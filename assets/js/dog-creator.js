/*
 * Dog Creator: explore the 256-dimensional PCA latent space of a convolutional
 * autoencoder trained (in PyTorch) on AFHQ dog faces. The decoder runs in the
 * browser with onnxruntime-web; each slider moves along one principal component.
 */
(function () {
  "use strict";

  var area = document.getElementById("dog-area");
  var canvas = document.getElementById("dog-canvas");
  if (!area || !canvas) return;
  var ctx = canvas.getContext("2d");

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
  var N = 256;
  var session = null, meta = null, loading = false, ready = false;
  var coords = new Float32Array(N);          // current PCA coordinates
  var sliders = [];
  var off = document.createElement("canvas"); off.width = 64; off.height = 64;
  var offCtx = off.getContext("2d");
  var statusEl = document.getElementById("dog-status");

  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "";
  }

  // deterministic-enough gaussian
  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  function buildSliders() {
    var wrap = document.getElementById("dog-sliders");
    if (!wrap) return;
    wrap.innerHTML = "";
    for (var i = 0; i < N; i++) {
      (function (i) {
        var div = document.createElement("div");
        div.className = "dog-slider";
        var lab = document.createElement("div");
        lab.className = "small text-center";
        lab.style.lineHeight = "1.1";
        lab.textContent = "PC #" + (i + 1);
        var inp = document.createElement("input");
        inp.type = "range"; inp.min = "-300"; inp.max = "300"; inp.value = "0";
        inp.style.width = "100%";
        inp.addEventListener("input", function () {
          coords[i] = (parseFloat(inp.value) / 100) * meta.stds[i];
          var live = document.getElementById("dog-live");
          if (live && live.checked) requestRender();
        });
        div.appendChild(lab); div.appendChild(inp);
        wrap.appendChild(div);
        sliders.push(inp);
      })(i);
    }
  }

  function syncSliders() {
    for (var i = 0; i < N; i++) {
      var v = meta.stds[i] > 1e-8 ? (coords[i] / meta.stds[i]) * 100 : 0;
      sliders[i].value = Math.max(-300, Math.min(300, Math.round(v)));
    }
  }

  var rendering = false, pending = false;
  function requestRender() {
    if (rendering) { pending = true; return; }
    render();
  }
  function render() {
    if (!ready) return;
    rendering = true;
    var input = new window.ort.Tensor("float32", coords.slice(), [1, N]);
    session.run({ p: input }).then(function (out) {
      var d = out.img.data;                  // 1x3x64x64, [0,1]
      var img = offCtx.createImageData(64, 64);
      var plane = 64 * 64;
      for (var k = 0; k < plane; k++) {
        img.data[4 * k] = Math.round(255 * d[k]);
        img.data[4 * k + 1] = Math.round(255 * d[plane + k]);
        img.data[4 * k + 2] = Math.round(255 * d[2 * plane + k]);
        img.data[4 * k + 3] = 255;
      }
      offCtx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      rendering = false;
      if (pending) { pending = false; render(); }
    }).catch(function (e) {
      rendering = false;
      setStatus("Render failed: " + e, "Échec du rendu : " + e);
    });
  }

  function init() {
    if (loading || ready) return;
    loading = true;
    setStatus("Loading model…", "Chargement du modèle…");
    var base = area.getAttribute("data-base");
    Promise.all([
      loadScript(ORT_URL),
      fetch(base + "dog_data.json").then(function (r) { return r.json(); })
    ]).then(function (rs) {
      meta = rs[1];
      return window.ort.InferenceSession.create(base + "dog_decoder.onnx");
    }).then(function (s) {
      session = s; ready = true; loading = false;
      setStatus("", "");
      buildSliders();
      render();                              // average dog
    }).catch(function (e) {
      loading = false;
      setStatus("Failed to load the model: " + e, "Échec du chargement du modèle : " + e);
    });
  }

  // ---- buttons ----
  function on(id, fn) { var el = document.getElementById(id); if (el) el.addEventListener("click", fn); }
  on("dog-render", function () { requestRender(); });
  on("dog-avg", function () {
    if (!ready) return;
    coords.fill(0); syncSliders(); requestRender();
  });
  on("dog-random", function () {
    if (!ready) return;
    for (var i = 0; i < N; i++) coords[i] = randn() * meta.stds[i] * 0.85;
    syncSliders(); requestRender();
  });
  on("dog-random-ds", function () {
    if (!ready || !meta.samples || !meta.samples.length) return;
    var row = meta.samples[Math.floor(Math.random() * meta.samples.length)];
    for (var i = 0; i < N; i++) coords[i] = row[i];
    syncSliders(); requestRender();
  });

  // lazy init when the tab becomes visible (called from game-of-life.js)
  window.__dogShow = init;
})();
