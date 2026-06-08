/*
 * Dog Diffusion: a DDPM (epsilon-prediction UNet) trained in
 * PyTorch on AFHQ dog faces. The UNet runs in the browser with onnxruntime-web
 * and a DDIM sampler; the canvas shows the image denoising step by step.
 */
(function () {
  "use strict";

  var area = document.getElementById("ddpm-area");
  var canvas = document.getElementById("ddpm-canvas");
  if (!area || !canvas) return;
  var ctx = canvas.getContext("2d");

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js";
  // try the GPU execution provider first, fall back to plain wasm
  function createSession(url) {
    return window.ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] })
      .catch(function () { return window.ort.InferenceSession.create(url); });
  }
  var session = null, meta = null, loading = false, ready = false;
  var running = false, runId = 0;
  var IMG = 32;
  var off = document.createElement("canvas");
  var offCtx = null;
  var statusEl = document.getElementById("ddpm-status");

  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "&nbsp;";
  }

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      if (window.ort) { res(); return; }
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  function draw(x) {
    // x: Float32Array 3*IMG*IMG in [-1,1] (CHW)
    var img = offCtx.createImageData(IMG, IMG);
    var plane = IMG * IMG;
    for (var k = 0; k < plane; k++) {
      img.data[4 * k] = Math.max(0, Math.min(255, Math.round((x[k] + 1) * 127.5)));
      img.data[4 * k + 1] = Math.max(0, Math.min(255, Math.round((x[plane + k] + 1) * 127.5)));
      img.data[4 * k + 2] = Math.max(0, Math.min(255, Math.round((x[2 * plane + k] + 1) * 127.5)));
      img.data[4 * k + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function steps() {
    return 100;                                // fixed DDIM step count (more = sharper, linearly slower)
  }

  function generate() {
    if (!ready || running) { if (running) runId++; return; }
    running = true;
    var myRun = ++runId;
    var K = steps(), T = meta.tsteps, acp = meta.acp;
    var plane = 3 * IMG * IMG;
    var x = new Float32Array(plane);
    for (var i = 0; i < plane; i++) x[i] = randn();
    draw(x);

    // DDIM timestep sequence: T-1 ... 0, K entries
    var seq = [];
    for (var k = 0; k < K; k++) seq.push(Math.round((T - 1) * (1 - k / (K - 1))));

    var stepIdx = 0;
    function step() {
      if (myRun !== runId) { running = false; setStatus("", ""); generate(); return; }
      if (stepIdx >= K) {
        running = false;
        setStatus("Done.", "Terminé.");
        return;
      }
      var ti = seq[stepIdx];
      setStatus("Denoising… step " + (stepIdx + 1) + "/" + K + " (t=" + ti + ")",
                "Débruitage… étape " + (stepIdx + 1) + "/" + K + " (t=" + ti + ")");
      var xin = new window.ort.Tensor("float32", x.slice(), [1, 3, IMG, IMG]);
      var tin = new window.ort.Tensor("float32", new Float32Array([ti]), [1]);
      session.run({ x: xin, t: tin }).then(function (out) {
        var eps = out.eps.data;
        var a = acp[ti];
        var aPrev = stepIdx + 1 < K ? acp[seq[stepIdx + 1]] : 1.0;
        var sa = Math.sqrt(a), s1a = Math.sqrt(1 - a);
        var sap = Math.sqrt(aPrev), s1ap = Math.sqrt(1 - aPrev);
        for (var i = 0; i < plane; i++) {
          var x0 = (x[i] - s1a * eps[i]) / sa;
          if (x0 > 1) x0 = 1; else if (x0 < -1) x0 = -1;
          x[i] = sap * x0 + s1ap * eps[i];
        }
        draw(x);
        stepIdx++;
        setTimeout(step, 30);                  // let the canvas update visibly
      }).catch(function (e) {
        running = false;
        setStatus("Sampling failed: " + e, "Échec de l'échantillonnage : " + e);
      });
    }
    step();
  }

  function init() {
    if (loading || ready) return;
    loading = true;
    setStatus("Loading model…", "Chargement du modèle…");
    var base = area.getAttribute("data-base");
    Promise.all([
      loadScript(ORT_URL),
      fetch(base + "dog_diffusion.json?v=5").then(function (r) { return r.json(); })
    ]).then(function (rs) {
      meta = rs[1];
      IMG = meta.img;
      off.width = IMG; off.height = IMG;
      offCtx = off.getContext("2d");
      return createSession(base + "dog_diffusion.onnx?v=5");
    }).then(function (s) {
      session = s; ready = true; loading = false;
      setStatus("Ready - press Generate.", "Prêt - clique sur Générer.");
    }).catch(function (e) {
      loading = false;
      setStatus("Failed to load the model: " + e, "Échec du chargement du modèle : " + e);
    });
  }

  var genBtn = document.getElementById("ddpm-generate");
  if (genBtn) genBtn.addEventListener("click", generate);

  window.__ddpmShow = init;
})();
