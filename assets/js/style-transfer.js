/*
 * Style transfer: fast neural style networks (Johnson et al., trained with
 * PyTorch - pytorch/examples fast_neural_style, exported to ONNX in the ONNX
 * model zoo). Pick a style, upload a photo (or use the sample) and the network
 * repaints it in the browser with onnxruntime-web.
 */
(function () {
  "use strict";

  var area = document.getElementById("style-area");
  var srcC = document.getElementById("style-src");
  var outC = document.getElementById("style-out");
  if (!area || !srcC || !outC) return;
  var srcCtx = srcC.getContext("2d");
  var outCtx = outC.getContext("2d");

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
  // try the GPU execution provider first, fall back to plain wasm
  var wasmOnly = false;
  function createSession(url) {
    if (wasmOnly) return window.ort.InferenceSession.create(url);
    return window.ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] })
      .catch(function () { return window.ort.InferenceSession.create(url); });
  }
  var SIZE = 224;                       // model input resolution
  var sessions = {};                    // fast style -> InferenceSession
  var adainSession = null;              // AdaIN arbitrary-style session
  var adainStyles = {};                 // painting key -> Float32Array (CHW 0-255)
  var cur = { type: "fast", style: "starry-night" };
  var srcImage = null;                  // current content image (Image element)
  var busy = false, queued = false;
  var lastStyled = null, lastSrcPx = null;   // cached last result for re-blending
  var statusEl = document.getElementById("style-status");

  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "&nbsp;";
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      if (window.ort) { res(); return; }
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  function getSession(style) {
    if (sessions[style]) return Promise.resolve(sessions[style]);
    var base = area.getAttribute("data-base");
    setStatus("Loading style model…", "Chargement du modèle de style…");
    return createSession(base + style + ".onnx").then(function (s) {
      sessions[style] = s;
      return s;
    });
  }

  function getAdainSession() {
    if (adainSession) return Promise.resolve(adainSession);
    var base = area.getAttribute("data-base");
    setStatus("Loading AdaIN model (~28 MB)…", "Chargement du modèle AdaIN (~28 Mo)…");
    return createSession(base + "adain.onnx?v=2").then(function (s) {
      adainSession = s;
      return s;
    });
  }

  function getStyleTensor(key) {
    if (adainStyles[key]) return Promise.resolve(adainStyles[key]);
    var stylesBase = area.getAttribute("data-styles");
    return new Promise(function (res, rej) {
      var im = new Image();
      im.onload = function () {
        var tmp = document.createElement("canvas");
        tmp.width = SIZE; tmp.height = SIZE;
        var tctx = tmp.getContext("2d");
        // cover-crop the painting to a square
        var w = im.naturalWidth, h = im.naturalHeight, m = Math.min(w, h);
        tctx.drawImage(im, (w - m) / 2, (h - m) / 2, m, m, 0, 0, SIZE, SIZE);
        var px = tctx.getImageData(0, 0, SIZE, SIZE).data;
        var plane = SIZE * SIZE;
        var arr = new Float32Array(3 * plane);
        for (var k = 0; k < plane; k++) {
          arr[k] = px[4 * k];
          arr[plane + k] = px[4 * k + 1];
          arr[2 * plane + k] = px[4 * k + 2];
        }
        adainStyles[key] = arr;
        res(arr);
      };
      im.onerror = rej;
      im.src = stylesBase + key + ".jpg";
    });
  }

  function drawSource() {
    if (!srcImage) return;
    // cover-crop to square
    var w = srcImage.naturalWidth, h = srcImage.naturalHeight, m = Math.min(w, h);
    srcCtx.clearRect(0, 0, srcC.width, srcC.height);
    srcCtx.drawImage(srcImage, (w - m) / 2, (h - m) / 2, m, m, 0, 0, srcC.width, srcC.height);
  }

  function contentTensor() {
    // sample the source canvas down to 224x224
    var tmp = document.createElement("canvas");
    tmp.width = SIZE; tmp.height = SIZE;
    var tctx = tmp.getContext("2d");
    tctx.drawImage(srcC, 0, 0, SIZE, SIZE);
    var px = tctx.getImageData(0, 0, SIZE, SIZE).data;
    lastSrcPx = px;
    var plane = SIZE * SIZE;
    var input = new Float32Array(3 * plane);    // CHW, RGB in [0,255]
    for (var k = 0; k < plane; k++) {
      input[k] = px[4 * k];
      input[plane + k] = px[4 * k + 1];
      input[2 * plane + k] = px[4 * k + 2];
    }
    return new window.ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
  }

  function stylize() {
    if (!srcImage) return;
    if (busy) { queued = true; return; }
    busy = true;
    var run;
    if (cur.type === "adain") {
      run = loadScript(ORT_URL).then(function () {
        return Promise.all([getAdainSession(), getStyleTensor(cur.style)]);
      }).then(function (rs) {
        setStatus("Painting…", "Peinture en cours…");
        return rs[0].run({
          content: contentTensor(),
          style: new window.ort.Tensor("float32", rs[1].slice(), [1, 3, SIZE, SIZE]),
          alpha: new window.ort.Tensor("float32", new Float32Array([1]), [1])
        });
      });
    } else {
      run = loadScript(ORT_URL).then(function () {
        return getSession(cur.style);
      }).then(function (session) {
        setStatus("Painting…", "Peinture en cours…");
        return session.run({ input1: contentTensor() });
      });
    }
    run.then(function (out) {
      lastStyled = out.output1.data;
      drawBlend();
      setStatus("", "");
      busy = false;
      if (queued) { queued = false; stylize(); }
    }).catch(function (e) {
      busy = false;
      if (!wasmOnly) {
        // a kernel unsupported on WebGPU: drop cached sessions, retry on wasm
        wasmOnly = true;
        sessions = {}; adainSession = null;
        stylize();
        return;
      }
      setStatus("Failed: " + e, "Échec : " + e);
    });
  }

  function strength() {
    var el = document.getElementById("style-strength");
    return el ? parseInt(el.value, 10) / 100 : 1;
  }

  // blend the cached stylized output with the original at the chosen intensity
  function drawBlend() {
    if (!lastStyled || !lastSrcPx) return;
    var a = strength(), plane = SIZE * SIZE;
    var tmp = document.createElement("canvas");
    tmp.width = SIZE; tmp.height = SIZE;
    var tctx = tmp.getContext("2d");
    var img = tctx.createImageData(SIZE, SIZE);
    for (var k = 0; k < plane; k++) {
      var r = a * lastStyled[k] + (1 - a) * lastSrcPx[4 * k];
      var g = a * lastStyled[plane + k] + (1 - a) * lastSrcPx[4 * k + 1];
      var b = a * lastStyled[2 * plane + k] + (1 - a) * lastSrcPx[4 * k + 2];
      img.data[4 * k] = Math.max(0, Math.min(255, Math.round(r)));
      img.data[4 * k + 1] = Math.max(0, Math.min(255, Math.round(g)));
      img.data[4 * k + 2] = Math.max(0, Math.min(255, Math.round(b)));
      img.data[4 * k + 3] = 255;
    }
    tctx.putImageData(img, 0, 0);
    outCtx.imageSmoothingEnabled = true;
    outCtx.clearRect(0, 0, outC.width, outC.height);
    outCtx.drawImage(tmp, 0, 0, outC.width, outC.height);
  }

  function setImageFromURL(url) {
    var im = new Image();
    im.onload = function () { srcImage = im; drawSource(); stylize(); };
    im.src = url;
  }

  // ---- controls ----
  var styleBtns = document.querySelectorAll("[data-style]");
  var adainBtns = document.querySelectorAll("[data-adain]");
  function clearActive() {
    var j;
    for (j = 0; j < styleBtns.length; j++) styleBtns[j].classList.remove("active");
    for (j = 0; j < adainBtns.length; j++) adainBtns[j].classList.remove("active");
  }
  for (var i = 0; i < styleBtns.length; i++) {
    (function (btn) {
      btn.addEventListener("click", function () {
        clearActive();
        btn.classList.add("active");
        cur = { type: "fast", style: btn.getAttribute("data-style") };
        stylize();
      });
    })(styleBtns[i]);
  }
  for (var a = 0; a < adainBtns.length; a++) {
    (function (btn) {
      btn.addEventListener("click", function () {
        clearActive();
        btn.classList.add("active");
        cur = { type: "adain", style: btn.getAttribute("data-adain") };
        stylize();
      });
    })(adainBtns[a]);
  }

  var strengthIn = document.getElementById("style-strength");
  if (strengthIn) strengthIn.addEventListener("input", function () {
    var lab = document.getElementById("style-strength-val");
    if (lab) lab.textContent = strengthIn.value + "%";
    drawBlend();                                 // re-blend cached result, no re-run
  });

  var fileIn = document.getElementById("style-file");
  if (fileIn) fileIn.addEventListener("change", function () {
    if (!fileIn.files || !fileIn.files[0]) return;
    var url = URL.createObjectURL(fileIn.files[0]);
    setImageFromURL(url);
  });

  var inited = false;
  function init() {
    if (inited) return;
    inited = true;
    setImageFromURL(area.getAttribute("data-sample"));   // sample dog photo
  }
  window.__styleShow = init;
})();
