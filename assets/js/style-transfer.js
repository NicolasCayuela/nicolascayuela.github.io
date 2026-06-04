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

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
  var SIZE = 224;                       // model input resolution
  var sessions = {};                    // style -> InferenceSession
  var curStyle = "rain-princess-9";
  var srcImage = null;                  // current content image (Image element)
  var busy = false, ortLoaded = false;
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
    return window.ort.InferenceSession.create(base + style + ".onnx").then(function (s) {
      sessions[style] = s;
      return s;
    });
  }

  function drawSource() {
    if (!srcImage) return;
    // cover-crop to square
    var w = srcImage.naturalWidth, h = srcImage.naturalHeight, m = Math.min(w, h);
    srcCtx.clearRect(0, 0, srcC.width, srcC.height);
    srcCtx.drawImage(srcImage, (w - m) / 2, (h - m) / 2, m, m, 0, 0, srcC.width, srcC.height);
  }

  function stylize() {
    if (!srcImage || busy) return;
    busy = true;
    loadScript(ORT_URL).then(function () {
      return getSession(curStyle);
    }).then(function (session) {
      setStatus("Painting…", "Peinture en cours…");
      // sample the source canvas down to 224x224
      var tmp = document.createElement("canvas");
      tmp.width = SIZE; tmp.height = SIZE;
      var tctx = tmp.getContext("2d");
      tctx.drawImage(srcC, 0, 0, SIZE, SIZE);
      var px = tctx.getImageData(0, 0, SIZE, SIZE).data;
      var plane = SIZE * SIZE;
      var input = new Float32Array(3 * plane);    // CHW, RGB in [0,255]
      for (var k = 0; k < plane; k++) {
        input[k] = px[4 * k];
        input[plane + k] = px[4 * k + 1];
        input[2 * plane + k] = px[4 * k + 2];
      }
      var tensor = new window.ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
      return session.run({ input1: tensor });
    }).then(function (out) {
      var d = out.output1.data, plane = SIZE * SIZE;
      var tmp = document.createElement("canvas");
      tmp.width = SIZE; tmp.height = SIZE;
      var tctx = tmp.getContext("2d");
      var img = tctx.createImageData(SIZE, SIZE);
      for (var k = 0; k < plane; k++) {
        img.data[4 * k] = Math.max(0, Math.min(255, Math.round(d[k])));
        img.data[4 * k + 1] = Math.max(0, Math.min(255, Math.round(d[plane + k])));
        img.data[4 * k + 2] = Math.max(0, Math.min(255, Math.round(d[2 * plane + k])));
        img.data[4 * k + 3] = 255;
      }
      tctx.putImageData(img, 0, 0);
      outCtx.imageSmoothingEnabled = true;
      outCtx.clearRect(0, 0, outC.width, outC.height);
      outCtx.drawImage(tmp, 0, 0, outC.width, outC.height);
      setStatus("", "");
      busy = false;
    }).catch(function (e) {
      busy = false;
      setStatus("Failed: " + e, "Échec : " + e);
    });
  }

  function setImageFromURL(url) {
    var im = new Image();
    im.onload = function () { srcImage = im; drawSource(); stylize(); };
    im.src = url;
  }

  // ---- controls ----
  var styleBtns = document.querySelectorAll("[data-style]");
  for (var i = 0; i < styleBtns.length; i++) {
    (function (btn) {
      btn.addEventListener("click", function () {
        for (var j = 0; j < styleBtns.length; j++) styleBtns[j].classList.remove("active");
        btn.classList.add("active");
        curStyle = btn.getAttribute("data-style");
        stylize();
      });
    })(styleBtns[i]);
  }

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
