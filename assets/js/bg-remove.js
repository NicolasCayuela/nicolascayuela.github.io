/*
 * Background remover: U^2-Net (small) salient-object segmentation running
 * locally with onnxruntime-web. The input image is resized to 320x320 for the
 * network; the predicted mask is min-max normalised (as in rembg), shaped by
 * the threshold / hardness sliders and applied as the alpha channel of the
 * full-resolution image. No data ever leaves the browser.
 */
(function () {
  "use strict";

  var drop = document.getElementById("bgr-drop");
  if (!drop) return;

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
  var MODEL_URL = "assets/models/u2netp.onnx?v=1";
  var NET = 320;                                  // network input size
  var MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  var fileInput = document.getElementById("bgr-file");
  var statusEl = document.getElementById("bgr-status");
  var resultEl = document.getElementById("bgr-result");
  var origImg = document.getElementById("bgr-orig");
  var outCanvas = document.getElementById("bgr-out");
  var hardEl = document.getElementById("bgr-hard");
  var threshEl = document.getElementById("bgr-thresh");

  var session = null, ortLoading = null;
  var srcCanvas = null;                           // full-res original pixels
  var mask = null;                                // Float32Array NET*NET in [0,1]

  function setStatus(en, fr) {
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "";
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  function getSession() {
    if (session) return Promise.resolve(session);
    if (!ortLoading) {
      setStatus("Loading model (4.5 MB)…", "Chargement du modèle (4,5 Mo)…");
      // wasm only: the WebGPU EP rejects u2netp (MaxPool with ceil_mode),
      // and the model is small enough that wasm inference is fast anyway
      ortLoading = loadScript(ORT_URL)
        .then(function () {
          return window.ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
        })
        .then(function (s) { session = s; return s; });
    }
    return ortLoading;
  }

  // ---- preprocessing: image -> normalized 1x3x320x320 tensor ----
  function toTensor(img) {
    var c = document.createElement("canvas");
    c.width = NET; c.height = NET;
    var cx = c.getContext("2d");
    cx.drawImage(img, 0, 0, NET, NET);
    var px = cx.getImageData(0, 0, NET, NET).data;
    var n = NET * NET;
    var data = new Float32Array(3 * n);
    // rembg normalises by the per-image max before mean/std
    var mx = 0;
    for (var i = 0; i < n * 4; i++) { if (i % 4 !== 3 && px[i] > mx) mx = px[i]; }
    mx = mx || 255;
    for (var p = 0; p < n; p++) {
      data[p]         = (px[p * 4]     / mx - MEAN[0]) / STD[0];
      data[n + p]     = (px[p * 4 + 1] / mx - MEAN[1]) / STD[1];
      data[2 * n + p] = (px[p * 4 + 2] / mx - MEAN[2]) / STD[2];
    }
    return new window.ort.Tensor("float32", data, [1, 3, NET, NET]);
  }

  function runModel(img) {
    return getSession().then(function (s) {
      setStatus("Removing background…", "Suppression du fond…");
      var feeds = {};
      feeds[s.inputNames[0]] = toTensor(img);
      return s.run(feeds, [s.outputNames[0]]);    // first output = fused mask d0
    }).then(function (out) {
      var d = out[Object.keys(out)[0]].data;
      // min-max normalisation, as rembg does
      var mi = Infinity, ma = -Infinity;
      for (var i = 0; i < d.length; i++) { if (d[i] < mi) mi = d[i]; if (d[i] > ma) ma = d[i]; }
      var r = ma - mi || 1;
      mask = new Float32Array(d.length);
      for (var j = 0; j < d.length; j++) mask[j] = (d[j] - mi) / r;
      compose();
      resultEl.classList.remove("d-none");
      setStatus("", "");
    });
  }

  // ---- shape the raw mask with the sliders and apply it as alpha ----
  function shapedAlpha(v) {
    var t = threshEl.value / 100;                       // mask value mapped to 0.5
    var k = 1 + (hardEl.value / 100) * 24;              // sigmoid steepness
    var a = 1 / (1 + Math.exp(-k * (v - t) * 4));
    return a < 0.004 ? 0 : a > 0.996 ? 1 : a;           // snap near-extremes
  }

  function compose() {
    if (!srcCanvas || !mask) return;
    var w = srcCanvas.width, h = srcCanvas.height;
    outCanvas.width = w; outCanvas.height = h;
    var octx = outCanvas.getContext("2d");
    octx.drawImage(srcCanvas, 0, 0);
    var id = octx.getImageData(0, 0, w, h);
    var px = id.data;
    // bilinear sample of the 320x320 mask at every full-res pixel
    var sx = (NET - 1) / (w - 1 || 1), sy = (NET - 1) / (h - 1 || 1);
    for (var y = 0; y < h; y++) {
      var fy = y * sy, y0 = Math.floor(fy), y1 = Math.min(y0 + 1, NET - 1), wy = fy - y0;
      for (var x = 0; x < w; x++) {
        var fx = x * sx, x0 = Math.floor(fx), x1 = Math.min(x0 + 1, NET - 1), wx = fx - x0;
        var m = mask[y0 * NET + x0] * (1 - wx) * (1 - wy)
              + mask[y0 * NET + x1] * wx * (1 - wy)
              + mask[y1 * NET + x0] * (1 - wx) * wy
              + mask[y1 * NET + x1] * wx * wy;
        px[(y * w + x) * 4 + 3] = Math.round(shapedAlpha(m) * 255);
      }
    }
    octx.putImageData(id, 0, 0);
  }

  // ---- input handling ----
  function handleFile(file) {
    if (!file || file.type.indexOf("image/") !== 0) return;
    var url = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () {
      // cap the working resolution to keep composition fast
      var MAXSIDE = 2048;
      var sc = Math.min(1, MAXSIDE / Math.max(img.width, img.height));
      srcCanvas = document.createElement("canvas");
      srcCanvas.width = Math.round(img.width * sc);
      srcCanvas.height = Math.round(img.height * sc);
      srcCanvas.getContext("2d").drawImage(img, 0, 0, srcCanvas.width, srcCanvas.height);
      origImg.src = srcCanvas.toDataURL("image/png");
      URL.revokeObjectURL(url);
      runModel(srcCanvas).catch(function (e) {
        setStatus("Failed to load the model: " + e.message,
                  "Échec du chargement du modèle : " + e.message);
      });
    };
    img.src = url;
  }

  drop.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { handleFile(fileInput.files[0]); });
  drop.addEventListener("dragover", function (e) { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", function () { drop.classList.remove("dragover"); });
  drop.addEventListener("drop", function (e) {
    e.preventDefault(); drop.classList.remove("dragover");
    handleFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", function (e) {
    var items = (e.clipboardData || {}).items || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image/") === 0) { handleFile(items[i].getAsFile()); break; }
    }
  });

  var ct;
  function recompose() { clearTimeout(ct); ct = setTimeout(compose, 60); }
  hardEl.addEventListener("input", recompose);
  threshEl.addEventListener("input", recompose);

  document.getElementById("bgr-download").addEventListener("click", function () {
    var a = document.createElement("a");
    a.download = "image-alpha.png";
    a.href = outCanvas.toDataURL("image/png");
    a.click();
  });
  document.getElementById("bgr-reset").addEventListener("click", function () {
    resultEl.classList.add("d-none");
    srcCanvas = null; mask = null;
    fileInput.value = "";
    setStatus("", "");
  });
})();
