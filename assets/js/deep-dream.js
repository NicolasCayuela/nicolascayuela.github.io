/*
 * Deep Dream: gradient ascent on the activations of a pretrained ImageNet CNN
 * (MobileNet v1, loaded from the TF.js model hub). The image is nudged to make
 * a chosen conv layer fire harder, so the network's favourite patterns - swirls,
 * eyes, dog snouts (ImageNet is dog-heavy) - hallucinate into the picture.
 *
 * This is the one playground tab that uses TensorFlow.js instead of
 * onnxruntime-web: Deep Dream needs gradients w.r.t. the input pixels, and
 * onnxruntime-web is forward-only. TF.js has autograd (tf.grad), so the whole
 * ascent loop runs in the browser on WebGL. Inceptionism: Mordvintsev et al. (2015).
 */
(function () {
  "use strict";

  var area = document.getElementById("dream-area");
  if (!area) return;
  var outC = document.getElementById("dream-canvas");
  if (!outC) return;

  var TF_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  var MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json";

  // shallow -> deep: shallow layers dream textures/swirls, deep layers dream
  // object parts (eyes, snouts). MobileNet v1 pointwise-conv ReLU layer names.
  var LAYERS = {
    textures: "conv_pw_5_relu",
    patterns: "conv_pw_9_relu",
    objects:  "conv_pw_11_relu",
    deep:     "conv_pw_13_relu"
  };

  var base = null;              // full MobileNet LayersModel
  var dreamModels = {};         // layer name -> sub-model (inputs -> that layer)
  var srcImage = null;          // current source Image element
  var running = false, stopReq = false;
  var inited = false;

  var statusEl = document.getElementById("dream-status");
  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = (en || fr)
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "&nbsp;";
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      if (window.tf) { res(); return; }
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  function getBase() {
    if (base) return Promise.resolve(base);
    setStatus("Loading MobileNet (~17 MB)…", "Chargement de MobileNet (~17 Mo)…");
    return window.tf.loadLayersModel(MODEL_URL).then(function (m) {
      base = m;
      return m;
    });
  }

  // MobileNet's declared input is fixed at 224x224, but Deep Dream runs the
  // image at several octave resolutions. MobileNet v1 is a purely sequential
  // stack of size-agnostic conv layers, so we rebuild a fully-convolutional
  // sub-model on a flexible [null,null,3] input (re-applying the same layer
  // objects keeps the pretrained weights) and stop at the chosen layer.
  function getDreamModel(layerName) {
    if (dreamModels[layerName]) return dreamModels[layerName];
    var tf = window.tf;
    var inp = tf.input({ shape: [null, null, 3] });
    var x = inp, layers = base.layers;
    for (var i = 0; i < layers.length; i++) {
      if (layers[i].getClassName() === "InputLayer") continue;
      x = layers[i].apply(x);
      if (layers[i].name === layerName) break;
    }
    var sub = tf.model({ inputs: inp, outputs: x });
    dreamModels[layerName] = sub;
    return sub;
  }

  // ---- control readers ----
  function ctrlVal(id, dflt) {
    var el = document.getElementById(id);
    return el ? parseFloat(el.value) : dflt;
  }
  function layerName() {
    var el = document.getElementById("dream-layer");
    var key = el ? el.value : "patterns";
    return LAYERS[key] || LAYERS.patterns;
  }

  // draw a normalized [1,H,W,3] tensor (values in [-1,1]) to the output canvas
  function renderTensor(img4) {
    var tf = window.tf;
    return tf.tidy(function () {
      var px = img4.add(1).div(2).clipByValue(0, 1).squeeze([0]);   // [H,W,3] in [0,1]
      return px;
    });
  }
  async function drawTensor(img4) {
    var tf = window.tf;
    var px = renderTensor(img4);
    var h = px.shape[0], w = px.shape[1];
    if (outC.width !== w || outC.height !== h) { outC.width = w; outC.height = h; }
    await tf.browser.toPixels(px, outC);
    px.dispose();
  }

  // pull the source image into a [1,H,W,3] tensor normalized to [-1,1] at a
  // working resolution (longest side = maxSide)
  function sourceTensor(maxSide) {
    var tf = window.tf;
    var w = srcImage.naturalWidth, h = srcImage.naturalHeight;
    var scale = maxSide / Math.max(w, h);
    var tw = Math.max(32, Math.round(w * scale));
    var th = Math.max(32, Math.round(h * scale));
    var tmp = document.createElement("canvas");
    tmp.width = tw; tmp.height = th;
    tmp.getContext("2d").drawImage(srcImage, 0, 0, tw, th);
    return tf.tidy(function () {
      var p = tf.browser.fromPixels(tmp).toFloat().div(127.5).sub(1);  // [th,tw,3]
      return p.expandDims(0);                                          // [1,th,tw,3]
    });
  }

  async function run() {
    if (running || !srcImage) return;
    running = true; stopReq = false;
    var btn = document.getElementById("dream-run");
    if (btn) btn.innerHTML = '<i class="fas fa-stop"></i> <span class="lang-en">Stop</span><span class="lang-fr">Arrêter</span>';

    try {
      await loadScript(TF_URL);
      await getBase();
      var tf = window.tf;
      var model = getDreamModel(layerName());

      var iters = Math.round(ctrlVal("dream-iters", 15));
      var lr = ctrlVal("dream-step", 0.012);
      var octaves = Math.round(ctrlVal("dream-octaves", 3));
      var maxSide = (typeof navigator !== "undefined" && navigator.gpu) ? 480 : 300;

      // loss = mean of squared activations at the chosen layer
      var gradFn = tf.grad(function (x) {
        return model.predict(x).square().mean();
      });

      // octave sizes, smallest -> largest
      var full = sourceTensor(maxSide);          // [1,H,W,3]
      var H = full.shape[1], W = full.shape[2];
      var octaveScale = 1.4;

      var cur = tf.tidy(function () {
        var s = Math.pow(octaveScale, -(octaves - 1));
        return tf.image.resizeBilinear(full, [Math.round(H * s), Math.round(W * s)]);
      });

      for (var o = 0; o < octaves && !stopReq; o++) {
        var s = Math.pow(octaveScale, -(octaves - 1 - o));
        var oh = Math.round(H * s), ow = Math.round(W * s);
        var resized = tf.tidy(function () { return tf.image.resizeBilinear(cur, [oh, ow]); });
        cur.dispose(); cur = resized;

        setStatus("Dreaming… octave " + (o + 1) + "/" + octaves,
                  "Rêve… octave " + (o + 1) + "/" + octaves);

        for (var i = 0; i < iters && !stopReq; i++) {
          var next = tf.tidy(function () {
            var g = gradFn(cur);
            // normalize the gradient by its std so the step size is scale-free
            var std = tf.moments(g).variance.sqrt().add(1e-8);
            var step = g.div(std).mul(lr);
            return cur.add(step).clipByValue(-1, 1);
          });
          cur.dispose(); cur = next;
          if (i % 2 === 0) { await drawTensor(cur); await tf.nextFrame(); }
        }
        await drawTensor(cur);
      }

      full.dispose();
      // final upscale render to full working size for a crisp result
      var finalImg = tf.tidy(function () { return tf.image.resizeBilinear(cur, [H, W]); });
      cur.dispose();
      await drawTensor(finalImg);
      finalImg.dispose();
      setStatus(stopReq ? "Stopped." : "Done.", stopReq ? "Arrêté." : "Terminé.");
    } catch (e) {
      setStatus("Failed: " + e, "Échec : " + e);
    } finally {
      running = false;
      var b = document.getElementById("dream-run");
      if (b) b.innerHTML = '<i class="fas fa-magic"></i> <span class="lang-en">Dream</span><span class="lang-fr">Rêver</span>';
    }
  }

  function setImageFromURL(url) {
    var im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = function () {
      srcImage = im;
      // draw the plain source so the canvas isn't blank before the first dream
      var w = im.naturalWidth, h = im.naturalHeight, m = Math.max(w, h), s = 320 / m;
      outC.width = Math.round(w * s); outC.height = Math.round(h * s);
      outC.getContext("2d").drawImage(im, 0, 0, outC.width, outC.height);
      setStatus("", "");
    };
    im.onerror = function () { setStatus("Couldn't load that image.", "Impossible de charger cette image."); };
    im.src = url;
  }

  // ---- controls ----
  var runBtn = document.getElementById("dream-run");
  if (runBtn) runBtn.addEventListener("click", function () {
    if (running) { stopReq = true; return; }
    run();
  });
  var fileIn = document.getElementById("dream-file");
  if (fileIn) fileIn.addEventListener("change", function () {
    if (!fileIn.files || !fileIn.files[0]) return;
    setImageFromURL(URL.createObjectURL(fileIn.files[0]));
  });
  var resetBtn = document.getElementById("dream-reset");
  if (resetBtn) resetBtn.addEventListener("click", function () {
    if (running) { stopReq = true; }
    if (srcImage) setImageFromURL(srcImage.src);
  });
  // live slider value labels
  ["dream-iters", "dream-step", "dream-octaves"].forEach(function (id) {
    var el = document.getElementById(id), lab = document.getElementById(id + "-val");
    if (el && lab) {
      var upd = function () { lab.textContent = el.value; };
      el.addEventListener("input", upd); upd();
    }
  });

  function init() {
    if (inited) return;
    inited = true;
    setImageFromURL(area.getAttribute("data-sample"));
  }
  window.__dreamShow = init;
})();
