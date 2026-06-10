/*
 * Deep Dream (dogs everywhere): the classic Inceptionism look. We run a
 * pretrained ImageNet classifier (MobileNet v1, TF.js model hub) and, instead
 * of maximizing a layer's overall activation, we push the image toward the
 * network's DOG classes (ImageNet has ~120 dog breeds, indices 151-268). By
 * applying the classifier's 1x1 conv directly to the last conv feature map -
 * skipping the global pool - we get a *spatial* dog-logit map, so the gradient
 * ascent grows a dog face at every location: dogs hallucinate everywhere.
 *
 * This is the one playground tab on TensorFlow.js instead of onnxruntime-web:
 * Deep Dream needs gradients w.r.t. the input pixels, which the forward-only
 * ORT-web runtime can't provide. Inceptionism: Mordvintsev et al. (2015).
 */
(function () {
  "use strict";

  var area = document.getElementById("dream-area");
  if (!area) return;
  var outC = document.getElementById("dream-canvas");
  if (!outC) return;

  var TF_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  var MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json";
  var FEAT_LAYER = "conv_pw_13_relu";   // last conv feature map, [.,H,W,1024]

  var base = null;          // full MobileNet LayersModel
  var gradFn = null;        // image -> gradient that grows dogs everywhere
  var srcImage = null;      // current source Image element
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
    return window.tf.loadLayersModel(MODEL_URL).then(function (m) { base = m; return m; });
  }

  // Build the dog-maximizing gradient function once. MobileNet's declared input
  // is fixed at 224x224, but Deep Dream runs the image at several octave
  // resolutions; MobileNet v1 is a purely sequential stack of size-agnostic
  // conv layers, so we rebuild a fully-convolutional sub-model on a flexible
  // [null,null,3] input (re-applying the same layer objects keeps the
  // pretrained weights) up to the last conv map, then apply the classifier's
  // 1x1 conv on top of it to read off a spatial dog-logit map.
  function buildGradFn() {
    if (gradFn) return;
    var tf = window.tf;
    var inp = tf.input({ shape: [null, null, 3] });
    var x = inp, layers = base.layers;
    for (var i = 0; i < layers.length; i++) {
      if (layers[i].getClassName() === "InputLayer") continue;
      x = layers[i].apply(x);
      if (layers[i].name === FEAT_LAYER) break;
    }
    var featModel = tf.model({ inputs: inp, outputs: x });
    var convPreds = base.getLayer("conv_preds");      // 1x1 conv classifier head
    var dogIdx = [];
    for (var c = 151; c <= 268; c++) dogIdx.push(c);  // ImageNet dog breeds
    var dogT = tf.tensor1d(dogIdx, "int32");          // kept alive for the session

    gradFn = tf.grad(function (img) {
      var feat = featModel.predict(img);              // [1,H,W,1024]
      var logits = convPreds.apply(feat);             // [1,H,W,1000]
      return tf.gather(logits, dogT, 3).mean();       // mean over dog channels
    });
  }

  // ---- control readers ----
  function ctrlVal(id, dflt) {
    var el = document.getElementById(id);
    return el ? parseFloat(el.value) : dflt;
  }

  async function drawTensor(img4) {
    var tf = window.tf;
    var px = tf.tidy(function () {
      return img4.add(1).div(2).clipByValue(0, 1).squeeze([0]);  // [H,W,3] in [0,1]
    });
    var h = px.shape[0], w = px.shape[1];
    if (outC.width !== w || outC.height !== h) { outC.width = w; outC.height = h; }
    await tf.browser.toPixels(px, outC);
    px.dispose();
  }

  // source image -> [1,H,W,3] normalized to [-1,1] at a working resolution
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
      return tf.browser.fromPixels(tmp).toFloat().div(127.5).sub(1).expandDims(0);
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
      buildGradFn();

      var iters = Math.round(ctrlVal("dream-iters", 20));
      var lr = ctrlVal("dream-step", 0.012);
      var octaves = Math.round(ctrlVal("dream-octaves", 3));
      var maxSide = (typeof navigator !== "undefined" && navigator.gpu) ? 480 : 320;
      var octaveScale = 1.4;

      var full = sourceTensor(maxSide);
      var H = full.shape[1], W = full.shape[2];
      var cur = tf.tidy(function () {
        var s = Math.pow(octaveScale, -(octaves - 1));
        return tf.image.resizeBilinear(full, [Math.round(H * s), Math.round(W * s)]);
      });

      for (var o = 0; o < octaves && !stopReq; o++) {
        var s = Math.pow(octaveScale, -(octaves - 1 - o));
        var oh = Math.round(H * s), ow = Math.round(W * s);
        var resized = tf.tidy(function () { return tf.image.resizeBilinear(cur, [oh, ow]); });
        cur.dispose(); cur = resized;

        setStatus("Growing dogs… octave " + (o + 1) + "/" + octaves,
                  "Pousse des chiens… octave " + (o + 1) + "/" + octaves);

        for (var i = 0; i < iters && !stopReq; i++) {
          var next = tf.tidy(function () {
            var g = gradFn(cur);
            // normalize by std so the step size is scale-free
            var std = tf.moments(g).variance.sqrt().add(1e-8);
            return cur.add(g.div(std).mul(lr)).clipByValue(-1, 1);
          });
          cur.dispose(); cur = next;
          if (i % 2 === 0) { await drawTensor(cur); await tf.nextFrame(); }
        }
        await drawTensor(cur);
      }

      full.dispose();
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
