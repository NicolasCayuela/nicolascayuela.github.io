/*
 * Deep Dream (the classic Inceptionism look): gradient ascent on a deep layer of
 * Inception, so the network's favourite shapes - dog faces, fur, eyes (ImageNet
 * is dog-heavy) - hallucinate all over the image. We maximize the L2 of a deep
 * conv layer (Inception v3 "mixed6"); the dogs come for free, no class steering.
 *
 * Model: Inception v3 truncated at mixed6, trained on ImageNet, converted from
 * Keras to a TF.js LayersModel (~14 MB, hosted with the site). The BN layers are
 * rebuilt with scale=True so their gradient works in the browser.
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
  var MODEL_URL = area.getAttribute("data-model") + "model.json";

  var model = null;         // Inception v3 -> mixed6 (LayersModel)
  var gradFn = null;        // image -> gradient that grows dogs
  var srcImage = null;      // current source Image element
  var running = false, stopReq = false;
  var inited = false;
  var tfPromise = null, modelPromise = null;   // cached so preload + click never download twice

  var statusEl = document.getElementById("dream-status");
  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = (en || fr)
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "&nbsp;";
  }

  function loadScript(src) {
    if (window.tf) return Promise.resolve();
    if (tfPromise) return tfPromise;
    tfPromise = new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
    return tfPromise;
  }

  function getModel() {
    if (modelPromise) return modelPromise;
    setStatus("Loading dream model (~14 MB)…", "Chargement du modèle (~14 Mo)…");
    modelPromise = window.tf.loadLayersModel(MODEL_URL).then(function (m) {
      model = m;
      gradFn = window.tf.grad(function (img) { return m.predict(img).square().mean(); });
      return m;
    });
    return modelPromise;
  }

  // ---- control readers ----
  function ctrlVal(id, dflt) {
    var el = document.getElementById(id);
    return el ? parseFloat(el.value) : dflt;
  }

  // roll (wrap-around shift) by sy rows / sx cols on a [1,H,W,3] tensor. Used for
  // the per-step jitter that keeps Deep Dream's patterns coherent instead of a
  // pixel grid. Done outside the gradient (eager), so it isn't differentiated.
  function roll2d(t, sy, sx) {
    return window.tf.tidy(function () {
      var H = t.shape[1], W = t.shape[2];
      sy = ((sy % H) + H) % H; sx = ((sx % W) + W) % W;
      var r = t;
      if (sy) r = window.tf.concat([r.slice([0, H - sy, 0, 0], [-1, sy, -1, -1]),
                                    r.slice([0, 0, 0, 0], [-1, H - sy, -1, -1])], 1);
      if (sx) r = window.tf.concat([r.slice([0, 0, W - sx, 0], [-1, -1, sx, -1]),
                                    r.slice([0, 0, 0, 0], [-1, -1, W - sx, -1])], 2);
      return r;
    });
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

  // source image -> [1,H,W,3] normalized to [-1,1] (Inception preprocessing)
  function sourceTensor(maxSide) {
    var tf = window.tf;
    var w = srcImage.naturalWidth, h = srcImage.naturalHeight;
    var scale = maxSide / Math.max(w, h);
    var tw = Math.max(75, Math.round(w * scale));
    var th = Math.max(75, Math.round(h * scale));
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

    var tf, base = null, img = null, detail = null;
    try {
      await loadScript(TF_URL);
      await getModel();
      tf = window.tf;

      var iters = Math.round(ctrlVal("dream-iters", 25));
      var lr = ctrlVal("dream-step", 0.04);
      var octaves = Math.round(ctrlVal("dream-octaves", 4));
      var jitter = 16, octaveScale = 1.4;
      // bigger working size on the fast WebGL backend; smaller on plain CPU
      var maxSide = (tf.getBackend && tf.getBackend() === "webgl") ? 500 : 300;

      base = sourceTensor(maxSide);
      var H = base.shape[1], W = base.shape[2];

      // octave sizes, smallest -> largest; carry "detail" (the dreamed-in change)
      // up across scales so the hallucination is multi-scale and coherent
      var sizes = [];
      for (var o = 0; o < octaves; o++) {
        var sc = Math.pow(octaveScale, -(octaves - 1 - o));
        sizes.push([Math.max(75, Math.round(H * sc)), Math.max(75, Math.round(W * sc))]);
      }
      detail = tf.zeros([1, sizes[0][0], sizes[0][1], 3]);

      for (var k = 0; k < sizes.length && !stopReq; k++) {
        var oh = sizes[k][0], ow = sizes[k][1];
        if (img) { img.dispose(); }
        img = tf.tidy(function () { return tf.image.resizeBilinear(base, [oh, ow]); });
        var d2 = tf.tidy(function () { return tf.image.resizeBilinear(detail, [oh, ow]); });
        detail.dispose(); detail = d2;
        var x = tf.tidy(function () { return img.add(detail); });

        setStatus("Growing dogs… octave " + (k + 1) + "/" + octaves,
                  "Pousse des chiens… octave " + (k + 1) + "/" + octaves);

        for (var i = 0; i < iters && !stopReq; i++) {
          var sy = (Math.random() * (2 * jitter + 1) | 0) - jitter;
          var sx = (Math.random() * (2 * jitter + 1) | 0) - jitter;
          var rolled = roll2d(x, sy, sx); x.dispose();
          var stepped = tf.tidy(function () {
            var g = gradFn(rolled);
            var norm = g.abs().mean().add(1e-8);          // scale-free step
            return rolled.add(g.div(norm).mul(lr)).clipByValue(-1, 1);
          });
          rolled.dispose();
          x = roll2d(stepped, -sy, -sx); stepped.dispose();
          if (i % 2 === 0) { await drawTensor(x); await tf.nextFrame(); }
        }

        var nd = tf.tidy(function () { return x.sub(img); });   // detail = dreamed change
        detail.dispose(); detail = nd;
        await drawTensor(x);
        x.dispose();
      }
      setStatus(stopReq ? "Stopped." : "Done.", stopReq ? "Arrêté." : "Terminé.");
    } catch (e) {
      setStatus("Failed: " + e, "Échec : " + e);
    } finally {
      if (base && !base.isDisposed) base.dispose();
      if (img && !img.isDisposed) img.dispose();
      if (detail && !detail.isDisposed) detail.dispose();
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
      if (!(modelPromise && !model)) setStatus("", "");   // don't clobber the "loading model" message
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

  // download tf.js + the dream model in the background when the tab opens, so the
  // first Dream click is instant. Skip on metered / slow links.
  function warm() {
    var c = navigator.connection;
    if (c && (c.saveData || /^(slow-2g|2g|3g)$/.test(c.effectiveType || ""))) return;
    loadScript(TF_URL)
      .then(getModel)
      .then(function () { if (!running) setStatus("Ready - press Dream.", "Prêt - clique sur Rêver."); })
      .catch(function () {});
  }

  function init() {
    if (inited) return;
    inited = true;
    setImageFromURL(area.getAttribute("data-sample"));
    warm();
  }
  window.__dreamShow = init;
})();
