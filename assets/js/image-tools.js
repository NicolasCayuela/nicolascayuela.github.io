/*
 * Image toolbox: convert/resize/crop, compress (before/after), AI 4x upscale
 * (Real-ESRGAN general-x4v3 via onnxruntime-web, tiled), EXIF strip and QR
 * code generation. Everything runs locally; libraries are lazy-loaded.
 */
(function () {
  "use strict";

  var drop = document.getElementById("it-drop");
  if (!drop) return;

  var ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
  var HEIC_URL = "https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js";
  var EXIFR_URL = "https://cdn.jsdelivr.net/npm/exifr@7.1.3/dist/full.umd.js";
  var QR_URL = "https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js";
  var ESRGAN_URL = "assets/models/realesrgan_x4.onnx?v=1";

  var fileInput = document.getElementById("it-file");
  var statusEl = document.getElementById("it-status");
  var resultEl = document.getElementById("it-result");
  var runBtn = document.getElementById("it-run");
  var hintEl = document.getElementById("it-hint");
  var prevWrap = document.getElementById("it-preview-wrap");
  var prevCanvas = document.getElementById("it-preview");
  var metaEl = document.getElementById("it-meta");
  var qrUi = document.getElementById("it-qr-ui");
  var cropBtn = document.getElementById("it-crop-btn");

  var tool = "convert";
  var img = null;            // current image as a canvas (full resolution)
  var srcFile = null;
  var crop = null;           // {x,y,w,h} in image pixels

  var TOOLS = {
    convert:  { needsFile: true, opts: ["format", "quality", "w", "h"], crop: true,
                hint: ["Choose format/quality, set width or height (the other follows), drag on the preview to crop.",
                       "Choisis format/qualité, fixe largeur ou hauteur (l'autre suit), glisse sur l'aperçu pour rogner."] },
    compress: { needsFile: true, opts: ["format", "quality"],
                hint: ["Re-encodes the image at the chosen quality and shows the size before/after.",
                       "Réencode l'image à la qualité choisie et montre la taille avant/après."] },
    upscale:  { needsFile: true, opts: [],
                hint: ["4x super-resolution with Real-ESRGAN (4.9 MB model, tiled inference). Inputs larger than 1024 px are downscaled first.",
                       "Super-résolution 4x avec Real-ESRGAN (modèle de 4,9 Mo, inférence par tuiles). Les images de plus de 1024 px sont d'abord réduites."] },
    exif:     { needsFile: true, opts: ["format", "quality"],
                hint: ["Shows the metadata found (camera, date, GPS…) and produces a clean copy without any of it.",
                       "Affiche les métadonnées trouvées (appareil, date, GPS…) et produit une copie propre sans aucune d'elles."] },
    qr:       { needsFile: false, opts: [],
                hint: ["Generates a QR code locally - nothing is sent anywhere.",
                       "Génère un QR code localement - rien n'est envoyé nulle part."] }
  };
  var OPT_IDS = ["format", "quality", "w", "h"];

  function setStatus(en, fr) {
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + (fr || en) + "</span>"
      : "";
  }

  var loaded = {};
  function loadScript(src) {
    if (!loaded[src]) {
      loaded[src] = new Promise(function (res, rej) {
        var s = document.createElement("script");
        s.src = src; s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
      });
    }
    return loaded[src];
  }

  function fmtSize(b) { return b > 1048576 ? (b / 1048576).toFixed(2) + " MB" : Math.round(b / 1024) + " KB"; }
  function canvasBlob(c, type, q) {
    return new Promise(function (res) { c.toBlob(res, type, q); });
  }
  function resultLink(blob, name, extraEn, extraFr) {
    var url = URL.createObjectURL(blob);
    return '<a class="btn btn-sm btn-outline-dark mb-1" download="' + name + '" href="' + url + '">' +
      '<i class="fas fa-download"></i> ' + name + " (" + fmtSize(blob.size) + ")" +
      (extraEn ? ' <span class="lang-en">' + extraEn + '</span><span class="lang-fr">' + (extraFr || extraEn) + "</span>" : "") + "</a>";
  }

  // ---------- file loading (incl. HEIC) ----------
  async function fileToCanvas(file) {
    var blob = file;
    if (/heic|heif/i.test(file.type) || /\.hei[cf]$/i.test(file.name)) {
      setStatus("Decoding HEIC…", "Décodage HEIC…");
      await loadScript(HEIC_URL);
      blob = await window.heic2any({ blob: file, toType: "image/png" });
    }
    var bmp = await createImageBitmap(blob);
    var c = document.createElement("canvas");
    c.width = bmp.width; c.height = bmp.height;
    c.getContext("2d").drawImage(bmp, 0, 0);
    return c;
  }

  // ---------- preview + crop selection ----------
  function drawPreview() {
    if (!img) return;
    var maxW = 480;
    var sc = Math.min(1, maxW / img.width);
    prevCanvas.width = Math.round(img.width * sc);
    prevCanvas.height = Math.round(img.height * sc);
    var cx = prevCanvas.getContext("2d");
    cx.drawImage(img, 0, 0, prevCanvas.width, prevCanvas.height);
    if (crop) {
      cx.fillStyle = "rgba(0,0,0,.45)";
      cx.fillRect(0, 0, prevCanvas.width, prevCanvas.height);
      cx.clearRect(crop.x * sc, crop.y * sc, crop.w * sc, crop.h * sc);
      cx.drawImage(img, crop.x, crop.y, crop.w, crop.h,
                   crop.x * sc, crop.y * sc, crop.w * sc, crop.h * sc);
      cx.strokeStyle = "#1c54e5"; cx.lineWidth = 2;
      cx.strokeRect(crop.x * sc, crop.y * sc, crop.w * sc, crop.h * sc);
    }
    metaEl.textContent = img.width + " × " + img.height + " px" +
      (crop ? "  ·  crop " + Math.round(crop.w) + " × " + Math.round(crop.h) : "") +
      (srcFile ? "  ·  " + fmtSize(srcFile.size) : "");
  }

  var dragStart = null;
  function evPos(e) {
    var r = prevCanvas.getBoundingClientRect();
    var px = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    var py = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    var sc = img.width / r.width;
    return { x: Math.max(0, Math.min(img.width, px * sc)), y: Math.max(0, Math.min(img.height, py * sc)) };
  }
  function dragMove(e) {
    if (!dragStart || !img || tool !== "convert") return;
    var p = evPos(e);
    crop = { x: Math.min(dragStart.x, p.x), y: Math.min(dragStart.y, p.y),
             w: Math.abs(p.x - dragStart.x), h: Math.abs(p.y - dragStart.y) };
    if (crop.w < 4 || crop.h < 4) crop = null;
    drawPreview();
    e.preventDefault();
  }
  prevCanvas.addEventListener("mousedown", function (e) { if (img) { dragStart = evPos(e); } });
  prevCanvas.addEventListener("touchstart", function (e) { if (img) { dragStart = evPos(e); e.preventDefault(); } });
  window.addEventListener("mousemove", dragMove);
  prevCanvas.addEventListener("touchmove", dragMove);
  window.addEventListener("mouseup", function () { dragStart = null; });
  prevCanvas.addEventListener("touchend", function () { dragStart = null; });
  cropBtn.addEventListener("click", function () { crop = null; drawPreview(); });

  // ---------- tools ----------
  function workCanvas() {
    // apply crop + resize, return the canvas to encode
    var sx = 0, sy = 0, sw = img.width, sh = img.height;
    if (tool === "convert" && crop) { sx = crop.x; sy = crop.y; sw = crop.w; sh = crop.h; }
    var tw = parseInt(document.getElementById("it-w").value, 10) || 0;
    var th = parseInt(document.getElementById("it-h").value, 10) || 0;
    if (tool !== "convert") { tw = 0; th = 0; }
    if (!tw && !th) { tw = sw; th = sh; }
    else if (!th) { th = Math.round(sh * tw / sw); }
    else if (!tw) { tw = Math.round(sw * th / sh); }
    var c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(tw)); c.height = Math.max(1, Math.round(th));
    c.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
    return c;
  }

  async function doConvert() {
    var fmt = document.getElementById("it-format").value;
    var q = document.getElementById("it-quality").value / 100;
    var c = workCanvas();
    var blob = await canvasBlob(c, "image/" + fmt, fmt === "png" ? undefined : q);
    var ext = fmt === "jpeg" ? "jpg" : fmt;
    resultEl.innerHTML = resultLink(blob, baseName(srcFile.name) + "." + ext,
      "- " + c.width + " × " + c.height, "- " + c.width + " × " + c.height);
  }

  async function doCompress() {
    var fmt = document.getElementById("it-format").value;
    if (fmt === "png") fmt = "jpeg";                 // png ignores quality
    var q = document.getElementById("it-quality").value / 100;
    var blob = await canvasBlob(img, "image/" + fmt, q);
    var pct = Math.round(100 * blob.size / srcFile.size);
    resultEl.innerHTML = resultLink(blob, baseName(srcFile.name) + "-min." + (fmt === "jpeg" ? "jpg" : fmt),
      "(" + fmtSize(srcFile.size) + " → " + fmtSize(blob.size) + ", " + pct + "%)",
      "(" + fmtSize(srcFile.size) + " → " + fmtSize(blob.size) + ", " + pct + "%)");
  }

  var esrgan = null;
  async function doUpscale() {
    await loadScript(ORT_URL);
    if (!esrgan) {
      setStatus("Loading the upscaler (4.9 MB)…", "Chargement de l'upscaler (4,9 Mo)…");
      esrgan = await window.ort.InferenceSession.create(ESRGAN_URL, { executionProviders: ["wasm"] });
    }
    // cap the input so 4x output stays reasonable
    var MAX = 1024;
    var src = img;
    if (Math.max(img.width, img.height) > MAX) {
      var sc = MAX / Math.max(img.width, img.height);
      var d = document.createElement("canvas");
      d.width = Math.round(img.width * sc); d.height = Math.round(img.height * sc);
      d.getContext("2d").drawImage(img, 0, 0, d.width, d.height);
      src = d;
    }
    var W = src.width, H = src.height;
    var out = document.createElement("canvas");
    out.width = W * 4; out.height = H * 4;
    var octx = out.getContext("2d");

    var TILE = 128, PAD = 8;
    var sctx = src.getContext("2d");
    var nTiles = Math.ceil(W / TILE) * Math.ceil(H / TILE), done = 0;
    for (var ty = 0; ty < H; ty += TILE) {
      for (var tx = 0; tx < W; tx += TILE) {
        var x0 = Math.max(0, tx - PAD), y0 = Math.max(0, ty - PAD);
        var x1 = Math.min(W, tx + TILE + PAD), y1 = Math.min(H, ty + TILE + PAD);
        var tw = x1 - x0, th = y1 - y0;
        var px = sctx.getImageData(x0, y0, tw, th).data;
        var n = tw * th;
        var data = new Float32Array(3 * n);
        for (var p = 0; p < n; p++) {
          data[p] = px[p * 4] / 255;
          data[n + p] = px[p * 4 + 1] / 255;
          data[2 * n + p] = px[p * 4 + 2] / 255;
        }
        var feeds = {}; feeds[esrgan.inputNames[0]] = new ort.Tensor("float32", data, [1, 3, th, tw]);
        var res = await esrgan.run(feeds);
        var o = res[esrgan.outputNames[0]].data;        // 1x3x(4th)x(4tw)
        var OW = tw * 4, OH = th * 4, on = OW * OH;
        var oid = octx.createImageData(OW, OH);
        for (var op = 0; op < on; op++) {
          oid.data[op * 4]     = Math.max(0, Math.min(255, Math.round(o[op] * 255)));
          oid.data[op * 4 + 1] = Math.max(0, Math.min(255, Math.round(o[on + op] * 255)));
          oid.data[op * 4 + 2] = Math.max(0, Math.min(255, Math.round(o[2 * on + op] * 255)));
          oid.data[op * 4 + 3] = 255;
        }
        // paste only the un-padded core of the tile
        var cx0 = (tx - x0) * 4, cy0 = (ty - y0) * 4;
        var cw = Math.min(TILE, W - tx) * 4, ch = Math.min(TILE, H - ty) * 4;
        var tmp = document.createElement("canvas");
        tmp.width = OW; tmp.height = OH;
        tmp.getContext("2d").putImageData(oid, 0, 0);
        octx.drawImage(tmp, cx0, cy0, cw, ch, tx * 4, ty * 4, cw, ch);
        done++;
        setStatus("Upscaling… tile " + done + "/" + nTiles, "Agrandissement… tuile " + done + "/" + nTiles);
        await new Promise(function (r) { setTimeout(r, 0); });   // let the UI breathe
      }
    }
    var blob = await canvasBlob(out, "image/png");
    resultEl.innerHTML = resultLink(blob, baseName(srcFile.name) + "-x4.png",
      "- " + out.width + " × " + out.height, "- " + out.width + " × " + out.height);
  }

  async function doExif() {
    await loadScript(EXIFR_URL);
    var tags = null;
    try { tags = await window.exifr.parse(srcFile, { gps: true, tiff: true, exif: true }); } catch (e) {}
    var rows = [];
    if (tags) {
      var interesting = { Make: 1, Model: 1, DateTimeOriginal: 1, CreateDate: 1, Software: 1,
                          latitude: 1, longitude: 1, GPSAltitude: 1, LensModel: 1, ISO: 1,
                          FNumber: 1, ExposureTime: 1, FocalLength: 1 };
      Object.keys(tags).forEach(function (k) {
        if (interesting[k] && tags[k] != null) rows.push("<tr><td class='pr-3'>" + k + "</td><td>" + tags[k] + "</td></tr>");
      });
    }
    var fmt = document.getElementById("it-format").value;
    var q = document.getElementById("it-quality").value / 100;
    var blob = await canvasBlob(img, "image/" + fmt, fmt === "png" ? undefined : q);
    var ext = fmt === "jpeg" ? "jpg" : fmt;
    resultEl.innerHTML =
      (rows.length
        ? '<div class="small mb-2"><strong><span class="lang-en">Metadata found (will be removed):</span><span class="lang-fr">Métadonnées trouvées (elles seront supprimées) :</span></strong>' +
          '<table class="small text-muted">' + rows.join("") + "</table></div>"
        : '<div class="small text-muted mb-2"><span class="lang-en">No notable metadata found - the clean copy is still guaranteed metadata-free.</span><span class="lang-fr">Pas de métadonnées notables - la copie propre reste garantie sans métadonnées.</span></div>') +
      resultLink(blob, baseName(srcFile.name) + "-clean." + ext, "(no metadata)", "(sans métadonnées)");
  }

  async function doQr() {
    await loadScript(QR_URL);
    var text = document.getElementById("it-qr-text").value.trim();
    if (!text) throw new Error("empty text / texte vide");
    var size = parseInt(document.getElementById("it-qr-size").value, 10);
    var ecc = document.getElementById("it-qr-ecc").value;
    var qr = window.qrcode(0, ecc);                 // type 0 = auto
    qr.addData(text); qr.make();
    var nm = qr.getModuleCount();
    var quiet = 4;
    var cell = Math.floor(size / (nm + 2 * quiet));
    var dim = cell * (nm + 2 * quiet);
    var c = document.createElement("canvas");
    c.width = dim; c.height = dim;
    var cx = c.getContext("2d");
    cx.fillStyle = "#fff"; cx.fillRect(0, 0, dim, dim);
    cx.fillStyle = "#000";
    for (var r = 0; r < nm; r++) {
      for (var col = 0; col < nm; col++) {
        if (qr.isDark(r, col)) cx.fillRect((col + quiet) * cell, (r + quiet) * cell, cell, cell);
      }
    }
    var blob = await canvasBlob(c, "image/png");
    resultEl.innerHTML = '<div class="mb-2"><img src="' + URL.createObjectURL(blob) +
      '" style="width:180px; border:1px solid #e3e8f0; border-radius:8px;" alt="QR code"></div>' +
      resultLink(blob, "qrcode.png");
  }

  var RUNNERS = { convert: doConvert, compress: doCompress, upscale: doUpscale, exif: doExif, qr: doQr };

  function baseName(name) { return name.replace(/\.[^.]+$/, ""); }

  // ---------- UI wiring ----------
  function refresh() {
    var t = TOOLS[tool];
    drop.classList.toggle("d-none", !t.needsFile);
    qrUi.classList.toggle("d-none", tool !== "qr");
    OPT_IDS.forEach(function (o) {
      document.getElementById("it-opt-" + o).classList.toggle("d-none", t.opts.indexOf(o) === -1);
    });
    cropBtn.classList.toggle("d-none", !(t.crop && img));
    prevWrap.classList.toggle("d-none", !(t.needsFile && img));
    runBtn.disabled = t.needsFile ? !img : false;
    var h = t.hint;
    hintEl.innerHTML = '<span class="lang-en">' + h[0] + "</span><span class=\"lang-fr\">" + h[1] + "</span>";
    drawPreview();
  }

  document.querySelectorAll("[data-it-tool]").forEach(function (b) {
    b.addEventListener("click", function () {
      tool = b.getAttribute("data-it-tool");
      document.querySelectorAll("[data-it-tool]").forEach(function (x) { x.classList.toggle("active", x === b); });
      crop = null; resultEl.innerHTML = ""; setStatus("", "");
      refresh();
    });
  });

  function takeFile(f) {
    if (!f) return;
    srcFile = f;
    setStatus("Loading image…", "Chargement de l'image…");
    fileToCanvas(f).then(function (c) {
      img = c; crop = null;
      setStatus("", "");
      resultEl.innerHTML = "";
      refresh();
    }).catch(function (e) {
      setStatus("Cannot read this image: " + e.message, "Impossible de lire cette image : " + e.message);
    });
  }
  drop.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { takeFile(fileInput.files[0]); fileInput.value = ""; });
  drop.addEventListener("dragover", function (e) { e.preventDefault(); drop.style.borderColor = "#1c54e5"; });
  drop.addEventListener("dragleave", function () { drop.style.borderColor = "#cdd9f3"; });
  drop.addEventListener("drop", function (e) {
    e.preventDefault(); drop.style.borderColor = "#cdd9f3";
    takeFile(e.dataTransfer.files[0]);
  });

  document.getElementById("it-quality").addEventListener("input", function () {
    document.getElementById("it-quality-val").textContent = this.value;
  });

  runBtn.addEventListener("click", function () {
    runBtn.disabled = true;
    resultEl.innerHTML = "";
    setStatus("Working…", "Traitement…");
    RUNNERS[tool]().then(function () { setStatus("Done.", "Terminé."); })
      .catch(function (e) { setStatus("Error: " + e.message, "Erreur : " + e.message); })
      .then(function () { refresh(); });
  });

  refresh();
})();
