/*
 * Image toolbox: convert/resize/crop, compress (before/after), AI 4x upscale
 * (Real-ESRGAN general-x4v3 via onnxruntime-web, tiled), EXIF strip and QR
 * code generation. Everything runs locally; libraries are lazy-loaded.
 */
(function () {
  "use strict";

  var drop = document.getElementById("it-drop");
  if (!drop) return;

  var HEIC_URL = "https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js";
  var QR_URL = "https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js";

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

  var RUNNERS = { convert: doConvert, compress: doCompress, qr: doQr };

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
