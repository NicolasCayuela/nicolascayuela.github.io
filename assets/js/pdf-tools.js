/*
 * PDF toolbox: merge, split/extract, compress (rasterize), rotate, delete
 * pages, images->PDF, PDF->images, PDF->Markdown (with embedded images).
 * pdf-lib does the document surgery, PDF.js does the rendering/parsing,
 * JSZip packages multi-file outputs. All libraries are lazy-loaded from CDNs
 * on first run; the user's files never leave the browser.
 */
(function () {
  "use strict";

  var drop = document.getElementById("pt-drop");
  if (!drop) return;

  var PDFLIB_URL = "https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js";
  var PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
  var PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  var JSZIP_URL = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";

  var fileInput = document.getElementById("pt-file");
  var filesEl = document.getElementById("pt-files");
  var statusEl = document.getElementById("pt-status");
  var resultEl = document.getElementById("pt-result");
  var runBtn = document.getElementById("pt-run");
  var hintEl = document.getElementById("pt-hint");

  var files = [];          // current selection, in order
  var tool = "merge";

  // ---------- tool registry ----------
  var TOOLS = {
    merge:    { accept: "application/pdf", multiple: true,  min: 2, opts: [],
                hint: ["Select at least two PDFs - they are merged in the order of the list.",
                       "Sélectionne au moins deux PDF - ils sont fusionnés dans l'ordre de la liste."] },
    split:    { accept: "application/pdf", multiple: false, min: 1, opts: ["range"],
                hint: ["Extract the given pages into a new PDF.",
                       "Extrait les pages indiquées dans un nouveau PDF."] },
    compress: { accept: "application/pdf", multiple: false, min: 1, opts: ["quality", "scale"],
                hint: ["Rasterizes each page to JPEG - big size reduction, text no longer selectable.",
                       "Rastérise chaque page en JPEG - forte réduction de taille, texte non sélectionnable."] },
    rotate:   { accept: "application/pdf", multiple: false, min: 1, opts: ["angle", "range"],
                hint: ["Rotate all pages, or only the pages listed in the range field.",
                       "Pivote toutes les pages, ou seulement celles indiquées dans le champ pages."] },
    delete:   { accept: "application/pdf", multiple: false, min: 1, opts: ["range"],
                hint: ["Remove the given pages and download the rest.",
                       "Supprime les pages indiquées et télécharge le reste."] },
    img2pdf:  { accept: "image/*", multiple: true, min: 1, opts: [],
                hint: ["Each image becomes one PDF page (in list order).",
                       "Chaque image devient une page du PDF (dans l'ordre de la liste)."] },
    pdf2img:  { accept: "application/pdf", multiple: false, min: 1, opts: ["format", "scale"],
                hint: ["Renders every page as an image; several pages are packaged in a ZIP.",
                       "Convertit chaque page en image ; plusieurs pages sont livrées dans un ZIP."] },
    ocr:      { accept: "application/pdf,image/*", multiple: false, min: 1, opts: ["lang"],
                hint: ["Recognizes the text of a scanned PDF or photo (Tesseract, ~15 MB language data on first run) and downloads it as .txt.",
                       "Reconnaît le texte d'un PDF scanné ou d'une photo (Tesseract, ~15 Mo de données de langue au premier lancement) et le télécharge en .txt."] },
    sign:     { accept: "application/pdf", multiple: false, min: 1, opts: [],
                hint: ["Draw or import a signature, pick the page, click on the preview where it should go, then run.",
                       "Dessine ou importe une signature, choisis la page, clique sur l'aperçu à l'endroit voulu, puis lance."] },
    watermark:{ accept: "application/pdf", multiple: false, min: 1, opts: [],
                hint: ["Stamps the text on every page with the chosen opacity and size.",
                       "Appose le texte sur chaque page avec l'opacité et la taille choisies."] }
  };
  var OPT_IDS = ["range", "angle", "quality", "scale", "format", "lang"];
  var UI_PANELS = { sign: "pt-sign-ui", watermark: "pt-wm-ui" };

  function setStatus(en, fr) {
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + (fr || en) + "</span>"
      : "";
  }
  function setHint() {
    var h = TOOLS[tool].hint;
    hintEl.innerHTML = '<span class="lang-en">' + h[0] + "</span><span class=\"lang-fr\">" + h[1] + "</span>";
  }

  // ---------- lazy library loading ----------
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
  function needPdfLib() { return loadScript(PDFLIB_URL); }
  function needPdfJs() {
    return loadScript(PDFJS_URL).then(function () {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
    });
  }
  function needZip() { return loadScript(JSZIP_URL); }

  // ---------- helpers ----------
  function parseRange(str, n) {
    // "1-3,5" -> [0,1,2,4]; empty -> all pages
    if (!str || !str.trim()) return Array.from({ length: n }, function (_, i) { return i; });
    var out = [];
    str.split(",").forEach(function (part) {
      var m = part.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
      if (!m) return;
      var a = parseInt(m[1], 10), b = m[2] ? parseInt(m[2], 10) : a;
      for (var i = Math.min(a, b); i <= Math.max(a, b); i++) {
        if (i >= 1 && i <= n && out.indexOf(i - 1) === -1) out.push(i - 1);
      }
    });
    return out;
  }
  function downloadBlob(blob, name) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 30000);
  }
  function baseName(name) { return name.replace(/\.[^.]+$/, ""); }
  function fmtSize(b) { return b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.round(b / 1024) + " KB"; }
  function resultLink(blob, name, labelEn, labelFr) {
    var url = URL.createObjectURL(blob);
    return '<a class="btn btn-sm btn-outline-dark mb-1" download="' + name + '" href="' + url + '">' +
           '<i class="fas fa-download"></i> <span class="lang-en">' + labelEn + '</span><span class="lang-fr">' + (labelFr || labelEn) + "</span> (" + fmtSize(blob.size) + ")</a>";
  }

  // ---------- the tools ----------
  async function doMerge() {
    await needPdfLib();
    var out = await PDFLib.PDFDocument.create();
    for (var i = 0; i < files.length; i++) {
      setStatus("Merging " + (i + 1) + "/" + files.length + "…", "Fusion " + (i + 1) + "/" + files.length + "…");
      var src = await PDFLib.PDFDocument.load(await files[i].arrayBuffer(), { ignoreEncryption: true });
      var pages = await out.copyPages(src, src.getPageIndices());
      pages.forEach(function (p) { out.addPage(p); });
    }
    var bytes = await out.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), "merged.pdf");
  }

  async function doSplit() {
    await needPdfLib();
    var src = await PDFLib.PDFDocument.load(await files[0].arrayBuffer(), { ignoreEncryption: true });
    var idx = parseRange(document.getElementById("pt-range").value, src.getPageCount());
    if (!idx.length) throw new Error("empty page range");
    var out = await PDFLib.PDFDocument.create();
    (await out.copyPages(src, idx)).forEach(function (p) { out.addPage(p); });
    var bytes = await out.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), baseName(files[0].name) + "-pages.pdf");
  }

  async function doDelete() {
    await needPdfLib();
    var src = await PDFLib.PDFDocument.load(await files[0].arrayBuffer(), { ignoreEncryption: true });
    var n = src.getPageCount();
    var del = parseRange(document.getElementById("pt-range").value, n);
    var keep = [];
    for (var i = 0; i < n; i++) if (del.indexOf(i) === -1) keep.push(i);
    if (!keep.length) throw new Error("no pages left");
    var out = await PDFLib.PDFDocument.create();
    (await out.copyPages(src, keep)).forEach(function (p) { out.addPage(p); });
    var bytes = await out.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), baseName(files[0].name) + "-edited.pdf");
  }

  async function doRotate() {
    await needPdfLib();
    var doc = await PDFLib.PDFDocument.load(await files[0].arrayBuffer(), { ignoreEncryption: true });
    var angle = parseInt(document.getElementById("pt-angle").value, 10);
    var idx = parseRange(document.getElementById("pt-range").value, doc.getPageCount());
    idx.forEach(function (i) {
      var p = doc.getPage(i);
      p.setRotation(PDFLib.degrees((p.getRotation().angle + angle) % 360));
    });
    var bytes = await doc.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), baseName(files[0].name) + "-rotated.pdf");
  }

  async function doCompress() {
    await needPdfLib(); await needPdfJs();
    var q = document.getElementById("pt-quality").value / 100;
    var scale = parseFloat(document.getElementById("pt-scale").value);
    var srcBytes = await files[0].arrayBuffer();
    var doc = await pdfjsLib.getDocument({ data: srcBytes.slice(0) }).promise;
    var out = await PDFLib.PDFDocument.create();
    for (var i = 1; i <= doc.numPages; i++) {
      setStatus("Compressing page " + i + "/" + doc.numPages + "…", "Compression page " + i + "/" + doc.numPages + "…");
      var page = await doc.getPage(i);
      var vp = page.getViewport({ scale: scale });
      var c = document.createElement("canvas");
      c.width = Math.round(vp.width); c.height = Math.round(vp.height);
      await page.render({ canvasContext: c.getContext("2d"), viewport: vp }).promise;
      var jpg = await out.embedJpg(c.toDataURL("image/jpeg", q));
      var p1 = page.getViewport({ scale: 1 });
      var np = out.addPage([p1.width, p1.height]);
      np.drawImage(jpg, { x: 0, y: 0, width: p1.width, height: p1.height });
    }
    var bytes = await out.save();
    var blob = new Blob([bytes], { type: "application/pdf" });
    resultEl.innerHTML = resultLink(blob, baseName(files[0].name) + "-compressed.pdf",
      "Download (" + fmtSize(files[0].size) + " → " + fmtSize(blob.size) + ")",
      "Télécharger (" + fmtSize(files[0].size) + " → " + fmtSize(blob.size) + ")");
  }

  async function doImg2Pdf() {
    await needPdfLib();
    var out = await PDFLib.PDFDocument.create();
    for (var i = 0; i < files.length; i++) {
      setStatus("Adding image " + (i + 1) + "/" + files.length + "…", "Ajout image " + (i + 1) + "/" + files.length + "…");
      var buf = await files[i].arrayBuffer();
      var img;
      if (files[i].type === "image/jpeg") img = await out.embedJpg(buf);
      else if (files[i].type === "image/png") img = await out.embedPng(buf);
      else {
        // anything else (webp, gif, bmp…): round-trip through a canvas to PNG
        var bmp = await createImageBitmap(new Blob([buf]));
        var c = document.createElement("canvas");
        c.width = bmp.width; c.height = bmp.height;
        c.getContext("2d").drawImage(bmp, 0, 0);
        img = await out.embedPng(c.toDataURL("image/png"));
      }
      var p = out.addPage([img.width, img.height]);
      p.drawImage(img, { x: 0, y: 0, width: img.width, height: img.height });
    }
    var bytes = await out.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), "images.pdf");
  }

  async function doPdf2Img() {
    await needPdfJs();
    var fmt = document.getElementById("pt-format").value;       // jpeg | png
    var ext = fmt === "png" ? "png" : "jpg";
    var scale = parseFloat(document.getElementById("pt-scale").value);
    var doc = await pdfjsLib.getDocument({ data: await files[0].arrayBuffer() }).promise;
    var blobs = [];
    for (var i = 1; i <= doc.numPages; i++) {
      setStatus("Rendering page " + i + "/" + doc.numPages + "…", "Rendu page " + i + "/" + doc.numPages + "…");
      var page = await doc.getPage(i);
      var vp = page.getViewport({ scale: scale });
      var c = document.createElement("canvas");
      c.width = Math.round(vp.width); c.height = Math.round(vp.height);
      await page.render({ canvasContext: c.getContext("2d"), viewport: vp }).promise;
      blobs.push(await new Promise(function (res) { c.toBlob(res, "image/" + fmt, 0.92); }));
    }
    var base = baseName(files[0].name);
    if (blobs.length === 1) {
      downloadBlob(blobs[0], base + "." + ext);
      return;
    }
    await needZip();
    var zip = new JSZip();
    blobs.forEach(function (b, j) { zip.file(base + "-page" + (j + 1) + "." + ext, b); });
    var zblob = await zip.generateAsync({ type: "blob" });
    resultEl.innerHTML = resultLink(zblob, base + "-images.zip", "Download ZIP (" + blobs.length + " pages)", "Télécharger le ZIP (" + blobs.length + " pages)");
  }

  // ---- OCR (Tesseract.js) ----
  var TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js";
  async function doOcr() {
    await loadScript(TESSERACT_URL);
    var lang = document.getElementById("pt-lang").value;
    setStatus("Loading OCR engine + language data…", "Chargement du moteur OCR + données de langue…");
    var worker = await Tesseract.createWorker(lang.split("+"));
    var text = "";
    try {
      if (files[0].type === "application/pdf" || /\.pdf$/i.test(files[0].name)) {
        await needPdfJs();
        var doc = await pdfjsLib.getDocument({ data: await files[0].arrayBuffer() }).promise;
        for (var i = 1; i <= doc.numPages; i++) {
          setStatus("OCR page " + i + "/" + doc.numPages + "…", "OCR page " + i + "/" + doc.numPages + "…");
          var page = await doc.getPage(i);
          var vp = page.getViewport({ scale: 2 });
          var c = document.createElement("canvas");
          c.width = Math.round(vp.width); c.height = Math.round(vp.height);
          await page.render({ canvasContext: c.getContext("2d"), viewport: vp }).promise;
          var r = await worker.recognize(c);
          text += (i > 1 ? "\n\n----- page " + i + " -----\n\n" : "") + r.data.text;
        }
      } else {
        setStatus("Recognizing text…", "Reconnaissance du texte…");
        var res = await worker.recognize(files[0]);
        text = res.data.text;
      }
    } finally { await worker.terminate(); }
    var blob = new Blob([text], { type: "text/plain" });
    resultEl.innerHTML =
      '<textarea class="form-control form-control-sm mb-2" rows="8" readonly>' +
      text.replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</textarea>" +
      resultLink(blob, baseName(files[0].name) + ".txt", "Download text", "Télécharger le texte");
  }

  // ---- Sign PDF ----
  var signPlaced = null;           // {nx, ny} fractional position on the preview
  var signPageDims = null;         // {w, h} of the previewed page in PDF points
  var padDirty = false;

  async function renderSignPreview() {
    if (!files.length) return;
    await needPdfJs();
    var doc = await pdfjsLib.getDocument({ data: await files[0].arrayBuffer() }).promise;
    var pn = Math.min(Math.max(1, parseInt(document.getElementById("pt-sign-page").value, 10) || 1), doc.numPages);
    document.getElementById("pt-sign-page").value = pn;
    var page = await doc.getPage(pn);
    var vp = page.getViewport({ scale: 1.2 });
    var c = document.getElementById("pt-sign-preview");
    c.width = Math.round(vp.width); c.height = Math.round(vp.height);
    await page.render({ canvasContext: c.getContext("2d"), viewport: vp }).promise;
    var p1 = page.getViewport({ scale: 1 });
    signPageDims = { w: p1.width, h: p1.height };
    if (signPlaced) drawSignMarker();
  }

  function signatureCanvas() {
    // returns the drawn pad if used, else the uploaded image canvas
    var pad = document.getElementById("pt-sign-pad");
    if (padDirty) return Promise.resolve(pad);
    if (signImg) return Promise.resolve(signImg);
    return Promise.resolve(null);
  }
  var signImg = null;

  function drawSignMarker() {
    var c = document.getElementById("pt-sign-preview");
    var cx = c.getContext("2d");
    signatureCanvas().then(function (sig) {
      var w = parseInt(document.getElementById("pt-sign-size").value, 10) * (c.width / signPageDims.w);
      var h = sig ? w * sig.height / sig.width : w * 0.33;
      var x = signPlaced.nx * c.width - w / 2, y = signPlaced.ny * c.height - h / 2;
      if (sig) cx.drawImage(sig, x, y, w, h);
      cx.strokeStyle = "#1c54e5"; cx.setLineDash([5, 4]);
      cx.strokeRect(x, y, w, h);
      cx.setLineDash([]);
    });
  }

  async function doSign() {
    var sig = await signatureCanvas();
    if (!sig) throw new Error("no signature drawn or imported / aucune signature dessinée ou importée");
    if (!signPlaced) throw new Error("click on the preview to place it / clique sur l'aperçu pour la placer");
    await needPdfLib();
    var doc = await PDFLib.PDFDocument.load(await files[0].arrayBuffer(), { ignoreEncryption: true });
    var pn = Math.min(Math.max(1, parseInt(document.getElementById("pt-sign-page").value, 10) || 1), doc.getPageCount());
    var page = doc.getPage(pn - 1);
    var png = await doc.embedPng(sig.toDataURL ? sig.toDataURL("image/png") : sig);
    var w = parseInt(document.getElementById("pt-sign-size").value, 10);
    var h = w * png.height / png.width;
    var pw = page.getWidth(), ph = page.getHeight();
    page.drawImage(png, {
      x: signPlaced.nx * pw - w / 2,
      y: ph - signPlaced.ny * ph - h / 2,       // PDF origin is bottom-left
      width: w, height: h
    });
    var bytes = await doc.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), baseName(files[0].name) + "-signed.pdf");
  }

  // ---- Watermark ----
  async function doWatermark() {
    var text = document.getElementById("pt-wm-text").value.trim();
    if (!text) throw new Error("empty watermark text / texte de filigrane vide");
    await needPdfLib();
    var doc = await PDFLib.PDFDocument.load(await files[0].arrayBuffer(), { ignoreEncryption: true });
    var font = await doc.embedFont(PDFLib.StandardFonts.HelveticaBold);
    var opacity = document.getElementById("pt-wm-opacity").value / 100;
    var size = parseInt(document.getElementById("pt-wm-size").value, 10);
    var diag = document.getElementById("pt-wm-diag").checked;
    doc.getPages().forEach(function (page) {
      var pw = page.getWidth(), ph = page.getHeight();
      var tw = font.widthOfTextAtSize(text, size);
      page.drawText(text, {
        x: pw / 2 - (diag ? tw * 0.353 : tw / 2),         // cos45/2 when rotated
        y: ph / 2 - (diag ? tw * 0.353 : size / 2),
        size: size, font: font,
        color: PDFLib.rgb(0.55, 0.55, 0.6),
        opacity: opacity,
        rotate: diag ? PDFLib.degrees(45) : PDFLib.degrees(0)
      });
    });
    var bytes = await doc.save();
    downloadBlob(new Blob([bytes], { type: "application/pdf" }), baseName(files[0].name) + "-watermarked.pdf");
  }

  var RUNNERS = { merge: doMerge, split: doSplit, compress: doCompress, rotate: doRotate,
                  delete: doDelete, img2pdf: doImg2Pdf, pdf2img: doPdf2Img,
                  ocr: doOcr, sign: doSign, watermark: doWatermark };

  // ---------- UI wiring ----------
  function refresh() {
    var t = TOOLS[tool];
    fileInput.accept = t.accept;
    fileInput.multiple = t.multiple;
    OPT_IDS.forEach(function (o) {
      document.getElementById("pt-opt-" + o).classList.toggle("d-none", t.opts.indexOf(o) === -1);
    });
    Object.keys(UI_PANELS).forEach(function (k) {
      document.getElementById(UI_PANELS[k]).classList.toggle("d-none", tool !== k);
    });
    filesEl.innerHTML = files.map(function (f, i) {
      return '<li><i class="far fa-file"></i> ' + f.name + " (" + fmtSize(f.size) + ") " +
             '<a href="#" data-pt-rm="' + i + '" class="text-danger ml-1"><i class="fas fa-times"></i></a></li>';
    }).join("");
    runBtn.disabled = files.length < t.min;
    setHint();
  }

  document.querySelectorAll("[data-pt-tool]").forEach(function (b) {
    b.addEventListener("click", function () {
      tool = b.getAttribute("data-pt-tool");
      document.querySelectorAll("[data-pt-tool]").forEach(function (x) { x.classList.toggle("active", x === b); });
      files = []; resultEl.innerHTML = ""; setStatus("", "");
      refresh();
    });
  });

  filesEl.addEventListener("click", function (e) {
    var rm = e.target.closest("[data-pt-rm]");
    if (!rm) return;
    e.preventDefault();
    files.splice(parseInt(rm.getAttribute("data-pt-rm"), 10), 1);
    refresh();
  });

  function addFiles(list) {
    var t = TOOLS[tool];
    for (var i = 0; i < list.length; i++) {
      var f = list[i];
      var isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
      var isImg = f.type.indexOf("image/") === 0;
      var isTxt = /\.(md|markdown|txt)$/i.test(f.name) || f.type.indexOf("text/") === 0;
      var okType;
      if (t.accept === "image/*") okType = isImg;
      else if (t.accept.indexOf("image/*") !== -1) okType = isImg || isPdf;
      else if (t.accept.indexOf(".md") !== -1) okType = isTxt;
      else okType = isPdf;
      if (!okType) continue;
      if (!t.multiple) files = [];
      files.push(f);
    }
    resultEl.innerHTML = ""; setStatus("", "");
    refresh();
    if (tool === "sign" && files.length) {
      signPlaced = null;
      renderSignPreview().catch(function () {});
    }
  }

  // ---- signature pad + preview wiring ----
  (function () {
    var pad = document.getElementById("pt-sign-pad");
    var pcx = pad.getContext("2d");
    pcx.lineWidth = 2.2; pcx.lineCap = "round"; pcx.strokeStyle = "#1a2a6c";
    var drawing = false;
    function pos(e) {
      var r = pad.getBoundingClientRect();
      return { x: (e.touches ? e.touches[0].clientX : e.clientX) - r.left,
               y: (e.touches ? e.touches[0].clientY : e.clientY) - r.top };
    }
    function start(e) { drawing = true; var p = pos(e); pcx.beginPath(); pcx.moveTo(p.x, p.y); e.preventDefault(); }
    function move(e) { if (!drawing) return; var p = pos(e); pcx.lineTo(p.x, p.y); pcx.stroke(); padDirty = true; e.preventDefault(); }
    pad.addEventListener("mousedown", start); pad.addEventListener("touchstart", start);
    pad.addEventListener("mousemove", move); pad.addEventListener("touchmove", move);
    window.addEventListener("mouseup", function () { drawing = false; });
    pad.addEventListener("touchend", function () { drawing = false; });
    document.getElementById("pt-sign-clear").addEventListener("click", function () {
      pcx.clearRect(0, 0, pad.width, pad.height); padDirty = false; signImg = null;
    });
    document.getElementById("pt-sign-file").addEventListener("change", function () {
      var f = this.files[0];
      if (!f) return;
      createImageBitmap(f).then(function (bmp) {
        var c = document.createElement("canvas");
        c.width = bmp.width; c.height = bmp.height;
        c.getContext("2d").drawImage(bmp, 0, 0);
        signImg = c; padDirty = false;
        pcx.clearRect(0, 0, pad.width, pad.height);
        var sc = Math.min(pad.width / c.width, pad.height / c.height);
        pcx.drawImage(c, 0, 0, c.width * sc, c.height * sc);
      });
    });
    document.getElementById("pt-sign-page").addEventListener("change", function () {
      signPlaced = null;
      renderSignPreview().catch(function () {});
    });
    document.getElementById("pt-sign-preview").addEventListener("click", function (e) {
      var r = this.getBoundingClientRect();
      signPlaced = { nx: (e.clientX - r.left) / r.width, ny: (e.clientY - r.top) / r.height };
      renderSignPreview().catch(function () {});
    });
  })();

  drop.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { addFiles(fileInput.files); fileInput.value = ""; });
  drop.addEventListener("dragover", function (e) { e.preventDefault(); drop.style.borderColor = "#1c54e5"; });
  drop.addEventListener("dragleave", function () { drop.style.borderColor = "#cdd9f3"; });
  drop.addEventListener("drop", function (e) {
    e.preventDefault(); drop.style.borderColor = "#cdd9f3";
    addFiles(e.dataTransfer.files);
  });

  runBtn.addEventListener("click", function () {
    runBtn.disabled = true;
    resultEl.innerHTML = "";
    setStatus("Working…", "Traitement…");
    RUNNERS[tool]().then(function () {
      setStatus("Done.", "Terminé.");
    }).catch(function (e) {
      setStatus("Error: " + e.message, "Erreur : " + e.message);
    }).then(function () { runBtn.disabled = files.length < TOOLS[tool].min; });
  });

  refresh();
})();
