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
    pdf2md:   { accept: "application/pdf", multiple: false, min: 1, opts: [],
                hint: ["Extracts the text as Markdown (headings inferred from font size) and keeps the embedded images - download is a ZIP with document.md + images/.",
                       "Extrait le texte en Markdown (titres déduits de la taille de police) et conserve les images intégrées - téléchargement en ZIP avec document.md + images/."] }
  };
  var OPT_IDS = ["range", "angle", "quality", "scale", "format"];

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

  // ---- PDF -> Markdown with image preservation ----
  function imageToPngBlob(img) {
    // img comes from PDF.js' object store: either {bitmap} or raw {data,width,height,kind}
    var c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    var cx = c.getContext("2d");
    if (img.bitmap) {
      cx.drawImage(img.bitmap, 0, 0);
    } else if (img.data) {
      var id = cx.createImageData(img.width, img.height);
      var n = img.width * img.height, d = img.data, o = id.data;
      if (d.length === n * 4) { o.set(d); }
      else if (d.length === n * 3) {
        for (var p = 0; p < n; p++) { o[p*4] = d[p*3]; o[p*4+1] = d[p*3+1]; o[p*4+2] = d[p*3+2]; o[p*4+3] = 255; }
      } else if (d.length === n) {                     // 8-bit grayscale
        for (var g = 0; g < n; g++) { o[g*4] = o[g*4+1] = o[g*4+2] = d[g]; o[g*4+3] = 255; }
      } else { return Promise.resolve(null); }         // unsupported (1bpp masks…)
      cx.putImageData(id, 0, 0);
    } else { return Promise.resolve(null); }
    return new Promise(function (res) { c.toBlob(res, "image/png"); });
  }

  async function doPdf2Md() {
    await needPdfJs(); await needZip();
    var doc = await pdfjsLib.getDocument({ data: await files[0].arrayBuffer() }).promise;
    var md = [], images = [], seenObj = {};

    for (var i = 1; i <= doc.numPages; i++) {
      setStatus("Converting page " + i + "/" + doc.numPages + "…", "Conversion page " + i + "/" + doc.numPages + "…");
      var page = await doc.getPage(i);

      // -- text: rebuild lines, infer headings from the font-size distribution --
      var tc = await page.getTextContent();
      var items = tc.items.filter(function (it) { return it.str && it.str.trim(); })
        .map(function (it) {
          return { str: it.str, x: it.transform[4], y: it.transform[5],
                   size: Math.hypot(it.transform[2], it.transform[3]) };
        });
      items.sort(function (a, b) { return b.y - a.y || a.x - b.x; });

      var lines = [];
      items.forEach(function (it) {
        var L = lines[lines.length - 1];
        if (L && Math.abs(L.y - it.y) < Math.max(L.size, it.size) * 0.5) {
          L.str += (it.x - L.endX > it.size * 0.25 ? " " : "") + it.str;
          L.endX = it.x + it.str.length * it.size * 0.5;
          L.size = Math.max(L.size, it.size);
        } else {
          lines.push({ str: it.str, y: it.y, size: it.size,
                       endX: it.x + it.str.length * it.size * 0.5 });
        }
      });

      var sizes = lines.map(function (l) { return l.size; }).sort(function (a, b) { return a - b; });
      var med = sizes.length ? sizes[Math.floor(sizes.length / 2)] : 12;

      var prev = null;
      lines.forEach(function (l) {
        var txt = l.str.trim();
        if (!txt) return;
        var pre = "";
        if (l.size >= med * 1.7) pre = "# ";
        else if (l.size >= med * 1.35) pre = "## ";
        else if (l.size >= med * 1.15) pre = "### ";
        if (prev !== null && (prev.y - l.y) > prev.size * 1.7) md.push("");   // paragraph gap
        if (pre && md.length && md[md.length - 1] !== "") md.push("");
        md.push(pre + txt);
        if (pre) md.push("");
        prev = l;
      });

      // -- images: pull every painted XObject out of the operator list --
      try {
        var ops = await page.getOperatorList();
        for (var k = 0; k < ops.fnArray.length; k++) {
          if (ops.fnArray[k] !== pdfjsLib.OPS.paintImageXObject) continue;
          var objId = ops.argsArray[k][0];
          if (seenObj[objId]) continue;
          seenObj[objId] = true;
          var img = null;
          try { img = page.objs.get(objId); }
          catch (e) { try { img = page.commonObjs.get(objId); } catch (e2) {} }
          if (!img || img.width < 8 || img.height < 8) continue;
          var blob = await imageToPngBlob(img);
          if (!blob) continue;
          var name = "p" + i + "-img" + (images.length + 1) + ".png";
          images.push({ name: name, blob: blob });
          md.push("");
          md.push("![" + name + "](images/" + name + ")");
          md.push("");
        }
      } catch (e) { /* image extraction is best-effort */ }

      md.push("");
    }

    var mdText = md.join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
    var base = baseName(files[0].name);
    if (!images.length) {
      downloadBlob(new Blob([mdText], { type: "text/markdown" }), base + ".md");
      return;
    }
    var zip = new JSZip();
    zip.file("document.md", mdText);
    var dir = zip.folder("images");
    images.forEach(function (im) { dir.file(im.name, im.blob); });
    var zblob = await zip.generateAsync({ type: "blob" });
    resultEl.innerHTML = resultLink(zblob, base + "-markdown.zip",
      "Download ZIP (.md + " + images.length + " images)",
      "Télécharger le ZIP (.md + " + images.length + " images)");
  }

  var RUNNERS = { merge: doMerge, split: doSplit, compress: doCompress, rotate: doRotate,
                  delete: doDelete, img2pdf: doImg2Pdf, pdf2img: doPdf2Img, pdf2md: doPdf2Md };

  // ---------- UI wiring ----------
  function refresh() {
    var t = TOOLS[tool];
    fileInput.accept = t.accept;
    fileInput.multiple = t.multiple;
    OPT_IDS.forEach(function (o) {
      document.getElementById("pt-opt-" + o).classList.toggle("d-none", t.opts.indexOf(o) === -1);
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
      var okType = t.accept === "image/*" ? f.type.indexOf("image/") === 0
                                          : f.type === "application/pdf" || /\.pdf$/i.test(f.name);
      if (!okType) continue;
      if (!t.multiple) files = [];
      files.push(f);
    }
    resultEl.innerHTML = ""; setStatus("", "");
    refresh();
  }

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
