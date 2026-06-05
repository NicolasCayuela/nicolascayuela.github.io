/*
 * Word embeddings explorer: the 20000 most frequent words of GloVe 6B 50d
 * (word2vec-style embeddings). 3D PCA scatter, nearest-neighbour search and
 * vector arithmetic (king - man + woman ≈ queen), all in the browser.
 */
(function () {
  "use strict";

  var area = document.getElementById("w2v-area");
  var canvas = document.getElementById("w2v-canvas");
  if (!area || !canvas) return;
  var ctx = canvas.getContext("2d");

  var data = null, ready = false, loading = false;
  var index = {};                  // word -> row
  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var SIZE = 380;
  var yaw = 0.5, pitch = 0.3, autoSpin = true;
  var highlights = [];             // [{i, color, label}]
  var statusEl = document.getElementById("w2v-status");

  function setStatus(en, fr) {
    if (!statusEl) return;
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + fr + "</span>"
      : "&nbsp;";
  }

  // ---- math ----
  function cosine(a, b) {
    var dot = 0, na = 0, nb = 0;
    for (var k = 0; k < a.length; k++) { dot += a[k] * b[k]; na += a[k] * a[k]; nb += b[k] * b[k]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
  }
  function nearest(vec, exclude, topK) {
    var best = [];
    for (var i = 0; i < data.words.length; i++) {
      if (exclude.indexOf(i) !== -1) continue;
      var c = cosine(vec, data.v[i]);
      if (best.length < topK || c > best[best.length - 1].c) {
        best.push({ i: i, c: c });
        best.sort(function (a, b) { return b.c - a.c; });
        if (best.length > topK) best.pop();
      }
    }
    return best;
  }

  // ---- 3D scatter ----
  function layoutSize() {
    var w = canvas.parentNode.clientWidth || 380;
    SIZE = Math.max(260, Math.min(420, w));
    canvas.style.width = SIZE + "px"; canvas.style.height = SIZE + "px";
    canvas.width = SIZE * DPR; canvas.height = SIZE * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  var CAM = 2.8, FOV = 1.6, zoom = 1;
  function project(p) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var x = p[0] * cy + p[2] * sy;
    var z1 = -p[0] * sy + p[2] * cy;
    var y = p[1] * cp - z1 * sp;
    var z = p[1] * sp + z1 * cp;
    var s = zoom * FOV / (CAM - z);
    return { sx: SIZE * (0.5 + x * s), sy: SIZE * (0.5 - y * s), z: z };
  }
  function render() {
    if (!ready) return;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, SIZE, SIZE);
    var i, pr;
    ctx.fillStyle = "rgba(90,100,120,0.45)";
    for (i = 0; i < data.xyz.length; i++) {
      pr = project(data.xyz[i]);
      ctx.fillRect(pr.sx - 1, pr.sy - 1, 2, 2);
    }
    // highlighted words on top, with labels
    for (i = 0; i < highlights.length; i++) {
      var h = highlights[i];
      pr = project(data.xyz[h.i]);
      ctx.beginPath();
      ctx.arc(pr.sx, pr.sy, 4.5, 0, 6.2832);
      ctx.fillStyle = h.color; ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.font = "bold 12px sans-serif";
      ctx.fillStyle = h.color;
      ctx.fillText(data.words[h.i], pr.sx + 7, pr.sy - 6);
    }
  }
  function tick() {
    if (autoSpin && !dragging) { yaw += 0.0035; render(); }
    requestAnimationFrame(tick);
  }

  // ---- interactions ----
  var dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("mousedown", function (e) { dragging = true; autoSpin = false; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    yaw += (e.clientX - lastX) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - lastY) * 0.01));
    lastX = e.clientX; lastY = e.clientY;
    render();
  });
  window.addEventListener("mouseup", function () { dragging = false; });
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault(); dragging = true; autoSpin = false;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  }, { passive: false });
  canvas.addEventListener("touchmove", function (e) {
    e.preventDefault(); if (!dragging) return;
    yaw += (e.touches[0].clientX - lastX) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.touches[0].clientY - lastY) * 0.01));
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    render();
  }, { passive: false });
  canvas.addEventListener("touchend", function () { dragging = false; pinchDist = 0; });

  // ---- zoom: mouse wheel + two-finger pinch ----
  function setZoom(z) { zoom = Math.max(0.5, Math.min(6, z)); render(); }
  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    setZoom(zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12));
  }, { passive: false });
  var pinchDist = 0;
  function dist2(t) {
    var dx = t[0].clientX - t[1].clientX, dy = t[0].clientY - t[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }
  canvas.addEventListener("touchstart", function (e) {
    if (e.touches.length === 2) { e.preventDefault(); dragging = false; pinchDist = dist2(e.touches); }
  }, { passive: false });
  canvas.addEventListener("touchmove", function (e) {
    if (e.touches.length === 2 && pinchDist > 0) {
      e.preventDefault();
      var d = dist2(e.touches);
      setZoom(zoom * d / pinchDist);
      pinchDist = d;
    }
  }, { passive: false });

  function lookup(w) {
    if (!w) return -1;
    w = w.trim().toLowerCase();
    return index.hasOwnProperty(w) ? index[w] : -1;
  }

  // clickable word chips with a similarity bar; clicking explores that word
  function showList(el, items) {
    el.innerHTML = "";
    items.forEach(function (it) {
      var chip = document.createElement("span");
      chip.className = "wv-chip";
      var frac = Math.max(0, Math.min(1, it.c));
      chip.innerHTML = "<strong>" + data.words[it.i] + "</strong>" +
        '<span class="wv-bar"><span style="width:' + Math.round(frac * 100) + '%"></span></span>' +
        '<span class="text-muted" style="font-size:.68rem;">' + it.c.toFixed(2) + "</span>";
      chip.title = it.c.toFixed(3);
      chip.addEventListener("click", function () {
        document.getElementById("w2v-search").value = data.words[it.i];
        doSearch();
      });
      el.appendChild(chip);
    });
  }

  function doSearch() {
    if (!ready) return;
    var inp = document.getElementById("w2v-search");
    var i = lookup(inp.value);
    var list = document.getElementById("w2v-neighbors");
    if (i < 0) {
      setStatus('"' + inp.value + '" is not in the 20,000-word vocabulary.',
                "« " + inp.value + " » n'est pas dans le vocabulaire de 20 000 mots.");
      return;
    }
    setStatus("", "");
    var nb = nearest(data.v[i], [i], 8);
    highlights = [{ i: i, color: "#e51c23" }];
    nb.forEach(function (n) { highlights.push({ i: n.i, color: "#1c54e5" }); });
    showList(list, nb);
    render();
  }

  function doMath() {
    if (!ready) return;
    var a = lookup(document.getElementById("w2v-a").value);
    var b = lookup(document.getElementById("w2v-b").value);
    var c = lookup(document.getElementById("w2v-c").value);
    var list = document.getElementById("w2v-result");
    if (a < 0 || b < 0 || c < 0) {
      setStatus("All three words must be in the 20,000-word vocabulary.",
                "Les trois mots doivent appartenir au vocabulaire de 20 000 mots.");
      return;
    }
    setStatus("", "");
    var dim = data.dim, target = new Array(dim);
    for (var k = 0; k < dim; k++) target[k] = data.v[a][k] - data.v[b][k] + data.v[c][k];
    var res = nearest(target, [a, b, c], 5);
    highlights = [
      { i: a, color: "#0a8f1a" }, { i: b, color: "#f57c00" }, { i: c, color: "#8e24aa" },
      { i: res[0].i, color: "#e51c23" }
    ];
    var best = document.getElementById("w2v-best");
    if (best) best.textContent = "= " + data.words[res[0].i];
    showList(list, res.slice(1));
    render();
  }

  function on(id, ev, fn) { var el = document.getElementById(id); if (el) el.addEventListener(ev, fn); }
  on("w2v-search-btn", "click", doSearch);
  on("w2v-search", "keydown", function (e) { if (e.key === "Enter") doSearch(); });
  on("w2v-math-btn", "click", doMath);
  ["w2v-a", "w2v-b", "w2v-c"].forEach(function (id) {
    on(id, "keydown", function (e) { if (e.key === "Enter") doMath(); });
  });
  // example chips
  Array.prototype.forEach.call(document.querySelectorAll("[data-w2v-word]"), function (el) {
    el.addEventListener("click", function () {
      document.getElementById("w2v-search").value = el.getAttribute("data-w2v-word");
      doSearch();
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-w2v-analogy]"), function (el) {
    el.addEventListener("click", function () {
      var p = el.getAttribute("data-w2v-analogy").split(",");
      document.getElementById("w2v-a").value = p[0];
      document.getElementById("w2v-b").value = p[1];
      document.getElementById("w2v-c").value = p[2];
      doMath();
    });
  });

  function init() {
    if (loading || ready) return;
    loading = true;
    setStatus("Loading embeddings (~5 MB)…", "Chargement des plongements (~5 Mo)…");
    layoutSize();
    fetch(area.getAttribute("data-json")).then(function (r) { return r.json(); }).then(function (j) {
      data = j;
      var sc = data.scale || 1;
      // xyz needs real units for the projection; v stays integer
      // (cosine and vector arithmetic are scale-invariant)
      for (var i = 0; i < data.xyz.length; i++) {
        data.xyz[i] = [data.xyz[i][0] / sc, data.xyz[i][1] / sc, data.xyz[i][2] / sc];
      }
      for (var w = 0; w < data.words.length; w++) index[data.words[w]] = w;
      ready = true; loading = false;
      setStatus("", "");
      // default demo: paris - france + italy = rome
      document.getElementById("w2v-a").value = "paris";
      document.getElementById("w2v-b").value = "france";
      document.getElementById("w2v-c").value = "italy";
      doMath();
      requestAnimationFrame(tick);
    }).catch(function (e) {
      loading = false;
      setStatus("Failed to load embeddings: " + e, "Échec du chargement : " + e);
    });
  }

  var rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt); rt = setTimeout(function () { layoutSize(); render(); }, 200);
  });
  window.__w2vShow = init;
})();
