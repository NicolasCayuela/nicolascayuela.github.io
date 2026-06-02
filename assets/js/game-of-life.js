/*
 * Two cellular automata in one widget:
 *   - Conway's Game of Life (draw cells, stamp classic patterns)
 *   - Forest fire (Drossel-Schwabe): trees grow, lightning/neighbours ignite.
 * Toroidal grid. Bilingual labels live in the HTML include.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("gol-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var wrap = document.getElementById("gol-wrap");

  var CELL = 13, DPR = 1, cols = 0, rows = 0;
  var grid, next;
  var mode = "life";           // "life" | "fire"
  var fireDirty = false;       // grid was overwritten by forest fire -> reset on return to life
  var running = false;
  var acc = 0, last = 0;

  // forest-fire colours / cell states: 0 empty, 1 tree, 2 burning
  var COL_EMPTY = "#efe7d6", COL_TREE = "#2f9e44", COL_BURN = "#ff5a1f";

  // ---- Game of Life patterns (relative cell coords) ----
  var PATTERNS = {
    glider: [[1,0],[2,1],[0,2],[1,2],[2,2]],
    lwss: [[1,0],[4,0],[0,1],[0,2],[4,2],[0,3],[1,3],[2,3],[3,3]],
    mwss: [[1,0],[4,0],[0,1],[0,2],[5,2],[0,3],[1,3],[2,3],[3,3],[4,3],[2,4]],
    blinker: [[0,0],[1,0],[2,0]],
    toad: [[1,0],[2,0],[3,0],[0,1],[1,1],[2,1]],
    beacon: [[0,0],[1,0],[0,1],[1,1],[2,2],[3,2],[2,3],[3,3]],
    block: [[0,0],[1,0],[0,1],[1,1]],
    beehive: [[1,0],[2,0],[0,1],[3,1],[1,2],[2,2]],
    loaf: [[1,0],[2,0],[0,1],[3,1],[1,2],[3,2],[2,3]],
    pulsar: [
      [2,0],[3,0],[4,0],[8,0],[9,0],[10,0],
      [0,2],[5,2],[7,2],[12,2],[0,3],[5,3],[7,3],[12,3],[0,4],[5,4],[7,4],[12,4],
      [2,5],[3,5],[4,5],[8,5],[9,5],[10,5],
      [2,7],[3,7],[4,7],[8,7],[9,7],[10,7],
      [0,8],[5,8],[7,8],[12,8],[0,9],[5,9],[7,9],[12,9],[0,10],[5,10],[7,10],[12,10],
      [2,12],[3,12],[4,12],[8,12],[9,12],[10,12]
    ],
    pentadecathlon: [
      [2,0],[7,0],
      [0,1],[1,1],[3,1],[4,1],[5,1],[6,1],[8,1],[9,1],
      [2,2],[7,2]
    ],
    rpentomino: [[1,0],[2,0],[0,1],[1,1],[1,2]],
    acorn: [[1,0],[3,1],[0,2],[1,2],[4,2],[5,2],[6,2]],
    diehard: [[6,0],[0,1],[1,1],[1,2],[5,2],[6,2],[7,2]],
    gun: [
      [24,0],[22,1],[24,1],
      [12,2],[13,2],[20,2],[21,2],[34,2],[35,2],
      [11,3],[15,3],[20,3],[21,3],[34,3],[35,3],
      [0,4],[1,4],[10,4],[16,4],[20,4],[21,4],
      [0,5],[1,5],[10,5],[14,5],[16,5],[17,5],[22,5],[24,5],
      [10,6],[16,6],[24,6],[11,7],[15,7],[12,8],[13,8]
    ]
  };
  var activePattern = null;    // null => pen mode (life)
  var firePaint = "plant";     // "plant" | "ignite"

  function idx(c, r) { return r * cols + c; }
  // nerfed scales: slider 0..100 -> growth 0..0.01, lightning 0..0.0001
  function growthP() { var el = document.getElementById("gol-growth"); return el ? (+el.value) / 10000 : 0.003; }
  function lightP() { var el = document.getElementById("gol-light"); return el ? (+el.value) / 1000000 : 0.00003; }

  function build() {
    var w = wrap.clientWidth || 600;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    cols = Math.max(10, Math.floor(w / CELL));
    rows = 26;
    var cssW = cols * CELL, cssH = rows * CELL;
    canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    canvas.width = cssW * DPR; canvas.height = cssH * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    var ng = new Uint8Array(cols * rows);
    if (grid) {
      var oc = window.__golCols || cols, or = window.__golRows || rows;
      for (var r = 0; r < Math.min(or, rows); r++)
        for (var c = 0; c < Math.min(oc, cols); c++)
          ng[idx(c, r)] = grid[r * oc + c] || 0;
    }
    grid = ng; next = new Uint8Array(cols * rows);
    window.__golCols = cols; window.__golRows = rows;
    draw();
  }

  function draw() {
    var cssW = cols * CELL, cssH = rows * CELL, r, c, v;
    ctx.clearRect(0, 0, cssW, cssH);
    if (mode === "fire") { ctx.fillStyle = COL_EMPTY; ctx.fillRect(0, 0, cssW, cssH); }
    ctx.strokeStyle = "rgba(0,0,0,0.07)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = 0; x <= cols; x++) { ctx.moveTo(x * CELL + 0.5, 0); ctx.lineTo(x * CELL + 0.5, cssH); }
    for (var y = 0; y <= rows; y++) { ctx.moveTo(0, y * CELL + 0.5); ctx.lineTo(cssW, y * CELL + 0.5); }
    ctx.stroke();
    for (r = 0; r < rows; r++) {
      for (c = 0; c < cols; c++) {
        v = grid[idx(c, r)];
        if (mode === "life") {
          if (v) { ctx.fillStyle = "#111"; ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1); }
        } else if (v === 1) {
          ctx.fillStyle = COL_TREE; ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1);
        } else if (v === 2) {
          ctx.fillStyle = COL_BURN; ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1);
        }
      }
    }
  }

  function stepLife() {
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var n = 0;
      for (var dr = -1; dr <= 1; dr++) for (var dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        n += grid[idx((c + dc + cols) % cols, (r + dr + rows) % rows)];
      }
      var a = grid[idx(c, r)];
      next[idx(c, r)] = (a && (n === 2 || n === 3)) || (!a && n === 3) ? 1 : 0;
    }
  }

  function stepFire() {
    var p = growthP(), f = lightP();
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var v = grid[idx(c, r)], out;
      if (v === 2) { out = 0; }                       // burning -> empty
      else if (v === 1) {                              // tree
        var burningNb = false;
        for (var dr = -1; dr <= 1 && !burningNb; dr++) for (var dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          if (grid[idx((c + dc + cols) % cols, (r + dr + rows) % rows)] === 2) { burningNb = true; break; }
        }
        out = (burningNb || Math.random() < f) ? 2 : 1; // neighbour or lightning
      } else { out = Math.random() < p ? 1 : 0; }       // empty -> maybe grow
      next[idx(c, r)] = out;
    }
  }

  function step() {
    if (mode === "life") stepLife(); else stepFire();
    var t = grid; grid = next; next = t; draw();
  }

  function loop(now) {
    if (!running) return;
    requestAnimationFrame(loop);
    if (!last) last = now;
    acc += (now - last) / 1000; last = now;
    var interval = 1 / (mode === "fire" ? 14 : 11);
    if (acc >= interval) { acc = 0; step(); }
  }

  var playBtn = document.getElementById("gol-play");
  function setRunning(on) {
    running = on; last = 0; acc = 0;
    playBtn.innerHTML = on
      ? '<i class="fas fa-pause"></i> <span class="lang-en">Pause</span><span class="lang-zh">Pause</span>'
      : '<i class="fas fa-play"></i> <span class="lang-en">Play</span><span class="lang-zh">Lancer</span>';
    if (on) requestAnimationFrame(loop);
  }

  // ---- pointer ----
  function cellAt(e) {
    var rect = canvas.getBoundingClientRect();
    return { c: Math.floor((e.clientX - rect.left) / CELL), r: Math.floor((e.clientY - rect.top) / CELL) };
  }
  function inBounds(p) { return p.c >= 0 && p.c < cols && p.r >= 0 && p.r < rows; }

  function stamp(pat, c0, r0) {
    var maxc = 0, maxr = 0, i;
    for (i = 0; i < pat.length; i++) { if (pat[i][0] > maxc) maxc = pat[i][0]; if (pat[i][1] > maxr) maxr = pat[i][1]; }
    var oc = c0 - (maxc >> 1), or = r0 - (maxr >> 1);
    for (i = 0; i < pat.length; i++) {
      grid[idx(((oc + pat[i][0]) % cols + cols) % cols, ((or + pat[i][1]) % rows + rows) % rows)] = 1;
    }
    draw();
  }

  var painting = false, paintVal = 1;
  function apply(p) {
    if (mode === "fire") {
      grid[idx(p.c, p.r)] = firePaint === "ignite" ? 2 : 1;
    } else {
      grid[idx(p.c, p.r)] = paintVal;
    }
    draw();
  }
  canvas.addEventListener("mousedown", function (e) {
    var p = cellAt(e); if (!inBounds(p)) return;
    if (mode === "life" && activePattern) { stamp(PATTERNS[activePattern], p.c, p.r); return; }
    if (mode === "life") paintVal = grid[idx(p.c, p.r)] ? 0 : 1;
    painting = true; apply(p);
  });
  canvas.addEventListener("mousemove", function (e) {
    if (!painting) return;
    if (mode === "life" && activePattern) return;
    var p = cellAt(e); if (inBounds(p)) apply(p);
  });
  window.addEventListener("mouseup", function () { painting = false; });
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault();
    var p = cellAt(e.touches[0]); if (!inBounds(p)) return;
    if (mode === "life" && activePattern) { stamp(PATTERNS[activePattern], p.c, p.r); return; }
    if (mode === "life") grid[idx(p.c, p.r)] = grid[idx(p.c, p.r)] ? 0 : 1; else apply(p);
    draw();
  }, { passive: false });

  // ---- controls ----
  function on(id, fn) { var el = document.getElementById(id); if (el) el.addEventListener("click", fn); }
  on("gol-play", function () { setRunning(!running); });
  on("gol-step", function () { setRunning(false); step(); });
  on("gol-clear", function () { grid = new Uint8Array(cols * rows); setRunning(false); draw(); });
  on("gol-random", function () {
    if (mode === "fire") {
      for (var i = 0; i < grid.length; i++) grid[i] = Math.random() < 0.6 ? 1 : 0;
      grid[(Math.random() * grid.length) | 0] = 2;       // one spark
    } else {
      for (var j = 0; j < grid.length; j++) grid[j] = Math.random() < 0.28 ? 1 : 0;
    }
    draw();
  });

  // Game of Life pattern buttons
  var patBtns = document.querySelectorAll("[data-gol-pattern]");
  function setActive(name, btn) {
    activePattern = name;
    for (var i = 0; i < patBtns.length; i++) patBtns[i].classList.remove("active");
    if (btn) btn.classList.add("active");
    canvas.style.cursor = name ? "crosshair" : "pointer";
  }
  for (var pi = 0; pi < patBtns.length; pi++) (function (b) {
    b.addEventListener("click", function () {
      var name = b.getAttribute("data-gol-pattern");
      if (name === "") { setActive(null, b); return; }
      setActive(activePattern === name ? null : name, activePattern === name ? null : b);
    });
  })(patBtns[pi]);

  // Forest-fire paint buttons
  var fireBtns = document.querySelectorAll("[data-gol-fire]");
  for (var fi = 0; fi < fireBtns.length; fi++) (function (b) {
    b.addEventListener("click", function () {
      firePaint = b.getAttribute("data-gol-fire");
      for (var i = 0; i < fireBtns.length; i++) fireBtns[i].classList.remove("active");
      b.classList.add("active");
      canvas.style.cursor = "crosshair";
    });
  })(fireBtns[fi]);

  // Mode switch (life / fire / symmetry)
  function show(id, vis) { var el = document.getElementById(id); if (el) el.classList.toggle("d-none", !vis); }
  function setMode(m, btn) {
    setRunning(false);
    document.getElementById("gol-mode-life").classList.toggle("active", m === "life");
    document.getElementById("gol-mode-fire").classList.toggle("active", m === "fire");
    var symBtn = document.getElementById("gol-mode-sym");
    if (symBtn) symBtn.classList.toggle("active", m === "sym");
    var gdBtn = document.getElementById("gol-mode-gd");
    if (gdBtn) gdBtn.classList.toggle("active", m === "gd");

    show("gol-area", m === "life" || m === "fire");
    show("sym-area", m === "sym");
    show("gd-area", m === "gd");

    if (m === "sym") { if (window.__symResize) window.__symResize(); return; }
    if (m === "gd") { if (window.__gdResize) window.__gdResize(); return; }

    mode = m;
    show("gol-life-controls", m === "life");
    show("gol-fire-controls", m === "fire");
    canvas.style.cursor = "pointer";
    if (m === "fire") {
      for (var j = 0; j < grid.length; j++) grid[j] = Math.random() < 0.6 ? 1 : 0; // fresh random forest
      fireDirty = true;
    } else if (m === "life" && fireDirty) {        // returning from fire -> clean Life board
      grid = new Uint8Array(cols * rows);
      stamp(PATTERNS.glider, 4, 3);
      fireDirty = false;
    }
    draw();
  }
  on("gol-mode-life", function () { setMode("life"); });
  on("gol-mode-fire", function () { setMode("fire"); });
  on("gol-mode-sym", function () { setMode("sym"); });
  on("gol-mode-gd", function () { setMode("gd"); });

  // ---- init ----
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(build, 200); });
  build();
  stamp(PATTERNS.glider, 4, 3);
  setRunning(false);
})();
