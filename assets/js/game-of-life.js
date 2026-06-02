/*
 * Conway's Game of Life — small interactive widget.
 * Click / drag to draw black cells. Pick a pattern then click to stamp it.
 * Toroidal (wrap-around) grid so gliders keep travelling.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("gol-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var wrap = document.getElementById("gol-wrap");

  var CELL = 13;            // logical px per cell
  var DPR = 1;
  var cols = 0, rows = 0;
  var grid, next;
  var running = false;
  var speed = 11;          // generations per second
  var acc = 0, last = 0;

  // ---- patterns (relative cell coords) ----
  var PATTERNS = {
    glider: [[1,0],[2,1],[0,2],[1,2],[2,2]],
    blinker: [[0,0],[1,0],[2,0]],
    toad: [[1,0],[2,0],[3,0],[0,1],[1,1],[2,1]],
    beacon: [[0,0],[1,0],[0,1],[1,1],[2,2],[3,2],[2,3],[3,3]],
    lwss: [[1,0],[4,0],[0,1],[0,2],[4,2],[0,3],[1,3],[2,3],[3,3]],
    pulsar: [
      [2,0],[3,0],[4,0],[8,0],[9,0],[10,0],
      [0,2],[5,2],[7,2],[12,2],
      [0,3],[5,3],[7,3],[12,3],
      [0,4],[5,4],[7,4],[12,4],
      [2,5],[3,5],[4,5],[8,5],[9,5],[10,5],
      [2,7],[3,7],[4,7],[8,7],[9,7],[10,7],
      [0,8],[5,8],[7,8],[12,8],
      [0,9],[5,9],[7,9],[12,9],
      [0,10],[5,10],[7,10],[12,10],
      [2,12],[3,12],[4,12],[8,12],[9,12],[10,12]
    ],
    gun: [
      [24,0],
      [22,1],[24,1],
      [12,2],[13,2],[20,2],[21,2],[34,2],[35,2],
      [11,3],[15,3],[20,3],[21,3],[34,3],[35,3],
      [0,4],[1,4],[10,4],[16,4],[20,4],[21,4],
      [0,5],[1,5],[10,5],[14,5],[16,5],[17,5],[22,5],[24,5],
      [10,6],[16,6],[24,6],
      [11,7],[15,7],
      [12,8],[13,8]
    ]
  };

  var activePattern = null;   // null => pen mode (single cells)

  function idx(c, r) { return r * cols + c; }

  function build() {
    var w = wrap.clientWidth || 600;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    cols = Math.max(10, Math.floor(w / CELL));
    rows = 26;
    var cssW = cols * CELL, cssH = rows * CELL;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = cssW * DPR;
    canvas.height = cssH * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    var ng = new Uint8Array(cols * rows);
    if (grid) {              // preserve overlap on resize
      var oc = window.__golCols || cols, or = window.__golRows || rows;
      for (var r = 0; r < Math.min(or, rows); r++)
        for (var c = 0; c < Math.min(oc, cols); c++)
          ng[idx(c, r)] = grid[r * oc + c] || 0;
    }
    grid = ng;
    next = new Uint8Array(cols * rows);
    window.__golCols = cols; window.__golRows = rows;
    draw();
  }

  function draw() {
    var cssW = cols * CELL, cssH = rows * CELL;
    ctx.clearRect(0, 0, cssW, cssH);
    // grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.07)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = 0; x <= cols; x++) { ctx.moveTo(x * CELL + 0.5, 0); ctx.lineTo(x * CELL + 0.5, cssH); }
    for (var y = 0; y <= rows; y++) { ctx.moveTo(0, y * CELL + 0.5); ctx.lineTo(cssW, y * CELL + 0.5); }
    ctx.stroke();
    // live cells
    ctx.fillStyle = "#111";
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        if (grid[idx(c, r)]) ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1);
      }
    }
  }

  function step() {
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var n = 0;
        for (var dr = -1; dr <= 1; dr++) {
          for (var dc = -1; dc <= 1; dc++) {
            if (!dr && !dc) continue;
            var rr = (r + dr + rows) % rows;   // wrap (toroidal)
            var cc = (c + dc + cols) % cols;
            n += grid[idx(cc, rr)];
          }
        }
        var alive = grid[idx(c, r)];
        next[idx(c, r)] = (alive && (n === 2 || n === 3)) || (!alive && n === 3) ? 1 : 0;
      }
    }
    var t = grid; grid = next; next = t;
    draw();
  }

  function loop(now) {
    if (!running) return;
    requestAnimationFrame(loop);
    if (!last) last = now;
    acc += (now - last) / 1000; last = now;
    var interval = 1 / speed;
    if (acc >= interval) { acc = 0; step(); }
  }

  function setRunning(on) {
    running = on; last = 0; acc = 0;
    playBtn.innerHTML = on
      ? '<i class="fas fa-pause"></i> <span class="lang-en">Pause</span><span class="lang-zh">Pause</span>'
      : '<i class="fas fa-play"></i> <span class="lang-en">Play</span><span class="lang-zh">Lancer</span>';
    if (on) requestAnimationFrame(loop);
  }

  // ---- pointer interaction ----
  function cellAt(e) {
    var rect = canvas.getBoundingClientRect();
    var x = (e.clientX - rect.left), y = (e.clientY - rect.top);
    return { c: Math.floor(x / CELL), r: Math.floor(y / CELL) };
  }

  function stamp(pat, c0, r0) {
    // centre the pattern on the click
    var maxc = 0, maxr = 0, i;
    for (i = 0; i < pat.length; i++) { if (pat[i][0] > maxc) maxc = pat[i][0]; if (pat[i][1] > maxr) maxr = pat[i][1]; }
    var oc = c0 - (maxc >> 1), or = r0 - (maxr >> 1);
    for (i = 0; i < pat.length; i++) {
      var c = ((oc + pat[i][0]) % cols + cols) % cols;
      var r = ((or + pat[i][1]) % rows + rows) % rows;
      grid[idx(c, r)] = 1;
    }
    draw();
  }

  var painting = false, paintVal = 1;
  canvas.addEventListener("mousedown", function (e) {
    var p = cellAt(e);
    if (p.c < 0 || p.c >= cols || p.r < 0 || p.r >= rows) return;
    if (activePattern) { stamp(PATTERNS[activePattern], p.c, p.r); return; }
    paintVal = grid[idx(p.c, p.r)] ? 0 : 1;   // toggle based on first cell
    painting = true;
    grid[idx(p.c, p.r)] = paintVal;
    draw();
  });
  canvas.addEventListener("mousemove", function (e) {
    if (!painting || activePattern) return;
    var p = cellAt(e);
    if (p.c < 0 || p.c >= cols || p.r < 0 || p.r >= rows) return;
    grid[idx(p.c, p.r)] = paintVal;
    draw();
  });
  window.addEventListener("mouseup", function () { painting = false; });

  // touch support
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault();
    var t = e.touches[0], p = cellAt(t);
    if (p.c < 0 || p.c >= cols || p.r < 0 || p.r >= rows) return;
    if (activePattern) { stamp(PATTERNS[activePattern], p.c, p.r); return; }
    grid[idx(p.c, p.r)] = grid[idx(p.c, p.r)] ? 0 : 1; draw();
  }, { passive: false });

  // ---- controls ----
  var playBtn = document.getElementById("gol-play");
  function on(id, fn) { var el = document.getElementById(id); if (el) el.addEventListener("click", fn); }

  on("gol-play", function () { setRunning(!running); });
  on("gol-step", function () { setRunning(false); step(); });
  on("gol-clear", function () { grid = new Uint8Array(cols * rows); setRunning(false); draw(); });
  on("gol-random", function () {
    for (var i = 0; i < grid.length; i++) grid[i] = Math.random() < 0.28 ? 1 : 0;
    draw();
  });

  // pattern buttons set the active stamp; click again (or "Pen") to go back to drawing
  var patBtns = document.querySelectorAll("[data-gol-pattern]");
  function setActive(name, btn) {
    activePattern = name;
    for (var i = 0; i < patBtns.length; i++) patBtns[i].classList.remove("active");
    if (btn) btn.classList.add("active");
    canvas.style.cursor = name ? "crosshair" : "pointer";
  }
  for (var i = 0; i < patBtns.length; i++) {
    (function (b) {
      b.addEventListener("click", function () {
        var name = b.getAttribute("data-gol-pattern");
        if (name === "") { setActive(null, null); return; }      // Pen
        setActive(activePattern === name ? null : name, activePattern === name ? null : b);
      });
    })(patBtns[i]);
  }

  // ---- init ----
  var rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(build, 200);
  });
  build();

  // a friendly starting state: one glider
  stamp(PATTERNS.glider, 4, 3);
  setRunning(false);
})();
