/*
 * 3D gradient descent visualiser — from scratch (no three.js).
 * A loss surface z = f(x,y) is meshed, colour-mapped (jet) and projected to 2D
 * with a rotatable orthographic camera (drag to rotate). A marker rolls down
 * the surface following the gradient (SGD or momentum), leaving a trail.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("gd-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var wrap = document.getElementById("gd-wrap");

  var DPR = 1, W = 0, H = 0;
  var D = 2;                 // domain half-width: x,y in [-D, D]
  var N = 30;                // mesh resolution
  var ISO_YAW = Math.PI / 4, ISO_PITCH = 0.6155;   // isometric view
  var yaw = ISO_YAW, pitch = ISO_PITCH;

  // ---- loss surfaces (value + gradient) ----
  var SURF = {
    bowl: { f: function (x, y) { return 0.5 * (x * x + y * y); },
            g: function (x, y) { return [x, y]; }, lr: 0.25, start: 1.7 },
    saddle: { f: function (x, y) { return 0.5 * (x * x - y * y); },
              g: function (x, y) { return [x, -y]; }, lr: 0.12, start: 1.6, noMin: true },
    ripples: { f: function (x, y) { return Math.sin(2 * x) * Math.cos(2 * y) + 0.18 * (x * x + y * y); },
               g: function (x, y) { return [2 * Math.cos(2 * x) * Math.cos(2 * y) + 0.36 * x,
                                            -2 * Math.sin(2 * x) * Math.sin(2 * y) + 0.36 * y]; }, lr: 0.06, start: 1.8 },
    rosenbrock: { f: function (x, y) { var a = 1 - x, b = y - x * x; return 0.02 * (a * a + 20 * b * b); },
                  g: function (x, y) { var b = y - x * x;
                                       return [0.02 * (-2 * (1 - x) - 80 * x * b), 0.02 * (40 * b)]; }, lr: 0.04, start: 1.7 },
    volcano: { f: function (x, y) { var r = x * x + y * y - 1; return r * r; },               // ring of minima
               g: function (x, y) { var r = x * x + y * y - 1; return [4 * r * x, 4 * r * y]; }, lr: 0.08, start: 1.7 },
    fourwells: { f: function (x, y) { var a = x * x - 1, b = y * y - 1; return a * a + b * b; }, // 4 minima
                 g: function (x, y) { return [4 * x * (x * x - 1), 4 * y * (y * y - 1)]; }, lr: 0.06, start: 1.7 },
    monkey: { f: function (x, y) { return 0.3 * (x * x * x - 3 * x * y * y); },                 // monkey saddle (no min)
              g: function (x, y) { return [0.3 * (3 * x * x - 3 * y * y), 0.3 * (-6 * x * y)]; }, lr: 0.05, start: 1.4, noMin: true }
  };
  var surf = "ripples", optimizer = "momentum", lrScale = 1;

  // marker state
  var px = 0, py = 0, vx = 0, vy = 0, trail = [], paused = false, settled = false, settleTimer = 0;
  var aMx = 0, aMy = 0, aVx = 0, aVy = 0, aT = 0;   // Adam moments
  var target = null;   // global minimum (goal)

  function computeTarget() {
    if (SURF[surf].noMin) { target = null; return; }
    var f = SURF[surf].f, best = 1e9, bx = 0, by = 0, M = 180;
    for (var i = 0; i <= M; i++) for (var j = 0; j <= M; j++) {
      var x = -D + 2 * D * i / M, y = -D + 2 * D * j / M, z = f(x, y);
      if (z < best) { best = z; bx = x; by = y; }
    }
    target = [bx, by];
  }

  function jet(t) {
    if (t < 0) t = 0; else if (t > 1) t = 1;
    var r = Math.max(0, Math.min(1, Math.min(4 * t - 1.5, -4 * t + 4.5)));
    var g = Math.max(0, Math.min(1, Math.min(4 * t - 0.5, -4 * t + 3.5)));
    var b = Math.max(0, Math.min(1, Math.min(4 * t + 0.5, -4 * t + 2.5)));
    return "rgb(" + ((r * 255) | 0) + "," + ((g * 255) | 0) + "," + ((b * 255) | 0) + ")";
  }

  // mesh height cache for colour normalisation
  var zmin = 0, zmax = 1;
  function computeZRange() {
    var f = SURF[surf].f, lo = 1e9, hi = -1e9;
    for (var i = 0; i <= N; i++) for (var j = 0; j <= N; j++) {
      var x = -D + 2 * D * i / N, y = -D + 2 * D * j / N, z = f(x, y);
      if (z < lo) lo = z; if (z > hi) hi = z;
    }
    zmin = lo; zmax = hi <= lo ? lo + 1 : hi;
  }
  function zn(z) { return (z - zmin) / (zmax - zmin); }

  // project a world point (domain x,y + normalised height) to screen
  function project(x, y, zheight) {
    var ca = Math.cos(yaw), sa = Math.sin(yaw);
    var x1 = x * ca - y * sa, y1 = x * sa + y * ca, z1 = zheight;
    var cb = Math.cos(pitch), sb = Math.sin(pitch);
    var up = y1 * sb + z1 * cb;          // height goes UP on screen
    var depth = y1 * cb - z1 * sb;        // along the view direction (larger = farther)
    var scale = Math.min(W, H) / (2 * D) * 0.62;
    return { X: W / 2 + scale * x1, Y: H * 0.56 - scale * up, depth: depth };
  }

  function worldZ(z) { return (zn(z) - 0.45) * (D * 1.25); }   // height in domain units

  function resize() {
    var w = wrap.clientWidth || 600;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = w; H = 380;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    canvas.width = W * DPR; canvas.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    render();
  }

  function render() {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);

    var f = SURF[surf].f, quads = [], i, j;
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        var x0 = -D + 2 * D * i / N, x1 = -D + 2 * D * (i + 1) / N;
        var y0 = -D + 2 * D * j / N, y1 = -D + 2 * D * (j + 1) / N;
        var p1 = project(x0, y0, worldZ(f(x0, y0)));
        var p2 = project(x1, y0, worldZ(f(x1, y0)));
        var p3 = project(x1, y1, worldZ(f(x1, y1)));
        var p4 = project(x0, y1, worldZ(f(x0, y1)));
        var zc = (f(x0, y0) + f(x1, y0) + f(x1, y1) + f(x0, y1)) / 4;
        quads.push({ p: [p1, p2, p3, p4], depth: (p1.depth + p2.depth + p3.depth + p4.depth) / 4, t: zn(zc) });
      }
    }
    quads.sort(function (a, b) { return b.depth - a.depth; });   // painter: far (larger depth) first
    for (i = 0; i < quads.length; i++) {
      var q = quads[i].p;
      ctx.beginPath();
      ctx.moveTo(q[0].X, q[0].Y);
      ctx.lineTo(q[1].X, q[1].Y); ctx.lineTo(q[2].X, q[2].Y); ctx.lineTo(q[3].X, q[3].Y);
      ctx.closePath();
      ctx.fillStyle = jet(quads[i].t); ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.12)"; ctx.lineWidth = 0.5; ctx.stroke();
    }

    // trail
    if (trail.length > 1) {
      ctx.beginPath();
      for (i = 0; i < trail.length; i++) {
        var pp = project(trail[i][0], trail[i][1], worldZ(f(trail[i][0], trail[i][1])) + 0.03);
        if (i === 0) ctx.moveTo(pp.X, pp.Y); else ctx.lineTo(pp.X, pp.Y);
      }
      ctx.strokeStyle = "#111"; ctx.lineWidth = 2; ctx.stroke();
    }
    // global-minimum target (the goal)
    if (target) {
      var tp = project(target[0], target[1], worldZ(f(target[0], target[1])) + 0.04);
      ctx.beginPath(); ctx.arc(tp.X, tp.Y, 9, 0, 6.2832);
      ctx.strokeStyle = "#13a10e"; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tp.X - 12, tp.Y); ctx.lineTo(tp.X + 12, tp.Y);
      ctx.moveTo(tp.X, tp.Y - 12); ctx.lineTo(tp.X, tp.Y + 12);
      ctx.strokeStyle = "rgba(19,161,14,0.6)"; ctx.lineWidth = 1; ctx.stroke();
      var label = document.documentElement.classList.contains("lang-active-zh") ? "Minimum global" : "Global minimum";
      ctx.font = "600 12px Lato, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#13a10e";
      ctx.fillText(label, tp.X, tp.Y + 24);
    }
    // marker (green once it has reached the minimum)
    var mp = project(px, py, worldZ(f(px, py)) + 0.05);
    ctx.beginPath(); ctx.arc(mp.X, mp.Y, 6, 0, 6.2832);
    ctx.fillStyle = settled ? "#13a10e" : "#d61f1f"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }

  function restart() {
    var s = SURF[surf].start;
    px = (Math.random() * 2 - 1) * s; py = (Math.random() * 2 - 1) * s;
    vx = 0; vy = 0; aMx = 0; aMy = 0; aVx = 0; aVy = 0; aT = 0;
    trail = [[px, py]]; settled = false; settleTimer = 0; settleHold = 0;
  }

  function sliderFrac() { var el = document.getElementById("gd-lr"); return (el ? +el.value : 50) / 100; }   // 0.01..1

  function step() {
    var sx = px, sy = py;
    var g = SURF[surf].g(px, py), m = 0.85, f = sliderFrac();
    if (optimizer === "adam") {
      var b1 = 0.9, b2 = 0.999, eps = 1e-8, a = 0.05 * f;   // small Adam step
      aT++;
      aMx = b1 * aMx + (1 - b1) * g[0]; aMy = b1 * aMy + (1 - b1) * g[1];
      aVx = b2 * aVx + (1 - b2) * g[0] * g[0]; aVy = b2 * aVy + (1 - b2) * g[1] * g[1];
      var mhx = aMx / (1 - Math.pow(b1, aT)), mhy = aMy / (1 - Math.pow(b1, aT));
      var vhx = aVx / (1 - Math.pow(b2, aT)), vhy = aVy / (1 - Math.pow(b2, aT));
      px -= a * mhx / (Math.sqrt(vhx) + eps);
      py -= a * mhy / (Math.sqrt(vhy) + eps);
    } else {
      var L = SURF[surf].lr * f * 0.3;                      // drastically reduced lr
      if (optimizer === "momentum") {
        vx = m * vx - L * g[0]; vy = m * vy - L * g[1];
        px += vx; py += vy;
      } else {
        px -= L * g[0]; py -= L * g[1];
      }
    }
    // keep the marker inside the visible domain (saddle would otherwise fly off)
    if (px > D) { px = D; vx = 0; } else if (px < -D) { px = -D; vx = 0; }
    if (py > D) { py = D; vy = 0; } else if (py < -D) { py = -D; vy = 0; }
    trail.push([px, py]);
    if (trail.length > 400) trail.shift();
    // stop once it essentially stops moving (a minimum, or a clamped edge for the saddle)
    var moved = Math.sqrt((px - sx) * (px - sx) + (py - sy) * (py - sy));
    if (moved < 0.004) { settleTimer++; if (settleTimer > 30) settled = true; }
    else settleTimer = 0;
  }

  // ---- animation ----
  var frame = 0, settleHold = 0;
  function loop() {
    requestAnimationFrame(loop);
    if (canvas.offsetParent === null) return;     // hidden tab -> idle
    if (!paused) {
      if (!settled) { frame++; if (frame % 3 === 0) step(); }
      else { settleHold++; if (settleHold > 180) restart(); }   // ~3s after converging -> relaunch
    }
    render();
  }

  // ---- rotate by drag ----
  var dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("mousedown", function (e) { dragging = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    yaw -= (e.clientX - lx) * 0.01;
    pitch += (e.clientY - ly) * 0.01;
    lx = e.clientX; ly = e.clientY;
  });
  window.addEventListener("mouseup", function () { dragging = false; });
  canvas.addEventListener("touchstart", function (e) { dragging = true; lx = e.touches[0].clientX; ly = e.touches[0].clientY; }, { passive: true });
  canvas.addEventListener("touchmove", function (e) {
    if (!dragging) return;
    yaw -= (e.touches[0].clientX - lx) * 0.01;
    pitch += (e.touches[0].clientY - ly) * 0.01;
    lx = e.touches[0].clientX; ly = e.touches[0].clientY;
  }, { passive: true });
  canvas.addEventListener("touchend", function () { dragging = false; });

  // ---- controls ----
  function $(id) { return document.getElementById(id); }
  function setSurf(name, btn) {
    surf = name;
    var bs = document.querySelectorAll("[data-gd-surf]");
    for (var i = 0; i < bs.length; i++) bs[i].classList.toggle("active", bs[i] === btn);
    computeZRange(); computeTarget(); restart();
  }
  var sbtn = document.querySelectorAll("[data-gd-surf]");
  for (var si = 0; si < sbtn.length; si++) (function (b) {
    b.addEventListener("click", function () { setSurf(b.getAttribute("data-gd-surf"), b); });
  })(sbtn[si]);

  var obtn = document.querySelectorAll("[data-gd-opt]");
  for (var oi = 0; oi < obtn.length; oi++) (function (b) {
    b.addEventListener("click", function () {
      optimizer = b.getAttribute("data-gd-opt");
      for (var i = 0; i < obtn.length; i++) obtn[i].classList.toggle("active", obtn[i] === b);
      restart();
    });
  })(obtn[oi]);

  if ($("gd-play")) $("gd-play").addEventListener("click", function () {
    paused = !paused;
    this.innerHTML = paused
      ? '<i class="fas fa-play"></i> <span class="lang-en">Play</span><span class="lang-zh">Lancer</span>'
      : '<i class="fas fa-pause"></i> <span class="lang-en">Pause</span><span class="lang-zh">Pause</span>';
  });
  if ($("gd-restart")) $("gd-restart").addEventListener("click", restart);
  if ($("gd-resetview")) $("gd-resetview").addEventListener("click", function () { yaw = ISO_YAW; pitch = ISO_PITCH; });

  // ---- init ----
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
  computeZRange();
  computeTarget();
  restart();
  resize();
  requestAnimationFrame(loop);
  window.__gdResize = resize;     // re-layout when its tab is shown
})();
