/*
 * CIFAR-100 dataset explorer (inspired by the MNIST latent-space demo).
 * Right: a 3D point cloud where each real CIFAR-100 thumbnail is a dot
 * coloured by its class. Drag to rotate the cloud; move the cursor and the
 * left panel shows the image at the nearest point.
 */
(function () {
  "use strict";

  var mapC = document.getElementById("cifar-map");
  var imgC = document.getElementById("cifar-img");
  if (!mapC || !imgC) return;
  var mctx = mapC.getContext("2d");
  var ictx = imgC.getContext("2d");

  // one distinct colour per class, generated with golden-angle hue steps
  var COLORS = [];
  function makeColors(n) {
    COLORS = [];
    for (var k = 0; k < n; k++) {
      var h = (k * 137.508) % 360;
      var l = 38 + 18 * (k % 3);           // 38 / 56 / 74 % lightness bands
      COLORS.push("hsl(" + h.toFixed(1) + ",85%," + l + "%)");
    }
  }

  var data = null, sprite = null, ready = false;

  // deterministic PRNG so the 3D layout is stable across reloads
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 3D positions: class centers on a fibonacci sphere, points jittered around them
  var pts = [];                    // [{x,y,z,c,i}]
  function build3D() {
    var n = data.classes.length, centers = [], golden = Math.PI * (3 - Math.sqrt(5));
    var rnd = mulberry32(7);
    for (var k = 0; k < n; k++) {
      var yy = 1 - 2 * (k + 0.5) / n, rr = Math.sqrt(1 - yy * yy), th = golden * k;
      // vary the shell radius a bit so 100 clusters fill the volume, not one shell
      var R = 0.42 + 0.32 * rnd();
      centers.push([R * rr * Math.cos(th), R * yy, R * rr * Math.sin(th)]);
    }
    var it = data.items, J = 0.07;   // tight jitter: many small clusters
    pts = [];
    for (var i = 0; i < it.length; i++) {
      var c = centers[it[i].c];
      // uniform jitter in a small ball
      var jx, jy, jz;
      do {
        jx = 2 * rnd() - 1; jy = 2 * rnd() - 1; jz = 2 * rnd() - 1;
      } while (jx * jx + jy * jy + jz * jz > 1);
      pts.push({ x: c[0] + J * jx, y: c[1] + J * jy, z: c[2] + J * jz, c: it[i].c, i: i });
    }
  }

  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var SIZE = 360;                  // map logical size (square)
  var nearest = 0;
  var yaw = 0.6, pitch = 0.35;     // view angles
  var pointer = { x: 0.5, y: 0.5, active: false };
  var autoSpin = true;

  var base = mapC.getAttribute("data-json");
  var spriteUrl = mapC.getAttribute("data-sprite");

  fetch(base).then(function (r) { return r.json(); }).then(function (j) {
    data = j;
    makeColors(data.classes.length);
    build3D();
    sprite = new Image();
    sprite.onload = function () { ready = true; drawImg(); requestAnimationFrame(tick); };
    sprite.src = spriteUrl;
  });

  function layoutSize() {
    var w = mapC.parentNode.clientWidth || 360;
    SIZE = Math.max(220, Math.min(360, w));
  }

  function resize() {
    layoutSize(); sizeCanvas();
    if (ready) drawImg();
  }
  function sizeCanvas() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    mapC.style.width = SIZE + "px"; mapC.style.height = SIZE + "px";
    mapC.width = SIZE * DPR; mapC.height = SIZE * DPR;
    mctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  // rotate + perspective-project a 3D point to screen space
  var CAM = 2.6, FOV = 1.9;
  function project(p) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var x = p.x * cy + p.z * sy;
    var z1 = -p.x * sy + p.z * cy;
    var y = p.y * cp - z1 * sp;
    var z = p.y * sp + z1 * cp;
    var s = FOV / (CAM - z);
    return { sx: SIZE * (0.5 + x * s), sy: SIZE * (0.5 - y * s), z: z, s: s };
  }

  var projected = [];              // last projected positions (for picking)
  function render() {
    if (!ready) return;
    mctx.clearRect(0, 0, SIZE, SIZE);
    mctx.fillStyle = "#fff";
    mctx.fillRect(0, 0, SIZE, SIZE);

    projected = [];
    var i, pr;
    for (i = 0; i < pts.length; i++) {
      pr = project(pts[i]);
      pr.c = pts[i].c; pr.i = pts[i].i;
      projected.push(pr);
    }
    // painter's algorithm: far points first
    var order = projected.slice().sort(function (a, b) { return a.z - b.z; });
    for (i = 0; i < order.length; i++) {
      pr = order[i];
      var rad = Math.max(1.6, 3.4 * pr.s);
      mctx.beginPath();
      mctx.arc(pr.sx, pr.sy, rad, 0, 6.2832);
      mctx.fillStyle = COLORS[pr.c];
      mctx.globalAlpha = 0.55 + 0.45 * Math.max(0, Math.min(1, (pr.z + 1) / 2));
      mctx.fill();
      mctx.globalAlpha = 1;
    }
    // highlight nearest point
    for (i = 0; i < projected.length; i++) {
      if (projected[i].i === nearest) {
        pr = projected[i];
        mctx.beginPath();
        mctx.arc(pr.sx, pr.sy, Math.max(5, 4.2 * pr.s) + 3, 0, 6.2832);
        mctx.strokeStyle = "rgba(40,40,40,0.9)"; mctx.lineWidth = 2.5; mctx.stroke();
        break;
      }
    }
  }

  function pickNearest() {
    if (!projected.length || !pointer.active) return;
    var mx = pointer.x * SIZE, my = pointer.y * SIZE;
    var best = 1e18, bi = nearest;
    for (var i = 0; i < projected.length; i++) {
      var dx = projected[i].sx - mx, dy = projected[i].sy - my;
      // bias toward closer (front) points when screen distances tie
      var d = dx * dx + dy * dy - projected[i].z * 4;
      if (d < best) { best = d; bi = projected[i].i; }
    }
    if (bi !== nearest) { nearest = bi; drawImg(); }
  }

  // ---- latent vector display: fixed random projection R^3 -> R^8 of the
  // point's 3D position, so nearby points show similar vectors ----
  var VDIM = 8, VW = [], vecRows = null;
  (function () {
    var rnd = mulberry32(11);
    for (var k = 0; k < VDIM; k++) {
      VW.push([2 * rnd() - 1, 2 * rnd() - 1, 2 * rnd() - 1, 0.6 * rnd() - 0.3]); // wx,wy,wz,b
    }
  })();
  function buildVecRows() {
    var box = document.getElementById("cifar-vec");
    if (!box) return;
    box.innerHTML = "";
    vecRows = [];
    for (var k = 0; k < VDIM; k++) {
      var row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:.4rem;margin-bottom:3px;";
      var barBox = document.createElement("div");
      barBox.style.cssText = "position:relative;width:84px;height:9px;background:#eef1f6;border-radius:4px;overflow:hidden;flex-shrink:0;";
      var bar = document.createElement("div");
      bar.style.cssText = "position:absolute;top:0;height:100%;border-radius:4px;";
      barBox.appendChild(bar);
      var num = document.createElement("span");
      num.style.cssText = "font-size:.68rem;color:#566075;font-variant-numeric:tabular-nums;min-width:3rem;";
      row.appendChild(barBox); row.appendChild(num);
      box.appendChild(row);
      vecRows.push({ bar: bar, num: num });
    }
  }
  function updateVec() {
    if (!vecRows) buildVecRows();
    if (!vecRows) return;
    var p = pts[nearest];
    for (var k = 0; k < VDIM; k++) {
      var w = VW[k];
      var v = w[0] * p.x + w[1] * p.y + w[2] * p.z + w[3];
      var frac = Math.max(-1, Math.min(1, v / 1.2));     // bar scale
      var half = 42;                                      // px, half of bar box
      if (frac >= 0) {
        vecRows[k].bar.style.left = half + "px";
        vecRows[k].bar.style.width = (frac * half) + "px";
        vecRows[k].bar.style.background = "#1c54e5";
      } else {
        vecRows[k].bar.style.left = (half + frac * half) + "px";
        vecRows[k].bar.style.width = (-frac * half) + "px";
        vecRows[k].bar.style.background = "#f57c00";
      }
      vecRows[k].num.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
    }
  }

  function drawImg() {
    if (!ready) return;
    var it = data.items[nearest], T = data.thumb;
    var L = imgC.width;            // device px (square)
    ictx.imageSmoothingEnabled = false;
    ictx.clearRect(0, 0, L, L);
    ictx.drawImage(sprite, it.g * T, it.r * T, T, T, 0, 0, L, L);
    var label = document.getElementById("cifar-label");
    if (label) label.textContent = data.classes[it.c].replace(/_/g, " ");
    updateVec();
  }

  function tick() {
    if (autoSpin && !dragging) yaw += 0.004;
    render();
    pickNearest();
    requestAnimationFrame(tick);
  }

  // ---- pointer: drag rotates, move picks nearest ----
  var dragging = false, lastX = 0, lastY = 0;
  function localPos(e) {
    var rect = mapC.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    return {
      x: Math.max(0, Math.min(1, (src.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (src.clientY - rect.top) / rect.height)),
      cx: src.clientX, cy: src.clientY
    };
  }
  mapC.addEventListener("mousedown", function (e) {
    dragging = true; autoSpin = false;
    var p = localPos(e); lastX = p.cx; lastY = p.cy;
  });
  window.addEventListener("mousemove", function (e) {
    var p = localPos(e);
    if (dragging) {
      yaw += (p.cx - lastX) * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + (p.cy - lastY) * 0.01));
      lastX = p.cx; lastY = p.cy;
    }
    if (e.target === mapC || dragging) {
      pointer.x = p.x; pointer.y = p.y; pointer.active = true;
    }
  });
  window.addEventListener("mouseup", function () { dragging = false; });
  mapC.addEventListener("mouseleave", function () { if (!dragging) pointer.active = false; });
  mapC.addEventListener("touchstart", function (e) {
    e.preventDefault(); dragging = true; autoSpin = false;
    var p = localPos(e); lastX = p.cx; lastY = p.cy;
    pointer.x = p.x; pointer.y = p.y; pointer.active = true;
  }, { passive: false });
  mapC.addEventListener("touchmove", function (e) {
    e.preventDefault();
    if (!dragging) return;
    var p = localPos(e);
    yaw += (p.cx - lastX) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (p.cy - lastY) * 0.01));
    lastX = p.cx; lastY = p.cy;
    pointer.x = p.x; pointer.y = p.y; pointer.active = true;
  }, { passive: false });
  mapC.addEventListener("touchend", function () { dragging = false; });

  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
  layoutSize(); sizeCanvas();
  window.__cifarResize = resize;
})();
