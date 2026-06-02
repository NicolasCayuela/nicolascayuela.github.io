/*
 * Elastic-wave metamaterial background.
 *
 * A periodic lattice of nodes (the "unit cells" of a phononic crystal /
 * acoustic metamaterial). Radial elastic waves travel through it, displacing
 * the nodes; the links between neighbours light up with the local strain, so
 * you literally see the wavefronts ripple through the periodic medium.
 * Moving the pointer injects small ripples, clicking emits a strong pulse.
 *
 * Pure <canvas>, no dependency. Runs behind the page content.
 */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var CFG = {
    spacing: 46,        // lattice pitch in px (unit-cell size)
    jitter: 0.18,       // random lattice disorder (0 = perfect crystal)
    amp: 19,            // peak node displacement (px)
    wavelength: 190,    // spatial period of a wave (px)
    speed: 130,         // wave phase speed (px/s)
    waveLife: 15,       // seconds a ripple stays alive
    maxWaves: 40,       // perf backstop only; high so live ripples are never cut
    autoMin: 2,         // min seconds between random excitations
    autoMax: 5,         // max seconds between random excitations
    linkDist: 1.9,      // neighbour link cutoff, in lattice pitches
    baseAlpha: 0.22,    // resting link opacity (idle = faint blue, COMSOL low end)
    peakAlpha: 0.95,    // link opacity at the crest (COMSOL high end)
    nodeAlpha: 0.6,
    opacity: 0.6,       // whole-canvas opacity
    fps: 60
  };

  // COMSOL "Rainbow" (jet) colormap: t in [0,1] -> [r,g,b] 0..255
  function jet(t) {
    if (t < 0) t = 0; else if (t > 1) t = 1;
    var r = Math.max(0, Math.min(1, Math.min(4 * t - 1.5, -4 * t + 4.5)));
    var g = Math.max(0, Math.min(1, Math.min(4 * t - 0.5, -4 * t + 3.5)));
    var b = Math.max(0, Math.min(1, Math.min(4 * t + 0.5, -4 * t + 2.5)));
    return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
  }

  var canvas = document.createElement("canvas");
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.cssText =
    "position:fixed;top:0;left:0;width:100%;height:100%;" +
    "z-index:-1;pointer-events:none;opacity:" + CFG.opacity;
  var ctx = canvas.getContext("2d");
  (document.body || document.documentElement).appendChild(canvas);

  var W = 0, H = 0, DPR = 1;
  var nodes = [];       // {ox, oy} rest positions
  var cols = 0, rows = 0;
  var links = [];       // [iA, iB] precomputed neighbour pairs
  var waves = [];       // {x, y, t, amp, k, life}

  // deterministic pseudo-random so the lattice disorder is stable across resizes
  function rand(seed) {
    var x = Math.sin(seed * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  function build() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

    var s = CFG.spacing;
    cols = Math.ceil(W / s) + 2;
    rows = Math.ceil(H / s) + 2;
    nodes = [];
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var seed = r * 73.13 + c * 31.7 + 1;
        var jx = (rand(seed) - 0.5) * 2 * CFG.jitter * s;
        var jy = (rand(seed + 0.5) - 0.5) * 2 * CFG.jitter * s;
        nodes.push({
          ox: (c - 0.5) * s + jx,
          oy: (r - 0.5) * s + jy,
          x: 0, y: 0, strain: 0
        });
      }
    }

    // neighbour links: right, down, both diagonals (within cutoff)
    var cut = (CFG.linkDist * s) * (CFG.linkDist * s);
    links = [];
    function idx(c, r) { return r * cols + c; }
    for (var rr = 0; rr < rows; rr++) {
      for (var cc = 0; cc < cols; cc++) {
        var a = idx(cc, rr);
        var cand = [[cc + 1, rr], [cc, rr + 1], [cc + 1, rr + 1], [cc - 1, rr + 1]];
        for (var n = 0; n < cand.length; n++) {
          var nc = cand[n][0], nr = cand[n][1];
          if (nc < 0 || nc >= cols || nr < 0 || nr >= rows) continue;
          var b = idx(nc, nr);
          var dx = nodes[a].ox - nodes[b].ox, dy = nodes[a].oy - nodes[b].oy;
          if (dx * dx + dy * dy <= cut) links.push([a, b]);
        }
      }
    }
  }

  function spawn(x, y, amp) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    waves.push({
      x: x, y: y, t: 0,
      amp: amp,
      k: (2 * Math.PI) / CFG.wavelength,
      life: CFG.waveLife
    });
  }

  // displacement contribution of one wave at distance d, plus its radial dir handled by caller
  function waveField(w, d) {
    var front = CFG.speed * w.t;          // current radius of the wavefront
    var ring = d - front;                 // signed distance to the front
    // narrow gaussian envelope -> thin, sharp wavefronts (background stays calm
    // even with many overlapping ripples), plus global time decay
    var env = Math.exp(-(ring * ring) / (2 * 850));
    var decay = 1 - w.t / w.life;
    if (decay < 0) decay = 0;
    return w.amp * decay * env * Math.sin(w.k * ring);
  }

  var last = 0, acc = 0, autoTimer = 0, nextAuto = 0;
  var frameInterval = 1 / CFG.fps;

  function scheduleAuto() {
    nextAuto = CFG.autoMin + Math.random() * (CFG.autoMax - CFG.autoMin);
  }
  scheduleAuto();

  function frame(now) {
    requestAnimationFrame(frame);
    if (!last) last = now;
    var dt = (now - last) / 1000;
    last = now;
    if (dt > 0.1) dt = 0.1;            // clamp after tab switch
    acc += dt;
    if (acc < frameInterval) return;   // throttle to target fps
    var step = acc; acc = 0;

    // advance waves
    autoTimer += step;
    if (autoTimer >= nextAuto) {
      autoTimer = 0;
      scheduleAuto();   // new random delay in [autoMin, autoMax] seconds
      spawn(Math.random() * W, Math.random() * H, CFG.amp * (1.4 + Math.random() * 0.8));
    }
    for (var wI = waves.length - 1; wI >= 0; wI--) {
      waves[wI].t += step;
      if (waves[wI].t >= waves[wI].life) waves.splice(wI, 1);
    }

    // displace nodes
    var i, w, n, dx, dy, d, disp;
    for (i = 0; i < nodes.length; i++) {
      n = nodes[i];
      var ux = 0, uy = 0;
      for (var k = 0; k < waves.length; k++) {
        w = waves[k];
        dx = n.ox - w.x; dy = n.oy - w.y;
        d = Math.sqrt(dx * dx + dy * dy) + 0.001;
        disp = waveField(w, d);
        ux += (dx / d) * disp;          // longitudinal: displace along radius
        uy += (dy / d) * disp;
      }
      n.x = n.ox + ux;
      n.y = n.oy + uy;
      n.strain = Math.sqrt(ux * ux + uy * uy);
    }

    draw();
  }

  var LEVELS = 16;              // colormap quantisation; each level = one batched stroke
  // normalise against a single wave's crest (not the dynamic max) so every
  // wavefront reaches red all the way round; interference just stays clamped at red.
  var REF = CFG.amp * 0.95;       // sets where the colormap saturates; lower -> more orange/yellow at the fronts
  var HUECAP = 0.82;              // compress colormap so the top is bright orange-red, not dark red
  function draw() {
    ctx.clearRect(0, 0, W, H);

    // links bucketed by field amplitude -> COMSOL Rainbow (jet) colormap
    var paths = [], li, bb;
    for (bb = 0; bb < LEVELS; bb++) paths.push([]);

    for (li = 0; li < links.length; li++) {
      var a = nodes[links[li][0]], c = nodes[links[li][1]];
      var s = (a.strain + c.strain) * 0.5 / REF; // 0..1 normalised amplitude
      if (s > 1) s = 1;
      var lv = (s * (LEVELS - 1) + 0.5) | 0;
      paths[lv].push(a.x, a.y, c.x, c.y);
    }

    for (bb = 0; bb < LEVELS; bb++) {
      var arr = paths[bb];
      if (!arr.length) continue;
      var t = bb / (LEVELS - 1);
      var col = jet(t * HUECAP);
      var alpha = CFG.baseAlpha + (CFG.peakAlpha - CFG.baseAlpha) * t;
      ctx.strokeStyle = "rgba(" + col[0] + "," + col[1] + "," + col[2] + "," + alpha.toFixed(3) + ")";
      ctx.lineWidth = 0.6 + t * 1.6;
      ctx.beginPath();
      for (var j = 0; j < arr.length; j += 4) {
        ctx.moveTo(arr[j], arr[j + 1]);
        ctx.lineTo(arr[j + 2], arr[j + 3]);
      }
      ctx.stroke();
    }

    // nodes coloured by the same colormap, brighter/larger where the field is strong
    for (var ni = 0; ni < nodes.length; ni++) {
      var nd = nodes[ni];
      var ns = nd.strain / maxStrain; if (ns > 1) ns = 1;
      var nc = jet(ns * HUECAP);
      ctx.fillStyle = "rgba(" + nc[0] + "," + nc[1] + "," + nc[2] + "," +
        (CFG.nodeAlpha * (0.4 + 0.6 * ns)).toFixed(3) + ")";
      var rad = 0.9 + ns * 1.9;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, rad, 0, 6.2832);
      ctx.fill();
    }
  }

  // static render for reduced-motion users: one frozen wavefront, no animation
  function renderStatic() {
    spawn(W * 0.06, H * 0.5, CFG.amp * 1.8);
    waves[0].t = (CFG.wavelength * 1.2) / CFG.speed;
    var maxStrain = 1e-4, i, n, w, dx, dy, d, disp, ux, uy;
    for (i = 0; i < nodes.length; i++) {
      n = nodes[i]; ux = 0; uy = 0;
      w = waves[0];
      dx = n.ox - w.x; dy = n.oy - w.y; d = Math.sqrt(dx * dx + dy * dy) + 0.001;
      disp = waveField(w, d);
      ux = (dx / d) * disp; uy = (dy / d) * disp;
      n.x = n.ox + ux; n.y = n.oy + uy;
      n.strain = Math.sqrt(ux * ux + uy * uy);
      if (n.strain > maxStrain) maxStrain = n.strain;
    }
    draw(maxStrain);
  }

  // pointer = wave source
  var moveAcc = 0;
  window.addEventListener("pointermove", function (e) {
    var t = performance.now();
    if (t - moveAcc < 90) return;     // throttle ripple injection
    moveAcc = t;
    spawn(e.clientX, e.clientY, CFG.amp * 0.55);
  }, { passive: true });
  window.addEventListener("pointerdown", function (e) {
    spawn(e.clientX, e.clientY, CFG.amp * 1.8);
  }, { passive: true });

  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      build();
      if (reduce) renderStatic();
    }, 150);
  });

  build();
  if (reduce) {
    renderStatic();
  } else {
    spawn(W * 0.06, H * 0.5, CFG.amp * (1.4 + Math.random() * 0.8));   // start at the left edge, not behind the centred text
    requestAnimationFrame(frame);
  }
})();
