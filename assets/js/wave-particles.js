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
    amp: 13,            // peak node displacement (px)
    wavelength: 190,    // spatial period of a wave (px)
    speed: 130,         // wave phase speed (px/s)
    waveLife: 4.2,      // seconds a ripple stays alive
    maxWaves: 14,       // hard cap on simultaneous ripples
    autoEvery: 2.6,     // seconds between ambient auto-ripples
    linkDist: 1.9,      // neighbour link cutoff, in lattice pitches
    baseAlpha: 0.16,    // resting link opacity
    nodeAlpha: 0.5,
    hueLow: 188,        // calm wavefront hue (cyan)
    hueHigh: 286,       // high-strain hue (violet) -> AI/elastic field look
    opacity: 0.55,      // whole-canvas opacity
    fps: 60
  };

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
    // gaussian envelope around the travelling front + global time decay
    var env = Math.exp(-(ring * ring) / (2 * 9000));
    var decay = 1 - w.t / w.life;
    if (decay < 0) decay = 0;
    return w.amp * decay * env * Math.sin(w.k * ring);
  }

  var last = 0, acc = 0, autoTimer = 0;
  var frameInterval = 1 / CFG.fps;

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
    if (autoTimer >= CFG.autoEvery) {
      autoTimer = 0;
      spawn(Math.random() * W, Math.random() * H, CFG.amp * (0.5 + Math.random() * 0.5));
    }
    for (var wI = waves.length - 1; wI >= 0; wI--) {
      waves[wI].t += step;
      if (waves[wI].t >= waves[wI].life) waves.splice(wI, 1);
    }

    // displace nodes
    var i, w, n, dx, dy, d, disp, maxStrain = 1e-4;
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
      if (n.strain > maxStrain) maxStrain = n.strain;
    }

    draw(maxStrain);
  }

  function draw(maxStrain) {
    ctx.clearRect(0, 0, W, H);

    // links, bucketed by strain so we stroke a handful of paths instead of thousands
    var BUCKETS = 7;
    var paths = [];
    for (var b = 0; b < BUCKETS; b++) paths.push([]);

    for (var li = 0; li < links.length; li++) {
      var a = nodes[links[li][0]], c = nodes[links[li][1]];
      var s = (a.strain + c.strain) * 0.5 / maxStrain; // 0..1 normalised strain
      if (s > 1) s = 1;
      var bi = (s * (BUCKETS - 1)) | 0;
      var p = paths[bi];
      p.push(a.x, a.y, c.x, c.y);
    }

    for (var bk = 0; bk < BUCKETS; bk++) {
      var arr = paths[bk];
      if (!arr.length) continue;
      var t = bk / (BUCKETS - 1);
      var hue = CFG.hueLow + (CFG.hueHigh - CFG.hueLow) * t;
      var alpha = CFG.baseAlpha + (0.85 - CFG.baseAlpha) * t;
      ctx.strokeStyle = "hsla(" + hue.toFixed(0) + ",85%,58%," + alpha.toFixed(3) + ")";
      ctx.lineWidth = 0.6 + t * 1.3;
      ctx.beginPath();
      for (var j = 0; j < arr.length; j += 4) {
        ctx.moveTo(arr[j], arr[j + 1]);
        ctx.lineTo(arr[j + 2], arr[j + 3]);
      }
      ctx.stroke();
    }

    // nodes (brighter where strained -> they trace the wavefronts)
    for (var ni = 0; ni < nodes.length; ni++) {
      var nd = nodes[ni];
      var ns = nd.strain / maxStrain; if (ns > 1) ns = 1;
      var nh = CFG.hueLow + (CFG.hueHigh - CFG.hueLow) * ns;
      ctx.fillStyle = "hsla(" + nh.toFixed(0) + ",90%,60%," +
        (CFG.nodeAlpha * (0.35 + 0.65 * ns)).toFixed(3) + ")";
      var rad = 0.9 + ns * 1.8;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, rad, 0, 6.2832);
      ctx.fill();
    }
  }

  // static render for reduced-motion users: one frozen wavefront, no animation
  function renderStatic() {
    spawn(W * 0.5, H * 0.42, CFG.amp);
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
    spawn(W * 0.5, H * 0.4, CFG.amp);
    requestAnimationFrame(frame);
  }
})();
