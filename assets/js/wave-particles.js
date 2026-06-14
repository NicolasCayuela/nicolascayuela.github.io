/*
 * Elastic-wave metamaterial background (v2).
 *
 * A periodic lattice of nodes (the "unit cells" of a phononic crystal /
 * acoustic metamaterial). Elastic waves travel through it and displace the
 * nodes; the links between neighbours light up with the local strain, so you
 * literally see the wavefronts ripple through the periodic medium.
 *
 * Two wave kinds run together:
 *   - radial pulses (point sources, like a tap on the medium), and
 *   - plane waves that sweep across the lattice along a direction (Bloch-like
 *     propagation along a path).
 * Each carries a longitudinal part (motion along propagation) and a smaller
 * transverse/shear part a quarter-phase out, so nodes trace little ellipses,
 * the way particles move in a real surface elastic wave.
 *
 * Moving the pointer injects small ripples, clicking emits a strong pulse.
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
    autoMin: 1.2,       // min seconds between random excitations
    autoMax: 3,         // max seconds between random excitations
    planeProb: 0.34,    // share of auto excitations that are sweeping plane waves
    shear: 0.42,        // transverse amplitude as a fraction of longitudinal
    frontWidth: 30,     // wavefront thickness (px, gaussian std) -> sharpness
    gapRadius: 0.13,    // band-gap inclusion radius, fraction of min(W,H)
    gapDamp: 0.10,      // residual node response inside the inclusion (the gap)
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
  var waves = [];       // {kind, x, y, nx, ny, t, amp, k, life}
  var SIG2 = 2 * CFG.frontWidth * CFG.frontWidth;
  var gapCx = 0, gapCy = 0, gapR = 0;   // band-gap inclusion (mass-loaded zone)

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

    // Band-gap inclusion: a circular cluster of mass-loaded "heavy" unit cells.
    // Sits off the centred page text (right side). Nodes inside barely respond,
    // so wavefronts ripple up to it and the region stays calm: a visible gap.
    gapR = CFG.gapRadius * Math.min(W, H);
    gapCx = W * 0.80;
    gapCy = H * 0.30;
    var edge = s * 1.6;
    for (var gi = 0; gi < nodes.length; gi++) {
      var gn = nodes[gi];
      var gd = Math.sqrt((gn.ox - gapCx) * (gn.ox - gapCx) + (gn.oy - gapCy) * (gn.oy - gapCy));
      var tt = (gd - gapR) / edge; if (tt < 0) tt = 0; else if (tt > 1) tt = 1;
      var sm = tt * tt * (3 - 2 * tt);            // smoothstep ramp at the edge
      gn.gap = CFG.gapDamp + (1 - CFG.gapDamp) * sm;
      gn.heavy = gd < gapR;                       // drawn as fixed resonators
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

  function spawnRadial(x, y, amp) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    waves.push({
      kind: 0, x: x, y: y, nx: 0, ny: 0, t: 0,
      amp: amp, k: (2 * Math.PI) / CFG.wavelength, life: CFG.waveLife
    });
  }

  // plane wave: a flat front entering from one edge and sweeping across, the
  // origin point sits just outside that edge so the front crosses the screen.
  function spawnPlane(amp) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    var ang = Math.random() * Math.PI * 2;
    var nx = Math.cos(ang), ny = Math.sin(ang);
    var cx = W * 0.5, cy = H * 0.5, span = Math.sqrt(W * W + H * H) * 0.5;
    waves.push({
      kind: 1, x: cx - nx * span, y: cy - ny * span, nx: nx, ny: ny, t: 0,
      amp: amp, k: (2 * Math.PI) / CFG.wavelength, life: CFG.waveLife
    });
  }

  // scalar field of one wave at signed propagation coordinate `p`. quad=true
  // returns the quarter-phase (cosine) component used for the transverse part.
  function waveField(w, p, quad) {
    var ring = p - CFG.speed * w.t;          // signed distance to the front
    var env = Math.exp(-(ring * ring) / SIG2);
    var decay = 1 - w.t / w.life;
    if (decay < 0) decay = 0;
    var ph = w.k * ring;
    return w.amp * decay * env * (quad ? Math.cos(ph) : Math.sin(ph));
  }

  // Displace every node by the superposition of all live waves. Each wave adds
  // a longitudinal term (along propagation) and a smaller transverse term a
  // quarter-phase out, so a node traces an ellipse. Returns the peak strain.
  function displaceNodes() {
    var maxStrain = 1e-4, i, k, w, dx, dy, d, p, dirx, diry, lon, sh;
    for (i = 0; i < nodes.length; i++) {
      var n = nodes[i], ux = 0, uy = 0;
      for (k = 0; k < waves.length; k++) {
        w = waves[k];
        if (w.kind === 0) {                  // radial
          dx = n.ox - w.x; dy = n.oy - w.y;
          d = Math.sqrt(dx * dx + dy * dy) + 0.001;
          dirx = dx / d; diry = dy / d; p = d;
        } else {                             // plane
          dirx = w.nx; diry = w.ny;
          p = (n.ox - w.x) * w.nx + (n.oy - w.y) * w.ny;
        }
        lon = waveField(w, p, false);
        sh = waveField(w, p, true) * CFG.shear;
        ux += dirx * lon - diry * sh;        // longitudinal + transverse (perp)
        uy += diry * lon + dirx * sh;
      }
      ux *= n.gap; uy *= n.gap;          // band-gap inclusion damps the response
      n.x = n.ox + ux;
      n.y = n.oy + uy;
      n.strain = Math.sqrt(ux * ux + uy * uy);
      if (n.strain > maxStrain) maxStrain = n.strain;
    }
    return maxStrain;
  }

  var last = 0, acc = 0, autoTimer = 0, nextAuto = 0;
  var frameInterval = 1 / CFG.fps;

  function scheduleAuto() {
    nextAuto = CFG.autoMin + Math.random() * (CFG.autoMax - CFG.autoMin);
  }
  scheduleAuto();

  function autoExcite() {
    if (Math.random() < CFG.planeProb) {
      spawnPlane(CFG.amp * (1.0 + Math.random() * 0.6));
    } else {
      spawnRadial(Math.random() * W, Math.random() * H, CFG.amp * (1.4 + Math.random() * 0.8));
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!last) last = now;
    var dt = (now - last) / 1000;
    last = now;
    if (dt > 0.1) dt = 0.1;            // clamp after tab switch
    acc += dt;
    if (acc < frameInterval) return;   // throttle to target fps
    var step = acc; acc = 0;

    autoTimer += step;
    if (autoTimer >= nextAuto) {
      autoTimer = 0;
      scheduleAuto();
      autoExcite();
    }
    for (var wI = waves.length - 1; wI >= 0; wI--) {
      waves[wI].t += step;
      if (waves[wI].t >= waves[wI].life) waves.splice(wI, 1);
    }

    displaceNodes();
    draw(REF);                    // normalise node colour/size against the crest reference
  }

  var lastDark = null;          // tracks theme to update canvas opacity only on change
  var LEVELS = 16;              // colormap quantisation; each level = one batched stroke
  // normalise against a single wave's crest (not the dynamic max) so every
  // wavefront reaches red all the way round; interference just stays clamped at red.
  var REF = CFG.amp * 0.95;       // sets where the colormap saturates; lower -> more orange/yellow at the fronts
  var HUECAP = 0.82;              // compress colormap so the top is bright orange-red, not dark red
  function draw(maxStrain) {
    ctx.clearRect(0, 0, W, H);

    // dark theme: the jet low end (dark blue) vanishes on black, so lift the
    // colormap floor and the resting opacity to keep the lattice visible
    var dark = document.documentElement.classList.contains("theme-dark");
    if (dark !== lastDark) {
      lastDark = dark;
      canvas.style.opacity = dark ? 0.9 : CFG.opacity;
    }
    var tFloor = dark ? 0.10 : 0;
    var baseA = dark ? 0.7 : CFG.baseAlpha;
    // pure jet blue is too dim on black: blend resting colors toward white,
    // fading the lift out as amplitude rises so crests stay saturated
    function lift(col, t) {
      if (!dark) return col;
      var f = 0.7 * (1 - t);
      return [
        (col[0] + (255 - col[0]) * f) | 0,
        (col[1] + (255 - col[1]) * f) | 0,
        (col[2] + (255 - col[2]) * f) | 0
      ];
    }

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
      var col = lift(jet(tFloor + t * (HUECAP - tFloor)), t);
      var alpha = baseA + (CFG.peakAlpha - baseA) * t;
      ctx.strokeStyle = "rgba(" + col[0] + "," + col[1] + "," + col[2] + "," + alpha.toFixed(3) + ")";
      ctx.lineWidth = 0.6 + t * 1.6;
      ctx.beginPath();
      for (var j = 0; j < arr.length; j += 4) {
        ctx.moveTo(arr[j], arr[j + 1]);
        ctx.lineTo(arr[j + 2], arr[j + 3]);
      }
      ctx.stroke();
    }

    // mass-loaded resonators of the band-gap inclusion: opaque fixed markers
    // (source-over, before the additive passes so they read on light theme too)
    ctx.fillStyle = dark ? "rgba(150,172,205,0.85)" : "rgba(86,108,150,0.7)";
    for (var hi = 0; hi < nodes.length; hi++) {
      var hn = nodes[hi];
      if (!hn.heavy) continue;
      ctx.beginPath();
      ctx.arc(hn.x, hn.y, 2.4, 0, 6.2832);
      ctx.fill();
    }

    // crest bloom: re-stroke the brightest buckets wide and faint with additive
    // blending so wavefronts glow where they overlap. Few links live up here,
    // so it is cheap. Strongest on dark; a gentle touch on light.
    var bloomFrom = (LEVELS * 0.72) | 0;
    ctx.globalCompositeOperation = "lighter";
    for (bb = bloomFrom; bb < LEVELS; bb++) {
      var barr = paths[bb];
      if (!barr.length) continue;
      var bt = bb / (LEVELS - 1);
      var bcol = jet(tFloor + bt * (HUECAP - tFloor));
      ctx.strokeStyle = "rgba(" + bcol[0] + "," + bcol[1] + "," + bcol[2] + "," +
        ((dark ? 0.16 : 0.08) * bt).toFixed(3) + ")";
      ctx.lineWidth = 3 + bt * 5;
      ctx.beginPath();
      for (var bj = 0; bj < barr.length; bj += 4) {
        ctx.moveTo(barr[bj], barr[bj + 1]);
        ctx.lineTo(barr[bj + 2], barr[bj + 3]);
      }
      ctx.stroke();
    }

    // nodes coloured by the same colormap, brighter/larger where the field is
    // strong. Additive so crossing wavefronts bloom at the nodes too.
    for (var ni = 0; ni < nodes.length; ni++) {
      var nd = nodes[ni];
      if (nd.heavy) continue;            // resonators drawn opaque in their own pass
      var ns = nd.strain / maxStrain; if (ns > 1) ns = 1;
      var nc = lift(jet(tFloor + ns * (HUECAP - tFloor)), ns);
      ctx.fillStyle = "rgba(" + nc[0] + "," + nc[1] + "," + nc[2] + "," +
        ((dark ? 0.9 : CFG.nodeAlpha) * (0.4 + 0.6 * ns)).toFixed(3) + ")";
      var rad = 0.9 + ns * 1.9;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, rad, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
  }

  // static render for reduced-motion users: one frozen wavefront, no animation
  function renderStatic() {
    waves = [];
    spawnRadial(W * 0.06, H * 0.5, CFG.amp * 1.8);
    waves[0].t = (CFG.wavelength * 1.2) / CFG.speed;
    draw(displaceNodes());
  }

  // pointer = wave source
  var moveAcc = 0;
  window.addEventListener("pointermove", function (e) {
    var t = performance.now();
    if (t - moveAcc < 90) return;     // throttle ripple injection
    moveAcc = t;
    spawnRadial(e.clientX, e.clientY, CFG.amp * 0.55);
  }, { passive: true });
  window.addEventListener("pointerdown", function (e) {
    spawnRadial(e.clientX, e.clientY, CFG.amp * 1.8);
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
    // Animated mode re-reads the theme every frame; the static render runs once,
    // so it must repaint when the user toggles dark/light (theme.js fires this).
    window.addEventListener("themechange", renderStatic);
  } else {
    spawnRadial(W * 0.06, H * 0.5, CFG.amp * (1.4 + Math.random() * 0.8));   // start at the left edge, not behind the centred text
    requestAnimationFrame(frame);
  }
})();
