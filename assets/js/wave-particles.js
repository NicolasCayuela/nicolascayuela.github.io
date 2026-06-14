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
    amp: 12,            // peak node displacement (px)
    wavelength: 175,    // spatial period of a wave (px)
    speed: 115,         // wave phase speed (px/s), longitudinal (P) reference
    shearSpeedFrac: 0.62, // shear (S) waves travel slower than P (dispersion)
    edgeSpeedFrac: 0.80,  // topological edge mode speed, fraction of P
    fadeIn: 1.1,        // seconds to ramp the canvas in on load
    waveLife: 18,       // seconds a ripple stays alive
    maxWaves: 50,       // perf backstop only; high so live ripples are never cut
    autoMin: 1,         // min seconds between random excitations
    autoMax: 2,         // max seconds between random excitations
    planeProb: 0.34,    // share of auto excitations that are sweeping plane waves
    topoProb: 0.16,     // share that are robust topological edge modes (border path)
    shearProb: 0.32,    // share of waves that are shear-dominant (transverse mode)
    shear: 0.45,        // transverse amplitude as a fraction of longitudinal (P mode)
    frontWidth: 26,     // wavefront thickness (px, gaussian std) -> sharpness
    cullWidth: 4,       // node-vs-wave cull band, in frontWidths (perf)
    clickCooldown: 600, // ms between pointer-click pulses (anti-spam)
    linkDist: 1.6,      // neighbour link cutoff, in lattice pitches
    vigMin: 0.26,       // field opacity behind the centred content (0 = hidden)
    vigAx: 0.55,        // half-width of the dimmed central band (frac of W/2)
    vigAy: 0.90,        // half-height of the dimmed central band (frac of H/2)
    baseAlpha: 0.16,    // resting link opacity (idle = faint blue, COMSOL low end)
    peakAlpha: 0.90,    // link opacity at the crest
    nodeAlpha: 0.5,
    opacity: 0.55,      // whole-canvas opacity
    fps: 60
  };

  // "Rainbow" (jet) colormap: t in [0,1] -> [r,g,b] 0..255
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

    // Square lattice of unit cells with a little disorder.
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
    for (var rrr = 0; rrr < rows; rrr++) {
      for (var cc = 0; cc < cols; cc++) {
        var a = idx(cc, rrr);
        var cand = [[cc + 1, rrr], [cc, rrr + 1], [cc + 1, rrr + 1], [cc - 1, rrr + 1]];
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

  // lon/sh are the longitudinal and transverse (shear) weights of the wave.
  // P-wave: lon=1, sh=CFG.shear. Shear-dominant: lon small, sh large.
  function spawnRadial(x, y, amp, lon, sh, speed) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    waves.push({
      kind: 0, x: x, y: y, nx: 0, ny: 0, t: 0,
      amp: amp, k: (2 * Math.PI) / CFG.wavelength, life: CFG.waveLife,
      lon: lon == null ? 1 : lon, sh: sh == null ? CFG.shear : sh,
      speed: speed == null ? CFG.speed : speed
    });
  }

  // plane wave: a flat front entering from one edge and sweeping across, the
  // origin point sits just outside that edge so the front crosses the screen.
  function spawnPlane(amp, lon, sh, speed) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    var ang = Math.random() * Math.PI * 2;
    var nx = Math.cos(ang), ny = Math.sin(ang);
    var cx = W * 0.5, cy = H * 0.5, span = Math.sqrt(W * W + H * H) * 0.5;
    waves.push({
      kind: 1, x: cx - nx * span, y: cy - ny * span, nx: nx, ny: ny, t: 0,
      amp: amp, k: (2 * Math.PI) / CFG.wavelength, life: CFG.waveLife,
      lon: lon == null ? 1 : lon, sh: sh == null ? CFG.shear : sh,
      speed: speed == null ? CFG.speed : speed
    });
  }

  // robust topological edge mode: a wave packet that travels along the lattice
  // boundary (rectangular path, inset from the screen edges) and turns the
  // corners without backscattering, exciting only nodes within a thin channel
  // around the path. The rest of the lattice stays still, like a chiral edge
  // state in a topological phononic insulator.
  function spawnEdge(amp, speed) {
    if (waves.length >= CFG.maxWaves) waves.shift();
    var m = CFG.spacing * 2.2;            // channel inset from the border
    var pts = [[m, m], [W - m, m], [W - m, H - m], [m, H - m]];
    var segs = [], arc = 0;
    for (var i = 0; i < 4; i++) {
      var a = pts[i], b = pts[(i + 1) % 4];
      var dx = b[0] - a[0], dy = b[1] - a[1], len = Math.sqrt(dx * dx + dy * dy);
      segs.push({ x: a[0], y: a[1], ux: dx / len, uy: dy / len, len: len, arc: arc });
      arc += len;
    }
    var chHalf = CFG.spacing * 1.2;
    waves.push({
      kind: 2, t: 0, amp: amp, k: (2 * Math.PI) / CFG.wavelength, life: CFG.waveLife,
      segs: segs, L: arc, inset: m, chHalf: chHalf, chCull: chHalf * 3,
      sigA: CFG.wavelength * 1.1, dir: Math.random() < 0.5 ? 1 : -1,
      speed: speed == null ? CFG.speed : speed
    });
  }

  // scalar field of one wave at signed propagation coordinate `p`. quad=true
  // returns the quarter-phase (cosine) component used for the transverse part.
  function waveField(w, p, quad) {
    var ring = p - w.speed * w.t;            // signed distance to the front
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
    var cull = CFG.cullWidth * CFG.frontWidth;
    // per-wave front radius + cull band (squared) so the inner loop can skip
    // nodes outside the active ring without any sqrt.
    for (k = 0; k < waves.length; k++) {
      var fr = waves[k].speed * waves[k].t;
      waves[k]._lo = Math.max(0, fr - cull); waves[k]._hi = fr + cull;
      waves[k]._lo2 = waves[k]._lo * waves[k]._lo;
      waves[k]._hi2 = waves[k]._hi * waves[k]._hi;
    }
    for (i = 0; i < nodes.length; i++) {
      var n = nodes[i], ux = 0, uy = 0;
      for (k = 0; k < waves.length; k++) {
        w = waves[k];
        if (w.kind === 2) {                  // topological edge mode (border path)
          var mb = Math.min(n.ox, W - n.ox, n.oy, H - n.oy);
          if (Math.abs(mb - w.inset) > w.chCull) continue;   // not near the channel
          var best = 1e9, bArc = 0, bdx = 0, bdy = 0, sgi;
          for (sgi = 0; sgi < 4; sgi++) {
            var sg = w.segs[sgi];
            var pr = (n.ox - sg.x) * sg.ux + (n.oy - sg.y) * sg.uy;
            if (pr < 0) pr = 0; else if (pr > sg.len) pr = sg.len;
            var ex = n.ox - (sg.x + sg.ux * pr), ey = n.oy - (sg.y + sg.uy * pr);
            var e2 = ex * ex + ey * ey;
            if (e2 < best) { best = e2; bArc = sg.arc + pr; bdx = sg.ux; bdy = sg.uy; }
          }
          var trans = Math.sqrt(best);
          if (trans > w.chCull) continue;
          var da = bArc - w.dir * w.speed * w.t;
          da -= w.L * Math.round(da / w.L);                  // wrap around the loop
          var decE = 1 - w.t / w.life; if (decE < 0) decE = 0;
          var dE = w.amp * decE
            * Math.exp(-(da * da) / (2 * w.sigA * w.sigA))
            * Math.exp(-(trans * trans) / (2 * w.chHalf * w.chHalf))
            * Math.sin(w.k * da);
          ux += -bdy * dE; uy += bdx * dE;                   // transverse to the path
          continue;
        }
        if (w.kind === 0) {                  // radial
          dx = n.ox - w.x; dy = n.oy - w.y;
          var dsq = dx * dx + dy * dy;
          if (dsq > w._hi2 || dsq < w._lo2) continue;   // outside the active ring
          d = Math.sqrt(dsq) + 0.001;
          dirx = dx / d; diry = dy / d; p = d;
        } else {                             // plane
          dirx = w.nx; diry = w.ny;
          p = (n.ox - w.x) * w.nx + (n.oy - w.y) * w.ny;
          if (p < w._lo || p > w._hi) continue;         // front not here yet / gone
        }
        lon = waveField(w, p, false) * w.lon;
        sh = waveField(w, p, true) * w.sh;
        ux += dirx * lon - diry * sh;        // longitudinal + transverse (perp)
        uy += diry * lon + dirx * sh;
      }
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
    // robust topological edge mode now and then: travels the boundary, leaves
    // the bulk still. Edge mode runs at its own speed (dispersion).
    if (Math.random() < CFG.topoProb) {
      spawnEdge(CFG.amp * 1.15, CFG.speed * CFG.edgeSpeedFrac);
      return;
    }
    // otherwise a randomly polarised bulk wave. shear-dominant (S) waves are
    // mostly transverse and travel slower than the longitudinal (P) default,
    // so the two modes visibly separate as they propagate (dispersion).
    var shearMode = Math.random() < CFG.shearProb;
    var lon = shearMode ? 0.35 : 1;
    var sh = shearMode ? 1.0 : CFG.shear;
    var spd = shearMode ? CFG.speed * CFG.shearSpeedFrac : CFG.speed;
    if (Math.random() < CFG.planeProb) {
      spawnPlane(CFG.amp * (1.0 + Math.random() * 0.6), lon, sh, spd);
    } else {
      spawnRadial(Math.random() * W, Math.random() * H, CFG.amp * (1.4 + Math.random() * 0.8), lon, sh, spd);
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) { last = 0; return; }   // idle in background tabs (battery)
    if (!loadStart) loadStart = now;
    fadeAmt = CFG.fadeIn > 0 ? Math.min(1, (now - loadStart) / (CFG.fadeIn * 1000)) : 1;
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

  var fadeAmt = reduce ? 1 : 0; // load fade-in multiplier (0 -> 1 over CFG.fadeIn)
  var loadStart = 0;
  var LEVELS = 32;             // colormap quantisation; each level = one batched stroke
  // normalise against a single wave's crest (not the dynamic max) so every
  // wavefront reaches red all the way round; interference just stays clamped at red.
  var REF = CFG.amp * 0.95;       // sets where the colormap saturates; lower -> more orange/yellow at the fronts
  var HUECAP = 0.82;              // compress colormap so the top is bright orange-red, not dark red
  function draw(maxStrain) {
    ctx.clearRect(0, 0, W, H);

    // dark theme: the jet low end (dark blue) vanishes on black, so lift the
    // colormap floor and the resting opacity to keep the lattice visible
    var dark = document.documentElement.classList.contains("theme-dark");
    // whole-canvas opacity: theme target scaled by the load fade-in
    canvas.style.opacity = ((dark ? 0.9 : CFG.opacity) * fadeAmt).toFixed(3);
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

    // legibility vignette: fade the whole field inside a central ellipse so it
    // does not compete with the page text. One destination-out gradient pass,
    // removing up to (1 - vigMin) of the centre and nothing at the edges.
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.translate(W * 0.5, H * 0.5);
    ctx.scale(W * 0.5 * CFG.vigAx, H * 0.5 * CFG.vigAy);
    var vg = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    vg.addColorStop(0, "rgba(0,0,0," + (1 - CFG.vigMin).toFixed(3) + ")");
    vg.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = vg;
    ctx.beginPath();
    ctx.arc(0, 0, 1, 0, 6.2832);
    ctx.fill();
    ctx.restore();
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
  var clickAcc = 0;
  window.addEventListener("pointerdown", function (e) {
    var t = performance.now();
    if (t - clickAcc < CFG.clickCooldown) return;   // cooldown between click pulses
    clickAcc = t;
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
    // Random first excitation on every page load: position, polarisation and
    // wave kind all vary, so the background never opens the same way twice.
    autoExcite();
    requestAnimationFrame(frame);
  }
})();
