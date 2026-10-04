/*
 * Elastic-wave metamaterial background.
 *
 * A periodic lattice of nodes (the "unit cells" of a phononic crystal /
 * acoustic metamaterial). Elastic waves travel through it and displace the
 * nodes; the links between neighbours light up with the local strain, so you
 * literally see the wavefronts ripple through the periodic medium.
 *
 * Three wave kinds run together, each at its own speed (dispersion):
 *   - radial pulses (point sources, like a tap on the medium),
 *   - plane waves sweeping across the lattice along a direction, and
 *   - topological edge modes that travel the boundary and turn the corners,
 *     exciting only a thin channel and leaving the bulk still.
 * Bulk waves carry a longitudinal part (motion along propagation) and a
 * smaller transverse/shear part a quarter-phase out, so nodes trace little
 * ellipses, the way particles move in a real surface elastic wave.
 *
 * Pointer move injects small ripples; click emits a stronger pulse (cooldown).
 * Pure <canvas>, no dependency. Runs behind the page content.
 *
 * Threading: where OffscreenCanvas is available, this same file is started as
 * a Web Worker and the page transfers the canvas to it, so the simulation AND
 * the rasterisation (the bulk of the cost) run off the main thread: the page
 * stays responsive whatever the waves cost. The page side only forwards
 * pointer / resize / theme / visibility events. Otherwise it runs on the page.
 *
 * Perf: links and nodes are bucketed by colour level into one typed array
 * (counting sort, no per-frame allocation) and drawn with one path per level;
 * the centre vignette is a CSS mask, not a canvas pass. Profiling on a
 * throttled CPU without GPU showed ~80% of the cost is the browser rasterising
 * and uploading the full-screen canvas each frame (not our JS), so the tiers
 * mostly trade canvas resolution and frame rate. Weak devices (phones, <=4
 * cores or <=4 GB) start on tier 1. Any device whose rAF loop runs slow (a
 * saturated drawing thread, raster included) steps down at runtime, down to a
 * frozen frame, and steps back up once the page is smooth again (a load
 * spike, e.g. a heavy demo starting, must not freeze the background for
 * good). <body data-waves="static"> renders a single frozen frame.
 */
(function () {
  "use strict";

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
    maxWaves: 50,       // perf backstop; lowered on weak tiers
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
    opacity: 0.55       // whole-canvas opacity (light theme; dark uses 0.9)
  };
  // quality tier -> canvas resolution (x CSS px), frame rate, bloom.
  // 0 full | 1 low: 0.75x, 30 fps | 2 minimal: 0.6x, 20 fps | 3 frozen frame
  var TIER_FPS = [60, 30, 20, 20];
  var TIER_RES = [0, 0.75, 0.6, 0.6];      // 0 = device DPR (capped at 1.5)

  // "Rainbow" (jet) colormap: t in [0,1] -> [r,g,b] 0..255
  function jet(t) {
    if (t < 0) t = 0; else if (t > 1) t = 1;
    var r = Math.max(0, Math.min(1, Math.min(4 * t - 1.5, -4 * t + 4.5)));
    var g = Math.max(0, Math.min(1, Math.min(4 * t - 0.5, -4 * t + 3.5)));
    var b = Math.max(0, Math.min(1, Math.min(4 * t + 0.5, -4 * t + 2.5)));
    return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
  }

  // The simulation + drawing, independent of where it runs (worker with an
  // OffscreenCanvas, or the page as a fallback). env: canvas, W, H, dpr, dark,
  // hidden, reduce, weak, raf(fn), setOpacity(op), onTier(t).
  function createWave(env) {
    var canvas = env.canvas, ctx = canvas.getContext("2d");
    var reduce = env.reduce;
    var tier = env.weak ? 1 : 0;

    var W = 0, H = 0, DPR = 1;
    var nodes = [];       // {ox, oy} rest positions
    var cols = 0, rows = 0;
    var links = [];       // [iA, iB] precomputed neighbour pairs
    var waves = [];       // {kind, x, y, nx, ny, t, amp, k, life}
    var linkLv, linkBuf, nodeLv, nodeBuf;   // per-frame level + level-sorted coords
    var levelCount = new Int32Array(32), levelStart = new Int32Array(32);
    var SIG2 = 2 * CFG.frontWidth * CFG.frontWidth;

    // deterministic pseudo-random so the lattice disorder is stable across resizes
    function rand(seed) {
      var x = Math.sin(seed * 12.9898) * 43758.5453;
      return x - Math.floor(x);
    }

    function build() {
      DPR = TIER_RES[tier] || Math.min(env.dpr || 1, 1.5);
      W = env.W;
      H = env.H;
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
      linkLv = new Uint8Array(links.length);
      linkBuf = new Float32Array(links.length * 4);
      nodeLv = new Uint8Array(nodes.length);
      nodeBuf = new Float32Array(nodes.length * 2);
    }

    function maxWaves() { return tier ? 24 : CFG.maxWaves; }

    // lon/sh are the longitudinal and transverse (shear) weights of the wave.
    // P-wave: lon=1, sh=CFG.shear. Shear-dominant: lon small, sh large.
    function spawnRadial(x, y, amp, lon, sh, speed) {
      if (waves.length >= maxWaves()) waves.shift();
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
      if (waves.length >= maxWaves()) waves.shift();
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
      if (waves.length >= maxWaves()) waves.shift();
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
    var frameInterval = 1 / TIER_FPS[tier];
    // runtime step-down: smoothed rAF interval vs the fastest interval seen (the
    // display refresh). When the thread drawing the waves is saturated (our JS
    // + canvas raster) rAF slows down, which catches raster cost that timing our
    // own code cannot see. Relative, so a browser that caps rAF at 30 Hz
    // (battery saver) is not mistaken for a slow one.
    var rafEMA = 0, rafFrames = 0, rafMin = 1e9;
    // step-up: after `upWait` ms of smooth rAF, go back one tier (never above
    // the starting tier). The wait doubles on each step-up so a device that
    // really cannot keep up does not oscillate.
    var baseTier = tier, goodMs = 0, upWait = 4000;
    function setTier(t) {
      if (t === tier) return;
      tier = t;
      frameInterval = 1 / TIER_FPS[tier];
      rafEMA = 0; rafFrames = 0; goodMs = 0;
      build();                      // DPR change needs a canvas rebuild
      if (tier === 3) draw(REF);    // final frozen frame (the loop stops drawing)
      env.onTier(tier);
    }

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
      env.raf(frame);
      if (env.hidden) { last = 0; return; }        // idle in background tabs (battery)
      if (!loadStart) loadStart = now;
      fadeAmt = CFG.fadeIn > 0 ? Math.min(1, (now - loadStart) / (CFG.fadeIn * 1000)) : 1;
      if (!last) last = now;
      var dt = (now - last) / 1000;
      last = now;
      if (dt > 0 && dt < 0.5) {          // ignore tab-switch gaps
        var ms = dt * 1000;
        if (ms > 4 && ms < rafMin) rafMin = ms;
        rafEMA = rafFrames ? rafEMA * 0.95 + ms * 0.05 : ms;
        if (++rafFrames > 90 && tier < 3 && rafEMA > 1.45 * rafMin && rafEMA > 21) setTier(tier + 1);
        else if (tier > baseTier && rafFrames > 30) {
          goodMs = rafEMA < 1.2 * rafMin ? goodMs + ms : 0;
          if (goodMs > upWait) { upWait = Math.min(upWait * 2, 60000); setTier(tier - 1); }
        }
      }
      if (tier === 3) return;            // frozen
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
    // colour strings per level, rebuilt only when the theme flips
    var styleDark = null, linkStyle = [], bloomStyle = [], nodeStyle = [];
    function buildStyles(dark) {
      styleDark = dark;
      // dark theme: the jet low end (dark blue) vanishes on black, so lift the
      // colormap floor and the resting opacity to keep the lattice visible
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
      function rgba(c, al) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + al.toFixed(3) + ")"; }
      for (var bb = 0; bb < LEVELS; bb++) {
        var t = bb / (LEVELS - 1);
        var raw = jet(tFloor + t * (HUECAP - tFloor)), col = lift(raw, t);
        linkStyle[bb] = rgba(col, baseA + (CFG.peakAlpha - baseA) * t);
        bloomStyle[bb] = rgba(raw, (dark ? 0.16 : 0.08) * t);
        nodeStyle[bb] = rgba(col, (dark ? 0.9 : CFG.nodeAlpha) * (0.4 + 0.6 * t));
      }
    }

    // counting sort of n items by level lv[] into out (stride floats each);
    // get(i, out, offset) writes item i's coords. Leaves levelStart/levelCount set.
    function bucket(n, lv, out, stride, get) {
      var bb, i, sum = 0;
      for (bb = 0; bb < LEVELS; bb++) levelCount[bb] = 0;
      for (i = 0; i < n; i++) levelCount[lv[i]]++;
      for (bb = 0; bb < LEVELS; bb++) { levelStart[bb] = sum; sum += levelCount[bb]; }
      for (bb = 0; bb < LEVELS; bb++) levelCount[bb] = levelStart[bb];    // reuse as write cursor
      for (i = 0; i < n; i++) get(i, out, levelCount[lv[i]]++ * stride);
      for (bb = 0; bb < LEVELS; bb++) levelCount[bb] -= levelStart[bb];   // back to counts
    }
    function getLink(i, out, o) {
      var a = nodes[links[i][0]], c = nodes[links[i][1]];
      out[o] = a.x; out[o + 1] = a.y; out[o + 2] = c.x; out[o + 3] = c.y;
    }
    function getNode(i, out, o) { out[o] = nodes[i].x; out[o + 1] = nodes[i].y; }

    var shownOpacity = -1;
    function draw(maxStrain) {
      ctx.clearRect(0, 0, W, H);
      var dark = env.dark;
      if (dark !== styleDark) buildStyles(dark);
      // whole-canvas opacity: theme target scaled by the load fade-in (only
      // touch the style when it changes: a style recalc, or a worker message)
      var op = Math.round((dark ? 0.9 : CFG.opacity) * fadeAmt * 100) / 100;
      if (op !== shownOpacity) { env.setOpacity(op); shownOpacity = op; }

      var i, bb, j, e, top = LEVELS - 1;
      // links bucketed by field amplitude -> COMSOL Rainbow (jet) colormap
      for (i = 0; i < links.length; i++) {
        var s = (nodes[links[i][0]].strain + nodes[links[i][1]].strain) * 0.5 / REF;
        linkLv[i] = s >= 1 ? top : (s * top + 0.5) | 0;
      }
      bucket(links.length, linkLv, linkBuf, 4, getLink);
      for (bb = 0; bb < LEVELS; bb++) {
        if (!levelCount[bb]) continue;
        ctx.strokeStyle = linkStyle[bb];
        ctx.lineWidth = 0.6 + (bb / top) * 1.6;
        ctx.beginPath();
        for (j = levelStart[bb] * 4, e = j + levelCount[bb] * 4; j < e; j += 4) {
          ctx.moveTo(linkBuf[j], linkBuf[j + 1]);
          ctx.lineTo(linkBuf[j + 2], linkBuf[j + 3]);
        }
        ctx.stroke();
      }

      // crest bloom: re-stroke the brightest buckets wide and faint with additive
      // blending so wavefronts glow where they overlap. Few links live up here,
      // so it is cheap. Strongest on dark; a gentle touch on light. Tier 0 only.
      ctx.globalCompositeOperation = "lighter";
      if (tier === 0) {
        for (bb = (LEVELS * 0.72) | 0; bb < LEVELS; bb++) {
          if (!levelCount[bb]) continue;
          ctx.strokeStyle = bloomStyle[bb];
          ctx.lineWidth = 3 + (bb / top) * 5;
          ctx.beginPath();
          for (j = levelStart[bb] * 4, e = j + levelCount[bb] * 4; j < e; j += 4) {
            ctx.moveTo(linkBuf[j], linkBuf[j + 1]);
            ctx.lineTo(linkBuf[j + 2], linkBuf[j + 3]);
          }
          ctx.stroke();
        }
      }

      // nodes coloured by the same colormap, brighter/larger where the field is
      // strong. Additive so crossing wavefronts bloom at the nodes too. One path
      // per level; resting nodes (the vast majority) are drawn as cheap squares.
      for (i = 0; i < nodes.length; i++) {
        var ns = nodes[i].strain / maxStrain;
        nodeLv[i] = ns >= 1 ? top : (ns * top + 0.5) | 0;
      }
      bucket(nodes.length, nodeLv, nodeBuf, 2, getNode);
      for (bb = 0; bb < LEVELS; bb++) {
        if (!levelCount[bb]) continue;
        var rad = 0.9 + (bb / top) * 1.9;
        ctx.fillStyle = nodeStyle[bb];
        ctx.beginPath();
        for (j = levelStart[bb] * 2, e = j + levelCount[bb] * 2; j < e; j += 2) {
          if (bb < 3) { ctx.rect(nodeBuf[j] - rad, nodeBuf[j + 1] - rad, rad * 2, rad * 2); continue; }
          ctx.moveTo(nodeBuf[j] + rad, nodeBuf[j + 1]);
          ctx.arc(nodeBuf[j], nodeBuf[j + 1], rad, 0, 6.2832);
        }
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


    function redrawFrozen() { if (reduce) renderStatic(); else if (tier === 3) { displaceNodes(); draw(REF); } }

    // messages from the page: {type: "pointer", x, y, strong} | "resize" {W, H, dpr}
    // | "theme" {dark} | "hidden" {v}
    function handle(m) {
      if (m.type === "pointer") spawnRadial(m.x, m.y, CFG.amp * (m.strong ? 1.8 : 0.55));
      else if (m.type === "resize") { env.W = m.W; env.H = m.H; env.dpr = m.dpr; build(); redrawFrozen(); }
      else if (m.type === "theme") { env.dark = m.dark; redrawFrozen(); }
      else if (m.type === "hidden") env.hidden = m.v;
    }
    function start() {
      build();
      env.onTier(tier);
      if (reduce) { renderStatic(); return; }
      // Random first excitation on every page load: position, polarisation and
      // wave kind all vary, so the background never opens the same way twice.
      autoExcite();
      env.raf(frame);
    }
    return { handle: handle, start: start };
  }

  // ---- worker side: the page transferred the canvas, we draw off the main thread
  if (typeof window === "undefined") {
    var wcore = null;
    self.onmessage = function (e) {
      var m = e.data;
      if (m.type !== "init") { if (wcore) wcore.handle(m); return; }
      var raf = self.requestAnimationFrame ? self.requestAnimationFrame.bind(self)
        : function (f) { return setTimeout(function () { f(performance.now()); }, 16); };
      wcore = createWave({
        canvas: m.canvas, W: m.W, H: m.H, dpr: m.dpr, dark: m.dark, hidden: m.hidden,
        reduce: m.reduce, weak: m.weak, raf: raf,
        setOpacity: function (op) { self.postMessage({ type: "opacity", v: op }); },
        onTier: function (t) { self.postMessage({ type: "tier", v: t }); }
      });
      wcore.start();
    };
    return;
  }

  // ---- page side
  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (document.body && document.body.getAttribute("data-waves") === "static") reduce = true;
  var nav = window.navigator || {};
  var weak = (window.isMobileViewport && window.isMobileViewport()) ||
    (nav.hardwareConcurrency || 8) <= 4 || (nav.deviceMemory || 8) <= 4;
  var scriptSrc = document.currentScript && document.currentScript.src;
  var tierNow = -1;
  window.__waveTier = function () { return tierNow; };   // for perf debugging

  // legibility vignette: dim the field inside a central ellipse so it does not
  // compete with the page text. Done once by the compositor as a CSS mask
  // (alpha vigMin at the centre, 1 at the ellipse edge and beyond).
  var vig = "radial-gradient(ellipse " + (50 * CFG.vigAx) + "% " + (50 * CFG.vigAy) +
    "% at 50% 50%, rgba(0,0,0," + CFG.vigMin + ") 0%, #000 100%)";
  function makeCanvas() {
    var c = document.createElement("canvas");
    c.setAttribute("aria-hidden", "true");
    c.style.cssText =
      "position:fixed;top:0;left:0;width:100%;height:100%;" +
      "z-index:-1;pointer-events:none;opacity:0;" +
      "-webkit-mask-image:" + vig + ";mask-image:" + vig;
    (document.body || document.documentElement).appendChild(c);
    return c;
  }
  function isDark() { return document.documentElement.classList.contains("theme-dark"); }
  function state() {
    return { W: window.innerWidth, H: window.innerHeight, dpr: window.devicePixelRatio || 1,
             dark: isDark(), hidden: document.hidden, reduce: reduce, weak: weak };
  }

  var send;                         // forwards page events to the wave core
  function startOnPage(c) {
    var env = state();
    env.canvas = c;
    env.raf = window.requestAnimationFrame.bind(window);
    env.setOpacity = function (op) { c.style.opacity = op; };
    env.onTier = function (t) { tierNow = t; };
    var core = createWave(env);
    send = core.handle;
    core.start();
  }

  var canvas = makeCanvas();
  var worker = null;
  if (canvas.transferControlToOffscreen && window.Worker && scriptSrc) {
    try {
      var off = canvas.transferControlToOffscreen();
      worker = new Worker(scriptSrc);
      worker.onmessage = function (e) {
        if (e.data.type === "opacity") canvas.style.opacity = e.data.v;
        else if (e.data.type === "tier") tierNow = e.data.v;
      };
      // a worker that cannot run (blocked, no 2D OffscreenCanvas...) falls
      // back to drawing on the page, on a fresh canvas (the old one is detached)
      worker.onerror = function () {
        worker.terminate(); worker = null;
        canvas.remove(); canvas = makeCanvas(); startOnPage(canvas);
      };
      var init = state();
      init.type = "init"; init.canvas = off;
      worker.postMessage(init, [off]);
      send = function (m) { if (worker) worker.postMessage(m); };
    } catch (err) {
      worker = null; canvas.remove(); canvas = makeCanvas();
    }
  }
  if (!worker) startOnPage(canvas);

  // pointer = wave source (throttled here so the worker gets few messages)
  var moveAcc = 0, clickAcc = 0;
  window.addEventListener("pointermove", function (e) {
    var t = performance.now();
    if (t - moveAcc < 90) return;     // throttle ripple injection
    moveAcc = t;
    send({ type: "pointer", x: e.clientX, y: e.clientY, strong: false });
  }, { passive: true });
  window.addEventListener("pointerdown", function (e) {
    var t = performance.now();
    if (t - clickAcc < CFG.clickCooldown) return;   // cooldown between click pulses
    clickAcc = t;
    send({ type: "pointer", x: e.clientX, y: e.clientY, strong: true });
  }, { passive: true });
  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      send({ type: "resize", W: window.innerWidth, H: window.innerHeight, dpr: window.devicePixelRatio || 1 });
    }, 150);
  });
  // theme.js fires this when the user toggles dark/light
  window.addEventListener("themechange", function () { send({ type: "theme", dark: isDark() }); });
  document.addEventListener("visibilitychange", function () { send({ type: "hidden", v: document.hidden }); });
})();
