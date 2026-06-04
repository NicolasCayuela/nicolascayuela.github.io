/*
 * Wallpaper-symmetry drawing pad - built from scratch.
 * Whatever you draw is replicated by every symmetry of the chosen plane
 * group (one of the 17 wallpaper groups), tiled across the canvas.
 *
 * Each group is defined by its "general positions" (point operation + a
 * fractional lattice translation) in crystallographic convention. Those are
 * converted to pixel affine transforms via the lattice basis B, then tiled by
 * the integer lattice. Centred groups (cm, cmm) duplicate every op at +(1/2,1/2).
 */
(function () {
  "use strict";

  var canvas = document.getElementById("sym-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var wrap = document.getElementById("sym-wrap");

  var DPR = 1, W = 0, H = 0;
  var T = 150;                 // translation amount (px)
  var group = "p4m";
  var shapes = [];             // committed primitives
  var redo = [];
  var showGrid = true;

  var COLORS = {
    Black: "#000", Red: "#e51c23", Green: "#0a8f1a", Blue: "#1c54e5",
    Cyan: "#10b9c9", Magenta: "#d61fb5", Yellow: "#e6c20a",
    Orange: "#f57c00", Purple: "#8e24aa", Brown: "#795548"
  };

  // ---- group definitions: ops as [m11,m12,m21,m22, t1,t2] in fractional coords ----
  var ID = [1,0,0,1,0,0], R2 = [-1,0,0,-1,0,0];
  var mY = [-1,0,0,1,0,0], mX = [1,0,0,-1,0,0];     // mirror x=0 / y=0
  var R4 = [0,-1,1,0,0,0], R4b = [0,1,-1,0,0,0];     // square 4-fold
  var dP = [0,1,1,0,0,0], dM = [0,-1,-1,0,0,0];       // square diagonal mirrors
  var R3 = [0,-1,1,-1,0,0], R3b = [-1,1,-1,0,0,0];    // hex 3-fold

  var GROUPS = {
    p1:  { lat: "rect", c: false, ops: [ID] },
    pg:  { lat: "rect", c: false, ops: [ID, [-1,0,0,1,0,0.5]] },
    pm:  { lat: "rect", c: false, ops: [ID, mY] },
    cm:  { lat: "rect", c: true,  ops: [ID, mY] },
    p2:  { lat: "rect", c: false, ops: [ID, R2] },
    pgg: { lat: "rect", c: false, ops: [ID, R2, [1,0,0,-1,0.5,0.5], [-1,0,0,1,0.5,0.5]] },
    pmm: { lat: "rect", c: false, ops: [ID, R2, mY, mX] },
    cmm: { lat: "rect", c: true,  ops: [ID, R2, mY, mX] },
    pmg: { lat: "rect", c: false, ops: [ID, R2, [1,0,0,-1,0.5,0], [-1,0,0,1,0.5,0]] },
    p4:  { lat: "rect", c: false, ops: [ID, R4, R2, R4b] },
    p4m: { lat: "rect", c: false, ops: [ID, R4, R2, R4b, mY, mX, dP, dM] },
    p4g: { lat: "rect", c: false, ops: [ID, R4, R2, R4b,
            [-1,0,0,1,0.5,0.5], [1,0,0,-1,0.5,0.5], [0,1,1,0,0.5,0.5], [0,-1,-1,0,0.5,0.5]] },
    p3:  { lat: "hex", c: false, ops: [ID, R3, R3b] },
    p3m1:{ lat: "hex", c: false, ops: [ID, R3, R3b, [0,1,1,0,0,0], [1,-1,0,-1,0,0], [-1,0,-1,1,0,0]] },
    p31m:{ lat: "hex", c: false, ops: [ID, R3, R3b, [0,-1,-1,0,0,0], [-1,1,0,1,0,0], [1,0,1,-1,0,0]] },
    p6:  { lat: "hex", c: false, ops: [ID, R3, R3b, R2, [0,1,-1,1,0,0], [1,-1,1,0,0,0]] },
    p6m: { lat: "hex", c: false, ops: [ID, R3, R3b, R2, [0,1,-1,1,0,0], [1,-1,1,0,0,0],
            [0,1,1,0,0,0], [1,-1,0,-1,0,0], [-1,0,-1,1,0,0],
            [0,-1,-1,0,0,0], [-1,1,0,1,0,0], [1,0,1,-1,0,0]] }
  };

  // ---- frieze groups: 1D horizontal translation + a horizontal centre line ----
  var FRIEZE = {
    f_p111: ["E"],
    f_p112: ["E", "R2c"],
    f_p1m1: ["E", "mY"],
    f_pm11: ["E", "mXc"],
    f_pmm2: ["E", "R2c", "mXc", "mY"],
    f_p1a1: ["E", "glide"],
    f_pma2: ["E", "mY", "glide", "rot4"]
  };
  function isFrieze() { return group.indexOf("f_") === 0; }

  // ---- aperiodic "hat" monotile (Smith, Myers, Kaplan, Goodman-Strauss 2023) ----
  // Substitution construction adapted from hatviz, (c) 2023 Craig S. Kaplan,
  // BSD-3-Clause - https://github.com/isohedral/hatviz
  var HAT = (function () {
    var hr3 = 0.8660254037844386;
    function pt(x, y) { return { x: x, y: y }; }
    function hexPt(x, y) { return pt(x + 0.5 * y, hr3 * y); }
    function inv(T) {
      var det = T[0] * T[4] - T[1] * T[3];
      return [T[4] / det, -T[1] / det, (T[1] * T[5] - T[2] * T[4]) / det,
              -T[3] / det, T[0] / det, (T[2] * T[3] - T[0] * T[5]) / det];
    }
    function mul(A, B) {
      return [A[0] * B[0] + A[1] * B[3], A[0] * B[1] + A[1] * B[4], A[0] * B[2] + A[1] * B[5] + A[2],
              A[3] * B[0] + A[4] * B[3], A[3] * B[1] + A[4] * B[4], A[3] * B[2] + A[4] * B[5] + A[5]];
    }
    function padd(p, q) { return pt(p.x + q.x, p.y + q.y); }
    function psub(p, q) { return pt(p.x - q.x, p.y - q.y); }
    function trot(a) { var c = Math.cos(a), s = Math.sin(a); return [c, -s, 0, s, c, 0]; }
    function ttrans(tx, ty) { return [1, 0, tx, 0, 1, ty]; }
    function rotAbout(p, a) { return mul(ttrans(p.x, p.y), mul(trot(a), ttrans(-p.x, -p.y))); }
    function transPt(M, P) { return pt(M[0] * P.x + M[1] * P.y + M[2], M[3] * P.x + M[4] * P.y + M[5]); }
    function matchSeg(p, q) { return [q.x - p.x, p.y - q.y, p.x, q.y - p.y, q.x - p.x, p.y]; }
    function matchTwo(p1, q1, p2, q2) { return mul(matchSeg(p2, q2), inv(matchSeg(p1, q1))); }
    function intersect(p1, q1, p2, q2) {
      var d = (q2.y - p2.y) * (q1.x - p1.x) - (q2.x - p2.x) * (q1.y - p1.y);
      var uA = ((q2.x - p2.x) * (p1.y - p2.y) - (q2.y - p2.y) * (p1.x - p2.x)) / d;
      return pt(p1.x + uA * (q1.x - p1.x), p1.y + uA * (q1.y - p1.y));
    }

    var OUTLINE = [hexPt(0, 0), hexPt(-1, -1), hexPt(0, -2), hexPt(2, -2),
                   hexPt(2, -1), hexPt(4, -2), hexPt(5, -1), hexPt(4, 0),
                   hexPt(3, 0), hexPt(2, 2), hexPt(0, 3), hexPt(0, 2), hexPt(-1, 2)];
    var HATGEOM = { hat: true, shape: OUTLINE };

    function meta(shape) { return { hat: false, shape: shape, children: [] }; }
    function addChild(m, T, g) { m.children.push({ T: T, geom: g }); }
    function evalChild(m, n, i) { return transPt(m.children[n].T, m.children[n].geom.shape[i]); }
    function recentre(m) {
      var cx = 0, cy = 0, i;
      for (i = 0; i < m.shape.length; i++) { cx += m.shape[i].x; cy += m.shape[i].y; }
      cx /= m.shape.length; cy /= m.shape.length;
      for (i = 0; i < m.shape.length; i++) m.shape[i] = pt(m.shape[i].x - cx, m.shape[i].y - cy);
      var M = ttrans(-cx, -cy);
      for (i = 0; i < m.children.length; i++) m.children[i].T = mul(M, m.children[i].T);
    }

    function buildInit() {
      var H = meta([pt(0, 0), pt(4, 0), pt(4.5, hr3), pt(2.5, 5 * hr3), pt(1.5, 5 * hr3), pt(-0.5, hr3)]);
      addChild(H, matchTwo(OUTLINE[5], OUTLINE[7], H.shape[5], H.shape[0]), HATGEOM);
      addChild(H, matchTwo(OUTLINE[9], OUTLINE[11], H.shape[1], H.shape[2]), HATGEOM);
      addChild(H, matchTwo(OUTLINE[5], OUTLINE[7], H.shape[3], H.shape[4]), HATGEOM);
      addChild(H, mul(ttrans(2.5, hr3), mul([-0.5, -hr3, 0, hr3, -0.5, 0], [0.5, 0, 0, 0, -0.5, 0])), HATGEOM);
      var T = meta([pt(0, 0), pt(3, 0), pt(1.5, 3 * hr3)]);
      addChild(T, [0.5, 0, 0.5, 0, 0.5, hr3], HATGEOM);
      var P = meta([pt(0, 0), pt(4, 0), pt(3, 2 * hr3), pt(-1, 2 * hr3)]);
      addChild(P, [0.5, 0, 1.5, 0, 0.5, hr3], HATGEOM);
      addChild(P, mul(ttrans(0, 2 * hr3), mul([0.5, hr3, 0, -hr3, 0.5, 0], [0.5, 0, 0, 0, 0.5, 0])), HATGEOM);
      var F = meta([pt(0, 0), pt(3, 0), pt(3.5, hr3), pt(3, 2 * hr3), pt(-1, 2 * hr3)]);
      addChild(F, [0.5, 0, 1.5, 0, 0.5, hr3], HATGEOM);
      addChild(F, mul(ttrans(0, 2 * hr3), mul([0.5, hr3, 0, -hr3, 0.5, 0], [0.5, 0, 0, 0, 0.5, 0])), HATGEOM);
      return [H, T, P, F];
    }

    function constructPatch(H, T, P, F) {
      var rules = [
        ['H'], [0, 0, 'P', 2], [1, 0, 'H', 2], [2, 0, 'P', 2], [3, 0, 'H', 2],
        [4, 4, 'P', 2], [0, 4, 'F', 3], [2, 4, 'F', 3], [4, 1, 3, 2, 'F', 0],
        [8, 3, 'H', 0], [9, 2, 'P', 0], [10, 2, 'H', 0], [11, 4, 'P', 2],
        [12, 0, 'H', 2], [13, 0, 'F', 3], [14, 2, 'F', 1], [15, 3, 'H', 4],
        [8, 2, 'F', 1], [17, 3, 'H', 0], [18, 2, 'P', 0], [19, 2, 'H', 2],
        [20, 4, 'F', 3], [20, 0, 'P', 2], [22, 0, 'H', 2], [23, 4, 'F', 3],
        [23, 0, 'F', 3], [16, 0, 'P', 2], [9, 4, 0, 2, 'T', 2], [4, 0, 'F', 3]
      ];
      var ret = meta([]);
      var shapesMap = { H: H, T: T, P: P, F: F };
      for (var k = 0; k < rules.length; k++) {
        var r = rules[k];
        if (r.length === 1) {
          addChild(ret, [1, 0, 0, 0, 1, 0], shapesMap[r[0]]);
        } else if (r.length === 4) {
          var poly = ret.children[r[0]].geom.shape;
          var Tm = ret.children[r[0]].T;
          var Pp = transPt(Tm, poly[(r[1] + 1) % poly.length]);
          var Qp = transPt(Tm, poly[r[1]]);
          var nshp = shapesMap[r[2]], npoly = nshp.shape;
          addChild(ret, matchTwo(npoly[r[3]], npoly[(r[3] + 1) % npoly.length], Pp, Qp), nshp);
        } else {
          var chP = ret.children[r[0]], chQ = ret.children[r[2]];
          var Pq = transPt(chQ.T, chQ.geom.shape[r[3]]);
          var Qq = transPt(chP.T, chP.geom.shape[r[1]]);
          var nshp2 = shapesMap[r[4]], npoly2 = nshp2.shape;
          addChild(ret, matchTwo(npoly2[r[5]], npoly2[(r[5] + 1) % npoly2.length], Pq, Qq), nshp2);
        }
      }
      return ret;
    }

    function constructMetatiles(patch) {
      var PI = Math.PI;
      var bps1 = evalChild(patch, 8, 2);
      var bps2 = evalChild(patch, 21, 2);
      var rbps = transPt(rotAbout(bps1, -2 * PI / 3), bps2);
      var p72 = evalChild(patch, 7, 2);
      var p252 = evalChild(patch, 25, 2);
      var llc = intersect(bps1, rbps, evalChild(patch, 6, 2), p72);
      var w = psub(evalChild(patch, 6, 2), llc);

      var nH = [llc, bps1];
      w = transPt(trot(-PI / 3), w);
      nH.push(padd(nH[1], w));
      nH.push(evalChild(patch, 14, 2));
      w = transPt(trot(-PI / 3), w);
      nH.push(psub(nH[3], w));
      nH.push(evalChild(patch, 6, 2));
      var newH = meta(nH);
      [0, 9, 16, 27, 26, 6, 1, 8, 10, 15].forEach(function (ch) {
        addChild(newH, patch.children[ch].T, patch.children[ch].geom);
      });

      var newP = meta([p72, padd(p72, psub(bps1, llc)), bps1, llc]);
      [7, 2, 3, 4, 28].forEach(function (ch) {
        addChild(newP, patch.children[ch].T, patch.children[ch].geom);
      });

      var newF = meta([bps2, evalChild(patch, 24, 2), evalChild(patch, 25, 0),
                       p252, padd(p252, psub(llc, bps1))]);
      [21, 20, 22, 23, 24, 25].forEach(function (ch) {
        addChild(newF, patch.children[ch].T, patch.children[ch].geom);
      });

      var AAA = nH[2];
      var BBB = padd(nH[1], psub(nH[4], nH[5]));
      var CCC = transPt(rotAbout(BBB, -PI / 3), AAA);
      var newT = meta([BBB, CCC, AAA]);
      addChild(newT, patch.children[11].T, patch.children[11].geom);

      recentre(newH); recentre(newP); recentre(newF); recentre(newT);
      return [newH, newT, newP, newF];
    }

    var instances = null;          // flat list of hat transforms (tile coords)
    function build() {
      if (instances) return instances;
      var tiles = buildInit();
      for (var k = 0; k < 4; k++) {
        tiles = constructMetatiles(constructPatch(tiles[0], tiles[1], tiles[2], tiles[3]));
      }
      instances = [];
      (function collect(g, Tm) {
        if (g.hat) { instances.push(Tm); return; }
        for (var i = 0; i < g.children.length; i++) {
          collect(g.children[i].geom, mul(Tm, g.children[i].T));
        }
      })(tiles[0], [1, 0, 0, 0, 1, 0]);
      return instances;
    }

    // centroid of the hat outline (local coords)
    var cen = (function () {
      var cx = 0, cy = 0;
      for (var i = 0; i < OUTLINE.length; i++) { cx += OUTLINE[i].x; cy += OUTLINE[i].y; }
      return pt(cx / OUTLINE.length, cy / OUTLINE.length);
    })();

    return { build: build, outline: OUTLINE, centroid: cen,
             mul: mul, inv: inv, ttrans: ttrans, transPt: transPt };
  })();
  function isHat() { return group === "hat"; }

  var ax, ay, bx, by;          // lattice basis vectors (px)
  var pixelOps = [];           // affine {a,b,c,d,e,f}
  var frieze = false;
  var hat = false;
  var hatMats = [];            // screen-space hat transforms (for outlines)
  var hatRef = 0;              // index of the reference hat in hatMats

  function basis() {
    if (GROUPS[group].lat === "hex") { ax = T; ay = 0; bx = T / 2; by = T * Math.sqrt(3) / 2; }
    else { ax = T; ay = 0; bx = 0; by = T; }
  }

  function buildFrieze() {
    frieze = true;
    var yc = H / 2, t2 = 2 * yc, h = T / 2;
    var defs = {
      E:    { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 },
      mY:   { a: -1, b: 0, c: 0, d: 1, e: 0, f: 0 },            // vertical mirror x=0
      mXc:  { a: 1, b: 0, c: 0, d: -1, e: 0, f: t2 },           // horizontal mirror y=yc
      R2c:  { a: -1, b: 0, c: 0, d: -1, e: 0, f: t2 },          // 180 about (0,yc)
      glide:{ a: 1, b: 0, c: 0, d: -1, e: h, f: t2 },           // glide: x+T/2, mirror y=yc
      rot4: { a: -1, b: 0, c: 0, d: -1, e: h, f: t2 }           // 180 about (T/4,yc)
    };
    pixelOps = FRIEZE[group].map(function (k) { return defs[k]; });
    ax = T; ay = 0; bx = 0; by = 0;
  }

  function buildHat() {
    frieze = false; hat = true;
    ax = 0; ay = 0; bx = 0; by = 0;          // no lattice: ops already cover the plane
    var all = HAT.build();
    var s = T / 2;                            // hat is ~2 units wide here -> ~T px
    var S = HAT.mul(HAT.ttrans(W / 2, H / 2), [s, 0, 0, 0, -s, 0]);
    var m = 1.5 * T;                          // cull margin (strokes can overflow a tile)
    hatMats = [];
    var best = 1e18; hatRef = 0;
    for (var i = 0; i < all.length; i++) {
      var M = HAT.mul(S, all[i]);
      var c = HAT.transPt(M, HAT.centroid);
      if (c.x < -m || c.x > W + m || c.y < -m || c.y > H + m) continue;
      hatMats.push(M);
      var d = (c.x - W / 2) * (c.x - W / 2) + (c.y - H / 2) * (c.y - H / 2);
      if (d < best) { best = d; hatRef = hatMats.length - 1; }
    }
    var iR = HAT.inv(hatMats[hatRef]);
    pixelOps = hatMats.map(function (M) {
      var L = HAT.mul(M, iR);                 // maps ref-hat screen coords -> this hat
      return { a: L[0], b: L[3], c: L[1], d: L[4], e: L[2], f: L[5] };
    });
  }

  function buildOps() {
    if (isHat()) { buildHat(); return; }
    hat = false;
    if (isFrieze()) { buildFrieze(); return; }
    frieze = false;
    basis();
    var det = ax * by - bx * ay;
    var iB11 = by / det, iB12 = -bx / det, iB21 = -ay / det, iB22 = ax / det; // B^-1
    var raw = GROUPS[group].ops.slice();
    if (GROUPS[group].c) {                       // centring: duplicate at +(1/2,1/2)
      var ext = [];
      for (var i = 0; i < raw.length; i++) {
        var o = raw[i];
        ext.push([o[0], o[1], o[2], o[3], o[4] + 0.5, o[5] + 0.5]);
      }
      raw = raw.concat(ext);
    }
    pixelOps = raw.map(function (o) {
      var m11 = o[0], m12 = o[1], m21 = o[2], m22 = o[3];
      // BM = B * M
      var BM11 = ax * m11 + bx * m21, BM12 = ax * m12 + bx * m22;
      var BM21 = ay * m11 + by * m21, BM22 = ay * m12 + by * m22;
      // L = BM * B^-1
      var L11 = BM11 * iB11 + BM12 * iB21, L12 = BM11 * iB12 + BM12 * iB22;
      var L21 = BM21 * iB11 + BM22 * iB21, L22 = BM21 * iB12 + BM22 * iB22;
      var tx = ax * o[4] + bx * o[5], ty = ay * o[4] + by * o[5];
      return { a: L11, b: L21, c: L12, d: L22, e: tx, f: ty };
    });
  }

  function resize() {
    var w = wrap.clientWidth || 600;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = w; H = 460;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    canvas.width = W * DPR; canvas.height = H * DPR;
    buildOps();                 // frieze ops depend on canvas height
    render();
  }

  function setT(ctxT) { ctx.setTransform(DPR * ctxT.a, DPR * ctxT.b, DPR * ctxT.c, DPR * ctxT.d, DPR * ctxT.e, DPR * ctxT.f); }

  function drawPrimitive(s) {
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.width;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    var x1 = s.x1, y1 = s.y1, x2 = s.x2, y2 = s.y2;
    if (s.tool === "line") { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
    else if (s.tool === "freehand") {
      var p = s.points; if (!p || p.length < 2) return;
      ctx.beginPath(); ctx.moveTo(p[0], p[1]);
      for (var i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
      ctx.stroke();
    } else {
      var x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
      if (s.tool === "rect") ctx.strokeRect(x, y, w, h);
      else if (s.tool === "frect") ctx.fillRect(x, y, w, h);
      else {
        ctx.beginPath();
        ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, 6.2832);
        if (s.tool === "foval") ctx.fill(); else ctx.stroke();
      }
    }
  }

  function drawGrid() {
    setT({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
    ctx.strokeStyle = "rgba(0,0,0,0.18)"; ctx.lineWidth = 0.6;
    if (hat) {                                 // hat outlines; reference tile highlighted
      var o = HAT.outline, k, p, q;
      for (k = 0; k < hatMats.length; k++) {
        ctx.beginPath();
        p = HAT.transPt(hatMats[k], o[0]);
        ctx.moveTo(p.x, p.y);
        for (var v = 1; v < o.length; v++) { q = HAT.transPt(hatMats[k], o[v]); ctx.lineTo(q.x, q.y); }
        ctx.closePath();
        if (k === hatRef) {
          ctx.save();
          ctx.fillStyle = "rgba(28,84,229,0.08)"; ctx.fill();
          ctx.strokeStyle = "rgba(28,84,229,0.65)"; ctx.lineWidth = 1.4; ctx.stroke();
          ctx.restore();
          ctx.strokeStyle = "rgba(0,0,0,0.18)"; ctx.lineWidth = 0.6;
        } else {
          ctx.stroke();
        }
      }
      return;
    }
    var rng = latticeRange(), i, j;
    if (frieze) {
      var yc = H / 2;
      ctx.beginPath();
      for (i = rng.i0; i <= rng.i1; i++) { ctx.moveTo(i * T, 0); ctx.lineTo(i * T, H); }
      ctx.moveTo(0, yc); ctx.lineTo(W, yc);
      ctx.stroke();
      return;
    }
    ctx.beginPath();
    for (i = rng.i0; i <= rng.i1; i++) {           // lines along b
      ctx.moveTo(i * ax + rng.j0 * bx, i * ay + rng.j0 * by);
      ctx.lineTo(i * ax + rng.j1 * bx, i * ay + rng.j1 * by);
    }
    for (j = rng.j0; j <= rng.j1; j++) {           // lines along a
      ctx.moveTo(rng.i0 * ax + j * bx, rng.i0 * ay + j * by);
      ctx.lineTo(rng.i1 * ax + j * bx, rng.i1 * ay + j * by);
    }
    if (GROUPS[group].lat === "hex") {             // extra diagonal => triangular cells
      for (i = rng.i0; i <= rng.i1; i++)
        for (j = rng.j0; j <= rng.j1; j++) {
          ctx.moveTo(i * ax + j * bx, i * ay + j * by);
          ctx.lineTo((i + 1) * ax + (j - 1) * bx, (i + 1) * ay + (j - 1) * by);
        }
    }
    ctx.stroke();
  }

  function latticeRange() {
    if (hat) return { i0: 0, i1: 0, j0: 0, j1: 0 };   // ops already enumerate every tile
    if (frieze) return { i0: Math.floor(0 / T) - 2, i1: Math.ceil(W / T) + 2, j0: 0, j1: 0 };
    var det = ax * by - bx * ay;
    var iB11 = by / det, iB12 = -bx / det, iB21 = -ay / det, iB22 = ax / det;
    var minI = 1e9, maxI = -1e9, minJ = 1e9, maxJ = -1e9;
    var cs = [[0, 0], [W, 0], [0, H], [W, H]];
    for (var k = 0; k < 4; k++) {
      var li = iB11 * cs[k][0] + iB12 * cs[k][1];
      var lj = iB21 * cs[k][0] + iB22 * cs[k][1];
      if (li < minI) minI = li; if (li > maxI) maxI = li;
      if (lj < minJ) minJ = lj; if (lj > maxJ) maxJ = lj;
    }
    return {
      i0: Math.floor(minI) - 2, i1: Math.ceil(maxI) + 2,
      j0: Math.floor(minJ) - 2, j1: Math.ceil(maxJ) + 2
    };
  }

  function render(preview) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    if (showGrid) drawGrid();

    var list = preview ? shapes.concat([preview]) : shapes;
    if (!list.length) return;
    var rng = latticeRange();
    // safety cap on number of cells (tiny T)
    var cells = (rng.i1 - rng.i0 + 1) * (rng.j1 - rng.j0 + 1);
    if (cells > 4000) return;
    for (var o = 0; o < pixelOps.length; o++) {
      var op = pixelOps[o];
      for (var i = rng.i0; i <= rng.i1; i++) {
        for (var j = rng.j0; j <= rng.j1; j++) {
          var Lx = i * ax + j * bx, Ly = i * ay + j * by;
          setT({ a: op.a, b: op.b, c: op.c, d: op.d, e: op.e + Lx, f: op.f + Ly });
          for (var s = 0; s < list.length; s++) drawPrimitive(list[s]);
        }
      }
    }
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  // ---- drawing interaction ----
  function tool() { var el = document.querySelector('input[name="sym-tool"]:checked'); return el ? el.value : "freehand"; }
  function width() { var el = document.querySelector('input[name="sym-width"]:checked'); return el ? +el.value : 3; }
  function color() { var el = document.querySelector('input[name="sym-color"]:checked'); return el ? COLORS[el.value] : "#000"; }

  function pos(e) {
    var r = canvas.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    return { x: src.clientX - r.left, y: src.clientY - r.top };
  }

  var drawing = false, cur = null;
  function start(e) {
    if (e.cancelable) e.preventDefault();
    var p = pos(e);
    cur = { tool: tool(), color: color(), width: width(), x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    if (cur.tool === "freehand") cur.points = [p.x, p.y];
    drawing = true;
    render(cur);
  }
  function move(e) {
    if (!drawing) return;
    if (e.cancelable) e.preventDefault();
    var p = pos(e);
    cur.x2 = p.x; cur.y2 = p.y;
    if (cur.tool === "freehand") cur.points.push(p.x, p.y);
    render(cur);
  }
  function end() {
    if (!drawing) return;
    drawing = false;
    var ok = cur.tool === "freehand" ? cur.points.length >= 4 : (cur.x1 !== cur.x2 || cur.y1 !== cur.y2);
    if (ok) { shapes.push(cur); redo = []; updateButtons(); }
    cur = null;
    render();
  }
  canvas.addEventListener("mousedown", start);
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  canvas.addEventListener("touchstart", start, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", end);

  // ---- controls ----
  function $(id) { return document.getElementById(id); }
  function updateButtons() {
    $("sym-undo").disabled = shapes.length === 0;
    $("sym-redo").disabled = redo.length === 0;
    $("sym-clear").disabled = shapes.length === 0;
  }
  $("sym-undo").addEventListener("click", function () { if (shapes.length) { redo.push(shapes.pop()); updateButtons(); render(); } });
  $("sym-redo").addEventListener("click", function () { if (redo.length) { shapes.push(redo.pop()); updateButtons(); render(); } });
  $("sym-clear").addEventListener("click", function () { if (shapes.length) { shapes = []; redo = []; updateButtons(); render(); } });

  $("sym-apply").addEventListener("click", function () {
    var v = parseInt($("sym-trans").value, 10);
    if (v >= 20 && v <= 600) { T = v; buildOps(); render(); }
  });
  var grpRadios = document.querySelectorAll('input[name="sym-group"]');
  for (var g = 0; g < grpRadios.length; g++) grpRadios[g].addEventListener("change", function () {
    group = this.value; buildOps(); render();
  });
  $("sym-grid").addEventListener("change", function () { showGrid = this.checked; render(); });

  // ---- init ----
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
  buildOps();
  updateButtons();
  resize();
  window.__symResize = resize;   // let the tab switcher re-layout when shown
})();
