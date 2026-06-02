/*
 * Graph Neural Network - message passing / label propagation demo.
 * A graph of nodes; you mark a few "seed" nodes with a class colour, then the
 * labels diffuse along the edges (each node averages its neighbours' labels,
 * seeds stay clamped) until the whole graph is classified - exactly the
 * message-passing idea behind GCNs.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("gnn-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var wrap = document.getElementById("gnn-wrap");

  var C = 3;                                  // number of classes
  var COLORS = [[230, 25, 75], [60, 180, 75], [28, 84, 229]];  // red / green / blue
  var DPR = 1, W = 0, H = 380;
  var nodes = [];     // {x,y,h:[..],seed:-1}
  var edges = [];     // [i,j]
  var sel = 0;        // currently selected class for seeding
  var playing = true, frame = 0;

  function rnd(a, b) { return a + (b - a) * Math.random(); }

  function genGraph() {
    var N = 30;
    nodes = [];
    for (var i = 0; i < N; i++) {
      nodes.push({ x: rnd(0.08, 0.92), y: rnd(0.08, 0.92), h: new Array(C).fill(0), seed: -1 });
    }
    // connect each node to its ~3 nearest neighbours
    edges = [];
    var seen = {};
    for (i = 0; i < N; i++) {
      var ds = [];
      for (var j = 0; j < N; j++) if (j !== i) {
        var dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        ds.push([dx * dx + dy * dy, j]);
      }
      ds.sort(function (a, b) { return a[0] - b[0]; });
      for (var k = 0; k < 3; k++) {
        var a = Math.min(i, ds[k][1]), b = Math.max(i, ds[k][1]), key = a + "_" + b;
        if (!seen[key]) { seen[key] = 1; edges.push([a, b]); }
      }
    }
    // a couple of seeds so something propagates immediately
    seedNode((Math.random() * N) | 0, 0);
    seedNode((Math.random() * N) | 0, 1);
    seedNode((Math.random() * N) | 0, 2);
  }

  function seedNode(i, cls) {
    nodes[i].seed = cls;
    nodes[i].h = new Array(C).fill(0);
    nodes[i].h[cls] = 1;
  }

  // adjacency (built from edges each step is cheap for this size)
  function step() {
    var adj = nodes.map(function () { return []; });
    for (var e = 0; e < edges.length; e++) { adj[edges[e][0]].push(edges[e][1]); adj[edges[e][1]].push(edges[e][0]); }
    var nh = nodes.map(function (n) { return n.h.slice(); });
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].seed >= 0) { nh[i] = new Array(C).fill(0); nh[i][nodes[i].seed] = 1; continue; }
      var acc = new Array(C).fill(0), cnt = 0, c;
      acc = acc.map(function (v, idx) { return nodes[i].h[idx] * 0.5; });  // keep half of own state
      cnt += 0.5;
      for (var a = 0; a < adj[i].length; a++) {
        var hj = nodes[adj[i][a]].h;
        for (c = 0; c < C; c++) acc[c] += hj[c];
        cnt += 1;
      }
      var sum = 0;
      for (c = 0; c < C; c++) { acc[c] /= cnt; sum += acc[c]; }
      if (sum > 1e-6) for (c = 0; c < C; c++) acc[c] /= sum;   // normalise to a distribution
      nh[i] = acc;
    }
    for (i = 0; i < nodes.length; i++) nodes[i].h = nh[i];
  }

  function nodeColor(n) {
    var r = 235, g = 235, b = 235, sum = n.h[0] + n.h[1] + n.h[2];
    if (sum > 1e-4) {
      r = g = b = 0;
      for (var c = 0; c < C; c++) { r += n.h[c] * COLORS[c][0]; g += n.h[c] * COLORS[c][1]; b += n.h[c] * COLORS[c][2]; }
      // blend toward white when uncertain (low confidence)
      var conf = Math.max(n.h[0], n.h[1], n.h[2]);
      var t = 0.25 + 0.75 * conf;
      r = r * t + 235 * (1 - t); g = g * t + 235 * (1 - t); b = b * t + 235 * (1 - t);
    }
    return "rgb(" + (r | 0) + "," + (g | 0) + "," + (b | 0) + ")";
  }

  function render() {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    // edges
    ctx.strokeStyle = "rgba(0,0,0,0.18)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (var e = 0; e < edges.length; e++) {
      var a = nodes[edges[e][0]], b = nodes[edges[e][1]];
      ctx.moveTo(a.x * W, a.y * H); ctx.lineTo(b.x * W, b.y * H);
    }
    ctx.stroke();
    // nodes
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i], x = n.x * W, y = n.y * H;
      ctx.beginPath(); ctx.arc(x, y, 11, 0, 6.2832);
      ctx.fillStyle = nodeColor(n); ctx.fill();
      ctx.lineWidth = n.seed >= 0 ? 3 : 1;
      ctx.strokeStyle = n.seed >= 0 ? "#111" : "rgba(0,0,0,0.35)";
      ctx.stroke();
    }
  }

  function loop() {
    requestAnimationFrame(loop);
    if (canvas.offsetParent === null) return;
    if (playing) { frame++; if (frame % 6 === 0) step(); }
    render();
  }

  function resize() {
    var w = wrap.clientWidth || 600;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = w; H = 380;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    canvas.width = W * DPR; canvas.height = H * DPR;
    render();
  }

  // ---- pointer: click a node to seed it with the selected class; drag to move ----
  function nodeAt(e) {
    var r = canvas.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    var px = (src.clientX - r.left) / r.width, py = (src.clientY - r.top) / r.height;
    var best = -1, bd = 1e9;
    for (var i = 0; i < nodes.length; i++) {
      var dx = (nodes[i].x - px) * W, dy = (nodes[i].y - py) * H, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return { i: bd < 20 * 20 ? best : -1, px: px, py: py };
  }
  var drag = -1, downAt = null, moved = false;
  canvas.addEventListener("mousedown", function (e) { var h = nodeAt(e); drag = h.i; downAt = h; moved = false; });
  window.addEventListener("mousemove", function (e) {
    if (drag < 0) return;
    var r = canvas.getBoundingClientRect();
    nodes[drag].x = Math.max(0.03, Math.min(0.97, (e.clientX - r.left) / r.width));
    nodes[drag].y = Math.max(0.03, Math.min(0.97, (e.clientY - r.top) / r.height));
    moved = true;
  });
  window.addEventListener("mouseup", function () {
    if (drag >= 0 && !moved) {                  // a click = toggle seed
      if (nodes[drag].seed === sel) { nodes[drag].seed = -1; nodes[drag].h = new Array(C).fill(0); }
      else seedNode(drag, sel);
    }
    drag = -1;
  });
  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault(); var h = nodeAt(e);
    if (h.i >= 0) { if (nodes[h.i].seed === sel) { nodes[h.i].seed = -1; nodes[h.i].h = new Array(C).fill(0); } else seedNode(h.i, sel); }
  }, { passive: false });

  // ---- controls ----
  function $(id) { return document.getElementById(id); }
  var classBtns = document.querySelectorAll("[data-gnn-class]");
  for (var ci = 0; ci < classBtns.length; ci++) (function (b) {
    b.addEventListener("click", function () {
      sel = +b.getAttribute("data-gnn-class");
      for (var i = 0; i < classBtns.length; i++) classBtns[i].classList.remove("active");
      b.classList.add("active");
    });
  })(classBtns[ci]);
  if ($("gnn-play")) $("gnn-play").addEventListener("click", function () {
    playing = !playing;
    this.innerHTML = playing
      ? '<i class="fas fa-pause"></i> <span class="lang-en">Pause</span><span class="lang-fr">Pause</span>'
      : '<i class="fas fa-play"></i> <span class="lang-en">Play</span><span class="lang-fr">Lancer</span>';
  });
  if ($("gnn-step")) $("gnn-step").addEventListener("click", function () { playing = false; step(); render(); });
  if ($("gnn-reset")) $("gnn-reset").addEventListener("click", function () { genGraph(); render(); });
  if ($("gnn-clear")) $("gnn-clear").addEventListener("click", function () {
    for (var i = 0; i < nodes.length; i++) { nodes[i].seed = -1; nodes[i].h = new Array(C).fill(0); }
  });

  // ---- init ----
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
  genGraph();
  resize();
  requestAnimationFrame(loop);
  window.__gnnResize = resize;
})();
