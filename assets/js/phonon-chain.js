/*
 * 1D diatomic mass-spring chain: the classic phononics dispersion demo.
 * Left: dispersion relation w(q) with acoustic and optical branches and the
 * band gap. Drag the marker along the curves to pick a mode (q, branch);
 * the chain below animates that exact Bloch mode. A mass-ratio slider opens
 * and closes the gap.
 *
 * Units: spring constant k = 1, light mass m2 = 1, lattice constant a = 1.
 * Dispersion: w^2 = k(1/m1+1/m2) +/- k sqrt[(1/m1+1/m2)^2 - 4 sin^2(q/2)/(m1 m2)]
 */
(function () {
  "use strict";

  var area = document.getElementById("phonon-area");
  var dispC = document.getElementById("phonon-disp");
  var chainC = document.getElementById("phonon-chain");
  if (!area || !dispC || !chainC) return;
  var dctx = dispC.getContext("2d");
  var cctx = chainC.getContext("2d");

  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var DW = 420, DH = 300;          // dispersion logical size
  var CW = 760, CH = 130;          // chain logical size
  var ratio = 2;                   // m1/m2 (heavy over light)
  var q = Math.PI * 0.6;           // selected wavevector in [0, pi]
  var branch = "ac";               // "ac" | "opt"
  var t = 0, last = 0;
  var running = false;

  var K = 1, M2 = 1;
  function m1() { return ratio * M2; }

  function omega(qv, br) {
    var s = 1 / m1() + 1 / M2;
    var disc = s * s - 4 * Math.sin(qv / 2) * Math.sin(qv / 2) / (m1() * M2);
    disc = Math.max(0, disc);
    var w2 = K * s + (br === "opt" ? 1 : -1) * K * Math.sqrt(disc);
    return Math.sqrt(Math.max(0, w2));
  }
  function omegaMax() { return Math.sqrt(2 * K * (1 / m1() + 1 / M2)); }
  function gap() {                  // [top of acoustic, bottom of optical]
    return [Math.sqrt(2 * K / m1()), Math.sqrt(2 * K / M2)];
  }

  // complex amplitude ratio B/A for the light mass, picking the
  // better-conditioned of the two equivalent expressions
  function amplitudeRatio(qv, br) {
    var w2 = omega(qv, br); w2 = w2 * w2;
    var d1 = { re: K * (1 + Math.cos(qv)), im: K * Math.sin(qv) };       // k(1+e^{iq})
    var n1 = 2 * K - M2 * w2;                                            // (2k - m2 w^2)
    var d2 = { re: K * (1 + Math.cos(qv)), im: -K * Math.sin(qv) };      // k(1+e^{-iq})
    var n2 = 2 * K - m1() * w2;
    // B/A = d1/n1  ==  n2/d2
    if (Math.abs(n1) > 1e-6) return { re: d1.re / n1, im: d1.im / n1 };
    var m = d2.re * d2.re + d2.im * d2.im;
    if (m < 1e-12) return { re: 0, im: 0 };
    return { re: n2 * d2.re / m, im: -n2 * d2.im / m };
  }

  // ---- layout ----
  function sizeCanvas(c, ctx, w, h) {
    c.style.width = w + "px"; c.style.height = h + "px";
    c.width = w * DPR; c.height = h * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  function layout() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    var w = area.clientWidth || 760;
    DW = Math.max(300, Math.min(440, w * 0.55));
    CW = Math.max(300, Math.min(820, w - 16));
    sizeCanvas(dispC, dctx, DW, DH);
    sizeCanvas(chainC, cctx, CW, CH);
    drawDisp();
  }

  // dispersion plot mapping
  var PAD = { l: 44, r: 12, t: 14, b: 32 };
  function qToX(qv) { return PAD.l + (qv / Math.PI) * (DW - PAD.l - PAD.r); }
  function wToY(w) { return DH - PAD.b - (w / (omegaMax() * 1.06)) * (DH - PAD.t - PAD.b); }
  function xToQ(x) { return Math.max(0.001, Math.min(Math.PI, (x - PAD.l) / (DW - PAD.l - PAD.r) * Math.PI)); }

  function drawDisp() {
    dctx.clearRect(0, 0, DW, DH);
    dctx.fillStyle = "#fff"; dctx.fillRect(0, 0, DW, DH);
    var g = gap(), i, qq;

    // band gap shading
    dctx.fillStyle = "rgba(150,160,180,0.18)";
    dctx.fillRect(PAD.l, wToY(g[1]), DW - PAD.l - PAD.r, wToY(g[0]) - wToY(g[1]));

    // axes
    dctx.strokeStyle = "#c8cdd6"; dctx.lineWidth = 1;
    dctx.beginPath();
    dctx.moveTo(PAD.l, PAD.t); dctx.lineTo(PAD.l, DH - PAD.b); dctx.lineTo(DW - PAD.r, DH - PAD.b);
    dctx.stroke();
    dctx.fillStyle = "#8a93a3"; dctx.font = "11px sans-serif";
    dctx.fillText("ω", 8, PAD.t + 8);
    dctx.fillText("0", PAD.l - 10, DH - PAD.b + 14);
    dctx.fillText("q·a/π = 1", DW - PAD.r - 52, DH - PAD.b + 14);

    // branches
    var branches = [["ac", "#1c54e5"], ["opt", "#f57c00"]];
    for (var b = 0; b < 2; b++) {
      dctx.strokeStyle = branches[b][1]; dctx.lineWidth = 2.2;
      dctx.beginPath();
      for (i = 0; i <= 160; i++) {
        qq = (i / 160) * Math.PI;
        var x = qToX(qq), y = wToY(omega(qq, branches[b][0]));
        if (i === 0) dctx.moveTo(x, y); else dctx.lineTo(x, y);
      }
      dctx.stroke();
    }

    // gap label
    if (g[1] - g[0] > 0.02) {
      dctx.fillStyle = "#6a7385"; dctx.font = "italic 11px sans-serif";
      dctx.fillText("band gap", PAD.l + 8, (wToY(g[0]) + wToY(g[1])) / 2 + 4);
    }

    // marker
    var mx = qToX(q), my = wToY(omega(q, branch));
    dctx.beginPath(); dctx.arc(mx, my, 7, 0, 6.2832);
    dctx.fillStyle = branch === "ac" ? "#1c54e5" : "#f57c00";
    dctx.fill();
    dctx.strokeStyle = "#fff"; dctx.lineWidth = 2; dctx.stroke();

    // readout
    var label = document.getElementById("phonon-readout");
    if (label) {
      var w = omega(q, branch);
      label.innerHTML =
        '<span class="lang-en">' + (branch === "ac" ? "acoustic" : "optical") +
        " branch &nbsp; q·a/π = " + (q / Math.PI).toFixed(2) +
        " &nbsp; ω = " + w.toFixed(3) + " &nbsp; gap: [" + g[0].toFixed(2) + ", " + g[1].toFixed(2) + "]</span>" +
        '<span class="lang-fr">branche ' + (branch === "ac" ? "acoustique" : "optique") +
        " &nbsp; q·a/π = " + (q / Math.PI).toFixed(2) +
        " &nbsp; ω = " + w.toFixed(3) + " &nbsp; gap : [" + g[0].toFixed(2) + ", " + g[1].toFixed(2) + "]</span>";
    }
  }

  // ---- chain animation ----
  var NCELLS = 12;
  function drawChain() {
    cctx.clearRect(0, 0, CW, CH);
    cctx.fillStyle = "#fff"; cctx.fillRect(0, 0, CW, CH);
    var w = omega(q, branch);
    var BA = amplitudeRatio(q, branch);
    var cell = CW / (NCELLS + 1);
    var amp = cell * 0.22;                       // exaggerated displacement
    var y = CH / 2;
    var r1 = 11 * Math.cbrt(m1() / 2), r2 = 11 * Math.cbrt(M2 / 2);
    var xs = [], rs = [], n;

    for (n = 0; n < NCELLS; n++) {
      var phase = q * n - w * t;
      // heavy mass at cell origin, light mass half a cell later
      var u = amp * Math.cos(phase);
      var phase2 = q * (n + 0.5) - w * t;
      var v = amp * (BA.re * Math.cos(phase2) - BA.im * Math.sin(phase2));
      xs.push(cell * (0.8 + n) + u); rs.push(r1);
      xs.push(cell * (1.3 + n) + v); rs.push(r2);
    }
    // springs
    cctx.strokeStyle = "#b9c2d0"; cctx.lineWidth = 2;
    cctx.beginPath();
    for (n = 0; n < xs.length - 1; n++) { cctx.moveTo(xs[n], y); cctx.lineTo(xs[n + 1], y); }
    cctx.stroke();
    // masses
    for (n = 0; n < xs.length; n++) {
      cctx.beginPath(); cctx.arc(xs[n], y, rs[n], 0, 6.2832);
      cctx.fillStyle = n % 2 === 0 ? "#1c54e5" : "#f57c00";
      cctx.fill();
      cctx.strokeStyle = "#fff"; cctx.lineWidth = 1.5; cctx.stroke();
    }
  }

  function tick(ts) {
    if (!running) return;
    if (last) t += Math.min(0.05, (ts - last) / 1000) * 2.2;   // time scale
    last = ts;
    drawChain();
    requestAnimationFrame(tick);
  }

  // ---- interactions ----
  function pickMode(e) {
    var rect = dispC.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    var x = src.clientX - rect.left, yy = src.clientY - rect.top;
    var qv = xToQ(x);
    var dAc = Math.abs(wToY(omega(qv, "ac")) - yy);
    var dOpt = Math.abs(wToY(omega(qv, "opt")) - yy);
    q = qv; branch = dAc <= dOpt ? "ac" : "opt";
    drawDisp();
  }
  var dragging = false;
  dispC.addEventListener("mousedown", function (e) { dragging = true; pickMode(e); });
  window.addEventListener("mousemove", function (e) { if (dragging) pickMode(e); });
  window.addEventListener("mouseup", function () { dragging = false; });
  dispC.addEventListener("touchstart", function (e) { e.preventDefault(); dragging = true; pickMode(e); }, { passive: false });
  dispC.addEventListener("touchmove", function (e) { e.preventDefault(); if (dragging) pickMode(e); }, { passive: false });
  dispC.addEventListener("touchend", function () { dragging = false; });

  var slider = document.getElementById("phonon-ratio");
  if (slider) slider.addEventListener("input", function () {
    ratio = parseFloat(slider.value);
    var lab = document.getElementById("phonon-ratio-val");
    if (lab) lab.textContent = "m₁/m₂ = " + ratio.toFixed(2);
    drawDisp();
  });

  function show() {
    layout();
    if (!running) { running = true; last = 0; requestAnimationFrame(tick); }
  }
  function hide() { running = false; }

  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(layout, 200); });
  window.__phononShow = show;
  window.__phononHide = hide;
})();
