/*
 * 1D phononic band-gap mechanisms, after Laude, "Phononic Crystals" ch. 1-2.
 *
 * Two switchable models:
 *  - Bragg: diatomic mass-spring chain. Gap between Omega1 = sqrt(2K/m1) and
 *    Omega2 = sqrt(2K/m2), opened by periodicity (lambda ~ 2a).
 *    In the gap k = pi/a + i ki with
 *    ki a = acosh(1 - (w^2/O1^2 - 1)(w^2/O2^2 - 1))  -> symmetric loop.
 *  - Local resonance: mass-in-mass chain (outer M, inner resonator m_r, w_r).
 *    Dispersion M w^4 - w^2((M+m_r) w_r^2 + S) + S w_r^2 = 0, S = 2K(1-cos qa).
 *    Sub-wavelength gap pinned to w_r; Im(k) is an asymmetric peak and
 *    Re(k) flips from pi/a to 0 across the resonance.
 *
 * The left half of the plot shows Im(k) (complex band structure), the right
 * half Re(k). The chain below animates the exact Bloch mode at the marker.
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
  var DW = 460, DH = 300;
  var CW = 760, CH = 130;
  var mech = "bragg";              // "bragg" | "res"
  var ratio = 2;                   // Bragg: m1/m2
  var wr = 0.8;                    // resonance: internal resonator frequency
  var MR = 0.5;                    // resonance: m_r / M
  var q = Math.PI * 0.6;
  var branch = 0;                  // band index (0 = lower, 1 = upper)
  var t = 0, last = 0, running = false;

  var K = 1, M2 = 1;
  function m1() { return ratio * M2; }

  // ---- Bragg (diatomic) ----
  function braggOmega(qv, b) {
    var s = 1 / m1() + 1 / M2;
    var disc = Math.max(0, s * s - 4 * Math.sin(qv / 2) * Math.sin(qv / 2) / (m1() * M2));
    return Math.sqrt(Math.max(0, K * s + (b === 1 ? 1 : -1) * K * Math.sqrt(disc)));
  }
  function braggGap() { return [Math.sqrt(2 * K / m1()), Math.sqrt(2 * K / M2)]; }
  function braggKi(w) {            // ki*a inside the gap (kr = pi)
    var O1 = 2 * K / m1(), O2 = 2 * K / M2;
    var arg = 1 - (w * w / O1 - 1) * (w * w / O2 - 1);
    return arg >= 1 ? Math.acosh(arg) : 0;
  }
  function braggAmpRatio(qv, b) {  // B/A (light over heavy), complex
    var w2 = braggOmega(qv, b); w2 = w2 * w2;
    var n1 = 2 * K - M2 * w2;
    if (Math.abs(n1) > 1e-6) {
      return { re: K * (1 + Math.cos(qv)) / n1, im: K * Math.sin(qv) / n1 };
    }
    var d2re = K * (1 + Math.cos(qv)), d2im = -K * Math.sin(qv);
    var m = d2re * d2re + d2im * d2im;
    if (m < 1e-12) return { re: 0, im: 0 };
    var n2 = 2 * K - m1() * w2;
    return { re: n2 * d2re / m, im: -n2 * d2im / m };
  }

  // ---- Local resonance (mass-in-mass) ----
  // M w^4 - w^2((M+mr) wr^2 + S) + S wr^2 = 0 with S = 2K(1 - cos qa), M = 1
  function resOmega(qv, b) {
    var S = 2 * K * (1 - Math.cos(qv));
    var B = (1 + MR) * wr * wr + S, C = S * wr * wr;
    var disc = Math.max(0, B * B - 4 * C);
    var w2 = 0.5 * (B + (b === 1 ? 1 : -1) * Math.sqrt(disc));
    return Math.sqrt(Math.max(0, w2));
  }
  function resGap() { return [resOmega(Math.PI, 0), wr * Math.sqrt(1 + MR)]; }
  function resRHS(w) {             // cos(ka) as a function of omega
    var meff = 1 + MR * wr * wr / (wr * wr - w * w);
    return 1 - w * w * meff / (2 * K);
  }
  function resKi(w) {              // [kr (0 or pi), ki*a]
    var c = resRHS(w);
    if (c < -1) return [Math.PI, Math.acosh(-c)];
    if (c > 1) return [0, Math.acosh(c)];
    return [Math.acos(Math.max(-1, Math.min(1, c))), 0];
  }
  function resAmpRatio(w) {        // internal / outer amplitude (real)
    var d = wr * wr - w * w;
    if (Math.abs(d) < 1e-4) d = d < 0 ? -1e-4 : 1e-4;
    return Math.max(-3, Math.min(3, wr * wr / d));
  }

  // ---- model-agnostic accessors ----
  function omega(qv, b) { return mech === "bragg" ? braggOmega(qv, b) : resOmega(qv, b); }
  function gap() { return mech === "bragg" ? braggGap() : resGap(); }
  function omegaTop() { return omega(mech === "bragg" ? 0.0 : Math.PI, 1) * (mech === "bragg" ? 1 : 1); }
  function maxPlotW() {
    var w = mech === "bragg" ? Math.sqrt(2 * K * (1 / m1() + 1 / M2)) : resOmega(Math.PI, 1);
    return w * 1.06;
  }

  // ---- canvases ----
  function sizeCanvas(c, ctx, w, h) {
    c.style.width = w + "px"; c.style.height = h + "px";
    c.width = w * DPR; c.height = h * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  function layout() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    var w = area.clientWidth || 760;
    DW = Math.max(320, Math.min(480, w * 0.62));
    CW = Math.max(300, Math.min(820, w - 16));
    sizeCanvas(dispC, dctx, DW, DH);
    sizeCanvas(chainC, cctx, CW, CH);
    drawDisp();
  }

  // plot: x in [-0.55, 1] -> Im(k) on the left of 0, Re(k)/pi on the right
  var PAD = { l: 40, r: 12, t: 14, b: 32 }, IMW = 0.55;
  function uToX(u) { return PAD.l + (u + IMW) / (1 + IMW) * (DW - PAD.l - PAD.r); }
  function wToY(w) { return DH - PAD.b - (w / maxPlotW()) * (DH - PAD.t - PAD.b); }
  function xToU(x) { return (x - PAD.l) / (DW - PAD.l - PAD.r) * (1 + IMW) - IMW; }

  function drawDisp() {
    dctx.clearRect(0, 0, DW, DH);
    dctx.fillStyle = "#fff"; dctx.fillRect(0, 0, DW, DH);
    var g = gap(), i, qq, x, y;
    var x0 = uToX(0);

    // Im region background
    dctx.fillStyle = "rgba(240,243,248,0.9)";
    dctx.fillRect(PAD.l, PAD.t, x0 - PAD.l, DH - PAD.t - PAD.b);

    // gap shading
    if (g[1] > g[0] + 1e-4) {
      dctx.fillStyle = "rgba(150,160,180,0.18)";
      dctx.fillRect(PAD.l, wToY(g[1]), DW - PAD.l - PAD.r, wToY(g[0]) - wToY(g[1]));
    }

    // axes
    dctx.strokeStyle = "#c8cdd6"; dctx.lineWidth = 1;
    dctx.beginPath();
    dctx.moveTo(PAD.l, PAD.t); dctx.lineTo(PAD.l, DH - PAD.b); dctx.lineTo(DW - PAD.r, DH - PAD.b);
    dctx.moveTo(x0, PAD.t); dctx.lineTo(x0, DH - PAD.b);
    dctx.stroke();
    dctx.fillStyle = "#8a93a3"; dctx.font = "11px sans-serif";
    dctx.fillText("ω", 8, PAD.t + 8);
    dctx.fillText("Im(k)·a/π", PAD.l + 4, DH - PAD.b + 14);
    dctx.fillText("Re(k)·a/π", DW - PAD.r - 58, DH - PAD.b + 14);
    dctx.fillText("0", x0 - 3, DH - PAD.b + 14);

    // real branches
    var cols = ["#1c54e5", "#f57c00"];
    for (var b = 0; b < 2; b++) {
      dctx.strokeStyle = cols[b]; dctx.lineWidth = 2.2;
      dctx.beginPath();
      var started = false;
      for (i = 0; i <= 160; i++) {
        qq = (i / 160) * Math.PI;
        var w = omega(qq, b);
        if (!isFinite(w)) continue;
        x = uToX(qq / Math.PI); y = wToY(w);
        if (!started) { dctx.moveTo(x, y); started = true; } else dctx.lineTo(x, y);
      }
      dctx.stroke();
    }

    // complex band structure inside the gap (dashed, dark red)
    dctx.strokeStyle = "#c0392b"; dctx.lineWidth = 1.8; dctx.setLineDash([5, 4]);
    dctx.beginPath();
    var startedIm = false;
    for (i = 1; i < 120; i++) {
      var wg = g[0] + (g[1] - g[0]) * i / 120;
      var ki = mech === "bragg" ? braggKi(wg) : resKi(wg)[1];
      if (ki <= 0) { startedIm = false; continue; }
      x = uToX(-Math.min(IMW, ki / Math.PI)); y = wToY(wg);
      if (!startedIm) { dctx.moveTo(x, y); startedIm = true; } else dctx.lineTo(x, y);
    }
    dctx.stroke();
    dctx.setLineDash([]);

    // gap label
    if (g[1] - g[0] > 0.02) {
      dctx.fillStyle = "#6a7385"; dctx.font = "italic 11px sans-serif";
      dctx.fillText("band gap", x0 + 8, (wToY(g[0]) + wToY(g[1])) / 2 + 4);
    }

    // marker
    var mx = uToX(q / Math.PI), my = wToY(omega(q, branch));
    dctx.beginPath(); dctx.arc(mx, my, 7, 0, 6.2832);
    dctx.fillStyle = cols[branch]; dctx.fill();
    dctx.strokeStyle = "#fff"; dctx.lineWidth = 2; dctx.stroke();

    // readout
    var label = document.getElementById("phonon-readout");
    if (label) {
      var wv = omega(q, branch);
      var bEn = branch === 0 ? (mech === "bragg" ? "acoustic" : "lower") : (mech === "bragg" ? "optical" : "upper");
      var bFr = branch === 0 ? (mech === "bragg" ? "acoustique" : "basse") : (mech === "bragg" ? "optique" : "haute");
      var tail = " &nbsp; q·a/π = " + (q / Math.PI).toFixed(2) + " &nbsp; ω = " + wv.toFixed(3) +
                 " &nbsp; gap: [" + g[0].toFixed(2) + ", " + g[1].toFixed(2) + "]";
      label.innerHTML = '<span class="lang-en">' + bEn + " branch" + tail + "</span>" +
                        '<span class="lang-fr">branche ' + bFr + tail.replace("gap:", "gap :") + "</span>";
    }
  }

  // ---- chain animation ----
  var NCELLS = 12;
  function drawChain() {
    cctx.clearRect(0, 0, CW, CH);
    cctx.fillStyle = "#fff"; cctx.fillRect(0, 0, CW, CH);
    var w = omega(q, branch);
    var cell = CW / (NCELLS + 1);
    var y = CH / 2, n;

    if (mech === "bragg") {
      var BA = braggAmpRatio(q, branch);
      var amp = cell * 0.22;
      var r1 = 11 * Math.cbrt(m1() / 2), r2 = 11 * Math.cbrt(M2 / 2);
      var xs = [], rs = [];
      for (n = 0; n < NCELLS; n++) {
        var ph = q * n - w * t;
        xs.push(cell * (0.8 + n) + amp * Math.cos(ph)); rs.push(r1);
        var ph2 = q * (n + 0.5) - w * t;
        xs.push(cell * (1.3 + n) + amp * (BA.re * Math.cos(ph2) - BA.im * Math.sin(ph2))); rs.push(r2);
      }
      cctx.strokeStyle = "#b9c2d0"; cctx.lineWidth = 2;
      cctx.beginPath();
      for (n = 0; n < xs.length - 1; n++) { cctx.moveTo(xs[n], y); cctx.lineTo(xs[n + 1], y); }
      cctx.stroke();
      for (n = 0; n < xs.length; n++) {
        cctx.beginPath(); cctx.arc(xs[n], y, rs[n], 0, 6.2832);
        cctx.fillStyle = n % 2 === 0 ? "#1c54e5" : "#f57c00";
        cctx.fill(); cctx.strokeStyle = "#fff"; cctx.lineWidth = 1.5; cctx.stroke();
      }
    } else {
      // mass-in-mass: outer ring + internal resonator
      var ampO = cell * 0.16;
      var rOut = 15, rIn = 7;
      var ratioIn = resAmpRatio(w);
      var xsO = [];
      for (n = 0; n < NCELLS; n++) {
        var phn = q * n - w * t;
        xsO.push(cell * (1 + n) + ampO * Math.cos(phn));
      }
      cctx.strokeStyle = "#b9c2d0"; cctx.lineWidth = 2;
      cctx.beginPath();
      for (n = 0; n < xsO.length - 1; n++) { cctx.moveTo(xsO[n], y); cctx.lineTo(xsO[n + 1], y); }
      cctx.stroke();
      for (n = 0; n < NCELLS; n++) {
        var phn2 = q * n - w * t;
        var xi = xsO[n] + ampO * (ratioIn - 1) * Math.cos(phn2) * 0.5;
        cctx.beginPath(); cctx.arc(xsO[n], y, rOut, 0, 6.2832);
        cctx.fillStyle = "rgba(28,84,229,0.18)"; cctx.fill();
        cctx.strokeStyle = "#1c54e5"; cctx.lineWidth = 2; cctx.stroke();
        cctx.beginPath(); cctx.moveTo(xsO[n] - rOut + 3, y); cctx.lineTo(xi, y); cctx.stroke();
        cctx.beginPath(); cctx.arc(xi, y, rIn, 0, 6.2832);
        cctx.fillStyle = "#f57c00"; cctx.fill();
        cctx.strokeStyle = "#fff"; cctx.lineWidth = 1.5; cctx.stroke();
      }
    }
  }

  function tick(ts) {
    if (!running) return;
    if (last) t += Math.min(0.05, (ts - last) / 1000) * 2.2;
    last = ts;
    drawChain();
    requestAnimationFrame(tick);
  }

  // ---- interactions ----
  function pickMode(e) {
    var rect = dispC.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    var u = xToU(src.clientX - rect.left), yy = src.clientY - rect.top;
    if (u < 0) return;                          // Im region is display-only
    var qv = Math.max(0.001, Math.min(Math.PI, u * Math.PI));
    var d0 = Math.abs(wToY(omega(qv, 0)) - yy);
    var d1 = Math.abs(wToY(omega(qv, 1)) - yy);
    q = qv; branch = d0 <= d1 ? 0 : 1;
    drawDisp();
  }
  var dragging = false;
  dispC.addEventListener("mousedown", function (e) { dragging = true; pickMode(e); });
  window.addEventListener("mousemove", function (e) { if (dragging) pickMode(e); });
  window.addEventListener("mouseup", function () { dragging = false; });
  dispC.addEventListener("touchstart", function (e) { e.preventDefault(); dragging = true; pickMode(e); }, { passive: false });
  dispC.addEventListener("touchmove", function (e) { e.preventDefault(); if (dragging) pickMode(e); }, { passive: false });
  dispC.addEventListener("touchend", function () { dragging = false; });

  function $(id) { return document.getElementById(id); }
  function refreshControls() {
    var br = $("phonon-bragg-controls"), re = $("phonon-res-controls");
    if (br) br.classList.toggle("d-none", mech !== "bragg");
    if (re) re.classList.toggle("d-none", mech !== "res");
    var bb = $("phonon-mech-bragg"), rb = $("phonon-mech-res");
    if (bb) bb.classList.toggle("active", mech === "bragg");
    if (rb) rb.classList.toggle("active", mech === "res");
  }
  if ($("phonon-mech-bragg")) $("phonon-mech-bragg").addEventListener("click", function () {
    mech = "bragg"; refreshControls(); drawDisp();
  });
  if ($("phonon-mech-res")) $("phonon-mech-res").addEventListener("click", function () {
    mech = "res"; refreshControls(); drawDisp();
  });
  if ($("phonon-ratio")) $("phonon-ratio").addEventListener("input", function () {
    ratio = parseFloat(this.value);
    if ($("phonon-ratio-val")) $("phonon-ratio-val").textContent = "m₁/m₂ = " + ratio.toFixed(2);
    drawDisp();
  });
  if ($("phonon-wr")) $("phonon-wr").addEventListener("input", function () {
    wr = parseFloat(this.value);
    if ($("phonon-wr-val")) $("phonon-wr-val").textContent = "ω᷊ = " + wr.toFixed(2);
    drawDisp();
  });
  if ($("phonon-mr")) $("phonon-mr").addEventListener("input", function () {
    MR = parseFloat(this.value);
    if ($("phonon-mr-val")) $("phonon-mr-val").textContent = "m᷊/M = " + MR.toFixed(2);
    drawDisp();
  });

  function show() {
    layout(); refreshControls();
    if (!running) { running = true; last = 0; requestAnimationFrame(tick); }
  }
  function hide() { running = false; }

  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(layout, 200); });
  window.__phononShow = show;
  window.__phononHide = hide;
})();
