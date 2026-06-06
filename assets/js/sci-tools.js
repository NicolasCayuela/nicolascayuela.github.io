/*
 * Science & engineering tools: function plotter (function-plot), scientific
 * unit converter (incl. acoustic pressure / dB SPL), CIF crystal viewer
 * (3Dmol.js), matrix calculator (math.js) and LaTeX equation editor (MathJax
 * SVG with PNG/SVG export). Libraries are lazy-loaded from CDNs.
 */
(function () {
  "use strict";

  var card = document.getElementById("sc-plot-ui");
  if (!card) return;

  var FPLOT_URL = "https://cdn.jsdelivr.net/npm/function-plot@1.25.1/dist/function-plot.js";
  var MATHJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/mathjs/12.4.2/math.js";
  var MOL3D_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js";
  var MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";

  var statusEl = document.getElementById("sc-status");
  function setStatus(en, fr) {
    statusEl.innerHTML = en || fr
      ? '<span class="lang-en">' + en + "</span><span class=\"lang-fr\">" + (fr || en) + "</span>"
      : "";
  }

  var loaded = {};
  function loadScript(src) {
    if (!loaded[src]) {
      loaded[src] = new Promise(function (res, rej) {
        var s = document.createElement("script");
        s.src = src; s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
      });
    }
    return loaded[src];
  }

  // ---------- mode switch ----------
  var PANELS = { plot: "sc-plot-ui", units: "sc-units-ui", cif: "sc-cif-ui",
                 matrix: "sc-matrix-ui", latex: "sc-latex-ui" };
  document.querySelectorAll("[data-sc-tool]").forEach(function (b) {
    b.addEventListener("click", function () {
      var tool = b.getAttribute("data-sc-tool");
      document.querySelectorAll("[data-sc-tool]").forEach(function (x) { x.classList.toggle("active", x === b); });
      Object.keys(PANELS).forEach(function (k) {
        document.getElementById(PANELS[k]).classList.toggle("d-none", k !== tool);
      });
      setStatus("", "");
      if (tool === "plot") plot();
      if (tool === "units") initUnits();
      if (tool === "latex") renderLatex();
    });
  });

  // ---------- function plotter ----------
  function plot() {
    loadScript(FPLOT_URL).then(function () {
      var exprs = document.getElementById("sc-plot-fn").value.split(";")
        .map(function (s) { return s.trim(); }).filter(Boolean);
      var xmin = parseFloat(document.getElementById("sc-plot-xmin").value);
      var xmax = parseFloat(document.getElementById("sc-plot-xmax").value);
      if (!(xmax > xmin)) { xmin = -10; xmax = 10; }
      var host = document.getElementById("sc-plot-out");
      host.innerHTML = "";
      try {
        window.functionPlot({
          target: host,
          width: Math.min(820, host.clientWidth || 820),
          height: 420,
          grid: true,
          xAxis: { domain: [xmin, xmax] },
          data: exprs.map(function (fn) { return { fn: fn, sampler: "builtIn", graphType: "polyline" }; })
        });
        setStatus("", "");
      } catch (e) {
        setStatus("Cannot plot: " + e.message, "Tracé impossible : " + e.message);
      }
    });
  }
  document.getElementById("sc-plot-go").addEventListener("click", plot);
  document.getElementById("sc-plot-fn").addEventListener("keydown", function (e) {
    if (e.key === "Enter") plot();
  });

  // ---------- unit converter ----------
  // factor = value in the base unit of the category
  var UNITS = {
    "Length":      { base: "m", u: { "m": 1, "km": 1e3, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6, "nm": 1e-9, "Å": 1e-10, "in": 0.0254, "ft": 0.3048, "mile": 1609.344 } },
    "Mass":        { base: "kg", u: { "kg": 1, "g": 1e-3, "mg": 1e-6, "t": 1e3, "lb": 0.45359237, "oz": 0.028349523 } },
    "Time":        { base: "s", u: { "s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9, "min": 60, "h": 3600, "day": 86400 } },
    "Frequency":   { base: "Hz", u: { "Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12, "rad/s": 1 / (2 * Math.PI), "rpm": 1 / 60 } },
    "Pressure":    { base: "Pa", u: { "Pa": 1, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "mbar": 1e2, "atm": 101325, "psi": 6894.757, "mmHg": 133.322, "dB SPL": "dbspl" } },
    "Energy":      { base: "J", u: { "J": 1, "kJ": 1e3, "MJ": 1e6, "Wh": 3600, "kWh": 3.6e6, "eV": 1.602176634e-19, "cal": 4.184, "kcal": 4184 } },
    "Power":       { base: "W", u: { "W": 1, "mW": 1e-3, "kW": 1e3, "MW": 1e6, "hp": 745.699872, "dBm": "dbm" } },
    "Speed":       { base: "m/s", u: { "m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704, "knot": 0.514444, "c (light)": 299792458, "sound in air (343 m/s)": 343, "sound in steel (5960 m/s)": 5960 } },
    "Temperature": { base: "K", u: { "K": "k", "°C": "c", "°F": "f" } },
    "Angle":       { base: "rad", u: { "rad": 1, "deg": Math.PI / 180, "grad": Math.PI / 200, "turn": 2 * Math.PI } }
  };

  function toBase(cat, unit, v) {
    var f = UNITS[cat].u[unit];
    if (f === "dbspl") return 20e-6 * Math.pow(10, v / 20);          // dB SPL re 20 µPa
    if (f === "dbm") return 1e-3 * Math.pow(10, v / 10);             // dBm re 1 mW
    if (f === "k") return v;
    if (f === "c") return v + 273.15;
    if (f === "f") return (v - 32) * 5 / 9 + 273.15;
    return v * f;
  }
  function fromBase(cat, unit, b) {
    var f = UNITS[cat].u[unit];
    if (f === "dbspl") return 20 * Math.log10(Math.max(b, 1e-300) / 20e-6);
    if (f === "dbm") return 10 * Math.log10(Math.max(b, 1e-300) / 1e-3);
    if (f === "k") return b;
    if (f === "c") return b - 273.15;
    if (f === "f") return (b - 273.15) * 9 / 5 + 32;
    return b / f;
  }

  var unitsInit = false;
  function initUnits() {
    if (!unitsInit) {
      unitsInit = true;
      var catSel = document.getElementById("sc-units-cat");
      Object.keys(UNITS).forEach(function (c) {
        var o = document.createElement("option"); o.value = c; o.textContent = c;
        catSel.appendChild(o);
      });
      catSel.addEventListener("change", fillUnits);
      ["sc-units-val", "sc-units-from", "sc-units-to"].forEach(function (id) {
        document.getElementById(id).addEventListener("input", convertUnits);
      });
      catSel.value = "Pressure";
      fillUnits();
    }
    convertUnits();
  }
  function fillUnits() {
    var cat = document.getElementById("sc-units-cat").value;
    ["sc-units-from", "sc-units-to"].forEach(function (id, k) {
      var sel = document.getElementById(id);
      sel.innerHTML = "";
      Object.keys(UNITS[cat].u).forEach(function (u) {
        var o = document.createElement("option"); o.value = u; o.textContent = u;
        sel.appendChild(o);
      });
      sel.selectedIndex = Math.min(k, sel.options.length - 1);
    });
    convertUnits();
  }
  function convertUnits() {
    var cat = document.getElementById("sc-units-cat").value;
    if (!cat) return;
    var v = parseFloat(document.getElementById("sc-units-val").value);
    var from = document.getElementById("sc-units-from").value;
    var to = document.getElementById("sc-units-to").value;
    if (!isFinite(v) || !from || !to) return;
    var r = fromBase(cat, to, toBase(cat, from, v));
    var txt = Math.abs(r) >= 1e6 || (Math.abs(r) < 1e-4 && r !== 0) ? r.toExponential(6) : +r.toPrecision(8);
    document.getElementById("sc-units-out").textContent = v + " " + from + " = " + txt + " " + to;
    document.getElementById("sc-units-note").innerHTML = cat === "Pressure"
      ? '<span class="lang-en">dB SPL referenced to 20 µPa (threshold of hearing in air).</span><span class="lang-fr">dB SPL référencé à 20 µPa (seuil d\'audition dans l\'air).</span>'
      : "";
  }

  // ---------- CIF viewer ----------
  var NACL_CIF = "data_NaCl\n_cell_length_a 5.6402\n_cell_length_b 5.6402\n_cell_length_c 5.6402\n" +
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n_symmetry_space_group_name_H-M 'F m -3 m'\n" +
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n" +
    "Na1 Na 0 0 0\nCl1 Cl 0.5 0.5 0.5\n";
  var cifViewer = null, cifData = null;

  function renderCif() {
    if (!cifData) return;
    loadScript(MOL3D_URL).then(function () {
      var host = document.getElementById("sc-cif-view");
      if (!cifViewer) cifViewer = window.$3Dmol.createViewer(host, { backgroundColor: "#ffffff" });
      cifViewer.removeAllModels();
      cifViewer.removeAllShapes();
      var rep = parseInt(document.getElementById("sc-cif-rep").value, 10);
      var m = cifViewer.addModel(cifData, "cif", { duplicateAssemblyAtoms: true, normalizeAssembly: true });
      var style = document.getElementById("sc-cif-style").value;
      if (style === "sphere") cifViewer.setStyle({}, { sphere: { scale: 0.45 } });
      else if (style === "line") cifViewer.setStyle({}, { line: {} });
      else cifViewer.setStyle({}, { sphere: { scale: 0.25 }, stick: { radius: 0.12 } });
      if (document.getElementById("sc-cif-cell").checked) {
        cifViewer.addUnitCell(m, { box: { color: "#1c54e5" } });
      }
      if (rep > 1) cifViewer.replicateUnitCell(rep, rep, rep, m);
      cifViewer.zoomTo();
      cifViewer.render();
      setStatus("", "");
    }).catch(function (e) { setStatus("3Dmol failed: " + e.message, "Échec 3Dmol : " + e.message); });
  }
  document.getElementById("sc-cif-sample").addEventListener("click", function () {
    cifData = NACL_CIF; renderCif();
  });
  document.getElementById("sc-cif-file").addEventListener("change", function () {
    var f = this.files[0];
    if (!f) return;
    f.text().then(function (t) { cifData = t; renderCif(); });
  });
  ["sc-cif-style", "sc-cif-cell", "sc-cif-rep"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", renderCif);
  });

  // ---------- matrix calculator ----------
  function parseMat(id) {
    var t = document.getElementById(id).value.trim();
    if (!t) return null;
    return t.split(/\n|;/).map(function (row) {
      return row.trim().split(/[\s,]+/).map(Number);
    }).filter(function (r) { return r.length && r.every(isFinite); });
  }
  function fmtNum(x) {
    if (x && typeof x === "object" && "re" in x) {       // complex
      var im = x.im >= 0 ? " + " + fmt(x.im) + "i" : " - " + fmt(-x.im) + "i";
      return fmt(x.re) + (Math.abs(x.im) > 1e-12 ? im : "");
    }
    return fmt(x);
    function fmt(v) { return Math.abs(v) < 1e-12 ? "0" : +v.toPrecision(6) + ""; }
  }
  function fmtMat(M) {
    var arr = M.toArray ? M.toArray() : M;
    if (!Array.isArray(arr)) return fmtNum(arr);
    if (!Array.isArray(arr[0])) return "[ " + arr.map(fmtNum).join(", ") + " ]";
    return arr.map(function (r) { return "[ " + r.map(fmtNum).join("  ") + " ]"; }).join("\n");
  }
  document.querySelectorAll("[data-sc-mat]").forEach(function (b) {
    b.addEventListener("click", function () {
      var op = b.getAttribute("data-sc-mat");
      var out = document.getElementById("sc-mat-out");
      loadScript(MATHJS_URL).then(function () {
        try {
          var A = parseMat("sc-mat-a"), B = parseMat("sc-mat-b");
          if (!A) throw new Error("matrix A is empty / matrice A vide");
          var r;
          if (op === "eigs") {
            var e = window.math.eigs(A);
            var vals = (e.values.toArray ? e.values.toArray() : e.values);
            r = "eigenvalues:\n" + vals.map(fmtNum).join("\n");
            if (e.eigenvectors) {
              r += "\n\neigenvectors (columns):\n" + fmtMat(e.eigenvectors.map(function (ev) { return ev.vector.toArray ? ev.vector.toArray() : ev.vector; }));
            }
          }
          else if (op === "det") r = "det(A) = " + fmtNum(window.math.det(A));
          else if (op === "inv") r = fmtMat(window.math.inv(A));
          else if (op === "transpose") r = fmtMat(window.math.transpose(A));
          else if (op === "mul") {
            if (!B) throw new Error("matrix B required / matrice B requise");
            r = fmtMat(window.math.multiply(A, B));
          }
          else if (op === "solve") {
            if (!B) throw new Error("vector/matrix B required / vecteur ou matrice B requis");
            r = "x =\n" + fmtMat(window.math.lusolve(A, B));
          }
          out.textContent = r;
        } catch (e) {
          out.textContent = "Error / Erreur : " + e.message;
        }
      });
    });
  });

  // ---------- LaTeX equations ----------
  var mjReady = null;
  function needMathJax() {
    if (!mjReady) {
      window.MathJax = { startup: { typeset: false }, svg: { fontCache: "none" } };
      mjReady = loadScript(MATHJAX_URL).then(function () { return window.MathJax.startup.promise; });
    }
    return mjReady;
  }
  var ltTimer;
  function renderLatex() {
    clearTimeout(ltTimer);
    ltTimer = setTimeout(function () {
      needMathJax().then(function () {
        var tex = document.getElementById("sc-latex-in").value;
        var out = document.getElementById("sc-latex-out");
        try {
          var node = window.MathJax.tex2svg(tex, { display: true });
          out.innerHTML = "";
          out.appendChild(node);
        } catch (e) {
          out.textContent = "LaTeX error: " + e.message;
        }
      });
    }, 250);
  }
  document.getElementById("sc-latex-in").addEventListener("input", renderLatex);

  function latexSvg() {
    var svg = document.querySelector("#sc-latex-out svg");
    if (!svg) return null;
    var clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    return new XMLSerializer().serializeToString(clone);
  }
  document.getElementById("sc-latex-svg").addEventListener("click", function () {
    var s = latexSvg();
    if (!s) return;
    var a = document.createElement("a");
    a.download = "equation.svg";
    a.href = URL.createObjectURL(new Blob([s], { type: "image/svg+xml" }));
    a.click();
  });
  document.getElementById("sc-latex-png").addEventListener("click", function () {
    var s = latexSvg();
    if (!s) return;
    var img = new Image();
    img.onload = function () {
      var SCALE = 4;
      var c = document.createElement("canvas");
      c.width = img.width * SCALE; c.height = img.height * SCALE;
      var cx = c.getContext("2d");
      cx.fillStyle = "#ffffff"; cx.fillRect(0, 0, c.width, c.height);
      cx.drawImage(img, 0, 0, c.width, c.height);
      var a = document.createElement("a");
      a.download = "equation.png";
      a.href = c.toDataURL("image/png");
      a.click();
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(s);
  });
})();
