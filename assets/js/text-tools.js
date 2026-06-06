/*
 * Text & dev tools: text diff (jsdiff), JSON/YAML formatter-validator-converter
 * (js-yaml), and file hashing with WebCrypto (SHA-256 / SHA-1 / SHA-512).
 * Everything runs locally.
 */
(function () {
  "use strict";

  var ui = document.getElementById("tx-diff-ui");
  if (!ui) return;

  var DIFF_URL = "https://cdn.jsdelivr.net/npm/diff@5.2.0/dist/diff.min.js";
  var YAML_URL = "https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js";

  var statusEl = document.getElementById("tx-status");
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

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ---------- mode switch ----------
  var PANELS = { diff: "tx-diff-ui", json: "tx-json-ui", hash: "tx-hash-ui" };
  document.querySelectorAll("[data-tx-tool]").forEach(function (b) {
    b.addEventListener("click", function () {
      var tool = b.getAttribute("data-tx-tool");
      document.querySelectorAll("[data-tx-tool]").forEach(function (x) { x.classList.toggle("active", x === b); });
      Object.keys(PANELS).forEach(function (k) {
        document.getElementById(PANELS[k]).classList.toggle("d-none", k !== tool);
      });
      setStatus("", "");
    });
  });

  // ---------- diff ----------
  document.getElementById("tx-diff-go").addEventListener("click", function () {
    loadScript(DIFF_URL).then(function () {
      var a = document.getElementById("tx-diff-a").value;
      var b = document.getElementById("tx-diff-b").value;
      var mode = document.getElementById("tx-diff-mode").value;
      var parts = mode === "lines" ? Diff.diffLines(a, b)
                : mode === "chars" ? Diff.diffChars(a, b)
                : Diff.diffWords(a, b);
      var html = parts.map(function (p) {
        var t = esc(p.value);
        if (p.added) return '<span style="background:#d3f9d8; color:#0b6b2d;">' + t + "</span>";
        if (p.removed) return '<span style="background:#ffe3e3; color:#a61e1e; text-decoration:line-through;">' + t + "</span>";
        return t;
      }).join("");
      var added = 0, removed = 0;
      parts.forEach(function (p) { if (p.added) added++; if (p.removed) removed++; });
      document.getElementById("tx-diff-out").innerHTML = html ||
        '<span class="text-muted">(empty)</span>';
      setStatus(added + " additions, " + removed + " deletions", added + " ajouts, " + removed + " suppressions");
    });
  });

  // ---------- JSON / YAML ----------
  document.querySelectorAll("[data-tx-json]").forEach(function (b) {
    b.addEventListener("click", function () {
      var op = b.getAttribute("data-tx-json");
      var inEl = document.getElementById("tx-json-in");
      var out = document.getElementById("tx-json-out");
      loadScript(YAML_URL).then(function () {
        var src = inEl.value;
        try {
          var r;
          if (op === "fmt") r = JSON.stringify(JSON.parse(src), null, 2);
          else if (op === "min") r = JSON.stringify(JSON.parse(src));
          else if (op === "toyaml") r = window.jsyaml.dump(JSON.parse(src), { lineWidth: 100 });
          else if (op === "tojson") r = JSON.stringify(window.jsyaml.load(src), null, 2);
          else if (op === "validate") {
            JSON.parse(src);
            r = "✓ valid JSON";
          }
          out.textContent = r;
          setStatus("", "");
        } catch (e) {
          if (op === "validate") {
            // not JSON; try YAML before declaring failure
            try { window.jsyaml.load(src); out.textContent = "✓ valid YAML (not JSON)"; setStatus("", ""); return; }
            catch (e2) {}
          }
          out.textContent = "✗ " + e.message;
        }
      });
    });
  });

  // ---------- file hash ----------
  var hashDrop = document.getElementById("tx-hash-drop");
  var hashInput = document.getElementById("tx-hash-file");
  var hashOut = document.getElementById("tx-hash-out");
  var ALGOS = ["SHA-256", "SHA-1", "SHA-512"];

  function hex(buf) {
    return Array.prototype.map.call(new Uint8Array(buf), function (b) {
      return ("0" + b.toString(16)).slice(-2);
    }).join("");
  }
  async function hashFiles(list) {
    for (var i = 0; i < list.length; i++) {
      var f = list[i];
      setStatus("Hashing " + f.name + "…", "Calcul pour " + f.name + "…");
      var buf = await f.arrayBuffer();
      var rows = "";
      for (var a = 0; a < ALGOS.length; a++) {
        var d = await crypto.subtle.digest(ALGOS[a], buf);
        rows += "<tr><td class='pr-2 text-muted'>" + ALGOS[a] + "</td><td>" + hex(d) + "</td></tr>";
      }
      hashOut.innerHTML += '<div class="mb-2"><strong>' + esc(f.name) + "</strong> (" +
        Math.round(f.size / 1024) + ' KB)<table class="small">' + rows + "</table></div>";
    }
    setStatus("", "");
  }
  hashDrop.addEventListener("click", function () { hashInput.click(); });
  hashInput.addEventListener("change", function () { hashFiles(this.files); this.value = ""; });
  hashDrop.addEventListener("dragover", function (e) { e.preventDefault(); hashDrop.style.borderColor = "#1c54e5"; });
  hashDrop.addEventListener("dragleave", function () { hashDrop.style.borderColor = "#cdd9f3"; });
  hashDrop.addEventListener("drop", function (e) {
    e.preventDefault(); hashDrop.style.borderColor = "#cdd9f3";
    hashFiles(e.dataTransfer.files);
  });
})();
