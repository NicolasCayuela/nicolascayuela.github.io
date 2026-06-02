/*
 * CIFAR-10 dataset explorer (inspired by the MNIST latent-space demo).
 * Right: a 2D map where each real CIFAR-10 thumbnail is a dot coloured by its
 * class. Drag the cursor around the map; the left panel shows the nearest image.
 */
(function () {
  "use strict";

  var mapC = document.getElementById("cifar-map");
  var imgC = document.getElementById("cifar-img");
  if (!mapC || !imgC) return;
  var mctx = mapC.getContext("2d");
  var ictx = imgC.getContext("2d");

  var COLORS = ["#e6194b", "#3cb44b", "#ff1aff", "#4363d8", "#42d4f4",
                "#0000ff", "#f58231", "#3cf0a0", "#911eb4", "#bfef45"];

  var data = null, sprite = null, ready = false;
  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var SIZE = 360;                  // map logical size (square)
  var cursor = { x: 0.5, y: 0.5 };
  var nearest = 0;

  var base = mapC.getAttribute("data-json");
  var spriteUrl = mapC.getAttribute("data-sprite");

  fetch(base).then(function (r) { return r.json(); }).then(function (j) {
    data = j;
    sprite = new Image();
    sprite.onload = function () { ready = true; pickNearest(); render(); drawImg(); };
    sprite.src = spriteUrl;
  });

  function layoutSize() {
    var w = mapC.parentNode.clientWidth || 360;
    SIZE = Math.max(220, Math.min(360, w));
  }

  function resize() {
    if (!ready) { layoutSize(); sizeCanvas(); return; }
    layoutSize(); sizeCanvas(); render(); drawImg();
  }
  function sizeCanvas() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    mapC.style.width = SIZE + "px"; mapC.style.height = SIZE + "px";
    mapC.width = SIZE * DPR; mapC.height = SIZE * DPR;
    mctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function pickNearest() {
    if (!data) return;
    var best = 1e9, bi = 0, it = data.items;
    for (var i = 0; i < it.length; i++) {
      var dx = it[i].x - cursor.x, dy = it[i].y - cursor.y;
      var d = dx * dx + dy * dy;
      if (d < best) { best = d; bi = i; }
    }
    nearest = bi;
  }

  function render() {
    if (!ready) return;
    mctx.clearRect(0, 0, SIZE, SIZE);
    mctx.fillStyle = "#fafafa"; mctx.fillRect(0, 0, SIZE, SIZE);
    var it = data.items, i;
    for (i = 0; i < it.length; i++) {
      mctx.beginPath();
      mctx.arc(it[i].x * SIZE, it[i].y * SIZE, i === nearest ? 6 : 3.4, 0, 6.2832);
      mctx.fillStyle = COLORS[it[i].c];
      mctx.fill();
      if (i === nearest) { mctx.strokeStyle = "#000"; mctx.lineWidth = 1.5; mctx.stroke(); }
    }
    // cursor ring
    mctx.beginPath();
    mctx.arc(cursor.x * SIZE, cursor.y * SIZE, 11, 0, 6.2832);
    mctx.strokeStyle = "rgba(40,40,40,0.85)"; mctx.lineWidth = 3; mctx.stroke();
    mctx.fillStyle = "rgba(255,255,255,0.25)"; mctx.fill();
  }

  function drawImg() {
    if (!ready) return;
    var it = data.items[nearest], T = data.thumb;
    var L = imgC.width;            // device px (square)
    ictx.imageSmoothingEnabled = false;
    ictx.clearRect(0, 0, L, L);
    ictx.drawImage(sprite, it.g * T, it.r * T, T, T, 0, 0, L, L);
    var label = document.getElementById("cifar-label");
    if (label) label.textContent = data.classes[it.c];
  }

  // ---- pointer drag on the map ----
  var dragging = false;
  function setCursor(e) {
    var rect = mapC.getBoundingClientRect();
    var src = e.touches ? e.touches[0] : e;
    cursor.x = Math.max(0, Math.min(1, (src.clientX - rect.left) / rect.width));
    cursor.y = Math.max(0, Math.min(1, (src.clientY - rect.top) / rect.height));
    pickNearest(); render(); drawImg();
  }
  mapC.addEventListener("mousedown", function (e) { dragging = true; setCursor(e); });
  window.addEventListener("mousemove", function (e) { if (dragging) setCursor(e); });
  window.addEventListener("mouseup", function () { dragging = false; });
  mapC.addEventListener("touchstart", function (e) { e.preventDefault(); dragging = true; setCursor(e); }, { passive: false });
  mapC.addEventListener("touchmove", function (e) { e.preventDefault(); if (dragging) setCursor(e); }, { passive: false });
  mapC.addEventListener("touchend", function () { dragging = false; });

  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
  layoutSize(); sizeCanvas();
  window.__cifarResize = resize;
})();
