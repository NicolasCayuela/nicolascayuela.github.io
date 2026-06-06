/*
 * Audio tools: real-time microphone spectrum analyzer (WebAudio FFT with a
 * scrolling spectrogram), tone & frequency-sweep generator, and an audio
 * converter (decode anything the browser can play, re-encode to WAV or MP3
 * via lamejs). Everything runs locally; the microphone is never recorded.
 */
(function () {
  "use strict";

  var ui = document.getElementById("au-spectrum-ui");
  if (!ui) return;

  var LAME_URL = "https://cdn.jsdelivr.net/npm/lamejs@1.2.1/lame.min.js";

  var statusEl = document.getElementById("au-status");
  var resultEl = document.getElementById("au-result");
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
  var PANELS = { spectrum: "au-spectrum-ui", tone: "au-tone-ui", convert: "au-convert-ui" };
  document.querySelectorAll("[data-au-tool]").forEach(function (b) {
    b.addEventListener("click", function () {
      var tool = b.getAttribute("data-au-tool");
      document.querySelectorAll("[data-au-tool]").forEach(function (x) { x.classList.toggle("active", x === b); });
      Object.keys(PANELS).forEach(function (k) {
        document.getElementById(PANELS[k]).classList.toggle("d-none", k !== tool);
      });
      setStatus("", ""); resultEl.innerHTML = "";
      if (tool !== "spectrum") stopSpectrum();
      if (tool !== "tone") stopTone();
    });
  });

  var actx = null;
  function audioCtx() {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") actx.resume();
    return actx;
  }

  // ---------- spectrum analyzer ----------
  var specStream = null, specRaf = null;
  var startBtn = document.getElementById("au-spec-start");
  var stopBtn = document.getElementById("au-spec-stop");
  var specCanvas = document.getElementById("au-spec-canvas");

  function stopSpectrum() {
    if (specRaf) cancelAnimationFrame(specRaf);
    specRaf = null;
    if (specStream) { specStream.getTracks().forEach(function (t) { t.stop(); }); specStream = null; }
    startBtn.classList.remove("d-none");
    stopBtn.classList.add("d-none");
  }

  startBtn.addEventListener("click", function () {
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } })
      .then(function (stream) {
        specStream = stream;
        var ctx = audioCtx();
        var src = ctx.createMediaStreamSource(stream);
        var an = ctx.createAnalyser();
        an.fftSize = 4096;
        an.smoothingTimeConstant = 0.6;
        src.connect(an);
        var bins = an.frequencyBinCount;
        var data = new Uint8Array(bins);
        var nyq = ctx.sampleRate / 2;
        var FMAX = 12000;
        var useBins = Math.floor(bins * FMAX / nyq);

        specCanvas.classList.remove("d-none");
        startBtn.classList.add("d-none");
        stopBtn.classList.remove("d-none");
        var W = specCanvas.clientWidth || 800;
        specCanvas.width = W; specCanvas.height = 360;
        var cx = specCanvas.getContext("2d");
        var SPEC_H = 120, GAP = 16, SG_Y = SPEC_H + GAP;
        var SG_H = specCanvas.height - SG_Y;
        cx.fillStyle = "#0b0e14"; cx.fillRect(0, 0, W, specCanvas.height);

        function hue(v) { return "hsl(" + (250 - v) + ",90%," + (12 + v * 0.18) + "%)"; }

        function frame() {
          an.getByteFrequencyData(data);
          // instant spectrum
          cx.fillStyle = "#0b0e14"; cx.fillRect(0, 0, W, SPEC_H + 14);
          var peakI = 0;
          for (var i = 0; i < useBins; i++) {
            if (data[i] > data[peakI]) peakI = i;
            var x = i * W / useBins;
            var h = data[i] / 255 * SPEC_H;
            cx.fillStyle = "#3ba3ff";
            cx.fillRect(x, SPEC_H - h, Math.max(1, W / useBins - 0.5), h);
          }
          var peakHz = Math.round(peakI * nyq / bins);
          if (data[peakI] > 30) {
            document.getElementById("au-spec-peak").textContent = "peak: " + peakHz + " Hz";
          }
          // axis labels
          cx.fillStyle = "#8fa3c0"; cx.font = "10px sans-serif";
          for (var f = 0; f <= FMAX; f += 2000) {
            cx.fillText((f / 1000) + "k", f * W / FMAX + 2, SPEC_H + 11);
          }
          // spectrogram: scroll left by 1px
          cx.drawImage(specCanvas, 1, SG_Y, W - 1, SG_H, 0, SG_Y, W - 1, SG_H);
          for (var j = 0; j < SG_H; j++) {
            var bi = Math.floor((1 - j / SG_H) * useBins);
            cx.fillStyle = hue(data[bi]);
            cx.fillRect(W - 1, SG_Y + j, 1, 1);
          }
          specRaf = requestAnimationFrame(frame);
        }
        frame();
      })
      .catch(function (e) {
        setStatus("Microphone refused: " + e.message, "Micro refusé : " + e.message);
      });
  });
  stopBtn.addEventListener("click", stopSpectrum);

  // ---------- tone / sweep generator ----------
  var osc = null, oscGain = null;
  var playBtn = document.getElementById("au-tone-play");

  function stopTone() {
    if (osc) {
      try { osc.stop(); } catch (e) {}
      osc.disconnect(); oscGain.disconnect();
      osc = null;
      playBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
  }
  function sliderToFreq(v) { return Math.round(Math.pow(10, 1 + v / 100)); }  // 10 Hz .. ~200 kHz capped below
  function freqToSlider(f) { return Math.round((Math.log10(f) - 1) * 100); }

  document.getElementById("au-tone-slider").addEventListener("input", function () {
    var f = Math.min(22000, sliderToFreq(+this.value));
    document.getElementById("au-tone-freq").value = f;
    if (osc) osc.frequency.setValueAtTime(f, audioCtx().currentTime);
  });
  document.getElementById("au-tone-freq").addEventListener("input", function () {
    var f = Math.max(1, Math.min(22000, +this.value || 440));
    document.getElementById("au-tone-slider").value = freqToSlider(f);
    if (osc) osc.frequency.setValueAtTime(f, audioCtx().currentTime);
  });
  document.getElementById("au-tone-gain").addEventListener("input", function () {
    if (oscGain) oscGain.gain.setValueAtTime(+this.value / 100 * 0.5, audioCtx().currentTime);
  });

  playBtn.addEventListener("click", function () {
    if (osc) { stopTone(); return; }
    var ctx = audioCtx();
    osc = ctx.createOscillator();
    oscGain = ctx.createGain();
    osc.type = document.getElementById("au-tone-wave").value;
    osc.frequency.value = +document.getElementById("au-tone-freq").value || 440;
    oscGain.gain.value = +document.getElementById("au-tone-gain").value / 100 * 0.5;
    osc.connect(oscGain); oscGain.connect(ctx.destination);
    osc.start();
    playBtn.innerHTML = '<i class="fas fa-stop"></i>';
  });

  document.getElementById("au-sweep-play").addEventListener("click", function () {
    stopTone();
    var ctx = audioCtx();
    var f0 = Math.max(1, +document.getElementById("au-sweep-f0").value || 20);
    var f1 = Math.max(1, Math.min(22000, +document.getElementById("au-sweep-f1").value || 20000));
    var T = Math.max(1, Math.min(60, +document.getElementById("au-sweep-t").value || 5));
    var log = document.getElementById("au-sweep-mode").value === "log";
    osc = ctx.createOscillator();
    oscGain = ctx.createGain();
    osc.type = document.getElementById("au-tone-wave").value;
    oscGain.gain.value = +document.getElementById("au-tone-gain").value / 100 * 0.5;
    var t = ctx.currentTime;
    osc.frequency.setValueAtTime(f0, t);
    if (log) osc.frequency.exponentialRampToValueAtTime(f1, t + T);
    else osc.frequency.linearRampToValueAtTime(f1, t + T);
    osc.connect(oscGain); oscGain.connect(ctx.destination);
    osc.start(t); osc.stop(t + T);
    osc.onended = stopTone;
    playBtn.innerHTML = '<i class="fas fa-stop"></i>';
  });

  // ---------- audio converter ----------
  var auFile = null;
  var auDrop = document.getElementById("au-drop");
  var auInput = document.getElementById("au-file");
  var convBtn = document.getElementById("au-conv-run");

  function takeAudio(f) {
    if (!f) return;
    auFile = f;
    convBtn.disabled = false;
    setStatus(f.name + " (" + Math.round(f.size / 1024) + " KB)", f.name + " (" + Math.round(f.size / 1024) + " Ko)");
  }
  auDrop.addEventListener("click", function () { auInput.click(); });
  auInput.addEventListener("change", function () { takeAudio(this.files[0]); this.value = ""; });
  auDrop.addEventListener("dragover", function (e) { e.preventDefault(); auDrop.style.borderColor = "#1c54e5"; });
  auDrop.addEventListener("dragleave", function () { auDrop.style.borderColor = "#cdd9f3"; });
  auDrop.addEventListener("drop", function (e) {
    e.preventDefault(); auDrop.style.borderColor = "#cdd9f3";
    takeAudio(e.dataTransfer.files[0]);
  });
  document.getElementById("au-out-fmt").addEventListener("change", function () {
    document.getElementById("au-opt-kbps").classList.toggle("d-none", this.value !== "mp3");
  });

  function encodeWav(buf) {
    var ch = buf.numberOfChannels, len = buf.length, sr = buf.sampleRate;
    var bytes = 44 + len * ch * 2;
    var ab = new ArrayBuffer(bytes), dv = new DataView(ab);
    function ws(o, s) { for (var i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); }
    ws(0, "RIFF"); dv.setUint32(4, bytes - 8, true); ws(8, "WAVE");
    ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
    dv.setUint16(22, ch, true); dv.setUint32(24, sr, true);
    dv.setUint32(28, sr * ch * 2, true); dv.setUint16(32, ch * 2, true); dv.setUint16(34, 16, true);
    ws(36, "data"); dv.setUint32(40, len * ch * 2, true);
    var o = 44;
    var chans = [];
    for (var c = 0; c < ch; c++) chans.push(buf.getChannelData(c));
    for (var i = 0; i < len; i++) {
      for (var c2 = 0; c2 < ch; c2++) {
        var v = Math.max(-1, Math.min(1, chans[c2][i]));
        dv.setInt16(o, v < 0 ? v * 32768 : v * 32767, true);
        o += 2;
      }
    }
    return new Blob([ab], { type: "audio/wav" });
  }

  function floatTo16(f32) {
    var out = new Int16Array(f32.length);
    for (var i = 0; i < f32.length; i++) {
      var v = Math.max(-1, Math.min(1, f32[i]));
      out[i] = v < 0 ? v * 32768 : v * 32767;
    }
    return out;
  }

  async function encodeMp3(buf, kbps) {
    await loadScript(LAME_URL);
    var ch = Math.min(2, buf.numberOfChannels);
    var enc = new lamejs.Mp3Encoder(ch, buf.sampleRate, kbps);
    var L = floatTo16(buf.getChannelData(0));
    var R = ch === 2 ? floatTo16(buf.getChannelData(1)) : null;
    var BLOCK = 1152, parts = [];
    for (var i = 0; i < L.length; i += BLOCK) {
      var l = L.subarray(i, i + BLOCK);
      var d = ch === 2 ? enc.encodeBuffer(l, R.subarray(i, i + BLOCK)) : enc.encodeBuffer(l);
      if (d.length) parts.push(new Uint8Array(d));
    }
    var end = enc.flush();
    if (end.length) parts.push(new Uint8Array(end));
    return new Blob(parts, { type: "audio/mpeg" });
  }

  convBtn.addEventListener("click", async function () {
    if (!auFile) return;
    convBtn.disabled = true;
    resultEl.innerHTML = "";
    try {
      setStatus("Decoding…", "Décodage…");
      var buf = await audioCtx().decodeAudioData(await auFile.arrayBuffer());
      var fmt = document.getElementById("au-out-fmt").value;
      setStatus("Encoding " + fmt.toUpperCase() + "…", "Encodage " + fmt.toUpperCase() + "…");
      var blob = fmt === "wav" ? encodeWav(buf)
                               : await encodeMp3(buf, parseInt(document.getElementById("au-kbps").value, 10));
      var name = auFile.name.replace(/\.[^.]+$/, "") + "." + fmt;
      var url = URL.createObjectURL(blob);
      resultEl.innerHTML = '<a class="btn btn-sm btn-outline-dark" download="' + name + '" href="' + url + '">' +
        '<i class="fas fa-download"></i> ' + name + " (" + (blob.size / 1048576).toFixed(2) + " MB)</a>";
      setStatus("Done.", "Terminé.");
    } catch (e) {
      setStatus("Error: " + e.message, "Erreur : " + e.message);
    }
    convBtn.disabled = false;
  });
})();
