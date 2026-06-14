/*
 * Elastic-wave metamaterial background - WebGL field simulation (v3-gl).
 *
 * A real 2D scalar wave equation solved on the GPU by finite differences
 * (FDTD), ping-ponged between two RG32F textures (R = u_t, G = u_{t-1}).
 * The medium has a spatially varying speed map, so genuine wave physics
 * emerges rather than being prescribed:
 *   - a rigid circular inclusion (speed -> 0) that REFLECTS waves: a real
 *     band-gap scatterer, not a faked calm zone;
 *   - a slower vertical slab that refracts and focuses the plane wave: a
 *     metamaterial lens.
 * A steady plane-wave line source on the left feeds fronts across the screen;
 * the pointer injects radial pulses. The field has memory, so wavefronts and
 * their trails are intrinsic. Rendered through the COMSOL "jet" colormap.
 *
 * Falls back to the canvas2d background (wave-particles.js) when WebGL2 /
 * float render targets are unavailable, on phones, or for reduced-motion.
 */
(function () {
  "use strict";

  function fallback() {
    var s = document.createElement("script");
    s.src = "assets/js/wave-particles.js?v=v3";
    document.head.appendChild(s);
  }

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var mobile = window.isMobileViewport ? window.isMobileViewport()
    : !!(window.matchMedia && window.matchMedia("(max-width: 820px), (pointer: coarse)").matches);
  if (reduce || mobile) { fallback(); return; }

  var canvas = document.createElement("canvas");
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.cssText =
    "position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;pointer-events:none;";
  var gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: false, antialias: false, depth: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) { fallback(); return; }
  (document.body || document.documentElement).appendChild(canvas);

  // ---- tunables -----------------------------------------------------------
  var CFG = {
    down: 2.4,          // sim grid = screen / down (smaller = faster, coarser)
    maxW: 560,          // hard cap on sim grid width
    substeps: 2,        // FDTD steps per rendered frame
    c2: 0.20,           // base wave speed^2 (CFL: keep < 0.5)
    freq: 0.026,        // source temporal frequency (cycles per step)
    planeAmp: 0.055,    // steady plane-wave source strength
    pulseAmp: 0.9,      // pointer / auto radial pulse strength
    damp: 0.0009,       // tiny bulk loss
    gain: 5.5,          // field -> colormap gain
    opacity: 0.6,       // canvas opacity (light theme)
    opacityDark: 0.9
  };

  var GW = 0, GH = 0;                 // sim grid size
  var srcs = [];                      // radial pulses {x,y,age,amp}
  var planeAge = 0;

  // ---- GL helpers ---------------------------------------------------------
  function sh(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("wave-field shader:", gl.getShaderInfoLog(s)); return null;
    }
    return s;
  }
  function prog(vs, fs) {
    var v = sh(gl.VERTEX_SHADER, vs), f = sh(gl.FRAGMENT_SHADER, fs);
    if (!v || !f) return null;
    var p = gl.createProgram();
    gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.error("wave-field link:", gl.getProgramInfoLog(p)); return null;
    }
    return p;
  }

  var VS =
    "#version 300 es\n" +
    "in vec2 p; out vec2 uv;\n" +
    "void main(){ uv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }\n";

  // medium speed^2 map + the FDTD update, shared snippet for the sim shader
  var SIM_FS =
    "#version 300 es\n" +
    "precision highp float;\n" +
    "in vec2 uv; out vec2 outState;\n" +
    "uniform sampler2D st; uniform vec2 texel; uniform float aspect;\n" +
    "uniform float c2, damp, freq, planeAmp, planeAge;\n" +
    "uniform int nsrc; uniform vec4 srcs[8];\n" +   // xy, age, amp
    "uniform vec2 gapC; uniform float gapR;\n" +
    "uniform vec2 lensX;\n" +                        // slab x-range (lo,hi)
    "const float PI=3.14159265;\n" +
    // speed^2 at a point: rigid disc -> 0 (reflects), slab -> slower
    "float medium(vec2 q){\n" +
    "  vec2 d=(q-gapC); d.x*=aspect;\n" +
    "  if(length(d)<gapR) return 0.0;\n" +
    "  float m=c2;\n" +
    "  if(q.x>lensX.x && q.x<lensX.y) m*=0.34;\n" +
    "  return m;\n" +
    "}\n" +
    // absorbing sponge near the borders, near-lossless inside
    "float edgeDamp(vec2 q){\n" +
    "  float e=smoothstep(0.0,0.07,q.x)*smoothstep(0.0,0.07,1.0-q.x)\n" +
    "         *smoothstep(0.0,0.07,q.y)*smoothstep(0.0,0.07,1.0-q.y);\n" +
    "  return mix(0.16, damp, e);\n" +
    "}\n" +
    "void main(){\n" +
    "  vec2 s=texture(st,uv).rg; float uC=s.r, uP=s.g;\n" +
    "  float l=texture(st,uv-vec2(texel.x,0.)).r;\n" +
    "  float r=texture(st,uv+vec2(texel.x,0.)).r;\n" +
    "  float d=texture(st,uv-vec2(0.,texel.y)).r;\n" +
    "  float u=texture(st,uv+vec2(0.,texel.y)).r;\n" +
    "  float lap=l+r+u+d-4.0*uC;\n" +
    "  float m=medium(uv);\n" +
    "  float un=2.0*uC-uP+m*lap;\n" +
    "  un-=edgeDamp(uv)*(uC-uP);\n" +
    "  if(m<=0.0){ outState=vec2(0.0,0.0); return; }\n" +   // rigid inclusion
    // steady plane wave entering from the left
    "  un+=planeAmp*exp(-pow((uv.x-0.04)/0.012,2.0))*sin(2.0*PI*freq*planeAge);\n" +
    // radial pulses (pointer / auto)
    "  for(int i=0;i<8;i++){ if(i>=nsrc) break;\n" +
    "    vec4 sp=srcs[i]; vec2 dd=uv-sp.xy; dd.x*=aspect;\n" +
    "    float g=exp(-dot(dd,dd)/0.0009);\n" +
    "    float env=exp(-pow((sp.z-6.0)/9.0,2.0));\n" +
    "    un+=sp.w*g*sin(2.0*PI*freq*sp.z)*env;\n" +
    "  }\n" +
    "  outState=vec2(un,uC);\n" +
    "}\n";

  var REND_FS =
    "#version 300 es\n" +
    "precision highp float;\n" +
    "in vec2 uv; out vec4 frag;\n" +
    "uniform sampler2D st; uniform float gain, dark; uniform vec2 res;\n" +
    "uniform vec2 gapC; uniform float gapR, aspect;\n" +
    "vec3 jet(float t){ t=clamp(t,0.,1.);\n" +
    "  float r=clamp(min(4.*t-1.5,-4.*t+4.5),0.,1.);\n" +
    "  float g=clamp(min(4.*t-0.5,-4.*t+3.5),0.,1.);\n" +
    "  float b=clamp(min(4.*t+0.5,-4.*t+2.5),0.,1.);\n" +
    "  return vec3(r,g,b); }\n" +
    "void main(){\n" +
    "  float v=texture(st,uv).r;\n" +
    "  float t=clamp(abs(v)*gain,0.0,1.0);\n" +
    "  float floorT = dark>0.5 ? 0.10 : 0.0;\n" +
    "  float hue = floorT + t*(0.82-floorT);\n" +
    "  vec3 col=jet(hue);\n" +
    "  if(dark>0.5){ float f=0.7*(1.0-t); col+=(vec3(1.0)-col)*f; }\n" +
    // faint hex-ish lattice modulation to evoke discrete unit cells
    "  vec2 q=uv*res/26.0; float lat=0.5+0.5*cos(6.2831*q.x)*cos(6.2831*q.y);\n" +
    "  col*=0.92+0.08*lat;\n" +
    // mark the rigid inclusion with a faint resonator ring
    "  vec2 dd=uv-gapC; dd.x*=aspect; float rr=length(dd);\n" +
    "  float ring=smoothstep(0.004,0.0,abs(rr-gapR));\n" +
    "  vec3 steel = dark>0.5 ? vec3(0.6,0.68,0.82) : vec3(0.34,0.42,0.6);\n" +
    "  float baseA = dark>0.5 ? 0.7 : 0.22;\n" +
    "  float a = clamp(baseA + t*(0.96-baseA), 0.0, 1.0);\n" +
    "  col=mix(col,steel,ring*0.6); a=max(a,ring*0.5);\n" +
    "  frag=vec4(col*a, a);\n" +     // straight-alpha over the page
    "}\n";

  var simP = prog(VS, SIM_FS), rendP = prog(VS, REND_FS);
  if (!simP || !rendP) { canvas.remove(); fallback(); return; }

  // full-screen triangle
  var vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  var vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  var ploc = gl.getAttribLocation(simP, "p");
  gl.enableVertexAttribArray(ploc); gl.vertexAttribPointer(ploc, 2, gl.FLOAT, false, 0, 0);

  var texA, texB, fboA, fboB, DPR = 1, W = 0, H = 0;
  function makeTex(w, h) {
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, w, h, 0, gl.RG, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function makeFbo(t) {
    var f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return f;
  }

  function build() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * DPR; canvas.height = H * DPR;
    GW = Math.min(CFG.maxW, Math.round(W / CFG.down));
    GH = Math.round(GW * H / W);
    if (texA) { gl.deleteTexture(texA); gl.deleteTexture(texB); gl.deleteFramebuffer(fboA); gl.deleteFramebuffer(fboB); }
    texA = makeTex(GW, GH); fboA = makeFbo(texA);
    texB = makeTex(GW, GH); fboB = makeFbo(texB);
    // clear both states to zero
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboA); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboB); gl.clear(gl.COLOR_BUFFER_BIT);
  }

  // uniform locations
  function U(p, n) { return gl.getUniformLocation(p, n); }
  var uSt = U(simP, "st"), uTexel = U(simP, "texel"), uAspect = U(simP, "aspect"),
    uC2 = U(simP, "c2"), uDamp = U(simP, "damp"), uFreq = U(simP, "freq"),
    uPlaneAmp = U(simP, "planeAmp"), uPlaneAge = U(simP, "planeAge"),
    uNsrc = U(simP, "nsrc"), uSrcs = U(simP, "srcs"),
    uGapC = U(simP, "gapC"), uGapR = U(simP, "gapR"), uLensX = U(simP, "lensX");
  var rSt = U(rendP, "st"), rGain = U(rendP, "gain"), rDark = U(rendP, "dark"),
    rRes = U(rendP, "res"), rGapC = U(rendP, "gapC"), rGapR = U(rendP, "gapR"),
    rAspect = U(rendP, "aspect");

  // pointer = radial pulse source
  var moveAcc = 0;
  window.addEventListener("pointermove", function (e) {
    var t = performance.now(); if (t - moveAcc < 110) return; moveAcc = t;
    srcs.push({ x: e.clientX / W, y: 1 - e.clientY / H, age: 0, amp: CFG.pulseAmp * 0.5 });
  }, { passive: true });
  window.addEventListener("pointerdown", function (e) {
    srcs.push({ x: e.clientX / W, y: 1 - e.clientY / H, age: 0, amp: CFG.pulseAmp });
  }, { passive: true });

  var resizeTimer;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer); resizeTimer = setTimeout(build, 150);
  });

  var lastDark = null;
  function setOpacity() {
    var dark = document.documentElement.classList.contains("theme-dark");
    if (dark !== lastDark) { lastDark = dark; canvas.style.opacity = dark ? CFG.opacityDark : CFG.opacity; }
    return dark;
  }
  window.addEventListener("themechange", setOpacity);

  var autoT = 0, nextAuto = 1.5;
  function pushSrcUniform() {
    var n = Math.min(srcs.length, 8), flat = new Float32Array(32);
    for (var i = 0; i < n; i++) {
      flat[i * 4] = srcs[i].x; flat[i * 4 + 1] = srcs[i].y;
      flat[i * 4 + 2] = srcs[i].age; flat[i * 4 + 3] = srcs[i].amp;
    }
    gl.uniform1i(uNsrc, n); gl.uniform4fv(uSrcs, flat);
  }

  var gapC = [0.82, 0.3], gapR = 0.085, lensX = [0.58, 0.63];
  var srcA = texA, srcF = fboA, dstA = texB, dstF = fboB;

  function step() {
    gl.useProgram(simP);
    gl.uniform2f(uTexel, 1 / GW, 1 / GH);
    gl.uniform1f(uAspect, GW / GH);
    gl.uniform1f(uC2, CFG.c2); gl.uniform1f(uDamp, CFG.damp); gl.uniform1f(uFreq, CFG.freq);
    gl.uniform1f(uPlaneAmp, CFG.planeAmp); gl.uniform1f(uPlaneAge, planeAge);
    gl.uniform2f(uGapC, gapC[0], gapC[1]); gl.uniform1f(uGapR, gapR);
    gl.uniform2f(uLensX, lensX[0], lensX[1]);
    pushSrcUniform();
    gl.viewport(0, 0, GW, GH);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcA);
    gl.uniform1i(uSt, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstF);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // swap
    var t = srcA; srcA = dstA; dstA = t; t = srcF; srcF = dstF; dstF = t;
    planeAge += 1;
    for (var i = srcs.length - 1; i >= 0; i--) { srcs[i].age += 1; if (srcs[i].age > 40) srcs.splice(i, 1); }
  }

  var last = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    if (!last) last = now;
    var dt = (now - last) / 1000; last = now;
    autoT += dt;
    if (autoT >= nextAuto) {
      autoT = 0; nextAuto = 1.4 + Math.random() * 2.2;
      srcs.push({ x: 0.15 + Math.random() * 0.7, y: 0.15 + Math.random() * 0.7, age: 0, amp: CFG.pulseAmp * 0.7 });
    }
    var dark = setOpacity();
    for (var s = 0; s < CFG.substeps; s++) step();
    // render current state (srcA holds the latest after the last swap)
    gl.useProgram(rendP);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcA);
    gl.uniform1i(rSt, 0); gl.uniform1f(rGain, CFG.gain); gl.uniform1f(rDark, dark ? 1 : 0);
    gl.uniform2f(rRes, GW, GH); gl.uniform2f(rGapC, gapC[0], gapC[1]);
    gl.uniform1f(rGapR, gapR); gl.uniform1f(rAspect, GW / GH);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  build();
  setOpacity();
  requestAnimationFrame(frame);
})();
