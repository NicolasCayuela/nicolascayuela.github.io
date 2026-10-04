/* Site-wide UI without jQuery / Bootstrap JS: navbar toggle on small screens
   and the publications year scroll-spy. Images use native loading="lazy". */
(function () {
    "use strict";
    var btn = document.querySelector(".navbar-toggler");
    var menu = document.getElementById("navbarResponsive");
    if (btn && menu) {
        btn.addEventListener("click", function () {
            var open = menu.classList.toggle("show");
            btn.setAttribute("aria-expanded", open ? "true" : "false");
            btn.closest(".navbar").classList.toggle("menu-open", open);   // opaque while open
        });
    }

    // <body data-spy="scroll" data-target="#nav" data-offset="100">: mark the
    // nav link of the last section scrolled past as active
    var spy = document.querySelector('[data-spy="scroll"]');
    var nav = spy && document.querySelector(spy.getAttribute("data-target"));
    if (!nav) return;
    var offset = +spy.getAttribute("data-offset") || 0;
    var links = [].slice.call(nav.querySelectorAll('a[href^="#"]'));
    var targets = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); });
    var queued = false;
    function update() {
        queued = false;
        var cur = 0;
        for (var i = 0; i < targets.length; i++) {
            if (targets[i] && targets[i].getBoundingClientRect().top - offset <= 1) cur = i;
        }
        links.forEach(function (a, i) { a.classList.toggle("active", i === cur); });
    }
    window.addEventListener("scroll", function () {
        if (!queued) { queued = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
})();
