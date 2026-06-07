(function() {
    // Theme is applied earlier by the inline <head> script to avoid a flash;
    // re-apply here as a fallback for layouts that miss it.
    // Dark is the default; light only if explicitly chosen.
    if (localStorage.getItem('site-theme') !== 'light') {
        document.documentElement.classList.add('theme-dark');
    }

    function updateIcon() {
        var btn = document.getElementById('theme-toggle');
        if (!btn) return;
        var dark = document.documentElement.classList.contains('theme-dark');
        btn.innerHTML = dark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }

    window.toggleTheme = function() {
        var dark = document.documentElement.classList.toggle('theme-dark');
        localStorage.setItem('site-theme', dark ? 'dark' : 'light');
        updateIcon();
        // let canvas widgets redraw with theme-aware colors
        window.dispatchEvent(new Event('themechange'));
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateIcon);
    } else {
        updateIcon();
    }
})();
