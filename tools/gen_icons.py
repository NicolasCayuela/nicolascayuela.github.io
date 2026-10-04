"""Generate assets/css/icons.css: only the Font Awesome 5.15.1 Free icons the
site uses, as CSS masks (SVG data URIs), so <i class="fas fa-x"> keeps working
without the 13 KB CSS + 3 webfonts (~170 KB). Re-run after using a new icon:
    npm i @fortawesome/fontawesome-free@5.15.1   (anywhere)
    python tools/gen_icons.py <path to node_modules/@fortawesome/fontawesome-free/svgs>
"""
import os, re, sys, urllib.parse

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SVGS = sys.argv[1]
STYLE_DIR = {"fas": "solid", "fa": "solid", "far": "regular", "fab": "brands"}

used = set()
pat = re.compile(r"\b(fas|far|fab|fa) (fa-[a-z0-9-]+)")
for root, dirs, files in os.walk(SITE):
    dirs[:] = [d for d in dirs if d not in (".git", "scripts", "_site", "node_modules") and not d.startswith(".venv")]
    for f in files:
        if f.endswith((".html", ".js", ".yml", ".md")):
            t = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
            for m in pat.finditer(t):
                if m.group(2) not in ("fa-lg", "fa-2x", "fa-3x", "fa-fw"):
                    used.add((STYLE_DIR[m.group(1)], m.group(2)[3:]))
# swapped at runtime by JS (theme toggle, play/pause buttons)
used |= {("solid", "sun"), ("solid", "moon"), ("solid", "play"), ("solid", "pause"),
         ("solid", "school"), ("solid", "laptop-code")}   # cv.html section icons (built from a Liquid string)


def svg_uri(style, name, fill=None):
    svg = open(os.path.join(SVGS, style, name + ".svg"), encoding="utf-8").read()
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S).strip()
    w = int(re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).group(1))
    if fill:
        svg = svg.replace("<path ", '<path fill="%s" ' % fill)
    svg = svg.replace('"', "'")             # inside url("...")
    return w, "url(\"data:image/svg+xml," + urllib.parse.quote(svg, safe=" /=:;,'") + "\")"


out = ["/* Font Awesome Free 5.15.1 icons by @fontawesome - https://fontawesome.com",
       "   License: CC BY 4.0 (icons). Generated subset: only the icons this site uses,",
       "   drawn as CSS masks so they take the text colour (currentColor). */",
       ".fa, .fas, .far, .fab { display: inline-block; width: 1em; height: 1em; vertical-align: -.125em;",
       "    background-color: currentColor; -webkit-mask: var(--fa) center / contain no-repeat;",
       "    mask: var(--fa) center / contain no-repeat; font-style: normal; }",
       ".fa-lg { font-size: 1.333em; vertical-align: -.225em; }",
       ".fa-2x { font-size: 2em; } .fa-3x { font-size: 3em; }"]
sel = {"solid": ".fas.fa-%s, .fa.fa-%s", "regular": ".far.fa-%s", "brands": ".fab.fa-%s"}
for style, name in sorted(used, key=lambda x: (x[1], x[0])):
    w, uri = svg_uri(style, name)
    s = sel[style].replace("%s", name)
    out.append("%s { --fa: %s; width: %.4gem; }" % (s, uri, w / 512))
# white search-plus glyph for the publication cover zoom badge (::after)
_, zoom = svg_uri("solid", "search-plus", fill="#fff")
out.append(":root { --fa-zoom-white: %s; }" % zoom)
open(os.path.join(SITE, "assets", "css", "icons.css"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(len(used), "icons ->", os.path.getsize(os.path.join(SITE, "assets", "css", "icons.css")) // 1024, "KB")
