// Pre-render TeX ($$...$$ and \(...\)) to inline SVG at deploy time, then drop
// the MathJax runtime from the page: no ~1 MB script, no typesetting after
// load, no layout shift. Run by CI on the built _site:
//   npm i mathjax-full@3 && node tools/prerender-math.js _site/publications.html ...
// A page whose render fails is left untouched (it keeps client-side MathJax).
const fs = require("fs");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

for (const file of process.argv.slice(2)) {
  try {
    const src = fs.readFileSync(file, "utf8");
    if (!/mathjax/i.test(src)) continue;
    const doc = mathjax.document(src, {
      InputJax: new TeX({ packages: AllPackages, inlineMath: [["\\(", "\\)"]], displayMath: [["$$", "$$"]] }),
      OutputJax: new SVG({ fontCache: "global" })   // glyphs shared once per page
    });
    doc.render();
    const n = Array.from(doc.math).length;
    let out = adaptor.doctype(doc.document) + "\n" + adaptor.outerHTML(adaptor.root(doc.document));
    // the runtime is no longer needed: config block, loader and its preconnect
    out = out
      .replace(/<link rel="preconnect" href="https:\/\/cdnjs\.cloudflare\.com" crossorigin="?"?>\s*/g, "")
      .replace(/<script>\s*window\.MathJax\s*=[\s\S]*?<\/script>\s*/g, "")
      .replace(/<script src="[^"]*mathjax[^"]*"[^>]*><\/script>\s*/gi, "")
      .replace(/<!-- MathJax:[^>]*-->\s*/g, "");
    fs.writeFileSync(file, out);
    console.log(file + ": " + n + " formulas pre-rendered");
  } catch (e) {
    console.error(file + ": left for client-side MathJax (" + e.message + ")");
  }
}
