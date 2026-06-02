#!/usr/bin/env python3
"""
Build a small CIFAR-10 sample: a sprite sheet of 32x32 thumbnails plus a JSON of
2D coordinates (grouped by class) for the latent-space-style explorer.
Source: Hugging Face datasets-server rows API (uoft-cs/cifar10).
"""
import io, json, math, os, urllib.request
from PIL import Image

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
PER_CLASS = 30
THUMB = 32
COLS = 30                      # sprite grid columns
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "images", "cifar")
ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=uoft-cs/cifar10&config=plain_text&split=train&offset={off}&length=100")

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def fetch_img(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB").resize((THUMB, THUMB))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    buckets = {c: [] for c in CLASSES}
    need = PER_CLASS * len(CLASSES)
    off = 0
    while sum(len(v) for v in buckets.values()) < need and off < 4000:
        data = fetch_json(ROWS_URL.format(off=off))
        for row in data.get("rows", []):
            label = CLASSES[row["row"]["label"]]
            if len(buckets[label]) >= PER_CLASS:
                continue
            try:
                buckets[label].append(fetch_img(row["row"]["img"]["src"]))
            except Exception as e:
                print("skip", e)
        off += 100
        print("collected", sum(len(v) for v in buckets.values()), "/", need)

    # flatten in class order
    items = []  # (img, label_index)
    for ci, c in enumerate(CLASSES):
        for img in buckets[c]:
            items.append((img, ci))
    n = len(items)
    rows = math.ceil(n / COLS)

    sprite = Image.new("RGB", (COLS * THUMB, rows * THUMB), (255, 255, 255))
    coords = []
    # 2D layout: each class clustered around a point on a circle, jittered
    import random
    random.seed(7)
    centers = []
    for ci in range(len(CLASSES)):
        ang = 2 * math.pi * ci / len(CLASSES)
        centers.append((0.5 + 0.36 * math.cos(ang), 0.5 + 0.36 * math.sin(ang)))
    for i, (img, ci) in enumerate(items):
        r, cgrid = divmod(i, COLS)
        sprite.paste(img, (cgrid * THUMB, r * THUMB))
        cx, cy = centers[ci]
        x = min(0.98, max(0.02, cx + random.uniform(-0.13, 0.13)))
        y = min(0.98, max(0.02, cy + random.uniform(-0.13, 0.13)))
        coords.append({"x": round(x, 4), "y": round(y, 4), "c": ci, "r": r, "g": cgrid})

    sprite.save(os.path.join(OUT_DIR, "cifar_sprite.png"), optimize=True)
    with open(os.path.join(OUT_DIR, "cifar_data.json"), "w") as f:
        json.dump({"classes": CLASSES, "thumb": THUMB, "cols": COLS, "items": coords}, f)
    print("done:", n, "images,", rows, "sprite rows ->", OUT_DIR)

if __name__ == "__main__":
    main()
