#!/usr/bin/env python3
"""
Build a small CIFAR-100 sample: a sprite sheet of 32x32 thumbnails plus a JSON
of coordinates (grouped by class) for the latent-space-style explorer.
Source: Hugging Face datasets-server rows API (uoft-cs/cifar100).
"""
import io, json, math, os, urllib.request
from PIL import Image

CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]
PER_CLASS = 10
THUMB = 32
COLS = 32                      # sprite grid columns
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "images", "cifar")
ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=uoft-cs/cifar100&config=cifar100&split=train&offset={off}&length=100")

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
    while sum(len(v) for v in buckets.values()) < need and off < 20000:
        data = fetch_json(ROWS_URL.format(off=off))
        for row in data.get("rows", []):
            label = CLASSES[row["row"]["fine_label"]]
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
    # (kept for compatibility; the web widget builds its own 3D layout)
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
