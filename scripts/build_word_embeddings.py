#!/usr/bin/env python3
"""
Build the word-embedding demo data: take the 20000 most frequent alphabetic
words from GloVe 6B 50d (word2vec-style embeddings, frequency-ordered),
compute a 3D PCA projection for display, and write one JSON.
Vectors are stored as integers (x1000): cosine similarity and vector
arithmetic are scale-invariant, so the JS uses them as-is.

Input : scripts/glove.6B.50d.txt (downloaded separately)
Output: assets/models/word_embeddings.json
"""
import json, os, re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "glove.6B.50d.txt")
OUT = os.path.join(HERE, "..", "assets", "models", "word_embeddings.json")
N_WORDS = 20000
DIM = 50

def main():
    words, vecs = [], []
    pat = re.compile(r"^[a-z]+$")
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if not pat.match(w):
                continue
            words.append(w)
            vecs.append([float(x) for x in parts[1:]])
            if len(words) >= N_WORDS:
                break
    V = np.asarray(vecs, dtype=np.float32)          # N x 50
    print("words:", len(words))

    # 3D PCA for the scatter display
    mean = V.mean(axis=0)
    Vc = V - mean
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    P = Vc @ Vt[:3].T                                # N x 3
    P /= np.abs(P).max()                             # roughly [-1, 1]

    data = {
        "dim": DIM,
        "scale": 1000,
        "words": words,
        "v": [[int(round(float(x) * 1000)) for x in row] for row in V],
        "xyz": [[int(round(float(x) * 1000)) for x in row] for row in P],
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("json:", OUT, os.path.getsize(OUT), "bytes")

if __name__ == "__main__":
    main()
