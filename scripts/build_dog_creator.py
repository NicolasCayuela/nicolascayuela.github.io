#!/usr/bin/env python3
"""
Dog Creator pipeline (PyTorch):
1. Load AFHQ dog faces from the locally downloaded parquet shards (scripts/afhq_cache).
2. Train a convolutional autoencoder on 64x64 RGB dog faces.
3. Run PCA on the latent codes -> 256-dimensional slider space.
4. Export a single ONNX graph (PCA coords -> decoded image) for onnxruntime-web,
   plus a JSON with per-component stds and sample dataset projections.

Outputs:
  assets/models/dog_decoder.onnx
  assets/models/dog_data.json
  scripts/dog_ae.pt           (checkpoint, for re-runs)
  scripts/dog_preview.png     (reconstruction sanity check)
"""
import io, json, math, os, sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "afhq_cache")
OUT_MODELS = os.path.join(HERE, "..", "assets", "models")
IMG = 64
LATENT = 256
PCS = 64                      # principal components exposed as sliders
EPOCHS = 200
BATCH = 64
LR_MAX, LR_MIN = 1e-3, 1e-4   # cosine decay
THREADS = 6                   # leave most cores to the concurrent DDPM run
DOG_LABEL = 1                 # huggan/AFHQ: cat=0, dog=1, wild=2
SAMPLES_IN_JSON = 300

torch.manual_seed(7)
np.random.seed(7)


def load_dogs():
    """Decode all dog images from the parquet shards into a uint8 tensor."""
    imgs = []
    for shard in sorted(os.listdir(CACHE)):
        if not shard.endswith(".parquet"):
            continue
        table = pq.read_table(os.path.join(CACHE, shard), columns=["image", "label"])
        labels = table.column("label").to_numpy()
        images = table.column("image").to_pylist()
        for rec, lab in zip(images, labels):
            if lab != DOG_LABEL:
                continue
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
            imgs.append(np.asarray(im, dtype=np.uint8))
        print(f"{shard}: total dogs so far {len(imgs)}")
    data = torch.from_numpy(np.stack(imgs))          # N x 64 x 64 x 3 uint8
    return data.permute(0, 3, 1, 2).contiguous()     # N x 3 x 64 x 64


def up_block(cin, cout):
    """Upsample + conv avoids the checkerboard artifacts of ConvTranspose."""
    return [nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True)]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 4, 2, 1), nn.BatchNorm2d(48), nn.ReLU(True),    # 32
            nn.Conv2d(48, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True),   # 16
            nn.Conv2d(96, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(True), # 8
            nn.Conv2d(192, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.ReLU(True),# 4
            nn.Flatten(),
            nn.Linear(384 * 4 * 4, LATENT),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 384 * 4 * 4)
        self.net = nn.Sequential(
            *up_block(384, 192),                       # 8
            *up_block(192, 96),                        # 16
            *up_block(96, 48),                         # 32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(48, 3, 3, padding=1), nn.Sigmoid(),  # 64
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 384, 4, 4)
        return self.net(h)


class PCADecoder(nn.Module):
    """ONNX export wrapper: PCA coordinates -> image."""
    def __init__(self, decoder, mean, components):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("mean", mean)               # (LATENT,)
        self.register_buffer("components", components)   # (LATENT, LATENT) rows = PCs

    def forward(self, p):
        z = p @ self.components + self.mean
        return self.decoder(z)


def main():
    data = load_dogs()
    n = data.shape[0]
    print("dogs:", n)

    enc, dec = Encoder(), Decoder()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    ckpt_path = os.path.join(HERE, "dog_ae2.pt")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
        opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"]
        print("resumed at epoch", start_epoch)

    torch.set_num_threads(THREADS)
    for epoch in range(start_epoch, EPOCHS):
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        enc.train(); dec.train()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            x = data[idx].float() / 255.0
            flip = torch.rand(x.shape[0]) < 0.5      # horizontal-flip augmentation
            x[flip] = x[flip].flip(-1)
            opt.zero_grad()
            out = dec(enc(x))
            loss = loss_fn(out, x)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  mse {tot / nb:.5f}", flush=True)
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                    "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt_path)

    # ---- encode the whole dataset ----
    enc.eval(); dec.eval()
    zs = []
    with torch.no_grad():
        for b in range(0, n, 256):
            zs.append(enc(data[b:b + 256].float() / 255.0))
    Z = torch.cat(zs)                                    # N x LATENT

    # ---- PCA (SVD on centered latents) ----
    mean = Z.mean(0)
    Zc = Z - mean
    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False) # Vh: LATENT x LATENT
    stds = (S / math.sqrt(n - 1))                        # per-component std
    proj = Zc @ Vh.T                                     # N x LATENT, PCA coords

    # ---- ONNX export: p (1xPCS) -> image (1x3x64x64); only the top PCS
    # components are exposed, the rest stay at the mean ----
    os.makedirs(OUT_MODELS, exist_ok=True)
    wrapper = PCADecoder(dec, mean, Vh[:PCS]).eval()
    dummy = torch.zeros(1, PCS)
    onnx_path = os.path.join(OUT_MODELS, "dog_decoder.onnx")
    try:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17)
    print("onnx:", onnx_path, os.path.getsize(onnx_path), "bytes")

    # ---- JSON: stds + sample dataset projections ----
    pick = torch.randperm(n)[:SAMPLES_IN_JSON]
    samples = proj[pick, :PCS]
    js = {
        "latent": PCS,
        "stds": [round(float(v), 4) for v in stds[:PCS]],
        "samples": [[round(float(v), 2) for v in row] for row in samples],
    }
    with open(os.path.join(OUT_MODELS, "dog_data.json"), "w") as f:
        json.dump(js, f)
    print("json:", os.path.join(OUT_MODELS, "dog_data.json"))

    # ---- preview grid: original vs reconstruction vs mean-dog ----
    with torch.no_grad():
        x = data[:8].float() / 255.0
        rec = dec(enc(x))
        avg = wrapper(torch.zeros(1, PCS))
    grid = Image.new("RGB", (8 * IMG, 3 * IMG))
    for i in range(8):
        grid.paste(Image.fromarray((x[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)), (i * IMG, 0))
        grid.paste(Image.fromarray((rec[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)), (i * IMG, IMG))
    grid.paste(Image.fromarray((avg[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)), (0, 2 * IMG))
    grid.save(os.path.join(HERE, "dog_preview.png"))
    print("preview:", os.path.join(HERE, "dog_preview.png"))


if __name__ == "__main__":
    main()
