#!/usr/bin/env python3
"""
Dog Creator pipeline (PyTorch):
1. Load AFHQ dog faces from the locally downloaded parquet shards (scripts/afhq_cache).
2. Train a convolutional autoencoder on 128x128 RGB dog faces.
3. Run PCA on the latent codes -> 512-dimensional slider space.
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
IMG = 128
LATENT = 768                  # v6: wider latent for the residual+attention AE
SLIDERS = 64                  # PCA components exposed as UI sliders (top, most semantic)
PCS = LATENT                  # components fed to the decoder: full latent = no truncation blur
EPOCHS = 300
BATCH = 48                    # v6 is bigger (resblocks + attention); leave VRAM headroom
LR_MAX, LR_MIN = 2e-4, 2e-5   # cosine decay
THREADS = os.cpu_count() or 6
PERC_W = 0.2                  # weight of the VGG perceptual loss vs pixel MSE (v6: stronger)
Z_NOISE = 0.05                # latent noise (fraction of batch latent std) -> smoother sliders
DOG_LABEL = 1                 # huggan/AFHQ: cat=0, dog=1, wild=2
SAMPLES_IN_JSON = 300
WARM_START = None             # v6 arch is new (resblocks + attention) - no warm start
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    data = torch.from_numpy(np.stack(imgs))          # N x IMG x IMG x 3 uint8
    return data.permute(0, 3, 1, 2).contiguous()     # N x 3 x IMG x IMG


class ResBlock(nn.Module):
    """GroupNorm-SiLU-conv residual block (the diffusion v4 block minus the
    time embedding). GroupNorm + 1x1 skip; SiLU activation."""
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head spatial self-attention at 16x16: every position attends to
    all others -> global face coherence (symmetric eyes, consistent colour).
    Same module as build_dog_diffusion_v4.py; exports cleanly to ONNX/WebGPU."""
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.scale = (c // heads) ** -0.5

    def forward(self, x):
        n, c, h, w = x.shape
        hd, dh = self.heads, c // self.heads
        qkv = self.qkv(self.norm(x)).reshape(n, 3, hd, dh, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(n, c, h, w)
        return x + self.proj(out)


class Encoder(nn.Module):
    """v6: residual blocks + self-attention at 16x16, 128->4 px, flat latent."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),                    # 128
            ResBlock(48, 48), ResBlock(48, 48),
            nn.Conv2d(48, 96, 4, 2, 1),                        # 64
            ResBlock(96, 96), ResBlock(96, 96),
            nn.Conv2d(96, 192, 4, 2, 1),                       # 32
            ResBlock(192, 192), ResBlock(192, 192),
            nn.Conv2d(192, 320, 4, 2, 1),                      # 16
            ResBlock(320, 320), ResBlock(320, 320),
            SelfAttention(320),
            nn.Conv2d(320, 384, 4, 2, 1),                      # 8
            ResBlock(384, 384), ResBlock(384, 384),
            nn.Conv2d(384, 384, 4, 2, 1),                      # 4
            ResBlock(384, 384),
            nn.Flatten(),
            nn.Linear(384 * 4 * 4, LATENT),
        )

    def forward(self, x):
        return self.net(x)


def up(cin, cout):
    """Nearest upsample + 3x3 conv (anti-checkerboard) then a residual block."""
    return [nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1), ResBlock(cout, cout)]


class Decoder(nn.Module):
    """v6 mirror: flat latent -> 4x4, residual upsampling, attention at 16x16."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 384 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock(384, 384),                                # 4
            *up(384, 384),                                     # 8
            *up(384, 320),                                     # 16
            SelfAttention(320),
            *up(320, 192),                                     # 32
            *up(192, 96),                                      # 64
            *up(96, 48),                                       # 128
            nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv2d(48, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 384, 4, 4)
        return self.net(h)


class VGGFeats(nn.Module):
    """Frozen slice of the AdaIN-normalised VGG used as a multi-scale
    perceptual loss: feature-space MSE at relu1_2 / relu2_2 / relu3_1 / relu4_1
    makes reconstructions much sharper than pixel MSE alone (low layers add edge
    and texture detail, relu4_1 adds higher-level structure). Weights:
    scripts/adain_vgg.pth (MIT)."""
    def __init__(self):
        super().__init__()
        vgg = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3), nn.ReLU(),       # relu1_1
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),      # idx 6: relu1_2
            nn.MaxPool2d(2, 2),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU(),     # relu2_1
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),    # idx 13: relu2_2
            nn.MaxPool2d(2, 2),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3), nn.ReLU(),    # idx 17: relu3_1
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_2
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_3
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_4
            nn.MaxPool2d(2, 2),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3), nn.ReLU(),    # idx 30: relu4_1
        )
        vgg.load_state_dict(torch.load(os.path.join(HERE, "adain_vgg.pth"),
                                       map_location="cpu", weights_only=True), strict=False)
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.net = vgg.eval()
        self.taps = (7, 14, 18, 31)              # relu1_2 / relu2_2 / relu3_1 / relu4_1

    def forward(self, x):                        # x in [0, 1]
        feats, prev = [], 0
        for i in self.taps:
            x = self.net[prev:i](x)
            feats.append(x)
            prev = i
        return feats


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
    print("device:", DEVICE, flush=True)
    data = load_dogs().to(DEVICE)        # ~60 MB uint8, fits in VRAM
    n = data.shape[0]
    print("dogs:", n)

    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    vgg = VGGFeats().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR_MAX)
    loss_fn = nn.MSELoss()

    ckpt_path = os.path.join(HERE, "dog_ae6.pt")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
        opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"]
        print("resumed at epoch", start_epoch)
    elif WARM_START and os.path.exists(os.path.join(HERE, WARM_START)):
        ck = torch.load(os.path.join(HERE, WARM_START), map_location="cpu", weights_only=True)
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
        print("warm start from", WARM_START)

    torch.set_num_threads(THREADS)
    for epoch in range(start_epoch, EPOCHS):
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n, device=DEVICE)
        tot, nb = 0.0, 0
        enc.train(); dec.train()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            x = data[idx].float() / 255.0
            flip = torch.rand(x.shape[0], device=DEVICE) < 0.5  # horizontal-flip augmentation
            x[flip] = x[flip].flip(-1)
            opt.zero_grad()
            z = enc(x)
            # denoising in latent space: decoder must map a small neighborhood
            # of z to the same image -> smoother slider interpolation
            z = z + Z_NOISE * z.detach().std() * torch.randn_like(z)
            out = dec(z)
            pix = loss_fn(out, x)
            perc = sum(loss_fn(fo, ft) for fo, ft in zip(vgg(out), vgg(x))) / len(vgg.taps)
            loss = pix + PERC_W * perc
            loss.backward()
            opt.step()
            tot += pix.item(); nb += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  pix-mse {tot / nb:.5f}", flush=True)
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

    # ---- ONNX export: p (1xPCS) -> image (1x3xIMGxIMG); only the top PCS
    # components are exposed, the rest stay at the mean (CPU for a clean graph) ----
    os.makedirs(OUT_MODELS, exist_ok=True)
    wrapper = PCADecoder(dec, mean, Vh[:PCS]).eval().cpu()
    dummy = torch.zeros(1, PCS)
    onnx_path = os.path.join(OUT_MODELS, "dog_decoder.onnx")
    try:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17)
    from build_dog_diffusion import onnx_to_fp16
    onnx_to_fp16(onnx_path)
    print("onnx (fp16):", onnx_path, os.path.getsize(onnx_path), "bytes")

    # ---- JSON: stds + sample dataset projections ----
    pick = torch.randperm(n)[:SAMPLES_IN_JSON]
    samples = proj[pick, :PCS]
    js = {
        "latent": PCS,
        "sliders": SLIDERS,
        "img": IMG,
        "stds": [round(float(v), 4) for v in stds[:PCS]],
        "samples": [[round(float(v), 2) for v in row] for row in samples],
    }
    with open(os.path.join(OUT_MODELS, "dog_data.json"), "w") as f:
        json.dump(js, f)
    print("json:", os.path.join(OUT_MODELS, "dog_data.json"))

    # ---- preview grid: original vs reconstruction vs mean-dog ----
    # (dec/wrapper were moved to CPU for the ONNX export; do the preview on CPU)
    enc = enc.cpu()
    with torch.no_grad():
        x = data[:8].float().cpu() / 255.0
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
