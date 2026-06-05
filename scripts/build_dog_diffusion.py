#!/usr/bin/env python3
"""
Dog Diffusion pipeline (PyTorch):
Train a small DDPM (epsilon-prediction UNet, 32x32) on AFHQ dog faces from the
locally downloaded parquet shards, keep an EMA copy of the weights, then export
the UNet to ONNX for in-browser DDIM sampling with onnxruntime-web.

Outputs:
  assets/models/dog_diffusion.onnx
  assets/models/dog_diffusion.json   (alphas_cumprod schedule)
  scripts/dog_ddpm.pt                (checkpoint, for re-runs)
  scripts/dog_ddpm_preview.png       (sampling sanity check)
"""
import io, json, math, os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "afhq_cache")
OUT_MODELS = os.path.join(HERE, "..", "assets", "models")
IMG = 32
EPOCHS = 800
BATCH = 128
LR_MAX, LR_MIN = 2e-4, 2e-5   # cosine decay over the full run
CKPT_NAME = "dog_ddpm2.pt"    # v2: wider UNet with attention
TSTEPS = 1000
DOG_LABEL = 1                 # huggan/AFHQ: cat=0, dog=1, wild=2
EMA_DECAY = 0.9995              # ~50-epoch horizon, smoother late-training average

torch.manual_seed(7)
np.random.seed(7)


def load_dogs():
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
        print(f"{shard}: total dogs so far {len(imgs)}", flush=True)
    data = torch.from_numpy(np.stack(imgs))
    return data.permute(0, 3, 1, 2).contiguous()     # N x 3 x 32 x 32 uint8


# ---- model ----
TDIM = 128

def time_embedding(t):
    """Sinusoidal embedding; t is a float tensor of shape (B,)."""
    half = TDIM // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(TDIM, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions (used at 8x8)."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.scale = c ** -0.5

    def forward(self, x):
        n, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(n, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # n, c, hw
        attn = torch.softmax(q.transpose(1, 2) @ k * self.scale, dim=-1)  # n, hw, hw
        out = (v @ attn.transpose(1, 2)).reshape(n, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """v2: wider (48/96/192), two ResBlocks per level, attention at 8x8."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(TDIM, TDIM), nn.SiLU(), nn.Linear(TDIM, TDIM))
        self.stem = nn.Conv2d(3, 48, 3, padding=1)
        self.d1a = ResBlock(48, 48); self.d1b = ResBlock(48, 48)
        self.down1 = nn.Conv2d(48, 96, 4, 2, 1)            # 16
        self.d2a = ResBlock(96, 96); self.d2b = ResBlock(96, 96)
        self.down2 = nn.Conv2d(96, 192, 4, 2, 1)           # 8
        self.mid1 = ResBlock(192, 192)
        self.attn = SelfAttention(192)
        self.mid2 = ResBlock(192, 192)
        self.up1 = nn.ConvTranspose2d(192, 96, 4, 2, 1)    # 16
        self.u1a = ResBlock(192, 96); self.u1b = ResBlock(96, 96)
        self.up2 = nn.ConvTranspose2d(96, 48, 4, 2, 1)     # 32
        self.u2a = ResBlock(96, 48); self.u2b = ResBlock(48, 48)
        self.out_norm = nn.GroupNorm(8, 48)
        self.out_conv = nn.Conv2d(48, 3, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        emb = self.mlp(time_embedding(t))
        h1 = self.d1b(self.d1a(self.stem(x), emb), emb)            # 48ch, 32px
        h2 = self.d2b(self.d2a(self.down1(h1), emb), emb)          # 96ch, 16px
        m = self.mid2(self.attn(self.mid1(self.down2(h2), emb)), emb)
        u = self.u1b(self.u1a(torch.cat([self.up1(m), h2], 1), emb), emb)
        u = self.u2b(self.u2a(torch.cat([self.up2(u), h1], 1), emb), emb)
        return self.out_conv(self.act(self.out_norm(u)))


def main():
    data = load_dogs()
    n = data.shape[0]
    print("dogs:", n, flush=True)

    betas = torch.linspace(1e-4, 0.02, TSTEPS)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)               # alphas_cumprod
    sqrt_acp = acp.sqrt()
    sqrt_1macp = (1 - acp).sqrt()

    model = UNet()
    ema = UNet()
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    ckpt_path = os.path.join(HERE, CKPT_NAME)
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"]
        print("resumed at epoch", start_epoch, flush=True)

    torch.set_num_threads(os.cpu_count() or 4)
    for epoch in range(start_epoch, EPOCHS):
        # cosine learning-rate decay
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            x0 = data[idx].float() / 127.5 - 1.0     # [-1, 1]
            flip = torch.rand(x0.shape[0]) < 0.5     # horizontal-flip augmentation
            x0[flip] = x0[flip].flip(-1)
            t = torch.randint(0, TSTEPS, (x0.shape[0],))
            eps = torch.randn_like(x0)
            xt = sqrt_acp[t, None, None, None] * x0 + sqrt_1macp[t, None, None, None] * eps
            pred = model(xt, t.float())
            loss = nn.functional.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item(); nb += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {tot / nb:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "epoch": epoch + 1}, ckpt_path)

    # ---- DDIM sampling sanity check (50 steps, 8 dogs) ----
    ema.eval()
    K = 50
    seq = torch.linspace(0, TSTEPS - 1, K).long().flip(0)
    with torch.no_grad():
        x = torch.randn(8, 3, IMG, IMG)
        for i, ti in enumerate(seq):
            a = acp[ti]
            a_prev = acp[seq[i + 1]] if i + 1 < K else torch.tensor(1.0)
            epsm = ema(x, torch.full((8,), float(ti)))
            x0 = ((x - (1 - a).sqrt() * epsm) / a.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * epsm
    grid = Image.new("RGB", (8 * IMG, IMG))
    for i in range(8):
        arr = ((x[i].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        grid.paste(Image.fromarray(arr), (i * IMG, 0))
    grid.save(os.path.join(HERE, "dog_ddpm_preview.png"))
    print("preview:", os.path.join(HERE, "dog_ddpm_preview.png"), flush=True)

    # ---- ONNX export (EMA weights) ----
    os.makedirs(OUT_MODELS, exist_ok=True)
    onnx_path = os.path.join(OUT_MODELS, "dog_diffusion.onnx")
    dummy = (torch.zeros(1, 3, IMG, IMG), torch.zeros(1))
    try:
        torch.onnx.export(ema, dummy, onnx_path, input_names=["x", "t"],
                          output_names=["eps"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(ema, dummy, onnx_path, input_names=["x", "t"],
                          output_names=["eps"], opset_version=17)
    print("onnx:", onnx_path, os.path.getsize(onnx_path), "bytes", flush=True)

    with open(os.path.join(OUT_MODELS, "dog_diffusion.json"), "w") as f:
        json.dump({"img": IMG, "tsteps": TSTEPS,
                   "acp": [round(float(v), 6) for v in acp]}, f)
    print("json done", flush=True)


if __name__ == "__main__":
    main()
