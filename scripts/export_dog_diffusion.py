#!/usr/bin/env python3
"""
Sample from / export the current Dog Diffusion checkpoint without touching the
training loop. Usage:
  python export_dog_diffusion.py preview   # 8-dog sample grid only (low CPU)
  python export_dog_diffusion.py export    # preview + ONNX + JSON
"""
import json, os, sys

import torch
from PIL import Image
import numpy as np

from build_dog_diffusion import UNet, TSTEPS, IMG, HERE, OUT_MODELS, CKPT_NAME, onnx_to_fp16

CKPT = os.path.join(HERE, CKPT_NAME)


def main(mode):
    torch.set_num_threads(4)        # stay out of the training run's way
    ema = UNet()
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    print("checkpoint epoch:", ck["epoch"], flush=True)

    betas = torch.linspace(1e-4, 0.02, TSTEPS)
    acp = torch.cumprod(1.0 - betas, dim=0)

    K = 50
    seq = torch.linspace(0, TSTEPS - 1, K).long().flip(0)
    torch.manual_seed(11)
    with torch.no_grad():
        x = torch.randn(8, 3, IMG, IMG)
        for i, ti in enumerate(seq):
            a = acp[ti]
            a_prev = acp[seq[i + 1]] if i + 1 < K else torch.tensor(1.0)
            eps = ema(x, torch.full((8,), float(ti)))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    grid = Image.new("RGB", (8 * IMG, IMG))
    for i in range(8):
        arr = ((x[i].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        grid.paste(Image.fromarray(arr), (i * IMG, 0))
    out_png = os.path.join(HERE, f"dog_ddpm_preview_e{ck['epoch']}.png")
    grid.save(out_png)
    print("preview:", out_png, flush=True)

    if mode != "export":
        return

    os.makedirs(OUT_MODELS, exist_ok=True)
    onnx_path = os.path.join(OUT_MODELS, "dog_diffusion.onnx")
    dummy = (torch.zeros(1, 3, IMG, IMG), torch.zeros(1))
    try:
        torch.onnx.export(ema, dummy, onnx_path, input_names=["x", "t"],
                          output_names=["eps"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(ema, dummy, onnx_path, input_names=["x", "t"],
                          output_names=["eps"], opset_version=17)
    onnx_to_fp16(onnx_path)
    print("onnx (fp16):", onnx_path, os.path.getsize(onnx_path), "bytes", flush=True)
    with open(os.path.join(OUT_MODELS, "dog_diffusion.json"), "w") as f:
        json.dump({"img": IMG, "tsteps": TSTEPS,
                   "acp": [round(float(v), 6) for v in acp]}, f)
    print("json done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "preview")
