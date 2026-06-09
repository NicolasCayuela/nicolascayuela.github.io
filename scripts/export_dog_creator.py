#!/usr/bin/env python3
"""
Export the current Dog Creator checkpoint (dog_ae6.pt) to ONNX + JSON without
training - for early-stop deploys. Mirrors the export block of
build_dog_creator.main(): encode the dataset, PCA on the latents, export the
PCA->image decoder, write dog_data.json, save a preview grid.
"""
import json, math, os

import numpy as np
import torch
from PIL import Image

from build_dog_creator import (Encoder, Decoder, PCADecoder, load_dogs,
                               PCS, SLIDERS, IMG, OUT_MODELS, SAMPLES_IN_JSON, HERE)
from build_dog_diffusion import onnx_to_fp16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(HERE, "dog_ae6.pt")


def main():
    data = load_dogs().to(DEVICE)
    n = data.shape[0]
    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    enc.eval(); dec.eval()
    print("checkpoint epoch", ck["epoch"], flush=True)

    zs = []
    with torch.no_grad():
        for b in range(0, n, 256):
            zs.append(enc(data[b:b + 256].float() / 255.0))
    Z = torch.cat(zs)
    mean = Z.mean(0)
    Zc = Z - mean
    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    stds = (S / math.sqrt(n - 1))
    proj = (Zc @ Vh.T).cpu()

    os.makedirs(OUT_MODELS, exist_ok=True)
    wrapper = PCADecoder(dec.cpu(), mean.cpu(), Vh[:PCS].cpu()).eval()
    dummy = torch.zeros(1, PCS)
    onnx_path = os.path.join(OUT_MODELS, "dog_decoder.onnx")
    try:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(wrapper, dummy, onnx_path, input_names=["p"],
                          output_names=["img"], opset_version=17)
    onnx_to_fp16(onnx_path)
    print("onnx (fp16):", os.path.getsize(onnx_path), "bytes", flush=True)

    pick = torch.randperm(n)[:SAMPLES_IN_JSON]
    samples = proj[pick, :PCS]
    js = {"latent": PCS, "sliders": SLIDERS, "img": IMG,
          "stds": [round(float(v), 4) for v in stds[:PCS]],
          "samples": [[round(float(v), 2) for v in row] for row in samples]}
    with open(os.path.join(OUT_MODELS, "dog_data.json"), "w") as f:
        json.dump(js, f)
    print("json done", flush=True)

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
    out = os.path.join(HERE, "dog_preview_e%d.png" % ck["epoch"])
    grid.save(out)
    print("preview:", out, flush=True)


if __name__ == "__main__":
    main()
