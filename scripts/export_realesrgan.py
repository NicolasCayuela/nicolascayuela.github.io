#!/usr/bin/env python3
"""Export realesr-general-x4v3 (SRVGGNetCompact, BSD-3) to ONNX with dynamic
H/W for in-browser 4x upscaling with onnxruntime-web.

Outputs assets/models/realesrgan_x4.onnx
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "models", "realesrgan_x4.onnx")


class SRVGGNetCompact(nn.Module):
    """Plain VGG-style SR network (architecture from Real-ESRGAN, BSD-3)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale ** 2, 3, 1, 1)]
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.upsampler(self.body(x))
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main():
    ck = torch.load(os.path.join(HERE, "realesr-general-x4v3.pth"),
                    map_location="cpu", weights_only=True)
    params = ck.get("params_ema") or ck.get("params") or ck
    model = SRVGGNetCompact()
    model.load_state_dict(params, strict=True)
    model.eval()

    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model(dummy)
    assert y.shape == (1, 3, 256, 256), y.shape

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    kwargs = dict(input_names=["img"], output_names=["out"],
                  dynamic_axes={"img": {2: "h", 3: "w"}, "out": {2: "H", 3: "W"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, OUT, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, OUT, **kwargs)
    print("onnx:", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
