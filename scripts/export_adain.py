#!/usr/bin/env python3
"""
Export the AdaIN arbitrary-style-transfer model (Huang & Belongie, ICCV 2017)
to a single ONNX graph for the in-browser style-transfer demo.

Weights: https://github.com/naoto0804/pytorch-AdaIN (MIT), release v0.0.0
(scripts/adain_decoder.pth + scripts/adain_vgg.pth, downloaded separately).

Graph interface (matches the site's 0-255 RGB convention):
  content [1,3,224,224], style [1,3,224,224], alpha [1]  ->  output1 [1,3,224,224]
"""
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "models", "style", "adain.onnx")
SIZE = 224

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, 3), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, 3), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, 3), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, 3), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, 3), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, 3),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, 1),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, 3), nn.ReLU(),       # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, 3), nn.ReLU(),      # relu1_2
    nn.MaxPool2d(2, 2, 0),  # ceil_mode=False: identical at 224 (even dims), and WebGPU lacks ceil support
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, 3), nn.ReLU(),     # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, 3), nn.ReLU(),    # relu2_2
    nn.MaxPool2d(2, 2, 0),  # ceil_mode=False: identical at 224 (even dims), and WebGPU lacks ceil support
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, 3), nn.ReLU(),    # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_2
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_3
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),    # relu3_4
    nn.MaxPool2d(2, 2, 0),  # ceil_mode=False: identical at 224 (even dims), and WebGPU lacks ceil support
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, 3), nn.ReLU(),    # relu4_1
    # (deeper layers exist in the checkpoint but are not used)
)


def calc_mean_std(feat, eps=1e-5):
    # ONNX-friendly version (no .var()): unbiased variance computed by hand
    n, c, h, w = feat.shape
    flat = feat.reshape(n, c, h * w)
    mean = flat.mean(dim=2, keepdim=True)
    var = ((flat - mean) ** 2).sum(dim=2, keepdim=True) / (h * w - 1) + eps
    return mean.reshape(n, c, 1, 1), var.sqrt().reshape(n, c, 1, 1)


class AdaINNet(nn.Module):
    def __init__(self, encoder, dec):
        super().__init__()
        self.encoder = encoder
        self.dec = dec

    def forward(self, content, style, alpha):
        c = self.encoder(content / 255.0)
        s = self.encoder(style / 255.0)
        c_mean, c_std = calc_mean_std(c)
        s_mean, s_std = calc_mean_std(s)
        t = (c - c_mean) / c_std * s_std + s_mean
        t = alpha * t + (1.0 - alpha) * c
        return self.dec(t).clamp(0, 1) * 255.0


def main():
    vgg.load_state_dict(torch.load(os.path.join(HERE, "adain_vgg.pth"),
                                   map_location="cpu", weights_only=True), strict=False)
    decoder.load_state_dict(torch.load(os.path.join(HERE, "adain_decoder.pth"),
                                       map_location="cpu", weights_only=True))
    net = AdaINNet(vgg, decoder).eval()

    dummy = (torch.zeros(1, 3, SIZE, SIZE), torch.zeros(1, 3, SIZE, SIZE), torch.ones(1))
    try:
        torch.onnx.export(net, dummy, OUT,
                          input_names=["content", "style", "alpha"],
                          output_names=["output1"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(net, dummy, OUT,
                          input_names=["content", "style", "alpha"],
                          output_names=["output1"], opset_version=17)
    print("onnx:", OUT, os.path.getsize(OUT), "bytes")

    # visual check: chihuahua + Great Wave
    from PIL import Image
    import numpy as np
    def load(p):
        im = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None]
    content = load(os.path.join(HERE, "..", "assets", "images", "style_sample.jpg"))
    style = load(os.path.join(HERE, "..", "assets", "images", "styles", "wave.jpg"))
    with torch.no_grad():
        y = net(content, style, torch.ones(1))
    out = y[0].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(HERE, "adain_test.png"))
    print("test image:", os.path.join(HERE, "adain_test.png"))


if __name__ == "__main__":
    main()
