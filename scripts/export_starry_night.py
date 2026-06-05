#!/usr/bin/env python3
"""
Export the pretrained "Starry Night" fast-style-transfer model (PyTorch) to
ONNX for the in-browser style-transfer demo.

Weights: https://github.com/igreat/fast-style-transfer (MIT licence).
The wrapper bakes the repo's exact pre/de-processing into the graph so the
ONNX model has the same 0-255 RGB interface as the ONNX-zoo style models
(input1 [1,3,224,224] -> output1 [1,3,224,224]).
"""
import os, urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "models", "style", "starry-night.onnx")
PTH_URL = ("https://github.com/igreat/fast-style-transfer/raw/main/"
           "saved_models/starry_night_pretrained.pth")
PTH = os.path.join(HERE, "starry_night_pretrained.pth")


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ReflectConv(filters, filters, 3, stride=1)
        self.instance_norm1 = nn.InstanceNorm2d(filters, affine=True)
        self.conv2 = ReflectConv(filters, filters, 3, stride=1)
        self.instance_norm2 = nn.InstanceNorm2d(filters, affine=True)

    def forward(self, x):
        out = F.relu(self.instance_norm1(self.conv1(x)))
        out = self.instance_norm2(self.conv2(out))
        return out + x


class ReflectConv(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2, padding_mode="reflect")

    def forward(self, x):
        return self.conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, cin, cout, k, stride, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ReflectConv(cin, cout, k, stride)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.scale_factor, mode="nearest"))


class TransformationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ReflectConv(3, 32, 9, stride=1), nn.InstanceNorm2d(32, affine=True), nn.ReLU(),
            ReflectConv(32, 64, 3, stride=2), nn.InstanceNorm2d(64, affine=True), nn.ReLU(),
            ReflectConv(64, 128, 3, stride=2), nn.InstanceNorm2d(128, affine=True), nn.ReLU(),
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            ResidualBlock(128), ResidualBlock(128),
            UpsampleConv(128, 64, 3, stride=1, scale_factor=2),
            nn.InstanceNorm2d(64, affine=True), nn.ReLU(),
            UpsampleConv(64, 32, 3, stride=1, scale_factor=2),
            nn.InstanceNorm2d(32, affine=True), nn.ReLU(),
            ReflectConv(32, 3, 9, stride=1),
        )

    def forward(self, x):
        return self.layers(x)


class Wrapper(nn.Module):
    """0-255 RGB in -> 0-255 RGB out, replicating the repo's processing
    (note: the repo normalizes the raw 0-255 tensor by the ImageNet mean/std
    directly; the model was trained that way, so we replicate it exactly)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        h = (x - self.mean) / self.std
        y = self.model(h)
        return (y * self.std + self.mean).clamp(0, 1) * 255.0


def main():
    if not os.path.exists(PTH):
        print("downloading weights…")
        urllib.request.urlretrieve(PTH_URL, PTH)
    model = TransformationModel()
    ck = torch.load(PTH, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    wrapper = Wrapper(model).eval()

    dummy = torch.zeros(1, 3, 224, 224)
    dyn = {"input1": {2: "h", 3: "w"}, "output1": {2: "h", 3: "w"}}
    try:
        torch.onnx.export(wrapper, (dummy,), OUT, input_names=["input1"],
                          output_names=["output1"], opset_version=17,
                          dynamic_axes=dyn, dynamo=False)
    except TypeError:
        torch.onnx.export(wrapper, (dummy,), OUT, input_names=["input1"],
                          output_names=["output1"], opset_version=17,
                          dynamic_axes=dyn)
    print("onnx:", OUT, os.path.getsize(OUT), "bytes")

    # quick visual check on the site's sample image
    from PIL import Image
    import numpy as np
    sample = os.path.join(HERE, "..", "assets", "images", "style_sample.jpg")
    im = Image.open(sample).convert("RGB").resize((224, 224), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = wrapper(x)
    out = y[0].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(HERE, "starry_night_test.png"))
    print("test image:", os.path.join(HERE, "starry_night_test.png"))


if __name__ == "__main__":
    main()
