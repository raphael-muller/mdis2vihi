"""Pixel-wise MLP for MDIS 8-band -> VIHI hyperspectral upsampling.

Pure `nn.Module`: no Lightning, no loss, no optimizer. The training wrapper
lives in `mdis2vihi.lit.spectral_module.SpectralLitModule`.
"""

from __future__ import annotations

import torch.nn as nn


class SpectralMLP(nn.Module):
    """Fully-connected spectrum-to-spectrum mapper.

    Default shape `[8 -> 128 -> 256 -> 256 -> 231]` maps one MDIS I/F vector
    (8 bands at 433–996 nm) to a 5 nm × [300, 1450] nm grid (231 bins).
    """

    def __init__(
        self,
        in_features: int = 8,
        out_features: int = 231,
        hidden: tuple[int, ...] = (128, 256, 256),
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}[activation.lower()]
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
