"""Training side of the hollow correction.

`layer.py` loads a correction network that is already trained and applies it to the
mosaic. This module is the other half: the pieces needed to produce that network in the
first place, so the second form of the deliverable can be rebuilt from the input data
rather than taken on trust.

    output = base_model(x) + strength * [ c(x) @ B.T ]

The **base model** is the delivered model. It is never retrained here, so wherever the
correction strength is zero the output is the deliverable itself, by construction. `B`
holds the first `rank` spectral shapes of `target - base_model` measured on the hollow
pairs of step 7, and does not change during training; only the small network `c`, which sets how
much of each shape to add, is learned. Restricting the correction to a couple of fixed
shapes is what stopped it from spending 231 degrees of freedom on artefacts that varied
from one random seed to the next.

`c(x) @ B.T` is the basis-function form of spectral super-resolution (He et al. 2023,
Inf. Fusion, eq. 4) applied to the residual rather than to the spectrum; restricting a
multivariate response to a fixed low-rank subspace is reduced-rank regression (Anderson
1951; Izenman 1975).

The strength is supplied as a **label** during training (1 on the hollow pairs, 0 on the
background ones) and read from a spatial catalogue at inference. Nothing decides it from the
spectrum: a hollow and a facula are not separable in 8 MDIS bands.

Checkpoints written through `CorrectionTrainingModule` load back with
`layer.CorrectionNetwork.from_checkpoint`, same `model.residual.` prefix.
"""
from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from ..lit.spectral_module import SpectralLitModule
from ..models.mlp import SpectralMLP


def load_base_model(ckpt_path, in_features=9, out_features=231, hidden=(128, 256, 256)):
    """The delivered model, with every parameter locked so it cannot be retrained."""
    lit = SpectralLitModule.load_from_checkpoint(
        str(ckpt_path),
        model=SpectralMLP(in_features=in_features, out_features=out_features, hidden=hidden),
        map_location="cpu")
    base = lit.model
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    return base


def correction_basis(base_model, x9: np.ndarray, y: np.ndarray, rank: int = 2):
    """Spectral shapes of `target - base_model`, on the rows that are finite everywhere.

    Returns `(B, report)` with `B` of shape (n_bands, rank). The decomposition is
    deliberately **not** centred: the mean correction is itself part of the shape to
    reproduce, so removing it would throw away most of mode 1.

    Rank 2 is the default because the singular values give 91.2 % of the residual
    variance to mode 1, 4.1 % to mode 2 and only 0.5 % to mode 3. The decomposition runs
    on the rows finite at every band (1 827 of 3 627, recorded in `report`): the far NIR
    tail of a VIRS spectrum carries gaps, and a row with one missing band cannot enter an
    SVD. Rank 2 therefore holds 95.4 % of the variance of what the correction has to
    reproduce, and the third mode is at the level of the noise.
    """
    with torch.no_grad():
        a = base_model(torch.from_numpy(np.ascontiguousarray(x9, np.float32))).numpy().astype(np.float64)
    r = np.asarray(y, np.float64) - a
    full = np.all(np.isfinite(r), axis=1)
    r = r[full]
    _u, s, vt = np.linalg.svd(r, full_matrices=False)
    ev = s ** 2
    ratio = ev / ev.sum()
    report = {"n_rows_total": int(len(full)), "n_rows_used": int(full.sum()),
              "explained_variance_ratio": [float(x) for x in ratio[:max(rank, 5)]],
              "cumulative_at_rank": float(np.cumsum(ratio)[rank - 1]),
              "median_residual_norm": float(np.median(np.linalg.norm(r, axis=1)))}
    return vt[:rank].T.astype(np.float32), report


class LowRankCorrection(nn.Module):
    """`r(x) = c(x) @ B.T`: fixed spectral shapes, input-dependent amplitudes only."""

    def __init__(self, basis, in_features=9, coef_hidden=(32,), learn_basis=False):
        super().__init__()
        b = torch.as_tensor(np.asarray(basis), dtype=torch.float32)      # (n_bands, rank)
        if learn_basis:
            self.B = nn.Parameter(b.clone())
        else:
            self.register_buffer("B", b)
        layers, d = [], in_features
        for h in coef_hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, b.shape[1])]
        self.coef = nn.Sequential(*layers)

    def forward(self, x):
        return self.coef(x) @ self.B.t()


class ModelPlusCorrection(nn.Module):
    """The delivered model plus a correction, scaled by `strength`. `residual_module`
    is plugged in, so the same wrapper serves a full-size correction and a shape-restricted
    one."""

    def __init__(self, base_model, in_features=9, out_features=231,
                 res_hidden=(128, 128), residual_module=None):
        super().__init__()
        self.base_model = base_model
        self.residual = (residual_module if residual_module is not None
                         else SpectralMLP(in_features, out_features,
                                          hidden=tuple(res_hidden), activation="gelu"))

    def forward(self, x, strength):
        self.base_model.eval()
        with torch.no_grad():
            a = self.base_model(x)
        g = strength.reshape(-1, 1).to(x.dtype)
        return a + g * self.residual(x)


class CorrectionTrainingModule(L.LightningModule):
    """NaN-tolerant MSE on the corrected output. The strength being a label, the network
    only receives a gradient where it is meant to act; nothing is learned on the
    background."""

    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    @staticmethod
    def _masked_mse(yhat, y):
        m = torch.isfinite(y)
        return ((yhat[m] - y[m]) ** 2).mean()

    def training_step(self, batch, _):
        x, y, strength = batch
        loss = self._masked_mse(self.model(x, strength=strength), y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y, strength = batch
        self.log("val/loss", self._masked_mse(self.model(x, strength=strength), y), prog_bar=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"}}