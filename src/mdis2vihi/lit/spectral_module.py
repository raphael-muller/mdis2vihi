"""LightningModule wrapping a pixel-wise spectral MLP.

Loss: NaN-tolerant MSE on the resampled 5 nm grid. `photom_iof_5nm` may carry
NaN at the spectral extremes for spectra that don't cover [300, 1450] nm, and
those positions are masked out of the loss instead of dropping the sample.
SAM (Spectral Angle Mapper) is logged as a diagnostic on validation.

Optimizer: Adam + ReduceLROnPlateau on `val/loss`.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn


class SpectralLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _masked_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        if not mask.any():
            return torch.zeros((), device=y_pred.device, requires_grad=True)
        diff = y_pred[mask] - y_true[mask]
        return (diff * diff).mean()

    @staticmethod
    def _sam_deg(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        yp = torch.where(mask, y_pred, torch.zeros_like(y_pred))
        yt = torch.where(mask, y_true, torch.zeros_like(y_true))
        dot = (yp * yt).sum(dim=-1)
        norm = yp.norm(dim=-1) * yt.norm(dim=-1)
        cos = (dot / (norm + 1e-12)).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(cos)).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._masked_mse(self(x), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self._masked_mse(y_pred, y)
        sam = self._sam_deg(y_pred, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sam_deg", sam, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.log("test/loss", self._masked_mse(y_pred, y), on_epoch=True)
        self.log("test/sam_deg", self._sam_deg(y_pred, y), on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"},
        }