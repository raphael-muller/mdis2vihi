"""Train the delivered model.

No arguments: everything is fixed here, because this script produces the checkpoint the
mosaic is generated from.

  INPUT   the 8 MDIS I/F bands plus the emission angle (band 9), standardised on the
          training split only. Band 9 is the observing geometry of the mosaic pixel: it
          is a smooth function of latitude (r = -0.49) and nearly independent of the
          geometry of the MASCS spectrum it is paired with (r = 0.17), so what it supplies
          is a regional covariate. The gain it brings is empirical.
  TARGET  `lsf_5p0` from `data/processed/virs_lsf_target.parquet`.

SpectralMLP (128, 256, 256) with GELU, NaN-tolerant MSE, Adam 1e-3,
ReduceLROnPlateau(patience 5, factor 0.5), EarlyStopping(patience 15), batch 1024,
validation on fold 0, test split kept aside, deterministic kernels and a fixed seed. These
values come from a hyperparameter search whose record is not part of this repository.

`runs/final/emission_stats.json` carries the emission mean and standard deviation and is
**required** at inference.

Run:    python scripts/03_train_final.py    (111 907 training pairs; the delivered run
        stopped at epoch 93 of at most 100, GPU or CPU)
Writes: runs/final/lightning_logs/.../checkpoints/*.ckpt,
        runs/final/emission_stats.json, runs/final/eval/final_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # CUDA determinism

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")
from mdis2vihi.models.mlp import SpectralMLP  # noqa: E402
from mdis2vihi.lit.spectral_module import SpectralLitModule  # noqa: E402
from mdis2vihi.eval.metrics import metric_set, per_band, param_fidelity, knn_floor  # noqa: E402

PAIRS = REPO_ROOT / "data/processed/pairs.parquet"
SPLITS = REPO_ROOT / "data/processed/splits.parquet"
LSF = REPO_ROOT / "data/processed/virs_lsf_target.parquet"
RUN_DIR = REPO_ROOT / "runs/final"
EVAL_DIR = RUN_DIR / "eval"
STATS_JSON = RUN_DIR / "emission_stats.json"

SEED = 42
HIDDEN = (128, 256, 256)
TARGET_COL = "lsf_5p0"          # the adopted LSF target
BATCH = 1024
MAX_EPOCHS = 100


def load_splits():
    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "mdis_iof", "mdis_emission"])
    lsf = pd.read_parquet(LSF, columns=["ref_id", TARGET_COL])
    splits = pd.read_parquet(SPLITS)
    df = pairs.merge(lsf, on="ref_id").merge(splits, on="ref_id", how="inner")
    sel = {
        "train": ~df.split.isin(["fold0", "test"]),
        "val": df.split == "fold0",
        "test": df.split == "test",
    }
    out = {}
    for name, m in sel.items():
        d = df[m]
        out[name] = dict(
            X=np.stack(d.mdis_iof.to_list()).astype(np.float32),
            E=d.mdis_emission.to_numpy(np.float32)[:, None],
            Y=np.stack(d[TARGET_COL].to_list()).astype(np.float32),
        )
    return out


def build9(data, emean, estd):
    """8 raw I/F + z-standardized emission -> (N, 9)."""
    return {
        s: np.concatenate([data[s]["X"], (data[s]["E"] - emean) / estd], axis=1).astype(np.float32)
        for s in data
    }


def require_inputs():
    """Fail early and legibly if the training tables are not on disk."""
    missing = [p for p in (PAIRS, SPLITS, LSF) if not p.exists()]
    if missing:
        rel = "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        raise SystemExit(
            f"Missing input(s):\n  {rel}\n\n"
            "These are produced by scripts/01_build_pairs.py and\n"
            "scripts/02_build_lsf_target.py --build, from data that is not shipped\n"
            "with the repository. See docs/DATA.md and docs/REPRODUCTION.md.")


def main():
    # No options: this script produces the delivered checkpoint, so everything is fixed
    # above. The parser is here only so that `--help` prints the header and a mistyped
    # argument is refused, instead of silently starting a training run.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    require_inputs()
    L.seed_everything(SEED, workers=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_splits()
    emean = float(data["train"]["E"].mean())
    estd = float(data["train"]["E"].std() + 1e-8)
    X9 = build9(data, emean, estd)
    print(f"train {len(X9['train'])}  val {len(X9['val'])}  test {len(X9['test'])}  "
          f"| emission z-stats: mean={emean:.4f} std={estd:.4f}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X9["train"]), torch.from_numpy(data["train"]["Y"]))
    val_ds = TensorDataset(torch.from_numpy(X9["val"]), torch.from_numpy(data["val"]["Y"]))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    model = SpectralMLP(in_features=9, out_features=231, hidden=HIDDEN, activation="gelu")
    lit = SpectralLitModule(model, lr=1e-3, weight_decay=0.0,
                            scheduler_patience=5, scheduler_factor=0.5)

    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True)
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto", devices="auto",
        default_root_dir=str(RUN_DIR), deterministic=True, log_every_n_steps=50,
        callbacks=[EarlyStopping(monitor="val/loss", patience=15, mode="min"),
                   ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(lit, train_dl, val_dl)
    best = ckpt_cb.best_model_path
    print(f"best checkpoint: {best}", flush=True)

    STATS_JSON.write_text(json.dumps({
        "emission_mean": emean, "emission_std": estd,
        "in_features": 9, "hidden": list(HIDDEN), "activation": "gelu",
        "target_col": TARGET_COL, "seed": SEED, "best_ckpt": best,
        "note": "input = 8 raw MDIS I/F + (emission - emission_mean)/emission_std; "
                "inference: scripts/04_predict_mosaic.py --emission-band 9 --emission-stats this.json",
    }, indent=2), encoding="utf-8")
    print(f"wrote {STATS_JSON}", flush=True)

    # ---- full test audit ----
    best_lit = SpectralLitModule.load_from_checkpoint(
        best, model=SpectralMLP(in_features=9, out_features=231, hidden=HIDDEN, activation="gelu"),
        map_location="cpu")
    best_lit.eval()
    Xte = torch.from_numpy(X9["test"])
    with torch.no_grad():
        Yp = best_lit.model(Xte).numpy().astype(np.float64)
    Yt = data["test"]["Y"].astype(np.float64)

    pan = metric_set(Yp, Yt)
    floor = knn_floor(data["test"]["X"].astype(np.float64), Yt, k=5)  # 8-input floor (comparable)
    pan["knn_floor_k5"] = floor
    pan["mse"] = float(np.nanmean((Yp - Yt) ** 2))
    pan["mlp_over_floor"] = pan["mse"] / floor
    pan["n_test"] = int(len(Yt))
    pan["emission_mean"] = emean
    pan["emission_std"] = estd
    (EVAL_DIR / "final_test_metrics.json").write_text(json.dumps(pan, indent=2), encoding="utf-8")
    per_band(Yp, Yt).to_csv(EVAL_DIR / "final_per_band.csv", index=False)
    param_fidelity(Yp, Yt).to_csv(EVAL_DIR / "final_param_fidelity.csv", index=False)

    print("\n=== test metrics ===",
          flush=True)
    for k in ("mse", "knn_floor_k5", "mlp_over_floor", "rmse", "mae", "mrae",
              "sam_median", "sam_mean", "sga_median", "sid_mean"):
        print(f"  {k:16s} {pan[k]:.5g}", flush=True)
    print(f"wrote {EVAL_DIR}/final_test_metrics.json + per_band + param_fidelity", flush=True)


if __name__ == "__main__":
    main()