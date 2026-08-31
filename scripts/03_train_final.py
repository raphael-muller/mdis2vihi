"""Train the delivered model.

No arguments: everything is fixed here, because this script produces the checkpoint the
mosaic is generated from.

  INPUT   the 8 MDIS I/F bands plus band 9, standardised on the training split only.
          Band 9 is the count of 8-colour image sets averaged into the mosaic pixel
          (docs/DATA.md), so it is an instrumental covariate rather than a surface
          quantity: it proxies the pixel's signal-to-noise, since the mosaic's sigma
          planes follow 1 / sqrt(count), and it marks the acquisition regime, being a
          smooth function of latitude (r = -0.49) and nearly independent of the geometry
          of the MASCS spectrum it is paired with (r = 0.17).

          No observing angle enters the input, and none can: the MDIS mosaic has carried
          no angle backplane since MDR v3 (docs/DATA.md section 1), so an angle exists
          only where MASCS pointed while inference sweeps every pixel. The MASCS angles
          were nonetheless screened as inputs, on the five folds and on this exact
          protocol, to know what they were worth: seven input variants, each fold
          validating on itself and scoring on the same held-out test split, deltas read
          against their own fold-to-fold dispersion. The emission angle alone is neutral
          (CCC +0.0088 against a fold noise of 0.0097), while the three angles together
          are the strongest lever ever measured here (MSE over the k-NN floor 1.449 ->
          1.361, CCC +0.0397 = 5.7 times its noise, SAM improving 5/5 folds, 27 parameters
          out of 27 gaining on both CCC and OLS slope) and are still not adoptable, for the
          reason above. Three quarters of that gain come from the phase angle, which is
          99.8 % between-observation variance, so what the network learns there is largely
          a per-observation photometric recalibration rather than per-pixel physics. Read
          it as an oracle bounding what a deployable geometry proxy could buy.
  TARGET  `lsf_5p0` from `data/processed/virs_lsf_target.parquet`.

SpectralMLP (128, 256, 256) with GELU, NaN-tolerant MSE, Adam 1e-3,
ReduceLROnPlateau(patience 5, factor 0.5), EarlyStopping(patience 15), batch 1024,
validation on fold 0, test split kept aside, deterministic kernels and a fixed seed. These
values come from a hyperparameter search whose record is not part of this repository.

Adam rather than AdamW is not a decision: the weight decay is 0, and at zero decay the two
optimisers are the same update, since AdamW differs from Adam only by taking the decay out
of the gradient. Decay was left at 0 because the network is small (159 463 parameters) for
112 k training pairs and shows no overfitting to regularise away: the test split sits about
6 % above the training one (`runs/final/eval/final_per_split.csv`).

The three roles read out of `data/processed/splits.parquet` are disjoint by `obs_id` and
are not interchangeable. `test` is 10 % of the observations, drawn once at the pairs stage
(`scripts/01_build_pairs.py`) and never rotated: it enters no fold, contributes no
gradient, and is not what the early stop or the learning-rate schedule read. The five
folds are a *validation* rotation over the remaining 90 %, so the training set here is
folds 1..4 and the validation set is fold 0. Only that one fold is used because the
delivered product is one model, and which fold it validates on was measured not to matter:
retraining the same configuration on the five folds gives a fold-to-fold spread of about
0.1 %. Screening is a different matter and is always run
on the five folds, because a fold-0-only result once flipped sign across the other four
(the relaxed quality filter). Changing the fold changes the validation set, never the test
set: there is exactly one test split in this project, and no result anywhere in it comes
from a different one.

`runs/final/image_count_stats.json` carries the band-9 mean and standard deviation and is
**required** at inference.

Run:    python scripts/03_train_final.py    (111 907 training pairs; the delivered run
        stopped at epoch 93 of at most 100, GPU or CPU)
Writes: runs/final/lightning_logs/.../checkpoints/*.ckpt,
        runs/final/image_count_stats.json, runs/final/eval/final_test_metrics.json
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
import pyarrow.parquet as pq
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
STATS_JSON = RUN_DIR / "image_count_stats.json"

SEED = 42
HIDDEN = (128, 256, 256)
TARGET_COL = "lsf_5p0"          # the adopted LSF target
BATCH = 1024
MAX_EPOCHS = 100


COUNT_COL = "mdis_image_count"
COUNT_COL_LEGACY = "mdis_emission"


def pairs_count_column(path: Path = PAIRS) -> str:
    """Name of the band-9 column in `pairs.parquet`.

    It was renamed `mdis_emission` -> `mdis_image_count` on 2026-08-21, when band 9 was
    identified as the count of 8-colour image sets and not an emission angle. Only the name
    changed, so a table built before that date holds the same numbers and is read here
    under either name; `scripts/01_build_pairs.py` only ever writes the current one.
    """
    names = set(pq.ParquetFile(path).schema_arrow.names)
    for c in (COUNT_COL, COUNT_COL_LEGACY):
        if c in names:
            return c
    raise SystemExit(f"{path} carries neither {COUNT_COL} nor {COUNT_COL_LEGACY}: "
                     "rebuild it with scripts/01_build_pairs.py.")


def load_splits():
    """The three disjoint roles: fold 0 validates, folds 1..4 train, `test` is the held-out
    10 % of observations. Whole observations move together, so no MASCS spot in one role
    sits a few kilometres from its own neighbour in another."""
    ccol = pairs_count_column()
    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "mdis_iof", ccol])
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
            C=d[ccol].to_numpy(np.float32)[:, None],
            Y=np.stack(d[TARGET_COL].to_list()).astype(np.float32),
        )
    return out


def build9(data, cmean, cstd):
    """8 raw I/F + the z-standardized band-9 image count -> (N, 9)."""
    return {
        s: np.concatenate([data[s]["X"], (data[s]["C"] - cmean) / cstd], axis=1).astype(np.float32)
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
    cmean = float(data["train"]["C"].mean())
    cstd = float(data["train"]["C"].std() + 1e-8)
    X9 = build9(data, cmean, cstd)
    print(f"train {len(X9['train'])}  val {len(X9['val'])}  test {len(X9['test'])}  "
          f"| image-count z-stats: mean={cmean:.4f} std={cstd:.4f}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X9["train"]), torch.from_numpy(data["train"]["Y"]))
    val_ds = TensorDataset(torch.from_numpy(X9["val"]), torch.from_numpy(data["val"]["Y"]))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    model = SpectralMLP(in_features=9, out_features=231, hidden=HIDDEN, activation="gelu")
    # weight_decay = 0, so Adam and AdamW are the same update here: they differ only in
    # where the decay is applied. See the header for why no decay is used.
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
        "image_count_mean": cmean, "image_count_std": cstd,
        "in_features": 9, "hidden": list(HIDDEN), "activation": "gelu",
        "target_col": TARGET_COL, "seed": SEED, "best_ckpt": best,
        "band9_meaning": "count of 8-colour image sets at the pixel "
                         "(PDS MDR v3+ backplane a), see docs/DATA.md",
        "note": "input = 8 raw MDIS I/F + (band 9 - image_count_mean)/image_count_std; "
                "inference: scripts/04_predict_mosaic.py --count-band 9 --count-stats this.json",
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
    pan["image_count_mean"] = cmean
    pan["image_count_std"] = cstd
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