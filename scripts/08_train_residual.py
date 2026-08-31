"""Train the correction network of the hollow-correction layer.

Produces the checkpoint that `scripts/tools/build_hollow_correction.py` combines with the
spatial selection to make the layer.

    output = base_model(x) + strength * [ c(x) @ B.T ]

The base model is the delivered one and is never retrained, so the background is untouched
by construction. `B` holds the first `--rank` spectral shapes of `target - base model` over
the hollow pairs of step 7 and is fixed; only the small network `c` is learned.

Which base model, and why that one. `BASE_MODEL_CKPT` below is the checkpoint written by
`scripts/03_train_final.py`, the single global network the whole 231-band mosaic is
generated from: it is the model for every spectrum, not a hollow-specific one. Nothing here
retrains it. `load_base_model` sets `requires_grad = False` on all of its parameters and
puts it in eval mode, so the gradient reaches only `c`, and the layer is defined as a
correction *of that exact file*. Pointing this at another checkpoint would make the
resulting layer inconsistent with the mosaic it is added to, since `output = anchor + g *
[c(x) @ B.T]` only holds for the anchor the residual was fitted against.

The tables read here are the cached ones: pairs, splits, target and the step-7 hollow pairs
are all parquet with fixed-length arrays on the implicit 5 nm grid. The 30 GB `spectres-002.dat`
and its per-row wavelength lists are read only by steps 1, 2 and 6, once each.

Two points that matter for the result:

* the strength is a **label** (1 on the hollow pairs, 0 on the background ones), never learned from
  the spectrum, because a hollow and a facula are not separable in 8 MDIS bands. At
  inference it comes from a spatial catalogue;
* **validation runs on hollow pairs kept aside by footprint**, at full strength. Step 7
  emits three rows per footprint, so a row-wise split leaves the same footprint on both
  sides; validating on the background instead leaves `val/loss` almost flat. Those pairs
  carry the measured MDIS vector, so `val/loss` is read in the same input space inference
  runs in, and a held-out footprint is a footprint whose pixel the residual has never seen.

Writes, under `--out`: the checkpoint, `residual_basis_rank<k>.npz`, `residual_stats.json`
and the training metrics.

Usage
-----
  python scripts/08_train_residual.py
  python scripts/08_train_residual.py --validate runs/final/correction/residual_rank2.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # CUDA determinism

import lightning as L
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.correction.residual import (ModelPlusCorrection, LowRankCorrection,  # noqa: E402
                                         CorrectionTrainingModule, load_base_model,
                                         correction_basis)

# The delivered checkpoint of scripts/03_train_final.py (best epoch 93), frozen. Not a
# hollow-specific model: it is the one the whole mosaic is predicted from.
BASE_MODEL_CKPT = REPO / "runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt"
BASE_MODEL_STATS = REPO / "runs/final/image_count_stats.json"
SIMPAIRS = REPO / "data/processed/simpairs_S2.parquet"
PAIRS = REPO / "data/processed/pairs.parquet"
SPLITS = REPO / "data/processed/splits.parquet"
TARGET = REPO / "data/processed/virs_lsf_target.parquet"
OUT_DEFAULT = REPO / "runs/residual"
TARGET_COL = "lsf_5p0"
BATCH, MAX_EPOCHS = 1024, 100
GRID = np.arange(300.0, 1455.0, 5.0)


def to9(x8, count, mean, std):
    e = np.asarray(count, np.float32).reshape(-1, 1)
    return np.concatenate([x8, (e - mean) / std], axis=1).astype(np.float32)


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


def count_column_of(df: pd.DataFrame) -> str:
    """Same resolution as `pairs_count_column`, on a table already in memory."""
    for c in (COUNT_COL, COUNT_COL_LEGACY):
        if c in df.columns:
            return c
    raise SystemExit(f"no {COUNT_COL} column: rebuild with scripts/07_build_simpairs.py.")


def load_real(val_fold, mean, std):
    """Real training pairs: the background the residual must leave alone."""
    ccol = pairs_count_column()
    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "mdis_iof", ccol])
    lsf = pd.read_parquet(TARGET, columns=["ref_id", TARGET_COL])
    splits = pd.read_parquet(SPLITS)
    df = pairs.merge(lsf, on="ref_id").merge(splits, on="ref_id")
    df = df[~df.split.isin([val_fold, "test"])]
    x8 = np.stack(df.mdis_iof.to_list()).astype(np.float32)
    y = np.stack(df[TARGET_COL].to_list()).astype(np.float32)
    return to9(x8, df[ccol].to_numpy(), mean, std), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpairs", default=str(SIMPAIRS))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--coef-hidden", default="32")
    ap.add_argument("--syn-frac", type=float, default=0.30,
                    help="share of the sampled batch mass taken by the hollow pairs")
    ap.add_argument("--syn-val-frac", type=float, default=0.15,
                    help="share of the hollow footprints held out for validation")
    ap.add_argument("--val-fold", default="fold0")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--validate", metavar="CKPT",
                    help="reference RESIDUAL checkpoint this run is reproduced against, "
                         "normally the delivered one, "
                         "runs/final/correction/residual_rank2.ckpt. It is not the base "
                         "model and not a held-out score: the comparison is a "
                         "reproducibility check on the residual (basis alignment and "
                         "correlation of the correction over the hollow pairs). The "
                         "held-out score of this run is val/loss, on the hollow "
                         "footprints kept aside by --syn-val-frac.")
    args = ap.parse_args()

    coef_hidden = tuple(int(v) for v in args.coef_hidden.split(",") if v)
    out = Path(args.out).resolve()   # absolute: residual_stats.json stores repo-relative paths
    out.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)

    stats = json.loads(BASE_MODEL_STATS.read_text(encoding="utf-8"))
    cmean, cstd = float(stats["image_count_mean"]), float(stats["image_count_std"])
    base_model = load_base_model(BASE_MODEL_CKPT)

    sim = pd.read_parquet(args.simpairs)
    if "ref_id" not in sim.columns:
        raise SystemExit(
            f"{args.simpairs} carries no ref_id, so train and validation cannot be "
            "separated by footprint, rebuild it with scripts/07_build_simpairs.py.")
    x_sim = to9(np.stack(sim.mdis_iof.to_list()).astype(np.float32),
                sim[count_column_of(sim)].to_numpy(), cmean, cstd)
    y_sim = np.stack(sim[TARGET_COL].to_list()).astype(np.float32)
    sim_ref = sim.ref_id.to_numpy()
    print(f"hollow pairs: {len(x_sim)} from {len(np.unique(sim_ref))} footprints",
          flush=True)

    basis, report = correction_basis(base_model, x_sim, y_sim, rank=args.rank)
    print(f"basis: rank {args.rank}, {report['n_rows_used']}/{report['n_rows_total']} "
          f"complete rows, cumulative explained variance "
          f"{report['cumulative_at_rank']*100:.1f} %", flush=True)
    np.savez(out / f"residual_basis_rank{args.rank}.npz", B=basis, wavelength_nm=GRID)

    x_real, y_real = load_real(args.val_fold, cmean, cstd)
    print(f"real training pairs: {len(x_real):,}", flush=True)

    rng = np.random.default_rng(args.seed)
    uref = np.unique(sim_ref)
    rng.shuffle(uref)
    n_val_ref = max(1, int(round(args.syn_val_frac * len(uref))))
    val_ref = set(uref[:n_val_ref].tolist())
    is_val = np.array([r in val_ref for r in sim_ref])
    v_idx, t_idx = np.where(is_val)[0], np.where(~is_val)[0]
    print(f"validation footprints: {n_val_ref}/{len(uref)} "
          f"({len(v_idx)} pairs), no footprint on both sides", flush=True)

    x_train = np.concatenate([x_real, x_sim[t_idx]])
    y_train = np.concatenate([y_real, y_sim[t_idx]])
    strength = np.concatenate([np.zeros(len(x_real), np.float32),
                           np.ones(len(t_idx), np.float32)])
    w_syn = ((args.syn_frac * len(x_real)) / ((1.0 - args.syn_frac) * len(t_idx))
             if len(t_idx) else 1.0)
    weights = np.where(strength > 0.5, w_syn, 1.0).astype(np.float64)
    print(f"train {len(strength):,} (background {len(x_real):,} + hollow {len(t_idx):,}), "
          f"hollow weight {w_syn:.2f} | validation {len(v_idx)} hollow pairs",
          flush=True)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train),
                      torch.from_numpy(strength)),
        batch_size=BATCH,
        sampler=WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                      num_samples=len(weights), replacement=True))
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(x_sim[v_idx]), torch.from_numpy(y_sim[v_idx]),
                      torch.ones(len(v_idx))),
        batch_size=BATCH, shuffle=False)

    model = ModelPlusCorrection(base_model, in_features=9, out_features=len(GRID),
                          residual_module=LowRankCorrection(basis, in_features=9,
                                                          coef_hidden=coef_hidden))
    lit = CorrectionTrainingModule(model, lr=1e-3)
    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices="auto",
                        default_root_dir=str(out), deterministic=True,
                        log_every_n_steps=200, enable_progress_bar=False,
                        callbacks=[EarlyStopping(monitor="val/loss", patience=15, mode="min"),
                                   ckpt_cb])
    t0 = time.time()
    trainer.fit(lit, train_dl, val_dl)
    best = Path(ckpt_cb.best_model_path).resolve()
    print(f"best checkpoint: {best}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # Copy the best epoch to a stable name at the root of --out. Lightning names the file
    # after the epoch it happened to stop on, while the layer builder needs one path that
    # does not move between runs; `residual_stats.json` points here, not into lightning_logs.
    promoted = out / f"residual_rank{args.rank}.ckpt"
    promoted.write_bytes(best.read_bytes())
    print(f"promoted to: {promoted}", flush=True)

    (out / "residual_stats.json").write_text(json.dumps({
        "_comment": "Rank-constrained residual of the hollow-correction layer, "
                    "trained by scripts/08_train_residual.py. Read by "
                    "scripts/tools/build_hollow_correction.py.",
        "image_count_mean": cmean, "image_count_std": cstd,
        "best_ckpt": promoted.relative_to(REPO).as_posix(),
        "best_epoch_ckpt": best.relative_to(REPO).as_posix(),
        "seed": args.seed, "rank": args.rank, "coef_hidden": list(coef_hidden),
        "res_hidden": [128, 128], "learn_basis": False, "val_fold": args.val_fold,
        "syn_frac": args.syn_frac, "syn_val_frac": args.syn_val_frac,
        "simpairs": str(Path(args.simpairs).name),
        "base_model_ckpt": BASE_MODEL_CKPT.relative_to(REPO).as_posix(),
        "basis_report": report,
        "arch": "low-rank residual on measured MDIS inputs, spatial catalogue selection at inference",
    }, indent=2), encoding="utf-8")

    if args.validate:
        # `args.validate` is the reference RESIDUAL checkpoint, i.e. the delivered
        # runs/final/correction/residual_rank2.ckpt, not the base model of BASE_MODEL_CKPT.
        # This block answers "does this run reproduce the delivered residual", which is a
        # reproducibility check; the quality of the run is val/loss, measured above on the
        # hollow footprints held out by footprint.
        from mdis2vihi.correction.layer import CorrectionNetwork
        ref = CorrectionNetwork.from_checkpoint(args.validate)
        new = CorrectionNetwork.from_checkpoint(best)
        b_ref = ref.B.numpy().astype(np.float64)
        b_new = new.B.numpy().astype(np.float64)
        # a basis is defined up to the sign of each mode: compare the subspace it spans
        cos = [abs(float(np.dot(b_ref[:, k], b_new[:, k])
                        / (np.linalg.norm(b_ref[:, k]) * np.linalg.norm(b_new[:, k]))))
               for k in range(min(b_ref.shape[1], b_new.shape[1]))]
        g = np.ones(len(x_sim), np.float32)
        d_ref = ref.delta(ref.coefficients(x_sim), g)
        d_new = new.delta(new.coefficients(x_sim), g)
        rep = {"reference": str(args.validate),
               "basis_mode_alignment": [round(c, 6) for c in cos],
               "correction_correlation": float(np.corrcoef(d_ref.ravel(), d_new.ravel())[0, 1]),
               "correction_median_abs_reference": float(np.median(np.abs(d_ref))),
               "correction_median_abs_new": float(np.median(np.abs(d_new))),
               "correction_max_abs_diff": float(np.max(np.abs(d_ref - d_new))),
               "correction_rms_diff": float(np.sqrt(np.mean((d_ref - d_new) ** 2)))}
        print("\nvalidation against the delivered residual")
        print(json.dumps(rep, indent=2))
        (out / "residual_validation.json").write_text(json.dumps(rep, indent=2),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()