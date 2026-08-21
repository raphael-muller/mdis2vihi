"""Test-side audit of the delivered model.

Recomputes, on the delivered checkpoint, the diagnostics that say how the error is
distributed rather than how large it is on average:

  * MSE and spectral angle per split (train / val / test)  -> final_per_split.csv
  * spectral-angle percentiles, SGA and SCM               -> final_diagnostics.json
  * spectral angle per observation                        -> final_per_obs.csv
  * spectral angle per terrain type                       -> final_per_terrain.csv
  * spectral angle against the MASCS quality flags        -> final_diagnostics.json
  * spectral angle against observing geometry             -> final_diagnostics.json
  * k-NN lower bound under four neighbourhood definitions -> final_floor_variants.csv
  * per-parameter reliability ceiling and how much of it the model reaches
                                                          -> final_param_ceiling.csv
  * per-band and per-parameter fidelity (r, OLS slope, Lin's CCC, bias)
                                                          -> final_per_band.csv,
                                                             final_param_fidelity.csv

It also refreshes the metric panel inside `final_test_metrics.json`, merging into the
file rather than rewriting it, so the keys only `03_train_final.py` produces survive.

Run:    python scripts/05_eval_final.py
Writes: runs/final/eval/final_*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")
from mdis2vihi.models.mlp import SpectralMLP  # noqa: E402
from mdis2vihi.lit.spectral_module import SpectralLitModule  # noqa: E402
from mdis2vihi.eval.metrics import (sam_deg, sga_deg, pearson,  # noqa: E402
                                    metric_set, per_band, param_fidelity,
                                    reliability_ceiling)

PAIRS = REPO_ROOT / "data/processed/pairs.parquet"
SPLITS = REPO_ROOT / "data/processed/splits.parquet"
LSF = REPO_ROOT / "data/processed/virs_lsf_target.parquet"
MDISDATA = REPO_ROOT / "data/raw/mascs/Spectra_0_360_-90_90_MDISdata.dat"
QUALITY = REPO_ROOT / "data/raw/mascs/Spectra_0_360_-90_90_quality.dat"
RUN_DIR = REPO_ROOT / "runs/final"
EVAL_DIR = RUN_DIR / "eval"
STATS = json.loads((RUN_DIR / "image_count_stats.json").read_text(encoding="utf-8"))

TARGET_COL = "lsf_5p0"
HIDDEN = tuple(STATS["hidden"])


def scm_deg(Yp, Yt):
    """Spectral Correlation Mapper: the angle between mean-centred spectra, whose cosine
    is their Pearson correlation (de Carvalho & Meneses 2000). Insensitive to a level
    shift, so a high SAM with a low SCM means an amplitude error, not a shape error."""
    Pc = Yp - np.nanmean(Yp, axis=1, keepdims=True)
    Tc = Yt - np.nanmean(Yt, axis=1, keepdims=True)
    return sam_deg(Pc, Tc)


def load():
    pairs = pd.read_parquet(
        PAIRS, columns=["ref_id", "obs_id", "lat_center", "lon_center",
                        "ang_in", "ang_em", "ang_ph", "mdis_iof", "mdis_image_count"])
    lsf = pd.read_parquet(LSF, columns=["ref_id", TARGET_COL])
    splits = pd.read_parquet(SPLITS)
    df = pairs.merge(lsf, on="ref_id").merge(splits, on="ref_id", how="inner")
    return df


def to_arrays(d, emean, estd):
    X8 = np.stack(d.mdis_iof.to_list()).astype(np.float32)
    E = d.mdis_image_count.to_numpy(np.float32)[:, None]
    X9 = np.concatenate([X8, (E - emean) / estd], axis=1).astype(np.float32)
    Y = np.stack(d[TARGET_COL].to_list()).astype(np.float32)
    return X8, X9, Y


def knn_floor_idx(idx, Y):
    return float(np.nanmean(np.nanvar(Y[idx], axis=1)))


def knn_idx(Xq, Xref, k, drop_self):
    """k nearest neighbours of Xq among Xref (torch cdist)."""
    A = torch.from_numpy(np.ascontiguousarray(Xq)).double()
    B = torch.from_numpy(np.ascontiguousarray(Xref)).double()
    kk = k + 1 if drop_self else k
    idx = torch.cdist(A, B).topk(kk, largest=False).indices.numpy()
    return idx[:, 1:] if drop_self else idx


def sam_floor_idx(idx, Y):
    """Within-neighbourhood angular dispersion: mean spectral angle between each
    neighbour and the neighbourhood centroid. Angular counterpart of `knn_floor`, built
    by analogy rather than taken from a reference, since an angle has no variance
    decomposition, and k-dependent in the same way."""
    cen = np.nanmean(Y[idx], axis=1)                       # (N, B)
    out = []
    for j in range(idx.shape[1]):
        out.append(sam_deg(Y[idx[:, j]], cen))
    return float(np.median(np.mean(np.stack(out, 1), axis=1)))


def require_inputs():
    """Fail early and legibly if the pipeline inputs are not on disk."""
    missing = [p for p in (PAIRS, SPLITS, LSF, MDISDATA, QUALITY) if not p.exists()]
    if missing:
        rel = "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        raise SystemExit(
            f"Missing input(s):\n  {rel}\n\n"
            "The raw data is not shipped with the repository. See docs/DATA.md for\n"
            "provenance and the expected layout, then run 01_build_pairs.py and\n"
            "02_build_lsf_target.py --build first\n"
            "(docs/REPRODUCTION.md).")


def main():
    # No options: the audit is fully determined by the delivered checkpoint. The parser is
    # here only so that `--help` prints the header and a mistyped argument is refused,
    # instead of silently recomputing the whole panel.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    require_inputs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    emean, estd = STATS["image_count_mean"], STATS["image_count_std"]

    sub = {
        "train": df[~df.split.isin(["fold0", "test"])],
        "val": df[df.split == "fold0"],
        "test": df[df.split == "test"],
    }

    ckpt = STATS["best_ckpt"]
    lit = SpectralLitModule.load_from_checkpoint(
        ckpt, model=SpectralMLP(in_features=9, out_features=231, hidden=HIDDEN,
                                activation="gelu"), map_location="cpu")
    lit.eval()

    pred = {}
    per_split = []
    for name, d in sub.items():
        X8, X9, Y = to_arrays(d, emean, estd)
        with torch.no_grad():
            Yp = lit.model(torch.from_numpy(X9)).numpy().astype(np.float64)
        Yt = Y.astype(np.float64)
        pred[name] = (X8.astype(np.float64), Yp, Yt, d)
        mse = float(np.nanmean((Yp - Yt) ** 2))
        sam = sam_deg(Yp, Yt)
        per_split.append({"split": name, "n": len(d), "mse": mse,
                          "sam_median": float(np.median(sam)),
                          "sam_mean": float(np.mean(sam))})
    pd.DataFrame(per_split).to_csv(EVAL_DIR / "final_per_split.csv", index=False)

    # ---- test per-sample shape metrics ----
    X8t, Ypt, Ytt, dt = pred["test"]
    sam = sam_deg(Ypt, Ytt)
    scm = scm_deg(Ypt, Ytt)
    sga = sga_deg(Ypt, Ytt)
    mse_s = np.nanmean((Ypt - Ytt) ** 2, axis=1)

    diag = {
        "n_test": int(len(dt)),
        "sam_median": float(np.median(sam)),
        "sam_mean": float(np.mean(sam)),
        "sam_p95": float(np.percentile(sam, 95)),
        "sam_p99": float(np.percentile(sam, 99)),
        "sga_median": float(np.median(sga)),
        "scm_median": float(np.median(scm)),
        "pearson_sam_scm": pearson(sam, scm),
    }

    # ---- per obs_id ----
    tt = pd.DataFrame({"obs_id": dt.obs_id.values, "sam": sam, "mse": mse_s})
    g = tt.groupby("obs_id").agg(n=("sam", "size"), sam_median=("sam", "median"),
                                 sam_p95=("sam", lambda s: np.percentile(s, 95)))
    g.reset_index().to_csv(EVAL_DIR / "final_per_obs.csv", index=False)
    diag["per_obs_median_of_medians"] = float(g.sam_median.median())
    diag["per_obs_p95_of_medians"] = float(np.percentile(g.sam_median, 95))

    # ---- per terrain (MDISdata join) ----
    md = pd.read_csv(MDISDATA, usecols=["ref_id", "lrm", "sm_plain", "430_1000"])
    tt2 = pd.DataFrame({"ref_id": dt.ref_id.values, "sam": sam}).merge(md, on="ref_id", how="left")

    def terrain(r):
        if r.lrm == 1 and r.sm_plain == 1:
            return "LRM+smooth"
        if r.lrm == 1:
            return "LRM"
        if r.sm_plain == 1:
            return "smooth_plains"
        return "other"

    tt2["terrain"] = tt2.apply(terrain, axis=1)
    pt = (tt2.groupby("terrain").agg(n=("sam", "size"),
          sam_median=("sam", "median"),
          sam_p95=("sam", lambda s: np.percentile(s, 95)))
          .reset_index().sort_values("sam_median"))
    pt.to_csv(EVAL_DIR / "final_per_terrain.csv", index=False)

    # ---- SAM vs quality ----
    # q1..q4 are read straight from the MASCS quality table (the same file
    # 01_build_pairs.py filters on) so this audit has no dependency outside the
    # production chain. 336 MB CSV, 5 columns -> a few seconds.
    q = pd.read_csv(QUALITY, usecols=["ref_id", "q1", "q2", "q3", "q4"])
    tq = pd.DataFrame({"ref_id": dt.ref_id.values, "sam": sam}).merge(q, on="ref_id", how="left")
    diag["pearson_sam_quality"] = {qi: pearson(tq["sam"].values, tq[qi].values)
                                   for qi in ["q1", "q2", "q3", "q4"]}

    # ---- SAM vs geometry ----
    diag["pearson_sam_geometry"] = {
        "ang_in": pearson(sam, dt.ang_in.to_numpy(float)),
        "ang_em": pearson(sam, dt.ang_em.to_numpy(float)),
        "ang_ph": pearson(sam, dt.ang_ph.to_numpy(float)),
    }

    # ---- floor variants (LSF target) ----
    X8tr, _, Ytr, dtr = pred["train"]
    latlon_te = dt[["lon_center", "lat_center"]].to_numpy(float)
    latlon_tr = dtr[["lon_center", "lat_center"]].to_numpy(float)
    rows = []
    # input-space, X_test (self-drop)
    i = knn_idx(X8t, X8t, 5, drop_self=True)
    rows.append(("input_Xtest", knn_floor_idx(i, Ytt), mse_s.mean() / knn_floor_idx(i, Ytt),
                 sam_floor_idx(i, Ytt)))
    # input-space, X_train neighbours of test
    i = knn_idx(X8t, X8tr, 5, drop_self=False)
    f = knn_floor_idx(i, Ytr)
    rows.append(("input_Xtrain", f, mse_s.mean() / f, sam_floor_idx(i, Ytr)))
    # great-circle, X_test
    i = knn_idx(latlon_te, latlon_te, 5, drop_self=True)
    f = knn_floor_idx(i, Ytt)
    rows.append(("greatcircle_Xtest", f, mse_s.mean() / f, sam_floor_idx(i, Ytt)))
    # great-circle, X_train
    i = knn_idx(latlon_te, latlon_tr, 5, drop_self=False)
    f = knn_floor_idx(i, Ytr)
    rows.append(("greatcircle_Xtrain", f, mse_s.mean() / f, sam_floor_idx(i, Ytr)))
    fv = pd.DataFrame(rows, columns=["neighbourhood", "mse_floor", "mlp_over_floor", "sam_floor"])
    fv["mlp_over_sam_floor"] = float(np.median(sam)) / fv["sam_floor"]
    fv.to_csv(EVAL_DIR / "final_floor_variants.csv", index=False)

    (EVAL_DIR / "final_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # ---- per-band + per-parameter fidelity (r, OLS slope, Lin's CCC, bias) ----
    # Owned by the audit script rather than by 03_train_final.py, so the panel can be
    # refreshed without retraining. `final_test_metrics.json` is MERGED, not rewritten:
    # the keys 03 alone produces (knn_floor_k5, image_count_*) are preserved, while the
    # metric-panel keys are recomputed on this same test split.
    per_band(Ypt, Ytt).to_csv(EVAL_DIR / "final_per_band.csv", index=False)
    pf = param_fidelity(Ypt, Ytt)
    pf.to_csv(EVAL_DIR / "final_param_fidelity.csv", index=False)
    rc = reliability_ceiling(X8t, Ytt, Ypt)
    rc.to_csv(EVAL_DIR / "final_param_ceiling.csv", index=False)

    metrics_path = EVAL_DIR / "final_test_metrics.json"
    prev = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    mse_test = float(np.nanmean((Ypt - Ytt) ** 2))
    if "mse" in prev and abs(prev["mse"] - mse_test) > 1e-12 * max(1.0, abs(prev["mse"])):
        raise SystemExit(f"test split drift: stored mse {prev['mse']:.6e} != {mse_test:.6e} "
                         "refusing to merge panels computed on different data")
    prev.update(metric_set(Ypt, Ytt))
    prev["mse"] = mse_test
    if "knn_floor_k5" in prev:
        prev["mlp_over_floor"] = mse_test / prev["knn_floor_k5"]
    metrics_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")

    # ---- print summary ----
    print("\n=== per split (MSE / SAM med) ===")
    print(pd.DataFrame(per_split).to_string(index=False))
    print("\n=== diagnostics ===")
    print(json.dumps(diag, indent=2))
    print("\n=== per terrain ===")
    print(pt.to_string(index=False))
    print("\n=== floor variants (LSF target) ===")
    print(fv.to_string(index=False))
    print("\n=== metric panel (r vs ccc vs slope) ===")
    for k in ("r_key_balanced", "ccc_key_balanced", "slope_key_balanced"):
        print(f"  {k:20s} {prev[k]:.4f}")
    print("\n=== param fidelity (r blind to compression, ccc is not) ===")
    print(pf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n=== part du plafond informationnel atteinte ===")
    print(rc.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
