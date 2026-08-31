"""Test-side audit of the delivered model.

Recomputes, on the delivered checkpoint, the diagnostics that say how the error is
distributed rather than how large it is on average:

  * MSE and spectral angle per split (train / val / test)  -> final_per_split.csv
  * spectral-angle percentiles, SGA and SCM               -> final_diagnostics.json
  * spectral angle per observation                        -> final_per_obs.csv
  * spectral angle per terrain type                       -> final_per_terrain.csv
  * spectral angle against the MASCS quality flags        -> final_diagnostics.json
  * spectral angle against observing geometry             -> final_diagnostics.json
  * k-NN lower bound under six neighbourhood definitions  -> final_floor_variants.csv
  * per-parameter reliability ceiling and how much of it the model reaches
                                                          -> final_param_ceiling.csv
  * per-band and per-parameter fidelity (r, OLS slope, Lin's CCC, bias)
                                                          -> final_per_band.csv,
                                                             final_param_fidelity.csv

It also refreshes the metric panel inside `final_test_metrics.json`, merging into the
file rather than rewriting it, so the keys only `03_train_final.py` produces survive.

One caveat on the word "test". The split is a true hold-out in the sense that matters for
fitting: it is group-disjoint by `obs_id`, no gradient ever saw it, and neither the early
stop nor the learning-rate schedule reads it. It is not virgin in the model-selection
sense, because the screening campaign scored every candidate lever on this same split
rather than on a fresh one, which is repeated use of a hold-out and can flatter the number
it finally reports. Two things bound that risk. The published criterion is a *ratio* to
the k-NN floor recomputed on the same split, not an absolute error, so a split that
happens to be easy moves numerator and denominator together. And the error is ordered
val 4.287e-5 < train 4.376e-5 < test 4.650e-5 (`final_per_split.csv`): the test split
scores 8 % worse than the fold the model early-stopped on, which is the opposite of what
over-use of a hold-out produces.

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

# Datum sphere of the mosaic (2 439 400 m), in km: this is great-circle arithmetic on the
# sphere, not a projection, so it does not go through pyproj (see CLAUDE.md).
R_MERC_KM = 2439.4

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


def to_arrays(d, cmean, cstd):
    X8 = np.stack(d.mdis_iof.to_list()).astype(np.float32)
    C = d.mdis_image_count.to_numpy(np.float32)[:, None]
    X9 = np.concatenate([X8, (C - cmean) / cstd], axis=1).astype(np.float32)
    Y = np.stack(d[TARGET_COL].to_list()).astype(np.float32)
    return X8, X9, Y


def knn_floor_idx(idx, Y):
    return float(np.nanmean(np.nanvar(Y[idx], axis=1)))


def unit_sphere(lon_deg, lat_deg):
    """(lon, lat) in degrees -> 3-D coordinates on the unit sphere.

    A neighbour search on these coordinates ranks by chord length, which is monotone in
    the great-circle arc, so it returns the true geographic neighbours. Searching on
    (lon, lat) degrees directly does not: a degree of longitude is worth only cos(lat)
    degrees of latitude, and 359.9 deg sits next to 0.1 deg, not 359.8 deg away.
    """
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def knn_idx(Xq, Xref, k, drop_self, groups_q=None, groups_ref=None, block=2048):
    """k nearest neighbours of Xq among Xref (torch cdist), by blocks of query rows.

    The full distance matrix would be 13 GB for the test set against the training set, so
    the query is walked in blocks. `groups_q` / `groups_ref` forbid neighbours sharing a
    group label (used to ask what a neighbourhood looks like once same-observation
    neighbours are excluded).
    """
    A = torch.from_numpy(np.ascontiguousarray(Xq)).double()
    B = torch.from_numpy(np.ascontiguousarray(Xref)).double()
    kk = k + 1 if drop_self else k
    out = []
    for start in range(0, len(A), block):
        stop = min(start + block, len(A))
        D = torch.cdist(A[start:stop], B)
        if groups_q is not None:
            same = torch.from_numpy(groups_q[start:stop, None] == groups_ref[None, :])
            D[same] = float("inf")
        idx = D.topk(kk, largest=False).indices.numpy()
        out.append(idx[:, 1:] if drop_self else idx)
    return np.concatenate(out, axis=0)


def neighbourhood_stats(idx, lon, lat, obs, lon_ref=None, lat_ref=None, obs_ref=None):
    """Median great-circle distance to the k neighbours, and how many share the obs_id."""
    lon_ref = lon if lon_ref is None else lon_ref
    lat_ref = lat if lat_ref is None else lat_ref
    obs_ref = obs if obs_ref is None else obs_ref
    p1, p2 = np.radians(lat)[:, None], np.radians(lat_ref[idx])
    dlon = np.radians(lon_ref[idx] - lon[:, None])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    km = 2 * R_MERC_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(np.median(km)), float(np.mean(obs_ref[idx] == obs[:, None]))


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
    cmean, cstd = STATS["image_count_mean"], STATS["image_count_std"]

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
        X8, X9, Y = to_arrays(d, cmean, cstd)
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
    # The headline floor defines the neighbourhood **in input space**, and that is not a
    # convenience: the model is a deterministic function of the nine-band vector, so two
    # footprints with the same input necessarily receive the same prediction and the
    # spread of their targets is irreducible for it. A neighbourhood defined by ground
    # distance does not bound this model (two spots 15 km apart usually have different
    # MDIS inputs, which the model is free to map to different spectra); it measures
    # something else, namely how much of the target a spatially aware model could hope to
    # explain. Both are reported, with the median neighbour distance and the share of
    # neighbours coming from the same observation, which is what makes them readable.
    X8tr, _, Ytr, dtr = pred["train"]
    lon_te, lat_te = dt.lon_center.to_numpy(float), dt.lat_center.to_numpy(float)
    lon_tr, lat_tr = dtr.lon_center.to_numpy(float), dtr.lat_center.to_numpy(float)
    obs_te, obs_tr = dt.obs_id.to_numpy(), dtr.obs_id.to_numpy()
    sphere_te, sphere_tr = unit_sphere(lon_te, lat_te), unit_sphere(lon_tr, lat_tr)

    def add(tag, idx, Yref, lon_ref=None, lat_ref=None, obs_ref=None):
        f = knn_floor_idx(idx, Yref)
        km, share = neighbourhood_stats(idx, lon_te, lat_te, obs_te, lon_ref, lat_ref, obs_ref)
        rows.append((tag, f, mse_s.mean() / f, sam_floor_idx(idx, Yref), km, share))

    rows = []
    # input-space, neighbours inside the test split
    add("input_Xtest", knn_idx(X8t, X8t, 5, drop_self=True), Ytt)
    # input-space, same but never inside the same observation
    add("input_Xtest_crossobs",
        knn_idx(X8t, X8t, 5, drop_self=True, groups_q=obs_te, groups_ref=obs_te), Ytt)
    # input-space, neighbours taken from the training split
    add("input_Xtrain", knn_idx(X8t, X8tr, 5, drop_self=False), Ytr, lon_tr, lat_tr, obs_tr)
    # ground distance, neighbours inside the test split
    add("greatcircle_Xtest", knn_idx(sphere_te, sphere_te, 5, drop_self=True), Ytt)
    # ground distance, never inside the same observation: the same-track neighbours that
    # dominate the row above are what make it a repeatability measurement rather than a bound
    add("greatcircle_Xtest_crossobs",
        knn_idx(sphere_te, sphere_te, 5, drop_self=True, groups_q=obs_te, groups_ref=obs_te), Ytt)
    # ground distance, neighbours taken from the training split
    add("greatcircle_Xtrain", knn_idx(sphere_te, sphere_tr, 5, drop_self=False), Ytr,
        lon_tr, lat_tr, obs_tr)
    fv = pd.DataFrame(rows, columns=["neighbourhood", "mse_floor", "mlp_over_floor",
                                     "sam_floor", "median_km", "same_obs_share"])
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
