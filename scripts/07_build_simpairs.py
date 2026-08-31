"""Build the hollow training pairs from the footprints recovered by step 6.

Each pair is a real measurement on both sides: the input is the MDIS vector colocated under
the footprint, the target is that footprint's VIRS spectrum resampled to the 5 nm grid. The
pool carries both, written by step 6 as `mdis_iof` and `lsf_5p0`, so no part of the pair is
modelled.

That matters because the residual trained on these pairs is applied, at inference, to the
pixels of the MDIS mosaic. Feeding it the measured vector puts training and inference in the
same input space, and the two are not interchangeable on this pool: applied to the
footprints of the training set the per-band VIRS-to-WAC relation is unbiased (-0.4 % per
band, flat across the eight), but applied to hollow footprints the measured pixel comes out
+9.7 % brighter than that relation predicts, from +12.1 % at 433 nm down to +7.6 % at
996 nm, so it is bluer as well. Scored on held-out footprints with the mosaic's own vector,
a residual trained on measured inputs improves the prediction by 44 % of MSE and 0.27 deg of
spectral angle (3 seeds out of 3), and moves the background less than half as much when it
is wrongly gated.

Three rows per footprint go into one file: the footprint itself, and the same footprint
mixed with its nearest real training background at 1/3 and 2/3, since a mosaic pixel over a
hollow field is only partly filled. The mixture is linear and taken on both sides at once,

    x = f * x_hollow + (1 - f) * x_background      (measured MDIS, 8 bands)
    y = f * y_hollow + (1 - f) * y_background      (VIRS spectrum, 231 bins)

so a mixed row is still a pair of real measurements, and the residual learns how the
correction scales with the hollow fraction instead of seeing whole footprints only.
Backgrounds are drawn from the training folds, so the validation fold and the test split
stay out of the pairs. The image-set count carried with every row is the hollow footprint's
own, unmixed: it is an instrumental covariate of the pixel, not a surface quantity to
interpolate.

Nothing here is random, so a rerun on the same pool reproduces the file exactly. Every row
carries its `ref_id`, and three rows share one, so a split ignoring it would leave the same
footprint on both sides; `scripts/08_train_residual.py` groups on it.

Usage
-----
  python scripts/07_build_simpairs.py
  python scripts/07_build_simpairs.py --validate <reference pairs parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

R_KM = 2439.4
MIX_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)        # partial fill fractions of the hollow arm

POOL = REPO / "data/processed/hollow_pool.parquet"
PAIRS = REPO / "data/processed/pairs.parquet"
SPLITS = REPO / "data/processed/splits.parquet"
TARGET = REPO / "data/processed/virs_lsf_target.parquet"
OUT_DEFAULT = REPO / "data/processed/simpairs_S2.parquet"
VAL_FOLD = "fold0"
TARGET_COL = "lsf_5p0"

COUNT_COL = "mdis_image_count"
COUNT_COL_LEGACY = "mdis_emission"


def count_column(df: pd.DataFrame) -> str:
    """Name of the band-9 column in a pairs-like table.

    It was renamed `mdis_emission` -> `mdis_image_count` on 2026-08-21, when band 9 was
    identified as the count of 8-colour image sets and not an emission angle. Only the name
    changed, so a table written before that date is read as it stands.
    """
    for c in (COUNT_COL, COUNT_COL_LEGACY):
        if c in df.columns:
            return c
    raise SystemExit(f"no {COUNT_COL} column: rebuild with scripts/01_build_pairs.py.")


def haversine_km(lon1, lat1, lon2, lat2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((la1 - la2) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2)
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_backgrounds(pool, bg):
    """For each pool footprint, the index of the closest real training footprint."""
    blon, blat = bg.lon_center.to_numpy(), bg.lat_center.to_numpy()
    idx = np.empty(len(pool), dtype=int)
    for i, (lo, la) in enumerate(zip(pool.lon_center.to_numpy(), pool.lat_center.to_numpy())):
        idx[i] = int(np.argmin(haversine_km(blon, blat, lo, la)))
    return idx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--val-fold", default=VAL_FOLD,
                    help="fold kept out of the background pool, with the test split")
    ap.add_argument("--validate", metavar="PARQUET",
                    help="reference pairs file to compare this run against, column by "
                         "column; the build is deterministic, so a rerun on the same pool "
                         "must reproduce it")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    for col in ("mdis_iof", TARGET_COL):
        if col not in pool.columns:
            raise SystemExit(f"{args.pool} has no '{col}': rerun step 6 without --no-target.")
    pool = pool.sort_values(["group_id", "ref_id"]).reset_index(drop=True)
    pcount = count_column(pool)
    print(f"pool: {len(pool)} footprints, {pool.group_id.nunique()} groups", flush=True)

    refs = pool.ref_id.to_numpy()
    h_mdis = np.stack(pool.mdis_iof.to_list()).astype(np.float64)
    h_lsf = np.stack(pool[TARGET_COL].to_list()).astype(np.float64)
    h_count = pool[pcount].to_numpy(np.float64)

    # Backgrounds come from the training folds only: validation and test stay untouched.
    pairs = pd.read_parquet(PAIRS)
    bcount = count_column(pairs)
    pairs = pairs[["ref_id", "lat_center", "lon_center", "mdis_iof", bcount]]
    splits = pd.read_parquet(SPLITS)
    lsf = pd.read_parquet(TARGET, columns=["ref_id", TARGET_COL])
    bg = pairs.merge(splits, on="ref_id")
    bg = bg[~bg.split.isin([args.val_fold, "test"])].merge(lsf, on="ref_id") \
           .reset_index(drop=True)
    print(f"background candidates (training folds): {len(bg):,}", flush=True)

    idx = nearest_backgrounds(pool, bg)
    d_bg = haversine_km(bg.lon_center.to_numpy()[idx], bg.lat_center.to_numpy()[idx],
                        pool.lon_center.to_numpy(), pool.lat_center.to_numpy())
    print(f"distance to the paired background: median {np.median(d_bg):.1f} km, "
          f"p95 {np.percentile(d_bg, 95):.1f} km", flush=True)
    b_mdis = np.stack(bg.mdis_iof.iloc[idx].to_list()).astype(np.float64)
    b_lsf = np.stack(bg[TARGET_COL].iloc[idx].to_list()).astype(np.float64)

    # One row per (footprint, fill fraction), the whole footprint first.
    x, y, count, ref, frac = [], [], [], [], []
    for f in (1.0, *MIX_FRACTIONS):
        x.append(f * h_mdis + (1 - f) * b_mdis)
        y.append(f * h_lsf + (1 - f) * b_lsf)
        count.append(h_count)
        ref.append(refs)
        frac.append(np.full(len(pool), f))
    x, y = np.concatenate(x), np.concatenate(y)
    count, ref, frac = np.concatenate(count), np.concatenate(ref), np.concatenate(frac)

    ok = np.isfinite(x).all(axis=1) & np.isfinite(count)
    print(f"unusable rows: {int((~ok).sum())} of {len(ok)}", flush=True)

    out = pd.DataFrame({"ref_id": ref[ok].astype(np.int64),
                        "mdis_iof": list(x[ok].astype(np.float64)),
                        COUNT_COL: count[ok].astype(np.float64),
                        TARGET_COL: list(y[ok].astype(np.float64)),
                        "fill_fraction": frac[ok].astype(np.float64)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_pure = int((frac[ok] == 1.0).sum())
    print(f"\nwrote {args.out}: {len(out)} pairs "
          f"({n_pure} whole footprints + {len(out) - n_pure} partly filled)")

    if args.validate:
        ref_df = pd.read_parquet(args.validate)
        rep = {"n": len(out), "n_reference": len(ref_df),
               "same_count": len(out) == len(ref_df)}
        if rep["same_count"]:
            rcount = count_column(ref_df)
            for col, got in (("mdis_iof", x[ok]), (TARGET_COL, y[ok]), (rcount, count[ok])):
                want = (np.stack(ref_df[col].to_list()).astype(np.float64)
                        if ref_df[col].dtype == object else ref_df[col].to_numpy(np.float64))
                rep[col] = {"max_abs_diff": float(np.nanmax(np.abs(got - want))),
                            "allclose": bool(np.allclose(got, want, equal_nan=True))}
        print("\nvalidation against", args.validate)
        print(json.dumps(rep, indent=2))
        Path(args.out).with_suffix(".validation.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
