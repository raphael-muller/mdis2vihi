"""Simulate training pairs for hollows, from the footprints recovered by step 6.

A hollow footprint colocated on the mosaic carries as much colocation noise as signal, so
this step teaches the correspondence without that noise, by simulating the MDIS input
**from the target**:

    syn_k = Gaussian average of the naive 5 nm spectrum over WAC band k
    x_k   = slope_k * syn_k + offset_k + [sqrt(rho)*eta + sqrt(1-rho)*zeta_k] * resid_std_k

`slope`, `offset`, `resid_std` and `rho` all come from the per-band VIRS-to-WAC relation
measured on the 153 214 real pairs (`02_build_lsf_target.py --diag-b`). `eta` is drawn once
per footprint, `zeta_k` per band, so the noise has unit variance and inter-band correlation
exactly `rho`, measured at 0.965, because a footprint brighter than its MDIS pixel is
brighter at every wavelength. `--band-corr 0.5` reproduces the pre-2026-08-19 construction,
which injected several times too much colour scatter.

Two arms go into one file: one pair per footprint, and each footprint mixed with its
nearest real background at 1/3 and 2/3, since a real pixel over a hollow field is always
partly diluted. Every row carries its `ref_id`: three rows share one, so a split ignoring
it puts the same footprint on both sides; `scripts/08_train_residual.py` groups on it. The
noise is keyed on (ref_id, arm), so the file does not depend on row order;
`--match-reference` instead draws it positionally, reproducing a file made that way.

Usage
-----
  python scripts/07_build_simpairs.py
  python scripts/07_build_simpairs.py --match-reference <pool.parquet> --validate <pairs.parquet>
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

GRID = np.arange(300.0, 1455.0, 5.0)          # 231 bands, the deliverable grid
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
R_KM = 2439.4
DATA_SEED = 4242                              # noise drawn once, shared by every training seed
MIX_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)        # partial-fill fractions of the hollow arm
BAND_CORR_LEGACY = 0.5                        # hard-coded before rho was measured

POOL = REPO / "data/processed/hollow_pool.parquet"
PAIRS = REPO / "data/processed/pairs.parquet"
SPLITS = REPO / "data/processed/splits.parquet"
TARGET = REPO / "data/processed/virs_lsf_target.parquet"
WAC_TABLE = REPO / "runs/lsf_target/eval/virs_wac_consistency.csv"
OUT_DEFAULT = REPO / "data/processed/simpairs_S2.parquet"
VAL_FOLD = "fold0"


def load_wac(path):
    """Per-band VIRS-to-WAC calibration: centre, width, slope, offset, residual spread,
    and the measured inter-band correlation of the residual."""
    if not Path(path).exists():
        raise SystemExit(
            f"{path} not found, produce it first with:\n"
            f"    python scripts/02_build_lsf_target.py --diag-b")
    w = pd.read_csv(path)
    if "resid_corr_mean" not in w.columns:
        raise SystemExit(
            f"{path} predates the inter-band correlation measurement, regenerate it:\n"
            f"    python scripts/02_build_lsf_target.py --diag-b")
    return (w.centre_nm.to_numpy(), w.fwhm_nm.to_numpy(), w.slope.to_numpy(),
            w.offset.to_numpy(), w.resid_std.to_numpy(),
            float(w.resid_corr_mean.mean()))


def synthesise_bands(y, centres, fwhms):
    """5 nm spectra -> the 8 WAC bands, NaN-tolerant Gaussian average truncated at 3 sigma."""
    out = np.full((len(y), len(centres)), np.nan)
    for b in range(len(centres)):
        sigma = fwhms[b] * FWHM_TO_SIGMA
        g = np.exp(-0.5 * ((GRID - centres[b]) / sigma) ** 2)
        g[np.abs(GRID - centres[b]) > 3.0 * sigma] = 0.0
        finite = np.isfinite(y)
        num = np.nansum(np.where(finite, y, 0.0) * g[None, :], axis=1)
        den = np.where(finite, g[None, :], 0.0).sum(axis=1)
        ok = den > 0
        out[ok, b] = num[ok] / den[ok]
    return out


def keyed_noise(ref_ids, arm, n_bands, seed=DATA_SEED):
    """Noise keyed on (ref_id, arm): identical whatever order the rows come in."""
    eta = np.empty((len(ref_ids), 1))
    zeta = np.empty((len(ref_ids), n_bands))
    for i, ref in enumerate(ref_ids):
        r = np.random.default_rng([seed, int(ref), int(arm)])
        eta[i, 0] = r.standard_normal()
        zeta[i] = r.standard_normal(n_bands)
    return eta, zeta


def simulate_input(y_naive, eta, zeta, centres, fwhms, slopes, offsets, resids,
                   band_corr):
    """Target -> simulated MDIS input, with the measured per-band relation and its noise.

    `sqrt(rho)*eta + sqrt(1-rho)*zeta_k` has unit variance and inter-band correlation
    `rho` exactly. At rho = 0.5 it reduces to `(eta + zeta_k)/sqrt(2)`, the form used
    before the correlation was measured.
    """
    syn = synthesise_bands(y_naive, centres, fwhms)
    rho = float(band_corr)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"--band-corr must be in [0, 1], got {rho}")
    noise = (np.sqrt(rho) * eta + np.sqrt(1.0 - rho) * zeta) * resids[None, :]
    return slopes[None, :] * syn + offsets[None, :] + noise


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--wac", default=str(WAC_TABLE))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--val-fold", default=VAL_FOLD,
                    help="fold kept out of the background pool, with the test split")
    ap.add_argument("--band-corr", type=float, default=None,
                    help=f"inter-band correlation of the injected noise. Default: the "
                         "value measured on the real calibration residuals and carried "
                         "by virs_wac_consistency.csv. Pass "
                         f"{BAND_CORR_LEGACY} to reproduce the pre-2026-08-19 layer.")
    ap.add_argument("--validate", metavar="PARQUET",
                    help="reference simulated pairs to compare against")
    ap.add_argument("--match-reference", metavar="POOL",
                    help="reproduce a file made with a positional noise draw: take the pool row order "
                         "from this file and draw the noise positionally, as it did")
    args = ap.parse_args()

    centres, fwhms, slopes, offsets, resids, measured_corr = load_wac(args.wac)
    band_corr = measured_corr if args.band_corr is None else float(args.band_corr)
    print(f"inter-band noise correlation: {band_corr:.4f} "
          f"({'measured' if args.band_corr is None else 'forced'}; "
          f"measured value {measured_corr:.4f})", flush=True)
    pool = pd.read_parquet(args.pool)
    for col in ("naive_5nm", "lsf_5p0"):
        if col not in pool.columns:
            raise SystemExit(f"{args.pool} has no '{col}': rerun step 6 without --no-target.")
    if args.match_reference:
        order = pd.read_parquet(args.match_reference, columns=["ref_id"]).ref_id
        pool = pool.set_index("ref_id").loc[order].reset_index()
        print(f"row order taken from {args.match_reference}", flush=True)
    else:
        pool = pool.sort_values(["group_id", "ref_id"]).reset_index(drop=True)
    print(f"pool: {len(pool)} footprints, {pool.group_id.nunique()} groups", flush=True)

    refs = pool.ref_id.to_numpy()
    h_naive = np.stack(pool.naive_5nm.to_list()).astype(np.float64)
    h_lsf = np.stack(pool.lsf_5p0.to_list()).astype(np.float64)
    h_emis = pool.mdis_emission.to_numpy(np.float64)

    # Backgrounds come from the training folds only: validation and test stay untouched.
    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "lat_center", "lon_center",
                                            "photom_iof_5nm"])
    splits = pd.read_parquet(SPLITS)
    lsf = pd.read_parquet(TARGET, columns=["ref_id", "lsf_5p0"])
    bg = pairs.merge(splits, on="ref_id")
    bg = bg[~bg.split.isin([args.val_fold, "test"])].merge(lsf, on="ref_id") \
           .reset_index(drop=True)
    print(f"background candidates (training folds): {len(bg):,}", flush=True)

    idx = nearest_backgrounds(pool, bg)
    d_bg = haversine_km(bg.lon_center.to_numpy()[idx], bg.lat_center.to_numpy()[idx],
                        pool.lon_center.to_numpy(), pool.lat_center.to_numpy())
    print(f"distance to the paired background: median {np.median(d_bg):.1f} km, "
          f"p95 {np.percentile(d_bg, 95):.1f} km", flush=True)
    b_naive = np.stack(bg.photom_iof_5nm.iloc[idx].to_list()).astype(np.float64)
    b_lsf = np.stack(bg.lsf_5p0.iloc[idx].to_list()).astype(np.float64)

    mix_naive, mix_lsf, mix_emis, mix_ref = [], [], [], []
    for f in MIX_FRACTIONS:
        mix_naive.append(f * h_naive + (1 - f) * b_naive)
        mix_lsf.append(f * h_lsf + (1 - f) * b_lsf)
        mix_emis.append(h_emis)
        mix_ref.append(refs)
    mix_naive = np.concatenate(mix_naive)
    mix_lsf = np.concatenate(mix_lsf)
    mix_emis = np.concatenate(mix_emis)
    mix_ref = np.concatenate(mix_ref)

    if args.match_reference:
        # Positional draw of the original run: one stream, pure arm then mixtures.
        rng = np.random.default_rng(DATA_SEED)
        eta_p = rng.standard_normal((len(h_naive), 1))
        zeta_p = rng.standard_normal((len(h_naive), len(centres)))
        eta_m = rng.standard_normal((len(mix_naive), 1))
        zeta_m = rng.standard_normal((len(mix_naive), len(centres)))
    else:
        eta_p, zeta_p = keyed_noise(refs, 0, len(centres))
        arms = np.concatenate([np.full(len(h_naive), i + 1) for i in range(len(MIX_FRACTIONS))])
        eta_m = np.empty((len(mix_naive), 1)); zeta_m = np.empty((len(mix_naive), len(centres)))
        for a in np.unique(arms):
            sel = arms == a
            eta_m[sel], zeta_m[sel] = keyed_noise(mix_ref[sel], int(a), len(centres))

    x_pure = simulate_input(h_naive, eta_p, zeta_p, centres, fwhms, slopes, offsets, resids,
                            band_corr=band_corr)
    x_mix = simulate_input(mix_naive, eta_m, zeta_m, centres, fwhms, slopes, offsets, resids,
                           band_corr=band_corr)

    ok_pure = np.isfinite(x_pure).all(axis=1)
    ok_mix = np.isfinite(x_mix).all(axis=1)
    print(f"unusable simulated inputs: {int((~ok_pure).sum())} pure, "
          f"{int((~ok_mix).sum())} mixtures", flush=True)

    x = np.concatenate([x_pure[ok_pure], x_mix[ok_mix]])
    e = np.concatenate([h_emis[ok_pure], mix_emis[ok_mix]])
    y = np.concatenate([h_lsf[ok_pure], mix_lsf[ok_mix]])
    ref = np.concatenate([refs[ok_pure], mix_ref[ok_mix]])
    out = pd.DataFrame({"ref_id": ref.astype(np.int64),
                        "mdis_iof": list(x.astype(np.float64)),
                        "mdis_emission": e.astype(np.float64),
                        "lsf_5p0": list(y.astype(np.float64))})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} simulated pairs "
          f"({int(ok_pure.sum())} pure + {int(ok_mix.sum())} mixed)")

    if args.validate:
        ref = pd.read_parquet(args.validate)
        rep = {"n": len(out), "n_reference": len(ref), "same_count": len(out) == len(ref)}
        if rep["same_count"]:
            for col, got in (("mdis_iof", x), ("lsf_5p0", y),
                             ("mdis_emission", e)):
                want = (np.stack(ref[col].to_list()).astype(np.float64)
                        if ref[col].dtype == object else ref[col].to_numpy(np.float64))
                rep[col] = {"max_abs_diff": float(np.nanmax(np.abs(got - want))),
                            "allclose": bool(np.allclose(got, want, equal_nan=True))}
        print("\nvalidation against", args.validate)
        print(json.dumps(rep, indent=2))
        Path(args.out).with_suffix(".validation.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()