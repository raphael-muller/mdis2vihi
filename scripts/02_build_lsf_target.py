"""Build the training target on the common 5 nm grid, matched to the VIRS line-spread
function.

Step 1 resamples by linear interpolation. The 5 nm grid being coarser than the native
sampling (2.33 nm/px on the VIS detector, ~3.4 nm/px on the NIR), that amounts to
point-sampling: it keeps the full per-channel noise instead of averaging it. This step
redoes the resampling as a Gaussian average matched to the instrument response, which
removes 26-30 % of the high-frequency noise; genuine gaps (0.1 % of the grid points, in
the far NIR tail) stay NaN rather than being bridged.

  --build   Re-read `spectres-002.dat` and resample two ways: `naive`, reproducing step 1,
            and `lsf_*` at FWHM 4.7, 5.0 and 7.0 nm. **`lsf_5p0` is the target of the
            deliverable.** Resumable per chunk under `data/interim/virs_lsf/`.

  --floor   k-NN lower bound per target variant, on the test split and over the bands
            finite in all of them. Writes `virs_lsf_floor.csv`, `virs_lsf_perband.csv`.

  --diag-b  Per-band relation between the target convolved with each MDIS/WAC bandpass and
            the real MDIS reflectance: slope, offset, scatter, and how much the eight
            residuals correlate with each other. `07_build_simpairs.py` reads all four.
            Writes
            `virs_wac_consistency.csv` and `virs_wac_resid_corr.csv`.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.data.io import MASCS_SPECTRA_PATH  # noqa: E402

PROCESSED = REPO_ROOT / "data/processed"
INTERIM = REPO_ROOT / "data/interim/virs_lsf"
EVAL_DIR = REPO_ROOT / "runs/lsf_target/eval"  # target-side diagnostics (--floor, --diag-b)
FIG_DIR = EVAL_DIR / "figures"
PAIRS = PROCESSED / "pairs.parquet"
SPLITS = PROCESSED / "splits.parquet"
LSF_TARGET = PROCESSED / "virs_lsf_target.parquet"

GRID = np.arange(300.0, 1450.0 + 5.0, 5.0)  # 231 bins, identical to scripts/01
GRID_STEP = 5.0
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
LSF_FWHMS = (4.7, 5.0, 7.0)  # native control / grid-matched / denoise
FWHM_TAG = {4.7: "lsf_4p7", 5.0: "lsf_5p0", 7.0: "lsf_7p0"}

# The 8 MDIS/WAC filters of the colour mosaic, ascending lambda = mdis_iof band order.
# Checked against the published transmission curves served by the SVO Filter Profile
# Service (Messenger/MDIS_WAC): seven of the eight match its effective wavelength to
# <= 0.15 nm and its effective width W_eff to <= 0.04 nm, so the column holds W_eff,
# which is ~1.06x the FWHM for a near-Gaussian profile.
#
# Filter F is the exception (SVO: 427.71 nm, FWHM 22.23 nm, W_eff 23.34 nm; here 433.2 /
# 18.1) and Hawkins et al. (2007, Sect. on OCF filter calibration) says why: a Gaussian
# shape was fitted to the centre and width of every WAC filter "except for WAC filters 6
# (430 nm) and 11 (1,010 nm)", whose centre is the 50 % point of the cumulative
# transmission and whose width is the 75 %-25 % interquartile difference. Filter 6 is
# filter F, and an interquartile width is narrower than a FWHM by construction. The two
# numbers are two published conventions, not a discrepancy to fix. Substituting the SVO
# pair moves the F-band calibration slope from 0.9918 to 1.0070 and leaves resid_std
# unchanged to five decimals, the per-band affine fit absorbing it.
#
# Modelling each bandpass as a Gaussian follows the same paper, which fits Gaussians to
# the measured WAC transmissions.
WAC = [  # (filter, centre_nm, effective width in nm)
    ("F", 433.2, 18.1), ("C", 479.9, 10.1), ("D", 558.9, 5.8), ("E", 628.8, 5.5),
    ("G", 748.7, 5.1), ("L", 828.4, 5.2), ("J", 898.8, 5.1), ("I", 996.2, 14.3),
]


# --------------------------------------------------------------- resampling (A)
def resample_naive(waves: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Exact replica of scripts/01::resample_to_grid (linear, NaN outside support)."""
    f = interp1d(waves, vals, kind="linear", bounds_error=False, fill_value=np.nan)
    return f(GRID)


def resample_lsf(waves: np.ndarray, vals: np.ndarray, fwhm: float) -> np.ndarray:
    """VIRS-LSF-aware Gaussian resampling onto GRID.

    Weighted average of native points by exp(-1/2 (dlambda/sigma)^2), truncated at
    3 sigma. NaN where (i) the grid point is outside [waves.min, waves.max] (same
    extrapolation guard as naive) or (ii) no native point lies within +-half a grid
    step (a true gap, never bridged, unlike linear interp).

    FWHM 4.7 nm is the VIRS spectral resolution given by the instrument paper
    (McClintock & Lankton 2007) and by the VIRS SIS. The true profile is **not** Gaussian:
    in the Ebert-Fastie configuration of VIRS "the resulting imaging function is a
    trapezoid with a full-width half maximum equal to the greater of the entrance slit
    image or the exit slit width". Resampling 1 500 spectra with a trapezoid of the same
    FWHM instead of the Gaussian moves the result by 0.12 % of the reflectance in the
    median, which is 1 % of the residual high-frequency noise of the target, so the
    Gaussian is used, and the departure is measured rather than assumed."""
    sigma = fwhm * FWHM_TO_SIGMA
    lo, hi = waves.min(), waves.max()
    d = GRID[:, None] - waves[None, :]          # (231, n_native)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    w[np.abs(d) > 3.0 * sigma] = 0.0
    sw = w.sum(axis=1)
    out = np.full(GRID.shape, np.nan)
    nearest = np.abs(d).min(axis=1)
    ok = (sw > 0) & (GRID >= lo) & (GRID <= hi) & (nearest <= GRID_STEP / 2.0)
    out[ok] = (w[ok] * vals[None, :]).sum(axis=1) / sw[ok]
    return out


def _done_ref_ids() -> set:
    done = set()
    for p in sorted(INTERIM.glob("virs_chunk_*.parquet")):
        done |= set(pq.read_table(p, columns=["ref_id"]).to_pandas().ref_id.tolist())
    return done


def build(chunksize: int = 200_000):
    """Recompute the naive and lsf_* targets from the native VIRS spectra."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    wanted = set(pq.read_table(PAIRS, columns=["ref_id"]).to_pandas().ref_id.tolist())
    done = _done_ref_ids()
    if done:
        print(f"resuming: {len(done)} ref_id already built", flush=True)
    needed = wanted - done
    print(f"ref_id to build: {len(needed)} / {len(wanted)}", flush=True)

    t0 = time.time()
    for ci, chunk in enumerate(pd.read_csv(MASCS_SPECTRA_PATH, chunksize=chunksize)):
        out_path = INTERIM / f"virs_chunk_{ci:04d}.parquet"
        if out_path.exists():
            continue
        keep = chunk[chunk.ref_id.isin(needed)]
        if keep.empty:
            if not needed:
                break
            continue
        rows = []
        for _, r in keep.iterrows():
            try:
                waves = np.asarray(ast.literal_eval(r["waves"]), dtype=np.float64)
                vals = np.asarray(ast.literal_eval(r["photom_iof"]), dtype=np.float64)
            except (ValueError, SyntaxError):
                continue
            if waves.size < 2:
                continue
            rec = {"ref_id": int(r["ref_id"]), "n_native": int(waves.size),
                   "naive": resample_naive(waves, vals).tolist()}
            for fw in LSF_FWHMS:
                rec[FWHM_TAG[fw]] = resample_lsf(waves, vals, fw).tolist()
            rows.append(rec)
        needed -= set(keep.ref_id.tolist())
        if rows:
            pd.DataFrame(rows).to_parquet(out_path, engine="pyarrow", compression="zstd")
            print(f"chunk {ci}: +{len(rows)} rows, {len(needed)} left, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        if not needed:
            break

    parts = [pd.read_parquet(p) for p in sorted(INTERIM.glob("virs_chunk_*.parquet"))]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(LSF_TARGET, engine="pyarrow", compression="zstd")
    print(f"wrote {LSF_TARGET}  ({len(out)} rows)", flush=True)


# ------------------------------------------------------------------- floor (A)
def _stack(series) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=np.float64) for v in series])


def floor_report(k: int = 5):
    """k-NN lower bound for the naive target and each lsf variant, on the test split and
    over the bands finite in all of them."""
    from mdis2vihi.eval.metrics import knn_floor

    lsf = pd.read_parquet(LSF_TARGET)
    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "mdis_iof"])
    splits = pd.read_parquet(SPLITS)
    split_col = "test" if "test" in splits.columns else None
    df = lsf.merge(pairs, on="ref_id").merge(splits, on="ref_id")
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    elif split_col:
        df = df[df[split_col]]
    print(f"test split n={len(df)}", flush=True)

    X = _stack(df["mdis_iof"])
    variants = {"naive": _stack(df["naive"])}
    for fw in LSF_FWHMS:
        variants[FWHM_TAG[fw]] = _stack(df[FWHM_TAG[fw]])

    # common finite mask (naive AND grid-matched lsf) for the fair headline floor
    common = np.isfinite(variants["naive"]) & np.isfinite(variants["lsf_5p0"])

    rows = []
    base = None
    for name, Y in variants.items():
        Yc = np.where(common, Y, np.nan)          # fair: identical support
        floor_fair = knn_floor(X, Yc, k=k)
        floor_own = knn_floor(X, Y, k=k)          # own support (coverage-honest)
        cov = float(np.isfinite(Y).mean())
        if base is None:
            base = floor_fair
        rows.append({"variant": name, "floor_fair": floor_fair,
                     "floor_own_support": floor_own, "coverage": cov,
                     "d_vs_naive": floor_fair - base,
                     "rel_vs_naive": (floor_fair - base) / base})
    out = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(EVAL_DIR / "virs_lsf_floor.csv", index=False)

    # per-band: coverage + high-frequency noise proxy (std of 2nd difference)
    def hf_noise(Y):
        d2 = Y[:, 2:] - 2 * Y[:, 1:-1] + Y[:, :-2]
        n = np.full(GRID.shape, np.nan)
        n[1:-1] = np.nanstd(d2, axis=0)
        return n
    pb = pd.DataFrame({"wavelength_nm": GRID})
    for name, Y in variants.items():
        pb[f"cov_{name}"] = np.isfinite(Y).mean(axis=0)
        pb[f"hfnoise_{name}"] = hf_noise(Y)
    pb.to_csv(EVAL_DIR / "virs_lsf_perband.csv", index=False)

    print(out.to_string(index=False), flush=True)
    print(f"\ncheck: naive floor {rows[0]['floor_fair']:.4e} "
          f"\nwrote virs_lsf_floor.csv + virs_lsf_perband.csv",
          flush=True)


# ------------------------------------------------------------ VIRS-to-WAC diagnostic
def diag_b():
    """MDIS-from-VIRS WAC-bandpass synthesis vs the real `mdis_iof`, per band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pd.read_parquet(PAIRS, columns=["ref_id", "mdis_iof", "photom_iof_5nm"])
    Yt = _stack(pairs["photom_iof_5nm"])          # (N, 231) existing 5 nm target
    Xr = _stack(pairs["mdis_iof"])                # (N, 8) real MDIS input

    # synthesize x_syn[b] = sum_lambda target(lambda) * G_b(lambda) / sum G_b
    Xsyn = np.full((Yt.shape[0], len(WAC)), np.nan)
    for b, (_, c, fw) in enumerate(WAC):
        sigma = fw * FWHM_TO_SIGMA
        g = np.exp(-0.5 * ((GRID - c) / sigma) ** 2)
        g[np.abs(GRID - c) > 3.0 * sigma] = 0.0
        W = np.where(np.isfinite(Yt), g[None, :], 0.0)
        num = np.nansum(np.where(np.isfinite(Yt), Yt, 0.0) * g[None, :], axis=1)
        den = W.sum(axis=1)
        ok = den > 0
        Xsyn[ok, b] = num[ok] / den[ok]

    # per-band affine fit, and the residual kept aligned across bands so the inter-band
    # correlation can be measured on the same rows
    resid_all = np.full_like(Xr, np.nan)
    rows = []
    for b, (filt, c, fw) in enumerate(WAC):
        xr, xs = Xr[:, b], Xsyn[:, b]
        m = np.isfinite(xr) & np.isfinite(xs)
        xr, xs = xr[m], xs[m]
        # robust per-band affine fit  real ~ a*syn + b0
        a, b0 = np.polyfit(xs, xr, 1)
        resid = xr - (a * xs + b0)
        resid_all[m, b] = resid
        r = float(np.corrcoef(xr, xs)[0, 1])
        rows.append({
            "filter": filt, "centre_nm": c, "fwhm_nm": fw, "n": int(m.sum()),
            "corr_real_syn": r, "slope": float(a), "offset": float(b0),
            "resid_std": float(resid.std()),
            "resid_rel": float(resid.std() / np.abs(xr).mean()),
            "mean_real": float(xr.mean()), "mean_syn": float(xs.mean()),
            "mean_ratio_real_syn": float(xr.mean() / xs.mean()),
        })
    out = pd.DataFrame(rows)

    # Inter-band correlation of the colocation residual. A footprint brighter than its
    # MDIS pixel is brighter at every wavelength, so this is close to 1: the residual is
    # essentially one common brightness factor, not eight independent ones. `scripts/07`
    # reads `resid_corr_mean` to set the correlation of the noise it injects.
    ok = np.all(np.isfinite(resid_all), axis=1)
    C = np.corrcoef(resid_all[ok].T)
    names = [f for f, _, _ in WAC]
    out["resid_corr_mean"] = [(C[b].sum() - 1.0) / (len(WAC) - 1) for b in range(len(WAC))]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(C, index=names, columns=names).to_csv(EVAL_DIR / "virs_wac_resid_corr.csv")
    off = C[np.triu_indices(len(WAC), 1)]
    print(f"inter-band residual correlation on {int(ok.sum())} pairs: "
          f"median {np.median(off):.3f}, min {off.min():.3f}, max {off.max():.3f}", flush=True)
    out.to_csv(EVAL_DIR / "virs_wac_consistency.csv", index=False)

    # figure: real vs synthesized per band + the per-band ratio (the "scalar bias")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for b, (filt, c, fw) in enumerate(WAC):
        ax = axes.flat[b]
        xs, xr = Xsyn[:, b], Xr[:, b]
        m = np.isfinite(xr) & np.isfinite(xs)
        ax.hexbin(xs[m], xr[m], gridsize=60, mincnt=1, cmap="viridis")
        lim = [0, np.nanpercentile(np.r_[xs[m], xr[m]], 99)]
        ax.plot(lim, lim, "w--", lw=1)
        a, b0 = out.slope[b], out.offset[b]
        ax.plot(lim, [a * lim[0] + b0, a * lim[1] + b0], "r-", lw=1)
        ax.set_title(f"{filt} {c:.0f} nm  r={out.corr_real_syn[b]:.3f}  "
                     f"k={out.mean_ratio_real_syn[b]:.2f}")
        ax.set_xlabel("MDIS-from-VIRS (synthesised)")
        ax.set_ylabel("real MDIS I/F")
    fig.suptitle("WAC-bandpass synthesis from VIRS vs real MDIS (per filter)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "virs_wac_consistency.png", dpi=110)
    print(out.to_string(index=False), flush=True)
    print("\nwrote virs_wac_consistency.csv + virs_wac_resid_corr.csv "
          "+ figures/virs_wac_consistency.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="rebuild the target variants")
    ap.add_argument("--floor", action="store_true", help="k-NN lower bound per target variant")
    ap.add_argument("--diag-b", action="store_true", help="per-band VIRS-to-WAC diagnostic")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    needed = [PAIRS, SPLITS] + ([MASCS_SPECTRA_PATH] if args.build else [])
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        rel = "\n  ".join(str(Path(p).relative_to(REPO_ROOT)) for p in missing)
        raise SystemExit(
            f"Missing input(s):\n  {rel}\n\n"
            "pairs.parquet / splits.parquet come from scripts/01_build_pairs.py;\n"
            "the native VIRS spectra are third-party data. See docs/DATA.md and\n"
            "docs/REPRODUCTION.md.")

    if args.build:
        build(args.chunksize)
    if args.floor:
        floor_report(args.k)
    if args.diag_b:
        diag_b()
    if not (args.build or args.floor or args.diag_b):
        ap.error("pick at least one of --build / --floor / --diag-b")


if __name__ == "__main__":
    main()