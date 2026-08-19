"""Determine the strength of the hollow correction, and estimate how it generalises.

`build_hollow_correction.py --scale` cannot be read off the physics: it has to be
calibrated against craters whose hollow contrast is measured. With only a handful of such
craters, calibrating and validating on the same ones proves nothing, so this script does
both by leave-one-out:

* per crater, the scale `s*` making the corrected hollow/background contrast at 996 nm
  equal the contrast MASCS measures there;
* per crater, the median `s*` of the **others** applied to the one left out: the
  departure that remains estimates what happens on a crater the calibration never saw;
* the scale to deploy is the median of all `s*`.

Only craters kept out of the residual's training set are usable
(`scripts/06_build_hollow_pool.py::HELD_OUT`); the others would flatter the result.

Needs `runs/final/eval/hollows/footprints/crater_footprint_spectra.parquet`, built by
`build_crater_footprints.py` from data this repository does not redistribute (docs/DATA.md).

Usage: python scripts/tools/calibrate_hollow_scale.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.correction.layer import CorrectionNetwork, GRID_NM, coef_columns  # noqa: E402
from mdis2vihi.correction.residual import load_base_model  # noqa: E402

LD = REPO / "runs/final/correction"
DELIVERABLE = REPO / "runs/final/mdis2vihi_global_final_deflate.tif"
MDIS = REPO / ("data/raw/mdis_mosaic/MDIS_MDR_20170512_PDS16_64ppd_"
               "equirectangular_withbackplanes.tif")
FOOTPRINTS = REPO / "runs/final/eval/hollows/footprints/crater_footprint_spectra.parquet"
# Kept out of the residual's training set by scripts/06_build_hollow_pool.py.
CALIBRATION_CRATERS = ("Dominici", "Hopper", "Tyagaraja", "Warhol")
R_MERC = 2_439_400.0


def sample_bands(lon, lat, path):
    """The 9 MDIS input bands at each footprint centre."""
    lon = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    xy = list(zip(np.radians(lon) * R_MERC, np.radians(np.asarray(lat)) * R_MERC))
    with rasterio.open(path) as ds:
        v = np.array(list(ds.sample(xy, indexes=list(range(1, 10)))), dtype=np.float64)
    v[v <= -1e30] = np.nan
    return v


def gate_at(layer, transform, width, height, lon, lat):
    """Is the correction switched on at each footprint centre?"""
    lon = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    col, row = ~transform * (np.radians(lon) * R_MERC,
                             np.radians(np.asarray(lat)) * R_MERC)
    row = np.clip(np.floor(row).astype(np.int64), 0, height - 1)
    col = np.clip(np.floor(col).astype(np.int64), 0, width - 1)
    key = np.sort(layer.row.to_numpy().astype(np.int64) * width
                  + layer.col.to_numpy().astype(np.int64))
    k = row * width + col
    i = np.clip(np.searchsorted(key, k), 0, len(key) - 1)
    return (key[i] == k).astype(float)


def contrast(spec, hollow, back, band):
    """Median hollow reflectance over median background reflectance, at one band."""
    return float(np.nanmedian(spec[hollow], axis=0)[band]
                 / np.nanmedian(spec[back], axis=0)[band])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-nm", type=float, default=996.0,
                    help="wavelength the contrast is measured at")
    ap.add_argument("--smax", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args()

    if not FOOTPRINTS.exists():
        raise SystemExit(
            f"Missing input:\n  {FOOTPRINTS.relative_to(REPO)}\n\n"
            "One row per MASCS footprint over the calibration craters, with its measured\n"
            "spectrum and an on_hollow flag. Derived from the MASCS set, which is not\n"
            "redistributed with this repository, see docs/DATA.md.")

    cfg = json.loads((LD / "correction_config.json").read_text(encoding="utf-8"))
    layer = pd.read_parquet(LD / cfg["files"]["layer"])
    residual = CorrectionNetwork.from_checkpoint(REPO / cfg["residual"]["ckpt"])
    base = load_base_model(REPO / cfg["base_model_ckpt"])
    emean, estd = cfg["emission"]["mean"], cfg["emission"]["std"]
    with rasterio.open(DELIVERABLE) as ds:
        transform, W, H = ds.transform, ds.width, ds.height

    cr = pd.read_parquet(FOOTPRINTS)
    cr = cr[cr.crater.isin(CALIBRATION_CRATERS)].reset_index(drop=True)
    V = sample_bands(cr.lon.to_numpy(), cr.lat.to_numpy(), MDIS)
    ok = np.all(np.isfinite(V), axis=1)
    x9 = V[ok].astype(np.float32)
    x9[:, 8] = (x9[:, 8] - emean) / estd
    with torch.no_grad():
        anchor = base(torch.from_numpy(x9)).numpy().astype(np.float64)
    resid = residual.coefficients(x9) @ residual.B.numpy().T
    g = gate_at(layer, transform, W, H, cr.lon.to_numpy()[ok], cr.lat.to_numpy()[ok])
    measured = np.vstack(cr.photom_iof_5nm.to_numpy())[ok]
    hol = cr.on_hollow.to_numpy(bool)[ok]
    name = cr.crater.to_numpy()[ok]
    band = int(np.argmin(np.abs(GRID_NM - args.band_nm)))

    grid = np.arange(0.0, args.smax + 1e-9, args.step)
    star, target, anchor_c = {}, {}, {}
    for c in CALIBRATION_CRATERS:
        h, b = (name == c) & hol, (name == c) & ~hol
        if h.sum() < 5 or b.sum() < 5:
            print(f"  {c}: too few footprints ({int(h.sum())}/{int(b.sum())}), skipped")
            continue
        target[c] = contrast(measured, h, b, band)
        anchor_c[c] = contrast(anchor, h, b, band)
        curve = np.array([contrast(anchor + (s * g)[:, None] * resid, h, b, band)
                          for s in grid])
        star[c] = float(grid[np.argmin(np.abs(curve - target[c]))])

    print(f"\ncontrast at {GRID_NM[band]:.0f} nm, hollow over background\n")
    print(f"{'crater':11s} {'measured':>9s} {'deliverable':>12s} {'s*':>7s}")
    for c in star:
        print(f"{c:11s} {target[c]:9.3f} {anchor_c[c]:12.3f} {star[c]:7.3f}")

    print("\nleave-one-out: scale taken from the OTHER craters, applied to this one\n")
    print(f"{'left out':11s} {'s from others':>14s} {'contrast':>9s} {'measured':>9s} {'error':>8s}")
    errs = []
    for c in star:
        s_hat = float(np.median([star[o] for o in star if o != c]))
        h, b = (name == c) & hol, (name == c) & ~hol
        got = contrast(anchor + (s_hat * g)[:, None] * resid, h, b, band)
        err = got - target[c]
        errs.append(err)
        print(f"{c:11s} {s_hat:14.3f} {got:9.3f} {target[c]:9.3f} {err:+8.3f}")

    deploy = float(np.median(list(star.values())))
    base_err = [anchor_c[c] - target[c] for c in star]
    print(f"\n  leave-one-out error : median {np.median(np.abs(errs)):+.3f}, "
          f"max {np.max(np.abs(errs)):.3f}")
    print(f"  same error without any correction : median "
          f"{np.median(np.abs(base_err)):.3f}, max {np.max(np.abs(base_err)):.3f}")
    print(f"\n  scale to deploy (median of s*) : {deploy:.3f}")
    out = {"band_nm": float(GRID_NM[band]), "craters": list(star),
           "measured_contrast": target, "deliverable_contrast": anchor_c,
           "s_star": star, "loo_error": {c: float(e) for c, e in zip(star, errs)},
           "loo_abs_error_median": float(np.median(np.abs(errs))),
           "uncorrected_abs_error_median": float(np.median(np.abs(base_err))),
           "scale_to_deploy": deploy,
           "residual_ckpt": cfg["residual"]["ckpt"]}
    (LD / "scale_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {(LD / 'scale_calibration.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
