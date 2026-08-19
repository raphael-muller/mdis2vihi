"""Build the crater footprint table that calibrates the hollow correction.

`calibrate_hollow_scale.py` needs, for craters known to host hollows, every MASCS footprint
falling in the crater, its measured spectrum on the 5 nm grid, and whether it sits on the
hollow field or on the surrounding floor. This builds that table from the input data, so
the calibration is reproducible rather than resting on a file someone has to be given.

`on_hollow` is a colour proxy: inside the crater disk, on the MDIS mosaic, a pixel counts
as hollow when it is both bright (band 5) and blue (band 1 over band 5) above the given
percentiles of that disk, and a footprint counts when its centre pixel does. It is
deliberately independent of the Thomas and HORNET polygons the correction selects on, so
the calibration is not judged by the rule that produced it. The percentiles match those of
`scripts/06_build_hollow_pool.py`.

The last step streams `spectres-002.dat` (~31 GB) and takes a few minutes. Writes
`runs/final/eval/hollows/footprints/crater_footprint_spectra.parquet`.

Usage: python scripts/tools/build_crater_footprints.py
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol as rio_rowcol
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.data.io import MASCS_DIR, MDIS_MOSAIC_PATH  # noqa: E402
from mdis2vihi.eval.params import GRID  # noqa: E402

OUT = REPO / "runs/final/eval/hollows/footprints/crater_footprint_spectra.parquet"
R_MERC = 2_439_400.0
MPP_KM = 0.66524315270546

# (name, latitude, east longitude, diameter km). The first four are the craters
# scripts/06_build_hollow_pool.py keeps out of the residual's training set; Eminescu is
# included for inspection only and is not used to calibrate.
CRATERS = [("Dominici", 1.35, 323.40, 20), ("Hopper", -12.44, 304.04, 36),
           ("Tyagaraja", 3.89, 211.10, 97), ("Warhol", -2.55, 353.73, 91),
           ("Eminescu", 10.66, 114.21, 129)]
CTX = 1.6            # window half-width, in crater radii


def lon_to_x(lon):
    lon = np.asarray(lon, float)
    return np.radians(np.where(lon > 180.0, lon - 360.0, lon)) * R_MERC


def crater_footprints(ds, name, lat, lon, diam_km, gx, gy, bright_pct, blue_pct):
    """Footprints of one crater, with the on-hollow flag from the MDIS colour proxy."""
    tr = ds.transform
    half = max(20, int(CTX * diam_km / 2 / MPP_KM))
    rc, cc = rio_rowcol(tr, float(lon_to_x(lon)), np.radians(lat) * R_MERC)
    b = ds.read(indexes=list(range(1, 9)),
                window=Window(cc - half, rc - half, 2 * half, 2 * half)).astype(np.float64)
    valid = np.all(np.isfinite(b) & (b > -1e30), axis=0)
    H, W = valid.shape
    bright = b[4]
    blue = b[0] / np.clip(b[4], 1e-6, None)
    yy, xx = np.mgrid[0:H, 0:W]
    in_disk = (np.hypot(yy - half, xx - half) <= diam_km / 2 / MPP_KM) & valid
    if in_disk.sum() < 25:
        return None
    hollow = (in_disk
              & (bright >= np.nanpercentile(bright[in_disk], bright_pct))
              & (blue >= np.nanpercentile(blue[in_disk], blue_pct)))
    fc = (gx - tr.c) / tr.a - (cc - half)
    fr = (gy - tr.f) / tr.e - (rc - half)
    inwin = (fc >= 0) & (fc < W) & (fr >= 0) & (fr < H)
    sub = np.where(inwin)[0]
    ri = np.clip(fr[sub].astype(int), 0, H - 1)
    ci = np.clip(fc[sub].astype(int), 0, W - 1)
    return sub, hollow[ri, ci]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bright-pct", type=float, default=82.0)
    ap.add_argument("--blue-pct", type=float, default=65.0)
    ap.add_argument("--chunksize", type=int, default=20_000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    geometry = MASCS_DIR / "Spectra_0_360_-90_90_geometry.dat"
    quality = MASCS_DIR / "Spectra_0_360_-90_90_quality.dat"
    spectra = MASCS_DIR / "Spectra_0_360_-90_90_spectres-002.dat"
    pairs = REPO / "data/processed/pairs.parquet"
    missing = [p for p in (geometry, quality, spectra, MDIS_MOSAIC_PATH, pairs)
               if not Path(p).exists()]
    if missing:
        rel = "\n  ".join(str(Path(p).relative_to(REPO)) for p in missing)
        raise SystemExit(
            f"Missing input(s):\n  {rel}\n\n"
            "The MASCS spectra and the MDIS mosaic are third-party data and are not\n"
            "shipped with the repository (docs/DATA.md); pairs.parquet comes from\n"
            "scripts/01_build_pairs.py.")

    geo = pd.read_csv(geometry, usecols=["ref_id", "obs_id", "lat_center", "lon_center"])
    qual = pd.read_csv(quality, usecols=["ref_id", "q1", "q2", "q3", "q4"])
    good = geo.merge(qual, on="ref_id", how="left")
    kept = set(pd.read_parquet(pairs, columns=["ref_id"]).ref_id)
    good["in_pairs"] = good.ref_id.isin(kept)
    gx = lon_to_x(good.lon_center.to_numpy())
    gy = np.radians(good.lat_center.to_numpy()) * R_MERC

    rows = []
    with rasterio.open(MDIS_MOSAIC_PATH) as ds:
        for name, lat, lon, diam in CRATERS:
            got = crater_footprints(ds, name, lat, lon, diam, gx, gy,
                                    args.bright_pct, args.blue_pct)
            if got is None:
                print(f"  {name}: window unusable, skipped", flush=True)
                continue
            sub, on = got
            for j, k in enumerate(sub):
                g = good.iloc[k]
                rows.append({"crater": name, "ref_id": int(g.ref_id), "obs_id": g.obs_id,
                             "lat": float(g.lat_center), "lon": float(g.lon_center),
                             "on_hollow": bool(on[j]), "q1": float(g.q1),
                             "q2": float(g.q2), "q3": float(g.q3), "q4": float(g.q4),
                             "in_pairs": bool(g.in_pairs)})
            print(f"  {name}: {len(sub)} footprints, {int(on.sum())} on hollow", flush=True)
    cand = pd.DataFrame(rows)

    print(f"reading the measured spectra of {len(cand)} footprints "
          f"(streams the ~31 GB file)...", flush=True)
    want = set(cand.ref_id.astype("int64"))
    got = {}
    for ch in pd.read_csv(spectra, usecols=["ref_id", "waves", "photom_iof"],
                          dtype={"ref_id": "int64"}, chunksize=args.chunksize):
        for r in ch[ch.ref_id.isin(want)].itertuples(index=False):
            w = np.asarray(ast.literal_eval(r.waves), float)
            p = np.asarray(ast.literal_eval(r.photom_iof), float)
            o = np.argsort(w)
            got[int(r.ref_id)] = np.interp(GRID, w[o], p[o], left=np.nan, right=np.nan)
        if len(got) == len(want):
            break

    rid = list(got)
    out = cand.merge(pd.DataFrame({"ref_id": rid,
                                   "photom_iof_5nm": [got[r] for r in rid]}), on="ref_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} footprints, "
          f"{int(out.on_hollow.sum())} on hollow")
    print(out.groupby("crater").agg(n=("ref_id", "size"), on_hollow=("on_hollow", "sum"),
                                    pass_q1=("in_pairs", "sum")).to_string())


if __name__ == "__main__":
    main()
