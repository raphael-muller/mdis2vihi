"""Recover the hollow footprints that the quality filter of step 1 rejects.

`q1` compares the visible and near-infrared slopes, and a hollow spectrum fails it by
construction, so the model never sees the unit it is later asked to render. This step
selects those footprints on purpose, for step 7 to build training pairs from.

Selection rule: a MASCS footprint whose centre lies within `--radius-km` of a Thomas et al.
(2014) hollow-group centre (nearest group wins); groups within a kept-aside crater radius
plus 20 km are dropped whole; a bright+blue mask taken as percentiles of MDIS R749 and
R433/R749 over the disk, tested at the footprint centre pixel; the strict filter of step 1
with `q1` inverted; colocation through the `foot_geom` polygon with the step-1 machinery;
and a visible-reflectance minimum.

The bright+blue disk is a Euclidean **pixel** radius inside a square window, which on an
equirectangular grid is not a constant ground distance, kept as is because it is what the
delivered pool was built with.

On the reference data: 17 140 candidates -> 11 039 after quality -> 1 801 on the mask
-> **1 209 footprints over 227 groups and 345 observations**.

Usage
-----
  python scripts/06_build_hollow_pool.py
  python scripts/06_build_hollow_pool.py --check      # compare each stage with the above
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely import wkt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.data.io import (  # noqa: E402
    MASCS_DIR,
    MDIS_MOSAIC_PATH,
    lonlat_to_xy,
    mosaic_projector,
)

# Step 1 is a script, not a module: load it by path so the colocation used here is
# literally the one that built the training pairs, not a copy that could drift.
_spec = importlib.util.spec_from_file_location("_step01", REPO / "scripts/01_build_pairs.py")
_step01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step01)

_spec2 = importlib.util.spec_from_file_location("_step02", REPO / "scripts/02_build_lsf_target.py")
_step02 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_step02)

THOMAS_GROUPS = REPO / "data/raw/hollow/1-s2.0-S0019103513004909-mmc1.txt"
GEOMETRY = MASCS_DIR / "Spectra_0_360_-90_90_geometry.dat"
QUALITY = MASCS_DIR / "Spectra_0_360_-90_90_quality.dat"
SHAPES = MASCS_DIR / "Spectra_0_360_-90_90_shapes.dat"
OUT_DEFAULT = REPO / "data/processed/hollow_pool.parquet"

R_KM = 2439.4                      # Mercury radius used by the catalogue arithmetic
MPP_KM = 0.66524315270546
NODATA = -1e30
VIS_BANDS = slice(2, 5)            # 559, 629, 749 nm, as in step 1

# Free parameters of the selection, all choices of this project rather than published
# thresholds: the percentiles below keep the brightest 18 % / bluest 35 % of each disk,
# `--radius-km 15` is the search radius, `--vis-floor 0.008` drops footprints too dark to
# be a hollow floor. Changing any of them changes the delivered layer; `--check` compares.
R749_PCT, CI_PCT = 82, 65

# Kept aside evaluation craters: no footprint of theirs may reach training, so the
# correction can be judged on hollows the residual has never seen. The 20 km margin added
# to each radius is a choice. (name, latitude, east longitude, diameter km)
HELD_OUT = [("Dominici", 1.35, 323.4, 20), ("Hopper", -12.44, 304.04, 36),
            ("Tyagaraja", 3.89, 211.10, 97), ("Warhol", -2.55, 353.73, 91)]
HELD_OUT_MARGIN_KM = 20.0


def haversine_km(lon1, lat1, lon2, lat2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((la1 - la2) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2)
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_180(lon):
    lon = np.asarray(lon, float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def load_groups(path=THOMAS_GROUPS):
    """Thomas 2014 hollow-group centres.

    This is the `Group_id, Central_long, Central_lat` table (445 groups), **not** the
    supplementary table with geodesic areas that the correction selection uses: here the
    reference disk has a fixed radius, so no area is needed. See docs/DATA.md.
    """
    cat = pd.read_csv(path, sep="\t")
    missing = {"Group_id", "Central_long", "Central_lat"} - set(cat.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}: wrong Thomas table?")
    return cat.dropna(subset=["Group_id"]).reset_index(drop=True)


def drop_held_out_groups(cat):
    """Whole groups sitting on a kept aside crater are removed, not just their footprints:
    a group centre inside the crater cannot yield a footprint that is honestly unseen."""
    dropped = set()
    for _name, lat, lon, diam in HELD_OUT:
        d = haversine_km(to_180(lon), lat, cat.Central_long.to_numpy(float),
                         cat.Central_lat.to_numpy(float))
        dropped |= set(cat.Group_id[d <= diam / 2 + HELD_OUT_MARGIN_KM].astype(int))
    return cat[~cat.Group_id.astype(int).isin(dropped)].reset_index(drop=True), dropped


def candidates(cat, geom, radius_km):
    """Footprint centres within `radius_km` of a group centre, nearest group wins.

    The loop runs over the ~440 catalogue groups, not over the footprints, and each pass
    is already vectorised over the 3.17 M rows of `geometry.dat`. Removing it means the
    full 440 x 3.17 M distance matrix, about 11 GB in float64 for a result that is 99.5 %
    empty; the latitude band below discards most of those rows before any trigonometry.
    """
    lat = geom.lat_center.to_numpy(float)
    lon180 = to_180(geom.lon_center.to_numpy(float))
    ref = geom.ref_id.to_numpy()
    dlat = radius_km / (np.pi * R_KM / 180.0)          # latitude band, cheap prefilter
    out = []
    for k in range(len(cat)):
        gla, glo = float(cat.Central_lat[k]), float(cat.Central_long[k])
        near = np.abs(lat - gla) <= dlat
        if not near.any():
            continue
        idx = np.nonzero(near)[0]
        d = haversine_km(glo, gla, lon180[idx], lat[idx])
        hit = d <= radius_km
        if hit.any():
            out.append(pd.DataFrame({"ref_id": ref[idx[hit]],
                                     "group_id": int(cat.Group_id[k]),
                                     "d_group_km": d[hit]}))
    cand = pd.concat(out, ignore_index=True)
    return (cand.sort_values("d_group_km").drop_duplicates("ref_id", keep="first")
                .merge(geom, on="ref_id").reset_index(drop=True))


def bright_blue_hits(cand, cat, mosaic_path, radius_km):
    """Bright+blue mask, group by group, on the MDIS mosaic.

    A square window of `half = int(radius / pixel)` around the group centre, and a disk
    taken as a Euclidean **pixel** radius. On an equirectangular grid that is not a
    constant ground distance away from the equator; the convention is kept as is because
    it is what the delivered pool was built with, not because it is the better one.
    """
    centres = cat.set_index(cat.Group_id.astype(int))
    half = int(radius_km / MPP_KM)
    disk_px = radius_km / MPP_KM
    hits = np.zeros(len(cand), bool)
    with rasterio.open(mosaic_path) as src:
        T = src.transform
        tf = mosaic_projector(src.crs)
        for gid, sub in cand.groupby("group_id"):
            glo = float(centres.loc[gid, "Central_long"])
            gla = float(centres.loc[gid, "Central_lat"])
            gcol, grow = ~T * lonlat_to_xy(glo, gla, tf)
            gcol, grow = int(np.floor(gcol)), int(np.floor(grow))
            c0, r0 = max(gcol - half, 0), max(grow - half, 0)
            w = min(2 * half + 1 - (c0 - (gcol - half)), src.width - c0)
            h = min(2 * half + 1 - (r0 - (grow - half)), src.height - r0)
            if w <= 0 or h <= 0:
                continue
            b = src.read([1, 5], window=Window(c0, r0, w, h)).astype(np.float64)
            b[b <= NODATA] = np.nan
            r433, r749 = b[0], b[1]
            ci = np.divide(r433, r749, out=np.full_like(r433, np.nan), where=r749 > 0)
            yy, xx = np.mgrid[0:h, 0:w]
            rr = np.hypot((yy + r0) - grow, (xx + c0) - gcol)
            disk = (rr <= disk_px) & np.isfinite(r749) & np.isfinite(ci)
            if disk.sum() < 5:
                continue
            mask = (disk & (r749 >= np.nanpercentile(r749[disk], R749_PCT))
                         & (ci >= np.nanpercentile(ci[disk], CI_PCT)))
            fx, fy = lonlat_to_xy(to_180(sub.lon_center.to_numpy(float)),
                                  sub.lat_center.to_numpy(float), tf)
            fcol, frow = ~T * (fx, fy)
            cc = np.floor(fcol).astype(int) - c0
            rw = np.floor(frow).astype(int) - r0
            inside = (cc >= 0) & (cc < w) & (rw >= 0) & (rw < h)
            hit = np.zeros(len(sub), bool)
            hit[inside] = mask[rw[inside], cc[inside]]
            hits[sub.index.to_numpy()] = hit
    return hits


def quality_rule(df):
    """Strict filter of step 1 with `q1` inverted: this is the whole point of the pool.

    Why `q2`, `q3` and `q4` are kept while `q1` is inverted: only `q1` rejects hollows for
    what they are. It compares the visible and near-infrared slopes, so it is a prior on
    spectral shape and a hollow fails it by construction (median `q1` 1.99 over the pool).
    The other three say nothing about the surface: `q2` is the residual of the fit joining
    the two detector arrays, `q3` and `q4` are the fractions of VIS and NIR channels that
    survived Barraud's cleaning. A footprint failing them is a bad *measurement*, and this
    pool is the sole definition of the residual the correction layer will reproduce, so
    letting one in would put detector noise into the spectral basis of step 8.

    The bright+blue mask is not a substitute either: it is read on the MDIS mosaic, at
    665 m and in 8 bands, so it locates a hollow field on the ground but sees nothing of
    the quality of the VIRS spectrum that will serve as the target.

    The price is modest: the rule as a whole keeps 11 039 of the 17 140 candidates, and
    since the `q1` inversion rejects almost nothing here (the window it excludes is
    essentially empty of hollows), that 36 % loss is what `q2..q4` cost. It buys a pool
    whose spectra are on the same quality footing as the training set, with exactly one
    criterion relaxed, which is what makes the residual of step 8 a hollow signature and
    not a mixture of hollow signature and measurement noise.
    """
    return ((df.q2.abs() < 5) & (df.q3 > 80) & (df.q4 > 95)
            & ~((df.q1 >= 0.9) & (df.q1 <= 1.1)))


def colocate(pool, mosaic_path, vis_floor):
    """MDIS values under each footprint polygon, through the step-1 machinery."""
    shapes = _step01.filtered_chunked_read(SHAPES, set(pool.ref_id.tolist()),
                                           usecols=["ref_id", "foot_geom"])
    pool = pool.merge(shapes, on="ref_id")
    iof, sigma, count, vis = [], [], [], []
    with rasterio.open(mosaic_path) as src:
        tf = _step01.mosaic_projector(src.crs)
        for geom_wkt in pool.foot_geom.to_numpy():
            poly = _step01.project_polygon(wkt.loads(geom_wkt), tf)
            vals = _step01.coloc_mean(src, poly, n_bands=17) if poly is not None else None
            if vals is None:
                iof.append(None); sigma.append(None); count.append(np.nan); vis.append(np.nan)
                continue
            # 0-based: [0:8] I/F, [8] band 9 = image-set count, [9:17] their sigmas.
            iof.append(vals[0:8]); count.append(vals[8]); sigma.append(vals[9:17])
            vis.append(float(np.nanmedian(vals[VIS_BANDS])))
    pool = pool.assign(mdis_iof=iof, mdis_image_count=count, mdis_sigma=sigma,
                       vis_median_iof=vis)
    ok = pool.mdis_iof.notna() & np.isfinite(pool.mdis_image_count)
    n_bad = int((~ok).sum())
    pool = pool[ok]
    under = pool.vis_median_iof <= vis_floor
    n_floor = int(under.sum())
    return pool[~under].reset_index(drop=True), n_bad, n_floor


def spectral_target(pool, chunksize=200_000):
    """Native VIRS spectra of the pool, resampled onto the 5 nm grid two ways.

    `lsf_5p0` is the line-spread-aware resampling of step 2, and is the training target
    step 7 pairs with the colocated MDIS vector. `naive_5nm`, the linear interpolation of
    step 1, is kept beside it as the comparison point between the two resamplings on this
    pool. Both come from the step-2 functions, not from a copy.

    The native file is not re-read on every run of the chain: step 2 already caches the
    resampled spectra in `data/processed/virs_lsf_target.parquet`, one fixed-length array
    per variant with the 5 nm grid implicit, so no wavelength axis is stored or parsed
    again, and steps 3, 7 and 8 read only that. The pass below is the one exception, and
    it is forced by the selection rather than by the format: the pool is made of the
    footprints that *fail* `q1`, so its spectra were never written to that cache and exist
    only in `spectres-002.dat`. It runs once, its result is cached in the pool parquet on
    the same terms, and `--no-target` skips it.
    """
    import ast
    want = set(pool.ref_id.tolist())
    rows = []
    t0 = time.time()
    reader = pd.read_csv(_step01.MASCS_SPECTRA_PATH if hasattr(_step01, "MASCS_SPECTRA_PATH")
                         else MASCS_DIR / "Spectra_0_360_-90_90_spectres-002.dat",
                         usecols=["ref_id", "waves", "photom_iof"], chunksize=chunksize)
    for n, chunk in enumerate(reader, 1):
        hit = chunk[chunk.ref_id.isin(want)]
        for r in hit.itertuples(index=False):
            waves = np.asarray(ast.literal_eval(r.waves), float)
            vals = np.asarray(ast.literal_eval(r.photom_iof), float)
            order = np.argsort(waves)
            waves, vals = waves[order], vals[order]
            rows.append({"ref_id": r.ref_id, "n_native": int(len(waves)),
                         "naive_5nm": _step02.resample_naive(waves, vals).tolist(),
                         "lsf_5p0": _step02.resample_lsf(waves, vals, 5.0).tolist()})
        if n % 5 == 0:
            print(f"  chunk {n}: {len(rows)}/{len(want)} spectra, "
                  f"{time.time()-t0:.0f} s", flush=True)
        if len(rows) == len(want):
            break
    return pd.DataFrame(rows)


# Stage counts of the run that produced the delivered layer, at the default settings
# (--radius-km 15, --vis-floor 0.008). Also quoted in docs/REPRODUCTION.md.
REFERENCE_COUNTS = {"candidates": 17140, "after quality": 11039,
                    "bright+blue mask": 1801, "final pool": 1209,
                    "groups": 227, "observations": 345}


def validate(counts):
    """Compare each stage with the reference run. Needs nothing but this run."""
    return [{"stage": name, "n": int(counts[name]), "n_reference": ref,
             "identical": int(counts[name]) == ref}
            for name, ref in REFERENCE_COUNTS.items() if name in counts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--radius-km", type=float, default=15.0)
    ap.add_argument("--vis-floor", type=float, default=0.008)
    ap.add_argument("--mosaic", default=str(MDIS_MOSAIC_PATH))
    ap.add_argument("--no-target", action="store_true",
                    help="stop after colocation, skip the native-spectra pass")
    ap.add_argument("--check", action="store_true",
                    help="compare each stage count with the reference run and write "
                         "<out>.validation.json")
    args = ap.parse_args()

    t0 = time.time()
    cat = load_groups()
    cat, dropped = drop_held_out_groups(cat)
    print(f"catalogue: {len(cat) + len(dropped)} groups, {len(dropped)} dropped on "
          f"kept aside craters {sorted(dropped)}", flush=True)

    geom = pd.read_csv(GEOMETRY, usecols=["ref_id", "lat_center", "lon_center",
                                          "width", "length", "obs_id"])
    qual = pd.read_csv(QUALITY, usecols=["ref_id", "q1", "q2", "q3", "q4"])
    cand = candidates(cat, geom, args.radius_km).merge(qual, on="ref_id")
    print(f"candidates within {args.radius_km:.0f} km: {len(cand):,} footprints, "
          f"{cand.group_id.nunique()} groups  ({time.time()-t0:.0f} s)", flush=True)

    sig = quality_rule(cand)
    print(f"quality rule (q1 inverted): {int(sig.sum()):,} footprints, "
          f"{cand[sig].group_id.nunique()} groups", flush=True)

    hits = bright_blue_hits(cand, cat, args.mosaic, args.radius_km)
    print(f"bright+blue mask: {int(hits.sum()):,} footprints on the mask", flush=True)

    pool = cand[hits & sig].reset_index(drop=True)
    print(f"pool before colocation: {len(pool):,} footprints, "
          f"{pool.group_id.nunique()} groups, {pool.obs_id.nunique()} observations",
          flush=True)

    pool, n_bad, n_floor = colocate(pool, args.mosaic, args.vis_floor)
    print(f"colocation: {n_bad} footprint(s) without a usable polygon, "
          f"{n_floor} under the visible floor {args.vis_floor}", flush=True)

    cols = ["group_id", "ref_id", "obs_id", "lat_center", "lon_center", "d_group_km",
            "q1", "q2", "q3", "q4", "mdis_iof", "mdis_image_count", "mdis_sigma",
            "vis_median_iof", "foot_geom"]
    if not args.no_target:
        print(f"reading the native spectra of {len(pool)} footprints "
              f"(streams the ~30 GB file)...", flush=True)
        tgt = spectral_target(pool)
        pool = pool.merge(tgt, on="ref_id")
        cols += ["n_native", "naive_5nm", "lsf_5p0"]
        print(f"spectral target: {len(tgt)} spectra resampled", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool[cols].to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(pool):,} footprints, {pool.group_id.nunique()} groups, "
          f"{pool.obs_id.nunique()} observations  ({time.time()-t0:.0f} s)")

    if args.check:
        counts = {"candidates": len(cand), "after quality": int(sig.sum()),
                  "bright+blue mask": int(hits.sum()), "final pool": len(pool),
                  "groups": pool.group_id.nunique(), "observations": pool.obs_id.nunique()}
        checks = validate(counts)
        print("\nstage counts against the reference run")
        for c in checks:
            print(f"  {c['stage']:20s} {c['n']:6d} vs {c['n_reference']:6d}  "
                  f"{'MATCH' if c['identical'] else 'DIFFERS'}")
        (out.with_suffix(".validation.json")).write_text(
            json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()