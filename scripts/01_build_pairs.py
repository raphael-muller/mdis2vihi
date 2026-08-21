"""Build the (MDIS 8-band, MASCS/VIRS 231-bin) training pairs.

Reads `quality.dat` to filter on the strict thresholds (q1..q4), streams
`spectres-002.dat` (~31 GB CSV) in chunks, resamples each spectrum to the common
5 nm x [300, 1450 nm] grid, colocates the MDIS mosaic via the `foot_geom` WKT
polygons, and writes:

    data/processed/pairs.parquet    one row per (MDIS, MASCS) pair
    data/processed/splits.parquet   ref_id -> {test, fold0..fold4}

Intermediate per-chunk parquets land in `data/interim/` so a crashed run can
be resumed by re-running the script (already-written chunks are skipped).

The strict quality policy keeps 154 064 spectra / 5 629 obs_id (4.9 % of the good
set); colocation then leaves the 153 214 pairs the model trains on, see docs/DATA.md.

The MASCS reflectance is used as delivered (`photom_iof`, photometrically corrected
to i = 45 deg, e = 45 deg, alpha = 90 deg). No attempt is made to undo that
correction: MDIS is itself standardised to a fixed geometry, so the mismatch between
the two is a per-band scalar the network absorbs, and it is measurable with
`scripts/02_build_lsf_target.py --diag-b`.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import rasterio.windows
import shapely.geometry as sgeom
from rasterio.features import geometry_mask, geometry_window
from scipy.interpolate import interp1d
from shapely import wkt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.data.io import (  # noqa: E402
    MASCS_DIR,
    MASCS_SPECTRA_PATH,
    MDIS_MOSAIC_PATH,
    read_mascs_dat,
)

QUALITY_PATH = MASCS_DIR / "Spectra_0_360_-90_90_quality.dat"
GEOMETRY_PATH = MASCS_DIR / "Spectra_0_360_-90_90_geometry.dat"
SHAPES_PATH = MASCS_DIR / "Spectra_0_360_-90_90_shapes.dat"

INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "pairs.parquet"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"

R_MERCURY = 2_439_400.0
GRID = np.arange(300.0, 1450.0 + 5.0, 5.0)  # 231 bins

# Quality thresholds. Barraud defines q1..q4 and the value each takes on a well-behaved
# spectrum (q1 ~ 1, q2 ~ 0, q3 up to ~83 %, q4 up to ~100 %); the widths below are this
# project's. They are strict: q1 alone rejects 91 % of the corpus.
Q1_LO, Q1_HI = 0.9, 1.1
Q2_ABS_MAX = 5.0
Q3_MIN = 80.0
Q4_MIN = 95.0

TEST_FRACTION = 0.10
N_FOLDS = 5
SEED = 42

logger = logging.getLogger("01_build_pairs")


def resample_to_grid(waves: np.ndarray, vals: np.ndarray) -> np.ndarray:
    f = interp1d(waves, vals, kind="linear", bounds_error=False, fill_value=np.nan)
    return f(GRID)


def normalize_lon(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def _project_one(poly_lonlat: sgeom.Polygon):
    xs = np.array([c[0] for c in poly_lonlat.exterior.coords])
    if float(np.ptp(xs)) > 180.0:
        return None
    ys = np.array([c[1] for c in poly_lonlat.exterior.coords])
    lon_norm = normalize_lon(xs)
    return sgeom.Polygon(
        zip(np.deg2rad(lon_norm) * R_MERCURY, np.deg2rad(ys) * R_MERCURY)
    )


def project_polygon(geom):
    """MASCS WKT (lon° 0–360, lat°) -> projected metres in the MDIS Equirect CRS.

    Accepts a Polygon or a MultiPolygon (MultiPolygons appear when a footprint
    straddles the prime meridian, since MASCS uses lon ∈ [0°, 360°] so the seam is
    at 0/360, not at 180). Returns None for footprints that can't be projected
    without crossing a discontinuity:
      - any sub-polygon spanning > 180° of MASCS longitude;
      - a MultiPolygon whose projected union straddles the MDIS antimeridian
        (would force a near-full-width window).
    """
    if geom.geom_type == "Polygon":
        return _project_one(geom)
    if geom.geom_type == "MultiPolygon":
        parts: list[sgeom.Polygon] = []
        for sub in geom.geoms:
            proj = _project_one(sub)
            if proj is None:
                return None
            parts.append(proj)
        mp = sgeom.MultiPolygon(parts)
        minx, _, maxx, _ = mp.bounds
        if (maxx - minx) > np.pi * R_MERCURY:
            return None
        return mp
    return None


def coloc_mean(src, poly_xy: sgeom.Polygon, n_bands: int = 17):
    """Mean MDIS pixels under poly_xy, robust to the float32 nodata value.

    The average is uniform over the pixels the footprint touches: with
    `all_touched=True` a pixel clipped by the polygon edge weighs as much as one fully
    inside, which slightly enlarges the effective footprint. Weighting by the VIRS
    spatial response would need a profile the instrument does not publish.

    The MDIS nodata value is ~ float32 min (-3.4e38) but `src.nodata` is a Python
    float64, so strict equality can miss the last bit and let nodata values through,
    which then overflow the float32 sum in pix.mean(). Filtering on both NaN
    and `value > -1e30` catches every flavour of "garbage".
    """
    try:
        win = geometry_window(src, [sgeom.mapping(poly_xy)])
    except Exception:
        return None
    full = rasterio.windows.Window(0, 0, src.width, src.height)
    win = win.intersection(full)
    if win.width <= 0 or win.height <= 0:
        return None
    data = src.read(list(range(1, n_bands + 1)), window=win).astype(np.float64)
    mask = geometry_mask(
        [sgeom.mapping(poly_xy)],
        out_shape=(int(win.height), int(win.width)),
        transform=src.window_transform(win),
        invert=True,
        all_touched=True,
    )
    out = np.full(n_bands, np.nan, dtype=np.float64)
    for k in range(n_bands):
        pix = data[k][mask]
        pix = pix[np.isfinite(pix) & (pix > -1e30)]
        if pix.size:
            out[k] = pix.mean()
    return out


def filter_clean(quality_path: Path) -> pd.DataFrame:
    logger.info("reading %s", quality_path.name)
    q = read_mascs_dat(
        quality_path, usecols=["ref_id", "q1", "q2", "q3", "q4", "obs_id"]
    )
    mask = (
        q.q1.between(Q1_LO, Q1_HI)
        & q.q2.abs().lt(Q2_ABS_MAX)
        & q.q3.gt(Q3_MIN)
        & q.q4.gt(Q4_MIN)
    )
    clean = q.loc[mask, ["ref_id", "obs_id"]].reset_index(drop=True)
    logger.info(
        "clean spectra: %d / %d (%.2f%%), unique obs_id: %d",
        len(clean),
        len(q),
        mask.mean() * 100,
        clean.obs_id.nunique(),
    )
    return clean


def filtered_chunked_read(
    path: Path,
    ref_id_set: set,
    usecols: list[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Chunked read of a CSV sidecar, retaining only rows whose ref_id is in `ref_id_set`."""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        keep = chunk[chunk.ref_id.isin(ref_id_set)]
        if not keep.empty:
            parts.append(keep)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols or [])


def load_base(clean_ref_ids: pd.Series) -> pd.DataFrame:
    """Join geometry.dat (angles + lat/lon) and shapes.dat (foot_geom WKT)."""
    ref_set = set(clean_ref_ids.values.tolist())

    logger.info("reading %s (chunked)", GEOMETRY_PATH.name)
    geo = filtered_chunked_read(
        GEOMETRY_PATH,
        ref_set,
        usecols=[
            "ref_id",
            "obs_id",
            "ang_in",
            "ang_em",
            "ang_ph",
            "lat_center",
            "lon_center",
        ],
    )
    logger.info("geometry rows kept: %d", len(geo))

    logger.info("reading %s (chunked)", SHAPES_PATH.name)
    shp = filtered_chunked_read(SHAPES_PATH, ref_set, usecols=["ref_id", "foot_geom"])
    logger.info("shapes rows kept: %d", len(shp))

    base = geo.merge(shp, on="ref_id", how="inner").reset_index(drop=True)
    logger.info("base (geometry ∩ shapes): %d rows", len(base))
    return base


def already_done_ref_ids(interim_dir: Path) -> set:
    done = set()
    for p in sorted(interim_dir.glob("pairs_chunk_*.parquet")):
        df = pq.read_table(p, columns=["ref_id"]).to_pandas()
        done |= set(df.ref_id.values.tolist())
    return done


def process_chunks(
    base: pd.DataFrame,
    chunksize: int = 200_000,
    force: bool = False,
) -> list[Path]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if force:
        for p in INTERIM_DIR.glob("pairs_chunk_*.parquet"):
            p.unlink()
        already_done = set()
    else:
        already_done = already_done_ref_ids(INTERIM_DIR)
        if already_done:
            logger.info(
                "resuming: %d ref_id already in interim parquets", len(already_done)
            )

    base_by_ref = base.set_index("ref_id")
    needed = set(base.ref_id.values.tolist()) - already_done
    logger.info("ref_id still to colocate: %d", len(needed))

    written = sorted(INTERIM_DIR.glob("pairs_chunk_*.parquet"))
    t0 = time.time()

    with rasterio.open(MDIS_MOSAIC_PATH) as src:
        for ci, chunk in enumerate(
            pd.read_csv(MASCS_SPECTRA_PATH, chunksize=chunksize)
        ):
            out_path = INTERIM_DIR / f"pairs_chunk_{ci:04d}.parquet"
            if out_path.exists() and not force:
                continue

            keep = chunk[chunk.ref_id.isin(needed)]
            if keep.empty:
                if not needed:
                    break
                continue

            rows = []
            for _, spec_row in keep.iterrows():
                ref_id = spec_row["ref_id"]
                try:
                    meta = base_by_ref.loc[ref_id]
                except KeyError:
                    continue
                try:
                    waves = np.asarray(ast.literal_eval(spec_row["waves"]), dtype=np.float64)
                    photom_iof = np.asarray(
                        ast.literal_eval(spec_row["photom_iof"]), dtype=np.float64
                    )
                except (ValueError, SyntaxError):
                    continue

                photom_iof_5nm = resample_to_grid(waves, photom_iof)

                poly_xy = project_polygon(wkt.loads(meta.foot_geom))
                if poly_xy is None:
                    continue
                mean_17 = coloc_mean(src, poly_xy)
                if mean_17 is None or np.all(np.isnan(mean_17[:8])):
                    continue

                rows.append(
                    {
                        "ref_id": int(ref_id),
                        "obs_id": str(meta.obs_id),
                        "lat_center": float(meta.lat_center),
                        "lon_center": float(normalize_lon(meta.lon_center)),
                        "ang_in": float(meta.ang_in),
                        "ang_em": float(meta.ang_em),
                        "ang_ph": float(meta.ang_ph),
                        "foot_geom": meta.foot_geom,
                        "mdis_iof": mean_17[:8].tolist(),
                        "mdis_image_count": float(mean_17[8]),
                        "mdis_sigma": mean_17[9:17].tolist(),
                        "photom_iof_5nm": photom_iof_5nm.tolist(),
                    }
                )

            needed -= set(keep["ref_id"].values.tolist())

            if rows:
                df_chunk = pd.DataFrame(rows)
                df_chunk.to_parquet(out_path, engine="pyarrow", compression="zstd")
                written.append(out_path)
                elapsed = time.time() - t0
                logger.info(
                    "chunk %d: kept %d/%d rows, %d ref_id remaining, %.1f min elapsed",
                    ci,
                    len(rows),
                    len(keep),
                    len(needed),
                    elapsed / 60,
                )

            if not needed:
                logger.info("all ref_id colocated at chunk %d", ci)
                break

    if needed:
        logger.warning(
            "%d ref_id not found in spectres-002.dat (incomplete corpus or missing entries)",
            len(needed),
        )
    return written


def consolidate(written: list[Path], output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("concatenating %d interim parquets -> %s", len(written), output_path.name)
    df = pd.concat(
        (pd.read_parquet(p) for p in sorted(written)), ignore_index=True
    )
    df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    logger.info("wrote %s (%d rows, %d unique obs_id)", output_path, len(df), df.obs_id.nunique())
    return df


def build_splits(
    pairs: pd.DataFrame,
    output_path: Path,
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Group-wise split on obs_id: hold out `test_fraction` for test, K-fold the rest.

    Folds are assigned round-robin over a shuffled obs_id list, giving roughly
    equal obs_id counts per fold (sample counts within a fold scale with the
    variable spectra-per-obs distribution, which is acceptable here).
    """
    rng = np.random.default_rng(seed)
    obs_ids = np.array(sorted(pairs.obs_id.unique()))
    rng.shuffle(obs_ids)

    n_test = int(round(len(obs_ids) * test_fraction))
    test_obs = set(obs_ids[:n_test].tolist())
    cv_obs = obs_ids[n_test:]
    obs_to_fold = {o: i % n_folds for i, o in enumerate(cv_obs.tolist())}

    splits = pd.DataFrame(
        {"ref_id": pairs.ref_id.values, "obs_id": pairs.obs_id.values}
    )
    splits["split"] = [
        "test" if o in test_obs else f"fold{obs_to_fold[o]}"
        for o in splits.obs_id.values
    ]

    counts = splits.groupby("split").agg(
        n_pairs=("ref_id", "size"),
        n_obs_id=("obs_id", "nunique"),
    )
    logger.info("splits:\n%s", counts.to_string())

    splits[["ref_id", "split"]].to_parquet(
        output_path, engine="pyarrow", compression="zstd"
    )
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="rows per spectres-002 chunk"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore data/interim/ and re-process all chunks from scratch",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="skip colocation; recompute splits from an existing pairs.parquet",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the clean set to the first N obs_id (quick test). Default: no cap.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    if args.splits_only:
        if not PAIRS_PATH.exists():
            raise FileNotFoundError(PAIRS_PATH)
        pairs = pd.read_parquet(PAIRS_PATH)
        build_splits(pairs, SPLITS_PATH)
        return

    missing = [p for p in (QUALITY_PATH, GEOMETRY_PATH, SHAPES_PATH,
                           MASCS_SPECTRA_PATH, MDIS_MOSAIC_PATH) if not p.exists()]
    if missing:
        rel = "\n  ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
        raise SystemExit(
            f"Missing input(s):\n  {rel}\n\n"
            "The MDIS mosaic and the MASCS spectra are third-party data and are not\n"
            "shipped with the repository. See docs/DATA.md for provenance and the\n"
            "expected layout under data/raw/.")

    clean = filter_clean(QUALITY_PATH)
    if args.limit is not None:
        kept_obs = clean.obs_id.drop_duplicates().iloc[: args.limit]
        clean = clean[clean.obs_id.isin(kept_obs)].reset_index(drop=True)
        logger.info(
            "limit applied: %d spectra over %d obs_id",
            len(clean),
            clean.obs_id.nunique(),
        )
    base = load_base(clean["ref_id"])
    written = process_chunks(base, chunksize=args.chunksize, force=args.force)
    pairs = consolidate(written, PAIRS_PATH)
    build_splits(pairs, SPLITS_PATH)


if __name__ == "__main__":
    main()