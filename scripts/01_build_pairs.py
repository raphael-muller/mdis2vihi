"""Build the (MDIS 8-band, MASCS/VIRS 231-bin) training pairs.

Reads `quality.dat` to filter on the strict thresholds (q1..q4), streams
`spectres-002.dat` (~31 GB CSV) in chunks, resamples each spectrum to the common
5 nm x [300, 1450 nm] grid, colocates the MDIS mosaic via the `foot_geom` WKT
polygons, and writes:

    data/processed/pairs.parquet    one row per (MDIS, MASCS) pair
    data/processed/splits.parquet   ref_id -> {test, fold0..fold4}

The second file is the train/validation/test split, drawn once over whole observations:
10 % of the obs_id go to `test`, the rest are dealt into five folds. The delivered model
trains on fold1..fold4, validates on fold0 and reports on test; the folds are kept as
labels so the same file also supports 5-fold cross-validation.

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
from pyproj import Transformer
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
    mosaic_projector,
    read_mascs_dat,
)

QUALITY_PATH = MASCS_DIR / "Spectra_0_360_-90_90_quality.dat"
GEOMETRY_PATH = MASCS_DIR / "Spectra_0_360_-90_90_geometry.dat"
SHAPES_PATH = MASCS_DIR / "Spectra_0_360_-90_90_shapes.dat"

INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "pairs.parquet"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"

GRID = np.arange(300.0, 1450.0 + 5.0, 5.0)  # 231 bins

# Quality thresholds. Barraud defines q1..q4 and the value each takes on a well-behaved
# spectrum (q1 ~ 1, q2 ~ 0, q3 up to ~83 %, q4 up to ~100 %); the widths below are this
# project's. They are strict: q1 alone rejects 91 % of the corpus.
#
# The window on q1 is narrower than the published practice and the window on q2 is wider:
# Besse et al. (2015) work at q1 in [-1, 5] and |q2| < 0.5 %. Measured on the 3 172 422
# spectra of the good set, the two are not independent knobs and this filter is not the
# stricter one overall: it keeps 154 064 spectra (4.86 %, 5 629 obs_id) where Besse's joint
# operating point keeps 388 567 (12.25 %, 6 020 obs_id), 2.5 times more.
#
# The |q2| < 5 above does reject almost nothing (q2 alone passes 99.4 % of the corpus,
# against 9.0 % for q1), but Besse's 0.5 % is NOT implied by the strict q1, contrary to
# what the shared VIS/NIR-junction meaning of the two indicators suggests: only 24.4 % of
# the 154 064 delivered spectra already satisfy it (median |q2| 0.954, p90 2.00, p99 3.46),
# so adopting it would cut the training set by 75.6 %, to 37 612 spectra over 3 818
# observations. That trade has not been trained; the measurement is the reason not to make
# it blindly, and the reviewer's remark on q2 stands as a real looseness of this filter.
#
# The cost of the narrow q1 is not statistical but compositional: the criterion is a prior
# on spectral *shape*, and a hollow violates it by construction. None of the 1 209 hollow
# footprints of the Thomas et al. (2014) catalogue passes it, their median q1 being 1.99.
# Widening q1 to Besse's window was therefore screened end to end, on the delivered
# protocol: training on the strict folds plus the 1.15 M spectra the wider window adds
# (x11.3 pairs, seed 42), scored on the unchanged strict test split. It does recover the
# unit, closing 52 % of the 1400 nm contrast gap on the four kept-aside hollow craters
# (1.172 -> 1.257 against 1.334 measured by MASCS), but it breaks the colour panel (CCC on
# ci_415_750 0.794 -> 0.590, ci_750_415 -0.223, nir_slope -0.163) and over-contrasts the
# faculae (+0.17 against the MASCS truth, where the delivered model sits at +0.02 to
# +0.06), while MSE and spectral angle both move the wrong way: 5 mechanical criteria out
# of 6 fail. Hence: the strict window stays, and hollows are treated a posteriori by the
# frozen correction layer (scripts/06..08 and src/mdis2vihi/correction/).
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


def _project_one(poly_lonlat: sgeom.Polygon, tf: Transformer):
    xs = np.array([c[0] for c in poly_lonlat.exterior.coords])
    if float(np.ptp(xs)) > 180.0:
        return None
    ys = np.array([c[1] for c in poly_lonlat.exterior.coords])
    x, y = tf.transform(normalize_lon(xs), ys)
    return sgeom.Polygon(zip(x, y))


def project_polygon(geom, tf: Transformer | None = None):
    """MASCS WKT (lon°, lat°) -> projected metres in the MDIS mosaic CRS.

    The projection itself is PROJ's, through `mdis2vihi.data.io.mosaic_projector`;
    `tf` reuses a transformer the caller already built, otherwise one comes from the
    mosaic's own CRS.

    Longitude convention, measured on the whole good set (3 172 422 rows): the
    `shapes.dat` WKT is on [-180°, 180°], with 51.4 % of the footprints carrying
    negative longitudes, so it is NOT the [0°, 360°) convention of the rest of the
    corpus; `geometry.dat` is, both for `lon_center` and for `lon_c1..c4`.
    `normalize_lon` maps either onto the mosaic's [-180°, 180°).

    Accepts a Polygon or a MultiPolygon: 429 footprints are stored in two parts, 296
    split at the antimeridian and 133 at the prime meridian, so both seams occur in
    this file. Returns None for footprints that can't be projected without crossing a
    discontinuity:
      - any sub-polygon spanning > 180° of MASCS longitude;
      - a MultiPolygon whose projected union straddles the MDIS antimeridian
        (would force a near-full-width window).
    """
    tf = tf if tf is not None else mosaic_projector()
    if geom.geom_type == "Polygon":
        return _project_one(geom, tf)
    if geom.geom_type == "MultiPolygon":
        parts: list[sgeom.Polygon] = []
        for sub in geom.geoms:
            proj = _project_one(sub, tf)
            if proj is None:
                return None
            parts.append(proj)
        mp = sgeom.MultiPolygon(parts)
        minx, _, maxx, _ = mp.bounds
        if (maxx - minx) > float(tf.transform(180.0, 0.0)[0]):
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
    which then overflow the float32 sum of the mean. Filtering on both NaN
    and `value > -1e30` catches every flavour of "garbage".

    That -1e30 is a sentinel guard and not a reflectance floor: raising it into one, to
    drop shadowed pixels, was measured and is not needed here. On 4 000 random training
    footprints (132 663 pixels) NO pixel is below I/F 0.01 at 749 nm and 0.40 % are
    below 0.02, all of them at |lat| ~ 72 deg with a median image count of 3, i.e.
    grazing polar illumination rather than cast shadow; under the footprints the
    quality filter rejects, 0.14 % fall below 0.01, so the darkest cases are already
    screened upstream. Were such a floor ever wanted, it would have to be ONE mask over
    the pixel, `valid &= data[ref_band] > floor` broadcast to every band: a shadow is
    dark in all bands, and filtering band by band would average different pixel sets
    per band and move exactly the colour ratios the model is judged on.

    The 17 band means are reduced in one pass rather than band by band; `mask` is
    (h, w) and broadcasts against the (17, h, w) window. Bands whose valid count is
    zero stay NaN, as the caller expects.
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
    valid = mask & np.isfinite(data) & (data > -1e30)
    n_valid = valid.sum(axis=(1, 2))
    total = np.where(valid, data, 0.0).sum(axis=(1, 2))
    return np.divide(total, n_valid,
                     out=np.full(n_bands, np.nan, dtype=np.float64), where=n_valid > 0)


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
        tf = mosaic_projector(src.crs)
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

                poly_xy = project_polygon(wkt.loads(meta.foot_geom), tf)
                if poly_xy is None:
                    continue
                # The 17 mosaic bands in file order, so the indices below are 0-based:
                # [:8] the I/F filters 433..996 nm, [8] band 9 = the count of 8-colour
                # image sets stacked at that pixel (an instrument covariate, NOT an
                # angle: the MDR has carried no angle backplane since v3, docs/DATA.md
                # section 1), [9:17] their standard deviations. The observing angles of
                # the row (ang_in, ang_em, ang_ph) are the MASCS ones, from geometry.dat.
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


LAT_EDGES = (30.0, 60.0)   # bands of absolute latitude, in degrees
N_COUNT_BINS = 3           # image-count terciles
N_INC_BINS = 2             # incidence, split at its median


def _pair_weighted_bins(values: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin `values` at the quantiles of their pair-weighted distribution.

    The quantiles are weighted by the number of pairs each obs_id carries, so the bins
    hold equal numbers of *spectra* rather than equal numbers of observations, which is
    what the balance of the resulting splits depends on.
    """
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    edges = [np.interp(k / n_bins, cumulative, values[order]) for k in range(1, n_bins)]
    return np.digitize(values, edges)


def _strata(groups: pd.DataFrame) -> np.ndarray:
    """Stratum of every obs_id: |latitude| band x image-count tercile x incidence half."""
    weights = groups.n_pairs.to_numpy(float)
    lat = np.digitize(groups.abs_lat.to_numpy(), LAT_EDGES)
    count = _pair_weighted_bins(groups.image_count.to_numpy(), weights, N_COUNT_BINS)
    incidence = _pair_weighted_bins(groups.ang_in.to_numpy(), weights, N_INC_BINS)
    return (lat * N_COUNT_BINS + count) * N_INC_BINS + incidence


def _plain_draw(pairs: pd.DataFrame, rng, test_fraction: float, n_folds: int) -> dict:
    """The unstratified draw: shuffle the obs_id, cut the test set, deal the rest."""
    obs_ids = np.array(sorted(pairs.obs_id.unique()))
    rng.shuffle(obs_ids)
    n_test = int(round(len(obs_ids) * test_fraction))
    assignment = {o: "test" for o in obs_ids[:n_test].tolist()}
    # What is left after the test set is not "the training set cut into folds": the folds
    # ARE the validation rotation, and they define what training is. The delivered model
    # validates on fold0 and trains on fold1..fold4; the cross-validation of the campaign
    # rotates which fold plays validation, always against the same untouched test set.
    assignment.update(
        {o: f"fold{i % n_folds}" for i, o in enumerate(obs_ids[n_test:].tolist())}
    )
    return assignment


def _stratified_draw(pairs: pd.DataFrame, rng, test_fraction: float, n_folds: int) -> dict:
    """Draw whole obs_id inside strata, filling each split up to its quota of pairs.

    Groups are shuffled inside their stratum and handed out one at a time to whichever
    split is furthest behind its quota, counted in pairs. Every split therefore ends up
    with the same stratum composition and, unlike the round-robin deal, with the same
    number of pairs and not merely the same number of observations.
    """
    groups = pairs.groupby("obs_id").agg(
        n_pairs=("ref_id", "size"),
        abs_lat=("lat_center", lambda s: s.abs().mean()),
        image_count=("mdis_image_count", "mean"),
        ang_in=("ang_in", "mean"),
    )
    groups["stratum"] = _strata(groups)
    # The n_folds + 1 destinations, in the order their quotas are given below: the test set
    # held out for the final audit, then the folds, which are the validation rotation rather
    # than a subdivision of the training set (see build_splits).
    names = np.array(["test"] + [f"fold{i}" for i in range(n_folds)])
    quotas = np.array([test_fraction] + [(1 - test_fraction) / n_folds] * n_folds)
    filled = np.zeros(len(names))

    assignment = {}
    for _, block in groups.groupby("stratum"):
        obs_ids = np.array(sorted(block.index))
        rng.shuffle(obs_ids)
        for obs_id in obs_ids.tolist():
            k = int(np.argmin(filled / quotas))
            assignment[obs_id] = names[k]
            filled[k] += float(block.n_pairs[obs_id])
    return assignment


def build_splits(
    pairs: pd.DataFrame,
    output_path: Path,
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
    n_folds: int = N_FOLDS,
    stratify: bool = True,
) -> pd.DataFrame:
    """Group-wise split on obs_id: hold out `test_fraction` for test, K-fold the rest.

    The file this writes carries `n_folds` + 1 labels, and the folds are a *validation
    rotation*, not a subdivision of an already-defined training set: the training set is
    what the rotation leaves over. The delivered model validates on `fold0` and trains on
    `fold1`..`fold4`; the cross-validation of the campaign retrains the same configuration
    `n_folds` times, each time validating on a different fold, and every one of those runs
    reports on the same `test` split, which is drawn once and never rotated.

    Whole observations move together, which is the constraint that stops the spatial
    leak: two MASCS spots a few kilometres apart on the same track are nearly the same
    measurement, so splitting them across train and test would score memorisation. The
    draw is stratified *inside* that constraint, never against it: obs_id remains the
    unit that is drawn, and the strata only decide which observations compete with which.

    Strata are the group-level |latitude| band (0-30-60-90 deg) x image-count tercile x
    incidence half, the three covariates that shift with a random draw and along which
    accuracy varies (median spectral angle 2.87 deg below 30 deg of latitude against
    3.74 deg beyond 60 deg; 3.35 deg where fewer than ten image sets are stacked against
    2.9 deg above). Terrain class is not stratified: it is what a track shares along its
    whole length, so it is largely carried by the latitude and geometry bands already.

    Measured on this dataset, three stratified draws against three unstratified ones,
    each trained end to end:

      - covariate balance: worst |standardised mean difference| over absolute latitude,
        image count, the three angles, reflectance level and colour drops from a median
        of 0.117 over 500 unstratified draws (0.126 for the delivered one) to a median
        0.06 over 10 stratified draws;
      - fold sizes: the spread between the largest and the smallest fold falls from
        ~3 000 pairs to ~25, so cross-validation folds finally weigh the same;
      - representativity: reweighting the test set to the training population moves the
        test MSE by -0.27 +/- 0.32 % against +0.22 +/- 2.01 %, i.e. the estimate stops
        depending on the luck of the draw;
      - accuracy is unchanged: MSE / k-NN floor 1.568 +/- 0.027 against 1.556 +/- 0.007,
        median spectral angle 3.020 +/- 0.014 deg against 3.102 +/- 0.043 deg, mean CCC
        0.537 +/- 0.013 against 0.535 +/- 0.044. Stratification buys stability across
        draws rather than a better model.

    `stratify=False` restores the historical round-robin deal bit for bit: the delivered
    `splits.parquet`, and therefore the delivered checkpoint, was drawn that way.
    """
    rng = np.random.default_rng(seed)
    draw = _stratified_draw if stratify else _plain_draw
    assignment = draw(pairs, rng, test_fraction, n_folds)

    splits = pd.DataFrame(
        {"ref_id": pairs.ref_id.values, "obs_id": pairs.obs_id.values}
    )
    splits["split"] = [assignment[o] for o in splits.obs_id.values]

    counts = splits.groupby("split").agg(
        n_pairs=("ref_id", "size"),
        n_obs_id=("obs_id", "nunique"),
    )
    logger.info("splits (%s):\n%s", "stratified" if stratify else "plain", counts.to_string())
    train = [f"fold{i}" for i in range(1, n_folds) if f"fold{i}" in counts.index]
    logger.info(
        "delivered use: fold0 = validation, %s = training (%d pairs), test = final audit only",
        " + ".join(train), int(counts.loc[train, "n_pairs"].sum()),
    )

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
        "--no-stratify",
        action="store_true",
        help="draw the splits without strata (reproduces the delivered splits.parquet)",
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
        build_splits(pairs, SPLITS_PATH, stratify=not args.no_stratify)
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
    build_splits(pairs, SPLITS_PATH, stratify=not args.no_stratify)


if __name__ == "__main__":
    main()