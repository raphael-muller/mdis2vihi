"""I/O helpers for MDIS mosaics and MASCS/VIRS sidecar files.

MDIS side: thin `rasterio` wrappers for metadata and windowed reads of the
global GeoTIFF (never load it whole, it's ~18 GB).

MASCS side: every file under `data/raw/mascs/` is CSV (despite the `.dat`
extension); read them with `read_mascs_dat`. The per-spectrum reflectance file
`spectres-002.dat` is also CSV, but with `waves` and `photom_iof` stored as
stringified Python lists, and `read_mascs_spectra` parses those into numpy arrays.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyproj
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
MDIS_MOSAIC_PATH = (
    RAW_DIR
    / "mdis_mosaic"
    / "MDIS_MDR_20170512_PDS16_64ppd_equirectangular_withbackplanes.tif"
)
MASCS_DIR = RAW_DIR / "mascs"
MASCS_SPECTRA_PATH = MASCS_DIR / "Spectra_0_360_-90_90_spectres-002.dat"


# ------------------------------------------------------------------ MDIS mosaic


@dataclass(frozen=True)
class MosaicInfo:
    path: Path
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    transform: tuple
    bounds: tuple
    nodata: float | None
    block_shapes: tuple[tuple[int, int], ...]
    band_descriptions: tuple[str | None, ...]


def mosaic_info(path: str | Path = MDIS_MOSAIC_PATH) -> MosaicInfo:
    """Open an MDIS mosaic and return its metadata, no pixel data is read."""
    path = Path(path)
    with rasterio.open(path) as src:
        return MosaicInfo(
            path=path,
            width=src.width,
            height=src.height,
            count=src.count,
            dtypes=tuple(src.dtypes),
            crs=str(src.crs) if src.crs else None,
            transform=tuple(src.transform),
            bounds=tuple(src.bounds),
            nodata=src.nodata,
            block_shapes=tuple(src.block_shapes),
            band_descriptions=tuple(src.descriptions),
        )


def read_mdis_window(
    path: str | Path,
    window: Window | tuple[int, int, int, int],
    bands: Iterable[int] | None = None,
) -> np.ndarray:
    """Read one rectangular window from an MDIS mosaic.

    `window` is a rasterio `Window` or `(col_off, row_off, width, height)`.
    `bands` is 1-indexed per rasterio convention; `None` reads every band.
    Returns an array shaped `(n_bands, height, width)`.
    """
    if not isinstance(window, Window):
        window = Window(*window)
    with rasterio.open(path) as src:
        indexes = list(bands) if bands is not None else list(range(1, src.count + 1))
        return src.read(indexes=indexes, window=window)


# ------------------------------------------------------------------ projection


@lru_cache(maxsize=4)
def _crs_wkt_of(path: str) -> str:
    with rasterio.open(path) as src:
        return src.crs.to_wkt()


@lru_cache(maxsize=8)
def _transformer(crs_wkt: str) -> Transformer:
    crs = pyproj.CRS.from_wkt(crs_wkt)
    return Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)


def mosaic_projector(crs=None, path: str | Path = MDIS_MOSAIC_PATH) -> Transformer:
    """(lon°, lat°) -> projected metres, for `crs` or for the CRS `path` declares.

    PROJ does the transformation, and reads the sphere and the projection parameters
    from the raster itself, so a product delivered on another grid is projected onto
    that grid instead of silently landing on the wrong pixels. Pass the CRS of an
    already-open dataset when there is one. Cached: one transformer per CRS.
    """
    if crs is None:
        return _transformer(_crs_wkt_of(str(path)))
    return _transformer(crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs))


def lonlat_to_xy(lon, lat, tf: Transformer | None = None):
    """Degrees (lon, lat) -> metres (x, y) of the mosaic CRS, arrays or scalars.

    Longitudes are taken as the caller passes them: the [0°, 360°) convention of some
    MASCS files has to be folded onto [-180°, 180°) first, that is a file convention
    and not part of the projection.
    """
    tf = tf if tf is not None else mosaic_projector()
    x, y = tf.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


# ----------------------------------------------------------------- MASCS/VIRS


def read_mascs_dat(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Read any MASCS sidecar (`.dat` or `.csv`): they are all CSV.

    Forward `nrows`, `usecols`, `chunksize`, etc. to `pd.read_csv`. For the
    per-spectrum reflectance file `spectres-002.dat` the `waves` and
    `photom_iof` columns come back as raw strings; use `read_mascs_spectra`
    to get parsed numpy arrays instead.
    """
    return pd.read_csv(path, **read_csv_kwargs)


def read_mascs_spectra(
    path: str | Path = MASCS_SPECTRA_PATH,
    nrows: int | None = None,
    **read_csv_kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """Read `spectres-002.dat` and parse the stringified spectra.

    Returns `(waves, iof, meta)`:
    - `waves[i]`, `iof[i]` are 1-D arrays for spectrum `i`. Lengths can differ
      between spectra: Barraud's `q3 / q4` cleaning keeps a variable fraction
      of channels per spectrum, so do not assume a fixed bin count.
    - `meta` is a DataFrame with the remaining columns (`ref_id, id, obs_id`).

    The full file is ~30 GB CSV / 3.17 M rows; always pass `nrows` (or use
    `chunksize` via `read_csv_kwargs`) unless you intend a full-corpus read.
    """
    df = pd.read_csv(path, nrows=nrows, **read_csv_kwargs)
    waves = [np.asarray(ast.literal_eval(s), dtype=np.float64) for s in df["waves"]]
    iof = [np.asarray(ast.literal_eval(s), dtype=np.float64) for s in df["photom_iof"]]
    meta = df.drop(columns=["waves", "photom_iof"])
    return waves, iof, meta