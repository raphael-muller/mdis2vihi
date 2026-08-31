"""Selection for the correction layer: catalogues, spatial rasterisation, bright+blue stage.

Two stages, as fixed by the `unionbb0.8` configuration:

* **spatial**: membership of a Thomas 2016 disk (geodesic radius
  `R = max(sqrt(area/pi)*margin, floor)`) or of a cleaned HORNET 0.8 polygon (Bickel),
  dilated by `buffer_km`; rasterised ONCE on the deliverable grid, because a
  per-pixel point-in-polygon test over 265 M pixels is out of budget;
* **spectral**: a "bright+blue" anomaly (bright R749, blue R433/R749) measured on the
  MDIS mosaic against a **fixed** reference (the global Mercury background). This is the
  stage that protects faculae, which are bright but RED, including those co-located with
  hollows inside the Thomas disks.

This module uses only the 608-group table with geodesic areas (Thomas et al. 2016,
table ST1), since each disk radius comes from its group's area; the 445-row centre table
of Thomas et al. (2014) is what `scripts/06_build_hollow_pool.py` uses (docs/DATA.md).

The two stages are self-contained: they need the two catalogues and the MDIS mosaic,
nothing else. Rebuilding must select the same pixels as the committed layer, and that
identity is what verifies the rasterisation.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely import affinity, box
from shapely.geometry import Polygon

from ..data.io import mosaic_projector

R_MERC_KM = 2439.4

REPO = Path(__file__).resolve().parents[3]
THOMAS_CSV = REPO / "data/raw/hollow/1-s2.0-S0019103516302469-mmc4.txt"
HORNET_DIR = REPO / "data/raw/hollow/hornet"

# Bounds of the deliverable's equirectangular grid (no latitude beyond +/-90).
_LAT_MAX = 90.0


# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------
def load_thomas(path: Path | None = None):
    """Thomas et al. (2016) catalogue, supplementary table ST1 (608 hollow groups)
    -> (lon, lat, area km^2). Not the 445-row centre table of Thomas et al. (2014)."""
    c = pd.read_csv(path or THOMAS_CSV, sep="\t", skiprows=2)
    c = c.rename(columns={"Central.Longitude.dec.degrees": "lon",
                          "Central.Latitude.dec.degrees": "lat",
                          "Geodesic.area.km.2": "area"})[["lon", "lat", "area"]].dropna()
    return c.lon.to_numpy(float), c.lat.to_numpy(float), c.area.to_numpy(float)


def _wkt_rings(s):
    """Tolerant WKT parser: HORNET polygons have UNCLOSED rings, which
    `shapely.wkt.loads` rejects. Rebuild each coordinate group instead."""
    s = str(s).strip().upper()
    if not s.startswith(("POLYGON", "MULTIPOLYGON")):
        return []
    out = []
    for grp in re.findall(r"\(([^()]+)\)", s):
        pts = []
        for pair in grp.split(","):
            xy = pair.split()
            if len(xy) >= 2:
                try:
                    pts.append((float(xy[0]), float(xy[1])))
                except ValueError:
                    pass
        if len(pts) >= 3:
            out.append(Polygon(pts))
    return out


def _hornet_csv(variant="0.8"):
    tag = "08" if str(variant).endswith("8") else "07"
    hits = sorted(HORNET_DIR.glob(f"*hollows_{tag}*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"HORNET {variant} catalogue not found in {HORNET_DIR} "
            "(Bickel et al. 2025 data set, CSV with a WKT column, see docs/DATA.md).")
    return hits[0]


def _poly_area_km2(poly):
    """Area of a lon/lat polygon in km^2 (equirectangular, cos(lat) correction)."""
    return float(poly.area) * (R_MERC_KM * np.pi / 180.0) ** 2 * float(
        np.cos(np.radians(poly.centroid.y)))


def load_hornet(variant="0.8", clean=True, max_area_km2=50.0, path: Path | None = None):
    """HORNET polygons (lon in [-180,180] / lat), cleaned the same way each time.

    Cleaning: repair self-intersections with `buffer(0)`, drop oversized polygons
    (> `max_area_km2`), and drop polygons straddling the +/-180 seam.

    `max_area_km2 = 50` is a choice, not a threshold read off the data: the area
    distribution is a continuum (64 polygons between 25 and 50 km^2, 27 between 50 and
    100, only 3 above 5 000). At 50 it rejects 58 of the 3 263.
    """
    with open(path or _hornet_csv(variant), "r", encoding="latin-1") as f:
        head = f.readline()
    sep = ";" if head.count(";") >= head.count(",") else ","
    df = pd.read_csv(path or _hornet_csv(variant), sep=sep, encoding="latin-1")
    col = next((c for c in df.columns
                if any(k in str(c).lower() for k in ("geometry", "polygon", "wkt", "geom"))), None)
    if col is None:
        raise ValueError(f"No WKT column in {_hornet_csv(variant)}: {list(df.columns)}")

    polys, n_seam, n_degen = [], 0, 0
    for w in df[col].dropna():
        for p in _wkt_rings(w):
            xy = np.asarray(p.exterior.coords, float)
            if len(xy) < 4:
                continue
            lon = np.where(xy[:, 0] > 180.0, xy[:, 0] - 360.0, xy[:, 0])
            if float(lon.max() - lon.min()) > 180.0:
                n_seam += 1
                continue
            q = Polygon(np.column_stack([lon, xy[:, 1]]))
            if not q.is_valid:
                q = q.buffer(0)
            # a self-intersecting "bow tie" repairs into a MultiPolygon and is rejected;
            # keeping its largest lobe would add 5 polygons to the delivered 3 205
            if not q.is_empty and q.geom_type == "Polygon":
                polys.append(q)
            else:
                n_degen += 1

    stats = {"n_raw": len(polys) + n_seam + n_degen, "n_seam": n_seam,
             "n_degenerate": n_degen, "n_oversized": 0}
    if clean and max_area_km2 is not None:
        kept = [p for p in polys if _poly_area_km2(p) <= max_area_km2]
        stats["n_oversized"] = len(polys) - len(kept)
        polys = kept
    stats["n_kept"] = len(polys)
    return polys, stats


# ---------------------------------------------------------------------------
# Geometry: geodesic disks, +/-180 seam, projection
# ---------------------------------------------------------------------------
def geodesic_disk(lon0, lat0, r_km, n=128):
    """Lon/lat polygon of the geodesic disk of radius `r_km`, an EXACT circle on the
    sphere, not the approximate ellipse of the equirectangular plane. Longitudes stay
    continuous around the centre (they may leave [-180, 180]: the seam is handled by
    `split_seam`)."""
    d = float(r_km) / R_MERC_KM
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    la0, lo0 = np.radians(float(lat0)), np.radians(float(lon0))
    lat = np.arcsin(np.sin(la0) * np.cos(d) + np.cos(la0) * np.sin(d) * np.cos(th))
    lon = lo0 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la0),
                           np.cos(d) - np.sin(la0) * np.sin(lat))
    return Polygon(np.column_stack([np.degrees(lon), np.degrees(lat)]))


def split_seam(poly):
    """Bring a lon/lat polygon back into [-180, 180] x [-90, 90], cutting it at the
    +/-180 seam if needed (returns 1 or 2 pieces)."""
    clip = box(-180.0, -_LAT_MAX, 180.0, _LAT_MAX)
    out = []
    for shift in (-360.0, 0.0, 360.0):
        q = affinity.translate(poly, xoff=shift).intersection(clip)
        if q.is_empty:
            continue
        for g in getattr(q, "geoms", [q]):
            if g.geom_type == "Polygon" and not g.is_empty:
                out.append(g)
    return out


def to_projected(poly, tf=None):
    """Lon/lat polygon -> the projected coordinates of the reference grid, in metres.

    PROJ does the transformation (`tf`, or a transformer built from the MDIS mosaic's
    own CRS), so the grid's own projection parameters are used rather than a formula
    written out here.

    INTERIOR rings are preserved: dilating a crescent-shaped HORNET polygon by 1 km can
    close it into a ring, and filling that hole would switch on pixels the analytic selection
    leaves off (up to ~4 km from the edge, measured).
    """
    tf = tf if tf is not None else mosaic_projector()

    def ring(c):
        xy = np.asarray(c.coords, float)
        x, y = tf.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    return Polygon(ring(poly.exterior), [ring(r) for r in poly.interiors])


def spatial_polygons(margin=1.5, floor_km=3.0, buffer_km=1.0, hornet_variant="0.8",
                     n_vertices=128, max_area_km2=50.0):
    """Full geometry of the `Thomas union HORNET` spatial stage, in lon/lat.

    Thomas: geodesic disk of radius `max(sqrt(area/pi)*margin, floor_km)`. The floor
    sets the radius for 58 % of the 608 groups (median raw radius 2.29 km); 3 km is about
    4.5 pixels, below which a group rasterises to almost nothing. `margin` and `floor_km`
    are choices of this project. Of the two, only `margin` measurably changes what the
    gate selects, see `scripts/tools/build_hollow_correction.py`.
    HORNET: cleaned polygon, dilated by `buffer_km` converted to **degrees** and applied
    in lon/lat, which under-dilates at high latitude. Kept as is because it is what the
    delivered layer was built with.
    """
    lon, lat, area = load_thomas()
    R = np.maximum(np.sqrt(area / np.pi) * margin, floor_km)
    thomas = [geodesic_disk(lo, la, r, n=n_vertices) for lo, la, r in zip(lon, lat, R)]

    hornet, hstats = load_hornet(hornet_variant, max_area_km2=max_area_km2)
    if buffer_km > 0:
        bd = float(buffer_km) / (R_MERC_KM * np.pi / 180.0)      # km -> degrees
        hornet = [p.buffer(bd) for p in hornet]

    meta = {"n_thomas": len(thomas), "thomas_R_km": [float(R.min()), float(np.median(R)),
                                                     float(R.max())],
            "margin": margin, "floor_km": floor_km, "buffer_km": buffer_km,
            "hornet_variant": hornet_variant, "hornet": hstats,
            "n_disk_vertices": n_vertices}
    return thomas, hornet, meta


# ---------------------------------------------------------------------------
# Rasterisation on the deliverable grid
# ---------------------------------------------------------------------------
def grid_from(path):
    """(transform, crs, width, height) of the reference grid (deliverable or MDIS mosaic)."""
    with rasterio.open(path) as ds:
        return ds.transform, ds.crs, ds.width, ds.height


def rasterize_spatial(polys_lonlat, transform, width, height, crs=None):
    """Spatial stage -> uint8 mask (1 = pixel centre inside a catalogue object).

    `crs` is the CRS of the grid `transform` belongs to, as returned by `grid_from`;
    the polygons are projected into it. `all_touched=False`: the test is on the pixel
    CENTRE, as the point-in-polygon selection it replaces tested one footprint centre
    at a time.
    """
    tf = mosaic_projector(crs)
    shapes = []
    for p in polys_lonlat:
        for q in split_seam(p):
            shapes.append((to_projected(q, tf), 1))
    if not shapes:
        raise ValueError("no polygon to rasterise")
    return features.rasterize(shapes, out_shape=(height, width), transform=transform,
                              fill=0, all_touched=False, dtype="uint8")


def write_mask(path, arr, transform, crs, dtype=None, nodata=None, description=None):
    """Write a single-band mask on the reference grid (DEFLATE, tiled)."""
    arr = arr if dtype is None else arr.astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype.name, crs=crs, transform=transform,
                       nodata=nodata, compress="deflate", predictor=2, tiled=True,
                       blockxsize=512, blockysize=256, BIGTIFF="IF_SAFER") as dst:
        dst.write(arr, 1)
        if description:
            dst.set_band_description(1, description)
    return Path(path)


# ---------------------------------------------------------------------------
# Bright+blue spectral stage
# ---------------------------------------------------------------------------
def bright_blue_ref(pairs_path):
    """Bright+blue reference = the GLOBAL Mercury background, computed once on the
    training pairs (`mdis_iof`). Returns (mu_R749, sigma_R749, mu_ratio, sigma_ratio).

    The reference is global and fixed: recalibrating it on the pixels it is applied to
    widens the spread so much that the spectral stage stops selecting anything. It is
    calibrated on footprint averages (1-5 km) and applied to 665 m pixels; the layer
    builder measures and reports that discrepancy.
    """
    X = np.stack(pd.read_parquet(pairs_path, columns=["mdis_iof"]).mdis_iof.to_list()).astype(float)
    r749 = X[:, 4]
    ratio = np.divide(X[:, 0], r749, out=np.full(len(r749), np.nan), where=r749 > 0)
    return (float(np.nanmean(r749)), float(np.nanstd(r749)),
            float(np.nanmean(ratio)), float(np.nanstd(ratio)))


def bright_blue_ref_pixels(mdis_path, n_rows=48, seed=42):
    """Reference variant calibrated on the 665 m PIXELS themselves (a sample of rows of
    the MDIS mosaic) rather than on the MASCS footprints.

    **Not what the deliverable uses**, provided to measure the effect of the
    calibration scale: at pixel scale the spread of the R433/R749 ratio is several times
    larger than at footprint scale, so a threshold expressed in footprint sigmas is far
    less selective on blueness.
    """
    import rasterio as _rio
    from rasterio.windows import Window as _Win
    rng = np.random.default_rng(seed)
    with _rio.open(mdis_path) as ds:
        rows = rng.choice(ds.height, size=n_rows, replace=False)
        a, b = [], []
        for r in rows:
            d = ds.read([1, 5], window=_Win(0, int(r), ds.width, 1)).astype(np.float64)
            a.append(d[0, 0])
            b.append(d[1, 0])
    r433, r749 = np.concatenate(a), np.concatenate(b)
    ok = np.isfinite(r749) & (r749 > 0) & np.isfinite(r433) & (r433 > -1e30)
    r433, r749 = r433[ok], r749[ok]
    ratio = r433 / r749
    return (float(r749.mean()), float(r749.std()), float(ratio.mean()), float(ratio.std()))


def bright_blue_mask(r749, ratio, ref, z_thresh=0.5):
    """Bright+blue membership: R749 and R433/R749 both above `z_thresh` sigmas of the
    reference."""
    mu749, sd749, mur, sdr = ref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z749 = (np.asarray(r749, float) - mu749) / (sd749 + 1e-9)
        zr = (np.asarray(ratio, float) - mur) / (sdr + 1e-9)
    return (z749 > z_thresh) & (zr > z_thresh)
