#!/usr/bin/env python3
"""Extract a band subset from the mdis2vihi hyperspectral mosaic.

Separate adapter utility, NOT part of the training/inference deliverable.
Reads the Float32 hyperspectral GeoTIFF produced by scripts/04_predict_mosaic.py
and writes a smaller GeoTIFF with a user-chosen subset of bands. Preserves
CRS, transform, nodata. Tags R/G/B color interpretation when exactly 3 bands
are requested so QGIS opens the result as a natural-color image.

Bands can be specified either by 1-indexed band number (--bands) or by
wavelength in nm (--wavelengths), in which case the closest available band
is selected for each target wavelength.

Examples
--------
# Natural-color RGB at 750 / 560 / 435 nm (QGIS preview)
python scripts/tools/extract_bands.py \
    --input runs/final/mdis2vihi_global_final_deflate.tif \
    --output runs/final/mdis2vihi_preview_rgb.tif \
    --wavelengths 750 560 435

# Same result, specifying band indices directly
python scripts/tools/extract_bands.py \
    --input runs/final/mdis2vihi_global_final_deflate.tif \
    --output runs/final/mdis2vihi_preview_rgb.tif \
    --bands 91 53 28

# 8-band stack mimicking the MDIS WAC filter set
python scripts/tools/extract_bands.py \
    --input runs/final/mdis2vihi_global_final_deflate.tif \
    --output runs/final/mdis2vihi_mdis8.tif \
    --wavelengths 435 480 560 630 750 830 900 995
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


def parse_wavelength(description: str | None, band_index: int) -> float:
    """Return the wavelength (nm) of a band.

    Prefers parsing the rasterio band description (e.g. "750 nm"); falls back
    to the canonical 5 nm × [300, 1450 nm] grid formula
    λ = 300 + 5 × (band_index − 1).
    """
    if description:
        match = re.search(r"([-+]?\d*\.?\d+)", description)
        if match:
            return float(match.group(1))
    return 300.0 + 5.0 * (band_index - 1)


def closest_band(target_nm: float, wavelengths: list[float]) -> int:
    """Return the 1-indexed band whose wavelength is closest to target_nm."""
    idx = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - target_nm))
    return idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Source hyperspectral GeoTIFF")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination GeoTIFF (overwritten if it exists)")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--bands", type=int, nargs="+",
                           help="1-indexed band numbers to extract")
    selection.add_argument("--wavelengths", type=float, nargs="+",
                           help="Wavelengths in nm; closest band is selected")
    parser.add_argument("--compress", default="deflate",
                        help="GeoTIFF compression (default: deflate)")
    parser.add_argument("--predictor", type=int, default=2,
                        help="GeoTIFF predictor (default: 2, as used for the deliverable)")
    parser.add_argument("--blockxsize", type=int, default=512)
    parser.add_argument("--blockysize", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=1024,
                        help="Streaming chunk height in rows. Larger = less "
                             "per-chunk overhead, more RAM. Default 1024.")
    args = parser.parse_args()

    with rasterio.open(args.input) as src:
        wavelengths = [
            parse_wavelength(src.descriptions[i], i + 1) for i in range(src.count)
        ]

        if args.bands is not None:
            bands = list(args.bands)
            for b in bands:
                if not (1 <= b <= src.count):
                    raise ValueError(
                        f"--bands {b} out of range [1, {src.count}]"
                    )
        else:
            bands = [closest_band(t, wavelengths) for t in args.wavelengths]

        chosen_wl = [wavelengths[b - 1] for b in bands]
        print(f"source           : {args.input}")
        print(f"source bands     : {src.count}, dtype: {src.dtypes[0]}")
        print(f"selected bands   : {bands}")
        print(f"corresponding λ  : {['%.0f nm' % w for w in chosen_wl]}")

        profile = src.profile.copy()
        profile.update(
            count=len(bands),
            compress=args.compress,
            predictor=args.predictor,
            tiled=True,
            blockxsize=args.blockxsize,
            blockysize=args.blockysize,
        )
        est_bytes = src.width * src.height * len(bands) * 4
        if est_bytes > 3 * 1024**3:
            profile["BIGTIFF"] = "YES"

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(args.output, "w", **profile) as dst:
            # Stream chunked windows reading ALL requested bands in one shot.
            # A naive `src.read(b)` per band would decompress the whole source
            # N times when the input uses PIXEL interleave (gdal_translate's
            # default), orders of magnitude slower on a multi-GB DEFLATE TIFF.
            for r0 in range(0, src.height, args.chunk_rows):
                h = min(args.chunk_rows, src.height - r0)
                window = Window(0, r0, src.width, h)
                chunk = src.read(indexes=bands, window=window)
                dst.write(chunk, window=window)
                pct = 100.0 * (r0 + h) / src.height
                print(f"  rows {r0:>5}-{r0 + h:>5} / {src.height}  ({pct:5.1f} %)")

            for i, wl in enumerate(chosen_wl, 1):
                dst.set_band_description(i, f"{wl:.0f} nm")

            if len(bands) == 3:
                dst.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                )

    out_size_mb = args.output.stat().st_size / 1024**2
    print(f"\ndone: {args.output} ({out_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()