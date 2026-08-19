"""Apply the trained MLP pixel-by-pixel over the MDIS mosaic.

Usage
-----
Quick test on a small ROI first (final 9-input model -- emission flags required):
    python scripts/04_predict_mosaic.py \\
        --ckpt runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt \\
        --emission-band 9 --emission-stats runs/final/emission_stats.json \\
        --output runs/final/predict_roi.tif \\
        --roi 11000 5500 1024 1024

Full mosaic (~245 GB uncompressed -- write with --compress none, compress separate;
on the cluster use slurm/predict_final.sbatch then slurm/compress_final.sbatch):
    python scripts/04_predict_mosaic.py \\
        --ckpt runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt \\
        --emission-band 9 --emission-stats runs/final/emission_stats.json \\
        --output runs/final/mdis2vihi_global_final.tif

The output GeoTIFF is the project deliverable: Float32, 231 bands on
5 nm x [300, 1450 nm], CRS and transform inherited from the MDIS source,
nodata propagated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

from mdis2vihi.data.io import MDIS_MOSAIC_PATH
from mdis2vihi.inference.mosaic_predict import predict_mosaic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint path")
    ap.add_argument("--input", type=Path, default=MDIS_MOSAIC_PATH, help="MDIS source mosaic")
    ap.add_argument("--output", type=Path, required=True, help="Output GeoTIFF path")
    ap.add_argument("--tile-rows", type=int, default=32, help="Rows per streaming chunk")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--compress", default="lzw", choices=["lzw", "deflate", "none"])
    ap.add_argument("--predictor", type=int, default=None, choices=[1, 2, 3],
                    help="TIFF predictor (default: 3 for deflate, 2 for lzw, 1 otherwise). "
                         "Do NOT combine --compress deflate with predictor 3 here: it "
                         "heap-corrupts libgdal in the chunked writer. Write with "
                         "--compress none and compress afterwards (docs/CLUSTER.md).")
    ap.add_argument("--blockxsize", type=int, default=512, help="Output GeoTIFF internal tile width")
    ap.add_argument("--blockysize", type=int, default=None,
                    help="Output GeoTIFF internal tile height (default: tile_rows, "
                         "so chunk writes align with internal TIFF tiles)")
    ap.add_argument("--forward-batch", type=int, default=200_000, help="Max pixels per model.forward")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("COL", "ROW", "W", "H"),
                    default=None, help="Optional sub-window to process")
    ap.add_argument("--emission-band", type=int, default=None,
                    help="1-indexed mosaic band of the emission angle (9 for the MDIS "
                         "mosaic) to feed as a 9th input. Requires --emission-stats. "
                         "Omit for a model without the emission input.")
    ap.add_argument("--emission-stats", type=Path, default=None,
                    help="emission_stats.json (from scripts/03_train_final.py) holding "
                         "emission_mean / emission_std used to z-standardize band 9.")
    args = ap.parse_args()

    emission_stats = None
    if args.emission_band is not None:
        if args.emission_stats is None:
            ap.error("--emission-band requires --emission-stats")
        s = json.loads(Path(args.emission_stats).read_text(encoding="utf-8"))
        emission_stats = (s["emission_mean"], s["emission_std"])
        print(f"9-input model: emission band {args.emission_band}, "
              f"z-stats mean={emission_stats[0]:.4f} std={emission_stats[1]:.4f}")

    compress = None if args.compress == "none" else args.compress
    out = predict_mosaic(
        ckpt_path=args.ckpt,
        input_mosaic=args.input,
        output_path=args.output,
        tile_rows=args.tile_rows,
        device=args.device,
        compress=compress,
        blockxsize=args.blockxsize,
        blockysize=args.blockysize,
        predictor=args.predictor,
        forward_batch=args.forward_batch,
        roi=tuple(args.roi) if args.roi else None,
        emission_band=args.emission_band,
        emission_stats=emission_stats,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()