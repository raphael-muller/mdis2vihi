"""Build the `unionbb0.8` hollow-correction layer for the deliverable.

A separate step, outside the production chain (`scripts/01->05`): it touches neither the
model, nor the data, nor the mosaic. It produces a detachable sparse layer and its "where"
product.

    output = deliverable + g(lon,lat,reflectance) * [ c(x) @ B.T ]
    g      = 0.42 * [ (Thomas 2016 union HORNET 0.8) inter bright+blue(global Mercury ref.) ]

Three steps: rasterise the spatial stage on the deliverable grid, since a per-pixel
point-in-polygon test over 265 M pixels is out of budget; apply the bright+blue stage
against a fixed global Mercury reference and read the low-rank coefficients; audit that
reference, which is calibrated at MASCS footprint scale and applied to 665 m pixels.

The build reads only the documented input data and the artefacts committed under
`runs/final/`. Rebuilding must select the same pixels as the committed layer.

Outputs (in `--out`, default `runs/final/correction/`): `hollow_selection_spatial.tif`,
`hollow_selection_unionbb0.8.tif`, `hollow_layer_unionbb0.8.parquet`,
`residual_basis_rank2.npz`, `correction_config.json`.

Usage: python scripts/tools/build_hollow_correction.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")
from mdis2vihi.correction import selection as S  # noqa: E402
from mdis2vihi.correction.layer import CorrectionNetwork, build_layer  # noqa: E402

DELIVERABLE = REPO / "runs/final/mdis2vihi_global_final_deflate.tif"
MDIS = REPO / ("data/raw/mdis_mosaic/MDIS_MDR_20170512_PDS16_64ppd_"
               "equirectangular_withbackplanes.tif")
PAIRS = REPO / "data/processed/pairs.parquet"
BASE_MODEL_CKPT = REPO / "runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt"
# The trained correction network, versioned next to the layer so that rebuilding works
# from a bare checkout. Produced by scripts/08_train_residual.py.
RESIDUAL_STATS = REPO / "runs/final/correction/residual_stats.json"


def md5(path, nbytes=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read() if nbytes is None else f.read(nbytes))
    return h.hexdigest()[:12]


def audit_bb_ref(bb_ref, z_thresh, mdis_path, n_rows=24, seed=42):
    """Measure the effect of the bright+blue reference (calibrated on I/F averaged at MASCS
    footprint scale) once applied to 665 m PIXELS: spread, and how much of the background
    passes. A background passing too massively would make the spectral stage useless."""
    rng = np.random.default_rng(seed)
    with rasterio.open(mdis_path) as ds:
        rows = rng.choice(ds.height, size=n_rows, replace=False)
        r749, r433 = [], []
        for r in rows:
            d = ds.read([1, 5], window=Window(0, int(r), ds.width, 1)).astype(np.float64)
            r433.append(d[0, 0]), r749.append(d[1, 0])
    r433, r749 = np.concatenate(r433), np.concatenate(r749)
    ok = np.isfinite(r749) & (r749 > 0) & np.isfinite(r433) & (r433 > -1e30)
    r433, r749 = r433[ok], r749[ok]
    ratio = r433 / r749
    bb = S.bright_blue_mask(r749, ratio, bb_ref, z_thresh=z_thresh)
    return {"n_background_pixels": int(ok.sum()),
            "ref_footprints": {"mu_r749": bb_ref[0], "sd_r749": bb_ref[1],
                               "mu_ratio": bb_ref[2], "sd_ratio": bb_ref[3]},
            "pixels_665m": {"mu_r749": float(r749.mean()), "sd_r749": float(r749.std()),
                            "mu_ratio": float(ratio.mean()), "sd_ratio": float(ratio.std())},
            "sigma_pixel_over_footprint_r749": round(float(r749.std()) / bb_ref[1], 3),
            "sigma_pixel_over_footprint_ratio": round(float(ratio.std()) / bb_ref[3], 3),
            "background_passing_bright_blue_pct": round(100.0 * float(bb.mean()), 3)}


def main():
    # The five geometry/strength knobs below are choices of this project, fixed at the
    # values the delivered layer was built with.
    #
    # `--scale` is calibrated, not chosen: `calibrate_hollow_scale.py` derives it and
    # writes `scale_calibration.json`. No single value fits every crater (0.300 at
    # Dominici to 0.740 at Hopper), so the deployed value is their median; leave-one-out
    # puts the residual contrast error at 0.063 in the median, against 0.175 uncorrected.
    #
    # Measured on the crater footprints, `--z-thresh` does not change hollow recall
    # anywhere in [0, 1], it only trades background false positives (12.0 % at 0, 7.7 % at
    # 1). Of the geometry knobs only `--margin` bites: 1.0 gives 65 % recall, 1.5 gives
    # 82 % at 9.7 % false positives, 2.0 gives 83 % at 17.8 %. `--floor-km`, `--buffer-km`
    # and `--max-area` move recall by less than a point.
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/final/correction")
    ap.add_argument("--scale", type=float, default=0.422)   # calibrate_hollow_scale.py
    ap.add_argument("--z-thresh", type=float, default=0.5)
    ap.add_argument("--buffer-km", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--floor-km", type=float, default=3.0)
    ap.add_argument("--hornet", default="0.8")
    ap.add_argument("--residual-stats", default=str(RESIDUAL_STATS))
    ap.add_argument("--grid-from", default=str(DELIVERABLE),
                    help="raster giving the reference grid (default: the deliverable)")
    ap.add_argument("--bb-ref", choices=("footprints", "pixels"), default="footprints",
                    help="calibration of the bright+blue reference: 'footprints' = the "
                         "configuration used for the deliverable (pairs.parquet); 'pixels' = variant calibrated "
                         "on the 665 m pixels, for measurement only)")
    args = ap.parse_args()

    OUT = REPO / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    transform, crs, W, H = S.grid_from(args.grid_from)
    print(f"reference grid: {Path(args.grid_from).name}: {W}x{H}, "
          f"{transform.a:.6f} m/px", flush=True)

    # ---------------- 1. rasterised spatial stage ----------------
    thomas, hornet, gmeta = S.spatial_polygons(margin=args.margin, floor_km=args.floor_km,
                                               buffer_km=args.buffer_km,
                                               hornet_variant=args.hornet)
    print(f"spatial selection: {len(thomas)} Thomas disks (R {gmeta['thomas_R_km'][0]:.1f}-"
          f"{gmeta['thomas_R_km'][2]:.1f} km) union {len(hornet)} HORNET {args.hornet} polygons "
          f"(+{args.buffer_km} km), {gmeta['hornet']}", flush=True)
    mask = S.rasterize_spatial(thomas + hornet, transform, W, H)
    n_spatial = int(mask.sum())
    print(f"  -> {n_spatial} pixels on ({100.0*n_spatial/(W*H):.4f} % of the mosaic), "
          f"{time.time()-t0:.0f} s", flush=True)
    S.write_mask(OUT / "hollow_selection_spatial.tif", mask, transform, crs,
                 description="spatial selection Thomas union HORNET 0.8 (1 = inside an object)")

    # ---------------- 2. bright+blue stage + coefficients ----------------
    bb_ref = (S.bright_blue_ref(PAIRS) if args.bb_ref == "footprints"
              else S.bright_blue_ref_pixels(MDIS))
    print(f"bright+blue reference (global Mercury background, {args.bb_ref} calibration): "
          f"R749 {bb_ref[0]:.4f}+/-{bb_ref[1]:.4f}, ratio {bb_ref[2]:.4f}+/-{bb_ref[3]:.4f}",
          flush=True)

    stats = json.loads(Path(args.residual_stats).read_text(encoding="utf-8"))
    res_ckpt = Path(stats["best_ckpt"])
    if not res_ckpt.is_absolute():
        res_ckpt = REPO / res_ckpt
    residual = CorrectionNetwork.from_checkpoint(res_ckpt)
    print(f"residual: rank {residual.rank}, {residual.meta['n_params']} coefficient "
          f"parameters, fixed basis (231, {residual.rank})", flush=True)

    layer, lstats = build_layer(mask, MDIS, residual, bb_ref,
                                image_count_mean=stats["image_count_mean"],
                                image_count_std=stats["image_count_std"],
                                scale=args.scale, z_thresh=args.z_thresh)
    print(f"layer: {lstats['n_pixels_corrected']} pixels corrected "
          f"({lstats['fraction_mosaic_pct']} % of the mosaic); "
          f"{lstats['n_pixels_rejected_bright_blue']} rejected by the bright+blue stage, "
          f"{lstats['n_pixels_invalid_mdis']} invalid (MDIS nodata)", flush=True)

    # ---------------- 3. audit of the bright+blue reference ----------------
    bb_audit = audit_bb_ref(bb_ref, args.z_thresh, MDIS)
    print(f"reference audit: sigma(665 m pixels)/sigma(footprints) = "
          f"{bb_audit['sigma_pixel_over_footprint_r749']}x (R749), "
          f"{bb_audit['sigma_pixel_over_footprint_ratio']}x (ratio); background passing "
          f"bright+blue = {bb_audit['background_passing_bright_blue_pct']} %", flush=True)

    # ---------------- outputs ----------------
    layer.to_parquet(OUT / "hollow_layer_unionbb0.8.parquet", index=False)
    np.savez(OUT / "residual_basis_rank2.npz", B=residual.B.numpy(),
             wavelength_nm=np.arange(300.0, 1455.0, 5.0))
    selection_raster = np.zeros((H, W), np.float32)
    selection_raster[layer.row.to_numpy(), layer.col.to_numpy()] = layer.g.to_numpy()
    S.write_mask(OUT / "hollow_selection_unionbb0.8.tif", selection_raster, transform, crs,
                 description=f"correction strength = {args.scale}*[(Thomas union HORNET 0.8) "
                             "inter bright+blue]")

    config = {
        "description": "Hollow-contrast correction layer, unionbb0.8 configuration. "
                       "Built and applied as described in docs/REPRODUCTION.md, "
                       "steps 6-10.",
        "formula": "output = deliverable + g(lon,lat,reflectance)*[c(x) @ B.T] ; "
                   "g = scale*[(Thomas union HORNET 0.8) inter bright+blue(global Mercury ref.)]",
        "base_mosaic": str(DELIVERABLE.relative_to(REPO)),
        "base_model_ckpt": str(BASE_MODEL_CKPT.relative_to(REPO)),
        "residual": {"ckpt": stats["best_ckpt"].replace(str(REPO) + "/", ""),
                     "rank": residual.rank, "coef_hidden": stats["coef_hidden"],
                     # fingerprint of the SVD basis as just written, so that the value
                     # recorded here always designates a file this repository carries
                     "base_md5": md5(OUT / "residual_basis_rank2.npz"),
                     "n_params_coef": residual.meta["n_params"]},
        "selection": {"source": "unionbb0.8", "scale": args.scale, "z_thresh": args.z_thresh,
                 "buffer_km": args.buffer_km, "margin": args.margin,
                 "floor_km": args.floor_km, **gmeta,
                 "thomas_md5": md5(S.THOMAS_CSV), "hornet_md5": md5(S._hornet_csv(args.hornet)),
                 "bright_blue_ref": {"mu_r749": bb_ref[0], "sd_r749": bb_ref[1],
                                     "mu_ratio": bb_ref[2], "sd_ratio": bb_ref[3],
                                     "calibration": args.bb_ref,
                                     "source": (str(PAIRS.relative_to(REPO))
                                                if args.bb_ref == "footprints"
                                                else "MDIS 665 m pixels (48 rows)")}},
        "image_count": {"mean": stats["image_count_mean"],
                        "std": stats["image_count_std"],
                        "source": "runs/final/image_count_stats.json"},
        "grid": {"width": W, "height": H, "n_bands": 231,
                 "res_m": abs(transform.a), "crs": crs.to_string()},
        "layer": lstats, "bright_blue_ref_audit": bb_audit,
        "files": {"layer": "hollow_layer_unionbb0.8.parquet",
                  "selection": "hollow_selection_unionbb0.8.tif",
                  "spatial_selection": "hollow_selection_spatial.tif",
                  "basis": "residual_basis_rank2.npz"},
        "build_duration_s": round(time.time() - t0, 1),
    }
    (OUT / "correction_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    size = (OUT / "hollow_layer_unionbb0.8.parquet").stat().st_size / 1e6
    print(f"\nlayer written: {OUT/'hollow_layer_unionbb0.8.parquet'} ({size:.1f} MB)")
    print(f"configuration: {OUT/'correction_config.json'}")
    print(f"total time   : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()