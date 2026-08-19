"""Global Mercury-consistency check on the delivered mosaic (158 GB DEFLATE GeoTIFF).

Band statistics over the produced mosaic (per-band mean/median spectrum, key bands,
red slope 750/435 nm), read by tile-aligned windows scattered across the globe rather
than by random-pixel access, which on a file with no overviews would decompress far
more than it reads.

    python scripts/tools/analysis_mosaic_consistency.py [N_ROWS] [N_COLS]

The window grid defaults to 8 x 8, a quick pass. The committed reference statistics
come from a denser grid: 352 windows over 41 M valid pixels (see
`runs/final/eval/mosaic_consistency_final.json`), so a default run gives the same
figures only to within sampling noise.

Outputs: runs/final/eval/mosaic_band_stats_final.csv
         runs/final/eval/mosaic_consistency_final.json   (+ stdout summary)
"""
from __future__ import annotations
import sys, json
from pathlib import Path

# The default Windows console is cp1252 and crashes on any print() containing a
# mathematical symbol. Degrade instead of raising.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")
import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[2]
FP = REPO / "runs/final/mdis2vihi_global_final_deflate.tif"
NGRID_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NGRID_COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TILE_H, TILE_W = 256, 512
NODATA_THR = -1e30
RESERVOIR = 20000
SEED = 42

GRID = np.arange(300.0, 1450.0 + 1e-6, 5.0)  # 231


def main():
    rng = np.random.default_rng(SEED)
    with rasterio.open(FP) as ds:
        H, W, B = ds.height, ds.width, ds.count
        row0 = np.unique(np.clip(np.linspace(0, H - TILE_H, NGRID_ROWS).astype(int), 0, H - 1))
        col0 = np.unique(np.clip(np.linspace(0, W - TILE_W, NGRID_COLS).astype(int), 0, W - 1))
        sums = np.zeros(B, np.float64)
        sqs = np.zeros(B, np.float64)
        nval = 0
        per_win = max(1, RESERVOIR // max(1, len(row0) * len(col0)))
        samples = []
        nwin = 0
        for r in row0:
            h = min(TILE_H, H - r)
            for c in col0:
                w = min(TILE_W, W - c)
                arr = ds.read(window=Window(c, r, w, h)).astype(np.float32)  # (B,h,w)
                flat = arr.reshape(B, -1)
                valid = np.all(np.isfinite(flat) & (flat > NODATA_THR), axis=0)
                v = flat[:, valid]  # (B, nv)
                nv = v.shape[1]
                nwin += 1
                if nv:
                    sums += v.sum(1)
                    sqs += (v.astype(np.float64) ** 2).sum(1)
                    nval += nv
                    take = min(per_win, nv)
                    sel = rng.choice(nv, size=take, replace=False)
                    samples.append(v[:, sel].T)  # (take, B)
        mean = sums / nval
        std = np.sqrt(np.maximum(sqs / nval - mean ** 2, 0.0))
        sample = np.concatenate(samples, axis=0)
        rfill = sample.shape[0]
        median = np.median(sample, axis=0)

    def at(wl, vec):
        b = int(round((wl - 300) / 5))
        return float(vec[b])
    R435m, R750m, R1450m, R300m, R996m = (at(w, median) for w in (435, 750, 1450, 300, 996))
    red_slope = (R750m - R435m) / R435m * 100.0
    out = {
        "n_windows": nwin, "n_valid_px": int(nval), "reservoir": int(rfill),
        "median_iof_allbands": float(np.median(median)),
        "R435_median": R435m, "R750_median": R750m, "R1450_median": R1450m,
        "R300_median": R300m, "R996_median": R996m,
        "red_slope_750_435_pct": red_slope,
        "ratio_750_435": R750m / R435m,
    }
    # write per-band csv
    import csv
    ev = REPO / "runs/final/eval/mosaic_band_stats_final.csv"
    with open(ev, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["wavelength_nm", "mean", "median", "std"])
        for i in range(len(GRID)):
            wcsv.writerow([GRID[i], mean[i], median[i], std[i]])
    (REPO / "runs/final/eval/mosaic_consistency_final.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
