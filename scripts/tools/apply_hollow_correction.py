"""Apply the separate hollow-correction layer to the deliverable, into a NEW file.

A separate step, outside the production chain: the original deliverable is never modified.

Two modes:

* `rewrite` (default): streaming rewrite into a new mosaic, carrying the deliverable's
  profile over identically (DEFLATE/PREDICTOR=2, 512x256 tiles, PIXEL interleaving, BIGTIFF).
  Recompression being lossless, values outside the selection are preserved **bit for bit**;
  `--verify` measures it. Clean output, of a size comparable to the deliverable.
* `patch`: byte-for-byte copy, then rewrite of the affected tiles only. Faster, but libtiff
  cannot always put a recompressed tile back where it was: the file keeps as much dead space
  as there are rewritten tiles (~ +25 % here). Reserve it for tests.

Usage
-----
  python scripts/tools/apply_hollow_correction.py                    # rewrite + proof
  python scripts/tools/apply_hollow_correction.py --mode patch --reuse-copy
"""
from __future__ import annotations

import argparse
import json
import shutil
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
from mdis2vihi.correction.layer import (CorrectionNetwork, apply_layer,  # noqa: E402
                                      rewrite_with_layer, tiles_of)

DELIVERABLE = REPO / "runs/final/mdis2vihi_global_final_deflate.tif"
DEFAULT_OUT = REPO / "runs/final/mdis2vihi_global_final_postcorr_unionbb0.8_deflate.tif"
LAYER_DIR = REPO / "runs/final/correction"


def human(n):
    return f"{n/1e9:.1f} GB"


def copy_deliverable(src, dst, reuse=False):
    """Byte-for-byte copy (no decoding/re-encoding: this is what guarantees the deliverable
    is preserved exactly outside the corrected tiles)."""
    if dst.exists() and reuse:
        if dst.stat().st_size == src.stat().st_size:
            print(f"copy already present ({human(dst.stat().st_size)}) reused", flush=True)
            return
        print("copy present but of a different size -> copying again", flush=True)
    free = shutil.disk_usage(dst.parent).free
    if free < src.stat().st_size * 1.02:
        raise SystemExit(f"not enough space: {human(free)} free for "
                         f"{human(src.stat().st_size)} to copy.")
    t0 = time.time()
    print(f"copying {src.name} -> {dst.name} ({human(src.stat().st_size)})...", flush=True)
    shutil.copyfile(src, dst)
    print(f"  copy finished in {(time.time()-t0)/60:.1f} min", flush=True)


def verify(src, dst, layer, n_tiles=3, seed=42):
    """Proof of reversibility at RASTER level: on corrected tiles, compare source and output
    (a) outside the selection, expected difference EXACTLY 0; (b) inside the selection,
    difference =
    delta. Plus one never-touched tile, where everything must be identical."""
    rng = np.random.default_rng(seed)
    groups = list(tiles_of(layer))
    pick = rng.choice(len(groups), size=min(n_tiles, len(groups)), replace=False)
    touched = {(t[0], t[1]) for t in groups}
    res = {"corrected_tiles": [], "untouched_tile": None}
    with rasterio.open(src) as a, rasterio.open(dst) as b:
        bh, bw = a.block_shapes[0]
        for k in pick:
            tr, tc, sl = groups[int(k)]
            win = Window(tc * bw, tr * bh, min(bw, a.width - tc * bw),
                         min(bh, a.height - tr * bh))
            A, B = a.read(window=win), b.read(window=win)
            m = np.zeros(A.shape[1:], bool)
            m[sl.row.to_numpy() - tr * bh, sl.col.to_numpy() - tc * bw] = True
            d_off = np.abs(B[:, ~m] - A[:, ~m])
            d_on = np.abs(B[:, m] - A[:, m])
            res["corrected_tiles"].append(
                {"tile": [int(tr), int(tc)], "n_selected_pixels": int(m.sum()),
                 "max_diff_outside_selection": float(d_off.max()) if d_off.size else 0.0,
                 "max_diff_inside_selection": float(d_on.max()) if d_on.size else 0.0})
        # one never-touched tile
        for _ in range(200):
            tr, tc = int(rng.integers(0, a.height // bh)), int(rng.integers(0, a.width // bw))
            if (tr, tc) not in touched:
                break
        win = Window(tc * bw, tr * bh, bw, bh)
        A, B = a.read(window=win), b.read(window=win)
        res["untouched_tile"] = {"tile": [tr, tc], "max_diff": float(np.abs(B - A).max())}
    res["exact_reversibility"] = (
        all(t["max_diff_outside_selection"] == 0.0 for t in res["corrected_tiles"])
        and res["untouched_tile"]["max_diff"] == 0.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DELIVERABLE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--layer-dir", default=str(LAYER_DIR))
    ap.add_argument("--mode", choices=("rewrite", "patch"), default="rewrite")
    ap.add_argument("--chunk-cols", type=int, default=2560,
                    help="width of the rewrite windows (multiple of 512; "
                         "2560 ~ 0.6 GB peak memory)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and audit the correction without writing or copying anything")
    ap.add_argument("--reuse-copy", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--force-inplace", action="store_true",
                    help="patch mode: write into --source (deliverable not recoverable)")
    args = ap.parse_args()

    src, dst = Path(args.source), Path(args.out)
    ld = Path(args.layer_dir)
    cfg = json.loads((ld / "correction_config.json").read_text(encoding="utf-8"))
    layer = pd.read_parquet(ld / cfg["files"]["layer"])
    residual = CorrectionNetwork.from_checkpoint(REPO / cfg["residual"]["ckpt"])
    print(f"layer {cfg['selection']['source']}: {len(layer)} pixels, scale "
          f"{cfg['selection']['scale']}, residual rank {residual.rank}", flush=True)

    if args.dry_run:
        audit = apply_layer(src, layer, residual, dry_run=True)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    t0 = time.time()
    if args.mode == "rewrite":
        # The destination is deleted before writing, so the space it currently occupies
        # is part of the budget: counting only `free` would refuse every regeneration
        # once a previous output exists on a nearly full volume.
        free = shutil.disk_usage(dst.parent).free + (dst.stat().st_size if dst.exists() else 0)
        if free < src.stat().st_size * 1.02:
            raise SystemExit(f"not enough space: {human(free)} usable for "
                             f"{human(src.stat().st_size)} to write.")
        dst.unlink(missing_ok=True)                 # a truncated BIGTIFF would fail the write
        print(f"streaming rewrite -> {dst.name}...", flush=True)
        audit = rewrite_with_layer(src, dst, layer, residual, chunk_cols=args.chunk_cols)
        print(f"rewrite: {audit['n_windows']} windows, "
              f"{audit['n_pixels_written']} pixels corrected, {audit['duration_s']/60:.1f} min",
              flush=True)
    else:
        if args.force_inplace:
            dst = src
            print("!! writing IN PLACE into the deliverable (--force-inplace)", flush=True)
        else:
            copy_deliverable(src, dst, reuse=args.reuse_copy)
        audit = apply_layer(dst, layer, residual)
        print(f"patch: {audit['n_tiles']} tiles, {audit['n_pixels_written']} pixels, "
              f"{audit['duration_s']/60:.1f} min", flush=True)
    print(f"  max |delta| {audit['delta_max_abs']:.4f}, median dR996 "
          f"{audit['delta_med_996']:+.5f}, {audit['n_negative_after']} negative values",
          flush=True)

    ver = None
    if not args.no_verify and not args.force_inplace:
        ver = verify(src, dst, layer)
        print(f"raster reversibility: "
              f"{'EXACT' if ver['exact_reversibility'] else 'BROKEN'} "
              f"(outside the selection, on corrected tiles and on an untouched tile)", flush=True)

    out_cfg = dict(cfg)
    # as_posix(): the log stays identical whatever the operating system
    out_cfg["application"] = {"mode": args.mode, "output": dst.relative_to(REPO).as_posix(),
                              "source": src.relative_to(REPO).as_posix(),
                              "size_bytes": dst.stat().st_size,
                              "source_size_bytes": src.stat().st_size,
                              "audit": audit, "verification": ver,
                              "total_duration_min": round((time.time() - t0) / 60, 1)}
    (ld / "correction_applied.json").write_text(json.dumps(out_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ncorrected mosaic: {dst} ({human(dst.stat().st_size)})")
    print(f"log             : {ld/'correction_applied.json'}")


if __name__ == "__main__":
    main()
