"""The sparse correction layer: the trained network, how the layer is built, how it is
applied.

The correction is non-zero on only ~0.04 % of the mosaic pixels, and the residual is
constrained to rank 2: the whole layer therefore amounts to **five numbers per corrected
pixel** (`row, col, g, c0, c1`) plus a fixed 231x2 basis `B`, a few megabytes, easy to
version, and detachable from the deliverable.

    delta(pixel) = g * [ (c0, c1) @ B.T ]        (231 bands)
    output       = deliverable + delta

Since the deliverable is exactly the output of the fixed base model, there is nothing to
recompute: outside the selection `delta = 0` and the deliverable's bytes are untouched:
reversibility holds by construction, not by measurement.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window

N_BANDS = 231
GRID_NM = np.arange(300.0, 1450.0 + 5.0, 5.0)
NODATA_THRESHOLD = -1e30
MDIS_INPUT_BANDS = (1, 2, 3, 4, 5, 6, 7, 8, 9)      # 8 I/F + emission angle


def coef_columns(df):
    """Coefficient columns `c0, c1, ...`, deliberately not a `startswith('c')`, which
    would also catch `col`."""
    return [c for c in df.columns if re.fullmatch(r"c\d+", str(c))]


class CorrectionNetwork:
    """The trained correction `r(x) = c(x) @ B.T`, rebuilt from its checkpoint without
    depending on Lightning.

    `c` is a small perceptron 9 -> 32 -> 2 (GELU), `B` holds two fixed SVD modes of
    `target - base model`. Weights are loaded as they are, so the correction applied here
    is numerically the one that was trained.
    """

    def __init__(self, coef: nn.Module, B: torch.Tensor, meta: dict | None = None):
        self.coef = coef.eval()
        self.B = B                                   # (231, rank)
        self.meta = meta or {}
        self.rank = int(B.shape[1])

    @classmethod
    def from_checkpoint(cls, ckpt_path, prefix="model.residual."):
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
        w = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if "B" not in w:
            raise ValueError(f"{ckpt_path}: no basis 'B': this is not a low-rank residual.")
        idx = sorted({int(k.split(".")[1]) for k in w if k.startswith("coef.")})
        layers, prev = [], None
        for i in idx:
            lin = nn.Linear(w[f"coef.{i}.weight"].shape[1], w[f"coef.{i}.weight"].shape[0])
            lin.weight.data, lin.bias.data = w[f"coef.{i}.weight"], w[f"coef.{i}.bias"]
            if prev is not None:
                layers.append(nn.GELU())             # coefficient net: Linear/GELU/Linear
            layers.append(lin)
            prev = i
        coef = nn.Sequential(*layers)
        for p in coef.parameters():
            p.requires_grad_(False)
        return cls(coef, w["B"], meta={"ckpt": str(ckpt_path),
                                       "n_params": sum(p.numel() for p in coef.parameters())})

    def coefficients(self, x9: np.ndarray) -> np.ndarray:
        """(N, 9) -> (N, rank). `x9` = 8 raw I/F bands + z-standardised emission."""
        with torch.no_grad():
            return self.coef(torch.from_numpy(np.ascontiguousarray(x9, dtype=np.float32))).numpy()

    def delta(self, coef: np.ndarray, g: np.ndarray) -> np.ndarray:
        """(N, rank), (N,) -> (N, 231): the correction to add to the deliverable."""
        B = self.B.numpy().astype(np.float32)
        return (np.asarray(g, np.float32)[:, None] * (np.asarray(coef, np.float32) @ B.T))


# ---------------------------------------------------------------------------
# Building the layer
# ---------------------------------------------------------------------------
def build_layer(spatial_mask, mdis_path, residual: CorrectionNetwork, bb_ref,
                emission_mean, emission_std, scale=0.50, z_thresh=0.5,
                row_block=256, progress=True):
    """Final selection + coefficients, on the pixels of the spatial stage only.

    Reads the 9 MDIS bands by blocks of rows containing lit pixels (the source mosaic is
    line-stripped: full-width windows are the right access pattern).

    Returns (DataFrame `row, col, g, c0..`, statistics).
    """
    H, W = spatial_mask.shape
    rows_lit = np.nonzero(spatial_mask.any(axis=1))[0]
    blocks = sorted({int(r) // row_block for r in rows_lit})
    out, n_spatial, n_invalid, n_bb_reject = [], 0, 0, 0
    t0 = time.time()

    with rasterio.open(mdis_path) as ds:
        for k, b in enumerate(blocks, start=1):
            r0 = b * row_block
            h = min(row_block, H - r0)
            sub = spatial_mask[r0:r0 + h]
            rr, cc = np.nonzero(sub)
            if not len(rr):
                continue
            n_spatial += len(rr)
            data = ds.read(list(MDIS_INPUT_BANDS),
                           window=Window(0, r0, W, h)).astype(np.float32)
            v = data[:, rr, cc]                                    # (9, n)
            valid = np.all(np.isfinite(v) & (v > NODATA_THRESHOLD), axis=0)
            n_invalid += int((~valid).sum())

            r749 = v[4]
            ratio = np.divide(v[0], r749, out=np.full(len(rr), np.nan, np.float32),
                              where=r749 > 0)
            from mdis2vihi.correction.selection import bright_blue_mask
            bb = bright_blue_mask(r749, ratio, bb_ref, z_thresh=z_thresh) & valid
            n_bb_reject += int((valid & ~bb).sum())
            if not bb.any():
                continue

            x9 = v[:, bb].T.copy()
            x9[:, 8] = (x9[:, 8] - emission_mean) / emission_std
            coef = residual.coefficients(x9)
            df = pd.DataFrame({"row": (r0 + rr[bb]).astype(np.int32),
                               "col": cc[bb].astype(np.int32),
                               "g": np.full(int(bb.sum()), scale, np.float32)})
            for j in range(coef.shape[1]):
                df[f"c{j}"] = coef[:, j].astype(np.float32)
            out.append(df)
            if progress:
                sys.stderr.write(f"\r  block {k}/{len(blocks)} (row {r0}): "
                                 f"{sum(len(d) for d in out)} pixels kept, "
                                 f"{time.time()-t0:.0f} s")
                sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")

    layer = (pd.concat(out, ignore_index=True) if out
             else pd.DataFrame({"row": [], "col": [], "g": [], "c0": [], "c1": []}))
    stats = {"n_pixels_spatial": int(n_spatial),
             "n_pixels_invalid_mdis": int(n_invalid),
             "n_pixels_rejected_bright_blue": int(n_bb_reject),
             "n_pixels_corrected": int(len(layer)),
             "fraction_mosaic_pct": round(100.0 * len(layer) / (H * W), 5),
             "bright_blue_rate_in_spatial_selection": round(
                 len(layer) / max(n_spatial - n_invalid, 1), 4)}
    return layer, stats


# ---------------------------------------------------------------------------
# Applying it to the mosaic
# ---------------------------------------------------------------------------
def tiles_of(layer, block_rows=256, block_cols=512):
    """Group the layer pixels by internal tile of the deliverable (one tile = one DEFLATE
    read/write; with PIXEL interleaving one spatial area maps to one tile)."""
    tr = layer.row.to_numpy() // block_rows
    tc = layer.col.to_numpy() // block_cols
    key = tr.astype(np.int64) * 10_000 + tc
    order = np.argsort(key, kind="stable")
    key, ordered = key[order], layer.iloc[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    for part in np.split(np.arange(len(ordered)), bounds):
        sl = ordered.iloc[part]
        yield int(sl.row.iloc[0]) // block_rows, int(sl.col.iloc[0]) // block_cols, sl


def apply_layer(mosaic_path, layer, residual: CorrectionNetwork, progress=True,
                dry_run=False):
    """Add `g*residual` to the listed pixels, tile by tile, **in place** in `mosaic_path`.

    Only ever apply this to a COPY of the deliverable (the original must stay recoverable).
    Nodata pixels are left untouched: the layer never creates a value where the deliverable
    has none. `dry_run=True` computes and audits without writing.
    """
    audit = {"n_tiles": 0, "n_pixels_written": 0, "n_pixels_nodata_skipped": 0,
             "delta_max_abs": 0.0, "n_negative_after": 0, "delta_med_996": None}
    i996 = int(np.argmin(np.abs(GRID_NM - 996.0)))
    d996 = []
    mode = "r" if dry_run else "r+"
    t0 = time.time()
    with rasterio.open(mosaic_path, mode) as ds:
        bh, bw = ds.block_shapes[0]
        nod = ds.nodata
        groups = list(tiles_of(layer, bh, bw))
        for k, (tr, tc, sl) in enumerate(groups, start=1):
            win = Window(tc * bw, tr * bh, min(bw, ds.width - tc * bw),
                         min(bh, ds.height - tr * bh))
            arr = ds.read(window=win)                              # (231, h, w)
            rr = sl.row.to_numpy() - tr * bh
            cc = sl.col.to_numpy() - tc * bw
            cur = arr[:, rr, cc]                                   # (231, n)
            ok = np.all(np.isfinite(cur) & (cur > NODATA_THRESHOLD), axis=0)
            audit["n_pixels_nodata_skipped"] += int((~ok).sum())
            if not ok.any():
                continue
            coef = sl[coef_columns(sl)].to_numpy(np.float32)[ok]
            delta = residual.delta(coef, sl.g.to_numpy(np.float32)[ok]).T   # (231, n_ok)
            new = cur[:, ok] + delta
            audit["delta_max_abs"] = max(audit["delta_max_abs"], float(np.abs(delta).max()))
            audit["n_negative_after"] += int((new < 0).sum())
            d996.append(delta[i996])
            arr[:, rr[ok], cc[ok]] = new
            if not dry_run:
                ds.write(arr, window=win)
            audit["n_tiles"] += 1
            audit["n_pixels_written"] += int(ok.sum())
            if progress:
                sys.stderr.write(f"\r  tile {k}/{len(groups)}: "
                                 f"{audit['n_pixels_written']} pixels, {time.time()-t0:.0f} s")
                sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    if d996:
        audit["delta_med_996"] = float(np.median(np.concatenate(d996)))
    audit["duration_s"] = round(time.time() - t0, 1)
    audit["nodata"] = float(nod)
    return audit


def rewrite_with_layer(src_path, dst_path, layer, residual: CorrectionNetwork,
                       chunk_cols=2560, progress=True):
    """Write a NEW corrected mosaic by streaming through the deliverable.

    Alternative to `apply_layer`: instead of editing a copy in place, which leaves as much
    dead space in the file as there are rewritten tiles, libtiff not always being able to
    put a recompressed tile back where it was, a clean file is written out. The
    deliverable's profile is carried over identically (DEFLATE/PREDICTOR=2, 512x256 tiles,
    PIXEL interleaving, BIGTIFF, nodata, band descriptions); only the compression is redone,
    and it is lossless: values outside the selection are preserved bit for bit.

    The traversal follows the natural tile order (tile row, then columns), each tile being
    written exactly once.
    """
    audit = {"n_pixels_written": 0, "n_pixels_nodata_skipped": 0, "delta_max_abs": 0.0,
             "n_negative_after": 0, "delta_med_996": None, "n_windows": 0}
    i996 = int(np.argmin(np.abs(GRID_NM - 996.0)))
    d996 = []
    t0 = time.time()
    with rasterio.open(src_path) as src:
        bh, bw = src.block_shapes[0]
        prof = src.profile.copy()
        prof.update(BIGTIFF="YES", compress="deflate", predictor=2, tiled=True,
                    blockxsize=bw, blockysize=bh, interleave="pixel",
                    num_threads="all_cpus")
        chunk_cols = max(bw, (chunk_cols // bw) * bw)      # aligned on the tiles
        # (row, column) index -> position in the layer, for a per-window lookup
        key = (layer.row.to_numpy().astype(np.int64) * src.width
               + layer.col.to_numpy().astype(np.int64))
        order = np.argsort(key)
        key = key[order]
        gg = layer.g.to_numpy(np.float32)[order]
        cc = layer[coef_columns(layer)].to_numpy(np.float32)[order]
        rows_l = layer.row.to_numpy()[order]
        cols_l = layer.col.to_numpy()[order]

        n_win = ((src.height + bh - 1) // bh) * ((src.width + chunk_cols - 1) // chunk_cols)
        with rasterio.open(dst_path, "w", **prof) as dst:
            for i, lam in enumerate(GRID_NM, start=1):
                dst.set_band_description(i, f"{lam:.0f} nm")
            k = 0
            for r0 in range(0, src.height, bh):
                h = min(bh, src.height - r0)
                inr = (rows_l >= r0) & (rows_l < r0 + h)
                for c0 in range(0, src.width, chunk_cols):
                    w = min(chunk_cols, src.width - c0)
                    win = Window(c0, r0, w, h)
                    arr = src.read(window=win)
                    sel = inr & (cols_l >= c0) & (cols_l < c0 + w)
                    if sel.any():
                        rr, ccl = rows_l[sel] - r0, cols_l[sel] - c0
                        cur = arr[:, rr, ccl]
                        ok = np.all(np.isfinite(cur) & (cur > NODATA_THRESHOLD), axis=0)
                        audit["n_pixels_nodata_skipped"] += int((~ok).sum())
                        if ok.any():
                            delta = residual.delta(cc[sel][ok], gg[sel][ok]).T
                            new = cur[:, ok] + delta
                            arr[:, rr[ok], ccl[ok]] = new
                            audit["delta_max_abs"] = max(audit["delta_max_abs"],
                                                         float(np.abs(delta).max()))
                            audit["n_negative_after"] += int((new < 0).sum())
                            audit["n_pixels_written"] += int(ok.sum())
                            d996.append(delta[i996])
                    dst.write(arr, window=win)
                    k += 1
                    audit["n_windows"] = k
                    if progress and (k % 10 == 0 or k == n_win):
                        el = time.time() - t0
                        sys.stderr.write(f"\r  window {k}/{n_win}: {el/60:.1f} min "
                                         f"(ETA {el*(n_win-k)/max(k,1)/60:.1f} min), "
                                         f"{audit['n_pixels_written']} pixels corrected")
                        sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    if d996:
        audit["delta_med_996"] = float(np.median(np.concatenate(d996)))
    audit["duration_s"] = round(time.time() - t0, 1)
    return audit