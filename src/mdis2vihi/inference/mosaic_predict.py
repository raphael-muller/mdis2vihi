"""Streamed pixel-by-pixel inference of the trained MLP over the MDIS mosaic.

Produces the project deliverable: a Float32 multi-band GeoTIFF on the
5 nm x [300, 1450 nm] grid (231 bands), CRS / transform / nodata inherited
from the MDIS source.

Memory note
-----------
A full hyperspectral mosaic is 23 040 x 11 521 x 231 x 4 B ~ 245 GB, never
materialized in RAM. The mosaic is processed in row-stripped chunks
(`tile_rows` rows at a time); peak RAM is bounded by the chunk size
(default 32 rows ~ 680 MB for the output buffer, ~27 MB for the 9 input bands).

Compression note
----------------
Write this output **uncompressed** (`compress=None`) and compress it afterwards with
`gdal_translate` (PREDICTOR=2). `deflate` + `predictor=3` corrupts the heap inside
libgdal in this chunked-writer path, see docs/CLUSTER.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from mdis2vihi.lit.spectral_module import SpectralLitModule
from mdis2vihi.models.mlp import SpectralMLP


GRID_NM = np.arange(300.0, 1450.0 + 5.0, 5.0)
N_BANDS_OUT = len(GRID_NM)
MDIS_IF_BANDS = (1, 2, 3, 4, 5, 6, 7, 8)
NODATA_GENERIC_THRESHOLD = -1e30


def _load_model(ckpt_path: Path, device: str, in_features: int = len(MDIS_IF_BANDS)) -> torch.nn.Module:
    skeleton = SpectralMLP(in_features=in_features, out_features=N_BANDS_OUT)
    lit = SpectralLitModule.load_from_checkpoint(
        str(ckpt_path), model=skeleton, map_location="cpu"
    )
    lit.eval()
    return lit.model.to(device)


def predict_mosaic(
    ckpt_path: Path,
    input_mosaic: Path,
    output_path: Path,
    tile_rows: int = 32,
    device: str = "auto",
    compress: str | None = "lzw",
    blockxsize: int = 512,
    blockysize: int | None = None,
    predictor: int | None = None,
    forward_batch: int = 200_000,
    roi: tuple[int, int, int, int] | None = None,
    count_band: int | None = None,
    count_stats: tuple[float, float] | None = None,
    band_indexes: Sequence[int] = MDIS_IF_BANDS,
    progress: bool = True,
) -> Path:
    """Run pixel-wise inference and write the hyperspectral GeoTIFF.

    Parameters
    ----------
    ckpt_path
        Lightning checkpoint produced by `scripts/03_train_final.py`.
    input_mosaic
        Path to the MDIS source mosaic.
    output_path
        Where the hyperspectral GeoTIFF will be written. BIGTIFF=YES.
    tile_rows
        Rows per streaming chunk. Trade-off: larger = fewer I/O calls but more
        RAM (out_chunk ~ tile_rows * 23 040 * 231 * 4 B = 21 MB * tile_rows).
    device
        `"cuda"`, `"cpu"` or `"auto"`.
    compress
        GeoTIFF compression: `"lzw"`, `"deflate"`, or `None`. Use `None` for the full
        mosaic and compress afterwards (see the module docstring).
    blockxsize
        Output GeoTIFF internal tile width.
    blockysize
        Output GeoTIFF internal tile height. If `None`, defaults to `tile_rows`
        so each streaming write covers an integer number of internal tile rows
        (avoids libtiff read-modify-write amplification at tile boundaries).
    predictor
        TIFF predictor for compressed output. If `None`, picks 3 (float
        predictor) for `deflate`, 2 for `lzw`, 1 otherwise. Do not use 3 here:
        `deflate` + `predictor=3` heap-corrupts libgdal in this chunked writer.
    forward_batch
        Max pixels per `model.forward` call. Caps GPU/CPU memory regardless of
        chunk size.
    roi
        Optional `(col_off, row_off, width, height)` to only process a window.
        Output dims will match the ROI; useful for quick tests.
    band_indexes
        1-indexed band positions of the 8 MDIS I/F filters in the source mosaic.
    progress
        Print progress to stderr.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if count_band is not None and count_stats is None:
        raise ValueError("count_band set but count_stats (mean, std) not given")
    read_bands = list(band_indexes) + ([count_band] if count_band is not None else [])
    in_features = len(read_bands)
    model = _load_model(Path(ckpt_path), device, in_features=in_features)

    if blockysize is None:
        blockysize = tile_rows
    if predictor is None:
        predictor = {"deflate": 3, "lzw": 2}.get(compress, 1)

    with rasterio.open(input_mosaic) as src:
        src_nodata = src.nodata
        if roi is None:
            col_off, row_off = 0, 0
            width, height = src.width, src.height
            transform = src.transform
        else:
            col_off, row_off, width, height = roi
            transform = src.window_transform(Window(col_off, row_off, width, height))

        profile = src.profile.copy()
        profile.update(
            count=N_BANDS_OUT,
            dtype="float32",
            nodata=src_nodata,
            width=width,
            height=height,
            transform=transform,
            BIGTIFF="YES",
            compress=compress,
            tiled=True,
            blockxsize=blockxsize,
            blockysize=blockysize,
            interleave="band",
            predictor=predictor,
        )
        profile.pop("photometric", None)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with rasterio.open(output_path, "w", **profile) as dst:
            for i, lam in enumerate(GRID_NM, start=1):
                dst.set_band_description(i, f"{lam:.0f} nm")

            for chunk_row in range(0, height, tile_rows):
                n = min(tile_rows, height - chunk_row)
                src_window = Window(col_off, row_off + chunk_row, width, n)
                dst_window = Window(0, chunk_row, width, n)

                t_a = time.time()
                data = src.read(read_bands, window=src_window).astype(np.float32)
                valid = np.all(
                    np.isfinite(data) & (data > NODATA_GENERIC_THRESHOLD), axis=0
                )
                t_read = time.time() - t_a

                t_a = time.time()
                out_chunk = np.full(
                    (n, width, N_BANDS_OUT), src_nodata, dtype=np.float32
                )
                n_valid = int(valid.sum())
                if n_valid > 0:
                    x = data.transpose(1, 2, 0)[valid]
                    if count_band is not None:
                        emean, estd = count_stats
                        x[:, -1] = (x[:, -1] - emean) / estd  # I/F raw, band 9 z-standardized
                    preds = np.empty((n_valid, N_BANDS_OUT), dtype=np.float32)
                    for j in range(0, n_valid, forward_batch):
                        xt = torch.from_numpy(x[j : j + forward_batch]).to(device)
                        with torch.no_grad():
                            preds[j : j + forward_batch] = model(xt).cpu().numpy()
                    out_chunk[valid] = preds
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd = time.time() - t_a

                t_a = time.time()
                dst.write(out_chunk.transpose(2, 0, 1), window=dst_window)
                t_write = time.time() - t_a

                if progress:
                    done = chunk_row + n
                    pct = 100.0 * done / height
                    elapsed = time.time() - t0
                    eta = elapsed * (height - done) / max(done, 1)
                    sys.stderr.write(
                        f"\r  rows {done}/{height} ({pct:5.1f}%) "
                        f"elapsed {elapsed/60:5.1f} min ETA {eta/60:5.1f} min "
                        f"[chunk: read {t_read:5.1f}s fwd {t_fwd:5.1f}s write {t_write:5.1f}s]\n"
                    )
                    sys.stderr.flush()
            if progress:
                sys.stderr.write("\n")

    return Path(output_path)