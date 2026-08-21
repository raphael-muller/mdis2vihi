# The deliverable

A **full hyperspectral mosaic of Mercury**, 231 bands, as a georeferenced Float32
GeoTIFF, in **two forms**:

```
runs/final/mdis2vihi_global_final_deflate.tif                       158.37 GB   the model output
runs/final/mdis2vihi_global_final_postcorr_unionbb0.8_deflate.tif   158.38 GB   + the hollow-correction layer
```

Both are the deliverable. They share one specification, one grid and one set of
values everywhere except on 115 112 pixels (0.043 % of the mosaic), where the
second carries the hollow-contrast correction. Which one you read is your call;
neither is a draft of the other, and the first is never modified to produce the
second.

Neither file is **in this repository**: they are 158 GB each. The repository
carries everything needed to regenerate either one
([REPRODUCTION.md](REPRODUCTION.md)) or to verify a copy you were given.

---

## Specification

Read back from the delivered file, not copied from a design document.

| Property | Value |
|---|---|
| Size | 23040 × 11521 × **231 bands** |
| Data type | Float32, no quantisation, no rescaling |
| Spectral grid | 5 nm steps over **[300, 1450] nm** |
| Band description | `'<λ> nm'`, band 1 = `300 nm`, band 231 = `1450 nm` |
| Nodata | `-3.4028226550889e+38` on every band |
| CRS | Equirectangular Mercury, datum `D_Mercury`, sphere R = 2 439 400 m, EPSG method 1028 |
| Resolution | 665.24315270546 m/px |
| Bounds (projected, m) | X ∈ [−7 663 601, +7 663 601], Y ∈ [−3 832 465.8, +3 831 800.6] |
| Container | BIGTIFF, tiled 512 × 256 |
| Compression | DEFLATE with PREDICTOR = 2, **lossless** |
| Overviews | none |

A pixel is nodata on *all* 231 bands as soon as any of the 9 input bands is
non-finite in the MDIS mosaic: the mask includes band 9, the image-set count, not
only the eight reflectances.

---

## Reading it

```python
import rioxarray                                    # pip install rioxarray, not a
                                                    # dependency of the pipeline itself
da = rioxarray.open_rasterio("mdis2vihi_global_final_deflate.tif", chunks=True)
```

or, with `rasterio` alone, for a windowed read of a few bands:

```python
import rasterio
from rasterio.windows import Window

with rasterio.open("mdis2vihi_global_final_deflate.tif") as src:
    wl = [float(d.split()[0]) for d in src.descriptions]   # 300, 305, ... 1450
    b750 = wl.index(750.0) + 1                             # rasterio bands are 1-indexed
    tile = src.read(b750, window=Window(11000, 5500, 1024, 1024))
```

**The file has no overviews.** Any decimated full-extent read decompresses every
tile of every band it touches, which on 158 GB is very slow.

---

## Values to check yours against

Over 41 million valid pixels in 352 windows
(`runs/final/eval/mosaic_consistency_final.json`):

| Quantity | Value |
|---|---|
| Median I/F | 0.047 |
| Reflectance at 435 / 750 / 1450 nm | 0.0248 / 0.0407 / 0.0628 |
| Red slope, 750/435 nm | +64.1 % |

[`notebooks/02_check_mosaic.ipynb`](../notebooks/02_check_mosaic.ipynb) re-runs these
checks on your own copy, and verifies that it really is this checkpoint's output.

---

## The second form: the hollow-corrected mosaic

```
mdis2vihi_global_final_postcorr_unionbb0.8_deflate.tif      158.38 GB
```

The model under-predicts the brightness contrast of hollows, because the quality
filter keeps almost no hollow-floor footprint in training (see
[Known limitations](#known-limitations)). A correction layer restores it on the
catalogued hollows and leaves every other pixel bit-identical:

```
output = model output + g(lon, lat, reflectance) * [ c(x) @ B.T ]
g      = 0.42 * [ (Thomas 2016 union HORNET 0.8) inter bright+blue ]
```

The layer itself is small enough to live in this repository
(`runs/final/correction/`, 2.2 MB: 115 112 pixels, five numbers each, plus the
small network it applies). [REPRODUCTION.md](REPRODUCTION.md) covers the model in
steps 1-5 and the layer in steps 6-10: the selection is rebuilt from the two hollow
catalogues and must select the same 115 112 pixels, while the rank-2 residual is
carried here as a 644 kB checkpoint, the way the model checkpoint is.

**The strength is calibrated, and its generalisation measured.**
`scripts/tools/calibrate_hollow_scale.py` derives `0.42` from the four craters whose
footprints are kept out of the residual's training set, then leaves each of them out in
turn. Calibrated on three and applied to the fourth, the remaining hollow/background
contrast error at 996 nm is **0.063 in the median and 0.139 at worst, against 0.175 and
0.283 with no correction**: the layer removes about two thirds of the deficit on a
crater it has never seen. Per-crater numbers in `correction/scale_calibration.json`.

How the correction reads spectrally is measured on the rasters, not asserted here: the
last two sections of
[`notebooks/02_check_mosaic.ipynb`](../notebooks/02_check_mosaic.ipynb) difference the two
mosaics over a hollow field and over a control patch, check that the difference is exactly
the fixed residual inside the selection and exactly zero outside, and report the contrast
gained band by band. The figures depend on the window, so run them on your own copy.

Same specification as the first form in every respect: grid, CRS, bounds, band
descriptions, nodata, compression. The two differ only inside the selection.

---

## Known limitations

Read these before drawing conclusions. Every figure comes from `runs/final/eval/`, on the
test split never used in training (15 047 spectra over 562 observations, grouped by
`obs_id`).

**The error has a lower bound and the model sits near it, not at zero.** A 665 m MDIS
pixel maps to a MASCS footprint of 1 to 5 km, so footprints with near-identical 8-band
inputs carry genuinely different spectra: the mapping is one-to-many. Against that k-NN
bound the model is at **1.557x**, and the headroom is on amplitude, not shape (spectral
angle at **1.06x** its own angular bound). The bound is an estimate and the ratio depends
on the convention: 1.557x is k = 5, ddof = 0, Euclidean distance on the 8 raw I/F bands,
while other reasonable conventions give 1.1 to 1.9. See `knn_floor` in
`src/mdis2vihi/eval/metrics.py`. Judged per parameter against what the input makes
knowable at all (`final_param_ceiling.csv`), the model reaches 85 to 97 % of the ceiling
everywhere except the NIR slope.

**Hollows are extrapolation.** The `q1` filter rejects almost every hollow-floor
footprint, so the model never saw the unit it is asked to render and under-predicts hollow
contrast. That is why the second form exists, and its correction is confined to
**catalogued** hollows: an uncatalogued one gets nothing. Neither form should be used to
*discover* hollows.

**Beyond 996 nm the model extrapolates.** The input stops at 996 nm and the output runs to
1450 nm, so the last 90 bands rest on the MASCS target alone. Per-band correlation falls
from **r about 0.65 in the visible to 0.54 at 1450 nm** (`final_per_band.csv`); per-band
bias stays under 2.2e-4 throughout, so the tail loses fidelity without drifting.

**Accuracy is not uniform.** Median spectral angle runs from 2.99 deg on smooth plains to
**3.45 deg on low-reflectance material** (`final_per_terrain.csv`), and depends mildly on
observing geometry of the MASCS spectrum (incidence +0.37, emission -0.33 in
correlation), which reads as
grazing-geometry signal-to-noise rather than an artefact.

**Coverage follows MASCS, not a grid.** The 153 214 pairs sit where VIRS happened to
sample, and the strict quality filter keeps 4.9 % of the good set. Regions far from any
retained footprint are interpolated with no local evidence.

**This is a simulation, not a measurement.** The output emulates what a VIHI-like
instrument might record, from MESSENGER-era data. It is early input for Mercury work
ahead of real BepiColombo/SIMBIO-SYS data, and cannot substitute for them.
