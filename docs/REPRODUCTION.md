# Reproducing the hyperspectral mosaic, step by step

This is the complete recipe: from two third-party datasets to a
231-band GeoTIFF covering the whole surface of Mercury. Follow the steps in order;
each one states what it needs, what it writes, and how to check it
worked before moving on.

If you only want to *use* the mosaic, you do not need any of this, see
[DELIVERABLE.md](DELIVERABLE.md).

---

## 0. The chain

| # | Script | What it does | Disk written |
|---|--------|--------------|-------|
| 1 | `scripts/01_build_pairs.py` | Colocate MASCS/VIRS spectra with the MDIS mosaic; quality filter; train / validation / test split by `obs_id` | ~0.6 GB + ~2 GB intermediates |
| 2 | `scripts/02_build_lsf_target.py --build` | Rebuild the training target with VIRS line-spread-aware resampling | ~1 GB |
| 3 | `scripts/03_train_final.py` | Train the delivered model (9 inputs, deterministic) | ~2 MB |
| 4 | `scripts/04_predict_mosaic.py` | Pixel-by-pixel inference over the whole mosaic | **~245 GB** |
| 4b | `gdal_translate` (separate step) | Lossless DEFLATE compression | ~158 GB |
| 5 | `scripts/05_eval_final.py` | Test-side audit of the delivered checkpoint | ~1 MB |

That chain produces the **first form** of the deliverable. The second one, the
hollow-corrected mosaic, continues from the same data:

| # | Script | What it does | Disk written |
|---|--------|--------------|-------|
| 6 | `scripts/06_build_hollow_pool.py` | Recover the hollow footprints the `q1` filter rejects, from the hollow catalogue | ~6 MB |
| 7 | `scripts/07_build_simpairs.py` | Pair each pool footprint with its measured MDIS vector | ~7 MB |
| 8 | `scripts/08_train_residual.py` | Train the rank-2 residual on those pairs | ~1 MB |
| 8a | `scripts/tools/build_crater_footprints.py` | Build the calibration table from MASCS + MDIS | ~13 MB |
| 8b | `scripts/tools/calibrate_hollow_scale.py` | Calibrate the correction strength, leave-one-out | small |
| 9 | `scripts/tools/build_hollow_correction.py` | Combine selection + correction network into the sparse layer | 3.2 MB |
| 10 | `scripts/tools/apply_hollow_correction.py` | Rewrite the mosaic through the layer | ~158 GB |

Everything but step 4 runs comfortably on a workstation; step 4 is the one that wants a
cluster, read [CLUSTER.md](CLUSTER.md) before launching it. Step 10 is local and took
46 minutes here, but it streams 158 GB in and out, so budget by your disk.

**Shortcut.** The checkpoint is committed
(`runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt`, 1.9 MB)
together with `runs/final/image_count_stats.json`. If you only want to regenerate the
mosaic, you can skip steps 1–3 entirely and go straight to step 4. You still need
the MDIS mosaic, but not the MASCS spectra.

---

## 1. Prerequisites

### Software

```bash
git clone https://github.com/raphael-muller/mdis2vihi.git
cd mdis2vihi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.11. No installation step is needed for the package itself: every script
inserts `<repo>/src` on `sys.path`, so a bare checkout works. Check it:

```bash
python -c "import sys; sys.path.insert(0,'src'); import mdis2vihi; print(mdis2vihi.__version__)"
```

Step 4b also needs the **GDAL command-line tools** (`gdal_translate`), which pip does
not provide: install the system package (`gdal-bin` on Debian/Ubuntu, `gdal` on conda
or Homebrew). Nothing else in the chain calls them.

### Data

See [DATA.md](DATA.md) for provenance and the exact directory layout.
In short, you need under `data/raw/`:

```
data/raw/mdis_mosaic/MDIS_MDR_20170512_PDS16_64ppd_equirectangular_withbackplanes.tif
data/raw/mascs/Spectra_0_360_-90_90_{spectres-002,quality,geometry,shapes}.dat
data/raw/mascs/Spectra_0_360_-90_90_MDISdata.dat                      step 5 only
```

Run [`notebooks/01_check_input_data.ipynb`](../notebooks/01_check_input_data.ipynb)
before step 1. It takes a minute and tells you whether your files are the ones this
project was built on, which step 1 would take hours to find out.

### Disk

| What | Size |
|------|------|
| MDIS mosaic | 18 GB |
| MASCS files the pipeline reads | ~34 GB (the `spectres-002.dat` alone is 31 GB) |
| Intermediates (`data/interim/`) | ~2 GB, deletable after step 1 |
| Training tables (`data/processed/`) | ~1.6 GB |
| Uncompressed mosaic (step 4) | **~245 GB** |
| Compressed deliverable (step 4b) | **~158 GB** |

Plan for **~400 GB free** during step 4, because the uncompressed and compressed
mosaics coexist until you delete the former.

---

## 2. Step 1: build the training pairs

```bash
python scripts/01_build_pairs.py
```

Reads `quality.dat` and keeps the spectra passing the strict thresholds
(`q1 ∈ [0.9, 1.1]`, `|q2| < 5`, `q3 > 80`, `q4 > 95`); streams `spectres-002.dat`
in chunks; resamples each spectrum onto the common 5 nm grid over [300, 1450] nm
(231 bins); averages the 8 MDIS I/F bands over each MASCS footprint polygon
(`shapes.dat::foot_geom`, uniform `all_touched` mean); then builds the split,
grouped on `obs_id`.

| Flag | Use |
|------|-----|
| `--limit N` | cap to the first N `obs_id` |
| `--force` | ignore `data/interim/` and reprocess every chunk |
| `--splits-only` | recompute the split from an existing `pairs.parquet` |
| `--chunksize` | rows per chunk of `spectres-002.dat` |

**Resumable.** Per-chunk parquets land in `data/interim/`; re-running the script
after a crash skips the chunks already written.

**Outputs**: `data/processed/pairs.parquet` (one row per pair, ~0.55 GB) and
`data/processed/splits.parquet`, which labels every `ref_id` `test` or `fold0`…`fold4`.

**What those labels mean.** They are a three-way train / validation / test split, made
once here and reused by every later step. Whole observations are split, not individual
spectra: 10 % of the `obs_id` go to `test`, the remaining 90 % are dealt into five folds
(a round-robin deal for the delivered file, a stratified fill in the current code, see
below). Step 3 then trains on
`fold1`…`fold4` (111 907 pairs), validates on `fold0` (26 260 pairs, used for early
stopping and checkpoint selection) and touches `test` (15 047 pairs) only to report the
final numbers. The five folds are stored as fold labels rather than a single
`train`/`val` column so that the same file also supports 5-fold cross-validation, but
the delivered model uses the one assignment above. Splitting by `obs_id` is what makes
the test set meaningful: consecutive spectra of one observation overlap on the ground, so
a split drawn spectrum by spectrum would put near-duplicates on both sides of the
boundary and flatter the model.

**The draw is stratified inside that grouping.** `obs_id` remains the unit that moves,
which is what stops the leak; the strata only decide which observations compete with which.
A group's stratum is its |latitude| band (0-30-60-90°) × image-count tercile × incidence
half, the three covariates along which accuracy actually varies (median spectral angle
2.87° below 30° of latitude against 3.74° beyond 60°; 3.35° where fewer than ten image sets
are stacked against 2.9° above). Inside a stratum, groups are handed out to whichever split
is furthest behind its quota counted in pairs, so the five folds hold the same number of
spectra and not merely the same number of observations. Terrain class is not stratified: it
is what a track shares along its length and is largely carried by the latitude and geometry
bands already.

Measured over three stratified draws against three unstratified ones, each trained end to
end: the worst standardised mean difference between test and
the rest falls from a median 0.117 over 500 unstratified draws to a median 0.06 over 10
stratified ones; the spread between the largest and the smallest fold falls from ~3 000
pairs to ~25; reweighting the test set to the training population moves the test MSE by
−0.27 ± 0.32 % against +0.22 ± 2.01 %. Accuracy is unchanged (MSE / k-NN floor 1.568 ± 0.027
against 1.556 ± 0.007, median spectral angle 3.020 ± 0.014° against 3.102 ± 0.043°):
stratification buys stability across draws, not a better model.

**The delivered `splits.parquet` predates this** and was drawn without strata; `--no-stratify`
reproduces it bit for bit, and with it the delivered checkpoint. On that draw the test set
matched the rest to a standardised mean difference of 0.12 on absolute latitude and 0.13 on
the image count, below 0.10 elsewhere, terrain fractions within 2 points, which put it at
the 94th-97th percentile of the 500 random draws on those two covariates; reweighting to the
training population lowered its MSE by 1.5 % (4.651e-5 → 4.580e-5) and the median spectral
angle by 0.03°, so the numbers below are slightly pessimistic rather than flattering.
Regenerating the file with strata means retraining and re-running the mosaic inference, so
it is left as delivered.

**Check before continuing**

```python
import pandas as pd
p = pd.read_parquet("data/processed/pairs.parquet")
print(len(p), p.obs_id.nunique())          # expect 153214 rows, 5624 obs_id
```

A different count is not necessarily wrong but means you are not using the same
MASCS spectra or quality flags.

---

## 3. Step 2: build the training target

```bash
python scripts/02_build_lsf_target.py --build
```

Redoes the 5 nm resampling as a Gaussian average matched to the VIRS line-spread
function, instead of the linear interpolation of step 1: the grid being coarser than the
native sampling, interpolating amounts to point-sampling and carries the full per-channel
noise into the target. Averaging removes 26-30 % of it and lowers the k-NN bound by 4 %;
genuine gaps stay NaN. Re-reads `spectres-002.dat`, resumable per chunk.

**Output**: `data/processed/virs_lsf_target.parquet` (~1 GB), with the variants
`naive`, `lsf_4p7`, `lsf_5p0` and `lsf_7p0`.

**The deliverable trains on `lsf_5p0`.**

Optional check: `python scripts/02_build_lsf_target.py --floor` writes
`runs/lsf_target/eval/virs_lsf_floor.csv`; the `lsf_5p0` bound should land at
≈ 2.99e-5.

---

## 4. Step 3: train the model

```bash
python scripts/03_train_final.py
```

No arguments: everything is fixed in the script, because this produces the
delivered checkpoint. It trains on `fold1`…`fold4`, validates on `fold0` and leaves
`test` untouched until the final evaluation.

**Outputs**: the best `epoch=*.ckpt`, `runs/final/image_count_stats.json`
(**required at inference**: it carries the band-9 mean and standard deviation and the
architecture) and `runs/final/eval/final_test_metrics.json`.

**Check before continuing.** The criterion is not MSE ≈ 0 but MSE against the k-NN
lower bound **on the same split**:

```python
import json; m = json.load(open("runs/final/eval/final_test_metrics.json"))
print(m["mse"], m["knn_floor_k5"], m["mlp_over_floor"], m["sam_median"])
```

The delivered run gives MSE 4.65e-5 against a floor of 2.99e-5 (ratio **1.557×**)
and a median spectral angle of **3.09°**. A ratio far above ~1.6 means something is
wrong.

---

## 5. Step 4: inference over the full mosaic

The full mosaic:

```bash
python scripts/04_predict_mosaic.py \
    --ckpt runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt \
    --count-band 9 --count-stats runs/final/image_count_stats.json \
    --output runs/final/mdis2vihi_global_final.tif \
    --device cuda --tile-rows 256 --forward-batch 5000000 --compress none
```
On the cluster: `sbatch -A <account> slurm/predict_final.sbatch`.

**The two band-9 flags are not optional.** The model has nine inputs:
`--count-band 9` reads band 9 of the MDIS mosaic, the count of 8-colour image sets
stacked at that pixel (see [DATA.md](DATA.md)), and `--count-stats` supplies the
training standardisation. Without them the checkpoint will not load, or will load and
produce nonsense.

**Do not enable compression here.** `--compress deflate --predictor 3` corrupts the
heap inside libgdal in this chunked-writer path. Compression is a separate step.

### Step 4b: compress, afterwards

```bash
gdal_translate \
    -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=YES -co TILED=YES \
    -co BLOCKXSIZE=512 -co BLOCKYSIZE=256 -co NUM_THREADS=ALL_CPUS \
    runs/final/mdis2vihi_global_final.tif \
    runs/final/mdis2vihi_global_final_deflate.tif
```

On the cluster: `sbatch -A <account> slurm/compress_final.sbatch`. Lossless,
245 GB → ~158 GB, source untouched. **`PREDICTOR=2`.**

### Check what came out

[`notebooks/02_check_mosaic.ipynb`](../notebooks/02_check_mosaic.ipynb) verifies the
file against the specification, re-predicts a window from the checkpoint and compares it
with the mosaic pixel by pixel, and compares the band statistics with the reference run.
It catches a mosaic written without the band-9 input, or from the wrong checkpoint.

---

## 6. Step 5: verify the model

```bash
python scripts/05_eval_final.py
```

Writes `runs/final/eval/final_{diagnostics.json, per_split.csv, per_obs.csv,
per_terrain.csv, floor_variants.csv, per_band.csv, param_fidelity.csv}`, and refreshes
the metric panel inside `final_test_metrics.json` (merging into it, so the keys step 3
alone produces are kept). Reference values from the delivered run:

| Quantity | Value |
|---|---|
| Test MSE / RMSE / MAE | 4.65e-5 / 6.82e-3 / 4.46e-3 |
| Spectral angle, median / p95 / p99 | 3.09° / 6.52° / 9.88° |
| k-NN lower bound (k = 5, `lsf_5p0`) | 2.987e-5 |
| MSE over the bound | 1.557× |
| Train / val / test spectral angle | 3.03° / 3.08° / 3.09° |

To verify that the mosaic really is this checkpoint's output, use section 2 of [`notebooks/02_check_mosaic.ipynb`](../notebooks/02_check_mosaic.ipynb).

---

## 7. Steps 6-10: the hollow correction layer

This step produces the **second form of the deliverable**, not an accessory. The
model under-predicts the contrast of hollows, because the quality filter keeps
almost no hollow-floor footprint in training; the layer restores it on 115 112
pixels (0.043 % of the mosaic) and leaves everything else bit-identical. The
first form is never modified: the corrected mosaic is a separate file, and
reading one or the other is your call.

The layer is committed under `runs/final/correction/`, so if all you want is the
corrected mosaic, jump straight to step 10:

```bash
python scripts/tools/apply_hollow_correction.py --dry-run   # streams, writes nothing
python scripts/tools/apply_hollow_correction.py             # writes the corrected mosaic
```

Expect 115 112 pixels written, zero negative reflectances, and a result of about
158.4 GB: the whole file is recompressed on the way out, only the corrected tiles change.

### Rebuilding the layer from the data

```bash
python scripts/06_build_hollow_pool.py --check        # --check compares each stage count
python scripts/07_build_simpairs.py
python scripts/08_train_residual.py --out runs/final/correction
python scripts/tools/build_crater_footprints.py       # calibration table, streams ~31 GB
python scripts/tools/build_hollow_correction.py       # first pass, any scale
python scripts/tools/calibrate_hollow_scale.py        # derives --scale, leave-one-out
python scripts/tools/build_hollow_correction.py --scale <calibrated>
```

The strength is calibrated on the rasterised layer, so step 9 runs twice: once to produce
the gate, then again at the scale step 8b derives from it. On the reference data that scale
is **0.797**, the median of the four per-crater values (0.595 at Dominici to 1.750 at
Hopper); leave-one-out puts the residual contrast error at 0.041 in the median against
0.175 uncorrected, i.e. the calibration transfers to a crater it never saw.

Step 6 recovers the footprints the strict filter rejects: the `q1` criterion
compares the VIS and NIR slopes, and a hollow spectrum fails it by construction,
so the model never sees the unit it is asked to render. It must report **17 140**
candidates within 15 km of a hollow group, **11 039** after the quality rule,
**1 801** on the bright+blue mask and a final pool of **1 209 footprints over 227
groups and 345 observations**.

Step 7 turns them into **3 627** training pairs, both sides measured: the input is the MDIS
vector colocated under the footprint, the target that footprint's VIRS spectrum. Each
footprint gives three rows, itself and two linear mixtures with its nearest training
background at 1/3 and 2/3, so the residual learns how the correction scales with the hollow
fill fraction. Training and inference therefore read the same input space, which matters:
scored on held-out footprints with the mosaic's own vector, the residual improves the
prediction by 44 % of MSE and 0.27 deg of spectral angle (3 seeds out of 3), and moves the
background less than half as much when it is wrongly gated.

Step 9 must select the same **115 112** pixels, reject 317 135 on the bright+blue
stage, and report 608 Thomas disks and 3 205 HORNET polygons.

> **One thing does not reproduce bit for bit, by nature.** Step 8 trains a network:
> a rerun lands on equivalent weights, not identical ones. `08 --validate` measures
> how close, comparing the spectral shapes and the correction the two networks
> actually apply. Everything before it is deterministic and order-independent: step 6
> selects the same footprints, and step 7 only reads and combines measured columns,
> so `07 --validate` against a previous run must match exactly.

The last two sections of
[`notebooks/02_check_mosaic.ipynb`](../notebooks/02_check_mosaic.ipynb) check the result
on the real rasters: outside the selection the two mosaics must be bit-identical, and inside
it the difference must be exactly what the correction network adds.