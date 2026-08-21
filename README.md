# mdis2vihi

**Synthesising a hyperspectral mosaic of Mercury from 8-band imagery.**

A neural network maps, at a single pixel, the 8-band MDIS reflectance spectrum
(MESSENGER multispectral imagery) to a 231-band VIHI-like spectrum on a common
5 nm grid over 300–1450 nm, the spectral range of the VIHI channel of SIMBIO-SYS
on BepiColombo. Applied pixel by pixel across the MDIS global mosaic, it produces a
**synthetic hyperspectral mosaic of the whole planet**, intended as an early
stand-in for real VIHI data.

Research internship in planetary science and machine learning, LIRA /
Observatoire de Paris.

---

## What this repository contains

Everything needed to reproduce, verify and reuse the product: the full pipeline
(colocation, target construction, training, tiled inference, evaluation), the
**delivered checkpoint** (1.9 MB) with its standardisation statistics, the
evaluation tables behind every published number, the **hollow-correction layer**
(2.2 MB) with the small network it applies, and two notebooks that check your
input data and your produced mosaic.

The deliverable comes in two forms, the model output and the hollow-corrected
mosaic.

It does **not** contain the input data (third-party, ~52 GB) nor either mosaic
(158 GB each).

## Where to start

| If you want to… | Read |
|---|---|
| use the mosaic you were given | [docs/DELIVERABLE.md](docs/DELIVERABLE.md): spec, how to read it, **limitations** |
| regenerate the mosaic | [docs/REPRODUCTION.md](docs/REPRODUCTION.md): step by step |
| get the input data | [docs/DATA.md](docs/DATA.md): provenance and layout |
| run it on a cluster | [docs/CLUSTER.md](docs/CLUSTER.md): SLURM |
| check what you have | the two notebooks: [inputs](notebooks/01_check_input_data.ipynb), [mosaic](notebooks/02_check_mosaic.ipynb) |

---

## The model

A fully-connected MLP, `[9 → 128 → 256 → 256 → 231]`, GELU. Input: the 8 MDIS I/F
bands plus band 9, the **count of 8-colour image sets** stacked at that pixel,
transformed and z-standardised. Target: the
co-located MASCS/VIRS spectrum resampled with the instrument's line-spread
function. Loss: NaN-tolerant MSE.

Test split, never used in training (15 047 spectra over 562 observations,
grouped by `obs_id`):

| MSE | RMSE | Spectral angle, median | p95 | k-NN lower bound (k = 5) | MSE / bound |
|---|---|---|---|---|---|
| 4.65e-5 | 6.82e-3 | **3.09°** | 6.52° | 2.99e-5 | **1.557×** |

The ratio to that lower bound is the number that matters. A 665 m MDIS pixel maps to a
1–5 km MASCS footprint, so the mapping is one-to-many and the achievable MSE is
bounded below by the target variance in a small input neighbourhood. **Do not
evaluate this model against MSE = 0**: evaluate it against the k-NN lower bound computed
on the same split. See :
`runs/final/eval/final_test_metrics.json`.

The bound is an estimate and the ratio is convention-dependent (1.557× is k = 5,
ddof = 0, 8 raw bands; other reasonable conventions give 1.1 to 1.9).

---

## Quick start

```bash
git clone https://github.com/raphael-muller/mdis2vihi.git
cd mdis2vihi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No install step for the package: the scripts put `src/` on `sys.path` themselves.

```bash
# regenerate the mosaic from the committed checkpoint, quick test first
python scripts/04_predict_mosaic.py \
    --ckpt runs/final/lightning_logs/version_0/checkpoints/epoch=93-step=10340.ckpt \
    --count-band 9 --count-stats runs/final/image_count_stats.json \
    --output runs/final/predict_roi.tif --roi 11000 5500 1024 1024
```

The full chain (needing the input data) is five numbered scripts:

| Step | Script | Output |
|---|---|---|
| 1 | `01_build_pairs.py` | `data/processed/{pairs,splits}.parquet` |
| 2 | `02_build_lsf_target.py --build` | `data/processed/virs_lsf_target.parquet` |
| 3 | `03_train_final.py` | `runs/final/` checkpoint + `image_count_stats.json` |
| 4 | `04_predict_mosaic.py` | the mosaic (~245 GB, compressed to ~158 GB) |
| 5 | `05_eval_final.py` | `runs/final/eval/final_*` |

Three more (`06`-`08`, plus the `build_`/`apply_hollow_correction.py` pair) rebuild
the hollow-correction layer and the second form of the mosaic.

Full instructions : [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Layout

```
src/mdis2vihi/      data/ models/ lit/ inference/ eval/ correction/
scripts/            production chain 01-05, correction layer 06-08; tools in tools/
slurm/              the two cluster jobs, as submitted on MesoPSL
notebooks/          01 check the input data, 02 check the mosaic
docs/               reproduction, deliverable, data, cluster
runs/final/         delivered checkpoint, band-9 stats, evaluation, correction layer
```

`scripts/tools/` holds the five extra tools that live outside the chain:
`extract_bands.py` (the band-subset adapter for whoever uses the mosaic next),
`analysis_mosaic_consistency.py` (band statistics of a produced mosaic),
`calibrate_hollow_scale.py` (derives the strength of the hollow correction and measures
how it generalises) and the `build_`/`apply_hollow_correction.py` pair.

---

## Data, licence and credit

The **code and documentation** are MIT-licensed ([LICENSE](LICENSE)).

The **data are not redistributed here** and keep their own terms: the
MESSENGER/MDIS mosaic and other NASA/PDS products, and the MASCS/VIRS spectra
**curated by Océane Barraud** (PhD thesis, Part II), which are the training data of
this entire work and are obtained from the team, not from this repository.

If you use this software, the checkpoint or the mosaic, please cite this work
([CITATION.cff](CITATION.cff)) *and* the underlying datasets and instrument papers
listed in [docs/DATA.md](docs/DATA.md).

Work carried out at LIRA (Laboratoire d'Instrumentation et de Recherche en
Astrophysique), Observatoire de Paris, PSL, by Raphaël Müller, under the
internship supervision of Michele Lissoni and Alain Doressoundiram. Computation on
the MesoPSL GPU cluster.