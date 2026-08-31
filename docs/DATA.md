# Input data: provenance and layout

None of this data is redistributed in this repository. It is third-party, large,
and in one case not ours to publish. This document says what you need, where to
get it, and where to put it.

| What | Needed for | Where it comes from |
|---|---|---|
| MDIS multispectral map | everything | [MESSENGER mission site, JHU/APL](https://messenger.jhuapl.edu/Explore/Images.html#global-mosaics), see §1 |
| MASCS/VIRS spectra | training the model | **LIRA / Observatoire de Paris team**: the curated set is not public, see §2 |
| Hollow catalogues | the correction layer only | published supporting information, open, see §3 |

**The one that will stop you is the MASCS set.** Everything else is downloadable;
that one has to be asked for.

Once it is in place, [`notebooks/01_check_input_data.ipynb`](../notebooks/01_check_input_data.ipynb)
verifies it against the reference run in about a minute.

## Expected layout

```
data/
  raw/
    mdis_mosaic/
      MDIS_MDR_20170512_PDS16_64ppd_equirectangular_withbackplanes.tif    18 GB
    mascs/
      Spectra_0_360_-90_90_spectres-002.dat         31 GB   spectra (required)
      Spectra_0_360_-90_90_quality.dat             336 MB   q1..q4 flags (required)
      Spectra_0_360_-90_90_geometry.dat            589 MB   footprint geometry (required)
      Spectra_0_360_-90_90_shapes.dat              1.5 GB   footprint polygons (required)
      Spectra_0_360_-90_90_MDISdata.dat            197 MB   terrain indicators (step 5 only)
    hollow/                                                 §3, second form only
      1-s2.0-S0019103516302469-mmc4.txt                    groups + geodesic areas
      1-s2.0-S0019103513004909-mmc1.txt                    group centres only
      hornet/Bickel_et_al_2024_MDIS_hollows_08_v1.0.csv
  interim/      written by step 1, deletable afterwards
  processed/    written by steps 1-2
```

`data/` is git-ignored in its entirety. The MASCS and MDIS paths are hard-coded in
`src/mdis2vihi/data/io.py`, the hollow ones in `src/mdis2vihi/correction/selection.py`
and `scripts/06_build_hollow_pool.py`.

---

## 1. MDIS global mosaic: the model input

MESSENGER Mercury Dual Imaging System, multispectral map, 64 pixels/degree,
equirectangular, with backplanes. Photometric processing per Denevi et al. (2018).

**Where to get it.** From the MESSENGER mission site at JHU/APL,
[Explore -> Images -> Global Mosaics](https://messenger.jhuapl.edu/Explore/Images.html#global-mosaics).
Take the multispectral global mosaic at 64 pixels/degree, equirectangular, **with
backplanes**: the file name records the PDS release it was built from. The mission
site assembles the map into one global file; the
[PDS Cartography and Imaging Sciences Node](https://pds-imaging.jpl.nasa.gov/volumes/mess.html)
archives the same data as 54 separate tiles, which this code does **not** read.

23040 × 11521, **17 bands, Float32**, nodata `-3.4028226550889e+38`, block shape
`(1, 23040)`; the file is *line-stripped*, so read full-row windows and never load it
whole.

| Bands | Content |
|-------|---------|
| 1–8 | The 8 MDIS/WAC I/F filters, ascending: **433, 480, 559, 629, 749, 828, 899, 996 nm** |
| 9 | **Count of 8-colour image sets** averaged into that pixel: an integer, 0 to 86 |
| 10–17 | Standard deviation of that average, one per I/F band, same order as 1–8 |

The model uses **bands 1–9**.

---

## 2. MASCS/VIRS point spectra: the model target

MESSENGER Atmospheric and Surface Composition Spectrometer, Visible and Infrared
Spectrograph, non-imaging: it samples spectra at points, not images.

**Where to get it.** The raw archive is the MASCS/VIRS Derived Data Record set at
the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/messenger/mascs.htm),
one spectrum per row with its geometry, 300–1050 nm (VIS) and 850–1450 nm (NIR).

**But that is not the set this project trains on.** The set used here was **curated
by Océane Barraud** (PhD thesis, Part II) on top of the PDS archive:
quality-screened, photometrically corrected, VIS and NIR merged into one spectrum,
with the `q1..q4` flags below. It is **not redistributed**, here or anywhere. Ask
the LIRA / Observatoire de Paris team for it, and cite Barraud's thesis and Barraud
et al. (2020) alongside the mission references.

**Format.** Every `.dat` sidecar is CSV despite the extension. In
`spectres-002.dat`, the `waves` and `photom_iof` columns are stringified Python
lists, parse them with `ast.literal_eval`. Spectra have variable length (341–525
bins), native sampling 2.33 nm between adjacent detector channels on both VIRS arrays
(VIRS SIS V5.4, `CHANNEL_WAVELENGTHS`: λ = 215.16 + 2.330·n − 1.89e−5·n² nm on the VIS
array, λ = 896.50 + 2.330·n + 1.48e−5·n² nm on the NIR one; measured 2.325 nm below
1050 nm and 2.34 nm above), resolution 4.7 nm FWHM, coverage ≈ 264–1489 nm.
The good set holds 3 172 422 spectra over 10 177 distinct `obs_id`.

**Quality flags `q1..q4`** (in `quality.dat`):

| Flag | Meaning | Target |
|---|---|---|
| `q1` | ratio of the 600–750 and 900–1050 nm slopes | ≈ 1 |
| `q2` | 750–800 nm extrapolated vs measured, % | ≈ 0 |
| `q3` | % of VIS points surviving cleaning | max ≈ 83 % |
| `q4` | % of NIR points surviving cleaning | max ≈ 100 % |

The pipeline keeps `q1 ∈ [0.9, 1.1]` (bounds included), `|q2| < 5`, `q3 > 80`,
`q4 > 95`: **154 064 spectra (4.9 %) over 5 629 `obs_id`**. Colocation on the mosaic then
drops the ~850 footprints that cannot be projected or that fall entirely on nodata,
leaving the 153 214 pairs over 5 624 `obs_id` that the model trains on. There is no
reflectance threshold at this stage; the `--vis-floor` of `scripts/06_build_hollow_pool.py`
applies to the hollow pool only.

The footprint average filters nodata (`> -1e30`, a sentinel guard) and nothing else, in
particular **no shadow floor**, and the measurement says none is needed: over 4 000 random
training footprints (132 663 MDIS pixels), not one pixel falls below I/F 0.01 at 749 nm and
0.40 % fall below 0.02, those at |lat| around 72° with a median image count of 3, which is
grazing polar illumination rather than cast shadow. The mosaic being an average of every
qualifying image, and the MASCS quality filter rejecting the worst-lit footprints (0.14 %
of pixels below 0.01 under the footprints it rejects, against 0.00 % under those it keeps),
the darkest cases are gone before this step. A floor here would also have to be applied per
pixel across all bands at once, never band by band, or the bands would be averaged over
different pixel sets and the colour ratios would shift.

**Footprints.** `geometry.dat` and `shapes.dat` describe the same elliptical
footprint in two different ways, and neither is a refinement of the other.
`geometry.dat` gives the ellipse parameters (centre, `length`, `width`, `azimuth`)
plus `lat_c1..c4`/`lon_c1..c4`, the four endpoints of its two axes (c1, c2 half a
`length` from the centre along the major axis, c3, c4 half a `width` along the minor
one) rather than the corners of a quadrilateral; `shapes.dat::foot_geom` gives the
boundary of that same ellipse as a WKT `POLYGON` of ten points evenly spaced in
angle. Use `foot_geom`, because rasterising a footprint needs a closed outline.

The two files also disagree on the **longitude convention**, which is worth checking
before projecting anything: `geometry.dat` is on `[0°, 360°)` (both `lon_center` and
`lon_c1..c4`), while the `shapes.dat` WKT is on `[-180°, 180°]`, 51.4 % of its
footprints carrying negative longitudes. Both seams appear in the WKT: of the 429
footprints stored in two parts, 296 are split at the antimeridian and 133 at the prime
meridian. On the ground they run from
~300 m to a few tens of km, with a median length of 3.7 km, against an MDIS pixel of
665 m, which is where the model's error floor comes from.

---

## 3. Hollow catalogues

Needed to **rebuild** the hollow correction layer, which produces the second form
of the deliverable ([REPRODUCTION.md](REPRODUCTION.md), steps 6-10). The layer itself
is committed, so *applying* it needs neither file; the first form of the
deliverable needs neither of them at all.

| Path | Used by | What |
|---|---|---|
| `data/raw/hollow/1-s2.0-S0019103516302469-mmc4.txt` | the selection | Hollow groups **with geodesic areas**. Tab-separated, **2 header rows**, 608 groups, and it must carry `Geodesic.area.km.2`: that column sets each disk's radius for the 42 % of groups large enough, the other 58 % falling on the 3 km floor.|
| `data/raw/hollow/1-s2.0-S0019103513004909-mmc1.txt` | the training pool | Hollow **group centres only**: `Group_id, Central_long, Central_lat`, 445 rows, one header line. No area column, and none is needed: the pool uses a fixed 15 km reference disk.|
| `data/raw/hollow/hornet/Bickel_et_al_2024_MDIS_hollows_08_v1.0.csv` | the selection | HORNET 0.8, 3 268 rows, semicolon-separated, latin-1, with hollow outlines as WKT in the last column.|

**Where to get them.** Three files, three separate sources, all open. **Keep the
names they download with**: the paths below are what the code expects, so nothing
has to be renamed.

| File here | Source | What to download |
|---|---|---|
| `1-s2.0-S0019103513004909-mmc1.txt` | **Thomas et al. (2014)**, [doi:10.1016/j.icarus.2013.11.018](https://doi.org/10.1016/j.icarus.2013.11.018) | Appendix A, *Supplementary data 1*. |
| `1-s2.0-S0019103516302469-mmc4.txt` | **Thomas et al. (2016)**, [doi:10.1016/j.icarus.2016.05.036](https://doi.org/10.1016/j.icarus.2016.05.036) | Appendix, **Table ST1**: the same inventory extended to data released up to March 2015, with a geodesic area per group. |
| `hornet/Bickel_et_al_2024_MDIS_hollows_08_v1.0.csv` | **Bickel, Deutsch & Blewett (2025)**, *Hollows on Mercury: Creation and Analysis of a Global Reference Catalog With Deep Learning*, [doi:10.1029/2024JH000431](https://doi.org/10.1029/2024JH000431) | The supporting data of that paper. |

> **The two group tables are not interchangeable.** They are the same inventory two
> years apart, and the code uses each for a different thing. Fed the 445-row centre
> table, the spatial selection would collapse every disk to its 3 km minimum radius
> and stop matching the delivered layer; fed the 608-row table, the pool builder
> would not find the `Group_id` key it selects on. The first line tells them apart:
> the `mmc4` file opens with a quoted `Supplementary Table ST1` title, the `mmc1`
> file goes straight to the column names.

---

## 4. Derived tables

Written by the pipeline, regenerable.

| File | Size | Written by | Contents |
|---|---|---|---|
| `data/processed/pairs.parquet` | 0.55 GB | step 1 | 153 214 rows, 5 624 `obs_id` |
| `data/processed/splits.parquet` | small | step 1 | the train / validation / test split: `ref_id → {fold0..fold4, test}`, drawn by `obs_id` ([how it is used](REPRODUCTION.md#2-step-1-build-the-training-pairs)) |
| `data/processed/virs_lsf_target.parquet` | 0.99 GB | step 2 | target variants; **`lsf_5p0` is what the deliverable trains on** |
| `data/interim/` | ~2 GB | steps 1–2 | per-chunk parquets enabling resume |

---

## References

Cite the instrument and processing papers alongside this work, and the catalogue
papers if you rebuild the correction layer.

- **Hawkins et al. (2007)**: MDIS instrument, WAC filter naming.
  *Space Sci. Rev.* 131, 247–338.
  [doi:10.1007/s11214-007-9266-3](https://doi.org/10.1007/s11214-007-9266-3)
- **Izenberg et al. (2014)**: MASCS/VIRS reflectance of Mercury's surface.
  *Icarus* 228, 364–374.
  [doi:10.1016/j.icarus.2013.10.023](https://doi.org/10.1016/j.icarus.2013.10.023)
- **Denevi et al. (2018)**: MDIS calibration, projection and final image products;
  the photometric standardisation of the map used here.
  *Space Sci. Rev.* 214, 2.
  [doi:10.1007/s11214-017-0440-y](https://doi.org/10.1007/s11214-017-0440-y)
- **Barraud et al. (2020)**: near-UV to near-IR spectral properties of hollows;
  the method behind the curated MASCS set, and the hollows diagnostic.
  *JGR: Planets* 125.
  [doi:10.1029/2020JE006497](https://doi.org/10.1029/2020JE006497)
- **Barraud, PhD thesis** (Part II, ch. 2–3): the curated MASCS set itself.
- **Thomas et al. (2014)**: the hollow inventory, and the group-centre table itself.
  *Icarus* 229, 221–235.
  [doi:10.1016/j.icarus.2013.11.018](https://doi.org/10.1016/j.icarus.2013.11.018)
- **Thomas et al. (2016)**: hollows as a constraint on Mercury's low-reflectance
  material; source of the extended group table with geodesic areas.
  *Icarus* 277, 455–465.
  [doi:10.1016/j.icarus.2016.05.036](https://doi.org/10.1016/j.icarus.2016.05.036)
- **Bickel, Deutsch & Blewett (2025)**: *Hollows on Mercury: Creation and Analysis of a
  Global Reference Catalog With Deep Learning*, the HORNET catalogue used by the
  correction layer. *JGR: Machine Learning and Computation* 2(1). Data under CC-BY.
  [doi:10.1029/2024JH000431](https://doi.org/10.1029/2024JH000431)
- **Deutsch et al. (2025)**: *Hollows on Mercury: Global Classification of Degradation
  States and Insight Into Hollow Evolution*, the second paper the HORNET archive supports
  and the source of its degradation states.
  *JGR: Planets*. [doi:10.1029/2024JE008747](https://doi.org/10.1029/2024JE008747)
- **Cremonese et al. (2020)**: SIMBIO-SYS / VIHI, the instrument this work emulates.
  *Space Sci. Rev.* 216, 75.
  [doi:10.1007/s11214-020-00704-8](https://doi.org/10.1007/s11214-020-00704-8)