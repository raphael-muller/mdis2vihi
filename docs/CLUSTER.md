# Running on a SLURM cluster

Only one step really needs a cluster: the full-mosaic inference, which writes
~245 GB and is then compressed to ~158 GB. Everything else fits on a workstation.

`slurm/` therefore contains only those two jobs:

| Script | Purpose                   |
|---|---------------------------|
| `predict_final.sbatch` | full mosaic, uncompressed |
| `compress_final.sbatch` | compression               |

Both are kept **verbatim as they were submitted on MesoPSL**, as the record of
what produced the deliverable, rather than turned into portable templates. Read
them as worked examples and edit them for your site.

---

## What to adapt

Two things, both at the top of each script:

1. **The scheduler directives.** `--clusters=gpu --partition=hi
   --qos=gpu_hi_normal` and `--gres=gpu:1` are MesoPSL names. (`compress_final`
   asks for a GPU it never uses: on MesoPSL the QoS granting a 6 h wall time
   lives on the gpu cluster, which requires `--gres`.)
2. **The conda block.** Three lines loading an environment module, sourcing the
   conda profile and activating `mdis2vihi_env`. Replace them with whatever
   makes the dependencies of `requirements.txt` importable at your site: a
   different module system, a virtual environment, a container.

## What not to change

The **wall times**. Extending a sparse ~245 GB BIGTIFF on a parallel filesystem
stalls for about 30 minutes on the first chunk, and a QoS capped at 1 h cannot
run the mosaic at all. The inference job asks for 24 h to cover a ~5-6 h write.

Also keep `--compress none` in the inference call and the separate compression
job: the chunked rasterio writer corrupts the heap on `deflate` + `predictor=3`.

## Submitting

Submit **from the repository root** (the scripts resolve it via
`SLURM_SUBMIT_DIR`) and pass the account on the command line:

```bash
sbatch -A <your_account> slurm/predict_final.sbatch
sbatch -A <your_account> slurm/compress_final.sbatch
```

Add `--mail-user=<you@example.org> --mail-type=ALL` if you want notifications.

Test on a small region first. Add `--roi COL ROW WIDTH HEIGHT` to the
`04_predict_mosaic.py` command inside `predict_final.sbatch` (for example
`--roi 11000 5500 4096 1024`) before committing.