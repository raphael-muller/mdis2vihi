#!/bin/bash
#
# Regenerate the deliverable mosaic from the FINAL 9-input model.
# Pixel-by-pixel inference over the full MDIS mosaic, feeding band 9 (the count of
# 8-colour image sets) as the 9th input, z-standardized with the train stats in
# image_count_stats.json.
# Writes uncompressed BIGTIFF (compress none: the chunked rasterio writer
# heap-corrupts on deflate+predictor; compress separate with compress_final.sbatch).
#
# Submission (from the repository root on the cluster):
#     sbatch -A <slurm_account> slurm/predict_final.sbatch
# Output: runs/final/mdis2vihi_global_final.tif  (~245 GB uncompressed)
#
# Prereqs: runs/final/ present (checkpoint + image_count_stats.json),
#          data/raw/mdis_mosaic/ present.
#
# THIS IS THE JOB AS SUBMITTED ON MesoPSL, kept verbatim as the record of what
# produced the deliverable. On another cluster, adapt:
#   - the --clusters/--partition/--qos/--gres directives below (MesoPSL names);
#   - the conda block (module names, profile path, environment), or replace it
#     entirely with whatever activates the dependencies at your site.
# Do NOT shorten the wall times: extending a sparse ~245 GB BIGTIFF on a
# parallel filesystem stalls ~30 min on the first chunk (docs/CLUSTER.md).
#
#SBATCH --job-name=mdis2vihi-predfinal
#SBATCH --time=1-00:00:00
#SBATCH --clusters=gpu
#SBATCH --partition=hi
#SBATCH --qos=gpu_hi_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/predfinal-%j.out
#SBATCH --error=slurm/logs/predfinal-%j.err

set -euo pipefail

# repository root: the directory the job was submitted from
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# ---- environment (MesoPSL; adapt to your site) -----------------------------
module load anaconda/3-2024.10
. /shared/apps/anaconda/3-2024.10/etc/profile.d/conda.sh
conda activate mdis2vihi_env

cd "$REPO"
mkdir -p slurm/logs

# best checkpoint (save_top_k=1 -> the single epoch=*.ckpt that is not last.ckpt);
# tail -1 picks the latest training version if several exist.
CKPT=$(ls "$REPO"/runs/final/lightning_logs/version_*/checkpoints/epoch=*.ckpt | tail -1)
STATS="$REPO/runs/final/image_count_stats.json"
OUTPUT="$REPO/runs/final/mdis2vihi_global_final.tif"

echo "[$(date -Iseconds)] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[$(date -Iseconds)] ckpt=$CKPT"
echo "[$(date -Iseconds)] stats=$STATS"
echo "[$(date -Iseconds)] output=$OUTPUT"

# remove leftover truncated BIGTIFF from a previous failed run (libtiff
# TIFFReadDirectory crash on re-open otherwise)
rm -f "$OUTPUT"

python scripts/04_predict_mosaic.py \
    --ckpt "$CKPT" \
    --output "$OUTPUT" \
    --device cuda \
    --tile-rows 256 \
    --forward-batch 5000000 \
    --compress none \
    --count-band 9 \
    --count-stats "$STATS"

echo "[$(date -Iseconds)] done"
ls -lh "$OUTPUT"