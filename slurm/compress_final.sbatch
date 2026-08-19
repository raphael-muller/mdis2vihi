#!/bin/bash
#
# Separate step lossless compression of the FINAL hyperspectral mosaic
# (slurm/predict_final.sbatch output). gdal_translate -> DEFLATE + PREDICTOR=2.
# Source is read-only; never altered. PREDICTOR=3 is banned (libgdal heap
# corruption on this cluster, see docs/CLUSTER.md).
#
# Submission (from the repository root on the cluster):
#     sbatch -A <slurm_account> slurm/compress_final.sbatch
# Output: runs/final/mdis2vihi_global_final_deflate.tif  (the deliverable)
#
# THIS IS THE JOB AS SUBMITTED ON MesoPSL, kept verbatim as the record of what
# produced the deliverable. On another cluster, adapt the
# --clusters/--partition/--qos/--gres directives and the conda block below.
# The GPU request is not used by gdal_translate: on MesoPSL the QoS that allows
# a 6 h wall time lives on the gpu cluster, which requires --gres.
#
#SBATCH --job-name=mdis2vihi-compfinal
#SBATCH --time=06:00:00
#SBATCH --clusters=gpu
#SBATCH --partition=hi
#SBATCH --qos=gpu_hi_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/compfinal-%j.out
#SBATCH --error=slurm/logs/compfinal-%j.err

set -euo pipefail

# repository root: the directory the job was submitted from
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# ---- environment (MesoPSL; adapt to your site) -----------------------------
module load anaconda/3-2024.10
. /shared/apps/anaconda/3-2024.10/etc/profile.d/conda.sh
conda activate mdis2vihi_env

cd "$REPO"
mkdir -p slurm/logs

SRC="$REPO/runs/final/mdis2vihi_global_final.tif"
DST="$REPO/runs/final/mdis2vihi_global_final_deflate.tif"

echo "[$(date -Iseconds)] node=$(hostname)"
echo "[$(date -Iseconds)] src=$(ls -lh $SRC)"
echo "[$(date -Iseconds)] scratch free: $(df -h "$(dirname "$DST")" | tail -1)"

rm -f "$DST"

gdal_translate \
    -co COMPRESS=DEFLATE \
    -co PREDICTOR=2 \
    -co BIGTIFF=YES \
    -co TILED=YES \
    -co BLOCKXSIZE=512 \
    -co BLOCKYSIZE=256 \
    -co NUM_THREADS=ALL_CPUS \
    "$SRC" "$DST"

echo "[$(date -Iseconds)] done"
ls -lh "$SRC" "$DST"
echo "[$(date -Iseconds)] gdalinfo on output:"
# pipefail + gdalinfo|head -> SIGPIPE false-FAIL guard (docs/CLUSTER.md)
set +o pipefail
gdalinfo "$DST" | head -30
set -o pipefail