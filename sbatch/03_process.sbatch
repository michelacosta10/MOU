#!/bin/bash
#SBATCH --job-name=mou_process
#SBATCH --partition=dss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Loads the estimation files and computes all derived quantities (process_MOU.py).
# Requires 02_estimate.sbatch to have finished first (same DATA_DIR/TAG/ENSEMBLE/C_VALUES/LAG).
# Submit from the MOU/ directory: sbatch sbatch/03_process.sbatch

set -euo pipefail

: "${DATA_DIR:=./data}"
: "${TAG:=6JUN}"
: "${ENSEMBLE:=GOE}"
: "${C_VALUES:=0.1 0.2 0.3 0.4 0.5 0.6 0.7}"
: "${LAG:=1}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running on $(hostname), $(nproc) CPUs allocated"
echo "DATA_DIR=$DATA_DIR  TAG=$TAG  ENSEMBLE=$ENSEMBLE  C_VALUES=$C_VALUES  LAG=$LAG"

python3 process_MOU.py \
    --data-dir "$DATA_DIR" \
    --ensemble "$ENSEMBLE" \
    --c-values $C_VALUES \
    --tag "$TAG" \
    --lag "$LAG"
