#!/bin/bash
#SBATCH --job-name=mou_estimate
#SBATCH --partition=dss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Estimates A, Q, Sigma_dt, Sigma_inf from the simulated trajectories (estimate_MOU.py).
# Requires 01_generate.sbatch to have finished first (same DATA_DIR/TAG/ENSEMBLE/C_VALUES).
# Submit from the MOU/ directory: sbatch sbatch/02_estimate.sbatch

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

python3 estimate_MOU.py \
    --data-dir "$DATA_DIR" \
    --ensemble "$ENSEMBLE" \
    --c-values $C_VALUES \
    --tag "$TAG" \
    --lag "$LAG"
