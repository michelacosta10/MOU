#!/bin/bash
#SBATCH --job-name=mou_generate
#SBATCH --partition=dss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Simulates MOU trajectories (generate_data_MOU.py).
# Submit from the MOU/ directory: sbatch sbatch/01_generate.sbatch
# Override defaults at submit time, e.g.:
#   sbatch --export=ALL,TAG=myrun,DATA_DIR=./data sbatch/01_generate.sbatch

set -euo pipefail

: "${DATA_DIR:=./data}"
: "${TAG:=6JUN}"
: "${ENSEMBLE:=GOE}"
: "${C_VALUES:=0.1 0.2 0.3 0.4 0.5 0.6 0.7}"

# One process per core, no oversubscription from BLAS threads inside each worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running on $(hostname), $(nproc) CPUs allocated"
echo "DATA_DIR=$DATA_DIR  TAG=$TAG  ENSEMBLE=$ENSEMBLE  C_VALUES=$C_VALUES"

python3 generate_data_MOU.py \
    --data-dir "$DATA_DIR" \
    --ensemble "$ENSEMBLE" \
    --c-values $C_VALUES \
    --tag "$TAG"
