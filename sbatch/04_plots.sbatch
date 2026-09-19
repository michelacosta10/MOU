#!/bin/bash
#SBATCH --job-name=mou_plots
#SBATCH --partition=dss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generates the analysis figures from the processed dataset (plots_MOU.py).
# Requires 03_process.sbatch to have finished first (same DATA_DIR/TAG/ENSEMBLE/LAG).
# Submit from the MOU/ directory: sbatch sbatch/04_plots.sbatch

set -euo pipefail

: "${DATA_DIR:=./data}"
: "${TAG:=6JUN}"
: "${ENSEMBLE:=GOE}"
: "${LAG:=1}"
: "${OUTPUT_DIR:=./figures}"

echo "Running on $(hostname)"

python3 plots_MOU.py \
    --dataset "${DATA_DIR}/processed/dataset_${ENSEMBLE}_lag${LAG}_${TAG}.pkl" \
    --output-dir "$OUTPUT_DIR"
