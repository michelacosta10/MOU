#!/bin/bash
#SBATCH --job-name=mou_phase_diagrams
#SBATCH --partition=dss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Theoretical resolvability phase diagrams (phase_diagrams_MOU.py).
# Independent of the rest of the pipeline: no simulated/estimated data needed.
# Submit from the MOU/ directory: sbatch sbatch/05_phase_diagrams.sbatch

set -euo pipefail

: "${OUTPUT_DIR:=./figures}"
: "${N:=100}"
: "${DELTA_T:=1.0}"
: "${Q:=10}"
: "${C_FIXED:=0.6}"

echo "Running on $(hostname)"

python3 phase_diagrams_MOU.py \
    --N "$N" \
    --delta-t "$DELTA_T" \
    --q "$Q" \
    --c-fixed "$C_FIXED" \
    --output-dir "$OUTPUT_DIR"
