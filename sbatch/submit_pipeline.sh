#!/bin/bash
# Submits the full MOU pipeline as a chain of SLURM jobs with dependencies:
# generate -> estimate -> process -> plots (and phase_diagrams, independently).
#
# Usage (from the MOU/ directory):
#   ./sbatch/submit_pipeline.sh
#   TAG=myrun DATA_DIR=./data ./sbatch/submit_pipeline.sh
#
# Each stage only starts if the previous one finished successfully (afterok).

set -euo pipefail

: "${DATA_DIR:=./data}"
: "${TAG:=6JUN}"
: "${ENSEMBLE:=GOE}"
: "${C_VALUES:=0.1 0.2 0.3 0.4 0.5 0.6 0.7}"
: "${LAG:=1}"
: "${OUTPUT_DIR:=./figures}"

mkdir -p logs "$DATA_DIR" "$OUTPUT_DIR"

EXPORT_VARS="ALL,DATA_DIR=${DATA_DIR},TAG=${TAG},ENSEMBLE=${ENSEMBLE},C_VALUES=${C_VALUES},LAG=${LAG},OUTPUT_DIR=${OUTPUT_DIR}"

JOB1=$(sbatch --export="$EXPORT_VARS" --parsable sbatch/01_generate.sbatch)
echo "Submitted 01_generate: job $JOB1"

JOB2=$(sbatch --export="$EXPORT_VARS" --dependency=afterok:$JOB1 --parsable sbatch/02_estimate.sbatch)
echo "Submitted 02_estimate: job $JOB2 (after $JOB1)"

JOB3=$(sbatch --export="$EXPORT_VARS" --dependency=afterok:$JOB2 --parsable sbatch/03_process.sbatch)
echo "Submitted 03_process: job $JOB3 (after $JOB2)"

JOB4=$(sbatch --export="$EXPORT_VARS" --dependency=afterok:$JOB3 --parsable sbatch/04_plots.sbatch)
echo "Submitted 04_plots: job $JOB4 (after $JOB3)"

JOB5=$(sbatch --export="$EXPORT_VARS" --parsable sbatch/05_phase_diagrams.sbatch)
echo "Submitted 05_phase_diagrams: job $JOB5 (independent)"

echo ""
echo "Track progress with: squeue -u $USER"
echo "Logs land in: logs/"
