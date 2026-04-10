#!/usr/bin/env bash
set -euo pipefail

# Example one-click launcher.
# You can override any env var below when invoking the script.

TASK="${TASK:-libero_uncond_2cam224_1e-4}"
CKPT="${CKPT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
HORIZONS="${HORIZONS:-5,9,13,17}"
SUITE="${SUITE:-libero_spatial}"
NUM_GPUS="${NUM_GPUS:-1}"
DREAM_MODE="${DREAM_MODE:-off}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./evaluate_results/horizon_sweep}"

python exoeriments/libero_horizon/run_horizon_sweep.py \
  --task "$TASK" \
  --ckpt "$CKPT" \
  --dataset-stats "$DATASET_STATS" \
  --horizons "$HORIZONS" \
  --suite "$SUITE" \
  --num-gpus "$NUM_GPUS" \
  --dream-mode "$DREAM_MODE" \
  --output-root "$OUTPUT_ROOT"
