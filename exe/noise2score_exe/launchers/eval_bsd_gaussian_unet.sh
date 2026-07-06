#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/labtest3/miniforge3/envs/noise2score/bin/python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

NOISE="${NOISE:-gaussian}"
NOISE_PARAM="${NOISE_PARAM:-0.1}"
SCORE_SIGMA="${SCORE_SIGMA:-0.1}"

SHAPE="${SHAPE:-128}"
CHANNELS="${CHANNELS:-1}"
INPUT_DIM="${INPUT_DIM:-$((CHANNELS * SHAPE * SHAPE))}"

CHECKPOINT="${CHECKPOINT:-checkpoints/ardae_unet/gaussian_002/best_model.pt}"
CLEAN_DATA="${CLEAN_DATA:-datasets/processed_test/bsd68_patches_${SHAPE}x${SHAPE}.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-results/n2s_unet_bsd68_${NOISE}_eval}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" ./scripts/run_noise2score.py \
  --checkpoint "$CHECKPOINT" \
  --clean-data "$CLEAN_DATA" \
  --input-dim "$INPUT_DIM" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --image-shape "$CHANNELS" "$SHAPE" "$SHAPE" \
  --noise-type "$NOISE" \
  --noise-param "$NOISE_PARAM" \
  --score-sigma "$SCORE_SIGMA" \
  --output-dir "$OUTPUT_DIR" \
  --save-output \
  --save-output-limit "${SAVE_OUTPUT_LIMIT:-64}" \
  --copy-info
