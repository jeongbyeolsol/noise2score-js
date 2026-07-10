#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/labtest3/miniforge3/envs/noise2score/bin/python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

NOISE="${NOISE:-poisson}"
POISSON_PEAK="${POISSON_PEAK:-${NOISE_PARAM:-50}}"
SCORE_SMOOTHING="${SCORE_SMOOTHING:-${SMOOTHING:-0.1}}"
SCORE_SIGMA="${SCORE_SIGMA:-0.1}"

SHAPE="${SHAPE:-128}"
STRIDE="${STRIDE:-64}"
CHANNELS="${CHANNELS:-3}"
INPUT_DIM="${INPUT_DIM:-$((CHANNELS * SHAPE * SHAPE))}"

CHECKPOINT="${CHECKPOINT:-checkpoints/ardae_unet/poisson_lam001_005_smoothing/best_model.pt}"
CLEAN_DATA="${CLEAN_DATA:-datasets/DIV2K_valid_HR_processed/0801.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-results/n2s_unet_div2k_${NOISE}_stitch}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" ./scripts/run_noise2score.py \
  --checkpoint "$CHECKPOINT" \
  --clean-data "$CLEAN_DATA" \
  --data-mode image-folder \
  --input-dim "$INPUT_DIM" \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --image-shape "$CHANNELS" "$SHAPE" "$SHAPE" \
  --patch-size "$SHAPE" \
  --stride "$STRIDE" \
  --channels "$CHANNELS" \
  --recursive-images \
  --noise-type "$NOISE" \
  --poisson-peak "$POISSON_PEAK" \
  --score-sigma "$SCORE_SIGMA" \
  --score-smoothing "$SCORE_SMOOTHING" \
  --score-smoothing-samples "${SCORE_SMOOTHING_SAMPLES:-${SMOOTHING_SAMPLES:-8}}" \
  --output-dir "$OUTPUT_DIR" \
  --stitch-output \
  --stitch-format "${STITCH_FORMAT:-both}" \
  --save-output \
  --save-output-limit "${SAVE_OUTPUT_LIMIT:-16}" \
  --copy-info
