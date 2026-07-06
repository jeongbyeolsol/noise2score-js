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

TRAIN_DATA="${TRAIN_DATA:-datasets/BSD400_processed}"
TEST_DATA="${TEST_DATA:-datasets/BSD68_processed}"
CLEAN_DATA="${CLEAN_DATA:-$TEST_DATA}"

BATCH_SIZE="${BATCH_SIZE:-128}"
ARDAE_BATCH_SIZE="${ARDAE_BATCH_SIZE:-$BATCH_SIZE}"
EPOCHS="${EPOCHS:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUN_NAME="${RUN_NAME:-bsd_${NOISE}_unet_train_test_eval_${SHAPE}}"
ARDAE_SAVE_DIR="${ARDAE_SAVE_DIR:-checkpoints/ardae_unet/${RUN_NAME}}"
ARDAE_TEST_OUTPUT_DIR="${ARDAE_TEST_OUTPUT_DIR:-results/${RUN_NAME}_ardae_test}"
OUTPUT_DIR="${OUTPUT_DIR:-results/n2s_${RUN_NAME}}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" ./scripts/run_noise2score.py \
  --clean-data "$CLEAN_DATA" \
  --ardae-train-data "$TRAIN_DATA" \
  --ardae-test-data "$TEST_DATA" \
  --input-dim "$INPUT_DIM" \
  --batch-size "$BATCH_SIZE" \
  --ardae-batch-size "$ARDAE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --image-shape "$CHANNELS" "$SHAPE" "$SHAPE" \
  --noise-type "$NOISE" \
  --noise-param "$NOISE_PARAM" \
  --score-sigma "$SCORE_SIGMA" \
  --train-ardae \
  --test-ardae \
  --ardae-save-dir "$ARDAE_SAVE_DIR" \
  --ardae-test-output-dir "$ARDAE_TEST_OUTPUT_DIR" \
  --ardae-epochs "$EPOCHS" \
  --ardae-backbone unet \
  --ardae-base-channels "${BASE_CHANNELS:-32}" \
  --ardae-channel-mults "${CHANNEL_MULTS:-1,2,4}" \
  --ardae-nonlinearity "${NONLINEARITY:-silu}" \
  --ardae-sigma-min "${SIGMA_MIN:-0.001}" \
  --ardae-sigma-max "${SIGMA_MAX:-0.5}" \
  --ardae-use-metric \
  --output-dir "$OUTPUT_DIR" \
  --save-output \
  --save-output-limit "${SAVE_OUTPUT_LIMIT:-64}" \
  --copy-info \
  --stitch-output \
  --ardae-train-data-mode image-folder \
  --stitch-format png
