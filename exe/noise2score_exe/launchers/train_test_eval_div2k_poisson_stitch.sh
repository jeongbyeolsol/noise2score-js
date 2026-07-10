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

TRAIN_DATA="${TRAIN_DATA:-datasets/DIV2K_train_HR_processed}"
CLEAN_DATA="${CLEAN_DATA:-datasets/DIV2K_valid_HR_processed/0801.npy}"
ARDAE_TEST_DATA="${ARDAE_TEST_DATA:-}"

BATCH_SIZE="${BATCH_SIZE:-64}"
ARDAE_BATCH_SIZE="${ARDAE_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUN_NAME="${RUN_NAME:-div2k_${NOISE}_unet_gaussian_perturb_${SHAPE}}"
ARDAE_SAVE_DIR="${ARDAE_SAVE_DIR:-checkpoints/ardae_unet/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/n2s_${RUN_NAME}}"

cmd=(
  "$PYTHON_BIN" ./scripts/run_noise2score.py
  --clean-data "$CLEAN_DATA"
  --data-mode image-folder
  --ardae-train-data "$TRAIN_DATA"
  --ardae-train-data-mode image-folder
  --input-dim "$INPUT_DIM"
  --batch-size "$BATCH_SIZE"
  --ardae-batch-size "$ARDAE_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device "$DEVICE"
  --image-shape "$CHANNELS" "$SHAPE" "$SHAPE"
  --patch-size "$SHAPE"
  --stride "$STRIDE"
  --channels "$CHANNELS"
  --recursive-images
  --noise-type "$NOISE"
  --poisson-peak "$POISSON_PEAK"
  --score-sigma "$SCORE_SIGMA"
  --score-smoothing "$SCORE_SMOOTHING"
  --score-smoothing-samples "${SCORE_SMOOTHING_SAMPLES:-${SMOOTHING_SAMPLES:-8}}"
  --train-ardae
  --ardae-save-dir "$ARDAE_SAVE_DIR"
  --ardae-epochs "$EPOCHS"
  --ardae-backbone unet
  --ardae-base-channels "${BASE_CHANNELS:-32}"
  --ardae-channel-mults "${CHANNEL_MULTS:-1,2,4}"
  --ardae-nonlinearity "${NONLINEARITY:-silu}"
  --ardae-patch-loader "${PATCH_LOADER:-stream}"
  --ardae-max-patches-per-image "${ARDAE_MAX_PATCHES_PER_IMAGE:-128}"
  --ardae-gaussian-perturbation
  --ardae-sigma-min "${SIGMA_MIN:-0.03}"
  --ardae-sigma-max "${SIGMA_MAX:-0.22}"
  --ardae-use-metric
  --output-dir "$OUTPUT_DIR"
  --stitch-output
  --stitch-format "${STITCH_FORMAT:-both}"
  --save-output
  --save-output-limit "${SAVE_OUTPUT_LIMIT:-16}"
  --copy-info
)

if [[ -n "$ARDAE_TEST_DATA" ]]; then
  cmd+=(--test-ardae --ardae-test-data "$ARDAE_TEST_DATA")
fi

CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
