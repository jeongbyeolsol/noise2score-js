#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/labtest3/miniforge3/envs/noise2score/bin/python}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-config/noise2score_blind_poisson_div2k_stitch.json}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" ./scripts/run_noise2score_blind_poisson.py \
  --config "$CONFIG" \
  "$@"
