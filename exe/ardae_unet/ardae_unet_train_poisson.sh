#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/DIV2K_train_HR"
GPU=0
EPOCH=128
SHAPE=128
CHANNELS=3
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
BATCH_SIZE=64
NUM_WORKERS=8
NOISE="poisson"

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/train_ardae.py \
  --data "$DATA" \
  --data-mode image-folder \
  --input-dim $INPUT_DIM \
  --epochs $EPOCH \
  --batch-size $BATCH_SIZE \
  --backbone unet \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 64 \
  --channels $CHANNELS \
  --max-patches-per-image 512 \
  --patch-loader stream \
  --num-workers $NUM_WORKERS \
  --persistent-workers \
  --prefetch-factor 4 \
  --base-channels 32 \
  --channel-mults 1,2,4 \
  --nonlinearity silu \
  --noise-type $NOISE \
  --noise-param 50 \
  --smoothing \
  --sigma-min 0.03 \
  --sigma-max 0.22 \
  --save-dir checkpoints/ardae_unet/poisson_lam001_005_smoothing_lazy \
  --use-metric \
  > log_ardae_unet_poisson_lam001_005_smoothing.txt
