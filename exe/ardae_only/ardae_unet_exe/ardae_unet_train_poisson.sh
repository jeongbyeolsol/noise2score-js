#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/DIV2K_train_HR_processed"
GPU=0
EPOCH=64
SHAPE=128
CHANNELS=3
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
BATCH_SIZE=128
NUM_WORKERS=0
NOISE="poisson"

CUDA_VISIBLE_DEVICES=$GPU python ./models/ardae/train_ardae.py \
  --data "$DATA" \
  --data-mode image-folder \
  --input-dim $INPUT_DIM \
  --epochs $EPOCH \
  --batch-size $BATCH_SIZE \
  --backbone unet \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 128 \
  --channels $CHANNELS \
  --max-patches-per-image 128 \
  --patch-loader stream \
  --num-workers $NUM_WORKERS \
  --persistent-workers \
  --prefetch-factor 4 \
  --base-channels 32 \
  --channel-mults 1,2,4 \
  --nonlinearity silu \
  --noise-type $NOISE \
  --poisson-peak 50 \
  --gaussian-perturbation 0.1 \
  --save-dir checkpoints/ardae_unet/poisson_lam001_005_gaussian_perturb \
  --use-metric \
  --recursive-images \
  > log_ardae_unet_poisson_lam001_005_gaussian_perturb.txt
