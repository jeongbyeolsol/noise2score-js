#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/processed_001/bsd400_patches_128x128.npy"
GPU=0
EPOCH=256
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE)) 
BATCH_SIZE=256
NOISE='gaussian'

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/train_ardae.py \
  --data "$DATA" \
  --input-dim $INPUT_DIM \
  --epochs $EPOCH \
  --batch-size $BATCH_SIZE \
  --backbone unet \
  --image-shape 1 $SHAPE $SHAPE \
  --base-channels 32 \
  --channel-mults 1,2,4 \
  --nonlinearity silu \
  --noise-type $NOISE \
  --noise-param 0.1 \
  --sigma-min 0.001 \
  --sigma-max 0.5 \
  --save-dir checkpoints/ardae_unet/${NOISE} \
  --use-metric \
  > log_ardae_unet.txt # 2>&1  // because of tqdm
