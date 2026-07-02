#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/processed/bsd400_patches_40x40.npy"
GPU=1
EPOCH=100
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE))  # 산술 연산 수정
BATCH_SIZE=256

CUDA_VISIBLE_DEVICES=$GPU python train_ardae.py \
  --data "$DATA" \
  --input-dim $INPUT_DIM \
  --epochs $EPOCH \
  --batch-size $BATCH_SIZE \
  --backbone unet \
  --image-shape 1 $SHAPE $SHAPE \
  --base-channels 32 \
  --channel-mults 1,2,4 \
  --nonlinearity silu \
  --noise-type gaussian \
  --noise-param 0.1 \
  --sigma-min 0.001 \
  --sigma-max 0.5 \
  --save-dir checkpoints/ardae_unet/gaussian \
  --use-metric \
  > log_ardae_unet.txt 2>&1

# 실험 결과 가우시안 제외 나빠짐 -> 가우시안 고정
