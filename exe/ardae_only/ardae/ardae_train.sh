#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/processed/bsd400_patches_40x40.npy"
GPU=1
EPOCH=200
INPUT_DIM=1600
BATCH_SIZE=8192

CUDA_VISIBLE_DEVICES=$GPU python ./models/ardae/train_ardae.py \
  --data "$DATA" \
  --input-dim $INPUT_DIM \
  --epochs $EPOCH \
  --batch-size $BATCH_SIZE \
  --h-dim 200 \
  --num-hidden-layers 2 \
  --noise-type gaussian \
  --noise-param 0.1 \
  --save-dir checkpoints/ardae/gaussian \
  --use-metric \
  > log_gaussian.txt 2>&1


#CUDA_VISIBLE_DEVICES=$GPU python ./models/ardae/train_ardae.py \
#  --data "$DATA" \
#  --input-dim $INPUT_DIM \
#  --epochs $EPOCH \
#  --batch-size $BATCH_SIZE \
#  --h-dim 200 \
#  --num-hidden-layers 2 \
#  --noise-type poisson \
#  --noise-param 30.0 \
#  --save-dir checkpoints/ardae/poisson \
#  --use-metric \
#  > log_poisson.txt 2>&1;

#CUDA_VISIBLE_DEVICES=$GPU python ./models/ardae/train_ardae.py \
#  --data "$DATA" \
#  --input-dim $INPUT_DIM \
#  --epochs $EPOCH \
#  --batch-size $BATCH_SIZE \
#  --h-dim 200 \
#  --num-hidden-layers 2 \
#  --noise-type gamma \
#  --noise-param 50.0 \
#  --save-dir checkpoints/ardae/gamma \
#  --use-metric \
#  > log_gamma.txt 2>&1;
