#!/usr/bin/env bash
set -euo pipefail

DATA="datasets/processed/bsd400_patches_8x8.npy"
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python train_ardae_only.py \
  --data "$DATA" \
  --input-dim 64 \
  --epochs 100 \
  --batch-size 8192 \
  --h-dim 1000 \
  --num-hidden-layers 2 \
  --noise-type gaussian \
  --noise-param 0.1 \
  --save-dir checkpoints/ardae/gaussian \
  > log_gaussian.txt 2>&1;

CUDA_VISIBLE_DEVICES=$GPU python train_ardae_only.py \
  --data "$DATA" \
  --input-dim 64 \
  --epochs 100 \
  --batch-size 8192 \
  --h-dim 1000 \
  --num-hidden-layers 2 \
  --noise-type poisson \
  --noise-param 30.0 \
  --save-dir checkpoints/ardae/poisson \
  > log_poisson.txt 2>&1;

CUDA_VISIBLE_DEVICES=$GPU python train_ardae_only.py \
  --data "$DATA" \
  --input-dim 64 \
  --epochs 100 \
  --batch-size 8192 \
  --h-dim 1000 \
  --num-hidden-layers 2 \
  --noise-type gamma \
  --noise-param 2.0 \
  --save-dir checkpoints/ardae/gamma \
  > log_gamma.txt 2>&1;