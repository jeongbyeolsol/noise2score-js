#!/usr/bin/env bash

GPU=0
DATA=CBSD100_noisy
NOISE=poisson
SHAPE=128
CHANNELS=3
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_lam001_005_smoothing"
OUTPUT_INDEX="_lam001_005_gaussian_perturb"

CLEAN_DATA="datasets/CBSD100_noisy_processed"


CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score_blind.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}/best_model.pt"  \
  --clean-data "$CLEAN_DATA" \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --data-mode image-folder \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 64 \
  --channels $CHANNELS \
  --noise-type "$NOISE" \
  --poisson-peak 50 \
  --candidate-params 20,25,30,40,50,67,80,100 \
  --candidate-score-smoothing 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --tv-weight 1.0 \
  --data-weight 0.0 \
  --output-dir "results/n2s_unet_blind_${DATA}_${NOISE}_${OUTPUT_INDEX}" \
  --save-output \
  --copy-info \
  --score-smoothing 0.1 \
  --stitch-output \
  --stitch-format png
