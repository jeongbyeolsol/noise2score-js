#!/usr/bin/env bash

GPU=0
DATA=DIV2K_valid_HR
NOISE=poisson
SHAPE=128
CHANNELS=3
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_lam001_005_smoothing"

if [ "$DATA" = "DIV2K_train_HR" ]; then
    CLEAN_DATA="datasets/DIV2K_train_HR_processed"
elif [ "$DATA" = "DIV2K_valid_HR" ]; then
    CLEAN_DATA="datasets/DIV2K_valid_HR_processed"
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score_blind.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}/best_model.pt"  \
  --clean-data "$CLEAN_DATA" \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --data-mode image-folder \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --noise-type "$NOISE" \
  --noise-param 0.1 \
  --candidate-params 20,30,50,80,100 \
  --score-sigma-mode fixed \
  --fixed-score-sigma 0.1 \
  --smoothing 0.1 \
  --tv-weight 1.0 \
  --data-weight 0.0 \
  --output-dir "results/n2s_unet_blind_${DATA}_${NOISE}_${INDEX}" \
  --save-output \
  --copy-info
