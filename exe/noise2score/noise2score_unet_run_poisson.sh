#!/usr/bin/env bash

GPU=0
DATA=DIV2K_valid_HR
NOISE=poisson
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE)) 
INDEX=""


if [ "$DATA" = "DIV2K_train_HR" ]; then
    PROCESSED=processed
elif [ "$DATA" = "DIV2K_valid_HR" ]; then 
    PROCESSED=processed_test
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}_lam001_005_smoothing/best_model.pt" \
  --clean-data "datasets/${PROCESSED}/${DATA}_patches_${SHAPE}x${SHAPE}.npy" \
  --input-dim $INPUT_DIM \
  --noise-type "$NOISE" \
  --noise-param 50 \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}" \
  --save-output \
  --save-output-limit 64 \
  --copy-info