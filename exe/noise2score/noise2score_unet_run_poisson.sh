#!/usr/bin/env bash

GPU=0
DATA=bsd68
NOISE=poisson
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE)) 
INDEX="_001"


if [ "$DATA" = "bsd400" ]; then
    PROCESSED=processed_001
elif [ "$DATA" = "bsd68" ]; then 
    PROCESSED=processed_test_001
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}/best_model.pt" \
  --clean-data "datasets/${PROCESSED}/${DATA}_patches_${SHAPE}x${SHAPE}.npy" \
  --input-dim $INPUT_DIM \
  --noise-type "$NOISE" \
  --noise-param 100 \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}" \
  --save-output \
  --save-output-limit 64