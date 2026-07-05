#!/usr/bin/env bash

GPU=0
DATA=bsd68
NOISE=gaussian
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE)) 

if [ "$DATA" = "bsd400" ]; then
    PROCESSED=processed
elif [ "$DATA" = "bsd68" ]; then 
    PROCESSED=processed_test
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae/${NOISE}/best_model.pt" \
  --clean-data "datasets/${PROCESSED}/${DATA}_patches_${SHAPE}x${SHAPE}.npy" \
  --input-dim $INPUT_DIM \
  --noise-type "$NOISE" \
  --noise-param 0.1 \
  --score-sigma 0.1 \
  --output-dir "results/n2s_${DATA}_${NOISE}" \
  --save-output \
  --save-output-limit 64