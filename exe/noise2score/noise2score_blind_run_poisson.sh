#!/usr/bin/env bash

GPU=0
DATA=bsd68
NOISE=poisson
SHAPE=128
INPUT_DIM=$((SHAPE * SHAPE))

if [ "$DATA" = "bsd400" ]; then
    PROCESSED=processed_001
elif [ "$DATA" = "bsd68" ]; then
    PROCESSED=processed_test_001
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score_blind.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}/best_model.pt" \
  --clean-data "datasets/${PROCESSED}/${DATA}_patches_${SHAPE}x${SHAPE}.npy" \
  --input-dim $INPUT_DIM \
  --batch-size 256 \
  --image-shape 1 $SHAPE $SHAPE \
  --noise-type "$NOISE" \
  --noise-param 0.1 \
  --candidate-params 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --tv-weight 1.0 \
  --data-weight 0.0 \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}_blind" \
  --save-output \
  --save-output-limit 256 \
  --copy-info