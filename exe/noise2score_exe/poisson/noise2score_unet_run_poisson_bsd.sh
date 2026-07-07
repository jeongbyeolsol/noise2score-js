#!/usr/bin/env bash

GPU=0
DATA=bsd68
NOISE=poisson
SHAPE=128
CHANNELS=1
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_001"



if [ "$DATA" = "bsd400" ]; then
    CLEAN_DATA="datasets/processed/bsd400_patches_128x128.npy"
elif [ "$DATA" = "bsd68" ]; then 
    CLEAN_DATA="datasets/processed_test/bsd68_patches_128x128.npy"
else
    echo "잘못된 DATA: $DATA"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}/best_model.pt" \
  --clean-data "$CLEAN_DATA" \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 128 \
  --channels $CHANNELS \
  --max-patches-per-image 128 \
  --num-workers 0 \
  --recursive-images \
  --noise-type "$NOISE" \
  --poisson-peak 100 \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}" \
  --save-output \
  --copy-info
