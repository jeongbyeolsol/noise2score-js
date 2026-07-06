#!/usr/bin/env bash

GPU=0
DATA=BSD68
NOISE=gaussian
SHAPE=128
CHANNELS=1
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_003"
CLEAN_DATA="datasets/${DATA}_processed/"


CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae_unet/bsd_gaussian_unet_train_test_eval_128_001/best_model.pt" \
  --clean-data ${CLEAN_DATA} \
  --data-mode image-folder \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 128 \
  --channels $CHANNELS \
  --noise-type "$NOISE" \
  --noise-param 0.1 \
  --score-sigma 0.1 \
  --recursive-images \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}" \
  --save-output \
  --copy-info \
  --stitch-output \
  --stitch-format png 
