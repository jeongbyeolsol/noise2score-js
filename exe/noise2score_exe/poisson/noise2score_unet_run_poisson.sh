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

CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score.py \
  --checkpoint "checkpoints/ardae_unet/${NOISE}${INDEX}/best_model.pt" \
  --clean-data "$CLEAN_DATA" \
  --data-mode image-folder \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --patch-size $SHAPE \
  --stride 128 \
  --channels $CHANNELS \
  --num-workers 1 \
  --recursive-images \
  --noise-type "$NOISE" \
  --noise-param 50 \
  --output-dir "results/n2s_unet_${DATA}_${NOISE}" \
  --save-output \
  --copy-info \
  --score-sigma 0.1 \
  --smoothing 0.1 \
  --stitch-output \
  --stitch-format both #\
  #--save-output-limit 64