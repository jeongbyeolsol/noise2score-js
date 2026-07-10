#!/usr/bin/env bash

GPU=0
DATA=DIV2K_valid_HR
NOISE=poisson
SHAPE=128
CHANNELS=3
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_lam001_005_smoothing_001"
OUTPUT_INDEX="_lam001_005_gaussian_perturb_001"

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
  --patch-size $SHAPE \
  --stride 64 \
  --channels $CHANNELS \
  --noise-type "$NOISE" \
  --poisson-peak 50 \
  --candidate-params 50 \
  --candidate-score-smoothing 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --tv-weight 1.0 \
  --data-weight 0.0 \
  --output-dir "results/poisson/n2s_unet_blind_${DATA}_${NOISE}_${OUTPUT_INDEX}" \
  --save-output \
  --copy-info \
  --score-smoothing 0.1 \
  --stitch-output \
  --stitch-format png \
  --save-output-limit 4
