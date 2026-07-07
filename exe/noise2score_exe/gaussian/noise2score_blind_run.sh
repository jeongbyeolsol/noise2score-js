GPU=0
DATA=BSD68
NOISE=gaussian
SHAPE=128
CHANNELS=1
INPUT_DIM=$((CHANNELS * SHAPE * SHAPE))
INDEX="_003"
CLEAN_DATA="datasets/${DATA}_processed/"


CUDA_VISIBLE_DEVICES=$GPU python ./scripts/run_noise2score_blind.py \
  --checkpoint "checkpoints/ardae_unet/bsd_gaussian_unet_train_test_eval_128_001/best_model.pt" \
  --clean-data ${CLEAN_DATA} \
  --input-dim $INPUT_DIM \
  --batch-size 128 \
  --image-shape $CHANNELS $SHAPE $SHAPE \
  --stride 128 \
  --channels $CHANNELS \
  --noise-type "$NOISE" \
  --noise-param 0.1 \
  --candidate-params 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --tv-weight 1.0 \
  --data-weight 0.0 \
  --output-dir "results/${NOISE}/n2s_unet_blind_${DATA}_${NOISE}" \
  --save-output \
  --copy-info \
  --stitch-output \
  --stitch-format png 