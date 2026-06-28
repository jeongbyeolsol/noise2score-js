set -euo pipefail

python train_ardae_only.py \
  --data datasets/processed/bsd400_patches_8x8.npy \
  --input-dim 64 \
  --epochs 10 \
  --batch-size 8192 \
  --noise-type gaussian \
  --noise-param 0.1  \
  --save-dir checkpoints/ardae/test_for_debug