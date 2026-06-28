python train_ardae_only.py \
  --data datasets/datasets/processed/bsd400_patches_8x8.npy \
  --input-dim 2 \
  --epochs 100 \
  --batch-size 128 \
  --noise-type gaussian \
  --noise-param 0.1