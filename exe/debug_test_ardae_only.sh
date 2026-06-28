python test_ardae.py \
  --checkpoint checkpoints/ardae/train_for_debug/best_epoch_0010.pt \
  --data datasets/processed/bsd400_patches_8x8.npy \
  --batch-size 8192 \
  --output-dir checkpoints/ardae/test_for_debug/test_result