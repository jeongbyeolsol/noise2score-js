filename='train_for_debug'

GPU=1

CUDA_VISIBLE_DEVICES=$GPU python test_ardae.py \
  --checkpoint checkpoints/ardae/$filename/best_epoch_0010.pt \
  --data datasets/processed/bsd400_patches_8x8.npy \
  --batch-size 8192 \
  --output-dir checkpoints/ardae/$filename/test_result