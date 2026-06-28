python run_noise2score.py \
  --checkpoint checkpoints/ardae/best_epoch_0100.pt \
  --clean-data data/test.npy \
  --input-dim 784 \
  --noise-type gaussian \
  --noise-param 0.1 \
  --score-sigma 0.01 \
  --output-dir results/n2s_gaussian_01