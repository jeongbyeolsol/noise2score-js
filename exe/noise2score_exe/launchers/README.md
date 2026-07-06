Noise2Score experiment launchers
================================

These scripts wrap the newer unified `scripts/run_noise2score.py` entrypoint.
They are intentionally controlled by environment variables so the same file can
be used for quick sanity checks and longer runs.

Common overrides:

```bash
GPU=0 ./exe/noise2score_exe/launchers/eval_div2k_poisson_stitch.sh
DEVICE=cpu ./exe/noise2score_exe/launchers/train_test_eval_bsd_gaussian_unet.sh
```

Scripts:

- `train_test_eval_bsd_gaussian_unet.sh`
  Trains ARDAE on BSD400 patch arrays, tests ARDAE on BSD68, then runs
  Noise2Score on BSD68.

- `eval_bsd_gaussian_unet.sh`
  Uses an existing Gaussian U-Net ARDAE checkpoint and runs Noise2Score on BSD68
  patch arrays.

- `eval_div2k_poisson_stitch.sh`
  Uses an existing Poisson U-Net ARDAE checkpoint, denoises image patches from
  DIV2K, and stitches denoised patches back into full images.

- `train_test_eval_div2k_poisson_stitch.sh`
  Trains ARDAE on DIV2K image folders, optionally tests it on an array file,
  then runs Noise2Score with stitched full-image output.

For stitched experiments, the default `CLEAN_DATA` is a single validation image
for a fast smoke test. Set `CLEAN_DATA=datasets/DIV2K_valid_HR_processed` for a
full validation run.
