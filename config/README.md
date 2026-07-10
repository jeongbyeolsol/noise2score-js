# Noise2Score Configs

Run scripts accept `--config path/to/file.json`.
Values in the config become argparse defaults, and explicit CLI arguments override them.

Top-level flat argparse names work:

```json
{
  "clean_data": "datasets/BSD68_processed",
  "input_dim": 16384,
  "noise_type": "gaussian"
}
```

Nested sections are also supported:

- `runtime`: `input_dim`, `batch_size`, `device`, `seed`, `num_workers`
- `data`: `clean_data`, `noisy_data`, `data_mode`, `key`, `noisy_key`
- `image`: `image_shape`, `patch_size`, `stride`, `channels`, `recursive_images`
- `noise`: `noise_type`, `noise_param`, `poisson_peak`, `score_sigma`, `score_smoothing`
- `output`: `output_dir`, `save_output`, `save_output_limit`, `stitch_output`, `stitch_format`, `copy_info`
- `checkpoint`: `checkpoint`, `noise2score_checkpoint_output`
- `blind`: `candidate_params`, `candidate_score_smoothing`, `score_sigma_mode`, quality weights
- `ardae`: ARDAE options without the `ardae_` prefix, plus `train`, `test`, and `gaussian_perturbation`

Example:

```bash
python ./scripts/run_noise2score_poisson.py \
  --config config/noise2score_poisson_div2k_stitch.json
```

Distribution-specific entrypoints also accept clearer names:

- Gaussian: `--sigma`, `--candidate-sigmas`, or config section `gaussian.sigma`
- Poisson: `--peak`, `--lam`, `--peak-min/--peak-max`, `--lam-min/--lam-max`, `--candidate-peaks`, `--candidate-lams`, or config section `poisson.peak`
- Gamma: `--alpha`, `--concentration`, `--candidate-alphas`, or config section `gamma.alpha`
