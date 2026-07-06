Noise2Score Notes
=================

This repo trains ARDAE score models and evaluates Noise2Score denoising.

Parameter Cheat Sheet
---------------------

### `noise_param`

`noise_param` is the parameter of the observation noise used to create synthetic
noisy images during evaluation.

For Gaussian:

```text
y = x + sigma * eps
noise_param = sigma
```

For Poisson, this code uses `peak`:

```text
y = Poisson(x * peak) / peak
noise_param = peak
```

If your task is written with `lam`:

```text
y = Poisson(x / lam) * lam
```

then:

```text
peak = 1 / lam
lam 0.01 -> peak 100
lam 0.05 -> peak 20
```

Larger `peak` means weaker noise. Larger `lam` means stronger noise.

### `score_sigma`

`score_sigma` is the ARDAE query noise level used when asking the trained ARDAE
for a score.

Use it only in these cases:

- Gaussian ARDAE / Gaussian evaluation
- Gaussian-smoothed ARDAE evaluation

Do not pass `--score-sigma` for plain non-smoothed Poisson/Gamma evaluation.
For a plain Poisson ARDAE, omit `--score-sigma`; the code will query ARDAE with
`--noise-param`.

Bad plain Poisson example:

```bash
--noise-type poisson \
--noise-param 50 \
--score-sigma 0.1
```

This asks a Poisson ARDAE for a score at parameter `0.1`, while the noisy image
was generated with `peak=50`. That is a severe mismatch.

Good plain Poisson example:

```bash
--noise-type poisson \
--noise-param 100
```

### `smoothing`

`smoothing` enables Gaussian-smoothed Noise2Score denoising.

When `--smoothing > 0`, non-Gaussian denoising uses:

```text
x_hat = y + smoothing^2 * score
```

In this mode, `--score-sigma` should usually match `--smoothing` or be close to
it:

```bash
--score-sigma 0.1 \
--smoothing 0.1
```

### `sigma_min` / `sigma_max`

These are training-time ARDAE query/noise-parameter ranges.

Their meaning depends on the training mode.

Plain Gaussian ARDAE:

```text
sigma_min / sigma_max = Gaussian noise sigma range
```

Plain non-smoothed Poisson ARDAE:

```text
sigma_min / sigma_max = Poisson peak range
```

So for a plain Poisson model intended for `peak=100`, use a fixed range:

```bash
--noise-type poisson \
--noise-param 100 \
--sigma-min 100 \
--sigma-max 100
```

For a plain Poisson model intended for `lam 0.01~0.05`, use:

```bash
--noise-type poisson \
--sigma-min 20 \
--sigma-max 100
```

Gaussian-smoothed Poisson ARDAE:

```text
sigma_min / sigma_max = Gaussian smoothing sigma range
```

Example:

```bash
--noise-type poisson \
--smoothing \
--sigma-min 0.03 \
--sigma-max 0.22
```

Gaussian-smoothed ARDAE Training
--------------------------------

`scripts/train_ardae.py` supports original-style Gaussian smoothing for ARDAE
training:

```bash
python scripts/train_ardae.py \
  --data datasets/DIV2K_train_HR_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --backbone unet \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 128 \
  --channels 3 \
  --noise-type poisson \
  --noise-param 100 \
  --smoothing \
  --sigma-min 0.03 \
  --sigma-max 0.22
```

With `--smoothing`, ARDAE adds Gaussian perturbations and trains with the
Gaussian denoising score matching target even if `--noise-type` is `poisson` or
`gamma`.

Smoothed ARDAE Evaluation
-------------------------

For a smoothed Poisson checkpoint, use `--score-sigma` and `--smoothing`
together:

```bash
python scripts/run_noise2score.py \
  --checkpoint checkpoints/ardae_unet/poisson_lam001_005_smoothing/best_model.pt \
  --clean-data datasets/DIV2K_valid_HR_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --batch-size 128 \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 128 \
  --channels 3 \
  --noise-type poisson \
  --noise-param 50 \
  --score-sigma 0.1 \
  --smoothing 0.1
```

Recommended sweep:

```text
score_sigma = smoothing = 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2
```

Blind Smoothed ARDAE Evaluation
-------------------------------

For a Gaussian-smoothed Poisson checkpoint, blind parameter search should sweep
the Gaussian smoothing sigma, not only the Poisson `peak`.

Recommended blind run:

```bash
python scripts/run_noise2score_blind.py \
  --checkpoint checkpoints/ardae_unet/poisson_lam001_005_smoothing/best_model.pt \
  --clean-data datasets/DIV2K_valid_HR_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --batch-size 128 \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 128 \
  --channels 3 \
  --noise-type poisson \
  --noise-param 50 \
  --candidate-params 50 \
  --candidate-smoothing 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --smoothing 0.1
```

With `--candidate-smoothing`, each candidate uses:

```text
score_sigma = smoothing_candidate
x_hat = y + smoothing_candidate^2 * score
```

The output summary records:

```text
estimated_noise_param = selected observation parameter/bookkeeping peak
estimated_smoothing   = selected Gaussian smoothing sigma
estimated_score_sigma = selected ARDAE query sigma
```

For smoothed Poisson ARDAE, `estimated_smoothing` is usually the important blind
selection result. If you only sweep `--candidate-params 20,30,50,80,100` while
keeping `--smoothing 0.1`, the denoising rule is effectively the same for every
candidate because smoothed non-Gaussian denoising uses the smoothing sigma.

Plain Poisson ARDAE Evaluation
------------------------------

For a non-smoothed Poisson checkpoint, do not use `--score-sigma` and do not use
`--smoothing`.

```bash
python scripts/run_noise2score.py \
  --checkpoint checkpoints/ardae_unet/poisson_001/best_model.pt \
  --clean-data datasets/processed_test/bsd68_patches_128x128.npy \
  --input-dim 16384 \
  --batch-size 128 \
  --image-shape 1 128 128 \
  --noise-type poisson \
  --noise-param 100
```

Data Modes
----------

### Array mode

Use `--data-mode array` or omit `--data-mode`.

Supported:

```text
.npy, .npz, .pt, .pth, .csv, .txt, .tsv
```

This mode loads the array as one dataset.

### Image-folder mode

Use:

```bash
--data-mode image-folder
```

Supported folder contents:

```text
PNG/JPEG images
per-image .npy files
```

Expected `.npy` shapes:

```text
RGB:       [H, W, 3] or [3, H, W]
Grayscale: [H, W], [H, W, 1], or [1, H, W]
```

For `.npy` image folders, use the streaming loader:

```bash
--patch-loader stream
```

Useful options:

```bash
--patch-size 128 \
--stride 128 \
--channels 3 \
--max-patches-per-image 128 \
--recursive-images
```
