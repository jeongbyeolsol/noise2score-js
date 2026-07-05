# Revised Changes

## 1. ARDAE Gaussian Smoothing

The main correction is in ARDAE training.

Added `--smoothing` to `scripts/train_ardae.py`.

Usage:

```bash
python scripts/train_ardae.py \
  --data datasets/train.npy \
  --input-dim 1600 \
  --noise-type poisson \
  --smoothing
```

With `--smoothing` enabled, ARDAE trains in the original Noise2Score-style
Gaussian smoothing mode:

```text
x_bar = x + sigma * eps
loss = MSE(sigma * score(x_bar, sigma), -eps)
```

This is used even when `--noise-type` is `poisson` or `gamma`.

## 2. Smoothing Sigma Modes

`--smoothing` can be used in two ways.

Use the existing sigma range:

```bash
--smoothing --sigma-min 0.001 --sigma-max 0.5
```

Use a fixed smoothing sigma:

```bash
--smoothing 0.1
```

Internally:

- `--smoothing` without a value samples sigma from `--sigma-min` to `--sigma-max`.
- `--smoothing <value>` fixes `sigma_min = sigma_max = value`.

## 3. ARDAE Model Changes

Updated `models/ardae.py`:

- Added `use_gaussian_smoothing`.
- When enabled, `ARDAE.add_noise(...)` always adds Gaussian noise.
- When enabled, `ARDAE.loss(...)` uses the Gaussian denoising score matching target.
- The original Poisson/Gamma analytic score losses remain available when smoothing is disabled.

This preserves the previous behavior by default.

## 4. Checkpoint Compatibility

Updated checkpoint loading in:

- `scripts/run_noise2score.py`
- `scripts/run_noise2score_blind.py`

The scripts now restore:

```text
use_gaussian_smoothing
```

from checkpoint args when available.

## 5. Config and Logging Improvements

Training config now records:

- `use_gaussian_smoothing`
- `smoothing`
- `smoothing_sigma`
- `best_checkpoint_path`
- `last_checkpoint_path`

Dataset config files copied during training now use a `data_` prefix to avoid
name collisions.

## 6. Config Copy Collision Handling

Updated `utils/file.py`:

- Added `make_unique_file_path(...)`.
- Added `suffix` support to `config_all_from_to(...)`.
- Added `on_conflict="suffix"` behavior.

If a copied file already exists, names like this can be created automatically:

```text
config.json
config_001.json
config_002.json
```

## 7. Additional Optional Noise2Score Path

I also added optional Gaussian-smoothed score averaging to the Noise2Score
evaluation path:

- `scripts/run_noise2score.py`
- `scripts/run_noise2score_blind.py`
- `models/noise2score.py`

This is disabled by default, so existing evaluation behavior is unchanged unless
`--smoothing` is explicitly passed to the evaluation scripts.

## 8. Verification

Checked with the `noise2score` conda environment:

```bash
conda run -n noise2score python -m py_compile \
  config/config.py \
  models/ardae.py \
  models/noise2score.py \
  models/noise2score_blind.py \
  scripts/train_ardae.py \
  scripts/run_noise2score.py \
  scripts/run_noise2score_blind.py \
  utils/file.py
```

Also ran small dummy API smoke tests for:

- ARDAE Gaussian smoothing loss
- Noise2Score smoothing compatibility
- Blind denoise compatibility
- Config copy collision handling

