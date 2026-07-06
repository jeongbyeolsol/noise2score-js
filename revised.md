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

## 9. Lazy Image Patch Loading

Added a memory-safe image patch dataset path for large image collections.

New training options:

```text
--data-mode image-folder
--patch-size
--stride
--channels
--max-patches-per-image
--recursive-images
```

Instead of precomputing every patch and storing all patches in one giant list or
`.npy` file, `ImagePatchDataset` stores only patch coordinates. During training,
`__getitem__` opens the needed image and slices one patch on demand.

Example:

```bash
python ./scripts/train_ardae.py \
  --data datasets/DIV2K_train_HR \
  --data-mode image-folder \
  --input-dim 49152 \
  --backbone unet \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 64 \
  --channels 3 \
  --max-patches-per-image 512
```

Updated `exe/ardae_unet/ardae_unet_train_poisson.sh` to use this lazy image
folder path.

## 10. Faster Streaming Patch Loader

Added a faster image-folder patch loader for training speed.

New options:

```text
--patch-loader stream
--persistent-workers
--prefetch-factor
```

`--patch-loader map` keeps the older lazy random-access behavior, where each
patch may reopen/decode an image. This is memory safe but slow.

`--patch-loader stream` opens one image, yields many patches from that decoded
image, then moves to the next image. This greatly reduces image decoding
overhead for large PNG/JPEG folders.

Recommended for image-folder training:

```bash
--patch-loader stream \
--num-workers 8 \
--persistent-workers \
--prefetch-factor 4
```

`exe/ardae_unet/ardae_unet_train_poisson.sh` now uses the streaming loader by
default.

## 11. NPY Image Folder Loading

Fixed image-folder training so folders of per-image `.npy` files are discovered.

Example supported layout:

```text
datasets/DIV2K_train_HR_processed/
  0001.npy
  0002.npy
  ...
```

Expected `.npy` shapes:

```text
RGB:       [H, W, 3] or [3, H, W]
Grayscale: [H, W], [H, W, 1], or [1, H, W]
```

Use `.npy` folders with:

```bash
--data-mode image-folder \
--patch-loader stream
```

The map loader is intentionally blocked for `.npy` folders because it is the
PIL image random-access path.

## 12. Noise2Score NPY Evaluation

Updated Noise2Score evaluation scripts to read per-image `.npy` folders too:

```text
scripts/run_noise2score.py
scripts/run_noise2score_blind.py
```

New evaluation options:

```bash
--data-mode image-folder \
--patch-size 128 \
--stride 128 \
--channels 3 \
--max-patches-per-image 128 \
--recursive-images
```

The evaluation loader uses `StreamingImagePatchDataset`, so it can read:

```text
datasets/DIV2K_valid_HR_processed/
  0801.npy
  0802.npy
  ...
```

without concatenating everything into one giant array.

Also fixed evaluation batch handling so both `TensorDataset` batches and direct
streamed tensor batches work.
