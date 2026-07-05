Noise2Score JS/Py playground
============================

Gaussian-smoothed ARDAE
-----------------------

`scripts/train_ardae.py` supports original-style Gaussian smoothing for ARDAE
training:

```bash
python scripts/train_ardae.py \
  --data datasets/train.npy \
  --input-dim 1600 \
  --noise-type poisson \
  --smoothing
```

When `--smoothing` is enabled, ARDAE adds Gaussian perturbations and trains with
the Gaussian denoising score matching target even if `--noise-type` is
`poisson` or `gamma`.

Use the configured sigma range:

```bash
--smoothing --sigma-min 0.001 --sigma-max 0.5
```

Or use a fixed sigma:

```bash
--smoothing 0.1
```
