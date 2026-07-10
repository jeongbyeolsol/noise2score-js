# Noise2Score JS

ARDAE score model을 학습하고, Noise2Score로 이미지/패치 단위 denoising을 수행하는 실험 코드입니다.

현재 구조는 다음 흐름을 기본으로 합니다.

```text
models/ardae/*              ARDAE 모델, 학습, 테스트
models/noise2score/*        Noise2Score 모델
scripts/run_noise2score_*   분포별 실행 entrypoint
config/*.json               실험 config
exe/*                       자주 쓰는 launcher
```

## 빠른 실행

Poisson DIV2K stitch 실험:

```bash
./scripts/run_noise2score_poisson.py \
  --config config/noise2score_poisson_div2k_stitch.json
```

Poisson blind stitch 실험:

```bash
./scripts/run_noise2score_blind_poisson.py \
  --config config/noise2score_blind_poisson_div2k_stitch.json
```

Launcher로 더 짧게 실행할 수도 있습니다.

```bash
./exe/noise2score_exe/config/run_poisson_div2k_stitch.sh
./exe/noise2score_exe/config/run_blind_poisson_div2k_stitch.sh
```

Config 값은 CLI로 덮어쓸 수 있습니다.

```bash
./scripts/run_noise2score_blind_poisson.py \
  --config config/noise2score_blind_poisson_div2k_stitch.json \
  --save-output-limit 8 \
  --output-dir results/debug_blind_poisson
```

## Entry Points

분포별 실행 파일을 우선 사용합니다.

```text
scripts/run_noise2score_gaussian.py
scripts/run_noise2score_poisson.py
scripts/run_noise2score_gamma.py

scripts/run_noise2score_blind_gaussian.py
scripts/run_noise2score_blind_poisson.py
scripts/run_noise2score_blind_gamma.py
```

호환용 공통 entrypoint도 남아 있습니다.

```text
scripts/run_noise2score.py
scripts/run_noise2score_blind.py
```

분포별 entrypoint는 더 직관적인 인자 이름을 지원합니다.

```bash
# Gaussian
python ./scripts/run_noise2score_gaussian.py --sigma 0.1 ...

# Poisson
python ./scripts/run_noise2score_poisson.py --peak 50 ...
python ./scripts/run_noise2score_poisson.py --lam 0.02 ...

# Gamma
python ./scripts/run_noise2score_gamma.py --alpha 20 ...
```

Blind 후보도 분포별 이름을 쓸 수 있습니다.

```bash
python ./scripts/run_noise2score_blind_gaussian.py --candidate-sigmas 0.05,0.1,0.2 ...
python ./scripts/run_noise2score_blind_poisson.py --candidate-peaks 30,50,80 ...
python ./scripts/run_noise2score_blind_poisson.py --candidate-lams 0.033,0.02,0.0125 ...
python ./scripts/run_noise2score_blind_gamma.py --candidate-alphas 10,20,50 ...
```

## Config

`--config`는 JSON을 기본 지원합니다. PyYAML이 설치되어 있으면 YAML도 사용할 수 있습니다.

Config 값은 argparse 기본값으로 들어가며, CLI 인자가 항상 config보다 우선합니다.

예시:

```json
{
  "checkpoint": {
    "checkpoint": "checkpoints/ardae_unet/poisson_lam001_005_smoothing_001/best_model.pt"
  },
  "data": {
    "clean_data": "datasets/DIV2K_valid_HR_processed",
    "data_mode": "image-folder"
  },
  "runtime": {
    "input_dim": 49152,
    "batch_size": 128,
    "device": "cuda"
  },
  "image": {
    "image_shape": [3, 128, 128],
    "patch_size": 128,
    "stride": 64,
    "channels": 3
  },
  "noise": {
    "noise_type": "poisson",
    "score_smoothing": 0.1
  },
  "poisson": {
    "lam_min": 0.01,
    "lam_max": 0.05,
    "candidate_peaks": [20, 50, 100]
  },
  "blind": {
    "candidate_score_smoothing": [0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2],
    "score_sigma_mode": "same"
  },
  "output": {
    "output_dir": "results/poisson_blind",
    "save_output": true,
    "stitch_output": true,
    "stitch_format": "png"
  }
}
```

지원되는 대표 section:

```text
runtime     input_dim, batch_size, device, seed, num_workers
data        clean_data, noisy_data, data_mode, key, noisy_key
image       image_shape, patch_size, stride, channels, recursive_images
noise       noise_type, noise_param, score_sigma, score_smoothing, score_smoothing_samples
gaussian    sigma, candidate_sigmas
poisson     peak, lam, peak_min/max, lam_min/max, candidate_peaks/lams
gamma       alpha, concentration, candidate_alphas
blind       candidate_score_smoothing, score_sigma_mode, tv_weight, data_weight
ardae       train/test 및 ARDAE 학습 옵션, gaussian_perturbation
output      output_dir, save_output, stitch_output, copy_info
checkpoint  checkpoint, noise2score_checkpoint_output
```

자세한 config 메모는 [config/README.md](config/README.md)를 봅니다.

## Checkpoint

기존에는 ARDAE를 따로 학습하고 Noise2Score가 그 checkpoint를 읽는 구조였습니다.

현재 실행 스크립트는 기존 ARDAE checkpoint도 읽지만, 실행 시 통합 Noise2Score checkpoint를 함께 저장합니다.

```text
output_dir/noise2score_checkpoint.pt
output_dir/noise2score_blind_checkpoint.pt
```

통합 checkpoint에는 다음 정보가 들어갑니다.

```text
checkpoint_type
noise2score 설정
ardae_checkpoint
ardae_checkpoint_path
실행 args
```

따라서 이후에는 ARDAE-only checkpoint 대신 Noise2Score checkpoint를 바로 넘길 수 있습니다.

```bash
python ./scripts/run_noise2score_poisson.py \
  --checkpoint results/run/noise2score_checkpoint.pt \
  --clean-data datasets/DIV2K_valid_HR_processed \
  --input-dim 49152 \
  --data-mode image-folder \
  --image-shape 3 128 128
```

## Poisson Parameter

Poisson은 `peak`를 기본 파라미터로 사용합니다.

```text
y = Poisson(x * peak) / peak
noise_param = peak
```

과제식이 다음 형태라면:

```text
y = Poisson(x / lam) * lam
```

관계는 다음과 같습니다.

```text
peak = 1 / lam
lam 0.01 -> peak 100
lam 0.02 -> peak 50
lam 0.05 -> peak 20
```

`peak`가 클수록 노이즈가 약하고, `lam`이 클수록 노이즈가 강합니다.

분포별 Poisson entrypoint에서는 둘 다 받을 수 있습니다.

```bash
--peak 50
--lam 0.02
```

Synthetic evaluation에서 이미지/샘플마다 다른 Poisson noise를 넣고 싶으면 range를 사용합니다.

```bash
--peak-min 20 \
--peak-max 100
```

또는 과제식의 `lam` 범위로 줄 수 있습니다.

```bash
--lam-min 0.01 \
--lam-max 0.05
```

내부적으로는 `peak = 1 / lam`으로 변환되어 `peak 20~100` 범위에서 샘플링됩니다. 기본 샘플링은 ARDAE 학습에서 쓰는 `make_noise_param`을 재사용하므로 log-scale입니다. 선형 Uniform 샘플링을 원하면 다음 옵션을 추가합니다.

```bash
--linear-noise-param
```

## Score Sigma와 Score Smoothing

`score_sigma`는 ARDAE에 score를 물어볼 때 쓰는 query noise level입니다.

Gaussian에서는 보통 observation sigma와 같습니다.

```bash
--sigma 0.1 \
--score-sigma 0.1
```

Plain non-smoothed Poisson/Gamma에서는 `--score-sigma`를 넘기지 않는 것이 안전합니다.

```bash
python ./scripts/run_noise2score_poisson.py \
  --checkpoint checkpoints/ardae_unet/poisson/best_model.pt \
  --clean-data datasets/test.npy \
  --input-dim 16384 \
  --peak 50
```

Non-Gaussian Noise2Score에서 `score_smoothing`은 score 주변을 Gaussian perturbation으로 Monte Carlo 평균내는 scale입니다.

```text
x_hat = y + score_smoothing^2 * score
```

이 경우 `score_sigma`는 보통 `score_smoothing`과 같거나 가까운 값을 씁니다.

```bash
--peak 50 \
--score-smoothing 0.1 \
--score-sigma 0.1
```

Blind score-smoothing sweep 예시:

```bash
python ./scripts/run_noise2score_blind_poisson.py \
  --checkpoint checkpoints/ardae_unet/poisson_lam001_005_smoothing_001/best_model.pt \
  --clean-data datasets/DIV2K_valid_HR_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 64 \
  --channels 3 \
  --peak 50 \
  --candidate-peaks 50 \
  --candidate-score-smoothing 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --stitch-output \
  --stitch-format png
```

Score-smoothed Poisson에서는 `candidate_score_smoothing`이 실제 denoising rule을 많이 좌우합니다.

용어 주의:

```text
ARDAE gaussian_perturbation
  학습 때 clean image에 Gaussian noise를 더한다.
  이미지 blur/kernel smoothing이 아니다.

Noise2Score score_smoothing
  평가 때 y 주변에 Gaussian perturbation을 여러 번 넣어 score를 평균낸다.
```

예전 옵션 이름인 `--smoothing`, `--candidate-smoothing`, `--ardae-smoothing`은 호환 alias로 남아 있지만 새 실험에서는 `--score-smoothing`, `--candidate-score-smoothing`, `--ardae-gaussian-perturbation`을 권장합니다.

## ARDAE Training

ARDAE 학습 코드는 `models/ardae` 아래에 있습니다.

```text
models/ardae/train_ardae.py
models/ardae/test_ardae.py
```

Noise2Score 실행 안에서 ARDAE를 같이 학습하려면 config나 CLI에서 `train_ardae`를 켭니다.

```json
{
  "ardae": {
    "train": true,
    "train_data": "datasets/DIV2K_train_HR_processed",
    "train_data_mode": "image-folder",
    "save_dir": "checkpoints/ardae_unet/div2k_poisson",
    "epochs": 64,
    "backbone": "unet",
    "base_channels": 32,
    "channel_mults": "1,2,4",
    "nonlinearity": "silu",
    "gaussian_perturbation": "range",
    "sigma_min": 0.03,
    "sigma_max": 0.22
  }
}
```

CLI로 직접 실행할 수도 있습니다.

```bash
python ./models/ardae/train_ardae.py \
  --data datasets/DIV2K_train_HR_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --backbone unet \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 128 \
  --channels 3 \
  --noise-type poisson \
  --poisson-peak 50 \
  --gaussian-perturbation \
  --sigma-min 0.03 \
  --sigma-max 0.22
```

`--gaussian-perturbation`으로 학습하는 ARDAE에서는 `sigma_min/sigma_max`가 Poisson peak가 아니라 Gaussian perturbation sigma 범위입니다.

## Data Modes

### Array mode

`--data-mode array`는 하나의 array 파일을 dataset으로 읽습니다.

지원 확장자:

```text
.npy, .npz, .pt, .pth, .csv, .txt, .tsv
```

### Image-folder mode

`--data-mode image-folder`는 이미지 또는 per-image `.npy` 파일을 patch 단위로 읽습니다.

지원:

```text
PNG/JPEG
.npy
```

`.npy` image shape:

```text
RGB:       [H, W, 3] or [3, H, W]
Grayscale: [H, W], [H, W, 1], or [1, H, W]
```

대표 옵션:

```bash
--data-mode image-folder \
--image-shape 3 128 128 \
--patch-size 128 \
--stride 64 \
--channels 3 \
--recursive-images
```

## Patch Stitching

`image-folder` 평가에서 patch denoising 결과를 원본 이미지 크기로 이어붙일 수 있습니다.

```bash
--stitch-output \
--stitch-format png
```

지원 format:

```text
npy
png
both
```

결과는 다음 위치에 저장됩니다.

```text
output_dir/stitched/clean
output_dir/stitched/noisy
output_dir/stitched/denoised
output_dir/stitched/score
output_dir/stitched/summary.json
```

`--save-output-limit`가 주어지고 `--save-output`이 켜져 있으면 해당 개수만 복원한 뒤 평가를 멈춥니다.

## Noisy-only Input

clean image가 없고 noisy image만 있을 때는 `--noisy-data`를 사용합니다.

```bash
python ./scripts/run_noise2score_blind_poisson.py \
  --checkpoint checkpoints/ardae_unet/poisson_lam001_005_smoothing_001/best_model.pt \
  --noisy-data datasets/CBSD100_noisy_processed \
  --data-mode image-folder \
  --input-dim 49152 \
  --image-shape 3 128 128 \
  --patch-size 128 \
  --stride 64 \
  --channels 3 \
  --peak 50 \
  --candidate-peaks 50 \
  --candidate-score-smoothing 0.03,0.05,0.075,0.1,0.125,0.15,0.2 \
  --score-sigma-mode same \
  --stitch-output \
  --stitch-format png
```

이 경우 metric 계산에 필요한 clean GT가 없으므로 `noisy_mse`, `denoised_mse`, PSNR 등은 `null`로 기록됩니다.

## Outputs

실행 결과 디렉토리에는 보통 다음 파일이 생깁니다.

```text
run_config.json
summary.json
candidate_history.json          blind 실행 시
noise2score_checkpoint.pt       기본 실행 시
noise2score_blind_checkpoint.pt blind 실행 시
output/*.npy                    --save-output 사용 시
stitched/*                      --stitch-output 사용 시
```

`--copy-info`를 켜면 checkpoint/data 쪽 config와 metrics 파일도 결과 디렉토리로 복사합니다.
