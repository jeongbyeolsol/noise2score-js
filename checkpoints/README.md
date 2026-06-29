# 체크포인트 설명

### 000

뒤에 숫자 없는 경우 -> best epoch만 저장, 또한 초기 버전

사용 권장 x


### 001~003

best달성 시 모두 저장함, **100 epoch으로 학습**

개선반영이 이루어지지 않음

사용 권장 x

### 004

**300 epoch 학습**

초기 버전에서는 ARDAE를 고정된 noise parameter에서 학습했다.
즉, 하나의 고정된 corruption scale에 대해 noisy input을 생성하고, 해당 조건에서 score 또는 denoising direction을 예측하도록 학습했다.

이 실험은 ARDAE 학습 파이프라인이 정상적으로 동작하는지 확인하기 위한 baseline 성격의 sanity check로 사용했다.

---

### 005

**1000 epoch 학습**

이후 버전에서는 noise parameter를 고정하지 않고, 각 batch sample마다 random하게 noise scale을 생성하도록 수정했다.
특히 Gaussian noise의 경우 `sigma_min=0.001`, `sigma_max=0.5` 범위에서 log-scale uniform sampling을 적용했다.

```python
def make_noise(x, sigma_min=0.001, sigma_max=0.5, use_log_scale=True):
    if use_log_scale:
        log_sigma_min = torch.log(torch.tensor(sigma_min, device=x.device, dtype=x.dtype))
        log_sigma_max = torch.log(torch.tensor(sigma_max, device=x.device, dtype=x.dtype))

        log_sigma = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype).uniform_(
            log_sigma_min,
            log_sigma_max,
        )
        noise_param = log_sigma.exp()
    else:
        noise_param = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype).uniform_(
            sigma_min,
            sigma_max,
        )

    return noise_param
```

이렇게 하면 ARDAE가 단일 noise level에만 대응하는 것이 아니라, 여러 noise scale에서의 score field를 학습하게 된다.
따라서 이후 Noise2Score에서 예측되는 noise distribution과 ARDAE가 학습한 noise distribution 사이의 관계를 분석하는 데 더 적합한 설정이라고 보았다.
