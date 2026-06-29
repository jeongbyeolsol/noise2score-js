# 체크포인트 설명

### 000

뒤에 숫자 없는 경우 -> best epoch만 저장, 또한 초기 버전

사용 권장 x


### 001~003

best달성 시 모두 저장함, **100 epoch으로 학습**

개선반영이 이루어지지 않음

사용 권장 x

### 004

**300 epoch으로 학습**

그러나 ardae가 고정된 denoising residual에 대해 예측

### 005

**1000 epoch으로 학습**

log scale로 생성된 random denoising residual에 대해 예측

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

    return noise_para
```

```python
noise_param = None
if config.sigma_min is not None and config.sigma_max is not None:
    noise_param = make_noise(x, sigma_min=config.sigma_min, sigma_max=config.sigma_max)
```