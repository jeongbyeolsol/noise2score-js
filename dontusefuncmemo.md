# 내가 새로 만든거

```python
def add_poisson_noise(input, peak=30.0, clamp=True):
    """
    Poisson noise.

    input: [0, 1] 범위의 이미지 텐서라고 가정
    peak: photon count scale. 작을수록 노이즈가 강함.
          예: 10 강함, 30 보통, 100 약함
    clamp: 결과를 [0, 1]로 자를지 여부

    return:
        noisy: 포아송 노이즈가 들어간 이미지
        eps: noisy - input, 실제로 더해진 residual noise
    """

    # Poisson rate는 음수일 수 없으므로 최소 0으로 제한
    rate = input.clamp_min(0) * peak

    # photon count 샘플링
    noisy_counts = torch.poisson(rate)

    # 다시 [0, 1] scale로 복원
    noisy = noisy_counts / peak

    if clamp:
        noisy = noisy.clamp(0, 1)

    eps = noisy - input
    return noisy, eps
```


```python
def add_gamma_noise(input, concentration=2.0, clamp=True):
    """
    Gamma multiplicative noise.

    input: [0, 1] 범위의 이미지 텐서라고 가정
    concentration: 감마분포 shape parameter.
                   작을수록 노이즈가 강함.
                   예: 0.5 매우 강함, 2 보통, 10 약함

    감마 노이즈는 평균이 1이 되도록 rate=concentration으로 둠.
    즉 gamma_noise ~ Gamma(alpha, alpha)
    평균 alpha / alpha = 1

    return:
        noisy: 감마 곱셈 노이즈가 들어간 이미지
        eps: noisy - input, 실제 residual noise
    """

    alpha = torch.full_like(input, concentration)
    rate = torch.full_like(input, concentration)

    gamma_noise = torch.distributions.Gamma(alpha, rate).sample()

    noisy = input * gamma_noise

    if clamp:
        noisy = noisy.clamp(0, 1)

    eps = noisy - input
    return noisy, eps
```

````python
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import add_gamma_noise, add_gaussian_noise, add_poisson_noise
from models.layers import MLP


class ARDAE(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=1000,
                 noise_param=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 ):
        super().__init__()
        
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.noise_param = noise_param
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type

        self.main = MLP(input_dim+1, h_dim, input_dim, use_nonlinearity_output=False, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity)

    def add_noise(self, input, noise_param=None):
        noise_param = self.noise_param if noise_param is None else noise_param

        if self.noise_type == "gaussian":
            return add_gaussian_noise(input, std=noise_param)

        elif self.noise_type == "poisson":
            return add_poisson_noise(input, peak=noise_param)

        elif self.noise_type == "gamma":
            return add_gamma_noise(input, concentration=noise_param)

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")
        

    def loss(self, glogprob, input, x_bar, eps, noise_param):
        if self.noise_type == "gaussian":
            target = -eps
            pred = noise_param * glogprob
            return F.mse_loss(pred, target)

        elif self.noise_type == "poisson":
            # Poisson은 discrete라서 엄밀한 score target을 잡기 어렵다.
            # 일단 heuristic residual target으로 둔다.
            target = -eps
            pred = glogprob
            return F.mse_loss(pred, target)

        elif self.noise_type == "gamma":
            # Gamma multiplicative noise의 conditional score target을 쓰는 버전
            # x_bar = input * gamma_noise
            # gamma_noise ~ Gamma(alpha, alpha)
            alpha = noise_param

            tiny = torch.finfo(input.dtype).eps
            input_safe = input.clamp_min(tiny)
            x_bar_safe = x_bar.clamp_min(tiny)

            target_score = (alpha - 1.0) / x_bar_safe - alpha / input_safe

            return F.mse_loss(glogprob, target_score)

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

    def forward(self, input, noise_param=None):
        # init
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        
        if noise_param is None:
            noise_param = input.new_full((batch_size, 1), self.noise_param)
        elif not torch.is_tensor(noise_param):
            noise_param = input.new_full((batch_size, 1), float(noise_param))
        else:
            noise_param = noise_param.to(device=input.device, dtype=input.dtype)
            if noise_param.ndim == 0:
                noise_param = noise_param.view(1, 1).expand(batch_size, 1)
            elif noise_param.ndim == 1:
                noise_param = noise_param.view(batch_size, 1)

        # add noise
        x_bar, eps = self.add_noise(input, noise_param)

        # concat
        h = torch.cat([x_bar, noise_param], dim=1)

        # predict
        glogprob = self.main(h)

        ''' get loss '''
        loss = self.loss(
            glogprob=glogprob,
            input=input,
            x_bar=x_bar,
            eps=eps,
            noise_param=noise_param,
        )

        # return
        return glogprob, loss

    def glogprob(self, input, noise_param=None):
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        if noise_param is None:
            noise_param = input.new_zeros(batch_size, 1)
        else:
            assert torch.is_tensor(noise_param)

        # concat
        h = torch.cat([input, noise_param], dim=1)

        # predict
        glogprob = self.main(h)

        return glogprob
      
    
```



---

원래 있던거


def add_gaussian_noise(input, std):
    eps = torch.randn_like(input)
    return input + std*eps, eps


def add_laplace_noise(input, scale):
    eps = sample_unit_laplace_noise(shape=input.size(), dtype=input.dtype, device=input.device)
    return input + scale*eps, eps


def add_uniform_noise(input, val):
    #raise NotImplementedError
    #eps = 2.*val*torch.rand_like(input) - val
    eps = torch.rand_like(input)
    return input + 2.*val*eps-val, eps