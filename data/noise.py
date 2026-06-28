import torch



def add_gaussian_noise(input, std):
    std = _view_param(std, input)
    eps = torch.randn_like(input)
    x_bar = input + std * eps
    return x_bar, eps


def add_poisson_noise(input, peak=30.0, clamp=True):
    peak = _view_param(peak, input)

    rate = input.clamp_min(0) * peak
    noisy_counts = torch.poisson(rate)
    x_bar = noisy_counts / peak

    if clamp:
        x_bar = x_bar.clamp(0, 1)

    eps = x_bar - input
    return x_bar, eps


def add_gamma_noise(input, concentration=2.0, clamp=True):
    concentration = _view_param(concentration, input)

    alpha = concentration.expand_as(input)
    rate = concentration.expand_as(input)

    gamma_noise = torch.distributions.Gamma(alpha, rate).sample()
    x_bar = input * gamma_noise

    if clamp:
        x_bar = x_bar.clamp(0, 1)

    eps = x_bar - input
    return x_bar, eps
  
  
def _view_param(param, input):
    """
    param을 input에 broadcast 가능한 shape으로 바꾼다.
    input: [B, D]
    param: scalar, [B], [B, 1] 가능
    """
    if not torch.is_tensor(param):
        param = input.new_tensor(float(param))

    param = param.to(device=input.device, dtype=input.dtype)

    if param.ndim == 0:
        return param

    if param.ndim == 1:
        return param.view(-1, 1)

    return param