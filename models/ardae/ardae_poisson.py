import torch
import torch.nn.functional as F

from models.ardae.ardae_base import ARDAE, _with_noise_type
from utils import add_poisson_noise


class PoissonARDAE(ARDAE):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "poisson")
        super().__init__(*args, **kwargs)

    def add_noise(self, input, noise_param=None):
        noise_param = self.noise_param if noise_param is None else noise_param
        return add_poisson_noise(input, peak=noise_param)

    def _loss_and_target_score(self, glogprob, input, x_bar, eps, noise_param):
        peak = self._view_param_like(noise_param.clamp_min(1e-6), input)
        tiny = 1e-6

        x_safe = input.clamp_min(tiny)
        y_safe = x_bar.clamp_min(0.0)

        count = peak * y_safe
        rate = peak * x_safe

        target_score = peak * (
            torch.log(rate.clamp_min(tiny)) -
            torch.digamma(count + 1.0)
        )
        target_score = target_score.clamp(-100.0, 100.0)

        return F.mse_loss(glogprob, target_score), target_score
