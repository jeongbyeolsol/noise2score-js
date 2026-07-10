import torch.nn.functional as F

from models.ardae.ardae_base import ARDAE, _with_noise_type
from utils import add_gamma_noise


class GammaARDAE(ARDAE):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "gamma")
        super().__init__(*args, **kwargs)

    def add_noise(self, input, noise_param=None):
        noise_param = self.noise_param if noise_param is None else noise_param
        return add_gamma_noise(input, concentration=noise_param)

    def _loss_and_target_score(self, glogprob, input, x_bar, eps, noise_param):
        alpha = self._view_param_like(noise_param.clamp_min(1e-6), input)
        tiny = 1.0 / 255.0

        x_safe = input.clamp_min(tiny)
        y_safe = x_bar.clamp_min(tiny)

        target_score = (alpha - 1.0) / y_safe - alpha / x_safe
        target_score = target_score.clamp(-100.0, 100.0)

        return F.mse_loss(glogprob, target_score), target_score
