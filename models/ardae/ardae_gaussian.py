import torch.nn.functional as F

from models.ardae.ardae_base import ARDAE, _with_noise_type
from utils import add_gaussian_noise


class GaussianARDAE(ARDAE):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "gaussian")
        super().__init__(*args, **kwargs)

    def add_noise(self, input, noise_param=None):
        noise_param = self.noise_param if noise_param is None else noise_param
        return add_gaussian_noise(input, std=noise_param)

    def _loss_and_target_score(self, glogprob, input, x_bar, eps, noise_param):
        sigma = self._view_param_like(noise_param, glogprob)
        safe_sigma = self._view_param_like(noise_param.clamp_min(1e-6), glogprob)
        target_score = -eps / safe_sigma
        loss = F.mse_loss(sigma * glogprob, -eps)
        return loss, target_score
