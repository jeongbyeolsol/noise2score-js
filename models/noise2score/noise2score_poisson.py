import torch

from models.noise2score.noise2score_base import Noise2Score, _with_noise_type


class PoissonNoise2Score(Noise2Score):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "poisson")
        super().__init__(*args, **kwargs)

    def denoise_from_score(self, y, score, noise_param, smoothing=0.0):
        smoothing = float(smoothing or 0.0)
        if smoothing > 0.0:
            x_hat = y + smoothing ** 2 * score
        else:
            peak = noise_param
            x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score / peak)

        if self.clamp:
            x_hat = x_hat.clamp(0, 1)
        return x_hat
