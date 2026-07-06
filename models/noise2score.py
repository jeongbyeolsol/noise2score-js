# models/noise2score.py

from pathlib import Path
import torch
import torch.nn as nn


from models.ardae import ARDAE
from config import ARDAEConfig


class Noise2Score(nn.Module):
    def __init__(
        self,
        ardae,
        noise_type="gaussian",
        noise_param=0.1,
        score_sigma=None,
        clamp=True,
    ):
        super().__init__()
        self.ardae = ardae
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.score_sigma = noise_param if score_sigma is None else score_sigma
        self.clamp = clamp

    @torch.no_grad()
    def score(
        self,
        y,
        noise_param=None,
        score_sigma=None,
        smoothing=0.0,
        smoothing_samples=1,
    ):
        noise_param = self._resolve_score_noise_param(noise_param, score_sigma)
        smoothing = float(smoothing or 0.0)
        smoothing_samples = int(smoothing_samples)

        if smoothing <= 0.0:
            return self.ardae.glogprob(y, noise_param=noise_param)

        if smoothing_samples < 1:
            raise ValueError("smoothing_samples must be >= 1.")

        score_sum = torch.zeros_like(y)
        for _ in range(smoothing_samples):
            y_smooth = y + smoothing * torch.randn_like(y)
            score_sum = score_sum + self.ardae.glogprob(
                y_smooth,
                noise_param=noise_param,
            )

        return score_sum / float(smoothing_samples)

    @torch.no_grad()
    def denoise(
        self,
        y,
        noise_param=None,
        score_sigma=None,
        smoothing=0.0,
        smoothing_samples=1,
    ):
        denoise_noise_param = self.noise_param if noise_param is None else noise_param
        score_noise_param = self._resolve_score_noise_param(
            noise_param,
            score_sigma,
        )
        score = self.score(
            y,
            noise_param=score_noise_param,
            smoothing=smoothing,
            smoothing_samples=smoothing_samples,
        )
        smoothing = float(smoothing or 0.0)

        if smoothing > 0.0 and self.noise_type != "gaussian":
            x_hat = y + smoothing ** 2 * score

        elif self.noise_type == "gaussian":
            sigma = denoise_noise_param
            x_hat = y + sigma ** 2 * score

        elif self.noise_type == "poisson":
            peak = denoise_noise_param
            x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score / peak)

        elif self.noise_type == "gamma":
            alpha = denoise_noise_param
            denom = (alpha - 1.0) - y * score
            denom = denom.clamp_min(1e-6)
            x_hat = alpha * y / denom

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

        if self.clamp:
            x_hat = x_hat.clamp(0, 1)

        return x_hat

    def _resolve_score_noise_param(self, noise_param=None, score_sigma=None):
        if score_sigma is not None:
            return score_sigma
        if noise_param is not None:
            return noise_param
        return self.score_sigma
    
    def _assign_ardae(self, ardae, config: ARDAEConfig = None):
        if isinstance(ardae, ARDAE):
            self.ardae = ardae
        elif isinstance(ardae, (str, Path)):
            self.ardae = ARDAE()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.ardae.load_state_dict(torch.load(str(ardae), map_location=torch.device(device)))
            except:
                raise ValueError('can\'t load model!!\n')
        else:
            if config is None:
                raise ValueError("Config가 제공되지 않아 새로운 ARDAE 모델을 생성할 수 없습니다.")
        
            self.ardae = ARDAE(input_dim=config.input_dim,
                 h_dim=config.h_dim,
                 noise_param = config.noise_param,
                 noise_min = config.noise_min,
                 noise_max = config.noise_max,
                 num_hidden_layers = config.num_hidden_layers,
                 nonlinearity = config.nonlinearity,
                 noise_type = config.noise_type,
                 use_metric = config.use_metric,
                 use_gaussian_smoothing = config.use_gaussian_smoothing,
            )
