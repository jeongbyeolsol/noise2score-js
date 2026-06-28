# models/noise2score.py

import torch
import torch.nn as nn


class Noise2Score(nn.Module):
    def __init__(
        self,
        ardae,
        noise_type="gaussian",
        noise_param=0.1,
        score_sigma=0.01,
        clamp=True,
    ):
        super().__init__()
        self.ardae = ardae
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.score_sigma = score_sigma
        self.clamp = clamp

    @torch.no_grad()
    def score(self, y, score_sigma=None):
        score_sigma = self.score_sigma if score_sigma is None else score_sigma
        return self.ardae.glogprob(y, noise_param=score_sigma)

    @torch.no_grad()
    def denoise(self, y, noise_param=None, score_sigma=None):
        noise_param = self.noise_param if noise_param is None else noise_param
        score = self.score(y, score_sigma=score_sigma)

        if self.noise_type == "gaussian":
            sigma = noise_param
            x_hat = y + sigma ** 2 * score

        elif self.noise_type == "poisson":
            peak = noise_param
            x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score / peak)

        elif self.noise_type == "gamma":
            alpha = noise_param
            denom = (alpha - 1.0) - y * score
            denom = denom.clamp_min(1e-6)
            x_hat = alpha * y / denom

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

        if self.clamp:
            x_hat = x_hat.clamp(0, 1)

        return x_hat