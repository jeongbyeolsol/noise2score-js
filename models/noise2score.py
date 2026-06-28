import torch
import torch.nn as nn

class Noise2Score(nn.Module):
    def __init__(self, ardae, noise_type="gaussian", noise_param=0.1):
        super().__init__()
        self.ardae = ardae
        self.noise_type = noise_type
        self.noise_param = noise_param

    @torch.no_grad()
    def denoise(self, y):
        score = self.ardae.glogprob(y)

        if self.noise_type == "gaussian":
            sigma = self.noise_param
            x_hat = y + sigma ** 2 * score

        elif self.noise_type == "poisson":
            peak = self.noise_param
            x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score)

        elif self.noise_type == "gamma":
            alpha = self.noise_param
            denom = (alpha - 1.0) - y * score
            denom = denom.clamp_min(1e-6)
            x_hat = alpha * y / denom

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

        return x_hat.clamp(0, 1)