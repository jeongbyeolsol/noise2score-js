import torch
import torch.nn as nn
import torch.nn.functional as F

from data import add_gamma_noise, add_gaussian_noise, add_poisson_noise
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

    def _prepare_noise_param(self, input, noise_param, default):
        batch_size = input.size(0)

        if noise_param is None:
            noise_param = input.new_full((batch_size, 1), default)
        elif not torch.is_tensor(noise_param):
            noise_param = input.new_full((batch_size, 1), float(noise_param))
        else:
            noise_param = noise_param.to(device=input.device, dtype=input.dtype)
            if noise_param.ndim == 0:
                noise_param = noise_param.view(1, 1).expand(batch_size, 1)
            elif noise_param.ndim == 1:
                if noise_param.numel() == 1:
                    noise_param = noise_param.view(1, 1).expand(batch_size, 1)
                else:
                    noise_param = noise_param.view(batch_size, 1)
            elif noise_param.ndim == 2:
                if noise_param.shape == (1, 1):
                    noise_param = noise_param.expand(batch_size, 1)
                elif noise_param.shape != (batch_size, 1):
                    raise ValueError(
                        f"noise_param must have shape [], [1], [{batch_size}], [1, 1], "
                        f"or [{batch_size}, 1], got {tuple(noise_param.shape)}"
                    )
            else:
                raise ValueError(f"noise_param must be scalar, 1D, or 2D, got {noise_param.ndim}D")

        return noise_param

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
        input = input.view(-1, self.input_dim)
        noise_param = self._prepare_noise_param(input, noise_param, self.noise_param)

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
        input = input.view(-1, self.input_dim)
        noise_param = self._prepare_noise_param(input, noise_param, 0.0)

        # concat
        h = torch.cat([input, noise_param], dim=1)

        # predict
        glogprob = self.main(h)

        return glogprob
      
    