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
                 use_metric = False
                 ):
        super().__init__()
        
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.noise_param = noise_param
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        self.use_metric = use_metric
        
        self.last_metrics = {}

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
    
    def _compute_score_metrics(self, pred_score, target_score):
        """
        pred_score: model output score, glogprob
        target_score: score target
        """

        with torch.no_grad():
            pred = pred_score.detach()
            target = target_score.detach()

            # 혹시 inf/nan이 섞였을 때 로그가 터지는 걸 방지
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

            mse = ((pred - target) ** 2).mean()
            target_energy = (target ** 2).mean()
            pred_energy = (pred ** 2).mean()

            nmse = mse / (target_energy + 1e-8)

            pred_flat = pred.flatten(start_dim=1)
            target_flat = target.flatten(start_dim=1)

            cos = F.cosine_similarity(
                pred_flat,
                target_flat,
                dim=1,
                eps=1e-8,
            ).mean()

            # Pearson correlation
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()

            corr = (pred_centered * target_centered).mean() / (
                pred_centered.std() * target_centered.std() + 1e-8
            )

        return {
            "score_mse": mse.item(),
            "score_nmse": nmse.item(),
            "score_cos": cos.item(),
            "score_corr": corr.item(),
            "target_energy": target_energy.item(),
            "pred_energy": pred_energy.item(),
        }

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
            peak = noise_param.clamp_min(1e-6)

            tiny = 1e-6
            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(0.0)

            count = peak * y_safe
            rate = peak * x_safe

            target_score = peak * (
                torch.log(rate.clamp_min(tiny)) -
                torch.digamma(count + 1.0)
            )

            # 너무 큰 score target이 학습을 망치지 않도록 완만하게 제한
            target_score = target_score.clamp(-100.0, 100.0)

            return F.mse_loss(glogprob, target_score)

        elif self.noise_type == "gamma":
            # Gamma multiplicative noise의 conditional score target을 쓰는 버전
            # x_bar = input * gamma_noise
            # gamma_noise ~ Gamma(alpha, alpha)
            alpha = noise_param.clamp_min(1e-6)

            # torch.finfo(...).eps는 너무 작아서 이미지 score에서는 폭발 방지로 부족함.
            # 1/255 근처를 lower bound로 두는 편이 더 안전함.
            tiny = 1.0 / 255.0

            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(tiny)

            target_score = (alpha - 1.0) / y_safe - alpha / x_safe

            # Gamma score는 어두운 픽셀에서 쉽게 폭발하므로 clamp 권장
            target_score = target_score.clamp(-100.0, 100.0)

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
        if self.use_metric:
            loss = self.loss_with_metric(
                glogprob=glogprob,
                input=input,
                x_bar=x_bar,
                eps=eps,
                noise_param=noise_param,
            )
        else:
            loss = self.loss(
                glogprob=glogprob,
                input=input,
                x_bar=x_bar,
                eps=eps,
                noise_param=noise_param,
            )

        # return
        return glogprob, loss

    def loss_with_metric(self, glogprob, input, x_bar, eps, noise_param):
        if self.noise_type == "gaussian":
            sigma = noise_param.clamp_min(1e-6)

            # Gaussian conditional score target
            # x_bar = input + sigma * eps
            # score target = -eps / sigma
            target_score = -eps / sigma

            # 기존 AR-DAE와 동치인 loss
            loss = F.mse_loss(sigma * glogprob, -eps)

            self.last_metrics = self._compute_score_metrics(
                pred_score=glogprob,
                target_score=target_score,
            )

            return loss

        elif self.noise_type == "poisson":
            peak = noise_param.clamp_min(1e-6)

            tiny = 1e-6
            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(0.0)

            # count = peak * y
            count = peak * y_safe
            rate = peak * x_safe

            # Continuous relaxation of Poisson score wrt y
            target_score = peak * (
                torch.log(rate.clamp_min(tiny))
                - torch.digamma(count + 1.0)
            )

            target_score = target_score.clamp(-100.0, 100.0)

            loss = F.mse_loss(glogprob, target_score)

            self.last_metrics = self._compute_score_metrics(
                pred_score=glogprob,
                target_score=target_score,
            )

            return loss

        elif self.noise_type == "gamma":
            alpha = noise_param.clamp_min(1e-6)

            tiny = 1.0 / 255.0

            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(tiny)

            # Gamma multiplicative noise score target
            # y = x * g, g ~ Gamma(alpha, alpha)
            target_score = (alpha - 1.0) / y_safe - alpha / x_safe

            target_score = target_score.clamp(-100.0, 100.0)

            loss = F.mse_loss(glogprob, target_score)

            self.last_metrics = self._compute_score_metrics(
                pred_score=glogprob,
                target_score=target_score,
            )

            return loss

        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

    def glogprob(self, input, noise_param=None):
        input = input.view(-1, self.input_dim)
        noise_param = self._prepare_noise_param(input, noise_param, 0.0)

        # concat
        h = torch.cat([input, noise_param], dim=1)

        # predict
        glogprob = self.main(h)

        return glogprob
      
    