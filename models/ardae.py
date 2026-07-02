import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import add_gamma_noise, add_gaussian_noise, add_poisson_noise
from models.layers import MLP, UNet


class ARDAE(nn.Module):
    def __init__(
        self,
        input_dim=2,
        h_dim=1000,
        noise_param=0.1,
        noise_min=0.001,
        noise_max=0.5,
        num_hidden_layers=1,
        nonlinearity='tanh',
        noise_type='gaussian',
        use_metric=False,
        backbone='mlp',
        image_shape=None,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        use_norm=True,
    ):
        super().__init__()

        self.backbone = backbone.lower()
        if self.backbone not in {"mlp", "unet"}:
            raise ValueError(f"backbone must be 'mlp' or 'unet', got {backbone!r}")

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.noise_param = noise_param
        self.noise_min = noise_min
        self.noise_max = noise_max

        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        self.use_metric = use_metric

        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.use_norm = use_norm
        self.image_shape = self._normalize_image_shape(image_shape, input_dim)

        self.last_metrics = {}

        if self.backbone == "mlp":
            self.main = MLP(
                input_dim + 1,
                h_dim,
                input_dim,
                use_nonlinearity_output=False,
                num_hidden_layers=num_hidden_layers,
                nonlinearity=nonlinearity,
            )
        else:
            in_channels = self.image_shape[0]
            self.main = UNet(
                in_channels=in_channels,
                out_channels=in_channels,
                base_channels=base_channels,
                channel_mults=self.channel_mults,
                nonlinearity=nonlinearity,
                use_norm=use_norm,
                use_noise_level=True,
            )

    def _normalize_image_shape(self, image_shape, input_dim):
        if self.backbone != "unet":
            return None

        if image_shape is not None:
            image_shape = tuple(int(v) for v in image_shape)
            if len(image_shape) == 2:
                image_shape = (1, *image_shape)
            if len(image_shape) != 3:
                raise ValueError("image_shape must be [H, W] or [C, H, W]")
            self.input_dim = reduce(mul, image_shape, 1)
            return image_shape

        if input_dim is None:
            raise ValueError("UNet backbone needs image_shape or input_dim for square single-channel inference.")

        side = int(math.sqrt(int(input_dim)))
        if side * side != int(input_dim):
            raise ValueError(
                "image_shape was not given and input_dim is not a square number. "
                "Pass --image-shape C H W, e.g. --image-shape 1 40 40."
            )
        return (1, side, side)

    def _format_input(self, input):
        if self.backbone == "mlp":
            return input.view(-1, self.input_dim)

        c, h, w = self.image_shape

        if input.dim() == 2:
            if input.size(1) != c * h * w:
                raise ValueError(
                    f"Flat input has dim {input.size(1)}, but image_shape={self.image_shape} "
                    f"requires {c*h*w}."
                )
            return input.view(input.size(0), c, h, w)

        if input.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            if c != 1:
                raise ValueError(
                    f"3D input is interpreted as [B, H, W], but image_shape has C={c}. "
                    "Use [B, C, H, W] input instead."
                )
            if tuple(input.shape[-2:]) != (h, w):
                raise ValueError(f"Input spatial shape {tuple(input.shape[-2:])} != {(h, w)}")
            return input.unsqueeze(1)

        if input.dim() == 4:
            if tuple(input.shape[1:]) != self.image_shape:
                raise ValueError(f"Input shape {tuple(input.shape[1:])} != image_shape {self.image_shape}")
            return input

        raise ValueError(f"Unsupported input shape for UNet ARDAE: {tuple(input.shape)}")

    def _sample_noise(self, input):
        return torch.empty(input.size(0), 1, device=input.device, dtype=input.dtype).uniform_(
            self.noise_min,
            self.noise_max,
        )

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
            elif noise_param.ndim >= 3:
                noise_param = noise_param.view(batch_size, -1)
                if noise_param.size(1) != 1:
                    raise ValueError(
                        f"noise_param with ndim >= 3 must contain one value per sample, got {tuple(noise_param.shape)}"
                    )
            else:
                raise ValueError(f"noise_param must be scalar, 1D, or 2D, got {noise_param.ndim}D")

        return noise_param

    def _view_param_like(self, param, target):
        if not torch.is_tensor(param):
            param = target.new_tensor(float(param))
        param = param.to(device=target.device, dtype=target.dtype)
        if param.ndim == 0:
            return param
        batch_size = target.size(0)
        if param.ndim == 1:
            param = param.view(batch_size, 1)
        if target.ndim <= 2:
            return param
        if param.ndim == 2:
            return param.view(batch_size, 1, *([1] * (target.ndim - 2)))
        return param

    def _compute_score_metrics(self, pred_score, target_score):
        """
        pred_score: model output score, glogprob
        target_score: score target
        """

        with torch.no_grad():
            pred = pred_score.detach()
            target = target_score.detach()

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

    def glogprob(self, input, noise_param=None):
        """노이즈가 이미 들어간 관측값 input에서 score/log-density-gradient를 예측한다."""
        input = self._format_input(input)
        noise_param = self._prepare_noise_param(input, noise_param, self.noise_param)

        if self.backbone == "mlp":
            h = torch.cat([input, noise_param], dim=1)
            return self.main(h)

        return self.main(input, noise_param)

    def loss(self, glogprob, input, x_bar, eps, noise_param):
        if self.noise_type == "gaussian":
            sigma = self._view_param_like(noise_param, glogprob)
            target = -eps
            pred = sigma * glogprob
            return F.mse_loss(pred, target)

        elif self.noise_type == "poisson":
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

            return F.mse_loss(glogprob, target_score)

        elif self.noise_type == "gamma":
            alpha = self._view_param_like(noise_param.clamp_min(1e-6), input)

            tiny = 1.0 / 255.0

            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(tiny)

            target_score = (alpha - 1.0) / y_safe - alpha / x_safe
            target_score = target_score.clamp(-100.0, 100.0)

            return F.mse_loss(glogprob, target_score)
        else:
            raise NotImplementedError(f"Unknown noise_type: {self.noise_type}")

    def forward(self, input, noise_param=None):
        input = self._format_input(input)
        noise_param = self._prepare_noise_param(input, noise_param, self.noise_param)

        x_bar, eps = self.add_noise(input, noise_param)
        glogprob = self.glogprob(x_bar, noise_param=noise_param)

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

        return glogprob, loss

    def loss_with_metric(self, glogprob, input, x_bar, eps, noise_param):
        if self.noise_type == "gaussian":
            sigma = self._view_param_like(noise_param.clamp_min(1e-6), glogprob)

            target_score = -eps / sigma
            loss = F.mse_loss(sigma * glogprob, -eps)

            self.last_metrics = self._compute_score_metrics(
                pred_score=glogprob,
                target_score=target_score,
            )

            return loss

        elif self.noise_type == "poisson":
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

            loss = F.mse_loss(glogprob, target_score)

            self.last_metrics = self._compute_score_metrics(
                pred_score=glogprob,
                target_score=target_score,
            )

            return loss

        elif self.noise_type == "gamma":
            alpha = self._view_param_like(noise_param.clamp_min(1e-6), input)
            tiny = 1.0 / 255.0

            x_safe = input.clamp_min(tiny)
            y_safe = x_bar.clamp_min(tiny)

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
