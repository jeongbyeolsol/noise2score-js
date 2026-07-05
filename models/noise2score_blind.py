import torch
from torch.nn import functional as F

from models.noise2score import Noise2Score


class Noise2ScoreBlind(Noise2Score):
    def __init__(
        self,
        ardae,
        noise_type="gaussian",
        noise_param=0.1,
        score_sigma=0.1,
        clamp=True,
    ):
        super().__init__(
            ardae=ardae,
            noise_type=noise_type,
            noise_param=noise_param,
            score_sigma=score_sigma,
            clamp=clamp,
        )

    @staticmethod
    def _make_param_grid(
        noise_type,
        device,
        dtype,
        param_min=None,
        param_max=None,
        num_candidates=50,
        candidate_params=None,
    ):
        """
        Blind denoising에서 탐색할 noise parameter grid 생성.

        gaussian:
            param = sigma, noise std
        poisson:
            param = peak
        gamma:
            param = alpha
        """
        if candidate_params is not None:
            return torch.as_tensor(candidate_params, device=device, dtype=dtype)

        if noise_type == "gaussian":
            param_min = 1e-3 if param_min is None else param_min
            param_max = 5e-1 if param_max is None else param_max

            return torch.logspace(
                torch.log10(torch.tensor(param_min, device=device, dtype=dtype)),
                torch.log10(torch.tensor(param_max, device=device, dtype=dtype)),
                steps=num_candidates,
                device=device,
                dtype=dtype,
            )

        if noise_type == "poisson":
            param_min = 1e-3 if param_min is None else param_min
            param_max = 1.0 if param_max is None else param_max

            return torch.logspace(
                torch.log10(torch.tensor(param_min, device=device, dtype=dtype)),
                torch.log10(torch.tensor(param_max, device=device, dtype=dtype)),
                steps=num_candidates,
                device=device,
                dtype=dtype,
            )

        if noise_type == "gamma":
            param_min = 2.0 if param_min is None else param_min
            param_max = 150.0 if param_max is None else param_max

            return torch.linspace(
                param_min,
                param_max,
                steps=num_candidates,
                device=device,
                dtype=dtype,
            )

        raise NotImplementedError(f"Unknown noise_type: {noise_type}")

    @staticmethod
    def _as_image_tensor(x, image_shape=None):
        """
        TV 계산용으로 x를 [B, C, H, W] 형태로 변환.
        U-Net이면 이미 4D일 가능성이 높고,
        MLP이면 image_shape=(1, H, W)를 넘겨줘야 함.
        """
        if x.dim() == 4:
            return x

        if x.dim() == 2 and image_shape is not None:
            return x.view(x.size(0), *image_shape)

        raise ValueError(
            "TV blind quality를 쓰려면 입력이 [B,C,H,W]이거나 "
            "MLP 입력에 대해 image_shape=(C,H,W)를 제공해야 합니다."
        )

    @staticmethod
    def _total_variation(x):
        """
        x: [B, C, H, W]
        """
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    @staticmethod
    def _range_penalty(x):
        below = F.relu(-x)
        above = F.relu(x - 1.0)
        return (below.pow(2) + above.pow(2)).mean()

    def _blind_quality(
        self,
        x_hat,
        y=None,
        image_shape=None,
        tv_weight=1.0,
        range_weight=0.0,
        data_weight=0.0,
    ):
        """
        Blind parameter 선택용 품질 함수.

        기본:
            Q(x_hat) = TV(x_hat)

        data_weight > 0:
            Q(x_hat) = TV(x_hat) + data_weight * MSE(x_hat, y)
        """
        x_img = self._as_image_tensor(x_hat, image_shape=image_shape)
        q = tv_weight * self._total_variation(x_img)

        if range_weight > 0:
            q = q + range_weight * self._range_penalty(x_hat)

        if data_weight > 0:
            if y is None:
                raise ValueError("data_weight > 0이면 y가 필요합니다.")
            q = q + data_weight * F.mse_loss(x_hat, y)

        return q

    @torch.no_grad()
    def blind_denoise(
        self,
        y,
        candidate_params=None,
        param_min=None,
        param_max=None,
        num_candidates=50,
        score_sigma_mode="same",
        fixed_score_sigma=None,
        image_shape=None,
        tv_weight=1.0,
        range_weight=0.0,
        data_weight=0.0,
        return_history=True,
    ):
        """
        Blind Noise2Score denoising.

        ARDAE는 재학습하지 않고,
        noise_param 후보를 sweep해서 quality가 가장 좋은 결과를 선택한다.
        """
        device = y.device
        dtype = y.dtype

        params = self._make_param_grid(
            noise_type=self.noise_type,
            device=device,
            dtype=dtype,
            param_min=param_min,
            param_max=param_max,
            num_candidates=num_candidates,
            candidate_params=candidate_params,
        )

        best_x = None
        best_param = None
        best_score_sigma = None
        best_q = None
        history = []

        for param in params:
            param_value = float(param.detach().cpu())

            if score_sigma_mode == "same":
                score_sigma = param_value

            elif score_sigma_mode == "fixed":
                if fixed_score_sigma is None:
                    raise ValueError(
                        'score_sigma_mode="fixed"이면 fixed_score_sigma를 제공해야 합니다.'
                    )
                score_sigma = fixed_score_sigma

            elif score_sigma_mode == "current":
                score_sigma = self.score_sigma

            else:
                raise ValueError(
                    f"Unknown score_sigma_mode: {score_sigma_mode}. "
                    'Choose from ["same", "fixed", "current"].'
                )

            x_hat = self.denoise(
                y,
                noise_param=param_value,
                score_sigma=score_sigma,
            )

            q = self._blind_quality(
                x_hat,
                y=y,
                image_shape=image_shape,
                tv_weight=tv_weight,
                range_weight=range_weight,
                data_weight=data_weight,
            )

            q_value = float(q.detach().cpu())

            if return_history:
                history.append(
                    {
                        "noise_param": param_value,
                        "score_sigma": float(score_sigma),
                        "quality": q_value,
                    }
                )

            if best_q is None or q_value < best_q:
                best_q = q_value
                best_param = param_value
                best_score_sigma = float(score_sigma)
                best_x = x_hat

        result = {
            "x_hat": best_x,
            "best_noise_param": best_param,
            "best_score_sigma": best_score_sigma,
            "best_quality": best_q,
        }

        if return_history:
            result["history"] = history

        return result