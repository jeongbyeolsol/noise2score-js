from models.noise2score.noise2score_base import Noise2Score, _with_noise_type


class GammaNoise2Score(Noise2Score):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "gamma")
        super().__init__(*args, **kwargs)

    def denoise_from_score(self, y, score, noise_param, smoothing=0.0):
        smoothing = float(smoothing or 0.0)
        if smoothing > 0.0:
            x_hat = y + smoothing ** 2 * score
        else:
            alpha = noise_param
            denom = (alpha - 1.0) - y * score
            denom = denom.clamp_min(1e-6)
            x_hat = alpha * y / denom

        if self.clamp:
            x_hat = x_hat.clamp(0, 1)
        return x_hat
