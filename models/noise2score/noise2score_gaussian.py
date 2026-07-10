from models.noise2score.noise2score_base import Noise2Score, _with_noise_type


class GaussianNoise2Score(Noise2Score):
    def __init__(self, *args, **kwargs):
        args, kwargs = _with_noise_type(args, kwargs, "gaussian")
        super().__init__(*args, **kwargs)

    def denoise_from_score(self, y, score, noise_param, smoothing=0.0):
        x_hat = y + noise_param ** 2 * score
        if self.clamp:
            x_hat = x_hat.clamp(0, 1)
        return x_hat
