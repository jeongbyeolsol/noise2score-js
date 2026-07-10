from config import add_config_argument, parse_args_with_config
from utils.evaluation import parse_float_list


def parse_noise2score_args(build_parser, noise_type=None, argv=None):
    parser = build_parser()
    add_config_argument(parser)
    _add_distribution_args(parser, noise_type)
    args = parse_args_with_config(
        parser,
        argv=argv,
        aliases=_config_aliases(noise_type),
    )
    _apply_distribution_aliases(args, noise_type)
    return args


def _add_distribution_args(parser, noise_type):
    if noise_type == "gaussian":
        parser.add_argument(
            "--sigma",
            type=float,
            dest="noise_param",
            help="Gaussian noise std. Alias for --noise-param.",
        )
        parser.add_argument(
            "--candidate-sigmas",
            type=str,
            dest="candidate_params",
            help="Comma-separated Gaussian sigma candidates for blind runs.",
        )
        return

    if noise_type == "poisson":
        parser.add_argument(
            "--peak",
            type=float,
            dest="poisson_peak",
            help="Poisson peak. Larger peak means weaker noise.",
        )
        parser.add_argument(
            "--lam",
            type=float,
            dest="poisson_lam",
            help="Poisson lam where peak = 1 / lam.",
        )
        parser.add_argument(
            "--candidate-peaks",
            type=str,
            dest="candidate_params",
            help="Comma-separated Poisson peak candidates for blind runs.",
        )
        parser.add_argument(
            "--candidate-lams",
            type=str,
            dest="candidate_lams",
            help="Comma-separated Poisson lam candidates; converted to peak candidates.",
        )
        return

    if noise_type == "gamma":
        parser.add_argument(
            "--alpha",
            type=float,
            dest="noise_param",
            help="Gamma alpha/concentration. Alias for --noise-param.",
        )
        parser.add_argument(
            "--concentration",
            type=float,
            dest="noise_param",
            help="Gamma concentration. Alias for --noise-param.",
        )
        parser.add_argument(
            "--candidate-alphas",
            type=str,
            dest="candidate_params",
            help="Comma-separated Gamma alpha candidates for blind runs.",
        )


def _config_aliases(noise_type):
    if noise_type == "gaussian":
        return {
            "sigma": "noise_param",
            "candidate_sigmas": "candidate_params",
            "gaussian": {
                "sigma": "noise_param",
                "candidate_sigmas": "candidate_params",
            },
        }

    if noise_type == "poisson":
        return {
            "peak": "poisson_peak",
            "lam": "poisson_lam",
            "candidate_peaks": "candidate_params",
            "candidate_lams": "candidate_lams",
            "poisson": {
                "peak": "poisson_peak",
                "lam": "poisson_lam",
                "candidate_peaks": "candidate_params",
                "candidate_lams": "candidate_lams",
            },
        }

    if noise_type == "gamma":
        return {
            "alpha": "noise_param",
            "concentration": "noise_param",
            "candidate_alphas": "candidate_params",
            "gamma": {
                "alpha": "noise_param",
                "concentration": "noise_param",
                "candidate_alphas": "candidate_params",
            },
        }

    return None


def _apply_distribution_aliases(args, noise_type):
    if noise_type == "poisson":
        poisson_lam = getattr(args, "poisson_lam", None)
        if poisson_lam is not None:
            if poisson_lam <= 0:
                raise ValueError("Poisson lam must be positive.")
            args.poisson_peak = 1.0 / float(poisson_lam)
            args.noise_param = args.poisson_peak

        candidate_lams = getattr(args, "candidate_lams", None)
        if candidate_lams is not None:
            lams = parse_float_list(candidate_lams)
            if any(value <= 0 for value in lams):
                raise ValueError("Poisson candidate lams must be positive.")
            args.candidate_params = [1.0 / value for value in lams]

    if noise_type is not None:
        args.noise_type = noise_type
