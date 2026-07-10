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
        parser.add_argument("--sigma-min", type=float, dest="noise_param_min")
        parser.add_argument("--sigma-max", type=float, dest="noise_param_max")
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
            "--peak-min",
            type=float,
            dest="noise_param_min",
            help="Min Poisson peak for synthetic eval.",
        )
        parser.add_argument(
            "--peak-max",
            type=float,
            dest="noise_param_max",
            help="Max Poisson peak for synthetic eval.",
        )
        parser.add_argument(
            "--lam-min",
            type=float,
            dest="poisson_lam_min",
            help="Min Poisson lam for synthetic eval. Converted to peak range.",
        )
        parser.add_argument(
            "--lam-max",
            type=float,
            dest="poisson_lam_max",
            help="Max Poisson lam for synthetic eval. Converted to peak range.",
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
        parser.add_argument("--alpha-min", type=float, dest="noise_param_min")
        parser.add_argument("--alpha-max", type=float, dest="noise_param_max")


def _config_aliases(noise_type):
    if noise_type == "gaussian":
        return {
            "sigma": "noise_param",
            "candidate_sigmas": "candidate_params",
            "sigma_min": "noise_param_min",
            "sigma_max": "noise_param_max",
            "gaussian": {
                "sigma": "noise_param",
                "candidate_sigmas": "candidate_params",
                "sigma_min": "noise_param_min",
                "sigma_max": "noise_param_max",
            },
        }

    if noise_type == "poisson":
        return {
            "peak": "poisson_peak",
            "lam": "poisson_lam",
            "candidate_peaks": "candidate_params",
            "candidate_lams": "candidate_lams",
            "peak_min": "noise_param_min",
            "peak_max": "noise_param_max",
            "lam_min": "poisson_lam_min",
            "lam_max": "poisson_lam_max",
            "poisson": {
                "peak": "poisson_peak",
                "lam": "poisson_lam",
                "candidate_peaks": "candidate_params",
                "candidate_lams": "candidate_lams",
                "peak_min": "noise_param_min",
                "peak_max": "noise_param_max",
                "lam_min": "poisson_lam_min",
                "lam_max": "poisson_lam_max",
            },
        }

    if noise_type == "gamma":
        return {
            "alpha": "noise_param",
            "concentration": "noise_param",
            "candidate_alphas": "candidate_params",
            "alpha_min": "noise_param_min",
            "alpha_max": "noise_param_max",
            "concentration_min": "noise_param_min",
            "concentration_max": "noise_param_max",
            "gamma": {
                "alpha": "noise_param",
                "concentration": "noise_param",
                "candidate_alphas": "candidate_params",
                "alpha_min": "noise_param_min",
                "alpha_max": "noise_param_max",
                "concentration_min": "noise_param_min",
                "concentration_max": "noise_param_max",
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

        lam_min = getattr(args, "poisson_lam_min", None)
        lam_max = getattr(args, "poisson_lam_max", None)
        if lam_min is not None or lam_max is not None:
            if lam_min is None or lam_max is None:
                raise ValueError("Use both --lam-min and --lam-max for a Poisson lam range.")
            if lam_min <= 0 or lam_max <= 0:
                raise ValueError("Poisson lam range values must be positive.")
            peak_min = 1.0 / float(max(lam_min, lam_max))
            peak_max = 1.0 / float(min(lam_min, lam_max))
            args.noise_param_min = peak_min
            args.noise_param_max = peak_max

        candidate_lams = getattr(args, "candidate_lams", None)
        if candidate_lams is not None:
            lams = parse_float_list(candidate_lams)
            if any(value <= 0 for value in lams):
                raise ValueError("Poisson candidate lams must be positive.")
            args.candidate_params = [1.0 / value for value in lams]

    if noise_type is not None:
        args.noise_type = noise_type
