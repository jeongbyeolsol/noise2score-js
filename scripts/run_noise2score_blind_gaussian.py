import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.noise2score_distribution_args import parse_noise2score_args
from scripts.run_noise2score_blind_base import build_parser, run


if __name__ == "__main__":
    args = parse_noise2score_args(build_parser, noise_type="gaussian")
    run(args, forced_noise_type="gaussian")
