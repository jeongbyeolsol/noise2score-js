# run_noise2score.py

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_array, preprocess_ardae_data
from models.ardae import ARDAE
from models.noise2score import Noise2Score
from utils import add_gaussian_noise, add_poisson_noise, add_gamma_noise


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clean-data", type=str, required=True)
    parser.add_argument("--key", type=str, default=None)

    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma"])
    parser.add_argument("--noise-param", type=float, default=0.1)
    parser.add_argument("--score-sigma", type=float, default=0.01)

    parser.add_argument("--output-dir", type=str, default="results/noise2score")

    return parser.parse_args()


def add_observation_noise(x, noise_type, noise_param):
    if noise_type == "gaussian":
        y, _ = add_gaussian_noise(x, std=noise_param)
        return y.clamp(0, 1)

    if noise_type == "poisson":
        y, _ = add_poisson_noise(x, peak=noise_param)
        return y

    if noise_type == "gamma":
        y, _ = add_gamma_noise(x, concentration=noise_param)
        return y

    raise NotImplementedError(noise_type)


def psnr_from_mse(mse, max_value=1.0):
    mse = max(float(mse), 1e-12)
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def load_ardae_from_checkpoint(path, input_dim, device):
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    model = ARDAE(
        input_dim=input_dim,
        h_dim=ckpt_args.get("h_dim", 1000),
        noise_param=ckpt_args.get("noise_param", 0.1),
        num_hidden_layers=ckpt_args.get("num_hidden_layers", 1),
        nonlinearity=ckpt_args.get("nonlinearity", "tanh"),
        noise_type=ckpt_args.get("noise_type", "gaussian"),
        use_metric=False,
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = load_array(args.clean_data, key=args.key).float()

    if clean.max() > 1.5:
        clean = clean / 255.0

    clean = preprocess_ardae_data(
        data=clean.clamp(0, 1),
        input_dim=args.input_dim,
        normalize=None,
        flatten=True,
    )

    loader = DataLoader(
        TensorDataset(clean),
        batch_size=args.batch_size,
        shuffle=False,
    )

    ardae = load_ardae_from_checkpoint(
        path=args.checkpoint,
        input_dim=args.input_dim,
        device=device,
    )

    n2s = Noise2Score(
        ardae=ardae,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        score_sigma=args.score_sigma,
    )

    total_count = 0
    noisy_mse_sum = 0.0
    denoised_mse_sum = 0.0
    cos_sum = 0.0

    for (x,) in loader:
        x = x.to(device)

        y = add_observation_noise(
            x=x,
            noise_type=args.noise_type,
            noise_param=args.noise_param,
        )

        x_hat = n2s.denoise(y)

        noisy_mse = F.mse_loss(y, x).item()
        denoised_mse = F.mse_loss(x_hat, x).item()

        score = n2s.score(y)
        cos = F.cosine_similarity(
            score.flatten(1),
            (x - y).flatten(1),
            dim=1,
            eps=1e-8,
        ).mean().item()

        batch_size = x.size(0)
        total_count += batch_size
        noisy_mse_sum += noisy_mse * batch_size
        denoised_mse_sum += denoised_mse * batch_size
        cos_sum += cos * batch_size

    noisy_mse = noisy_mse_sum / total_count
    denoised_mse = denoised_mse_sum / total_count
    score_cos = cos_sum / total_count

    summary = {
        "noise_type": args.noise_type,
        "noise_param": args.noise_param,
        "score_sigma": args.score_sigma,
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse),
        "denoised_psnr": psnr_from_mse(denoised_mse),
        "score_clean_direction_cos": score_cos,
        "improved_mse": denoised_mse < noisy_mse,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()