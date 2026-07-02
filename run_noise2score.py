# run_noise2score.py

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_array, preprocess_ardae_data
from models.ardae import ARDAE
from models.noise2score import Noise2Score
from utils import add_gaussian_noise, add_poisson_noise, add_gamma_noise
from utils import make_unique_save_dir, log_message, save_config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clean-data", type=str, required=True)
    parser.add_argument("--key", type=str, default=None)

    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-shape", type=int, nargs="+", default=None, help="Override/restore UNet image shape: C H W or H W.")

    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma"])
    parser.add_argument("--noise-param", type=float, default=0.1)
    parser.add_argument("--score-sigma", type=float, default=0.01)

    parser.add_argument("--output-dir", type=str, default="results/noise2score")

    
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-output-dir", type=str, default="results/output")
    parser.add_argument("--save-output-limit", type=int, default=64)
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


def _as_tuple(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def normalize_image_shape_arg(image_shape):
    image_shape = _as_tuple(image_shape)
    if image_shape is None:
        return None
    if len(image_shape) == 2:
        return (1, *image_shape)
    if len(image_shape) == 3:
        return image_shape
    raise ValueError("image_shape must be H W or C H W")


def infer_image_shape_from_data(clean, input_dim, ckpt_args, override=None):
    image_shape = normalize_image_shape_arg(override)
    if image_shape is not None:
        return image_shape

    image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
    if image_shape is not None:
        return image_shape

    if clean.ndim == 4:
        if clean.shape[1] in (1, 3):
            return tuple(int(v) for v in clean.shape[1:])
        if clean.shape[-1] in (1, 3):
            return (int(clean.shape[-1]), int(clean.shape[1]), int(clean.shape[2]))

    if clean.ndim == 3:
        return (1, int(clean.shape[1]), int(clean.shape[2]))

    side = int(round(float(input_dim) ** 0.5))
    if side * side == int(input_dim):
        return (1, side, side)

    raise ValueError("Could not infer image_shape for UNet checkpoint. Pass --image-shape C H W.")


def load_ardae_from_checkpoint(path, input_dim, device, clean=None, image_shape_override=None):
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "mlp")

    image_shape = None
    if backbone == "unet":
        if clean is None:
            image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
        else:
            image_shape = infer_image_shape_from_data(clean, input_dim, ckpt_args, override=image_shape_override)

    model = ARDAE(
        input_dim=input_dim,
        h_dim=ckpt_args.get("h_dim", 1000),
        noise_param=ckpt_args.get("noise_param", 0.1),
        num_hidden_layers=ckpt_args.get("num_hidden_layers", 1),
        nonlinearity=ckpt_args.get("nonlinearity", "tanh"),
        noise_type=ckpt_args.get("noise_type", "gaussian"),
        use_metric=False,
        backbone=backbone,
        image_shape=image_shape,
        base_channels=ckpt_args.get("base_channels", 64),
        channel_mults=_as_tuple(ckpt_args.get("channel_mults"), default=(1, 2, 4, 8)),
        use_norm=not ckpt_args.get("no_norm", False),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, backbone, image_shape


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device)
    output_dir = make_unique_save_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_clean = load_array(args.clean_data, key=args.key).float()

    if raw_clean.max() > 1.5:
        raw_clean = raw_clean / 255.0

    ardae, backbone, image_shape = load_ardae_from_checkpoint(
        path=args.checkpoint,
        input_dim=args.input_dim,
        device=device,
        clean=raw_clean,
        image_shape_override=args.image_shape,
    )

    clean = preprocess_ardae_data(
        data=raw_clean.clamp(0, 1),
        input_dim=args.input_dim,
        normalize=None,
        flatten=(backbone != "unet"),
        image_shape=image_shape,
    )

    loader = DataLoader(
        TensorDataset(clean),
        batch_size=args.batch_size,
        shuffle=False,
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
    
    save_clean = []
    save_noisy = []
    save_denoised = []
    save_score = []
    saved_count = 0

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
        
        if args.save_output and saved_count < args.save_output_limit:
            remain = args.save_output_limit - saved_count
            take = min(remain, x.size(0))

            save_clean.append(x[:take].detach().cpu())
            save_noisy.append(y[:take].detach().cpu())
            save_denoised.append(x_hat[:take].detach().cpu())
            save_score.append(score[:take].detach().cpu())

            saved_count += take

        batch_size = x.size(0)
        total_count += batch_size
        noisy_mse_sum += noisy_mse * batch_size
        denoised_mse_sum += denoised_mse * batch_size
        cos_sum += cos * batch_size

    noisy_mse = noisy_mse_sum / total_count
    denoised_mse = denoised_mse_sum / total_count
    score_cos = cos_sum / total_count

    summary = {
        "backbone": backbone,
        "image_shape": list(image_shape) if image_shape is not None else None,
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
    save_config(output_dir / "summary.json", summary)
    
    if args.save_output:
        save_output_dir = Path(args.save_output_dir)
        save_output_dir.mkdir(parents=True, exist_ok=True)

        clean_np = torch.cat(save_clean, dim=0).numpy()
        noisy_np = torch.cat(save_noisy, dim=0).numpy()
        denoised_np = torch.cat(save_denoised, dim=0).numpy()
        score_np = torch.cat(save_score, dim=0).numpy()

        np.save(save_output_dir / "clean.npy", clean_np)
        np.save(save_output_dir / "noisy.npy", noisy_np)
        np.save(save_output_dir / "denoised.npy", denoised_np)
        np.save(save_output_dir / "score.npy", score_np)

        print(f"[saved] outputs saved to {save_output_dir}")


if __name__ == "__main__":
    main()