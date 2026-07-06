# run_noise2score_blind.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import (
    StreamingImagePatchDataset,
    find_image_paths,
    load_array,
    preprocess_ardae_data,
)
from models.ardae import ARDAE
from utils import add_gaussian_noise, add_poisson_noise, add_gamma_noise
from utils import make_unique_save_dir, save_config, config_all_from_to

from models.noise2score_blind import Noise2ScoreBlind



def parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clean-data", type=str, required=True)
    parser.add_argument(
        "--data-mode",
        type=str,
        default="array",
        choices=["array", "image-folder"],
        help="array reads one array file; image-folder streams image or per-image npy files.",
    )
    parser.add_argument("--key", type=str, default=None)

    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs="+",
        default=None,
        help="Override/restore UNet image shape: C H W or H W.",
    )
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size for --data-mode image-folder.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride for --data-mode image-folder.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Channels for streamed image/npy folder data.")
    parser.add_argument("--max-patches-per-image", type=int, default=None)
    parser.add_argument("--recursive-images", action="store_true")

    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "poisson", "gamma"],
    )

    # 실제 관측 noisy image를 만들 때 쓰는 true parameter.
    # blind 평가에서는 metric 계산용 synthetic corruption에 필요하다.
    parser.add_argument("--noise-param", type=float, default=0.1)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Gaussian smoothing std for non-Gaussian Noise2Score denoising. 0 keeps the closed-form rule.",
    )
    parser.add_argument(
        "--smoothing-samples",
        type=int,
        default=8,
        help="Monte Carlo samples used when --smoothing > 0.",
    )

    # blind parameter candidates
    parser.add_argument(
        "--candidate-params",
        type=str,
        default=None,
        help="Comma-separated candidates, e.g. 0.03,0.05,0.075,0.1,0.125,0.15",
    )
    parser.add_argument(
        "--candidate-smoothing",
        type=str,
        default=None,
        help=(
            "Comma-separated Gaussian smoothing candidates for smoothed "
            "non-Gaussian Noise2Score, e.g. 0.03,0.05,0.075,0.1,0.125,0.15,0.2. "
            "With --score-sigma-mode same, score_sigma follows each smoothing candidate."
        ),
    )
    parser.add_argument("--param-min", type=float, default=None)
    parser.add_argument("--param-max", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=50)

    # ARDAE query sigma 설정
    parser.add_argument(
        "--score-sigma-mode",
        type=str,
        default="same",
        choices=["same", "fixed", "current"],
        help=(
            "same: score_sigma = candidate noise_param. "
            "fixed: use --fixed-score-sigma. "
            "current: use --score-sigma."
        ),
    )
    parser.add_argument("--score-sigma", type=float, default=0.1)
    parser.add_argument("--fixed-score-sigma", type=float, default=None)

    # blind quality Q(x_hat)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--range-weight", type=float, default=0.0)
    parser.add_argument("--data-weight", type=float, default=0.0)

    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/noise2score_blind")

    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-output-limit", type=int, default=64)

    parser.add_argument('--copy-info', action="store_true")
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
            return (
                int(clean.shape[-1]),
                int(clean.shape[1]),
                int(clean.shape[2]),
            )

    if clean.ndim == 3:
        return (1, int(clean.shape[1]), int(clean.shape[2]))

    side = int(round(float(input_dim) ** 0.5))
    if side * side == int(input_dim):
        return (1, side, side)

    raise ValueError(
        "Could not infer image_shape for UNet checkpoint. "
        "Pass --image-shape C H W."
    )


def load_ardae_from_checkpoint(path, input_dim, device, clean=None, image_shape_override=None):
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "mlp")

    image_shape = None
    if backbone == "unet":
        if clean is None:
            image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
        else:
            image_shape = infer_image_shape_from_data(
                clean,
                input_dim,
                ckpt_args,
                override=image_shape_override,
            )

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
        channel_mults=_as_tuple(
            ckpt_args.get("channel_mults"),
            default=(1, 2, 4, 8),
        ),
        use_norm=not ckpt_args.get("no_norm", False),
        use_gaussian_smoothing=ckpt_args.get("use_gaussian_smoothing", False),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, backbone, image_shape


def infer_stream_shape(args, ckpt_image_shape=None):
    image_shape = normalize_image_shape_arg(args.image_shape)
    if image_shape is None:
        image_shape = normalize_image_shape_arg(ckpt_image_shape)
    if image_shape is None:
        if args.patch_size is None:
            raise ValueError(
                "--data-mode image-folder requires --image-shape or --patch-size."
            )
        image_shape = (args.channels, args.patch_size, args.patch_size)

    channels, height, width = image_shape
    if height != width:
        raise ValueError("Streamed patch evaluation currently requires square patches.")
    if args.patch_size is None:
        args.patch_size = height
    if args.patch_size != height or args.patch_size != width:
        raise ValueError(
            f"--patch-size {args.patch_size} does not match image_shape {image_shape}."
        )
    if args.stride is None:
        args.stride = args.patch_size
    args.channels = channels
    return image_shape


def make_clean_loader(args, backbone, image_shape, device, raw_clean=None):
    if args.data_mode == "array":
        if raw_clean is None:
            raw_clean = load_array(args.clean_data, key=args.key)
        raw_clean = raw_clean.float()
        if raw_clean.max() > 1.5:
            raw_clean = raw_clean / 255.0

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
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        return loader, len(clean), str(clean.dtype)

    clean_paths = find_image_paths(args.clean_data, recursive=args.recursive_images)
    if len(clean_paths) == 0:
        raise ValueError(f"No image/npy files found in {args.clean_data}.")

    dataset = StreamingImagePatchDataset(
        image_paths=clean_paths,
        patch_size=args.patch_size,
        stride=args.stride,
        channels=args.channels,
        max_patches_per_image=args.max_patches_per_image,
        seed=0,
        flatten=(backbone != "unet"),
        dtype=torch.float32,
        shuffle_images=False,
        shuffle_patches=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    args.num_clean_files = len(clean_paths)
    args.num_patches = len(dataset)
    return loader, len(dataset), "torch.float32"


def info_dir_for(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def move_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    return batch.to(device, non_blocking=True)


def get_score_sigma(args, candidate_param, candidate_smoothing=None):
    if args.score_sigma_mode == "same":
        if candidate_smoothing is not None:
            return candidate_smoothing
        return candidate_param

    if args.score_sigma_mode == "fixed":
        if args.fixed_score_sigma is None:
            raise ValueError(
                '--score-sigma-mode fixed requires --fixed-score-sigma.'
            )
        return args.fixed_score_sigma

    if args.score_sigma_mode == "current":
        return args.score_sigma

    raise ValueError(f"Unknown score_sigma_mode: {args.score_sigma_mode}")


def build_candidate_jobs(args, n2s, device):
    candidate_params = parse_float_list(args.candidate_params)
    candidate_smoothing = parse_float_list(args.candidate_smoothing)

    params = n2s._make_param_grid(
        noise_type=args.noise_type,
        device=device,
        dtype=torch.float32,
        param_min=args.param_min,
        param_max=args.param_max,
        num_candidates=args.num_candidates,
        candidate_params=candidate_params,
    )
    params_list = [float(p.detach().cpu()) for p in params]

    if candidate_smoothing is None:
        return [
            {
                "noise_param": param_value,
                "smoothing": float(args.smoothing or 0.0),
                "uses_smoothing_candidate": False,
            }
            for param_value in params_list
        ], params_list, None

    if args.noise_type == "gaussian":
        raise ValueError("--candidate-smoothing is intended for non-Gaussian smoothing.")

    smoothing_list = [float(v) for v in candidate_smoothing]
    if any(v <= 0.0 for v in smoothing_list):
        raise ValueError("--candidate-smoothing values must be > 0.")

    if candidate_params is None and args.param_min is None and args.param_max is None:
        params_list = [float(args.noise_param)]

    jobs = []
    for smoothing_value in smoothing_list:
        for param_value in params_list:
            jobs.append(
                {
                    "noise_param": param_value,
                    "smoothing": smoothing_value,
                    "uses_smoothing_candidate": True,
                }
            )

    return jobs, params_list, smoothing_list


def denoise_from_score(y, score, noise_type, noise_param, clamp=True, smoothing=0.0):
    smoothing = float(smoothing or 0.0)

    if smoothing > 0.0 and noise_type != "gaussian":
        x_hat = y + smoothing ** 2 * score

    elif noise_type == "gaussian":
        sigma = noise_param
        x_hat = y + sigma ** 2 * score

    elif noise_type == "poisson":
        peak = noise_param
        x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score / peak)

    elif noise_type == "gamma":
        alpha = noise_param
        denom = (alpha - 1.0) - y * score
        denom = denom.clamp_min(1e-6)
        x_hat = alpha * y / denom

    else:
        raise NotImplementedError(f"Unknown noise_type: {noise_type}")

    if clamp:
        x_hat = x_hat.clamp(0, 1)

    return x_hat


@torch.no_grad()
def main():
    args = parse_args()

    if args.score_sigma_mode == "fixed" and args.fixed_score_sigma is None:
        raise ValueError("--score-sigma-mode fixed requires --fixed-score-sigma.")
    if args.smoothing < 0:
        raise ValueError("--smoothing must be >= 0.")
    smoothing_enabled = args.smoothing > 0 or args.candidate_smoothing is not None
    if smoothing_enabled and args.smoothing_samples < 1:
        raise ValueError(
            "--smoothing-samples must be >= 1 when smoothing is enabled."
        )

    device = torch.device(args.device)
    requested_output_dir = Path(args.output_dir)
    output_dir = make_unique_save_dir(requested_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "run_config.json", args)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_image_shape = ckpt.get("args", {}).get("image_shape")
    raw_clean = None
    if args.data_mode == "image-folder":
        stream_image_shape = infer_stream_shape(args, ckpt_image_shape)
    else:
        stream_image_shape = args.image_shape
        raw_clean = load_array(args.clean_data, key=args.key)

    ardae, backbone, image_shape = load_ardae_from_checkpoint(
        path=args.checkpoint,
        input_dim=args.input_dim,
        device=device,
        clean=raw_clean,
        image_shape_override=stream_image_shape,
    )

    loader, num_samples, clean_dtype = make_clean_loader(
        args,
        backbone,
        image_shape,
        device,
        raw_clean=raw_clean,
    )

    n2s = Noise2ScoreBlind(
        ardae=ardae,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        score_sigma=args.score_sigma,
        clamp=not args.no_clamp,
    )

    candidate_jobs, params_list, smoothing_list = build_candidate_jobs(
        args=args,
        n2s=n2s,
        device=device,
    )
    num_candidates = len(candidate_jobs)

    total_count = 0
    noisy_mse_sum = 0.0

    quality_sums = [0.0 for _ in range(num_candidates)]
    denoised_mse_sums = [0.0 for _ in range(num_candidates)]
    cos_sums = [0.0 for _ in range(num_candidates)]

    save_clean = []
    save_noisy = []
    saved_count = 0

    pbar = tqdm(loader, desc="Blind Noise2Score eval", total=len(loader))

    for batch in pbar:
        x = move_batch(batch, device)
        batch_size = x.size(0)

        y = add_observation_noise(
            x=x,
            noise_type=args.noise_type,
            noise_param=args.noise_param,
        )

        noisy_mse = F.mse_loss(y, x).item()
        noisy_mse_sum += noisy_mse * batch_size

        if args.save_output and saved_count < args.save_output_limit:
            remain = args.save_output_limit - saved_count
            take = min(remain, batch_size)

            save_clean.append(x[:take].detach().cpu())
            save_noisy.append(y[:take].detach().cpu())

            saved_count += take

        for idx, candidate in enumerate(candidate_jobs):
            param_value = candidate["noise_param"]
            smoothing_value = candidate["smoothing"]
            candidate_smoothing = (
                smoothing_value if candidate["uses_smoothing_candidate"] else None
            )
            score_sigma = get_score_sigma(
                args,
                param_value,
                candidate_smoothing=candidate_smoothing,
            )

            score = n2s.score(
                y,
                score_sigma=score_sigma,
                smoothing=smoothing_value,
                smoothing_samples=args.smoothing_samples,
            )

            x_hat = denoise_from_score(
                y=y,
                score=score,
                noise_type=args.noise_type,
                noise_param=param_value,
                clamp=not args.no_clamp,
                smoothing=smoothing_value,
            )

            q = n2s._blind_quality(
                x_hat=x_hat,
                y=y,
                image_shape=image_shape,
                tv_weight=args.tv_weight,
                range_weight=args.range_weight,
                data_weight=args.data_weight,
            ).item()

            denoised_mse = F.mse_loss(x_hat, x).item()

            cos = F.cosine_similarity(
                score.flatten(1),
                (x - y).flatten(1),
                dim=1,
                eps=1e-8,
            ).mean().item()

            quality_sums[idx] += q * batch_size
            denoised_mse_sums[idx] += denoised_mse * batch_size
            cos_sums[idx] += cos * batch_size

        total_count += batch_size

    noisy_mse = noisy_mse_sum / total_count

    candidate_history = []
    for idx, candidate in enumerate(candidate_jobs):
        param_value = candidate["noise_param"]
        smoothing_value = candidate["smoothing"]
        candidate_smoothing = (
            smoothing_value if candidate["uses_smoothing_candidate"] else None
        )
        score_sigma = get_score_sigma(
            args,
            param_value,
            candidate_smoothing=candidate_smoothing,
        )
        avg_quality = quality_sums[idx] / total_count
        avg_denoised_mse = denoised_mse_sums[idx] / total_count
        avg_cos = cos_sums[idx] / total_count

        candidate_history.append(
            {
                "noise_param": param_value,
                "smoothing": smoothing_value,
                "score_sigma": float(score_sigma),
                "quality": avg_quality,
                "denoised_mse": avg_denoised_mse,
                "denoised_psnr": psnr_from_mse(avg_denoised_mse),
                "score_clean_direction_cos": avg_cos,
            }
        )

    # blind 선택 기준: clean GT가 아니라 quality 최소화
    best_idx = min(range(num_candidates), key=lambda i: candidate_history[i]["quality"])
    best = candidate_history[best_idx]

    denoised_mse = best["denoised_mse"]
    score_cos = best["score_clean_direction_cos"]

    summary = {
        "mode": "blind",
        "backbone": backbone,
        "image_shape": list(image_shape) if image_shape is not None else None,
        "noise_type": args.noise_type,

        # synthetic eval에서 실제로 넣은 노이즈.
        # 실제 blind 상황에서는 알 수 없는 값이지만, 여기서는 평가용으로 기록.
        "true_noise_param_for_eval": args.noise_param,
        "smoothing": args.smoothing,
        "candidate_smoothing": smoothing_list,
        "smoothing_samples": args.smoothing_samples,

        "score_sigma_mode": args.score_sigma_mode,
        "fixed_score_sigma": args.fixed_score_sigma,
        "tv_weight": args.tv_weight,
        "range_weight": args.range_weight,
        "data_weight": args.data_weight,
        "data_mode": args.data_mode,
        "num_samples": num_samples,
        "clean_dtype": clean_dtype,

        "estimated_noise_param": best["noise_param"],
        "estimated_smoothing": best["smoothing"],
        "estimated_score_sigma": best["score_sigma"],
        "best_quality": best["quality"],

        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse),
        "denoised_psnr": psnr_from_mse(denoised_mse),
        "score_clean_direction_cos": score_cos,
        "improved_mse": denoised_mse < noisy_mse,
        "requested_output_dir": requested_output_dir,
        "output_dir": output_dir,
        "checkpoint": Path(args.checkpoint),
        "clean_data": Path(args.clean_data),

        "candidate_history": candidate_history,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    save_config(output_dir / Path("summary.json"), summary)

    save_config(output_dir / Path("candidate_history.json"), candidate_history)

    
    if args.copy_info:
        info_sources = {
            "checkpoint_": Path(args.checkpoint).parent,
            "data_": info_dir_for(args.clean_data),
        }

        for prefix, info_dir in info_sources.items():
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=False)
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=True)
    
    if args.save_output:
        save_output_dir = output_dir / Path("output")
        save_output_dir.mkdir(parents=True, exist_ok=True)

        clean_tensor = torch.cat(save_clean, dim=0)
        noisy_tensor = torch.cat(save_noisy, dim=0)

        best_noise_param = float(best["noise_param"])
        best_smoothing = float(best["smoothing"])
        best_score_sigma = float(best["score_sigma"])

        noisy_device = noisy_tensor.to(device)

        score = n2s.score(
            noisy_device,
            score_sigma=best_score_sigma,
            smoothing=best_smoothing,
            smoothing_samples=args.smoothing_samples,
        )
        denoised = denoise_from_score(
            y=noisy_device,
            score=score,
            noise_type=args.noise_type,
            noise_param=best_noise_param,
            clamp=not args.no_clamp,
            smoothing=best_smoothing,
        )

        clean_np = clean_tensor.numpy()
        noisy_np = noisy_tensor.numpy()
        denoised_np = denoised.detach().cpu().numpy()
        score_np = score.detach().cpu().numpy()

        np.save(save_output_dir / "clean.npy", clean_np)
        np.save(save_output_dir / "noisy.npy", noisy_np)
        np.save(save_output_dir / "denoised.npy", denoised_np)
        np.save(save_output_dir / "score.npy", score_np)

        print(f"[saved] outputs saved to {save_output_dir}")


if __name__ == "__main__":
    main()
