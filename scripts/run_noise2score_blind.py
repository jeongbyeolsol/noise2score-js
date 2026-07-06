# run_noise2score_blind.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

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
from scripts.run_noise2score import (
    load_clean_image_tensor,
    make_stitch_coords,
    save_stitched_tensor,
)
from utils import (
    add_observation_noise,
    config_all_from_to,
    denoise_from_score,
    make_unique_save_dir,
    move_batch,
    normalize_image_shape_arg,
    psnr_from_mse,
    save_config,
)
from utils.checkpoint import load_ardae_from_checkpoint

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
    parser.add_argument("--clean-data", type=str, default=None)
    parser.add_argument(
        "--noisy-data",
        type=str,
        default=None,
        help="Path to already-noisy data/image(s). If set, blind Noise2Score denoises without adding synthetic noise.",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="array",
        choices=["array", "image-folder"],
        help="array reads one array file; image-folder streams image or per-image npy files.",
    )
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--noisy-key", type=str, default=None)

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
    parser.add_argument(
        "--stitch-output",
        action="store_true",
        help="For --data-mode image-folder, blind-denoise patches and stitch them back into full images.",
    )
    parser.add_argument(
        "--stitch-format",
        type=str,
        default="npy",
        choices=["npy", "png", "both"],
        help="Format for stitched full-image outputs.",
    )

    parser.add_argument('--copy-info', action="store_true")
    return parser.parse_args()


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
    data_path = args.noisy_data or args.clean_data
    data_key = args.noisy_key if args.noisy_data is not None else args.key

    if args.data_mode == "array":
        if raw_clean is None:
            raw_clean = load_array(data_path, key=data_key)
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

    clean_paths = find_image_paths(data_path, recursive=args.recursive_images)
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


def quality_image_shape(args, image_shape):
    if image_shape is not None:
        return image_shape
    parsed = normalize_image_shape_arg(args.image_shape)
    if parsed is not None:
        return parsed
    if args.patch_size is not None:
        return (args.channels, args.patch_size, args.patch_size)
    return None


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


def stitch_blind_outputs(args, n2s, backbone, candidate_jobs, output_dir, device):
    if args.data_mode != "image-folder":
        raise ValueError("--stitch-output requires --data-mode image-folder.")
    if args.patch_size is None or args.stride is None:
        raise ValueError("--stitch-output requires resolved --patch-size and --stride.")
    if args.max_patches_per_image is not None:
        raise ValueError("--stitch-output needs all patches; omit --max-patches-per-image.")

    image_paths = find_image_paths(args.noisy_data or args.clean_data, recursive=args.recursive_images)
    if len(image_paths) == 0:
        raise ValueError(f"No image/npy files found in {args.noisy_data or args.clean_data}.")

    stitch_dir = output_dir / "stitched"
    flatten = backbone != "unet"
    patch_size = int(args.patch_size)
    stride = int(args.stride)

    image_summaries = []
    noisy_mse_sum = 0.0
    denoised_mse_sum = 0.0
    pixel_count_sum = 0
    has_clean = args.noisy_data is None

    for image_index, image_path in enumerate(tqdm(image_paths, desc="Blind stitch images")):
        if args.noisy_data is not None:
            clean = None
            noisy = load_clean_image_tensor(image_path, channels=args.channels).to(device)
        else:
            clean = load_clean_image_tensor(image_path, channels=args.channels).to(device)
            noisy = add_observation_noise(
                x=clean.unsqueeze(0),
                noise_type=args.noise_type,
                noise_param=args.noise_param,
            ).squeeze(0)

        channels, height, width = noisy.shape
        coords = make_stitch_coords(height, width, patch_size, stride)
        weight = torch.zeros((1, height, width), device=device, dtype=noisy.dtype)
        for y, x in coords:
            weight[:, y : y + patch_size, x : x + patch_size] += 1.0
        weight = weight.clamp_min(1.0)

        candidate_history = []
        best = None

        for candidate in candidate_jobs:
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

            denoised_acc = torch.zeros_like(noisy)
            score_acc = torch.zeros_like(noisy)

            for start in range(0, len(coords), args.batch_size):
                end = min(start + args.batch_size, len(coords))
                patch_coords = coords[start:end]
                noisy_patches = torch.stack(
                    [
                        noisy[:, y : y + patch_size, x : x + patch_size]
                        for y, x in patch_coords
                    ],
                    dim=0,
                )
                model_input = noisy_patches.reshape(noisy_patches.size(0), -1) if flatten else noisy_patches

                score = n2s.score(
                    model_input,
                    score_sigma=score_sigma,
                    smoothing=smoothing_value,
                    smoothing_samples=args.smoothing_samples,
                )
                denoised = denoise_from_score(
                    y=model_input,
                    score=score,
                    noise_type=args.noise_type,
                    noise_param=param_value,
                    clamp=not args.no_clamp,
                    smoothing=smoothing_value,
                )

                if flatten:
                    denoised = denoised.view(-1, channels, patch_size, patch_size)
                    score = score.view(-1, channels, patch_size, patch_size)

                for patch_index, (y, x) in enumerate(patch_coords):
                    denoised_acc[:, y : y + patch_size, x : x + patch_size] += denoised[patch_index]
                    score_acc[:, y : y + patch_size, x : x + patch_size] += score[patch_index]

            denoised_image = denoised_acc / weight
            score_image = score_acc / weight
            quality = n2s._blind_quality(
                x_hat=denoised_image.unsqueeze(0),
                y=noisy.unsqueeze(0),
                image_shape=(channels, height, width),
                tv_weight=args.tv_weight,
                range_weight=args.range_weight,
                data_weight=args.data_weight,
            ).item()

            denoised_mse = F.mse_loss(denoised_image, clean).item() if clean is not None else None
            history_item = {
                "noise_param": param_value,
                "smoothing": smoothing_value,
                "score_sigma": float(score_sigma),
                "quality": quality,
                "denoised_mse": denoised_mse,
                "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
            }
            candidate_history.append(history_item)

            if best is None or quality < best["quality"]:
                best = {
                    **history_item,
                    "denoised_image": denoised_image,
                    "score_image": score_image,
                }

        noisy_mse = F.mse_loss(noisy, clean).item() if clean is not None else None
        denoised_mse = best["denoised_mse"]
        num_pixels = int(noisy.numel())
        pixel_count_sum += num_pixels
        if clean is not None:
            noisy_mse_sum += noisy_mse * num_pixels
            denoised_mse_sum += denoised_mse * num_pixels

        stem = f"{image_index:04d}_{Path(image_path).stem}"
        if clean is not None:
            save_stitched_tensor(clean, stitch_dir / "clean" / stem, args.stitch_format)
        save_stitched_tensor(noisy, stitch_dir / "noisy" / stem, args.stitch_format)
        save_stitched_tensor(best["denoised_image"], stitch_dir / "denoised" / stem, args.stitch_format)
        save_stitched_tensor(best["score_image"], stitch_dir / "score" / stem, "npy")
        save_config(stitch_dir / "candidate_history" / f"{stem}.json", candidate_history)

        image_summaries.append(
            {
                "image": str(image_path),
                "height": height,
                "width": width,
                "channels": channels,
                "num_patches": len(coords),
                "estimated_noise_param": best["noise_param"],
                "estimated_smoothing": best["smoothing"],
                "estimated_score_sigma": best["score_sigma"],
                "best_quality": best["quality"],
                "noisy_mse": noisy_mse,
                "denoised_mse": denoised_mse,
                "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
                "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
                "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
            }
        )

    noisy_mse = noisy_mse_sum / max(pixel_count_sum, 1) if has_clean else None
    denoised_mse = denoised_mse_sum / max(pixel_count_sum, 1) if has_clean else None
    summary = {
        "mode": "blind",
        "input_mode": "noisy" if args.noisy_data is not None else "synthetic",
        "output_dir": stitch_dir,
        "num_images": len(image_paths),
        "num_pixels": pixel_count_sum,
        "patch_size": patch_size,
        "stride": stride,
        "format": args.stitch_format,
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
        "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
        "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
        "images": image_summaries,
    }
    save_config(stitch_dir / "summary.json", summary)
    return summary


@torch.no_grad()
def main():
    args = parse_args()

    if args.clean_data is None and args.noisy_data is None:
        raise ValueError("Provide --clean-data for synthetic-noise eval or --noisy-data for denoising noisy inputs.")
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
    eval_data_path = args.noisy_data or args.clean_data
    eval_key = args.noisy_key if args.noisy_data is not None else args.key
    if args.data_mode == "image-folder":
        stream_image_shape = infer_stream_shape(args, ckpt_image_shape)
    else:
        stream_image_shape = args.image_shape
        raw_clean = load_array(eval_data_path, key=eval_key)

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
    blind_quality_image_shape = quality_image_shape(args, image_shape)
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
        batch_tensor = move_batch(batch, device)
        if args.noisy_data is not None:
            x = None
            y = batch_tensor
        else:
            x = batch_tensor
            y = add_observation_noise(
                x=x,
                noise_type=args.noise_type,
                noise_param=args.noise_param,
            )

        batch_size = y.size(0)
        noisy_mse = F.mse_loss(y, x).item() if x is not None else None
        if x is not None:
            noisy_mse_sum += noisy_mse * batch_size

        if args.save_output and saved_count < args.save_output_limit:
            remain = args.save_output_limit - saved_count
            take = min(remain, batch_size)

            if x is not None:
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
                image_shape=blind_quality_image_shape,
                tv_weight=args.tv_weight,
                range_weight=args.range_weight,
                data_weight=args.data_weight,
            ).item()

            denoised_mse = F.mse_loss(x_hat, x).item() if x is not None else None

            cos = None
            if x is not None:
                cos = F.cosine_similarity(
                    score.flatten(1),
                    (x - y).flatten(1),
                    dim=1,
                    eps=1e-8,
                ).mean().item()

            quality_sums[idx] += q * batch_size
            if x is not None:
                denoised_mse_sums[idx] += denoised_mse * batch_size
                cos_sums[idx] += cos * batch_size

        total_count += batch_size

    has_clean = args.noisy_data is None
    noisy_mse = noisy_mse_sum / total_count if has_clean else None

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
        avg_denoised_mse = denoised_mse_sums[idx] / total_count if has_clean else None
        avg_cos = cos_sums[idx] / total_count if has_clean else None

        candidate_history.append(
            {
                "noise_param": param_value,
                "smoothing": smoothing_value,
                "score_sigma": float(score_sigma),
                "quality": avg_quality,
                "denoised_mse": avg_denoised_mse,
                "denoised_psnr": psnr_from_mse(avg_denoised_mse) if avg_denoised_mse is not None else None,
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
        "input_mode": "noisy" if args.noisy_data is not None else "synthetic",
        "num_samples": num_samples,
        "clean_dtype": clean_dtype,

        "estimated_noise_param": best["noise_param"],
        "estimated_smoothing": best["smoothing"],
        "estimated_score_sigma": best["score_sigma"],
        "best_quality": best["quality"],

        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
        "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
        "score_clean_direction_cos": score_cos,
        "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
        "requested_output_dir": requested_output_dir,
        "output_dir": output_dir,
        "checkpoint": Path(args.checkpoint),
        "clean_data": Path(args.clean_data) if args.clean_data is not None else None,
        "noisy_data": Path(args.noisy_data) if args.noisy_data is not None else None,

        "candidate_history": candidate_history,
    }

    save_config(output_dir / Path("candidate_history.json"), candidate_history)

    if args.stitch_output:
        summary["stitched"] = stitch_blind_outputs(
            args=args,
            n2s=n2s,
            backbone=backbone,
            candidate_jobs=candidate_jobs,
            output_dir=output_dir,
            device=device,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    save_config(output_dir / Path("summary.json"), summary)

    
    if args.copy_info:
        info_sources = {
            "checkpoint_": Path(args.checkpoint).parent,
            "data_": info_dir_for(eval_data_path),
        }

        for prefix, info_dir in info_sources.items():
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=False)
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=True)
    
    if args.save_output:
        save_output_dir = output_dir / Path("output")
        save_output_dir.mkdir(parents=True, exist_ok=True)

        clean_tensor = torch.cat(save_clean, dim=0) if save_clean else None
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

        clean_np = clean_tensor.numpy() if clean_tensor is not None else None
        noisy_np = noisy_tensor.numpy()
        denoised_np = denoised.detach().cpu().numpy()
        score_np = score.detach().cpu().numpy()

        if clean_np is not None:
            np.save(save_output_dir / "clean.npy", clean_np)
        np.save(save_output_dir / "noisy.npy", noisy_np)
        np.save(save_output_dir / "denoised.npy", denoised_np)
        np.save(save_output_dir / "score.npy", score_np)

        print(f"[saved] outputs saved to {save_output_dir}")


if __name__ == "__main__":
    main()
