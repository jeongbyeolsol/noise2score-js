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
from tqdm import tqdm

from data import (
    find_image_paths,
    load_array,
)
from utils import (
    add_observation_noise,
    config_all_from_to,
    denoise_from_score,
    make_observation_noise_param,
    make_unique_save_dir,
    move_batch,
    psnr_from_mse,
    save_config,
    summarize_noise_param,
)
from utils.checkpoint import (
    get_ardae_checkpoint,
    get_checkpoint_args,
    is_noise2score_checkpoint,
    load_noise2score_from_checkpoint,
    save_noise2score_checkpoint,
)
from utils.evaluation import (
    infer_stream_shape,
    info_dir_for,
    make_eval_loader as make_clean_loader,
    normalize_eval_data_modes,
    normalize_noise_param_aliases,
    parse_float_list,
    quality_image_shape,
)
from utils.image_io import (
    load_clean_image_tensor,
    make_stitch_coords,
    save_stitched_tensor,
)


def build_parser():
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
    parser.add_argument(
        "--noise-param",
        type=float,
        default=0.1,
        help="Distribution noise parameter: gaussian std, poisson peak, or gamma concentration.",
    )
    parser.add_argument(
        "--poisson-peak",
        type=float,
        default=None,
        help="Alias for --noise-param when --noise-type poisson. Larger peak means weaker Poisson noise.",
    )
    parser.add_argument(
        "--noise-param-min",
        type=float,
        default=None,
        help="Min observation noise parameter for synthetic eval. Samples one value per sample.",
    )
    parser.add_argument(
        "--noise-param-max",
        type=float,
        default=None,
        help="Max observation noise parameter for synthetic eval. Samples one value per sample.",
    )
    parser.add_argument(
        "--linear-noise-param",
        action="store_true",
        help="Sample observation noise parameters uniformly in linear scale instead of log scale.",
    )
    parser.add_argument(
        "--score-smoothing",
        "--smoothing",
        dest="smoothing",
        type=float,
        default=0.0,
        help="Gaussian perturbation std for Monte Carlo score smoothing. 0 keeps the closed-form rule. --smoothing is a deprecated alias.",
    )
    parser.add_argument(
        "--score-smoothing-samples",
        "--smoothing-samples",
        dest="smoothing_samples",
        type=int,
        default=8,
        help="Monte Carlo samples used when score smoothing is enabled.",
    )

    # blind parameter candidates
    parser.add_argument(
        "--candidate-params",
        type=str,
        default=None,
        help="Comma-separated candidate noise parameters. For poisson these are peak values.",
    )
    parser.add_argument(
        "--candidate-score-smoothing",
        "--candidate-smoothing",
        dest="candidate_smoothing",
        type=str,
        default=None,
        help=(
            "Comma-separated Gaussian perturbation candidates for Monte Carlo score smoothing, "
            "non-Gaussian Noise2Score, e.g. 0.03,0.05,0.075,0.1,0.125,0.15,0.2. "
            "With --score-sigma-mode same, score_sigma follows each score-smoothing candidate."
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
    parser.add_argument(
        "--noise2score-checkpoint-output",
        type=str,
        default=None,
        help="Path for the integrated blind Noise2Score checkpoint. Defaults to output_dir/noise2score_blind_checkpoint.pt.",
    )

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
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def force_noise_type(args, forced_noise_type):
    if forced_noise_type is None:
        return
    if args.noise_type != forced_noise_type:
        tqdm.write(
            f"[noise2score-blind] forcing --noise-type {forced_noise_type} "
            f"(was {args.noise_type})"
        )
    args.noise_type = forced_noise_type


def apply_noise2score_checkpoint_config(args, checkpoint, forced_noise_type=None):
    if not is_noise2score_checkpoint(checkpoint):
        return

    n2s_config = checkpoint.get("noise2score", {})
    ckpt_noise_type = n2s_config.get("noise_type")
    if forced_noise_type is not None and ckpt_noise_type != forced_noise_type:
        raise ValueError(
            f"This entrypoint is for {forced_noise_type}, but checkpoint stores "
            f"noise_type={ckpt_noise_type}."
        )

    for name in (
        "noise_type",
        "noise_param",
        "noise_param_min",
        "noise_param_max",
        "linear_noise_param",
        "score_sigma",
    ):
        if name in n2s_config and n2s_config[name] is not None:
            setattr(args, name, n2s_config[name])
    if n2s_config.get("score_smoothing") is not None:
        args.smoothing = n2s_config["score_smoothing"]
    if n2s_config.get("score_smoothing_samples") is not None:
        args.smoothing_samples = n2s_config["score_smoothing_samples"]

    if args.noise_type == "poisson" and n2s_config.get("poisson_peak") is not None:
        args.poisson_peak = n2s_config["poisson_peak"]


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


def _noise_param_list(noise_param):
    if torch.is_tensor(noise_param):
        return [float(v) for v in noise_param.detach().cpu().view(-1)]
    if noise_param is None:
        return None
    return [float(noise_param)]


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
        raise ValueError("--candidate-score-smoothing is intended for non-Gaussian score smoothing.")

    smoothing_list = [float(v) for v in candidate_smoothing]
    if any(v <= 0.0 for v in smoothing_list):
        raise ValueError("--candidate-score-smoothing values must be > 0.")

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
    if args.save_output and args.save_output_limit > 0:
        image_paths = image_paths[: args.save_output_limit]

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
            image_noise_param = None
        else:
            clean = load_clean_image_tensor(image_path, channels=args.channels).to(device)
            image_noise_param = make_observation_noise_param(args, clean.unsqueeze(0))
            noisy = add_observation_noise(
                x=clean.unsqueeze(0),
                noise_type=args.noise_type,
                noise_param=image_noise_param,
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
                "true_noise_param": _noise_param_list(image_noise_param),
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
        "noise_param_min": args.noise_param_min,
        "noise_param_max": args.noise_param_max,
        "linear_noise_param": args.linear_noise_param,
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
        "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
        "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
        "images": image_summaries,
    }
    save_config(stitch_dir / "summary.json", summary)
    return summary


def main(forced_noise_type=None):
    run(parse_args(), forced_noise_type=forced_noise_type)


@torch.no_grad()
def run(args, forced_noise_type=None):
    force_noise_type(args, forced_noise_type)

    if args.clean_data is None and args.noisy_data is None:
        raise ValueError("Provide --clean-data for synthetic-noise eval or --noisy-data for denoising noisy inputs.")
    normalize_eval_data_modes(args)
    normalize_noise_param_aliases(args)
    if args.score_sigma_mode == "fixed" and args.fixed_score_sigma is None:
        raise ValueError("--score-sigma-mode fixed requires --fixed-score-sigma.")
    if args.smoothing < 0:
        raise ValueError("--score-smoothing must be >= 0.")
    smoothing_enabled = args.smoothing > 0 or args.candidate_smoothing is not None
    if smoothing_enabled and args.smoothing_samples < 1:
        raise ValueError(
            "--score-smoothing-samples must be >= 1 when score smoothing is enabled."
        )

    device = torch.device(args.device)
    requested_output_dir = Path(args.output_dir)
    output_dir = make_unique_save_dir(requested_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "run_config.json", args)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    apply_noise2score_checkpoint_config(args, ckpt, forced_noise_type=forced_noise_type)
    normalize_noise_param_aliases(args)
    save_config(output_dir / "run_config.json", args)
    ardae_ckpt = get_ardae_checkpoint(ckpt)
    ckpt_image_shape = get_checkpoint_args(ckpt).get("image_shape")
    raw_clean = None
    eval_data_path = args.noisy_data or args.clean_data
    eval_key = args.noisy_key if args.noisy_data is not None else args.key
    if args.data_mode == "image-folder":
        stream_image_shape = infer_stream_shape(args, ckpt_image_shape)
    else:
        stream_image_shape = args.image_shape
        raw_clean = load_array(eval_data_path, key=eval_key)

    n2s, backbone, image_shape, loaded_checkpoint = load_noise2score_from_checkpoint(
        path=args.checkpoint,
        input_dim=args.input_dim,
        device=device,
        clean=raw_clean,
        image_shape_override=stream_image_shape,
        args=args,
        blind=True,
    )

    loader, num_samples, clean_dtype = make_clean_loader(
        args,
        backbone,
        image_shape,
        device,
        raw_clean=raw_clean,
    )

    checkpoint_output = (
        Path(args.noise2score_checkpoint_output)
        if args.noise2score_checkpoint_output is not None
        else output_dir / "noise2score_blind_checkpoint.pt"
    )
    save_noise2score_checkpoint(
        checkpoint_output,
        n2s,
        ardae_checkpoint=ardae_ckpt,
        ardae_checkpoint_path=args.checkpoint,
        args=args,
        blind=True,
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
    stop_after_saved_outputs = args.save_output and args.save_output_limit > 0
    noise_param_stats = []

    pbar = tqdm(loader, desc="Blind Noise2Score eval", total=len(loader))

    for batch in pbar:
        if stop_after_saved_outputs and saved_count >= args.save_output_limit:
            break

        batch_tensor = move_batch(batch, device)
        if args.noisy_data is not None:
            x = None
            y = batch_tensor
            observation_noise_param = None
        else:
            x = batch_tensor
            observation_noise_param = make_observation_noise_param(args, x)
            y = add_observation_noise(
                x=x,
                noise_type=args.noise_type,
                noise_param=observation_noise_param,
            )

        if stop_after_saved_outputs:
            remain = args.save_output_limit - saved_count
            take = min(remain, y.size(0))
            y = y[:take]
            if x is not None:
                x = x[:take]
            if torch.is_tensor(observation_noise_param):
                observation_noise_param = observation_noise_param[:take]

        batch_size = y.size(0)
        batch_noise_stats = summarize_noise_param(observation_noise_param)
        if batch_noise_stats is not None:
            batch_noise_stats["count"] = batch_size
            noise_param_stats.append(batch_noise_stats)
        noisy_mse = F.mse_loss(y, x).item() if x is not None else None
        if x is not None:
            noisy_mse_sum += noisy_mse * batch_size

        if stop_after_saved_outputs:
            if x is not None:
                save_clean.append(x.detach().cpu())
            save_noisy.append(y.detach().cpu())

            saved_count += batch_size

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
    sampled_noise_summary = None
    if noise_param_stats:
        count = sum(item["count"] for item in noise_param_stats)
        sampled_noise_summary = {
            "min": min(item["min"] for item in noise_param_stats),
            "max": max(item["max"] for item in noise_param_stats),
            "mean": sum(item["mean"] * item["count"] for item in noise_param_stats) / max(count, 1),
        }

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
        "true_noise_param_min_for_eval": args.noise_param_min,
        "true_noise_param_max_for_eval": args.noise_param_max,
        "linear_noise_param_for_eval": args.linear_noise_param,
        "sampled_true_noise_param_for_eval": sampled_noise_summary,
        "score_smoothing": args.smoothing,
        "candidate_score_smoothing": smoothing_list,
        "score_smoothing_samples": args.smoothing_samples,
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
        "num_samples": total_count,
        "available_num_samples": num_samples,
        "stopped_after_save_output_limit": stop_after_saved_outputs,
        "clean_dtype": clean_dtype,

        "estimated_noise_param": best["noise_param"],
        "estimated_score_smoothing": best["smoothing"],
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
        "checkpoint_type": loaded_checkpoint.get("checkpoint_type", "ardae"),
        "noise2score_checkpoint": checkpoint_output,
        "clean_data": Path(args.clean_data) if args.clean_data is not None else None,
        "noisy_data": Path(args.noisy_data) if args.noisy_data is not None else None,

        "candidate_history": candidate_history,
    }
    if args.noise_type == "poisson":
        summary["true_poisson_peak_for_eval"] = args.poisson_peak
        summary["true_poisson_lam_for_eval"] = args.poisson_lam
        summary["true_poisson_peak_min_for_eval"] = args.poisson_peak_min
        summary["true_poisson_peak_max_for_eval"] = args.poisson_peak_max
        summary["true_poisson_lam_min_for_eval"] = args.poisson_lam_min
        summary["true_poisson_lam_max_for_eval"] = args.poisson_lam_max
        summary["estimated_poisson_peak"] = best["noise_param"]
        summary["estimated_poisson_lam"] = 1.0 / float(best["noise_param"])

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
