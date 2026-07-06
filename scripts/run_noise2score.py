# run_noise2score.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
<<<<<<< HEAD
import math
<<<<<<< HEAD:run_noise2score.py
from pathlib import Path
=======
>>>>>>> main:scripts/run_noise2score.py
=======
>>>>>>> noise2score-codex
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import (
    StreamingImagePatchDataset,
    find_image_paths,
    load_array,
    preprocess_ardae_data,
)
from models.noise2score import Noise2Score
from utils import (
    add_observation_noise,
    config_all_from_to,
    make_unique_save_dir,
    move_batch,
    normalize_image_shape_arg,
    psnr_from_mse,
    save_config,
)
from utils.checkpoint import load_ardae_from_checkpoint

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-shape", type=int, nargs="+", default=None, help="Override/restore UNet image shape: C H W or H W.")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size for --data-mode image-folder.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride for --data-mode image-folder.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Channels for streamed image/npy folder data.")
    parser.add_argument("--max-patches-per-image", type=int, default=None)
    parser.add_argument("--recursive-images", action="store_true")

    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma"])
    parser.add_argument("--noise-param", type=float, default=0.1)
    parser.add_argument("--score-sigma", type=float, default=None)
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

    parser.add_argument("--output-dir", type=str, default="results/noise2score")

    
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--save-output-limit", type=int, default=64)
    
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


@torch.no_grad()
def main():
    args = parse_args()
    if args.smoothing < 0:
        raise ValueError("--smoothing must be >= 0.")
    if args.smoothing > 0 and args.smoothing_samples < 1:
        raise ValueError("--smoothing-samples must be >= 1 when --smoothing > 0.")
    if args.smoothing <= 0 and args.noise_type != "gaussian" and args.score_sigma is not None:
        raise ValueError(
            "--score-sigma should not be used for non-smoothed Poisson/Gamma runs. "
            "Omit it so ARDAE is queried with --noise-param, or enable --smoothing."
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

    for batch in tqdm(loader, desc="Noise2Score eval", total=len(loader)):
        x = move_batch(batch, device)

        y = add_observation_noise(
            x=x,
            noise_type=args.noise_type,
            noise_param=args.noise_param,
        )

        x_hat = n2s.denoise(
            y,
            smoothing=args.smoothing,
            smoothing_samples=args.smoothing_samples,
        )

        noisy_mse = F.mse_loss(y, x).item()
        denoised_mse = F.mse_loss(x_hat, x).item()

        score = n2s.score(
            y,
            smoothing=args.smoothing,
            smoothing_samples=args.smoothing_samples,
        )
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
        "smoothing": args.smoothing,
        "smoothing_samples": args.smoothing_samples,
        "score_sigma": args.score_sigma,
        "data_mode": args.data_mode,
        "num_samples": num_samples,
        "clean_dtype": clean_dtype,
        "requested_output_dir": requested_output_dir,
        "output_dir": output_dir,
        "checkpoint": Path(args.checkpoint),
        "clean_data": Path(args.clean_data),
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse),
        "denoised_psnr": psnr_from_mse(denoised_mse),
        "score_clean_direction_cos": score_cos,
        "improved_mse": denoised_mse < noisy_mse,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    save_config(output_dir / "summary.json", summary)
    
    
    if args.copy_info:
        info_sources = {
            "checkpoint_": Path(args.checkpoint).parent,
            "data_": info_dir_for(args.clean_data),
        }

        for prefix, info_dir in info_sources.items():
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=False)
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=True)

    
    if args.save_output:
        save_output_dir = output_dir / Path('output')
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
