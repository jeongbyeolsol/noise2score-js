# run_noise2score.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from argparse import Namespace
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
from scripts.test_ardae import (
    build_model as build_ardae_test_model,
    evaluate as evaluate_ardae,
    run_test,
    write_metrics_csv as write_ardae_metrics_csv,
)
from scripts.train_ardae import run_training
from utils import (
    add_observation_noise,
    config_all_from_to,
    make_unique_save_dir,
    move_batch,
    normalize_image_shape_arg,
    psnr_from_mse,
    save_config,
    set_seed,
)
from utils.checkpoint import load_ardae_from_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--clean-data", type=str, default=None)
    parser.add_argument(
        "--noisy-data",
        type=str,
        default=None,
        help="Path to already-noisy data/image(s). If set, Noise2Score denoises without adding synthetic noise.",
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
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--stitch-output",
        action="store_true",
        help="For --data-mode image-folder, denoise patches and stitch them back into full images.",
    )
    parser.add_argument(
        "--stitch-format",
        type=str,
        default="npy",
        choices=["npy", "png", "both"],
        help="Format for stitched full-image outputs.",
    )
    
    parser.add_argument('--copy-info', action="store_true")

    parser.add_argument("--train-ardae", action="store_true", help="Train ARDAE before running Noise2Score.")
    parser.add_argument("--test-ardae", action="store_true", help="Run ARDAE test before Noise2Score evaluation.")
    parser.add_argument("--ardae-train-data", type=str, default=None, help="Training data for ARDAE. Defaults to --clean-data.")
    parser.add_argument("--ardae-train-data-mode", type=str, default=None, choices=["array", "image-folder"])
    parser.add_argument("--ardae-test-data", type=str, default=None, help="Test data for ARDAE. Defaults to --clean-data.")
    parser.add_argument("--ardae-test-data-mode", type=str, default=None, choices=["array", "image-folder"])
    parser.add_argument("--ardae-max-patches-per-image", type=int, default=None)
    parser.add_argument("--ardae-save-dir", type=str, default="checkpoints/ardae")
    parser.add_argument("--ardae-test-output-dir", type=str, default=None)
    parser.add_argument("--ardae-use-metric", action="store_true")
    parser.add_argument("--ardae-save-every-best", action="store_true")
    parser.add_argument("--ardae-epochs", type=int, default=100)
    parser.add_argument("--ardae-batch-size", type=int, default=None)
    parser.add_argument("--ardae-lr", type=float, default=1e-3)
    parser.add_argument("--ardae-weight-decay", type=float, default=0.0)
    parser.add_argument("--ardae-val-ratio", type=float, default=0.1)
    parser.add_argument("--ardae-log-every", type=int, default=1)
    parser.add_argument("--ardae-normalize", type=str, default=None, choices=["standard", "minmax", "zero_one"])
    parser.add_argument("--ardae-no-flatten", action="store_true")
    parser.add_argument("--ardae-backbone", type=str, default="mlp", choices=["mlp", "unet"])
    parser.add_argument("--ardae-h-dim", type=int, default=1000)
    parser.add_argument("--ardae-num-hidden-layers", type=int, default=1)
    parser.add_argument("--ardae-nonlinearity", type=str, default="tanh")
    parser.add_argument("--ardae-base-channels", type=int, default=64)
    parser.add_argument("--ardae-channel-mults", type=str, default="1,2,4,8")
    parser.add_argument("--ardae-no-norm", action="store_true")
    parser.add_argument("--ardae-patch-loader", type=str, default="stream", choices=["stream", "map"])
    parser.add_argument("--ardae-sigma-min", type=float, default=0.001)
    parser.add_argument("--ardae-sigma-max", type=float, default=0.5)
    parser.add_argument("--ardae-linear-sigma", action="store_true")
    parser.add_argument(
        "--ardae-smoothing",
        nargs="?",
        const="range",
        default=None,
        help="ARDAE training smoothing. Use without value for range, or pass a fixed sigma.",
    )
    parser.add_argument("--ardae-test-max-batches", type=int, default=0)
    parser.add_argument("--ardae-test-save-samples", type=int, default=0)
    parser.add_argument("--ardae-test-no-metric", action="store_true")
    return parser.parse_args()


def build_ardae_train_args(args):
    if args.ardae_train_data is None and args.clean_data is None:
        raise ValueError("--train-ardae needs --ardae-train-data or --clean-data.")

    return Namespace(
        data=args.ardae_train_data or args.clean_data,
        data_mode=args.ardae_train_data_mode or args.data_mode,
        key=args.key,
        input_dim=args.input_dim,
        normalize=args.ardae_normalize,
        no_flatten=args.ardae_no_flatten,
        epochs=args.ardae_epochs,
        batch_size=args.ardae_batch_size or args.batch_size,
        lr=args.ardae_lr,
        weight_decay=args.ardae_weight_decay,
        val_ratio=args.ardae_val_ratio,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        seed=args.seed,
        device=args.device,
        h_dim=args.ardae_h_dim,
        num_hidden_layers=args.ardae_num_hidden_layers,
        nonlinearity=args.ardae_nonlinearity,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        save_dir=args.ardae_save_dir,
        save_every_best=args.ardae_save_every_best,
        log_every=args.ardae_log_every,
        use_metric=args.ardae_use_metric,
        backbone=args.ardae_backbone,
        image_shape=args.image_shape,
        patch_size=args.patch_size,
        stride=args.stride,
        channels=args.channels,
        max_patches_per_image=(
            args.ardae_max_patches_per_image
            if args.ardae_max_patches_per_image is not None
            else args.max_patches_per_image
        ),
        recursive_images=args.recursive_images,
        patch_loader=args.ardae_patch_loader,
        base_channels=args.ardae_base_channels,
        channel_mults=args.ardae_channel_mults,
        no_norm=args.ardae_no_norm,
        sigma_min=args.ardae_sigma_min,
        sigma_max=args.ardae_sigma_max,
        linear_sigma=args.ardae_linear_sigma,
        smoothing=args.ardae_smoothing,
    )


def build_ardae_test_args(args, checkpoint):
    if args.ardae_test_data is None and args.clean_data is None:
        raise ValueError("--test-ardae needs --ardae-test-data or --clean-data.")

    return Namespace(
        checkpoint=str(checkpoint),
        data=args.ardae_test_data or args.clean_data,
        key=args.key,
        input_dim=args.input_dim,
        normalize=args.ardae_normalize,
        no_flatten=args.ardae_no_flatten,
        batch_size=args.ardae_batch_size or args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.ardae_test_max_batches,
        seed=args.seed,
        device=args.device,
        noise_param=args.noise_param,
        noise_type=args.noise_type,
        no_metric=args.ardae_test_no_metric,
        output_dir=args.ardae_test_output_dir,
        save_samples=args.ardae_test_save_samples,
    )


def infer_path_data_mode(path, fallback="array"):
    if path is None:
        return fallback
    path = Path(path)
    if path.is_dir():
        return "image-folder"
    return fallback


def normalize_data_modes(args):
    eval_data_path = args.noisy_data or args.clean_data
    if args.data_mode == "array":
        args.data_mode = infer_path_data_mode(eval_data_path, fallback=args.data_mode)

    if args.ardae_train_data_mode is None:
        args.ardae_train_data_mode = infer_path_data_mode(
            args.ardae_train_data or args.clean_data,
            fallback=args.data_mode,
        )

    if args.ardae_test_data_mode is None:
        args.ardae_test_data_mode = infer_path_data_mode(
            args.ardae_test_data or args.clean_data,
            fallback=args.data_mode,
        )


def run_ardae_image_folder_test(args, checkpoint):
    test_args = build_ardae_test_args(args, checkpoint)
    set_seed(test_args.seed)

    device = torch.device(test_args.device)
    checkpoint_path = Path(test_args.checkpoint)
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, model_config, ckpt_args = build_ardae_test_model(
        checkpoint_obj,
        test_args,
        device,
    )

    image_shape = normalize_image_shape_arg(args.image_shape)
    if image_shape is None:
        image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
    if image_shape is None:
        if args.patch_size is None:
            raise ValueError(
                "--test-ardae with image-folder data requires --image-shape or --patch-size."
            )
        image_shape = (args.channels, args.patch_size, args.patch_size)

    channels, height, width = image_shape
    if args.patch_size is None:
        args.patch_size = height
    if args.stride is None:
        args.stride = args.patch_size
    args.channels = channels

    image_paths = find_image_paths(test_args.data, recursive=args.recursive_images)
    if len(image_paths) == 0:
        raise ValueError(f"No image/npy files found in {test_args.data}.")

    dataset = StreamingImagePatchDataset(
        image_paths=image_paths,
        patch_size=args.patch_size,
        stride=args.stride,
        channels=args.channels,
        max_patches_per_image=(
            args.ardae_max_patches_per_image
            if args.ardae_max_patches_per_image is not None
            else args.max_patches_per_image
        ),
        seed=test_args.seed,
        flatten=(model_config["backbone"] != "unet"),
        dtype=torch.float32,
        shuffle_images=False,
        shuffle_patches=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        num_workers=test_args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = test_args.output_dir
    if output_dir is None:
        output_dir = checkpoint_path.parent / "tests" / checkpoint_path.stem
    output_dir = make_unique_save_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    tqdm.write(f"checkpoint: {checkpoint_path}")
    tqdm.write(f"data: {test_args.data}")
    tqdm.write(f"num_images: {len(image_paths)}")
    tqdm.write(f"num_patches: {len(dataset)}")
    tqdm.write(f"device: {device}")
    tqdm.write(f"output_dir: {output_dir}")

    result = evaluate_ardae(
        model=model,
        loader=loader,
        device=device,
        max_batches=test_args.max_batches,
        save_samples=test_args.save_samples,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_obj.get("epoch", None),
        "data": test_args.data,
        "data_mode": "image-folder",
        "num_images": len(image_paths),
        "num_patches": len(dataset),
        "checkpoint_train_loss": checkpoint_obj.get("train_loss", None),
        "checkpoint_val_loss": checkpoint_obj.get("val_loss", None),
        "device": str(device),
        "model": model_config,
        "batch_size": test_args.batch_size,
        "max_batches": test_args.max_batches,
        "loss": result["loss"],
        "num_samples": result["num_samples"],
        "metrics": result["metrics"],
    }

    save_config(output_dir / "summary.json", summary)
    write_ardae_metrics_csv(output_dir / "metrics.csv", summary)

    if result["sample_x"] is not None:
        np.savez(
            output_dir / "samples.npz",
            x=result["sample_x"],
            score=result["sample_score"],
        )

    return {
        "output_dir": output_dir,
        "summary_path": output_dir / "summary.json",
        "metrics_path": output_dir / "metrics.csv",
        "summary": summary,
    }


def resolve_checkpoint(args):
    ardae_train_result = None
    ardae_test_result = None
    checkpoint = args.checkpoint

    if args.train_ardae:
        ardae_train_result = run_training(build_ardae_train_args(args))
        best_checkpoint = ardae_train_result["best_checkpoint_path"]
        checkpoint = best_checkpoint or ardae_train_result["last_checkpoint_path"]
        args.checkpoint = str(checkpoint)

    if checkpoint is None:
        raise ValueError("--checkpoint is required unless --train-ardae is enabled.")

    if args.test_ardae:
        test_data = args.ardae_test_data or args.clean_data
        test_data_mode = args.ardae_test_data_mode or infer_path_data_mode(
            test_data,
            fallback=args.data_mode,
        )
        if test_data_mode == "image-folder":
            ardae_test_result = run_ardae_image_folder_test(args, checkpoint)
        else:
            ardae_test_result = run_test(build_ardae_test_args(args, checkpoint))

    return Path(checkpoint), ardae_train_result, ardae_test_result


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


def make_stitch_coords(height, width, patch_size, stride):
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image size {(height, width)} is smaller than patch_size={patch_size}."
        )

    ys = list(range(0, height - patch_size + 1, stride))
    xs = list(range(0, width - patch_size + 1, stride))
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)

    return [(y, x) for y in ys for x in xs]


def load_clean_image_tensor(path, channels, dtype=torch.float32):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        if channels == 1:
            if image.ndim == 2:
                tensor = torch.as_tensor(image, dtype=dtype).unsqueeze(0)
            elif image.ndim == 3 and image.shape[-1] == 1:
                tensor = torch.as_tensor(image[..., 0], dtype=dtype).unsqueeze(0)
            elif image.ndim == 3 and image.shape[0] == 1:
                tensor = torch.as_tensor(image, dtype=dtype)
            else:
                raise ValueError(f"Unsupported grayscale npy shape {image.shape} from {path}")
        else:
            if image.ndim == 3 and image.shape[-1] == 3:
                tensor = torch.as_tensor(image, dtype=dtype).permute(2, 0, 1)
            elif image.ndim == 3 and image.shape[0] == 3:
                tensor = torch.as_tensor(image, dtype=dtype)
            else:
                raise ValueError(f"Unsupported RGB npy shape {image.shape} from {path}")

        if np.issubdtype(image.dtype, np.integer):
            tensor = tensor / 255.0
        return tensor.clamp(0, 1).contiguous()

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image stitching for non-npy files requires Pillow.") from exc

    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode)).copy()

    if channels == 1:
        tensor = torch.as_tensor(array, dtype=dtype).unsqueeze(0)
    else:
        tensor = torch.as_tensor(array, dtype=dtype).permute(2, 0, 1)

    return (tensor / 255.0).clamp(0, 1).contiguous()


def tensor_to_image_array(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() != 3:
        raise ValueError(f"Expected [C,H,W] tensor, got {tuple(tensor.shape)}")

    if tensor.size(0) == 1:
        array = tensor.squeeze(0).numpy()
    elif tensor.size(0) == 3:
        array = tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")

    return (array * 255.0).round().clip(0, 255).astype(np.uint8)


def save_stitched_tensor(tensor, path_base, output_format):
    path_base.parent.mkdir(parents=True, exist_ok=True)

    if output_format in {"npy", "both"}:
        np.save(path_base.with_suffix(".npy"), tensor.detach().cpu().numpy())

    if output_format in {"png", "both"}:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving stitched PNG outputs requires Pillow.") from exc

        Image.fromarray(tensor_to_image_array(tensor)).save(path_base.with_suffix(".png"))


def stitch_noise2score_outputs(args, n2s, backbone, output_dir, device):
    if args.data_mode != "image-folder":
        raise ValueError("--stitch-output requires --data-mode image-folder.")
    if args.patch_size is None or args.stride is None:
        raise ValueError("--stitch-output requires resolved --patch-size and --stride.")
    if args.max_patches_per_image is not None:
        raise ValueError("--stitch-output needs all patches; omit --max-patches-per-image.")

    image_paths = find_image_paths(args.noisy_data or args.clean_data, recursive=args.recursive_images)
    if len(image_paths) == 0:
        raise ValueError(f"No image/npy files found in {args.clean_data}.")

    stitch_dir = output_dir / "stitched"
    image_summaries = []
    noisy_mse_sum = 0.0
    denoised_mse_sum = 0.0
    pixel_count_sum = 0

    flatten = backbone != "unet"
    patch_size = int(args.patch_size)
    stride = int(args.stride)

    for image_index, image_path in enumerate(tqdm(image_paths, desc="Stitch images")):
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

        denoised_acc = torch.zeros_like(noisy)
        score_acc = torch.zeros_like(noisy)
        weight = torch.zeros((1, height, width), device=device, dtype=noisy.dtype)

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

            denoised = n2s.denoise(
                model_input,
                smoothing=args.smoothing,
                smoothing_samples=args.smoothing_samples,
            )
            score = n2s.score(
                model_input,
                smoothing=args.smoothing,
                smoothing_samples=args.smoothing_samples,
            )

            if flatten:
                denoised = denoised.view(-1, channels, patch_size, patch_size)
                score = score.view(-1, channels, patch_size, patch_size)

            for patch_index, (y, x) in enumerate(patch_coords):
                denoised_acc[:, y : y + patch_size, x : x + patch_size] += denoised[patch_index]
                score_acc[:, y : y + patch_size, x : x + patch_size] += score[patch_index]
                weight[:, y : y + patch_size, x : x + patch_size] += 1.0

        denoised_image = denoised_acc / weight.clamp_min(1.0)
        score_image = score_acc / weight.clamp_min(1.0)

        noisy_mse = None
        denoised_mse = None
        num_pixels = int(noisy.numel())
        pixel_count_sum += num_pixels
        if clean is not None:
            noisy_mse = F.mse_loss(noisy, clean).item()
            denoised_mse = F.mse_loss(denoised_image, clean).item()
            noisy_mse_sum += noisy_mse * num_pixels
            denoised_mse_sum += denoised_mse * num_pixels

        stem = f"{image_index:04d}_{Path(image_path).stem}"
        if clean is not None:
            save_stitched_tensor(clean, stitch_dir / "clean" / stem, args.stitch_format)
        save_stitched_tensor(noisy, stitch_dir / "noisy" / stem, args.stitch_format)
        save_stitched_tensor(denoised_image, stitch_dir / "denoised" / stem, args.stitch_format)
        save_stitched_tensor(score_image, stitch_dir / "score" / stem, "npy")

        image_summaries.append(
            {
                "image": str(image_path),
                "height": height,
                "width": width,
                "channels": channels,
                "num_patches": len(coords),
                "noisy_mse": noisy_mse,
                "denoised_mse": denoised_mse,
                "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
                "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
                "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
            }
        )

    has_clean = args.noisy_data is None
    noisy_mse = noisy_mse_sum / max(pixel_count_sum, 1) if has_clean else None
    denoised_mse = denoised_mse_sum / max(pixel_count_sum, 1) if has_clean else None
    summary = {
        "output_dir": stitch_dir,
        "num_images": len(image_paths),
        "num_pixels": pixel_count_sum,
        "patch_size": patch_size,
        "stride": stride,
        "format": args.stitch_format,
        "input_mode": "noisy" if args.noisy_data is not None else "synthetic",
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
        "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
        "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
        "images": image_summaries,
    }
    save_config(stitch_dir / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    if args.clean_data is None and args.noisy_data is None:
        raise ValueError("Provide --clean-data for synthetic-noise eval or --noisy-data for denoising noisy inputs.")
    normalize_data_modes(args)
    if args.smoothing < 0:
        raise ValueError("--smoothing must be >= 0.")
    if args.smoothing > 0 and args.smoothing_samples < 1:
        raise ValueError("--smoothing-samples must be >= 1 when --smoothing > 0.")
    if args.smoothing <= 0 and args.noise_type != "gaussian" and args.score_sigma is not None:
        raise ValueError(
            "--score-sigma should not be used for non-smoothed Poisson/Gamma runs. "
            "Omit it so ARDAE is queried with --noise-param, or enable --smoothing."
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path, ardae_train_result, ardae_test_result = resolve_checkpoint(args)
    requested_output_dir = Path(args.output_dir)
    output_dir = make_unique_save_dir(requested_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "run_config.json", args)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
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
        path=checkpoint_path,
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

        x_hat = n2s.denoise(
            y,
            smoothing=args.smoothing,
            smoothing_samples=args.smoothing_samples,
        )

        noisy_mse = F.mse_loss(y, x).item() if x is not None else None
        denoised_mse = F.mse_loss(x_hat, x).item() if x is not None else None

        score = n2s.score(
            y,
            smoothing=args.smoothing,
            smoothing_samples=args.smoothing_samples,
        )
        cos = None
        if x is not None:
            cos = F.cosine_similarity(
                score.flatten(1),
                (x - y).flatten(1),
                dim=1,
                eps=1e-8,
            ).mean().item()
        
        if args.save_output and saved_count < args.save_output_limit:
            remain = args.save_output_limit - saved_count
            take = min(remain, y.size(0))

            if x is not None:
                save_clean.append(x[:take].detach().cpu())
            save_noisy.append(y[:take].detach().cpu())
            save_denoised.append(x_hat[:take].detach().cpu())
            save_score.append(score[:take].detach().cpu())

            saved_count += take

        batch_size = y.size(0)
        total_count += batch_size
        if x is not None:
            noisy_mse_sum += noisy_mse * batch_size
            denoised_mse_sum += denoised_mse * batch_size
            cos_sum += cos * batch_size

    has_clean = args.noisy_data is None
    noisy_mse = noisy_mse_sum / total_count if has_clean else None
    denoised_mse = denoised_mse_sum / total_count if has_clean else None
    score_cos = cos_sum / total_count if has_clean else None

    summary = {
        "backbone": backbone,
        "image_shape": list(image_shape) if image_shape is not None else None,
        "noise_type": args.noise_type,
        "noise_param": args.noise_param,
        "smoothing": args.smoothing,
        "smoothing_samples": args.smoothing_samples,
        "score_sigma": args.score_sigma,
        "data_mode": args.data_mode,
        "input_mode": "noisy" if args.noisy_data is not None else "synthetic",
        "num_samples": num_samples,
        "clean_dtype": clean_dtype,
        "requested_output_dir": requested_output_dir,
        "output_dir": output_dir,
        "checkpoint": checkpoint_path,
        "clean_data": Path(args.clean_data) if args.clean_data is not None else None,
        "noisy_data": Path(args.noisy_data) if args.noisy_data is not None else None,
        "noisy_mse": noisy_mse,
        "denoised_mse": denoised_mse,
        "noisy_psnr": psnr_from_mse(noisy_mse) if noisy_mse is not None else None,
        "denoised_psnr": psnr_from_mse(denoised_mse) if denoised_mse is not None else None,
        "score_clean_direction_cos": score_cos,
        "improved_mse": denoised_mse < noisy_mse if denoised_mse is not None else None,
    }
    if ardae_train_result is not None:
        summary["ardae_train"] = {
            "save_dir": ardae_train_result["save_dir"],
            "best_checkpoint_path": ardae_train_result["best_checkpoint_path"],
            "last_checkpoint_path": ardae_train_result["last_checkpoint_path"],
            "metrics_path": ardae_train_result["metrics_path"],
            "best_val_loss": ardae_train_result["best_val_loss"],
            "best_epoch": ardae_train_result["best_epoch"],
        }
    if ardae_test_result is not None:
        summary["ardae_test"] = {
            "output_dir": ardae_test_result["output_dir"],
            "summary_path": ardae_test_result["summary_path"],
            "metrics_path": ardae_test_result["metrics_path"],
            "loss": ardae_test_result["summary"]["loss"],
            "num_samples": ardae_test_result["summary"]["num_samples"],
            "metrics": ardae_test_result["summary"]["metrics"],
        }

    if args.stitch_output:
        summary["stitched"] = stitch_noise2score_outputs(
            args=args,
            n2s=n2s,
            backbone=backbone,
            output_dir=output_dir,
            device=device,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    save_config(output_dir / "summary.json", summary)
    
    
    if args.copy_info:
        info_sources = {
            "checkpoint_": checkpoint_path.parent,
            "data_": info_dir_for(eval_data_path),
        }

        for prefix, info_dir in info_sources.items():
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=False)
            config_all_from_to(info_dir, output_dir, prefix=prefix, is_csv=True)

    
    if args.save_output:
        save_output_dir = output_dir / Path('output')
        save_output_dir.mkdir(parents=True, exist_ok=True)

        clean_np = torch.cat(save_clean, dim=0).numpy() if save_clean else None
        noisy_np = torch.cat(save_noisy, dim=0).numpy()
        denoised_np = torch.cat(save_denoised, dim=0).numpy()
        score_np = torch.cat(save_score, dim=0).numpy()

        if clean_np is not None:
            np.save(save_output_dir / "clean.npy", clean_np)
        np.save(save_output_dir / "noisy.npy", noisy_np)
        np.save(save_output_dir / "denoised.npy", denoised_np)
        np.save(save_output_dir / "score.npy", score_np)

        print(f"[saved] outputs saved to {save_output_dir}")


if __name__ == "__main__":
    main()
