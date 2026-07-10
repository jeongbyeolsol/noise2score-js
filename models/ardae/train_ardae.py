import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from datetime import datetime

import torch

from config import ARDAEConfig
from utils import (
    SCORE_METRIC_KEYS,
    average_metric_sums,
    config_all_from_to,
    log_message,
    make_loader_kwargs,
    make_noise_param,
    make_unique_save_dir,
    move_batch,
    normalize_image_shape_arg,
    parse_channel_mults,
    save_config,
    set_seed,
    update_metric_sums,
)


try:
    from tqdm.auto import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

    tqdm.write = print

from data import (
    find_image_paths,
    load_array,
    make_ardae_dataloaders,
    make_image_patch_dataloaders,
    make_streaming_image_patch_dataloaders,
)
from models.ardae import ARDAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARDAE only.")

    parser.add_argument("--data", type=str, required=True, help="Path to array file, image folder, image file, or image list.")
    parser.add_argument("--data-mode", type=str, default="array", choices=["array", "image-folder"], help="Use array data or lazy image patch loading.")
    parser.add_argument("--key", type=str, default=None, help="Key for npz or dict-style pt/pth files.")
    parser.add_argument("--input-dim", type=int, required=True, help="Feature dimension expected by ARDAE.")
    parser.add_argument("--normalize", type=str, default=None, choices=["standard", "minmax", "zero_one"])
    parser.add_argument("--no-flatten", action="store_true", help="Keep non-batch dimensions instead of flattening first.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--h-dim", type=int, default=1000)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--nonlinearity", type=str, default="tanh")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma"])
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

    parser.add_argument("--save-dir", type=str, default="checkpoints/ardae")
    parser.add_argument("--save-every-best", action="store_true", help="Save periodic checkpoints.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--use-metric", action="store_true", help="Log score NMSE/cosine/correlation metrics.")

    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "unet"], help="Score network backbone.")
    parser.add_argument("--image-shape", type=int, nargs="+", default=None, help="UNet image shape: C H W or H W. Example: --image-shape 1 40 40")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size for --data-mode image-folder.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride for --data-mode image-folder.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Image channels for lazy image patch loading.")
    parser.add_argument("--max-patches-per-image", type=int, default=None, help="Optional deterministic patch subsample per image.")
    parser.add_argument("--recursive-images", action="store_true", help="Find images recursively under --data when using image-folder mode.")
    parser.add_argument("--patch-loader", type=str, default="stream", choices=["stream", "map"], help="stream opens each image once and yields many patches; map keeps random patch access.")
    parser.add_argument("--base-channels", type=int, default=64, help="UNet base channel count.")
    parser.add_argument("--channel-mults", type=str, default="1,2,4,8", help="Comma-separated UNet channel multipliers.")
    parser.add_argument("--no-norm", action="store_true", help="Disable GroupNorm in UNet blocks.")

    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.001,
        help=(
            "Minimum ARDAE training noise level. With --gaussian-perturbation this is Gaussian "
            "perturbation sigma; otherwise it is the distribution parameter "
            "(poisson peak, not lam)."
        ),
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.5,
        help=(
            "Maximum ARDAE training noise level. With --gaussian-perturbation this is Gaussian "
            "perturbation sigma; otherwise it is the distribution parameter "
            "(poisson peak, not lam)."
        ),
    )
    parser.add_argument("--linear-sigma", action="store_true", help="Sample noise levels uniformly in linear scale instead of log scale.")
    parser.add_argument(
        "--gaussian-perturbation",
        "--smoothing",
        dest="smoothing",
        nargs="?",
        const="range",
        default=None,
        help=(
            "Train ARDAE with Gaussian perturbation noise. "
            "Use without a value to sample sigma from --sigma-min/--sigma-max, "
            "or pass a positive value for fixed sigma. --smoothing is kept as a deprecated alias."
        ),
    )

    args = parser.parse_args()
    return prepare_args(args)


def prepare_args(args):
    if not hasattr(args, "poisson_peak"):
        args.poisson_peak = None
    if args.poisson_peak is not None:
        if args.noise_type != "poisson":
            raise ValueError("--poisson-peak can only be used with --noise-type poisson.")
        args.noise_param = float(args.poisson_peak)

    if args.noise_type == "poisson":
        if args.noise_param <= 0:
            raise ValueError("Poisson peak must be positive.")
        args.poisson_peak = float(args.noise_param)
        args.poisson_lam = 1.0 / float(args.noise_param)
    else:
        args.poisson_lam = None

    args.use_gaussian_smoothing = args.smoothing is not None
    args.smoothing_sigma = None

    if args.smoothing not in (None, "range"):
        args.smoothing_sigma = float(args.smoothing)
        if args.smoothing_sigma <= 0:
            raise ValueError("--gaussian-perturbation value must be positive.")

    return args

def make_config(args):
    config = ARDAEConfig()
    config.sigma_min = args.sigma_min
    config.sigma_max = args.sigma_max
    config.use_log_scale = not args.linear_sigma
    config.use_gaussian_smoothing = args.use_gaussian_smoothing

    if args.smoothing_sigma is not None:
        config.sigma_min = args.smoothing_sigma
        config.sigma_max = args.smoothing_sigma
        config.use_log_scale = False

    return config


def infer_unet_image_shape(raw_data, input_dim, image_shape_arg):
    image_shape = normalize_image_shape_arg(image_shape_arg)
    if image_shape is not None:
        return image_shape

    if raw_data.ndim == 4:
        # Prefer NCHW. If data is NHWC with a small channel count, convert in dataset preprocessing.
        if raw_data.shape[1] in (1, 3):
            return tuple(int(v) for v in raw_data.shape[1:])
        if raw_data.shape[-1] in (1, 3):
            return (int(raw_data.shape[-1]), int(raw_data.shape[1]), int(raw_data.shape[2]))

    if raw_data.ndim == 3:
        return (1, int(raw_data.shape[1]), int(raw_data.shape[2]))

    if input_dim is not None:
        side = int(round(float(input_dim) ** 0.5))
        if side * side == int(input_dim):
            return (1, side, side)

    raise ValueError(
        "Could not infer UNet image shape. Pass --image-shape C H W, "
        "e.g. --image-shape 1 40 40."
    )


def infer_image_folder_shape(args):
    patch_size = args.patch_size
    if patch_size is None:
        image_shape = normalize_image_shape_arg(args.image_shape)
        if image_shape is None:
            raise ValueError(
                "--data-mode image-folder requires --patch-size or --image-shape."
            )
        channels, height, width = image_shape
        if height != width:
            raise ValueError("Lazy patch loading currently requires square patches.")
        patch_size = height
        args.patch_size = patch_size
        args.channels = channels

    if args.stride is None:
        args.stride = patch_size

    image_shape = normalize_image_shape_arg(args.image_shape)
    if image_shape is None:
        image_shape = (args.channels, patch_size, patch_size)
    else:
        channels, height, width = image_shape
        if height != patch_size or width != patch_size:
            raise ValueError(
                f"--image-shape {image_shape} does not match --patch-size {patch_size}."
            )
        args.channels = channels

    args.input_dim = int(args.channels * patch_size * patch_size)
    return image_shape


def train_one_epoch(model, loader, optimizer, device, config: ARDAEConfig, epoch=None):
    model.train()
    total_loss = 0.0
    total_count = 0

    metric_sums = {}
    metric_counts = {}

    progress = tqdm(loader, desc=f"train {epoch:04d}" if epoch is not None else "train", leave=False)

    for batch in progress:
        x = move_batch(batch, device)
        
        noise_param = None
        if config.sigma_min is not None and config.sigma_max is not None:
            noise_param = make_noise_param(
                x,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                use_log_scale=config.use_log_scale,
            )
            

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, noise_param)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        if getattr(model, "use_metric", False):
            update_metric_sums(
                metric_sums=metric_sums,
                metric_counts=metric_counts,
                metrics=getattr(model, "last_metrics", {}),
                batch_size=batch_size,
            )

        avg_loss = total_loss / max(total_count, 1)
        avg_metrics = average_metric_sums(metric_sums, metric_counts)

        postfix = {"loss": avg_loss}
        if "score_nmse" in avg_metrics:
            postfix["nmse"] = avg_metrics["score_nmse"]
        if "score_cos" in avg_metrics:
            postfix["cos"] = avg_metrics["score_cos"]

        progress.set_postfix(**postfix)

    return total_loss / max(total_count, 1), average_metric_sums(metric_sums, metric_counts)


@torch.no_grad()
def evaluate(model, loader, device, config: ARDAEConfig, epoch=None):
    model.eval()
    total_loss = 0.0
    total_count = 0

    metric_sums = {}
    metric_counts = {}

    progress = tqdm(loader, desc=f"valid {epoch:04d}" if epoch is not None else "valid", leave=False)

    for batch in progress:
            
        x = move_batch(batch, device)
        noise_param = None
        if config.sigma_min is not None and config.sigma_max is not None:
            noise_param = make_noise_param(
                x,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                use_log_scale=config.use_log_scale,
            )
        _, loss = model(x, noise_param)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        
        if getattr(model, "use_metric", False):
            update_metric_sums(
                metric_sums=metric_sums,
                metric_counts=metric_counts,
                metrics=getattr(model, "last_metrics", {}),
                batch_size=batch_size,
            )

        avg_loss = total_loss / max(total_count, 1)
        avg_metrics = average_metric_sums(metric_sums, metric_counts)

        postfix = {"loss": avg_loss}
        if "score_nmse" in avg_metrics:
            postfix["nmse"] = avg_metrics["score_nmse"]
        if "score_cos" in avg_metrics:
            postfix["cos"] = avg_metrics["score_cos"]

        progress.set_postfix(**postfix)

    return total_loss / max(total_count, 1), average_metric_sums(metric_sums, metric_counts)

def save_checkpoint(path, model, optimizer, epoch, train_loss, val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )

def run_training(args):
    args = prepare_args(args)
    config = make_config(args)
    set_seed(args.seed)

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.data_mode == "image-folder" and args.normalize is not None:
        raise ValueError("--normalize is not supported with lazy image-folder loading.")

    device = torch.device(args.device)
    loader_kwargs = make_loader_kwargs(args, device)
    requested_save_dir = Path(args.save_dir)
    save_dir = make_unique_save_dir(requested_save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    args.requested_save_dir = str(requested_save_dir)
    args.save_dir = str(save_dir)
    args.run_started_at = datetime.now().isoformat(timespec="seconds")

    log_path = save_dir / "train.log"
    metrics_path = save_dir / "metrics.csv"
    # image_shape/channel_mults는 raw_data를 본 뒤 확정되므로 아래에서 한 번 더 저장한다.
    save_config(save_dir / "model_config.json", args)

    log_message(f"save_dir: {save_dir}", log_path)
    if save_dir != requested_save_dir:
        log_message(f"requested_save_dir already existed; using: {save_dir}", log_path)
    log_message(f"device: {device}", log_path)

    log_message(f"data: {args.data}", log_path)

    channel_mults = parse_channel_mults(args.channel_mults)
    image_shape = None
    flatten = not args.no_flatten
    if args.data_mode == "image-folder":
        image_paths = find_image_paths(args.data, recursive=args.recursive_images)
        has_npy_paths = any(Path(path).suffix.lower() == ".npy" for path in image_paths)
        if has_npy_paths and args.patch_loader != "stream":
            raise ValueError(
                ".npy image folders require --patch-loader stream. "
                "The map loader is only for PIL-readable image files."
            )
        image_shape = infer_image_folder_shape(args)
        flatten = args.backbone != "unet"
        args.image_shape = list(image_shape)
        args.channel_mults = list(channel_mults)
        args.num_images = len(image_paths)
        log_message(f"num_images: {len(image_paths)}", log_path)
        log_message(f"patch_size: {args.patch_size}", log_path)
        log_message(f"stride: {args.stride}", log_path)
        log_message(f"unet_image_shape: {image_shape}", log_path)

        if args.patch_loader == "stream":
            train_loader, val_loader, train_dataset, val_dataset = make_streaming_image_patch_dataloaders(
                image_paths=image_paths,
                patch_size=args.patch_size,
                stride=args.stride,
                channels=args.channels,
                max_patches_per_image=args.max_patches_per_image,
                input_dim=args.input_dim,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                flatten=flatten,
                **loader_kwargs,
            )
            args.num_train_patches = len(train_dataset)
            args.num_val_patches = len(val_dataset)
            args.num_patches = args.num_train_patches + args.num_val_patches
        else:
            train_loader, val_loader, patch_dataset = make_image_patch_dataloaders(
                image_paths=image_paths,
                patch_size=args.patch_size,
                stride=args.stride,
                channels=args.channels,
                max_patches_per_image=args.max_patches_per_image,
                input_dim=args.input_dim,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                flatten=flatten,
                **loader_kwargs,
            )
            args.num_patches = len(patch_dataset)
            args.num_train_patches = len(train_loader.dataset)
            args.num_val_patches = len(val_loader.dataset)

        log_message(f"patch_loader: {args.patch_loader}", log_path)
        log_message(f"num_patches: {args.num_patches}", log_path)
        log_message(f"num_train_patches: {args.num_train_patches}", log_path)
        log_message(f"num_val_patches: {args.num_val_patches}", log_path)

    else:
        raw_data = load_array(args.data, key=args.key)
        log_message(f"raw_data_shape: {tuple(raw_data.shape)}", log_path)

        if args.backbone == "unet":
            image_shape = infer_unet_image_shape(raw_data, args.input_dim, args.image_shape)
            flatten = False
            args.image_shape = list(image_shape)
            args.channel_mults = list(channel_mults)
            log_message(f"unet_image_shape: {image_shape}", log_path)

        train_loader, val_loader = make_ardae_dataloaders(
            data=raw_data,
            input_dim=args.input_dim,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            normalize=args.normalize,
            flatten=flatten,
            image_shape=image_shape,
            seed=args.seed,
            **loader_kwargs,
        )

        config_all_from_to(
            Path(args.data).parent,
            save_dir,
            prefix="data_",
        )

    save_config(save_dir / "model_config.json", args)

    model = ARDAE(
        input_dim=args.input_dim,
        h_dim=args.h_dim,
        noise_param=args.noise_param,
        num_hidden_layers=args.num_hidden_layers,
        nonlinearity=args.nonlinearity,
        noise_type=args.noise_type,
        use_metric=args.use_metric,
        backbone=args.backbone,
        image_shape=image_shape,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        use_norm=not args.no_norm,
        use_gaussian_smoothing=config.use_gaussian_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = None

    base_fieldnames = ["epoch", "train_loss", "val_loss", "best_val_loss", "best_epoch"]

    metric_fieldnames = []
    if args.use_metric:
        for prefix in ["train", "val"]:
            for key in SCORE_METRIC_KEYS:
                metric_fieldnames.append(f"{prefix}_{key}")

    fieldnames = base_fieldnames + metric_fieldnames

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs")

    for epoch in epoch_progress:
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch, config=config)

        if len(val_loader.dataset) > 0:
            val_loss, val_metrics = evaluate(model, val_loader, device, epoch=epoch, config=config)
        else:
            val_loss = train_loss
            val_metrics = train_metrics

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if args.save_every_best:
                best_checkpoint_path = save_dir / f"best_epoch_{epoch:04d}.pt"
            else:
                best_checkpoint_path = save_dir / "best_model.pt"
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, train_loss, val_loss, args)


        epoch_progress.set_postfix(
            train_loss=train_loss,
            val_loss=val_loss,
            best_val=best_val_loss,
            best_epoch=best_epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

        if args.use_metric:
            for key in SCORE_METRIC_KEYS:
                row[f"train_{key}"] = train_metrics.get(key, "")
                row[f"val_{key}"] = val_metrics.get(key, "")

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metric_text = ""
            if args.use_metric:
                metric_text = (
                    f" train_nmse={train_metrics.get('score_nmse', float('nan')):.6f}"
                    f" val_nmse={val_metrics.get('score_nmse', float('nan')):.6f}"
                    f" train_cos={train_metrics.get('score_cos', float('nan')):.6f}"
                    f" val_cos={val_metrics.get('score_cos', float('nan')):.6f}"
                )

            log_message(
                f"epoch {epoch:04d}/{args.epochs:04d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val={best_val_loss:.6f}@{best_epoch:04d}"
                f"{metric_text}",
                log_path,
            )

    last_checkpoint_path = save_dir / f"last_epoch_{args.epochs:04d}.pt"
    save_checkpoint(last_checkpoint_path, model, optimizer, args.epochs, train_loss, val_loss, args)
    args.last_checkpoint_path = str(last_checkpoint_path)
    args.best_checkpoint_path = str(best_checkpoint_path) if best_checkpoint_path is not None else None
    save_config(save_dir / "model_config.json", args)
    log_message(f"saved last checkpoint: {last_checkpoint_path}", log_path)
    log_message(f"saved best checkpoint: {best_checkpoint_path}", log_path)
    log_message(f"saved metrics: {metrics_path}", log_path)

    return {
        "save_dir": save_dir,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "metrics_path": metrics_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def main():
    return run_training(parse_args())


if __name__ == "__main__":
    main()
