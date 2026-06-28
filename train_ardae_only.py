import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


SCORE_METRIC_KEYS = [
    "score_mse",
    "score_nmse",
    "score_cos",
    "score_corr",
    "target_energy",
    "pred_energy",
]


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

from data import load_array, make_ardae_dataloaders
from models.ardae import ARDAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARDAE only.")

    parser.add_argument("--data", type=str, required=True, help="Path to csv/txt/tsv/npy/npz/pt/pth data file.")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--h-dim", type=int, default=1000)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--nonlinearity", type=str, default="tanh")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma"])
    parser.add_argument("--noise-param", type=float, default=0.1)

    parser.add_argument("--save-dir", type=str, default="checkpoints/ardae")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoints. 0 disables it.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--use-metric", action="store_true", help="Log score NMSE/cosine/correlation metrics.")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    return batch.to(device, non_blocking=True)

def update_metric_sums(metric_sums, metric_counts, metrics, batch_size):
    if not metrics:
        return

    for key in SCORE_METRIC_KEYS:
        if key not in metrics:
            continue

        value = float(metrics[key])
        if not np.isfinite(value):
            continue

        metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
        metric_counts[key] = metric_counts.get(key, 0) + batch_size


def average_metric_sums(metric_sums, metric_counts):
    averaged = {}
    for key, total in metric_sums.items():
        count = metric_counts.get(key, 0)
        if count > 0:
            averaged[key] = total / count
    return averaged


def train_one_epoch(model, loader, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    total_count = 0

    metric_sums = {}
    metric_counts = {}

    progress = tqdm(loader, desc=f"train {epoch:04d}" if epoch is not None else "train", leave=False)

    for batch in progress:
        x = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x)
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
def evaluate(model, loader, device, epoch=None):
    model.eval()
    total_loss = 0.0
    total_count = 0

    metric_sums = {}
    metric_counts = {}

    progress = tqdm(loader, desc=f"valid {epoch:04d}" if epoch is not None else "valid", leave=False)

    for batch in progress:
        x = move_batch(batch, device)
        _, loss = model(x)

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


def make_unique_save_dir(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir

    for index in range(1, 10000):
        candidate = base_dir.with_name(f"{base_dir.name}_{index:03d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused save directory for {base_dir}")


def log_message(message, log_path):
    tqdm.write(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_config(path, args):
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


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


def main():
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")

    device = torch.device(args.device)
    requested_save_dir = Path(args.save_dir)
    save_dir = make_unique_save_dir(requested_save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    args.requested_save_dir = str(requested_save_dir)
    args.save_dir = str(save_dir)
    args.run_started_at = datetime.now().isoformat(timespec="seconds")

    log_path = save_dir / "train.log"
    metrics_path = save_dir / "metrics.csv"
    save_config(save_dir / "config.json", args)

    log_message(f"save_dir: {save_dir}", log_path)
    if save_dir != requested_save_dir:
        log_message(f"requested_save_dir already existed; using: {save_dir}", log_path)
    log_message(f"device: {device}", log_path)

    raw_data = load_array(args.data, key=args.key)
    log_message(f"data: {args.data}", log_path)
    log_message(f"raw_data_shape: {tuple(raw_data.shape)}", log_path)
    train_loader, val_loader = make_ardae_dataloaders(
        data=raw_data,
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        normalize=args.normalize,
        flatten=not args.no_flatten,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ARDAE(
        input_dim=args.input_dim,
        h_dim=args.h_dim,
        noise_param=args.noise_param,
        num_hidden_layers=args.num_hidden_layers,
        nonlinearity=args.nonlinearity,
        noise_type=args.noise_type,
        use_metric=args.use_metric,
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
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)

        if len(val_loader.dataset) > 0:
            val_loss, val_metrics = evaluate(model, val_loader, device, epoch=epoch)
        else:
            val_loss = train_loss
            val_metrics = train_metrics

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint_path = save_dir / f"best_epoch_{epoch:04d}.pt"
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, train_loss, val_loss, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, train_loss, val_loss, args)

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
    log_message(f"saved last checkpoint: {save_dir / 'last.pt'}", log_path)
    log_message(f"saved best checkpoint: {save_dir / 'best.pt'}", log_path)
    log_message(f"saved metrics: {metrics_path}", log_path)


if __name__ == "__main__":
    main()
