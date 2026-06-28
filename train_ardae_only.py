import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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


def train_one_epoch(model, loader, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    total_count = 0

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
        progress.set_postfix(loss=total_loss / max(total_count, 1))

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, epoch=None):
    model.eval()
    total_loss = 0.0
    total_count = 0

    progress = tqdm(loader, desc=f"valid {epoch:04d}" if epoch is not None else "valid", leave=False)

    for batch in progress:
        x = move_batch(batch, device)
        _, loss = model(x)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        progress.set_postfix(loss=total_loss / max(total_count, 1))

    return total_loss / max(total_count, 1)


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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss", "best_epoch"],
        )
        writer.writeheader()

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs")

    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)
        val_loss = evaluate(model, val_loader, device, epoch=epoch) if len(val_loader.dataset) > 0 else train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, train_loss, val_loss, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, train_loss, val_loss, args)

        epoch_progress.set_postfix(
            train_loss=train_loss,
            val_loss=val_loss,
            best_val=best_val_loss,
            best_epoch=best_epoch,
        )

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss", "best_epoch"],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                }
            )

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            log_message(
                f"epoch {epoch:04d}/{args.epochs:04d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val={best_val_loss:.6f}@{best_epoch:04d}",
                log_path,
            )

    save_checkpoint(save_dir / "last.pt", model, optimizer, args.epochs, train_loss, val_loss, args)
    log_message(f"saved last checkpoint: {save_dir / 'last.pt'}", log_path)
    log_message(f"saved best checkpoint: {save_dir / 'best.pt'}", log_path)
    log_message(f"saved metrics: {metrics_path}", log_path)


if __name__ == "__main__":
    main()
