import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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

from data import load_array, make_ardae_dataset
from models.ardae import ARDAE


SCORE_METRIC_KEYS = [
    "score_mse",
    "score_nmse",
    "score_cos",
    "score_corr",
    "target_energy",
    "pred_energy",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test an ARDAE checkpoint.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to an ARDAE checkpoint .pt file.")
    parser.add_argument("--data", type=str, required=True, help="Path to csv/txt/tsv/npy/npz/pt/pth test data file.")
    parser.add_argument("--key", type=str, default=None, help="Key for npz or dict-style pt/pth data files.")

    parser.add_argument("--input-dim", type=int, default=None, help="Override checkpoint input_dim.")
    parser.add_argument("--normalize", type=str, default=None, choices=["standard", "minmax", "zero_one"])
    parser.add_argument("--no-flatten", action="store_true", help="Keep non-batch dimensions instead of flattening first.")

    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="Evaluate only this many batches. 0 means all batches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--noise-param", type=float, default=None, help="Override checkpoint noise_param.")
    parser.add_argument("--noise-type", type=str, default=None, choices=["gaussian", "poisson", "gamma"])
    parser.add_argument("--no-metric", action="store_true", help="Disable score metrics and report only loss.")

    parser.add_argument("--output-dir", type=str, default=None, help="Directory for summary.json and metrics.csv.")
    parser.add_argument("--save-samples", type=int, default=0, help="Save this many x/score samples to samples.npz.")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_unique_dir(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir

    for index in range(1, 10000):
        candidate = base_dir.with_name(f"{base_dir.name}_{index:03d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused output directory for {base_dir}")


def get_ckpt_arg(ckpt_args, key, default=None):
    if ckpt_args is None:
        return default
    return ckpt_args.get(key, default)


def build_model(checkpoint, cli_args, device):
    ckpt_args = checkpoint.get("args", {})

    input_dim = cli_args.input_dim or int(get_ckpt_arg(ckpt_args, "input_dim", 2))
    h_dim = int(get_ckpt_arg(ckpt_args, "h_dim", 1000))
    num_hidden_layers = int(get_ckpt_arg(ckpt_args, "num_hidden_layers", 1))
    nonlinearity = get_ckpt_arg(ckpt_args, "nonlinearity", "tanh")
    noise_type = cli_args.noise_type or get_ckpt_arg(ckpt_args, "noise_type", "gaussian")
    noise_param = cli_args.noise_param
    if noise_param is None:
        noise_param = float(get_ckpt_arg(ckpt_args, "noise_param", 0.1))

    model = ARDAE(
        input_dim=input_dim,
        h_dim=h_dim,
        noise_param=noise_param,
        num_hidden_layers=num_hidden_layers,
        nonlinearity=nonlinearity,
        noise_type=noise_type,
        use_metric=not cli_args.no_metric,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    model_config = {
        "input_dim": input_dim,
        "h_dim": h_dim,
        "num_hidden_layers": num_hidden_layers,
        "nonlinearity": nonlinearity,
        "noise_type": noise_type,
        "noise_param": noise_param,
        "use_metric": not cli_args.no_metric,
    }
    return model, model_config, ckpt_args


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


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, save_samples=0):
    total_loss = 0.0
    total_count = 0
    metric_sums = {}
    metric_counts = {}
    sample_x = []
    sample_score = []

    progress = tqdm(loader, desc="test", leave=False)

    for batch_index, batch in enumerate(progress, start=1):
        x = move_batch(batch, device)
        glogprob, loss = model(x)

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

        if save_samples > 0 and len(sample_x) < save_samples:
            remaining = save_samples - len(sample_x)
            take = min(remaining, batch_size)
            sample_x.extend(x[:take].detach().cpu().numpy())
            sample_score.extend(glogprob[:take].detach().cpu().numpy())

        avg_loss = total_loss / max(total_count, 1)
        avg_metrics = average_metric_sums(metric_sums, metric_counts)
        postfix = {"loss": avg_loss}
        if "score_nmse" in avg_metrics:
            postfix["nmse"] = avg_metrics["score_nmse"]
        if "score_cos" in avg_metrics:
            postfix["cos"] = avg_metrics["score_cos"]
        progress.set_postfix(**postfix)

        if max_batches > 0 and batch_index >= max_batches:
            break

    return {
        "loss": total_loss / max(total_count, 1),
        "num_samples": total_count,
        "metrics": average_metric_sums(metric_sums, metric_counts),
        "sample_x": np.asarray(sample_x, dtype=np.float32) if sample_x else None,
        "sample_score": np.asarray(sample_score, dtype=np.float32) if sample_score else None,
    }


def write_metrics_csv(path, summary):
    row = {
        "loss": summary["loss"],
        "num_samples": summary["num_samples"],
    }
    for key in SCORE_METRIC_KEYS:
        row[key] = summary["metrics"].get(key, "")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, model_config, ckpt_args = build_model(checkpoint, args, device)

    input_dim = model_config["input_dim"]
    normalize = args.normalize
    if normalize is None:
        normalize = get_ckpt_arg(ckpt_args, "normalize", None)

    flatten = not args.no_flatten
    if not args.no_flatten:
        flatten = not bool(get_ckpt_arg(ckpt_args, "no_flatten", False))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = checkpoint_path.parent / "tests" / checkpoint_path.stem
    output_dir = make_unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_data = load_array(args.data, key=args.key)
    dataset = make_ardae_dataset(
        raw_data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    tqdm.write(f"checkpoint: {checkpoint_path}")
    tqdm.write(f"data: {args.data}")
    tqdm.write(f"raw_data_shape: {tuple(raw_data.shape)}")
    tqdm.write(f"test_tensor_shape: {tuple(dataset.x.shape)}")
    tqdm.write(f"device: {device}")
    tqdm.write(f"output_dir: {output_dir}")

    result = evaluate(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        save_samples=args.save_samples,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "data": args.data,
        "raw_data_shape": tuple(raw_data.shape),
        "test_tensor_shape": tuple(dataset.x.shape),
        "checkpoint_train_loss": checkpoint.get("train_loss", None),
        "checkpoint_val_loss": checkpoint.get("val_loss", None),
        "device": str(device),
        "tested_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_config,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "loss": result["loss"],
        "num_samples": result["num_samples"],
        "metrics": result["metrics"],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_metrics_csv(output_dir / "metrics.csv", summary)

    if result["sample_x"] is not None:
        np.savez(
            output_dir / "samples.npz",
            x=result["sample_x"],
            score=result["sample_score"],
        )

    metric_text = ""
    if summary["metrics"]:
        metric_text = (
            f" score_nmse={summary['metrics'].get('score_nmse', float('nan')):.6f}"
            f" score_cos={summary['metrics'].get('score_cos', float('nan')):.6f}"
        )

    tqdm.write(
        f"test_loss={summary['loss']:.6f} "
        f"num_samples={summary['num_samples']}"
        f"{metric_text}"
    )
    tqdm.write(f"saved summary: {output_dir / 'summary.json'}")
    tqdm.write(f"saved metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
