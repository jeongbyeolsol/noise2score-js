from pathlib import Path
from tqdm import tqdm
import json

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