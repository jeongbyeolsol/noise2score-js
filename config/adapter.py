import argparse
import json
from pathlib import Path


SECTION_KEY_MAP = {
    "runtime": {},
    "data": {},
    "image": {},
    "noise": {},
    "output": {},
    "checkpoint": {},
    "blind": {},
    "ardae": {
        "train": "train_ardae",
        "test": "test_ardae",
        "train_data": "ardae_train_data",
        "train_data_mode": "ardae_train_data_mode",
        "test_data": "ardae_test_data",
        "test_data_mode": "ardae_test_data_mode",
        "max_patches_per_image": "ardae_max_patches_per_image",
        "save_dir": "ardae_save_dir",
        "test_output_dir": "ardae_test_output_dir",
        "use_metric": "ardae_use_metric",
        "save_every_best": "ardae_save_every_best",
        "epochs": "ardae_epochs",
        "batch_size": "ardae_batch_size",
        "lr": "ardae_lr",
        "weight_decay": "ardae_weight_decay",
        "val_ratio": "ardae_val_ratio",
        "log_every": "ardae_log_every",
        "normalize": "ardae_normalize",
        "no_flatten": "ardae_no_flatten",
        "backbone": "ardae_backbone",
        "h_dim": "ardae_h_dim",
        "num_hidden_layers": "ardae_num_hidden_layers",
        "nonlinearity": "ardae_nonlinearity",
        "base_channels": "ardae_base_channels",
        "channel_mults": "ardae_channel_mults",
        "no_norm": "ardae_no_norm",
        "patch_loader": "ardae_patch_loader",
        "sigma_min": "ardae_sigma_min",
        "sigma_max": "ardae_sigma_max",
        "linear_sigma": "ardae_linear_sigma",
        "smoothing": "ardae_smoothing",
        "test_max_batches": "ardae_test_max_batches",
        "test_save_samples": "ardae_test_save_samples",
        "test_no_metric": "ardae_test_no_metric",
    },
}


def add_config_argument(parser):
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON/YAML config file. CLI arguments override config values.",
    )


def parse_args_with_config(parser, argv=None, aliases=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    if pre_args.config:
        config_values = load_config_defaults(pre_args.config, aliases=aliases)
        apply_config_defaults(parser, config_values)

    args = parser.parse_args(argv)
    args.config = pre_args.config
    return args


def load_config_defaults(path, aliases=None):
    path = Path(path)
    config = _load_mapping(path)
    if "defaults" in config and isinstance(config["defaults"], dict):
        config = config["defaults"]
    return normalize_config(config, aliases=aliases)


def apply_config_defaults(parser, config_values):
    valid_dests = {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }
    unknown = sorted(set(config_values) - valid_dests)
    if unknown:
        formatted = ", ".join(unknown)
        raise ValueError(f"Unknown config keys for this script: {formatted}")

    for action in parser._actions:
        if action.dest in config_values and getattr(action, "required", False):
            action.required = False

    parser.set_defaults(**config_values)


def normalize_config(config, aliases=None):
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping/object.")

    normalized = {}
    for raw_key, value in config.items():
        key = _normalize_key(raw_key)
        if isinstance(value, dict) and key in SECTION_KEY_MAP:
            normalized.update(_normalize_section(key, value))
        elif isinstance(value, dict) and aliases and key in aliases:
            normalized.update(_normalize_alias_section(value, aliases[key]))
        else:
            normalized[_resolve_alias(key, aliases)] = value
    return normalized


def _normalize_section(section, values):
    key_map = SECTION_KEY_MAP[section]
    normalized = {}
    for raw_key, value in values.items():
        key = _normalize_key(raw_key)
        dest = key_map.get(key, key)
        normalized[dest] = value
    return normalized


def _normalize_alias_section(values, alias_map):
    normalized = {}
    for raw_key, value in values.items():
        key = _normalize_key(raw_key)
        normalized[_resolve_alias(key, alias_map)] = value
    return normalized


def _resolve_alias(key, aliases):
    if aliases is None:
        return key
    if isinstance(aliases, dict):
        return aliases.get(key, key)
    return key


def _normalize_key(key):
    return str(key).strip().replace("-", "_")


def _load_mapping(path):
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config files require PyYAML. Use JSON or install pyyaml."
            ) from exc
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config file type: {suffix}")
