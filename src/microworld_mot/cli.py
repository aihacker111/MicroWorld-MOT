from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import config_from_dict, load_config
from .data import build_datasets
from .models import build_system
from .models.registry import resolved_config
from .training.engine import evaluate_epoch, resolve_device, train


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_overrides(config: Any, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"Unknown config key: {key}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"Unknown config key: {key}")
        setattr(target, parts[-1], _parse_value(raw_value))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Single-run MicroWorld-MOT training")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested JSON config field; may be repeated.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args.set)
    checkpoint = train(config)
    print(f"best_checkpoint={checkpoint.resolve()}")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MicroWorld-MOT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="", help="Optional config override; checkpoint config is default.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config(args.config) if args.config else config_from_dict(checkpoint["config"])
    config.model = resolved_config(config.model)
    _, validation = build_datasets(config.data, config.model, config.train.seed)
    loader = DataLoader(
        validation,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    system = build_system(config.model).to(device)
    system.load_state_dict(checkpoint["model"])
    metrics = evaluate_epoch(system, loader, config, device, int(checkpoint["epoch"]))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    train_main()

