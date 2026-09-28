from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from torch import nn

from ..config import ModelConfig
from .baselines import (
    ConstantVelocityModel,
    GRUWorldModel,
    KalmanWorldModel,
    MLPDeltaModel,
    TinySSMWorldModel,
)
from .system import MicroWorldMOT
from .world_model import ObjectWorldModel

_BUILDERS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "constant_velocity": ConstantVelocityModel,
    "kalman": KalmanWorldModel,
    "mlp_delta": MLPDeltaModel,
    "gru_tiny": GRUWorldModel,
    "ssm_tiny": TinySSMWorldModel,
    "microworld_det": ObjectWorldModel,
    "microworld_prob": ObjectWorldModel,
    "microworld_graph": ObjectWorldModel,
    "microworld_cf": ObjectWorldModel,
    "microworld_jepa": ObjectWorldModel,
    "microworld_adaptive": ObjectWorldModel,
}


_SIZE_PRESETS = {
    "nano": {"latent_dim": 48, "appearance_dim": 24, "graph_hidden_dim": 64},
    "tiny": {"latent_dim": 64, "appearance_dim": 32, "graph_hidden_dim": 96},
    "small": {"latent_dim": 96, "appearance_dim": 48, "graph_hidden_dim": 128},
}


def _split_name(name: str) -> tuple[str, str | None]:
    if name in _BUILDERS:
        return name, None
    for size in _SIZE_PRESETS:
        suffix = f"_{size}"
        if name.endswith(suffix):
            return name[: -len(suffix)], size
    return name, None


def resolved_config(config: ModelConfig) -> ModelConfig:
    base_name, size = _split_name(config.name)
    updates = dict(_SIZE_PRESETS[size]) if size is not None else {}
    # These are actual architectural ablations, not aliases. Counterfactual and
    # adaptive variants share the graph world-model weights but differ in the
    # inference controller that consumes them.
    if base_name == "microworld_det":
        updates.update(graph_layers=0, motion_modes=1)
    elif base_name == "microworld_prob":
        updates.update(graph_layers=0)
    return replace(config, name=base_name, **updates)


def list_models() -> list[str]:
    names = list(_BUILDERS)
    proposed = [name for name in names if name.startswith("microworld_")]
    names.extend(f"{name}_{size}" for name in proposed for size in _SIZE_PRESETS)
    return sorted(names)


def build_system(config: ModelConfig) -> MicroWorldMOT:
    config = resolved_config(config)
    name, _ = _split_name(config.name)
    if name not in _BUILDERS:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{config.name}'. Available models: {available}")
    return MicroWorldMOT(_BUILDERS[name](config), config)
