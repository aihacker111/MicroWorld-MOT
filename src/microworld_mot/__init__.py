"""MicroWorld-MOT: object-centric world modelling for online MOT."""

from .config import ExperimentConfig, load_config
from .models.registry import build_system, list_models
from .types import WorldPrediction, WorldState

__all__ = [
    "ExperimentConfig",
    "WorldPrediction",
    "WorldState",
    "build_system",
    "list_models",
    "load_config",
]

__version__ = "0.1.0"

