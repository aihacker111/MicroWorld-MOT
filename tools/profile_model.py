from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from microworld_mot.config import load_config
from microworld_mot.models import build_system
from microworld_mot.models.registry import resolved_config


class LinearProfiler:
    def __init__(self, module: nn.Module) -> None:
        self.macs = 0
        self.handles = [
            child.register_forward_hook(self._hook)
            for child in module.modules()
            if isinstance(child, nn.Linear)
        ]

    def _hook(self, module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.macs += output.numel() * module.in_features

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters and approximate dense GFLOPs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracks", type=int, default=32)
    parser.add_argument("--detections", type=int, default=40)
    args = parser.parse_args()
    config = load_config(args.config)
    config.model = resolved_config(config.model)
    system = build_system(config.model).eval()
    profiler = LinearProfiler(system)
    with torch.inference_mode():
        boxes = torch.rand(1, args.tracks, 4)
        boxes[..., 2:] = (0.03 + 0.08 * boxes[..., 2:]).log()
        appearance = torch.randn(1, args.tracks, config.model.observation_dim)
        state = system.initialize_state(boxes, appearance)
        prediction = system.predict(state)
        det_boxes = torch.rand(1, args.detections, 4)
        det_boxes[..., 2:] = (0.03 + 0.08 * det_boxes[..., 2:]).log()
        det_appearance = torch.randn(1, args.detections, config.model.observation_dim)
        scores = torch.rand(1, args.detections)
        system.association_logits(prediction.state, det_boxes, det_appearance, scores)
        aligned_boxes = prediction.state.boxes
        aligned_appearance = appearance
        system.correct(
            prediction.state,
            aligned_boxes,
            aligned_appearance,
            torch.ones(1, args.tracks),
            torch.ones(1, args.tracks, dtype=torch.bool),
        )
        base_macs = profiler.macs
        profiler.macs = 0
        system.predict(state)
        world_macs = profiler.macs
    profiler.close()
    parameters = sum(parameter.numel() for parameter in system.parameters())
    trainable = sum(parameter.numel() for parameter in system.parameters() if parameter.requires_grad)
    base_gflops = 2 * base_macs / 1e9
    world_gflops = 2 * world_macs / 1e9
    counterfactual_upper = base_gflops + (
        config.model.max_hypotheses * config.model.rollout_horizon * world_gflops
    )
    print(
        json.dumps(
            {
                "model": config.model.name,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "tracks": args.tracks,
                "detections": args.detections,
                "approx_base_gflops": base_gflops,
                "approx_world_step_gflops": world_gflops,
                "approx_counterfactual_upper_gflops": counterfactual_upper,
                "note": "Linear-layer FLOPs only; validate end-to-end latency on the target GPU.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
