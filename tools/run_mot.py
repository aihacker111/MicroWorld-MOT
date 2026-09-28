from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microworld_mot.config import config_from_dict
from microworld_mot.models import build_system
from microworld_mot.models.registry import resolved_config
from microworld_mot.tracking.tracker import MicroWorldTracker, cxcylogwh_to_xyxy, xyxy_to_cxcylogwh
from microworld_mot.types import Detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a checkpoint on MOTChallenge-format detections")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detections", required=True, help="MOT det.txt")
    parser.add_argument("--output", required=True, help="Output MOT result txt")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--features", default="", help="Optional .npy aligned with det.txt rows")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    config.model = resolved_config(config.model)
    system = build_system(config.model)
    system.load_state_dict(checkpoint["model"])
    tracker = MicroWorldTracker(system, config.tracker, device)

    rows = np.loadtxt(args.detections, delimiter=",", ndmin=2)
    features = np.load(args.features) if args.features else None
    if features is not None and len(features) != len(rows):
        raise ValueError("--features must contain one row per detection")
    results: list[list[float]] = []
    maximum_frame = int(rows[:, 0].max()) if len(rows) else 0
    for frame in range(1, maximum_frame + 1):
        row_indices = np.flatnonzero(rows[:, 0].astype(int) == frame)
        frame_rows = rows[row_indices]
        if len(frame_rows):
            xywh = torch.from_numpy(frame_rows[:, 2:6]).float()
            xyxy = torch.stack(
                (xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]),
                dim=-1,
            )
            boxes = xyxy_to_cxcylogwh(xyxy, (args.image_height, args.image_width))
            scores = torch.from_numpy(frame_rows[:, 6]).float()
            if features is None:
                appearance = torch.zeros(len(frame_rows), config.model.observation_dim)
            else:
                appearance = torch.from_numpy(features[row_indices]).float()
            detections = Detections(boxes=boxes, scores=scores, appearance=appearance)
        else:
            detections = Detections(
                boxes=torch.empty(0, 4),
                scores=torch.empty(0),
                appearance=torch.empty(0, config.model.observation_dim),
            )
        for track in tracker.step(detections):
            xyxy = cxcylogwh_to_xyxy(
                track.box.unsqueeze(0), (args.image_height, args.image_width)
            )[0]
            left, top, right, bottom = [float(value) for value in xyxy]
            results.append(
                [frame, track.track_id, left, top, right - left, bottom - top, track.score, -1, -1, -1]
            )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if results:
        np.savetxt(destination, np.asarray(results), delimiter=",", fmt="%.6f")
    else:
        destination.write_text("", encoding="utf-8")
    print(f"tracks={len(results)} output={destination.resolve()}")


if __name__ == "__main__":
    main()
