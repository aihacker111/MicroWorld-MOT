#!/usr/bin/env python3
"""Audit cached MOT clips for coverage, detector misses and protocol consistency."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _scalar(sample: Any, key: str, default: Any) -> Any:
    if key not in sample.files:
        return default
    return sample[key].item()


def audit(paths: list[Path]) -> dict[str, Any]:
    files = sorted(
        file
        for root in paths
        for file in (root.glob("*.npz") if root.is_dir() else [root])
        if file.suffix == ".npz"
    )
    if not files:
        raise FileNotFoundError("No .npz clips found")

    totals: Counter[str] = Counter()
    intervals: Counter[int] = Counter()
    protocols: Counter[int] = Counter()
    detector_sources: Counter[str] = Counter()
    preprocessing_configs: Counter[str] = Counter()
    sequences: set[str] = set()
    covered_tracks: set[tuple[str, int]] = set()
    for path in files:
        with np.load(path) as sample:
            sequence = str(_scalar(sample, "sequence", "unknown"))
            sequences.add(sequence)
            protocol = int(_scalar(sample, "protocol_version", 1))
            interval = int(_scalar(sample, "temporal_interval", 1))
            protocols[protocol] += 1
            intervals[interval] += 1
            detector_source = str(_scalar(sample, "detector_source", "unrecorded"))
            detector_sources[detector_source] += 1
            preprocessing_config = {
                "sequence_length": int(
                    _scalar(sample, "sequence_length", sample["boxes"].shape[0])
                ),
                "max_tracks": int(_scalar(sample, "max_tracks", sample["boxes"].shape[1])),
                "max_detections": int(
                    _scalar(sample, "max_detections", sample["detection_boxes"].shape[1])
                ),
                "min_track_frames": int(_scalar(sample, "min_track_frames", -1)),
                "iou_threshold": round(float(_scalar(sample, "iou_threshold", -1.0)), 6),
                "score_threshold": round(float(_scalar(sample, "score_threshold", -1.0)), 6),
                "sampling_seed": int(_scalar(sample, "sampling_seed", -1)),
            }
            preprocessing_configs[json.dumps(preprocessing_config, sort_keys=True)] += 1
            slot_valid = sample["slot_valid"].astype(bool)
            existence = sample["existence"].astype(bool) & slot_valid[None]
            visibility = sample["visibility"].astype(bool) & slot_valid[None]
            matched = (sample["assignment"] >= 0) & slot_valid[None]
            detection_valid = sample["detection_valid"].astype(bool)
            totals["clips"] += 1
            totals["frames"] += int(existence.shape[0])
            totals["valid_track_slots"] += int(slot_valid.sum())
            totals["existing_track_frames"] += int(existence.sum())
            totals["visible_track_frames"] += int(visibility.sum())
            totals["matched_track_frames"] += int((matched & existence).sum())
            totals["matched_visible_frames"] += int((matched & visibility).sum())
            totals["valid_detections"] += int(detection_valid.sum())
            if "track_ids" in sample.files:
                for track_id in sample["track_ids"][slot_valid]:
                    covered_tracks.add((sequence, int(track_id)))

    existing = max(1, totals["existing_track_frames"])
    visible = max(1, totals["visible_track_frames"])
    frames = max(1, totals["frames"])
    return {
        "files": len(files),
        "sequences": len(sequences),
        "covered_sequence_track_ids": len(covered_tracks),
        "protocol_versions": dict(sorted(protocols.items())),
        "detector_sources": dict(sorted(detector_sources.items())),
        "temporal_intervals": dict(sorted(intervals.items())),
        "preprocessing_configs": {
            config: count for config, count in sorted(preprocessing_configs.items())
        },
        "mean_tracks_per_clip": totals["valid_track_slots"] / totals["clips"],
        "mean_detections_per_frame": totals["valid_detections"] / frames,
        "detector_match_rate_existing": totals["matched_track_frames"] / existing,
        "detector_match_rate_visible": totals["matched_visible_frames"] / visible,
        "totals": dict(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Clip directories or .npz files")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit(args.paths)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
