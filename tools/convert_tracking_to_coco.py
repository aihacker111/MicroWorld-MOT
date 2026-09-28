#!/usr/bin/env python3
"""Convert DanceTrack, MOT16/17 and VisDrone-MOT to video-aware COCO JSON."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MOT_CATEGORIES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_motorized_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
}

VISDRONE_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",
    10: "motor",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    directory: Path
    image_directory: Path
    annotation_file: Path | None
    width: int
    height: int
    fps: float
    declared_length: int | None = None


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )


def _frame_number(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else fallback


def _relative_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _parse_seqinfo(sequence_dir: Path, images: list[Path]) -> tuple[int, int, float, int | None]:
    seqinfo = sequence_dir / "seqinfo.ini"
    if seqinfo.is_file():
        parser = configparser.ConfigParser()
        parser.read(seqinfo, encoding="utf-8")
        if parser.has_section("Sequence"):
            section = parser["Sequence"]
            width = section.getint("imWidth", fallback=0)
            height = section.getint("imHeight", fallback=0)
            fps = section.getfloat("frameRate", fallback=30.0)
            length = section.getint("seqLength", fallback=0) or None
            if width > 0 and height > 0:
                return width, height, fps, length
    if not images:
        raise ValueError(f"No images found under {sequence_dir}")
    width, height = _read_image_size(images[0])
    return width, height, 30.0, None


def _belongs_to_split(sequence_dir: Path, root: Path, split: str, dataset: str) -> bool:
    split = split.lower()
    if root.name.lower() == split:
        return True
    try:
        parts = [part.lower() for part in sequence_dir.relative_to(root).parts]
    except ValueError:
        parts = [part.lower() for part in sequence_dir.parts]
    if split in parts:
        return True
    return (
        dataset == "dancetrack"
        and split == "train"
        and any(re.fullmatch(r"train\d+", part) for part in parts)
    )


def _mot_sequence_dirs(root: Path, split: str, dataset: str) -> list[Path]:
    candidates: set[Path] = set()
    if (root / "img1").is_dir():
        candidates.add(root)
    for image_dir in root.rglob("img1"):
        if image_dir.is_dir():
            candidates.add(image_dir.parent)
    selected = [path for path in candidates if _belongs_to_split(path, root, split, dataset)]
    return sorted(selected, key=lambda path: path.as_posix().lower())


def _deduplicate_mot17(sequence_dirs: list[Path], detector: str) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(MOT17-\d{2})(?:-(DPM|FRCNN|SDP))?$", re.IGNORECASE)
    for sequence_dir in sequence_dirs:
        match = pattern.match(sequence_dir.name)
        grouped[match.group(1).upper() if match else sequence_dir.name].append(sequence_dir)

    selected: list[Path] = []
    suffix = f"-{detector.upper()}"
    for paths in grouped.values():
        preferred = [path for path in paths if path.name.upper().endswith(suffix)]
        selected.append(sorted(preferred or paths, key=lambda path: path.name)[0])
    return sorted(selected, key=lambda path: path.name)


def _mot_sequences(
    root: Path,
    split: str,
    dataset: str,
    mot17_detector: str,
    keep_mot17_duplicates: bool,
) -> list[SequenceInfo]:
    directories = _mot_sequence_dirs(root, split, dataset)
    if dataset == "mot17" and not keep_mot17_duplicates:
        directories = _deduplicate_mot17(directories, mot17_detector)

    sequences: list[SequenceInfo] = []
    for sequence_dir in directories:
        image_dir = sequence_dir / "img1"
        images = _image_files(image_dir)
        if not images:
            continue
        width, height, fps, length = _parse_seqinfo(sequence_dir, images)
        name = sequence_dir.name
        if dataset == "mot17" and not keep_mot17_duplicates:
            name = re.sub(r"-(DPM|FRCNN|SDP)$", "", name, flags=re.IGNORECASE)
        annotation = sequence_dir / "gt" / "gt.txt"
        sequences.append(
            SequenceInfo(
                name=name,
                directory=sequence_dir,
                image_directory=image_dir,
                annotation_file=annotation if annotation.is_file() else None,
                width=width,
                height=height,
                fps=fps,
                declared_length=length,
            )
        )
    return sequences


def _visdrone_split_token(split: str) -> str:
    aliases = {"test": "test-dev", "testdev": "test-dev", "challenge": "test-challenge"}
    return aliases.get(split.lower(), split.lower())


def _visdrone_sequences(root: Path, split: str) -> list[SequenceInfo]:
    token = _visdrone_split_token(split)
    dataset_roots: set[Path] = set()
    if (root / "sequences").is_dir():
        dataset_roots.add(root)
    for sequences_dir in root.rglob("sequences"):
        if sequences_dir.is_dir():
            dataset_roots.add(sequences_dir.parent)

    matching = [path for path in dataset_roots if token in path.name.lower()]
    if not matching and len(dataset_roots) == 1:
        only = next(iter(dataset_roots))
        if token in root.name.lower() or root == only:
            matching = [only]

    sequences: list[SequenceInfo] = []
    for dataset_root in sorted(matching, key=lambda path: path.as_posix().lower()):
        annotation_dir = dataset_root / "annotations"
        for image_dir in sorted((dataset_root / "sequences").iterdir(), key=lambda path: path.name):
            if not image_dir.is_dir():
                continue
            images = _image_files(image_dir)
            if not images:
                continue
            width, height = _read_image_size(images[0])
            annotation = annotation_dir / f"{image_dir.name}.txt"
            sequences.append(
                SequenceInfo(
                    name=image_dir.name,
                    directory=dataset_root,
                    image_directory=image_dir,
                    annotation_file=annotation if annotation.is_file() else None,
                    width=width,
                    height=height,
                    fps=30.0,
                    declared_length=None,
                )
            )
    return sequences


def _categories(dataset: str, category_mode: str) -> list[dict[str, Any]]:
    if category_mode == "person":
        return [{"id": 1, "name": "person", "supercategory": "person"}]
    if dataset == "dancetrack":
        mapping = {1: "person"}
    elif dataset in {"mot16", "mot17"}:
        mapping = MOT_CATEGORIES
    else:
        mapping = VISDRONE_CATEGORIES
    return [
        {"id": category_id, "name": name, "supercategory": "object"}
        for category_id, name in mapping.items()
    ]


def _mapped_category(dataset: str, source_category: int, category_mode: str) -> int | None:
    if category_mode == "person":
        if dataset == "visdrone":
            return 1 if source_category in {1, 2} else None
        return 1 if source_category == 1 else None
    if dataset == "dancetrack":
        return 1 if source_category == 1 else None
    if dataset in {"mot16", "mot17"}:
        return source_category if source_category in MOT_CATEGORIES else None
    return source_category if source_category in VISDRONE_CATEGORIES else None


def _clip_bbox(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    x1 = min(max(left, 0.0), float(image_width))
    y1 = min(max(top, 0.0), float(image_height))
    x2 = min(max(left + width, 0.0), float(image_width))
    y2 = min(max(top + height, 0.0), float(image_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    return [x1, y1, clipped_width, clipped_height]


def _read_rows(path: Path | None) -> list[list[str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle) if row and any(value.strip() for value in row)]


def _mot_annotations(
    sequence: SequenceInfo,
    dataset: str,
    category_mode: str,
    image_ids: dict[int, int],
    annotation_id: int,
) -> tuple[list[dict[str, Any]], int]:
    annotations: list[dict[str, Any]] = []
    for row in _read_rows(sequence.annotation_file):
        if len(row) < 6:
            continue
        frame_id = int(float(row[0]))
        track_id = int(float(row[1]))
        if frame_id not in image_ids or track_id < 0:
            continue
        mark = float(row[6]) if len(row) > 6 else 1.0
        if mark <= 0.0:
            continue
        source_category = (
            1 if dataset == "dancetrack" else int(float(row[7])) if len(row) > 7 else 1
        )
        category_id = _mapped_category(dataset, source_category, category_mode)
        if category_id is None:
            continue
        bbox = _clip_bbox(
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            sequence.width,
            sequence.height,
        )
        if bbox is None:
            continue
        visibility = float(row[8]) if len(row) > 8 else 1.0
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[frame_id],
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "track_id": track_id,
                "visibility": visibility,
                "source_category_id": source_category,
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def _visdrone_annotations(
    sequence: SequenceInfo,
    category_mode: str,
    image_ids: dict[int, int],
    annotation_id: int,
) -> tuple[list[dict[str, Any]], int]:
    annotations: list[dict[str, Any]] = []
    for row in _read_rows(sequence.annotation_file):
        if len(row) < 8:
            continue
        frame_id = int(float(row[0]))
        track_id = int(float(row[1]))
        score = float(row[6])
        source_category = int(float(row[7]))
        if frame_id not in image_ids or track_id < 0 or score <= 0.0:
            continue
        category_id = _mapped_category("visdrone", source_category, category_mode)
        if category_id is None:
            continue
        bbox = _clip_bbox(
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            sequence.width,
            sequence.height,
        )
        if bbox is None:
            continue
        truncation = int(float(row[8])) if len(row) > 8 else 0
        occlusion = int(float(row[9])) if len(row) > 9 else 0
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[frame_id],
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "track_id": track_id,
                "score": score,
                "truncation": truncation,
                "occlusion": occlusion,
                "visibility": max(0.0, 1.0 - 0.5 * occlusion),
                "source_category_id": source_category,
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def convert_split(
    dataset: str,
    root: Path,
    split: str,
    category_mode: str = "native",
    mot17_detector: str = "FRCNN",
    keep_mot17_duplicates: bool = False,
) -> dict[str, Any]:
    """Convert one dataset split and return a COCO-compatible dictionary."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if dataset == "visdrone":
        sequences = _visdrone_sequences(root, split)
    else:
        sequences = _mot_sequences(root, split, dataset, mot17_detector, keep_mot17_duplicates)
    if not sequences:
        raise ValueError(f"No {dataset} sequences found for split '{split}' under {root}")

    videos: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    annotation_id = 1

    for video_id, sequence in enumerate(sequences, start=1):
        image_paths = _image_files(sequence.image_directory)
        videos.append(
            {
                "id": video_id,
                "name": sequence.name,
                "width": sequence.width,
                "height": sequence.height,
                "fps": sequence.fps,
                "num_frames": len(image_paths),
            }
        )
        frame_to_image: dict[int, int] = {}
        for fallback_frame, image_path in enumerate(image_paths, start=1):
            frame_id = _frame_number(image_path, fallback_frame)
            frame_to_image[frame_id] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": _relative_name(image_path, root),
                    "width": sequence.width,
                    "height": sequence.height,
                    "video_id": video_id,
                    "frame_id": frame_id,
                }
            )
            image_id += 1

        if dataset == "visdrone":
            new_annotations, annotation_id = _visdrone_annotations(
                sequence, category_mode, frame_to_image, annotation_id
            )
        else:
            new_annotations, annotation_id = _mot_annotations(
                sequence, dataset, category_mode, frame_to_image, annotation_id
            )
        annotations.extend(new_annotations)

    return {
        "info": {
            "description": f"{dataset} {split} converted to video-aware COCO",
            "version": "1.0",
        },
        "licenses": [],
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "categories": _categories(dataset, category_mode),
    }


def _validate_coco(coco: dict[str, Any]) -> None:
    image_ids = [image["id"] for image in coco["images"]]
    annotation_ids = [annotation["id"] for annotation in coco["annotations"]]
    category_ids = {category["id"] for category in coco["categories"]}
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image ids were generated")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate annotation ids were generated")
    known_images = set(image_ids)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in known_images:
            raise ValueError(f"Unknown image_id in annotation {annotation['id']}")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Unknown category_id in annotation {annotation['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["dancetrack", "mot16", "mot17", "visdrone"], required=True
    )
    parser.add_argument("--root", type=Path, required=True, help="Extracted dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON files")
    parser.add_argument(
        "--splits", nargs="+", help="Splits to convert; defaults depend on the dataset"
    )
    parser.add_argument(
        "--category-mode",
        choices=["native", "person"],
        default="native",
        help="Keep evaluated native classes, or map person classes to one category",
    )
    parser.add_argument(
        "--mot17-detector",
        choices=["DPM", "FRCNN", "SDP"],
        default="FRCNN",
        help="MOT17 copy retained when duplicate detector folders are present",
    )
    parser.add_argument(
        "--keep-mot17-duplicates",
        action="store_true",
        help="Keep all DPM/FRCNN/SDP copies (normally incorrect for image training)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Indent JSON (uses substantially more disk space)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_splits = {
        "dancetrack": ["train", "val", "test"],
        "mot16": ["train", "test"],
        "mot17": ["train", "test"],
        "visdrone": ["train", "val", "test-dev", "test-challenge"],
    }
    splits = args.splits or default_splits[args.dataset]
    args.output.mkdir(parents=True, exist_ok=True)

    for split in splits:
        coco = convert_split(
            dataset=args.dataset,
            root=args.root,
            split=split,
            category_mode=args.category_mode,
            mot17_detector=args.mot17_detector,
            keep_mot17_duplicates=args.keep_mot17_duplicates,
        )
        _validate_coco(coco)
        safe_split = split.lower().replace("-", "_")
        output_path = args.output / f"{args.dataset}_{safe_split}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                coco,
                handle,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
            )
        print(
            f"{output_path}: {len(coco['videos'])} videos, "
            f"{len(coco['images'])} images, {len(coco['annotations'])} annotations"
        )


if __name__ == "__main__":
    main()
