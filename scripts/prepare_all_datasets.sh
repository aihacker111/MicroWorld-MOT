#!/usr/bin/env bash
set -euo pipefail

# Edit these four roots before running. Each command is resumable because
# detector outputs are cached per frame.
DANCETRACK_ROOT="/datasets/DanceTrack/dancetrack"
MOT17_ROOT="/datasets/MOT17/train"
MOT16_ROOT="/datasets/MOT16/train"
VISDRONE_ROOT="/datasets/VisDrone"
DEVICE="cuda:0"
DETECTOR="yolo11n"

python tools/prepare_real_data.py --format mot \
  --root "${DANCETRACK_ROOT}/train" --dataset-name dancetrack \
  --output data/clips/dancetrack/train --cache data/detector_cache/dancetrack/train \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format mot \
  --root "${DANCETRACK_ROOT}/val" --dataset-name dancetrack \
  --output data/clips/dancetrack/val --cache data/detector_cache/dancetrack/val \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

# Sequence-level splits prevent clip overlap between train and validation.
python tools/prepare_real_data.py --format mot \
  --root "${MOT17_ROOT}" --dataset-name mot17 \
  --include-sequences MOT17-02,MOT17-04,MOT17-05,MOT17-09,MOT17-10 \
  --output data/clips/mot17/train --cache data/detector_cache/mot17/train \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format mot \
  --root "${MOT17_ROOT}" --dataset-name mot17 \
  --include-sequences MOT17-11,MOT17-13 \
  --output data/clips/mot17/val --cache data/detector_cache/mot17/val \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format mot \
  --root "${MOT16_ROOT}" --dataset-name mot16 \
  --include-sequences MOT16-02,MOT16-04,MOT16-05,MOT16-09,MOT16-10 \
  --output data/clips/mot16/train --cache data/detector_cache/mot16/train \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format mot \
  --root "${MOT16_ROOT}" --dataset-name mot16 \
  --include-sequences MOT16-11,MOT16-13 \
  --output data/clips/mot16/val --cache data/detector_cache/mot16/val \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format visdrone \
  --root "${VISDRONE_ROOT}/VisDrone2019-MOT-train" --dataset-name visdrone \
  --output data/clips/visdrone/train --cache data/detector_cache/visdrone/train \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion

python tools/prepare_real_data.py --format visdrone \
  --root "${VISDRONE_ROOT}/VisDrone2019-MOT-val" --dataset-name visdrone \
  --output data/clips/visdrone/val --cache data/detector_cache/visdrone/val \
  --detector "${DETECTOR}" --device "${DEVICE}" --estimate-camera-motion
