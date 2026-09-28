#!/usr/bin/env bash
set -euo pipefail

# Fair protocol-v2 preparation. Override any root or DEVICE from the shell;
# defaults match scripts/download_datasets.sh and keep all outputs on the same
# disk as this repository.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
DANCETRACK_ROOT="${DANCETRACK_ROOT:-${DATA_ROOT}/DanceTrack}"
MOT17_ROOT="${MOT17_ROOT:-${DATA_ROOT}/MOT17/train}"
MOT16_ROOT="${MOT16_ROOT:-${DATA_ROOT}/MOT16/train}"
VISDRONE_ROOT="${VISDRONE_ROOT:-${DATA_ROOT}/VisDrone}"
DEVICE="${DEVICE:-cuda:0}"
DETECTOR="${DETECTOR:-yolo11n}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-128}"
UNCOMPRESSED_CLIPS="${UNCOMPRESSED_CLIPS:-0}"

cd "${PROJECT_ROOT}"

requested=("$@")
if [[ "${#requested[@]}" -eq 0 ]]; then
  requested=(all)
fi

want_dataset() {
  local candidate="$1"
  local item
  for item in "${requested[@]}"; do
    if [[ "${item}" == "all" || "${item}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

for item in "${requested[@]}"; do
  case "${item}" in
    all|dancetrack|mot17|mot16|visdrone) ;;
    *)
      echo "Unknown dataset: ${item}" >&2
      echo "Usage: bash scripts/prepare_all_datasets.sh [all|dancetrack|mot17|mot16|visdrone ...]" >&2
      exit 2
      ;;
  esac
done

audit_paths=()

train_protocol=(
  --detector "${DETECTOR}" --device "${DEVICE}"
  --score-threshold 0.15 --iou-threshold 0.50
  --sequence-length 24 --stride 8
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" --crop-batch-size "${CROP_BATCH_SIZE}"
  --temporal-intervals 1,2,4 --sampling-seed 7
  --min-track-frames 2 --max-tracks 96 --max-detections 192
  --estimate-camera-motion --overwrite-clips
)
val_protocol=(
  --detector "${DETECTOR}" --device "${DEVICE}"
  --score-threshold 0.15 --iou-threshold 0.50
  --sequence-length 24 --stride 8
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" --crop-batch-size "${CROP_BATCH_SIZE}"
  --temporal-intervals 1 --sampling-seed 7
  --min-track-frames 2 --max-tracks 96 --max-detections 192
  --estimate-camera-motion --overwrite-clips
)
if [[ "${UNCOMPRESSED_CLIPS}" == "1" ]]; then
  train_protocol+=(--uncompressed-clips)
  val_protocol+=(--uncompressed-clips)
fi

# DanceTrack's official train1/train2 names are archive shards of one semantic
# train split. Both write to one clip directory while retaining distinct caches.
if want_dataset dancetrack; then
  for part in train1 train2; do
    python tools/prepare_real_data.py --format mot \
      --root "${DANCETRACK_ROOT}/${part}" --dataset-name dancetrack \
      --output data/clips/dancetrack/train \
      --cache "data/detector_cache/dancetrack/${part}" \
      --gt-classes 1 --coco-labels 1 "${train_protocol[@]}"
  done

  python tools/prepare_real_data.py --format mot \
    --root "${DANCETRACK_ROOT}/val" --dataset-name dancetrack \
    --output data/clips/dancetrack/val --cache data/detector_cache/dancetrack/val \
    --gt-classes 1 --coco-labels 1 "${val_protocol[@]}"
  audit_paths+=(data/clips/dancetrack/train data/clips/dancetrack/val)
fi

# Sequence-level splits prevent overlap between windows in train and val.
if want_dataset mot17; then
  python tools/prepare_real_data.py --format mot \
    --root "${MOT17_ROOT}" --dataset-name mot17 \
    --include-sequences MOT17-02,MOT17-04,MOT17-05,MOT17-09,MOT17-10 \
    --output data/clips/mot17/train --cache data/detector_cache/mot17/train \
    --gt-classes 1 --coco-labels 1 "${train_protocol[@]}"

  python tools/prepare_real_data.py --format mot \
    --root "${MOT17_ROOT}" --dataset-name mot17 \
    --include-sequences MOT17-11,MOT17-13 \
    --output data/clips/mot17/val --cache data/detector_cache/mot17/val \
    --gt-classes 1 --coco-labels 1 "${val_protocol[@]}"
  audit_paths+=(data/clips/mot17/train data/clips/mot17/val)
fi

if want_dataset mot16; then
  python tools/prepare_real_data.py --format mot \
    --root "${MOT16_ROOT}" --dataset-name mot16 \
    --include-sequences MOT16-02,MOT16-04,MOT16-05,MOT16-09,MOT16-10 \
    --output data/clips/mot16/train --cache data/detector_cache/mot16/train \
    --gt-classes 1 --coco-labels 1 "${train_protocol[@]}"

  python tools/prepare_real_data.py --format mot \
    --root "${MOT16_ROOT}" --dataset-name mot16 \
    --include-sequences MOT16-11,MOT16-13 \
    --output data/clips/mot16/val --cache data/detector_cache/mot16/val \
    --gt-classes 1 --coco-labels 1 "${val_protocol[@]}"
  audit_paths+=(data/clips/mot16/train data/clips/mot16/val)
fi

if want_dataset visdrone; then
  python tools/prepare_real_data.py --format visdrone \
    --root "${VISDRONE_ROOT}/VisDrone2019-MOT-train" --dataset-name visdrone \
    --output data/clips/visdrone/train --cache data/detector_cache/visdrone/train \
    --gt-classes 1,2 --coco-labels 1 "${train_protocol[@]}"

  python tools/prepare_real_data.py --format visdrone \
    --root "${VISDRONE_ROOT}/VisDrone2019-MOT-val" --dataset-name visdrone \
    --output data/clips/visdrone/val --cache data/detector_cache/visdrone/val \
    --gt-classes 1,2 --coco-labels 1 "${val_protocol[@]}"
  audit_paths+=(data/clips/visdrone/train data/clips/visdrone/val)
fi

python tools/audit_dataset_clips.py "${audit_paths[@]}" \
  --output outputs/data_audit.json
