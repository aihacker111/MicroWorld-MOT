# Fair tracking-data protocol

This protocol is designed to make MicroWorld-MOT ablations comparable with
modern DanceTrack training recipes while avoiding detector-conditioned data
selection.

## Evidence behind the choices

- The official DanceTrack/ByteTrack recipe converts all training annotations
  to COCO and trains the detector on the complete train split.
- MOTR and MOTRv2 use temporal interval sampling and explicitly inject missed
  tracks and false positives rather than retaining only easy detections.
- MeMOTR uses a temporal-length curriculum with random intervals and hundreds
  of object queries.
- MOTIP uses longer 30-frame clips, interval sampling and trajectory-level
  occlusion/switch augmentation.
- DiffMOT constructs motion samples from every GT identity trajectory, not
  only identities detected in the first frame.

Primary references:

- <https://github.com/DanceTrack/DanceTrack/blob/main/ByteTrack/README.md>
- <https://github.com/megvii-research/MOTR/blob/main/configs/r50_motr_train.sh>
- <https://raw.githubusercontent.com/megvii-research/MOTRv2/main/configs/motrv2.args>
- <https://raw.githubusercontent.com/MCG-NJU/MeMOTR/main/configs/train_dancetrack.yaml>
- <https://raw.githubusercontent.com/MCG-NJU/MOTIP/main/configs/r50_deformable_detr_motip_dancetrack.yaml>
- <https://github.com/ShuCvlab/DiffMOT/blob/main/dancetrack_data_process.py>

## Protocol v2

1. Official train and validation sequences remain disjoint. Validation is
   never used for training or hyperparameter selection leakage.
2. Tracks are selected from GT visibility and longevity, not from a detector
   match in the first frame.
3. A detector miss is retained as `assignment=-1`. It does not remove the GT
   identity or the entire clip.
4. When a window exceeds `max_tracks`, coverage groups are emitted until every
   eligible identity appears in at least one clip.
5. One temporal interval is sampled reproducibly for each start position from
   the requested interval set. `delta_time` and sampled frame numbers are saved
   in every clip.
6. Frozen detector results are cached once. Training-time dropout, temporal
   miss spans, box/score noise, hard false positives and detection-slot
   shuffling are applied online and disabled for validation.
7. Each clip stores `protocol_version`, `track_ids`, `frame_numbers` and
   `temporal_interval`, plus detector thresholds and tensor capacities, so the
   experiment can be audited later.

## What “fair against other models” means

Use two separate tables rather than mixing evaluation regimes:

1. **Tracker-only comparison:** export one fixed detection set and give the
   exact same boxes, scores, video split and evaluation code to every tracker.
   Keep the detector threshold fixed. Report tracker parameters, FLOPs and FPS
   separately from the detector.
2. **End-to-end comparison:** run each method with its officially documented
   detector and training recipe. Report detector identity, detector training
   data and full-system cost. These numbers are not evidence that one tracking
   module alone is better.

Do not force sequence models such as MOTR/MOTIP or trajectory models such as
DiffMOT through this repository's NPZ representation merely to make the input
files look identical. Reproduce their native preprocessing, but freeze the
semantic train/validation split and evaluate all outputs with the same official
DanceTrack/TrackEval settings. For an architecture ablation inside this repo,
keep protocol-v2 clips, seeds, optimizer and augmentations identical. Also run
an augmentation-off ablation so data corruption is not mistaken for model
novelty.

## DanceTrack preparation

Use 24 sampled frames, intervals 1/2/4 and coverage limits 96/192 for training.
Validation uses consecutive frames only and has no online augmentation.

The shortest command for the repository's standard layout is:

```bash
bash scripts/prepare_all_datasets.sh dancetrack
```

The commands below show the equivalent settings explicitly.

### Faster preparation

The preparation path batches both full-frame detection and crop embeddings,
keeps each frame cache in RAM while overlapping clips are built, and saves the
ORB camera-motion result per sequence. These optimizations do not change the
protocol or targets. On a GPU with enough memory, increase both batch sizes:

```bash
INFERENCE_BATCH_SIZE=16 CROP_BATCH_SIZE=256 \
  bash scripts/prepare_all_datasets.sh dancetrack
```

If CUDA runs out of memory, try `8/128` (the defaults) or `4/64`. Re-running
the command reuses both detector and camera-motion caches. For maximum clip
writing speed, `UNCOMPRESSED_CLIPS=1` disables ZIP compression; this preserves
array values but can consume much more disk space:

```bash
UNCOMPRESSED_CLIPS=1 INFERENCE_BATCH_SIZE=16 CROP_BATCH_SIZE=256 \
  bash scripts/prepare_all_datasets.sh dancetrack
```

Do not run multiple YOLO preparation processes on the same GPU. If RAM is the
limitation, use `--no-frame-cache-memory` in a manual command; it is slower but
keeps less detector data resident.

```bash
python tools/prepare_real_data.py \
  --format mot \
  --root data/raw/DanceTrack/train1 \
  --dataset-name dancetrack \
  --output data/clips/dancetrack/train \
  --cache data/detector_cache/dancetrack/train1 \
  --detector yolo11n --device cuda:0 \
  --score-threshold 0.15 --iou-threshold 0.50 \
  --sequence-length 24 --stride 8 \
  --temporal-intervals 1,2,4 --sampling-seed 7 \
  --min-track-frames 2 --max-tracks 96 --max-detections 192 \
  --gt-classes 1 --coco-labels 1 \
  --estimate-camera-motion --overwrite-clips
```

Run the equivalent command for `train2`, changing only `--root` and `--cache`.
For validation, use `--temporal-intervals 1`.

With the repository and data on the HDD used in this project:

```bash
PROJECT_ROOT=/media/hung/HDD/workplaces/tin/cvpr2027/MicroWorld-MOT

python "$PROJECT_ROOT/tools/prepare_real_data.py" \
  --format mot --root "$PROJECT_ROOT/data/raw/DanceTrack/train2" \
  --dataset-name dancetrack \
  --output "$PROJECT_ROOT/data/clips/dancetrack/train" \
  --cache "$PROJECT_ROOT/data/detector_cache/dancetrack/train2" \
  --detector yolo11n --device cuda:0 \
  --score-threshold 0.15 --iou-threshold 0.50 \
  --sequence-length 24 --stride 8 --temporal-intervals 1,2,4 \
  --sampling-seed 7 --min-track-frames 2 \
  --max-tracks 96 --max-detections 192 \
  --gt-classes 1 --coco-labels 1 \
  --estimate-camera-motion --overwrite-clips

python "$PROJECT_ROOT/tools/prepare_real_data.py" \
  --format mot --root "$PROJECT_ROOT/data/raw/DanceTrack/val" \
  --dataset-name dancetrack \
  --output "$PROJECT_ROOT/data/clips/dancetrack/val" \
  --cache "$PROJECT_ROOT/data/detector_cache/dancetrack/val" \
  --detector yolo11n --device cuda:0 \
  --score-threshold 0.15 --iou-threshold 0.50 \
  --sequence-length 24 --stride 8 --temporal-intervals 1 \
  --sampling-seed 7 --min-track-frames 2 \
  --max-tracks 96 --max-detections 192 \
  --gt-classes 1 --coco-labels 1 \
  --estimate-camera-motion --overwrite-clips
```

## Audit and train

```bash
python tools/audit_dataset_clips.py \
  data/clips/dancetrack/train data/clips/dancetrack/val \
  --output outputs/dancetrack_data_audit.json

python tools/train.py --config configs/dancetrack_fair_tiny.json
```

The audit must show only protocol version 2, one detector source and the
intended preprocessing configurations (train and validation differ only where
documented). Keep the JSON report alongside the paper's experiment logs.
Detector-match rates are diagnostics, not filters.

## Reporting

Report the detector checkpoint, score threshold, IoU matching threshold,
sequence length, interval set, track/detection capacities, augmentation rates,
random seed and whether train+val was used. A final test submission may train on
train+val only if it is labeled separately; validation ablations must use train
only.
