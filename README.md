# MicroWorld-MOT

Research code for a tiny probabilistic object-world model that performs online
multi-object tracking through structured dynamics, sparse interactions,
Object-JEPA latent prediction and counterfactual association.

This repository is independent from Falcon-MOT. It does not contain or import
code from that project.

> **Research status:** this is an executable architecture prototype, not a
> pretrained model and not evidence of a CVPR claim. Numbers must come from
> real repeated experiments; the synthetic dataset is only a smoke test.

## What is the world model?

The world model is the block between track memory and data association:

```text
detector observations
        |
        v
previous object beliefs
        |
        v
+---------------------------------------+
| object world model                    |
| sparse graph -> structured transition |
| -> box distribution + future latent   |
+---------------------------------------+
        |
        v
predicted beliefs -> association -> correction
                         |
                         +-> counterfactual future rollout
```

It predicts object states, not pixels. The detector is intentionally outside
the trainable checkpoint so COCO-pretrained or task-specific detectors can be
changed without retraining the world-model interface.

## Implemented components

- Parameter-free constant-velocity and diagonal Kalman baselines.
- Learned MLP, GRU and diagonal SSM motion baselines.
- Structured object-world transition with a physical residual path.
- K-nearest-neighbour interaction graph.
- Gaussian-mixture box prediction and calibrated uncertainty output.
- Existence, occlusion and detector-refresh heads.
- Learned observation correction.
- Differentiable Sinkhorn association for training.
- Hungarian association for inference.
- Bounded counterfactual hypotheses with future rollouts.
- Online lifecycle management and uncertainty-driven detector scheduling.
- Real detector caching for MOT16, MOT17, DanceTrack and VisDrone.
- Independent detection slots, false positives and GT assignment targets.
- Frozen YOLO11n COCO detector and native YOLO11 crop embeddings.
- EMA target encoder and masked future-object latent prediction.
- Two-step open-loop Object-JEPA rollout objective.
- Single-run training, evaluation, parameter/GFLOP profiling and tests.
- GT-first, coverage-preserving clip construction with temporal-interval metadata.
- Online detector miss/false-positive/jitter augmentation disabled for validation.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for equations and tensor contracts.
See [docs/DATASETS.md](docs/DATASETS.md) for official sources and leakage notes.
See [docs/FAIR_DATA_PROTOCOL.md](docs/FAIR_DATA_PROTOCOL.md) for the reproducible
training-data protocol and its comparison with published MOT recipes.

## Installation

Python 3.10+ and PyTorch 2.0+ are required.

```bash
cd ~/Desktop/MicroWorld-MOT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,perception]'
```

For an existing Conda/PyTorch environment:

```bash
cd ~/Desktop/MicroWorld-MOT
python -m pip install -e '.[perception]'
```

## Verify the project

```bash
pytest
python tools/profile_model.py --config configs/microworld_tiny.json
python tools/profile_perception.py --weights yolo11n.pt --detections 40
```

## One-command training

The default configuration trains every trainable tracking component together:

```bash
python tools/train.py --config configs/microworld_tiny.json
```

Useful one-run overrides:

```bash
python tools/train.py \
  --config configs/microworld_tiny.json \
  --set train.device=cuda:0 \
  --set train.epochs=80 \
  --set train.batch_size=16 \
  --set data.kind=cached \
  --set data.path=/absolute/path/to/mot_clips
```

This uses one optimizer and produces:

```text
outputs/microworld_jepa_tiny/best.pt
outputs/microworld_jepa_tiny/last.pt
outputs/microworld_jepa_tiny/history.jsonl
outputs/microworld_jepa_tiny/config.json
```

AMP is enabled only on CUDA. Reduce `train.batch_size`, `data.max_tracks` or
`data.sequence_length` if one-GPU memory is insufficient.

## Pretrained perception

The turnkey real-data pipeline uses one official Ultralytics checkpoint:

```text
detector:          YOLO11n, COCO pretrained
appearance:        YOLO11n penultimate-layer crop embedding
feature size:      256 -> trainable projection -> 32
```

YOLO11n is frozen and Ultralytics downloads `yolo11n.pt` on the first cache
run. The same checkpoint performs detection and embeds each detected crop, so
there is no MobileNet or separate ReID backbone. The MicroWorld checkpoint
contains only the trainable 256-to-32 projection and tracker/world-model
components.

Ultralytics offers AGPL-3.0 and Enterprise licensing. Check which license is
appropriate before distributing a combined application or commercial model.

## Download and COCO conversion

Review and accept the official terms linked in `docs/DATASETS.md`, then install
the small dataset utility extra and run the resumable train/validation
downloader. Set `DATA_ROOT` to a disk with ample free space. DanceTrack test1
and test2 and the two VisDrone test splits are intentionally not downloaded:

```bash
python -m pip install -e '.[datasets]'
ACCEPT_DATASET_TERMS=1 bash scripts/download_datasets.sh all
```

One or more datasets can be selected instead of `all`, for example:

```bash
ACCEPT_DATASET_TERMS=1 bash scripts/download_datasets.sh dancetrack mot17
```

By default, data is stored under `data/raw` inside this repository, so it stays
on the same filesystem as the checkout. Set `DATA_ROOT` only when intentionally
using a different mounted disk.

Convert extracted tracking annotations to video-aware COCO JSON:

```bash
python tools/convert_tracking_to_coco.py \
  --dataset mot17 \
  --root data/raw/MOT17 \
  --output data/coco/mot17 \
  --splits train \
  --category-mode person
```

The converter supports `dancetrack`, `mot16`, `mot17` and `visdrone`. Its JSON
keeps standard COCO fields plus `videos`, `video_id`, `frame_id`, `track_id`,
visibility and VisDrone occlusion/truncation. Use `--category-mode person` to
map DanceTrack/MOT pedestrians and VisDrone pedestrian/people into one class.
For DanceTrack, `--splits train` automatically combines the official `train1`
and `train2` directories into one `dancetrack_train.json`; image and annotation
IDs remain globally unique.
MOT17 is deduplicated to its FRCNN copy by default; use
`--keep-mot17-duplicates` only when the detector-specific copies are genuinely
needed. Test splits without public GT produce image/video metadata and an empty
`annotations` list.

## Real dataset preparation

The project does not bypass dataset agreements or redistribute images.

Expected roots after `scripts/download_datasets.sh`:

```text
data/raw/DanceTrack/{train1,train2,val}/<sequence>/{img1,gt}
data/raw/MOT17/train/MOT17-xx-*/{img1,gt,det}
data/raw/MOT16/train/MOT16-xx/{img1,gt,det}
data/raw/VisDrone/VisDrone2019-MOT-{train,val}/{sequences,annotations}
```

The preparation script resolves its roots from the repository by default. You
can override `DATA_ROOT`, a dataset-specific root, or `DEVICE` from the shell:

```bash
bash scripts/prepare_all_datasets.sh all
```

To rebuild only DanceTrack, use `bash scripts/prepare_all_datasets.sh
dancetrack`. Multiple dataset names can be passed in one invocation.

The operation is resumable: detector outputs are saved per frame before clips
are assembled. To prepare one split manually:

```bash
python tools/prepare_real_data.py \
  --format mot \
  --root data/raw/DanceTrack/train1 \
  --dataset-name dancetrack \
  --output data/clips/dancetrack/train \
  --cache data/detector_cache/dancetrack/train1 \
  --detector yolo11n \
  --device cuda:0 \
  --score-threshold 0.15 \
  --iou-threshold 0.50 \
  --sequence-length 24 \
  --stride 8 \
  --temporal-intervals 1,2,4 \
  --sampling-seed 7 \
  --min-track-frames 2 \
  --max-tracks 96 \
  --max-detections 192 \
  --gt-classes 1 \
  --coco-labels 1 \
  --estimate-camera-motion \
  --overwrite-clips
```

Repeat this for `train2`, writing to the same clip directory but a separate
detector-cache directory. Prepare validation with `--temporal-intervals 1`.
The rationale and reproducibility checklist are in
[`docs/FAIR_DATA_PROTOCOL.md`](docs/FAIR_DATA_PROTOCOL.md).

For MOT16/17 public detections, use `--detector public`; YOLO11n still extracts
an appearance feature for every supplied box. For a new detector, emit the same
per-frame arrays (`boxes_xyxy`, `scores`, `labels`, `features`) and the rest of
the pipeline remains unchanged.

MOT17 contains DPM/FRCNN/SDP copies of the same videos. When a new detector is
used, the preparation tool deduplicates these automatically. MOT16 and MOT17
also overlap in underlying video content, so including both in joint training
is supported but is not an independent-data experiment. Never place overlapping
sequences on opposite train/validation sides.

Each cached clip contains:

| Key | Shape | Meaning |
|---|---:|---|
| `boxes` | `[T,N,4]` | GT normalized `(cx,cy,logw,logh)` |
| `slot_valid` | `[N]` | non-padding track slots |
| `existence` | `[T,N]` | object exists |
| `visibility` | `[T,N]` | object is visible |
| `detection_boxes` | `[T,M,4]` | unordered detector boxes |
| `detection_features` | `[T,M,256]` | frozen YOLO11n crop features |
| `detection_scores` | `[T,M]` | detector confidence |
| `detection_valid` | `[T,M]` | non-padding detection slots |
| `assignment` | `[T,N]` | matched detection index or `-1` |
| `appearance` | `[N,256]` | per-track feature prototype target |
| `camera_motion` | `[T,6]` | affine-motion token |
| `delta_time` | `[T]` | elapsed source frames per transition |
| `track_ids` | `[N]` | original GT identity or `-1` padding |
| `frame_numbers` | `[T]` | original frame numbers sampled into the clip |

Protocol metadata (detector, thresholds, capacities, interval, seed and
protocol version) is saved in every clip and summarized by
`tools/audit_dataset_clips.py`.

The assignment is only a training target. Detection order is independent of
track order, unmatched detections remain as false positives, and online
inference never receives this assignment.

## One-run training on all real datasets

After preparing the eight train/validation directories:

```bash
python tools/train.py \
  --config configs/real_multidataset_tiny.json \
  --set train.device=cuda:0
```

The multi-dataset loader balances datasets rather than allowing the largest one
to dominate. This remains one optimizer run and produces one final checkpoint.
The default real-data batch size is 2 for a single GPU.

During training, 35% of otherwise available matched observations are hidden
from the correction step. The world model must predict their future object
latents, with targets produced by an EMA copy of the 256-to-32 projection. This
adapts the latent-prediction principle of
[V-JEPA 2](https://arxiv.org/abs/2506.09985) to sparse object tokens; it does
not include or fine-tune the large V-JEPA 2 backbone.

## Evaluation

Evaluate the validation split stored in the checkpoint configuration:

```bash
python tools/evaluate.py \
  --checkpoint outputs/microworld_jepa_tiny/best.pt \
  --device cuda:0
```

Run the online tracker on an existing MOT `det.txt`:

```bash
python tools/run_mot.py \
  --checkpoint outputs/microworld_jepa_tiny/best.pt \
  --detections /datasets/MOT17/train/MOT17-02-FRCNN/det/det.txt \
  --image-width 1920 \
  --image-height 1080 \
  --output outputs/MOT17-02.txt \
  --device cuda:0
```

If `--features detections.npy` is omitted, appearance features are zero and the
run is motion-only. The feature file must have one row per `det.txt` row.

Run a complete cached validation split and write one MOT-format result per
sequence:

```bash
python tools/run_dataset.py \
  --checkpoint outputs/real_multidataset_tiny/best.pt \
  --format mot \
  --root data/raw/DanceTrack/val \
  --cache data/detector_cache/dancetrack/val \
  --output outputs/dancetrack_val \
  --device cuda:0 \
  --estimate-camera-motion
```

Add `--adaptive-detector` to test uncertainty-triggered detector skipping. The
result rows use
`frame,id,left,top,width,height,confidence,-1,-1,-1` and can be passed directly
to the official DanceTrack/MOTChallenge TrackEval scripts for HOTA, DetA, AssA,
MOTA and IDF1.

For a test split without public GT, create only the detector cache first:

```bash
python tools/prepare_real_data.py \
  --format mot \
  --root /datasets/MOT17/test \
  --dataset-name mot17 \
  --cache data/detector_cache/mot17/test \
  --detector yolo11n \
  --device cuda:0 \
  --cache-only
```

## Available model registry names

Baselines:

```text
constant_velocity
kalman
mlp_delta
gru_tiny
ssm_tiny
```

Proposed variants:

```text
microworld_det_{nano,tiny,small}
microworld_prob_{nano,tiny,small}
microworld_graph_{nano,tiny,small}
microworld_cf_{nano,tiny,small}
microworld_jepa_{nano,tiny,small}
microworld_adaptive_{nano,tiny,small}
```

`microworld_det` disables graph interaction and uses one motion mode;
`microworld_prob` adds multimodal prediction; `microworld_graph` adds sparse
interaction. `microworld_jepa` adds masked future-latent supervision with an
EMA target branch. Counterfactual and adaptive behavior remain controlled by
the tracker configuration.

## Important experimental rules

1. Keep the detector frozen for the initial paper study.
2. Report detector and tracker costs separately and end-to-end.
3. Compare against Kalman, GRU and SSM at matched state/feature inputs.
4. Report HOTA, AssA and IDF1 alongside parameter count, latency and energy.
5. Evaluate multiple detector cadences and occlusion lengths.
6. Use at least three seeds for final claims; one training run is enough only
   for architecture development, not statistical evidence.
