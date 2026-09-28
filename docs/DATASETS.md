# Dataset sources and split notes

The repository contains adapters, not dataset files. Accept each dataset's
terms and download it from the official source:

- DanceTrack: <https://github.com/DanceTrack/DanceTrack>
- MOT17: <https://motchallenge.net/data/MOT17/>
- MOT16: <https://motchallenge.net/data/MOT16/>
- VisDrone: <https://github.com/VisDrone/VisDrone-Dataset>

## Resumable download

Install `gdown` (needed for the official VisDrone Google Drive files), confirm
that you have reviewed the dataset terms, and choose a sufficiently large disk:

```bash
python -m pip install -e '.[datasets]'
ACCEPT_DATASET_TERMS=1 bash scripts/download_datasets.sh all
```

The script uses `curl --continue-at -` for HTTP downloads and `gdown
--continue` for VisDrone. It downloads only DanceTrack train1/train2/val and
VisDrone train/val; their test archives are intentionally skipped. MOT16/17
use the official benchmark archives, with train sequences split locally into
training and validation. Archives stay under `data/raw/.archives` by default,
so an interrupted run can resume. Set `KEEP_ARCHIVES=0` only if you want each
archive removed after successful extraction.

The default root is `<repository>/data/raw`. For example, when the repository
is `/media/hung/HDD/workplaces/tin/cvpr2027/MicroWorld-MOT`, the resolved data
root is `/media/hung/HDD/workplaces/tin/cvpr2027/MicroWorld-MOT/data/raw`.
Do not export `DATA_ROOT=/home/...` when the home partition is not the intended
storage device.

To resume only one DanceTrack archive after moving a partial download, set for
example `DANCETRACK_PARTS=val`; accepted values are `train1`, `train2`, `val`,
or a space-separated combination.

## COCO conversion

The converter accepts either a dataset-wide root or a split root and discovers
the official nested layout automatically:

```bash
python tools/convert_tracking_to_coco.py \
  --dataset dancetrack --root data/raw/DanceTrack \
  --output data/coco/dancetrack --splits train val

python tools/convert_tracking_to_coco.py \
  --dataset visdrone --root data/raw/VisDrone \
  --output data/coco/visdrone --splits train val \
  --category-mode native
```

`native` preserves labeled source classes (including MOT distractor/occluder
labels). `person` is the safer choice for standard MOT/DanceTrack experiments:
it creates one common person class from MOT/DanceTrack class 1 and VisDrone
classes 1 and 2. Ignored rows, invalid boxes and VisDrone's non-evaluated
category 11 are excluded. Bounding boxes are clipped to image boundaries. In
addition to standard COCO keys, the output preserves the temporal identifiers
needed by MOT code.

## Leakage warning

MOT17 repeats the same videos three times for DPM, FRCNN and SDP detections. The
preparation tool keeps one copy when a new detector is used. MOT17 also reuses
MOT16 sequences with revised annotations. Treating MOT16 and MOT17 as independent
training/validation domains therefore leaks video content.

The supplied preparation script uses sequence-level splits inside each dataset,
but the user is responsible for deciding whether to include both MOT16 and
MOT17 in the final paper protocol. A cleaner joint-training set is normally
DanceTrack + MOT17 + VisDrone, with MOT16 reported as an overlapping legacy
benchmark rather than an independent generalization set.

## Test annotations

Some official test annotations are private. `prepare_real_data.py` needs GT and
is therefore for train/validation preparation. Use `run_dataset.py` with cached
detections to generate test submissions without GT; submit the resulting files
through the official benchmark procedure.
