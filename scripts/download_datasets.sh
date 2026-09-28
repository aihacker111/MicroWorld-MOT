#!/usr/bin/env bash
set -euo pipefail

# Resumable train/validation downloader for the official DanceTrack, MOT16,
# MOT17 and VisDrone-MOT archives. Dataset terms still apply; this script does
# not redistribute or mirror any data.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${DATA_ROOT}/.archives}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"

usage() {
  cat <<'EOF'
Usage:
  ACCEPT_DATASET_TERMS=1 bash scripts/download_datasets.sh [all|DATASET ...]

Datasets:
  dancetrack  DanceTrack train1/train2/val from the official Hugging Face repo
  mot16       MOTChallenge MOT16 archive (use train sequences for train/val)
  mot17       MOTChallenge MOT17 archive (use train sequences for train/val)
  visdrone    VisDrone2019-MOT train/val

Environment variables:
  DATA_ROOT=/path          Optional override (default: <repository>/data/raw)
  ARCHIVE_ROOT=/path       Archive cache (default: DATA_ROOT/.archives)
  KEEP_ARCHIVES=0|1        Remove archives after extraction (default: 1)
  DANCETRACK_PARTS="..."   Optional subset: train1 train2 val

The full set is large. Downloads are resumable and existing extracted files
are not overwritten.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${ACCEPT_DATASET_TERMS:-0}" != "1" ]]; then
  echo "Refusing to download until you confirm the official dataset terms." >&2
  echo "Re-run with ACCEPT_DATASET_TERMS=1 after reviewing docs/DATASETS.md." >&2
  exit 2
fi

for command_name in curl unzip python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done

mkdir -p "${DATA_ROOT}" "${ARCHIVE_ROOT}"

download_url() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  if [[ -f "${output}" ]] && unzip -tq "${output}" >/dev/null 2>&1; then
    echo "Using complete archive: ${output}"
    return
  fi
  echo "Downloading $(basename "${output}")"
  curl --fail --location --retry 5 --retry-delay 5 \
    --continue-at - --output "${output}" "${url}"
  if ! unzip -tq "${output}" >/dev/null 2>&1; then
    echo "Downloaded file is not a valid ZIP archive: ${output}" >&2
    exit 1
  fi
}

download_gdrive() {
  local file_id="$1"
  local output="$2"
  if ! python3 -c 'import gdown' >/dev/null 2>&1; then
    echo "VisDrone download requires gdown." >&2
    echo "Install it with: python -m pip install -e '.[datasets]'" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${output}")"
  if [[ -f "${output}" ]] && unzip -tq "${output}" >/dev/null 2>&1; then
    echo "Using complete archive: ${output}"
    return
  fi
  echo "Downloading $(basename "${output}")"
  python3 -m gdown --id "${file_id}" --continue --output "${output}"
  if ! unzip -tq "${output}" >/dev/null 2>&1; then
    echo "Downloaded file is not a valid ZIP archive: ${output}" >&2
    exit 1
  fi
}

extract_zip() {
  local archive="$1"
  local destination="$2"
  mkdir -p "${destination}"
  echo "Extracting $(basename "${archive}") -> ${destination}"
  unzip -q -n "${archive}" -d "${destination}"
  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -f -- "${archive}"
  fi
}

download_dancetrack() {
  local destination="${DATA_ROOT}/DanceTrack"
  local requested_parts="${DANCETRACK_PARTS:-train1 train2 val}"
  local part
  for part in ${requested_parts}; do
    case "${part}" in
      train1|train2|val) ;;
      *)
        echo "Unknown DanceTrack part: ${part}; expected train1, train2 or val" >&2
        exit 2
        ;;
    esac
    local archive="${ARCHIVE_ROOT}/DanceTrack/${part}.zip"
    download_url \
      "https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/${part}.zip?download=true" \
      "${archive}"
    extract_zip "${archive}" "${destination}"
  done
}

download_mot16() {
  local archive="${ARCHIVE_ROOT}/MOT16/MOT16.zip"
  download_url "https://motchallenge.net/data/MOT16.zip" "${archive}"
  extract_zip "${archive}" "${DATA_ROOT}"
}

download_mot17() {
  local archive="${ARCHIVE_ROOT}/MOT17/MOT17.zip"
  download_url "https://motchallenge.net/data/MOT17.zip" "${archive}"
  extract_zip "${archive}" "${DATA_ROOT}"
}

download_visdrone() {
  local destination="${DATA_ROOT}/VisDrone"
  local names=(train val)
  local ids=(
    "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh"
    "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
  )
  local index
  for index in "${!names[@]}"; do
    local archive="${ARCHIVE_ROOT}/VisDrone/VisDrone2019-MOT-${names[$index]}.zip"
    download_gdrive "${ids[$index]}" "${archive}"
    extract_zip "${archive}" "${destination}"
  done
}

if [[ "$#" -eq 0 ]]; then
  set -- all
fi

datasets=()
for requested in "$@"; do
  requested_lower="$(printf '%s' "${requested}" | tr '[:upper:]' '[:lower:]')"
  case "${requested_lower}" in
    all)
      datasets=(dancetrack mot16 mot17 visdrone)
      break
      ;;
    dancetrack|mot16|mot17|visdrone)
      datasets+=("${requested_lower}")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown dataset: ${requested}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Repository root: ${PROJECT_ROOT}"
echo "Dataset root: ${DATA_ROOT}"
echo "Archive cache: ${ARCHIVE_ROOT}"
df -h "${DATA_ROOT}" | tail -n 1 || true

for dataset_name in "${datasets[@]}"; do
  "download_${dataset_name}"
done

echo "Download and extraction complete. Archives kept: ${KEEP_ARCHIVES}"
