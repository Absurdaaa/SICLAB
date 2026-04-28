#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="${WORKDIR:-}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
CKPT="${CKPT:-latest}"
SAMPLES_ROOT="${SAMPLES_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda:0}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-256}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-64}"
GRID_SIZE="${GRID_SIZE:-8}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"
STATS_CACHE="${STATS_CACHE:-}"

if [[ -z "${WORKDIR}" ]]; then
  echo "WORKDIR is required."
  echo "Example:"
  echo "  WORKDIR=/path/to/adagn CLASSIFIER_CKPT=/path/to/best_classifier.pt bash evaluate_conditional_checkpoint.sh"
  exit 1
fi

if [[ -z "${CLASSIFIER_CKPT}" ]]; then
  echo "CLASSIFIER_CKPT is required."
  echo "Example:"
  echo "  WORKDIR=/path/to/adagn CLASSIFIER_CKPT=/path/to/best_classifier.pt bash evaluate_conditional_checkpoint.sh"
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/scripts/evaluate_conditional_checkpoint.py"
  --workdir "${WORKDIR}"
  --classifier-ckpt "${CLASSIFIER_CKPT}"
  --ckpt "${CKPT}"
  --data-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --classifier-batch-size "${CLASSIFIER_BATCH_SIZE}"
  --fid-batch-size "${FID_BATCH_SIZE}"
  --grid-size "${GRID_SIZE}"
  --mode "${FID_MODE}"
)

if [[ -n "${SAMPLES_ROOT}" ]]; then
  CMD+=(--samples-root "${SAMPLES_ROOT}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${STATS_CACHE}" ]]; then
  CMD+=(--stats-cache "${STATS_CACHE}")
fi

"${CMD[@]}"
