#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Batch conditional sampling for CIFAR-10 classes in a single Python process.
# This avoids reloading the model/checkpoint once per class.

WORKDIR="${WORKDIR:-/nfs/tangwenhao/lhp/cd-conditional-student-ft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/tangwenhao/lhp/conditional_samples}"
CONFIG="${CONFIG:-configs/cifar10_student_conditional_ft.py}"
CONDITIONING_TYPE="${CONDITIONING_TYPE:-adagn}"
CKPT="${CKPT:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GPUS="${GPUS:-0,1,2,3}"
SAVE_META_EVERY="${SAVE_META_EVERY:-8}"

mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPUS}" python "${SCRIPT_DIR}/scripts/sample_conditional_all_classes.py" \
  --config "${CONFIG}" \
  --workdir "${WORKDIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --ckpt "${CKPT}" \
  --conditioning-type "${CONDITIONING_TYPE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${EVAL_BATCH_SIZE}"

echo "Done. Conditional samples saved under ${OUTPUT_ROOT}"
