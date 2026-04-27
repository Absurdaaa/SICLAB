#!/usr/bin/env bash
set -euo pipefail

# Batch conditional sampling for CIFAR-10 classes.
# This creates:
#   ${OUTPUT_ROOT}/class_0
#   ...
#   ${OUTPUT_ROOT}/class_9
# Each class directory contains the eval shards produced by jcm.main --mode eval.

WORKDIR="${WORKDIR:-/nfs/tangwenhao/lhp/cd-conditional-student-ft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/tangwenhao/lhp/conditional_samples}"
CONFIG="${CONFIG:-configs/cifar10_student_conditional_ft.py}"
CKPT="${CKPT:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GPUS="${GPUS:-0,1,2,3}"
SAVE_META_EVERY="${SAVE_META_EVERY:-8}"

mkdir -p "${OUTPUT_ROOT}"

for CLASS_ID in 0 1 2 3 4 5 6 7 8 9; do
  CLASS_DIR="${OUTPUT_ROOT}/class_${CLASS_ID}"
  mkdir -p "${CLASS_DIR}"
  echo "Sampling class ${CLASS_ID} -> ${CLASS_DIR}"

  CUDA_VISIBLE_DEVICES="${GPUS}" python -m jcm.main \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}" \
    --mode eval \
    --eval_folder "class_${CLASS_ID}" \
    --config.eval.begin_ckpt="${CKPT}" \
    --config.eval.end_ckpt="${CKPT}" \
    --config.eval.num_samples="${NUM_SAMPLES}" \
    --config.eval.batch_size="${EVAL_BATCH_SIZE}" \
    --config.eval.enable_loss=False \
    --config.eval.enable_bpd=False \
    --config.eval.enable_sampling=True \
    --config.eval.save_meta_every="${SAVE_META_EVERY}" \
    --config.eval.aggregate_samples=False \
    --config.sampling.class_label="${CLASS_ID}"

  rm -rf "${CLASS_DIR}"
  cp -r "${WORKDIR}/class_${CLASS_ID}" "${CLASS_DIR}"
done

echo "Done. Conditional samples saved under ${OUTPUT_ROOT}"
