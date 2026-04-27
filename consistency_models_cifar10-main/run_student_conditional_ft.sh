#!/usr/bin/env bash
set -euo pipefail

# Teacher-free conditional fine-tuning for a CD student.
# This script runs three variants:
#   1. true AdaGN conditioning
#   2. concatenation conditioning
#   3. cross-attention conditioning
# If the legacy WORKDIR override is set, treat it as the root directory and
# place each variant under a separate subdirectory to preserve compatibility.
WORKDIR_ROOT="${WORKDIR:-}"
WORKDIR_ADAGN="${WORKDIR_ADAGN:-${WORKDIR_ROOT:+${WORKDIR_ROOT}/adagn}}"
WORKDIR_CONCAT="${WORKDIR_CONCAT:-${WORKDIR_ROOT:+${WORKDIR_ROOT}/concat}}"
WORKDIR_CROSS="${WORKDIR_CROSS:-${WORKDIR_ROOT:+${WORKDIR_ROOT}/cross-attn}}"
WORKDIR_ADAGN="${WORKDIR_ADAGN:-/nfs/tangwenhao/lhp/cd-conditional-student-ft-adagn-true}"
WORKDIR_CONCAT="${WORKDIR_CONCAT:-/nfs/tangwenhao/lhp/cd-conditional-student-ft-concat}"
WORKDIR_CROSS="${WORKDIR_CROSS:-/nfs/tangwenhao/lhp/cd-conditional-student-ft-cross-attn}"
INIT_CKPT="${INIT_CKPT:-/nfs/tangwenhao/lhp/cd-lpips/checkpoints/checkpoint_25}"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-500}"
SNAPSHOT_FREQ="${SNAPSHOT_FREQ:-2000}"

run_train() {
  local workdir="$1"
  local conditioning_type="$2"

  CUDA_VISIBLE_DEVICES="${GPUS}" python -m jcm.main \
  --config configs/cifar10_student_conditional_ft.py \
  --workdir "${workdir}" \
  --mode train \
  --config.training.init_ckpt="${INIT_CKPT}" \
  --config.training.batch_size="${BATCH_SIZE}" \
  --config.optim.lr="${LR}" \
  --config.training.n_iters="${N_ITERS}" \
  --config.training.log_freq="${LOG_FREQ}" \
  --config.training.eval_freq="${EVAL_FREQ}" \
  --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
  --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}" \
  --config.model.class_conditional=True \
  --config.model.conditioning_type="${conditioning_type}" \
  --config.model.num_classes=10
}

run_train "${WORKDIR_CROSS}" "cross_attn"
run_train "${WORKDIR_ADAGN}" "adagn"
run_train "${WORKDIR_CONCAT}" "concat"

