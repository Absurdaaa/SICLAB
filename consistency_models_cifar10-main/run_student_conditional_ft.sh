#!/usr/bin/env bash
set -euo pipefail

# Teacher-free conditional fine-tuning for a CD student.
# This initializes the conditional student from an existing unconditional
# student checkpoint via `training.init_ckpt`, so only compatible weights are
# loaded and the new class embedding stays randomly initialized.

WORKDIR="${WORKDIR:-/nfs/tangwenhao/lhp/cd-conditional-student-ft}"
INIT_CKPT="${INIT_CKPT:-/nfs/tangwenhao/lhp/cd-lpips/checkpoints/checkpoint_25}"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-500}"
SNAPSHOT_FREQ="${SNAPSHOT_FREQ:-2000}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m jcm.main \
  --config configs/cifar10_student_conditional_ft.py \
  --workdir "${WORKDIR}" \
  --mode train \
  --config.training.init_ckpt="${INIT_CKPT}" \
  --config.training.batch_size="${BATCH_SIZE}" \
  --config.optim.lr="${LR}" \
  --config.training.n_iters="${N_ITERS}" \
  --config.training.log_freq="${LOG_FREQ}" \
  --config.training.eval_freq="${EVAL_FREQ}" \
  --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
  --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}"
