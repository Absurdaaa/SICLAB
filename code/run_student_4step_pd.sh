#!/usr/bin/env bash
set -euo pipefail

# Progressive distillation training for a CIFAR-10 student that reaches the
# 4-step stage during training.
#
# The schedule in configs/cifar10_ve_progressive_distillation.py halves the
# number of scales every `distill_steps_per_iter` steps. With
# `start_scales=4096` and `distill_steps_per_iter=50000`, the 4-step stage
# begins around step 500000.

WORKDIR="${WORKDIR:-/nfs/tangwenhao/lhp/pd-4step-cifar10}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/tangwenhao/lhp/SICLAB/consistency_models_cifar10-main/checkpoints/edm_cifar10_ema}"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-5e-5}"
N_ITERS="${N_ITERS:-800001}"
SNAPSHOT_FREQ="${SNAPSHOT_FREQ:-10000}"
PREEMPT_FREQ="${PREEMPT_FREQ:-5000}"
JITTED_STEPS="${JITTED_STEPS:-10}"
DISTILL_STEPS_PER_ITER="${DISTILL_STEPS_PER_ITER:-50000}"
START_SCALES="${START_SCALES:-4096}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m jcm.main \
  --config configs/cifar10_ve_progressive_distillation.py \
  --workdir "${WORKDIR}" \
  --mode train \
  --config.training.ref_model_path="${REF_MODEL_PATH}" \
  --config.training.n_iters="${N_ITERS}" \
  --config.training.n_jitted_steps="${JITTED_STEPS}" \
  --config.training.batch_size="${BATCH_SIZE}" \
  --config.training.distill_steps_per_iter="${DISTILL_STEPS_PER_ITER}" \
  --config.training.start_scales="${START_SCALES}" \
  --config.optim.lr="${LR}" \
  --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
  --config.training.snapshot_freq_for_preemption="${PREEMPT_FREQ}"
