#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remote-friendly launcher for conditional student fine-tuning.
#
# What it does:
# 1. Fine-tunes an existing unconditional student checkpoint with DSM loss.
# 2. Runs three conditioning variants: adagn / concat / cross_attn.
# 3. Performs small-scale conditional sampling after training.
# 4. Optionally runs classifier-based conditional evaluation if CLASSIFIER_CKPT is set.
#
# Minimal usage:
#   INIT_CKPT=/path/to/student/checkpoint_XX \
#   WORKDIR_ROOT=/path/to/outputs \
#   bash run_student_conditional_ft_remote.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/cifar10_student_conditional_ft.py}"
INIT_CKPT="${INIT_CKPT:-}"
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRIPT_DIR}/outputs/conditional_student_ft}"
GPUS="${GPUS:-0}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-200}"
SNAPSHOT_FREQ="${SNAPSHOT_FREQ:-1000}"

SAMPLE_CKPT="${SAMPLE_CKPT:-latest}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
SAVE_META_EVERY="${SAVE_META_EVERY:-2}"

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
DATA_ROOT="${DATA_ROOT:-./data}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-128}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-64}"
GRID_SIZE="${GRID_SIZE:-8}"
STATS_CACHE="${STATS_CACHE:-}"

if [[ -z "${INIT_CKPT}" ]]; then
  echo "INIT_CKPT is required, e.g. /path/to/checkpoints/checkpoint_25" >&2
  exit 1
fi

mkdir -p "${WORKDIR_ROOT}"

run_train() {
  local workdir="$1"
  local conditioning_type="$2"

  mkdir -p "${workdir}"
  echo "[train] ${conditioning_type} -> ${workdir}"

  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m jcm.main \
    --config "${CONFIG}" \
    --workdir "${workdir}" \
    --mode train \
    --config.training.init_ckpt="${INIT_CKPT}" \
    --config.training.batch_size="${TRAIN_BATCH_SIZE}" \
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

resolve_sampling_ckpt() {
  local workdir="$1"

  if [[ "${SAMPLE_CKPT}" != "latest" ]]; then
    printf '%s\n' "${SAMPLE_CKPT}"
    return
  fi

  local ckpt_dir="${workdir}/checkpoints"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "No checkpoint directory found under ${ckpt_dir}" >&2
    return 1
  fi

  local latest
  latest="$(find "${ckpt_dir}" -maxdepth 1 -type f -name 'checkpoint_*' | sed 's#.*/checkpoint_##' | sort -n | tail -1)"
  if [[ -z "${latest}" ]]; then
    echo "No checkpoint files found under ${ckpt_dir}" >&2
    return 1
  fi
  printf '%s\n' "${latest}"
}

run_sampling_and_eval() {
  local workdir="$1"
  local conditioning_type="$2"
  local sample_ckpt="$3"
  local samples_root="${workdir}/conditional_samples_ckpt_${sample_ckpt}"
  local metrics_root="${workdir}/conditional_metrics_ckpt_${sample_ckpt}"

  echo "[sample] ${conditioning_type} ckpt=${sample_ckpt}"
  WORKDIR="${workdir}" \
  OUTPUT_ROOT="${samples_root}" \
  CONFIG="${CONFIG}" \
  CONDITIONING_TYPE="${conditioning_type}" \
  CKPT="${sample_ckpt}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  GPUS="${GPUS}" \
  SAVE_META_EVERY="${SAVE_META_EVERY}" \
  bash "${SCRIPT_DIR}/run_conditional_sampling.sh"

  if [[ -z "${CLASSIFIER_CKPT}" ]]; then
    echo "[eval] skip ${conditioning_type}: CLASSIFIER_CKPT is not set"
    return
  fi

  local stats_cache_dir="${STATS_CACHE:-${metrics_root}/real_stats_cache}"
  echo "[eval] ${conditioning_type}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/evaluate_conditional_generation.py" \
    --samples-root "${samples_root}" \
    --classifier-ckpt "${CLASSIFIER_CKPT}" \
    --output-dir "${metrics_root}" \
    --data-root "${DATA_ROOT}" \
    --device "${EVAL_DEVICE}" \
    --classifier-batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --fid-batch-size "${FID_BATCH_SIZE}" \
    --grid-size "${GRID_SIZE}" \
    --stats-cache "${stats_cache_dir}"
}

declare -a TRAIN_JOBS=(
  "adagn:adagn"
  "concat:concat"
  "cross_attn:cross_attn"
)

for job in "${TRAIN_JOBS[@]}"; do
  name="${job%%:*}"
  conditioning_type="${job#*:}"
  workdir="${WORKDIR_ROOT}/${name}"

  run_train "${workdir}" "${conditioning_type}"
  sample_ckpt="$(resolve_sampling_ckpt "${workdir}")"
  run_sampling_and_eval "${workdir}" "${conditioning_type}" "${sample_ckpt}"
done

echo "All runs finished under ${WORKDIR_ROOT}"
