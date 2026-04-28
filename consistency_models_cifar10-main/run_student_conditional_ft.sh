#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Teacher-free conditional fine-tuning for a CD student.
# This script runs three variants:
#   1. true AdaGN conditioning
#   2. concatenation conditioning
#   3. cross-attention conditioning
# If the legacy WORKDIR override is set, treat it as the root directory and
# place each variant under a separate subdirectory to preserve compatibility.
resolve_workdir() {
  local specific="$1"
  local suffix="$2"
  local fallback="$3"

  if [[ -n "${specific}" ]]; then
    printf '%s\n' "${specific}"
  elif [[ -n "${WORKDIR:-}" ]]; then
    printf '%s/%s\n' "${WORKDIR}" "${suffix}"
  else
    printf '%s\n' "${fallback}"
  fi
}

WORKDIR_ADAGN="$(resolve_workdir "${WORKDIR_ADAGN:-}" "adagn" "/nfs/tangwenhao/lhp/cd-conditional-student-ft-adagn-true")"
WORKDIR_CONCAT="$(resolve_workdir "${WORKDIR_CONCAT:-}" "concat" "/nfs/tangwenhao/lhp/cd-conditional-student-ft-concat")"
WORKDIR_CROSS="$(resolve_workdir "${WORKDIR_CROSS:-}" "cross-attn" "/nfs/tangwenhao/lhp/cd-conditional-student-ft-cross-attn")"
CONFIG="${CONFIG:-configs/cifar10_student_conditional_ft.py}"
INIT_CKPT="${INIT_CKPT:-/nfs/tangwenhao/lhp/cd-lpips/checkpoints/checkpoint_25}"
GPUS="${GPUS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-500}"
SNAPSHOT_FREQ="${SNAPSHOT_FREQ:-2000}"
SAMPLE_CKPT="${SAMPLE_CKPT:-latest}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SAVE_META_EVERY="${SAVE_META_EVERY:-8}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
DATA_ROOT="${DATA_ROOT:-./data}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-256}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-64}"
GRID_SIZE="${GRID_SIZE:-8}"
STATS_CACHE="${STATS_CACHE:-}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"

run_train() {
  local workdir="$1"
  local conditioning_type="$2"

  CUDA_VISIBLE_DEVICES="${GPUS}" python -m jcm.main \
  --config "${CONFIG}" \
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
  --config.training.finetune_mode="${FINETUNE_MODE}" \
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
    echo "Skipping conditional evaluation for ${conditioning_type}: CLASSIFIER_CKPT is not set."
    return
  fi

  local stats_cache_dir="${STATS_CACHE:-${metrics_root}/real_stats_cache}"
  python "${SCRIPT_DIR}/scripts/evaluate_conditional_generation.py" \
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
  "${WORKDIR_CROSS}:cross_attn"
  "${WORKDIR_ADAGN}:adagn"
  "${WORKDIR_CONCAT}:concat"
)

for job in "${TRAIN_JOBS[@]}"; do
  workdir="${job%%:*}"
  conditioning_type="${job#*:}"
  run_train "${workdir}" "${conditioning_type}"
  sample_ckpt="$(resolve_sampling_ckpt "${workdir}")"
  run_sampling_and_eval "${workdir}" "${conditioning_type}" "${sample_ckpt}"
done
