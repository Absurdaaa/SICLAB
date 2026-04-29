#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
TIMED_ITERS="${TIMED_ITERS:-10}"
CLASS_LABEL="${CLASS_LABEL:-none}"

TEACHER_NAME="${TEACHER_NAME:-teacher}"
TEACHER_WORKDIR="${TEACHER_WORKDIR:-/path/to/teacher_workdir}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/cifar10_k_ve.py}"
TEACHER_CKPT="${TEACHER_CKPT:-1}"
TEACHER_METHOD="${TEACHER_METHOD:-heun}"
TEACHER_STEPS="${TEACHER_STEPS:--1}"

STUDENT_NAME="${STUDENT_NAME:-student}"
STUDENT_WORKDIR="${STUDENT_WORKDIR:-/path/to/student_workdir}"
STUDENT_CONFIG="${STUDENT_CONFIG:-configs/cifar10_ve_cd.py}"
STUDENT_CKPT="${STUDENT_CKPT:-1}"
STUDENT_METHOD="${STUDENT_METHOD:-onestep}"
STUDENT_STEPS="${STUDENT_STEPS:-1}"
STUDENT_STD="${STUDENT_STD:--1}"
STUDENT_CLASS_CONDITIONAL="${STUDENT_CLASS_CONDITIONAL:-0}"
STUDENT_CONDITIONING_TYPE="${STUDENT_CONDITIONING_TYPE:-}"

OUTDIR="${OUTDIR:-${SCRIPT_DIR}/speed_benchmarks}"
mkdir -p "${OUTDIR}"

teacher_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/benchmark_sampling_speed.py"
  --name "${TEACHER_NAME}"
  --config "${TEACHER_CONFIG}"
  --workdir "${TEACHER_WORKDIR}"
  --ckpt "${TEACHER_CKPT}"
  --batch-size "${BATCH_SIZE}"
  --warmup-iters "${WARMUP_ITERS}"
  --timed-iters "${TIMED_ITERS}"
  --sampling-method "${TEACHER_METHOD}"
  --class-label "${CLASS_LABEL}"
  --output-json "${OUTDIR}/${TEACHER_NAME}.json"
)

if [[ "${TEACHER_STEPS}" -ge 0 ]]; then
  teacher_cmd+=(--sampling-steps "${TEACHER_STEPS}")
fi

student_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/benchmark_sampling_speed.py"
  --name "${STUDENT_NAME}"
  --config "${STUDENT_CONFIG}"
  --workdir "${STUDENT_WORKDIR}"
  --ckpt "${STUDENT_CKPT}"
  --batch-size "${BATCH_SIZE}"
  --warmup-iters "${WARMUP_ITERS}"
  --timed-iters "${TIMED_ITERS}"
  --sampling-method "${STUDENT_METHOD}"
  --sampling-steps "${STUDENT_STEPS}"
  --class-label "${CLASS_LABEL}"
  --output-json "${OUTDIR}/${STUDENT_NAME}.json"
)

if [[ "${STUDENT_STD}" != "-1" ]]; then
  student_cmd+=(--std "${STUDENT_STD}")
fi
if [[ "${STUDENT_CLASS_CONDITIONAL}" == "1" ]]; then
  student_cmd+=(--class-conditional)
fi
if [[ -n "${STUDENT_CONDITIONING_TYPE}" ]]; then
  student_cmd+=(--conditioning-type "${STUDENT_CONDITIONING_TYPE}")
fi

echo "[teacher] ${teacher_cmd[*]}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${teacher_cmd[@]}"

echo "[student] ${student_cmd[*]}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${student_cmd[@]}"

echo "Saved benchmark JSON to ${OUTDIR}"
