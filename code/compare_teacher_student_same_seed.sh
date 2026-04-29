#!/usr/bin/env bash
set -euo pipefail

# Generate comparable samples from a teacher and a student using the same
# random seed so that the initial noise is aligned.
#
# The two runs are executed separately, but they share:
#   - the same SEED
#   - the same NUM_SAMPLES
#   - the same BATCH_SIZE
#   - the same optional CLASS_LABEL
#
# Output layout:
#   ${OUTDIR}/teacher/ckpt_${TEACHER_CKPT}_samples.npz
#   ${OUTDIR}/student/ckpt_${STUDENT_CKPT}_samples.npz

TEACHER_WORKDIR="${TEACHER_WORKDIR:-/nfs/tangwenhao/lhp/teacher_eval}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/cifar10_k_ve.py}"
TEACHER_CKPT="${TEACHER_CKPT:-1}"

STUDENT_WORKDIR="${STUDENT_WORKDIR:-/nfs/tangwenhao/lhp/cd-lpips}"
STUDENT_CONFIG="${STUDENT_CONFIG:-configs/cifar10_ve_cd.py}"
STUDENT_CKPT="${STUDENT_CKPT:-25}"

OUTDIR="${OUTDIR:-/path/to/compare_outputs}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GPUS="${GPUS:-0}"
SEED="${SEED:-42}"
CLASS_LABEL="${CLASS_LABEL:-}"
STUDENT_METHOD="${STUDENT_METHOD:-onestep}"
STUDENT_STEPS="${STUDENT_STEPS:-1}"

mkdir -p "${OUTDIR}/teacher" "${OUTDIR}/student"

teacher_cmd=(
  python -m jcm.main
  --config "${TEACHER_CONFIG}"
  --workdir "${TEACHER_WORKDIR}"
  --mode eval
  --eval_folder "teacher"
  --config.eval.begin_ckpt="${TEACHER_CKPT}"
  --config.eval.end_ckpt="${TEACHER_CKPT}"
  --config.eval.num_samples="${NUM_SAMPLES}"
  --config.eval.batch_size="${BATCH_SIZE}"
  --config.eval.enable_loss=False
  --config.eval.enable_bpd=False
  --config.eval.enable_sampling=True
  --config.eval.aggregate_samples=True
  --config.seed="${SEED}"
)

student_cmd=(
  python -m jcm.main
  --config "${STUDENT_CONFIG}"
  --workdir "${STUDENT_WORKDIR}"
  --mode eval
  --eval_folder "student"
  --config.eval.begin_ckpt="${STUDENT_CKPT}"
  --config.eval.end_ckpt="${STUDENT_CKPT}"
  --config.eval.num_samples="${NUM_SAMPLES}"
  --config.eval.batch_size="${BATCH_SIZE}"
  --config.eval.enable_loss=False
  --config.eval.enable_bpd=False
  --config.eval.enable_sampling=True
  --config.eval.aggregate_samples=True
  --config.seed="${SEED}"
  --config.sampling.method="${STUDENT_METHOD}"
  --config.sampling.n_steps="${STUDENT_STEPS}"
)

if [[ -n "${CLASS_LABEL}" ]]; then
  teacher_cmd+=(--config.sampling.class_label="${CLASS_LABEL}")
  student_cmd+=(--config.sampling.class_label="${CLASS_LABEL}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" "${teacher_cmd[@]}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${student_cmd[@]}"

teacher_npz="${TEACHER_WORKDIR}/teacher/ckpt_${TEACHER_CKPT}_samples.npz"
student_npz="${STUDENT_WORKDIR}/student/ckpt_${STUDENT_CKPT}_samples.npz"

if [[ ! -f "${teacher_npz}" ]]; then
  echo "Missing teacher samples: ${teacher_npz}" >&2
  exit 1
fi
if [[ ! -f "${student_npz}" ]]; then
  echo "Missing student samples: ${student_npz}" >&2
  exit 1
fi

cp "${teacher_npz}" "${OUTDIR}/teacher/ckpt_${TEACHER_CKPT}_samples.npz"
cp "${student_npz}" "${OUTDIR}/student/ckpt_${STUDENT_CKPT}_samples.npz"

echo "Saved teacher samples to ${OUTDIR}/teacher/ckpt_${TEACHER_CKPT}_samples.npz"
echo "Saved student samples to ${OUTDIR}/student/ckpt_${STUDENT_CKPT}_samples.npz"
