#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR_ROOT="${WORKDIR_ROOT:-/nfs/tangwenhao/lhp/conditional_student_ft_plus_head}"
VARIANT="${VARIANT:-adagn}"
CKPT="${CKPT:-latest}"
ROWS="${ROWS:-3}"
CELL_SIZE="${CELL_SIZE:-96}"
STRATEGY="${STRATEGY:-first}"
SEED="${SEED:-42}"
TITLE="${TITLE:-${VARIANT} conditional samples}"

resolve_ckpt() {
  local workdir="$1"

  if [[ "${CKPT}" != "latest" ]]; then
    printf '%s\n' "${CKPT}"
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

WORKDIR="${WORKDIR_ROOT}/${VARIANT}"
RESOLVED_CKPT="$(resolve_ckpt "${WORKDIR}")"
SAMPLES_ROOT="${WORKDIR}/conditional_samples_ckpt_${RESOLVED_CKPT}"
OUTPUT="${WORKDIR}/conditional_grid_ckpt_${RESOLVED_CKPT}.png"

if [[ ! -d "${SAMPLES_ROOT}" ]]; then
  echo "Samples root not found: ${SAMPLES_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/make_conditional_class_grid.py" \
  --samples-root "${SAMPLES_ROOT}" \
  --output "${OUTPUT}" \
  --rows "${ROWS}" \
  --cell-size "${CELL_SIZE}" \
  --strategy "${STRATEGY}" \
  --seed "${SEED}" \
  --title "${TITLE}"

echo "Saved grid image to ${OUTPUT}"
