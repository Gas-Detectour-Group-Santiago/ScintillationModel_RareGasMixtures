#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

# Toy statistics used by every primary fit. Override from the command line:
#   PRIMARY_FIT_N_TOYS=100 JOINT_IR_N_TOYS=100 bash run_all.sh
export PRIMARY_FIT_N_TOYS="${PRIMARY_FIT_N_TOYS:-300}"
export JOINT_IR_N_TOYS="${JOINT_IR_N_TOYS:-$PRIMARY_FIT_N_TOYS}"

run_step() {
  local directory="$1"
  local script="$2"
  shift 2

  if [[ ! -d "$directory" ]]; then
    printf '\n[skip] %s/ does not exist\n' "$directory"
    return 0
  fi
  if [[ ! -f "$directory/$script" ]]; then
    printf '\n[skip] %s/%s does not exist\n' "$directory" "$script"
    return 0
  fi

  printf '\n[run] %s/%s' "$directory" "$script"
  if (( $# > 0 )); then
    printf ' %s' "$*"
  fi
  printf '\n'

  (
    cd "$directory"
    python3 "$script" "$@"
  )
}

run_stage() {
  local title="$1"
  printf '\n\n============================================================\n'
  printf '[stage] %s\n' "$title"
  printf '============================================================\n'
}

# -----------------------------------------------------------------------------
# 1. ANALYSIS
# Rebuild all curated experimental, spectral, Degrad and Garfield++ CSV inputs.
# Set RUN_DATA_ANALYSIS=0 only when the existing curated CSVs must be reused.
# -----------------------------------------------------------------------------
run_stage "1/5 ANALYSIS"
if [[ "${RUN_DATA_ANALYSIS:-1}" != "0" ]]; then
  run_step data run_analysis.py
else
  printf '\n[skip] data analysis disabled with RUN_DATA_ANALYSIS=0\n'
fi

# -----------------------------------------------------------------------------
# 2. PRIMARY PREDICTIONS
# The main runner regenerates tables, bands, multibands and electron/X-ray plots.
# The two extra runners cover low-pressure IR and the separate joint-IR products.
# Existing PDFs/CSVs are overwritten; nothing is deleted before starting.
# -----------------------------------------------------------------------------
run_stage "2/5 PRIMARY PREDICTIONS"
run_step primary_predictions run_primary_predictions.py
run_step primary_predictions run_primary_ir_low_pressure_predictions.py
run_step primary_predictions run_joint_ir_predictions.py

# -----------------------------------------------------------------------------
# 3. SECONDARY PREDICTIONS
# Regenerates paper plots, comparison plots, metadata plots and secondary tables.
# UV and VUV are already included in this complete runner.
# -----------------------------------------------------------------------------
run_stage "3/5 SECONDARY PREDICTIONS"
run_step secondary_predictions run_secondary_predictions.py

# -----------------------------------------------------------------------------
# 4. SPECTRA
# Regenerates raw, generated, comparison and annotated spectra and VUV tables.
# -----------------------------------------------------------------------------
run_stage "4/5 SPECTRA"
run_step spectra run_all_spectra.py

# -----------------------------------------------------------------------------
# 5. FITS
# Deliberately last: regenerates all independent fits plus the joint IR fit,
# including parameter tables, fit plots, toys, covariance and correlations.
# -----------------------------------------------------------------------------
run_stage "5/5 PRIMARY FITS"
run_step primary_fits run_primary_fits.py

printf '\n[done] full pipeline finished in the requested order.\n'
printf '[done] PRIMARY_FIT_N_TOYS=%s JOINT_IR_N_TOYS=%s RUN_DATA_ANALYSIS=%s\n' \
  "$PRIMARY_FIT_N_TOYS" "$JOINT_IR_N_TOYS" "${RUN_DATA_ANALYSIS:-1}"
