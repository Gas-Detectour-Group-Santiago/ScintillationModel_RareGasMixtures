#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
# Paper default: keep pseudo-experiments high enough for stable bands.
# Override with, e.g.: PRIMARY_FIT_N_TOYS=100 bash run_all.sh
export PRIMARY_FIT_N_TOYS="${PRIMARY_FIT_N_TOYS:-300}"

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

  printf '\n[run] %s/%s %s\n' "$directory" "$script" "$*"
  (
    cd "$directory"
    python3 "$script" "$@"
  )
}

# Rebuild curated CSV inputs from raw pickles/TXT/ROOT files unless explicitly skipped.
# Use RUN_DATA_ANALYSIS=0 bash run_all.sh to reuse already exported CSVs.
if [[ "${RUN_DATA_ANALYSIS:-1}" != "0" ]]; then
  run_step data run_analysis.py
fi

# Fits must run before predictions when a fully fresh analysis is requested.
# These are the slowest/noisiest outputs; PRIMARY_FIT_N_TOYS defaults to 300 above.
run_step data run_analysis.py
run_step primary_fits run_primary_fits.py

run_step primary_predictions run_primary_predictions.py
run_step secondary_predictions run_secondary_predictions.py
run_step spectra run_all_spectra.py
run_step integral_comparations run_integral_comparisons.py
run_step cross_sections plot_cross_section.py
run_step populations_histograms run_population_histograms.py

printf '
[done] full analysis pipeline finished. PRIMARY_FIT_N_TOYS=%s RUN_DATA_ANALYSIS=%s
' "$PRIMARY_FIT_N_TOYS" "${RUN_DATA_ANALYSIS:-1}"
