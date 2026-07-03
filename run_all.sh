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

run_step primary_predictions run_primary_predictions.py
run_step secondary_predictions run_secondary_predictions.py
run_step spectra run_all_spectra.py
run_step integral_comparations run_integral_comparisons.py
run_step cross_sections plot_cross_section.py
run_step populations_histograms run_population_histograms.py

# Leave primary fits for the end: these are the slowest/noisiest outputs and
# PRIMARY_FIT_N_TOYS defaults to 300 above.
run_step primary_fits run_primary_fits.py

printf '\n[done] full plotting pipeline finished. PRIMARY_FIT_N_TOYS=%s\n' "$PRIMARY_FIT_N_TOYS"
