#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SCINTILLATION_PROJECT_ROOT="$ROOT_DIR"
if [[ -z "${SCINTILLATION_STYLE_PRESET:-}" && -f "$ROOT_DIR/config/styles/active.txt" ]]; then
  SCINTILLATION_STYLE_PRESET="$(tr -d '[:space:]' < "$ROOT_DIR/config/styles/active.txt")"
fi
export SCINTILLATION_STYLE_PRESET="${SCINTILLATION_STYLE_PRESET:-tfm}"
export SCINTILLATION_STYLE_FILE="${SCINTILLATION_STYLE_FILE:-$ROOT_DIR/config/styles/$SCINTILLATION_STYLE_PRESET.json}"
export RECOMPUTE_BANDS="${RECOMPUTE_BANDS:-0}"
export RECOMPUTE_TABLES="${RECOMPUTE_TABLES:-0}"

RUN_PRIMARY="${RUN_PRIMARY:-1}"
RUN_SECONDARY="${RUN_SECONDARY:-1}"
RUN_SPECTRA="${RUN_SPECTRA:-1}"
RUN_TABLES="${RUN_TABLES:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"

if [[ "${ARCHIVE_OUTPUTS:-0}" == "1" ]]; then
  python3 - <<'PY_ARCHIVE'
from scintillation.core.outputs import OutputManager
print(OutputManager('.').archive_outputs())
PY_ARCHIVE
fi

if [[ "$RUN_PRIMARY" != "1" && "$RUN_SECONDARY" != "1" && "$RUN_SPECTRA" != "1" && "$RUN_TABLES" != "1" && "$RUN_DIAGNOSTICS" != "1" ]]; then
  echo "Nothing selected: enable at least one RUN_* product stage." >&2
  exit 2
fi

run_stage() {
  local enabled="$1"
  local label="$2"
  shift 2
  if [[ "$enabled" == "1" ]]; then
    printf '\n[run_products] %s\n' "$label"
    "$@"
  fi
}

run_stage "$RUN_PRIMARY" "primary predictions" python3 workflows/primary.py
run_stage "$RUN_SECONDARY" "secondary predictions" python3 workflows/secondary.py
run_stage "$RUN_SPECTRA" "spectra" python3 workflows/spectra.py
run_stage "$RUN_TABLES" "LaTeX/reference tables" python3 workflows/report.py
run_stage "$RUN_DIAGNOSTICS" "optional diagnostics" python3 workflows/diagnostics.py --all

printf '\n[done] products completed. PDFs: outputs/figures | LaTeX: outputs/tables\n'
