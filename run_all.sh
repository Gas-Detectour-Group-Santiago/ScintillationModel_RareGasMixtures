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

RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_FITS="${RUN_FITS:-1}"
RUN_PRODUCTS="${RUN_PRODUCTS:-1}"
TOYS="${PRIMARY_FIT_N_TOYS:-100}"

if [[ "$RUN_ANALYSIS" != "1" && "$RUN_FITS" != "1" && "$RUN_PRODUCTS" != "1" ]]; then
  echo "Nothing selected: enable RUN_ANALYSIS, RUN_FITS or RUN_PRODUCTS." >&2
  exit 2
fi

if [[ "${ARCHIVE_OUTPUTS:-0}" == "1" ]]; then
  python3 - <<'PY'
from scintillation.core.outputs import OutputManager
print(OutputManager('.').archive_outputs())
PY
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  printf '\n[run_all] analysis\n'
  python3 workflows/analysis.py --refresh-runtime
fi

if [[ "$RUN_FITS" == "1" ]]; then
  printf '\n[run_all] fits (%s stat + %s syst toys)\n' "$TOYS" "$TOYS"
  python3 workflows/fits.py --toys "$TOYS"
fi

if [[ "$RUN_PRODUCTS" == "1" ]]; then
  # New fits invalidate cached prediction bands. If no fit was run, preserve the
  # caller choice and otherwise reuse existing caches by default.
  if [[ "$RUN_FITS" == "1" ]]; then
    export RECOMPUTE_BANDS=1
    export RECOMPUTE_TABLES=1
  else
    export RECOMPUTE_BANDS="${RECOMPUTE_BANDS:-0}"
    export RECOMPUTE_TABLES="${RECOMPUTE_TABLES:-0}"
  fi
  printf '\n[run_all] products\n'
  bash run_products.sh
fi

printf '\n[done] selected pipeline stages completed.\n'
