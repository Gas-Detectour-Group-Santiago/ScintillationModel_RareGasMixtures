#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"

failed=()

echo "Running all raw spectrum scripts..."
echo

for script in raw_spectra_*.py; do
    if [[ ! -f "$script" ]]; then
        continue
    fi

    echo ">>> $script"
    if "$PYTHON_BIN" "$script"; then
        echo "OK: $script"
    else
        echo "FAILED: $script"
        failed+=("$script")
    fi
    echo
done

if (( ${#failed[@]} > 0 )); then
    echo "Some scripts failed:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi

echo "All scripts finished successfully."
