#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primary_predictions.auxiliares.electrons_xray_predictions import (  # noqa: E402
    export_electrons_xray_predictions,
)


def main() -> None:
    export_electrons_xray_predictions(PROJECT_ROOT, make_plots=True)


if __name__ == "__main__":
    main()
