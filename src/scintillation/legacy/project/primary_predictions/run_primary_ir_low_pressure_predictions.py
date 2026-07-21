from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primary_predictions.auxiliares import PredictionRunner  # noqa: E402
from primary_predictions.configs import (  # noqa: E402
    OWN_NORM,
    PRIMARY_ADAPTERS,
    primary_ir_low_pressure_band_plots,
)


def main(*, make_plots: bool = True, overwrite_bands: bool = True):
    runner = PredictionRunner(PROJECT_ROOT, PRIMARY_ADAPTERS)
    runner.run_bands(
        primary_ir_low_pressure_band_plots(OWN_NORM),
        make_plots=make_plots,
        overwrite=overwrite_bands,
    )


if __name__ == "__main__":
    recompute = os.environ.get("RECOMPUTE_BANDS", "0").lower() in {"1", "true", "yes", "on"}
    main(overwrite_bands=recompute)
