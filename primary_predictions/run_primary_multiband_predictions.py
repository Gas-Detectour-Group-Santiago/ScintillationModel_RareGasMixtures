from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primary_predictions.auxiliares import PredictionRunner  # noqa: E402
from primary_predictions.configs import (  # noqa: E402
    PRIMARY_ADAPTERS,
    arcf4_ir_multiband_plots,
)


def main(*, overwrite_bands: bool = False):
    runner = PredictionRunner(PROJECT_ROOT, PRIMARY_ADAPTERS)
    runner.run_multi_bands(arcf4_ir_multiband_plots(), overwrite=overwrite_bands)


if __name__ == "__main__":
    main()
