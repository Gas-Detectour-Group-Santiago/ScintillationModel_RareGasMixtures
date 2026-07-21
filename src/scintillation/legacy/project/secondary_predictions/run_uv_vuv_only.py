from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secondary_predictions.auxiliares import PredictionRunner  # noqa: E402
from secondary_predictions.auxiliares.ar2nd_population_upgrade import (  # noqa: E402
    ensure_secondary_ar2nd_populations,
)
from secondary_predictions.configs import SECONDARY_ADAPTERS  # noqa: E402
from secondary_predictions import config_paper  # noqa: E402


def main(*, overwrite: bool = True):
    """Generate only the paper UV (200--400 nm) and VUV (100--200 nm) plots."""
    for report in ensure_secondary_ar2nd_populations(PROJECT_ROOT):
        if report.updated_rows:
            print(
                f"[secondary_predictions] Ar2nd populations actualizadas: "
                f"{report.population_csv} ({report.updated_rows} filas)"
            )
        if report.missing_level_csvs:
            print(
                f"[secondary_predictions] aviso: faltan {len(report.missing_level_csvs)} "
                f"tablas de niveles para {report.population_csv}"
            )

    configs = [
        cfg
        for cfg in config_paper.multiband_plots()
        if "200_400nm" in cfg.id or "100_200nm" in cfg.id
    ]
    runner = PredictionRunner(
        PROJECT_ROOT,
        SECONDARY_ADAPTERS,
        predictions_subdir=Path("Predictions") / "Secondary",
        log_prefix="[secondary_predictions]",
    )
    return runner.run_multi_bands(configs, overwrite=overwrite)


if __name__ == "__main__":
    main()
