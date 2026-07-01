from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primary_predictions.auxiliares import PredictionRunner  # noqa: E402
from primary_predictions.configs import (  # noqa: E402
    COMMON_ARCF4_NORM,
    COMMON_ARN2_NORM,
    OWN_NORM,
    PRIMARY_ADAPTERS,
    arcf4_ir_multiband_plots,
    primary_band_plots,
    selected_primary_points,
)


def main(
    *,
    make_tables: bool = True,
    make_bands: bool = True,
    make_plots: bool = True,
    make_multibands: bool = True,
    overwrite_bands: bool = True,
):
    runner = PredictionRunner(PROJECT_ROOT, PRIMARY_ADAPTERS)

    if make_tables:
        runner.run_points(
            selected_primary_points(COMMON_ARCF4_NORM),
            "primary_selected_yields_common_arcf4_norm",
            caption=(
                r"Predicciones primarias seleccionadas en ph/MeV usando una escala "
                r"común fijada por la normalización de Ar--CF$_4$."
            ),
            label="tab:primary_selected_yields_common_arcf4_norm",
        )

        runner.run_points(
            selected_primary_points(OWN_NORM),
            "primary_selected_yields_own_norm",
            caption=(
                r"Predicciones primarias seleccionadas en ph/MeV normalizadas con "
                r"la normalización propia de cada ajuste."
            ),
            label="tab:primary_selected_yields_own_norm",
        )

        runner.run_normalization_comparison_points(
            selected_primary_points(OWN_NORM, force_common_normalization=True),
            "primary_selected_yields_arcf4_vs_arn2_norm",
            left_normalization=COMMON_ARCF4_NORM,
            right_normalization=COMMON_ARN2_NORM,
            caption=(
                r"Predicciones primarias seleccionadas en ph/MeV evaluadas con "
                r"normalización de Ar--CF$_4$ y de Ar--N$_2$."
            ),
            label="tab:primary_selected_yields_arcf4_vs_arn2_norm",
        )

    if make_bands:
        runner.run_bands(primary_band_plots(OWN_NORM), make_plots=make_plots, overwrite=overwrite_bands)

    if make_multibands:
        runner.run_multi_bands(arcf4_ir_multiband_plots(), overwrite=overwrite_bands)


if __name__ == "__main__":
    main()
