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
    arn2_ir_multiband_plots_arcf4_norm,
    primary_band_plots,
    pure_ar_low_pressure_ir_points,
    selected_primary_points,
    vuv_primary_points,
)
from primary_predictions.auxiliares.n2_pure_energy_tables import (  # noqa: E402
    export_n2_pure_energy_prediction_table,
)
from primary_predictions.auxiliares.electrons_xray_predictions import (  # noqa: E402
    export_electrons_xray_predictions,
)


def main(
    *,
    make_tables: bool = True,
    make_bands: bool = True,
    make_plots: bool = True,
    make_multibands: bool = True,
    make_electrons_xray: bool = True,
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

        export_n2_pure_energy_prediction_table(
            PROJECT_ROOT,
            "N2_pure_predictions_by_incident_type",
            left_normalization=COMMON_ARCF4_NORM,
            right_normalization=COMMON_ARN2_NORM,
            caption=(
                r"Predicciones de N$_2$ puro para distintas entradas de Degrad: "
                r"rayos X de 12 keV y electrones de 1.5 MeV a 0 y 50 V/cm. "
                r"Las poblaciones de 1.5 MeV se dividen por 1500 keV antes de "
                r"convertir a ph/MeV."
            ),
            label="tab:n2_pure_predictions_by_incident_type",
        )

        runner.run_values_by_normalization(
            vuv_primary_points(OWN_NORM),
            "primary_vuv_absolute_yields_by_norm",
            normalizations={
                "value": OWN_NORM,
            },
            column_headings={
                "value": r"Valor",
            },
            caption=(
                r"Predicciones de las dos componentes VUV primarias añadidas al modelo "
                r"en ph/MeV: segundo continuo total de argón y rama "
                r"CF$_4^+{}^*(D)\to$CF$_4^+(X)$."
            ),
            label="tab:primary_vuv_absolute_yields_by_norm",
        )

        low_pressure_df = runner.run_normalization_comparison_points(
            pure_ar_low_pressure_ir_points(OWN_NORM),
            "primary_low_pressure_pure_ar_ir_arcf4_vs_arn2_norm",
            left_normalization=COMMON_ARCF4_NORM,
            right_normalization=COMMON_ARN2_NORM,
            caption=(
                r"Predicciones IR primarias en el límite de argón puro para la "
                r"extrapolación a baja presión, evaluadas con normalización de "
                r"Ar--CF$_4$ y de Ar--N$_2$. El límite se aproxima con la "
                r"menor fracción mostrada en cada extrapolación: 0.001\% de "
                r"CF$_4$ y 0.01\% de N$_2$."
            ),
            label="tab:primary_low_pressure_pure_ar_ir_arcf4_vs_arn2_norm",
        )

        runner.run_pure_ar_model_average_table(
            low_pressure_df,
            "primary_low_pressure_pure_ar_ir_model_average",
            caption=(
                r"Predicción IR primaria media en el límite de argón puro para la "
                r"extrapolación a baja presión. La columna media es el promedio "
                r"aritmético de las extrapolaciones Ar--CF$_4$ y Ar--N$_2$; "
                r"$\Delta_{\mathrm{modelo}}$ es la semidiferencia entre ambas. "
                r"Cada modelo se evalúa en la menor fracción mostrada en su "
                r"gráfica de baja presión."
            ),
            label="tab:primary_low_pressure_pure_ar_ir_model_average",
        )

    if make_bands:
        runner.run_bands(primary_band_plots(OWN_NORM), make_plots=make_plots, overwrite=overwrite_bands)

    if make_multibands:
        runner.run_multi_bands(
            arcf4_ir_multiband_plots() + arn2_ir_multiband_plots_arcf4_norm(),
            overwrite=overwrite_bands,
        )

    if make_electrons_xray:
        export_electrons_xray_predictions(PROJECT_ROOT, make_plots=make_plots)


if __name__ == "__main__":
    main()
