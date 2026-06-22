from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ("models", "primary_fits"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from auxiliares import DatasetSpec, FitConfig, Parameter, PlotSpec, PrimaryFitRunner, SystematicSource, ToySpec
from ArN2_infrarred import (
    W_ArN2,
    theory_yield_ArN2_Ir_696,
    theory_yield_ArN2_Ir_727,
    theory_yield_ArN2_Ir_750,
    theory_yield_ArN2_Ir_763,
    theory_yield_ArN2_Ir_772,
)


DATA_DIR = PROJECT_ROOT / "data"
IR_LINES = ("696", "727", "750", "763", "772")
TAUS = {"696": 28.3, "727": 28.3, "750": 21.7, "763": 29.4, "772": 28.3}

# Criterio IR limpio:
#   - solo las presiones indicadas;
#   - solo puntos con concentración estrictamente menor que 10 %.
# No se aplica ningún preproceso específico de IR en este archivo.
IR_FIT_PRESSURES = (1, 2, 3)
IR_MAX_CONCENTRATION_PERCENT = 10.0


def ir_parameters():
    params = []
    for line in IR_LINES:
        tau = TAUS[line]
        display = "764" if line == "763" else line
        params.extend(
            [
                Parameter(
                    f"PAr_star_{display}",
                    rf"$P_{{\mathrm{{Ar}}^* \ {display}\ \mathrm{{nm}}}}$",
                    0.0159,
                    0.0,
                    0.02,
                ),
                Parameter(
                    f"tau_N2_{display}",
                    rf"$\tau_{{\mathrm{{Ar}}^* \ {display}\ \mathrm{{nm}}}}$",
                    tau,
                    tau * 0.999999999999999,
                    tau * 1.000000000000001,
                    fixed=True,
                    fixed_value=tau,
                    fixed_error=0.1,
                ),
                Parameter(
                    f"K_Ar_Q_Ar_{display}",
                    rf"$K_{{\mathrm{{Ar}}^*,Q(\mathrm{{Ar}}) \ {display}\ \mathrm{{nm}}}}$",
                    1.0,
                    0.0,
                    1000.0,
                ),
                Parameter(
                    f"K_Ar_Q_N2_{display}",
                    rf"$K_{{\mathrm{{Ar}}^*,Q(\mathrm{{N}}_2) \ {display}\ \mathrm{{nm}}}}$",
                    1.0,
                    0.0,
                    1000.0,
                ),
            ]
        )
    return params


DATASETS = [
    DatasetSpec(
        key=line,
        csv_path=DATA_DIR / "Experimental" / "ArN2" / "csv" / f"{line}.csv",
        x_col="fN2",
        pressures=IR_FIT_PRESSURES,
        output_concentration_name="fN2",
        w_function=W_ArN2,
        max_concentration_percent=IR_MAX_CONCENTRATION_PERCENT,
    )
    for line in IR_LINES
]


EQUATIONS = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}


PLOTS = [
    PlotSpec(
        name=f"ArN2_IR_{line}",
        dataset_key=line,
        theory_key=line,
        pressures=IR_FIT_PRESSURES,
        concentration_grid=np.logspace(-6, np.log10(IR_MAX_CONCENTRATION_PERCENT / 100.0), 1000),
        title=rf"Primary ArN$_2$ IR ({line} nm) Yield fit",
        xlabel=r"Concentration of N$_2$ [$\%$]",
        ylabel="Normalized Yield",
        x_col="fN2",
        min_positive_x=1e-3,
        xlim=(1e-3, IR_MAX_CONCENTRATION_PERCENT * 1.1),
        ylim=(1e-5, 0.09),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArN2_IR" / f"ArN2_global_{line}.pdf",
        legend_kwargs={"ncol": 2, "fontsize": 9, "loc": "upper right"},
    )
    for line in IR_LINES
]


CONFIG = FitConfig(
    name="ArN2_IR_primary",
    model_name="ArN2_infrarred",
    degrad_csv=DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv",
    datasets=DATASETS,
    equations=EQUATIONS,
    parameters=ir_parameters(),
    plots=PLOTS,
    is_infrared=True,
    toy_spec=ToySpec(
        n_stat=10,
        n_syst=10,
        seed=44001,
        n_jobs=-1,
        syst_sources=tuple(SystematicSource(f"line_{line}_calibration", mode="by_dataset", datasets=(line,)) for line in IR_LINES),
    ),
    table_caption=r"Parámetros del ajuste primario IR en Ar--N$_2$.",
    table_label="tab:ArN2_IR_primary_stat_syst",
)


if __name__ == "__main__":
    PrimaryFitRunner(CONFIG, project_root=PROJECT_ROOT).run_all()
