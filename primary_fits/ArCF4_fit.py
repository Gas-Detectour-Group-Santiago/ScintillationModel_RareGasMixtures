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
from auxiliares.fit_io import drop_concentration_value, move_concentration_value
from ArCF4 import ion_potential as W_ArCF4
from ArCF4 import theory_yield_uv, theory_yield_vis


DATA_DIR = PROJECT_ROOT / "data"

# These two operations are applied before W(f) scaling. They modify only the
# DataFrame used by the fit/plot, never the source experimental CSV.
DROP_PURE_ARGON = drop_concentration_value("fCF4", 0.0)
MOVE_PURE_ARGON_TO_LOW_CF4 = move_concentration_value("fCF4", 0.0, 0.001)


PARAMETERS = [
    Parameter("Nnorm", r"$N_{\mathrm{norm}}$", 4.45646783e-03, 0.0, 0.99),
    Parameter("P_CF3_vis_dir", r"$\mathcal{W}_{\mathrm{CF}_3^*,\mathrm{vis}}^{\mathrm{dir}}$", 6.97523694e-02, 0.0, 1.0),
    Parameter("P_Ar_dbleStar", r"$\mathcal{W}_{\mathrm{Ar}^{**}}$", 1.50511150e-01, 0.0, 1.0),
    Parameter("K_Ar_dbleStar_Q_Ar", r"$K_{\mathrm{Ar}^{**},Q(\mathrm{Ar})}\,[\mathrm{ns}^{-1}]$", 3.89959971e-01, 0.0, 10000.0),
    Parameter("K_Ar_dbleStar_Q_CF4", r"$K_{\mathrm{Ar}^{**},Q(\mathrm{CF}_4)}\,[\mathrm{ns}^{-1}]$", 1.14938357e01, 0.0, 11.5),
    Parameter("inv_tau_dis_K_relax", r"$1/(\tau_{\mathrm{dis}}K_{\mathrm{relax}})$", 5.67737517e-03, 0.0, 10000.0),
    Parameter("tau_uv_K_CF4_Q_CF4", r"$\tau_{\mathrm{uv}}K_{\mathrm{CF_4}^{+,*}Q(\mathrm{CF}_4)}$", 6.5e-02, 0.065, 0.066, fixed=True, fixed_value=0.065),
    Parameter("P_CF4_dir", r"$\mathcal{W}_{\mathrm{CF}_4^{+,*}}^{\mathrm{dir}}$", 2.36307283e-01, 0.0, 1.0),
    Parameter("K_Ar3rd_Q_CF4", r"$K_{\mathrm{Ar}^{++},Q(\mathrm{CF}_4)}\,[\mathrm{ns}^{-1}]$", 5.005e01, 50.0, 50.2, fixed=True, fixed_value=50.05),
    Parameter("P_Ar3rd", r"$\mathcal{W}_{\mathrm{Ar}^{++}}$", 5.18971635e-01, 0.0, 1.0),
    Parameter("P_CF3_uv_dir", r"$\mathcal{B}r_{\mathrm{UV}}(\mathrm{CF}_3^*)$", 1.0e-04, 0.0, 1.0e-04),
]


DATASETS = [
    DatasetSpec(
        key="vis",
        csv_path=DATA_DIR / "Experimental" / "ArCF4" / "csv" / "vis.csv",
        x_col="fCF4",
        pressures=[1, 2, 2.5, 3, 4, 5],
        output_concentration_name="fCF4",
        w_function=W_ArCF4,
        preprocess_before_w=DROP_PURE_ARGON,
    ),
    DatasetSpec(
        key="uv",
        csv_path=DATA_DIR / "Experimental" / "ArCF4" / "csv" / "UV.csv",
        x_col="fCF4",
        pressures=[1, 2, 2.5, 3, 4, 5],
        output_concentration_name="fCF4",
        w_function=W_ArCF4,
        preprocess_before_w=MOVE_PURE_ARGON_TO_LOW_CF4,
    ),
]


PLOTS = [
    PlotSpec(
        name="ArCF4_visible",
        dataset_key="vis",
        theory_key="vis",
        pressures=[1, 3, 4],
        concentration_grid=np.logspace(-4, 0, 1000),
        title=r"Ar--CF$_4$ primary visible fit",
        xlabel=r"CF$_4$ concentration [%]",
        ylabel=r"Yield [arb. eV$^{-1}$]",
        x_col="fCF4",
        min_positive_x=1e-3,
        xlim=(0.1 * 0.9, 100 * 1.1),
       # ylim=(0.0006, 0.1),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArCF4_visible.pdf",
        legend_kwargs={"ncol": 2},
    ),
    PlotSpec(
        name="ArCF4_uv",
        dataset_key="uv",
        theory_key="uv",
        pressures=[1, 3, 4],
        concentration_grid=np.logspace(-6, 0, 1000),
        title=r"Ar--CF$_4$ primary UV fit",
        xlabel=r"CF$_4$ concentration [%]",
        ylabel=r"Yield [arb. eV$^{-1}$]",
        x_col="fCF4",
        min_positive_x=1e-5,
        xlim=(0.001 * 0.9, 100 * 1.1),
       # ylim=(0.005, 0.4),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArCF4_uv.pdf",
        legend_kwargs={"ncol": 2},
    ),
]


CONFIG = FitConfig(
    name="ArCF4_primary",
    model_name="ArCF4",
    degrad_csv=DATA_DIR / "Primary_DegradData" / "ArCF4.csv",
    datasets=DATASETS,
    equations={"vis": theory_yield_vis, "uv": theory_yield_uv},
    parameters=PARAMETERS,
    plots=PLOTS,
    toy_spec=ToySpec(
        n_stat=10,
        n_syst=10,
        seed=14001,
        n_jobs=-1,
        syst_sources=(
            SystematicSource("vis_calibration", mode="by_dataset", datasets=("vis",)),
            SystematicSource("uv_calibration", mode="by_dataset", datasets=("uv",)),
        ),
    ),
    table_caption=r"Parámetros del ajuste primario UV/VIS en Ar--CF$_4$.",
    table_label="tab:ArCF4_primary_stat_syst",
)


if __name__ == "__main__":
    PrimaryFitRunner(CONFIG, project_root=PROJECT_ROOT).run_all()
