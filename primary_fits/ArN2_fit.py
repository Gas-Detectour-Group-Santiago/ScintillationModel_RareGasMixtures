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
from ArN2 import W_ArN2, theory_yield_N2_uv


DATA_DIR = PROJECT_ROOT / "data"

to_m3 = 2.69e25 * 1e-9 * 273.15 / 300.0

tau_N2 = 1e2 / np.mean(np.array([2.6, 2.07, 3.3, 2.5, 2.74, 2.66]))
K_N2_Q_N2 = to_m3 * 1e-17 * np.mean(np.array([0.71, 1.12, 1, 1.4]))
K_N2_Q_Ar = to_m3 * 1e-19 * np.mean(np.array([5.6, 8.6]))
K_ArMeta_Q_N2c = to_m3 * 1e-17 * np.mean(np.array([3.2, 3.0, 1.1]))
K_ArMeta_Q_N2b = to_m3 * 1e-17 * np.mean(np.array([0.16]))
K_ArMeta_Q_2Ar = 1e-9 * np.mean(np.array([7.93e6]))
K_ArRes_Q_N2c = to_m3 * 1e-17 * np.mean(np.array([1.5, 3.6]))
K_ArRes_Q_N2b = to_m3 * 1e-17 * np.mean(np.array([1.5, 0]))
K_ArRes_Q_2Ar = 1e-9 * np.mean(np.array([9.24e5]))


def _semi(value):
    return float(value), float(value / 2.0), float(value * 2.0)


PARAMETERS = [
    Parameter("Nnorm", r"$N_{\mathrm{norm}}$", 0.0044564, 0.0, 1.0),
    Parameter("P_N2", r"$\mathcal{W}_{\mathrm{N}_2^*}$", 0.0, 0.0, 1.0),
    Parameter("tau_N2", r"$\tau_{\mathrm{N}_2(C)}$ [ns]", *_semi(tau_N2), fixed=True, fixed_value=tau_N2, fixed_error=0.376),
    Parameter("K_N2_Q_N2", r"$K_{\mathrm{N}_2(C)Q(\mathrm{N}_2)}$ [ns$^{-1}$]", *_semi(K_N2_Q_N2)),
    Parameter("K_N2_Q_Ar", r"$K_{\mathrm{N}_2(C)Q(\mathrm{Ar})}$ [ns$^{-1}$]", *_semi(K_N2_Q_Ar)),
    Parameter("K_ArMeta_Q_N2c", r"$K_{\mathrm{Ar}(1s_5)Q(\mathrm{N}_2(C))}$ [ns$^{-1}$]", *_semi(K_ArMeta_Q_N2c)),
    Parameter("K_ArMeta_Q_N2b", r"$K_{\mathrm{Ar}(1s_5)Q(\mathrm{N}_2(B))}$ [ns$^{-1}$]", *_semi(K_ArMeta_Q_N2b)),
    Parameter("K_ArMeta_Q_2Ar", r"$K_{\mathrm{Ar}(1s_5)Q(2\mathrm{Ar})}$ [ns$^{-1}$]", *_semi(K_ArMeta_Q_2Ar)),
    Parameter("K_ArRes_Q_N2c", r"$K_{\mathrm{Ar}(1s_4)Q(\mathrm{N}_2(C))}$ [ns$^{-1}$]", *_semi(K_ArRes_Q_N2c)),
    Parameter("K_ArRes_Q_N2b", r"$K_{\mathrm{Ar}(1s_4)Q(\mathrm{N}_2(B))}$ [ns$^{-1}$]", *_semi(K_ArRes_Q_N2b)),
    Parameter("K_ArRes_Q_2Ar", r"$K_{\mathrm{Ar}(1s_4)Q(2\mathrm{Ar})}$ [ns$^{-1}$]", *_semi(K_ArRes_Q_2Ar)),
    Parameter("P_Ar_dbleStar", r"$\mathcal{W}_{\mathrm{Ar}^{**}}^*$", 0.0, 0.0, 1.0),
    Parameter("frac_Ar_dbleStar", r"$f_{\mathrm{Ar}^{**}}$", 0.0, 0.0, 1.0),
]


def filter_n2_uv(df):
    out = df.copy()
    mask = (out["fN2"] != 120) & (out["fN2"] != 150)
    return out.loc[mask].reset_index(drop=True)


DATASETS = [
    DatasetSpec(
        key="vis",
        csv_path=DATA_DIR / "Experimental" / "ArN2" / "csv" / "yield_N2.csv",
        x_col="fN2",
        pressures=[1, 2, 3, 4, 5],
        output_concentration_name="fN2",
        w_function=W_ArN2,
        preprocess=filter_n2_uv,
    )
]


PLOTS = [
    PlotSpec(
        name="ArN2_global",
        dataset_key="vis",
        theory_key="vis",
        pressures=[1, 2, 3, 4, 5],
        concentration_grid=np.logspace(-4, 0, 1000),
        title=r"Ar--N$_2$ primary UV fit",
        xlabel=r"N$_2$ concentration [%]",
        ylabel=r"Yield [arb. units]",
        x_col="fN2",
        min_positive_x=1e-3,
        xlim=(0.1 * 0.9, 100 * 1.1),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArN2_global.pdf",
        legend_kwargs={"ncol": 2},
    ),
    PlotSpec(
        name="ArN2_global_components_1bar",
        dataset_key="vis",
        theory_key="vis",
        pressures=[1],
        concentration_grid=np.logspace(-4, 0, 1000),
        title=r"Ar--N$_2$ primary UV components, 1 bar",
        xlabel=r"N$_2$ concentration [%]",
        ylabel=r"Yield [arb. units]",
        x_col="fN2",
        min_positive_x=1e-3,
        xlim=(0.1 * 0.9, 100 * 1.1),
        ylim=(1e-5, 0.1),
        activate_components=True,
        line_label_fmt=("{p:g} bar completed", "{p:g} bar N2 dir", "{p:g} bar Ar* Meta", "{p:g} bar Ar* Res", "{p:g} bar Ar**"),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArN2_global_components_1bar.pdf",
        legend_kwargs={"ncol": 2},
    ),
    PlotSpec(
        name="ArN2_global_components_5bar",
        dataset_key="vis",
        theory_key="vis",
        pressures=[5],
        concentration_grid=np.logspace(-4, 0, 1000),
        title=r"Ar--N$_2$ primary UV components, 5 bar",
        xlabel=r"N$_2$ concentration [%]",
        ylabel=r"Yield [arb. units]",
        x_col="fN2",
        min_positive_x=1e-3,
        xlim=(0.1 * 0.9, 100 * 1.1),
        ylim=(1e-5, 0.1),
        activate_components=True,
        line_label_fmt=("{p:g} bar completed", "{p:g} bar N2 dir", "{p:g} bar Ar* Meta", "{p:g} bar Ar* Res", "{p:g} bar Ar**"),
        output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArN2_global_components_5bar.pdf",
        legend_kwargs={"ncol": 2},
    ),
]


CONFIG = FitConfig(
    name="ArN2_primary",
    model_name="ArN2",
    degrad_csv=DATA_DIR / "Primary_DegradData" / "ArN2.csv",
    datasets=DATASETS,
    equations={"vis": theory_yield_N2_uv},
    parameters=PARAMETERS,
    plots=PLOTS,
    is_infrared=True,
    toy_spec=ToySpec(
        n_stat=10,
        n_syst=10,
        seed=22001,
        n_jobs=-1,
        syst_sources=(SystematicSource("uv_calibration", mode="by_dataset", datasets=("vis",)),),
    ),
    table_caption=r"Parámetros del ajuste primario UV en Ar--N$_2$.",
    table_label="tab:ArN2_primary_stat_syst",
)


if __name__ == "__main__":
    PrimaryFitRunner(CONFIG, project_root=PROJECT_ROOT).run_all()
