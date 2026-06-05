import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scienceplots
plt.style.use("grid")

from band_utils import (
    configure_matplotlib,
    ensure_dir,
    infer_pressure_cols,
    read_primary_parameters,
    build_statistical_toy_dict,
    build_correlated_systematic_toy_dict,
    fit_toy_parameters,
    curves_from_parameters,
    percentile_curve_band,
    save_band_csv,
    export_parameter_products,
)

configure_matplotlib(plt, no_grid=True)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
FIT_DIR = ROOT_DIR / "primary_fits"

sys.path.extend([str(MODELS_DIR), str(DATA_DIR), str(FIT_DIR)])

from ArN2 import theory_yield_N2_uv  # noqa: E402
from read_experimental import read_experimental  # noqa: E402
from fiting import fitParameters  # noqa: E402

# =========================================================
# CONFIG
# =========================================================
DATA_DIR_EXP = DATA_DIR / "Experimental" / "ArN2"
DATA_DIR_DEGRAD = DATA_DIR / "Primary_DegradData"
DATA_DIR_PAR = DATA_DIR / "Parameters"
DATA_DIR_BANDS = DATA_DIR / "sistematic_stadistic_data"
PLOTS_DIR = ensure_dir(BASE_DIR / "plots")
TEX_DIR = ensure_dir(BASE_DIR / "tex_param")

archivo_entrada = DATA_DIR_EXP / "N2_primary_data_final.pkl"
yields = ["yield_N2"]
presiones = [1, 2, 2.5, 3, 4, 5]
plot_pressure = 1.0

N_STAT_TOYS = int(os.environ.get("N_STAT_TOYS", "150"))
N_SYST_TOYS = int(os.environ.get("N_SYST_TOYS", "150"))
STAT_SEED = 123456
SYST_SEED = 654321

# =========================================================
# DATA / MODEL
# =========================================================
degrad_data = pd.read_csv(DATA_DIR_DEGRAD / "ArN2.csv")

# Same configuration as the updated ArN2 primary fit.
to_m3 = 2.69e25 * 1e-9 * 273.15 / 300.0
tau_N2 = 1e2 / np.mean(np.array([2.6, 2.07, 3.3, 2.5, 2.74, 2.66]))
K_N2_Q_N2 = to_m3 * 1e-17 * np.mean(np.array([0.71, 1.12, 1.0, 1.4]))
K_N2_Q_Ar = to_m3 * 1e-19 * np.mean(np.array([5.6, 8.6]))
K_ArMeta_Q_N2c = to_m3 * 1e-17 * np.mean(np.array([3.2, 3.0, 1.1]))
K_ArMeta_Q_N2b = to_m3 * 1e-17 * np.mean(np.array([0.16]))
K_ArMeta_Q_2Ar = 1e-9 * np.mean(np.array([7.93e6]))
K_ArRes_Q_N2c = to_m3 * 1e-17 * np.mean(np.array([1.5, 3.6]))
K_ArRes_Q_N2b = to_m3 * 1e-17 * np.mean(np.array([1.5, 0.0]))
K_ArRes_Q_2Ar = 1e-9 * np.mean(np.array([9.24e5]))

x0_semifixed = np.array([
    0.0,
    0.0,
    tau_N2, K_N2_Q_N2, K_N2_Q_Ar,
    K_ArMeta_Q_N2c, K_ArMeta_Q_N2b, K_ArMeta_Q_2Ar,
    K_ArRes_Q_N2c, K_ArRes_Q_N2b, K_ArRes_Q_2Ar,
    0.0, 0.0,
], dtype=float)

lower_semifixed = x0_semifixed / 2.0
upper_semifixed = x0_semifixed * 2.0

lower_og = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
x0_og = np.array([0.0044564, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
upper_og = np.array([1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=float)

x0 = x0_og + x0_semifixed
bounds = (list(lower_og + lower_semifixed), list(upper_og + upper_semifixed))
fixed_idx = [0, 2]
fixed_error = 0.376

equations = {"vis": theory_yield_N2_uv}

names_tex = [
    r"$N_{\mathrm{norm}}$",
    r"$P_{\mathrm{N}_2}$",
    r"$\tau_{\mathrm{N}_2}$",
    r"$K_{\mathrm{N}_2,Q(\mathrm{N}_2)}$",
    r"$K_{\mathrm{N}_2,Q(\mathrm{Ar})}$",
    r"$K_{\mathrm{ArM},Q(\mathrm{N}_2,c)}$",
    r"$K_{\mathrm{ArM},Q(\mathrm{N}_2,b)}$",
    r"$K_{\mathrm{ArM},Q(2\mathrm{Ar})}$",
    r"$K_{\mathrm{ArR},Q(\mathrm{N}_2,c)}$",
    r"$K_{\mathrm{ArR},Q(\mathrm{N}_2,b)}$",
    r"$K_{\mathrm{ArR},Q(2\mathrm{Ar})}$",
    r"$P_{\mathrm{Ar}^{**}}$",
    r"$f_{\mathrm{Ar}^{**}}$",
]

names_csv = [
    "Nnorm", "P_N2", "tau_N2", "K_N2_Q_N2", "K_N2_Q_Ar",
    "K_ArMeta_Q_N2c", "K_ArMeta_Q_N2b", "K_ArMeta_Q_2Ar",
    "K_ArRes_Q_N2c", "K_ArRes_Q_N2b", "K_ArRes_Q_2Ar",
    "P_Ar_dbleStar", "frac_Ar_dbleStar",
]

# =========================================================
# HELPERS
# =========================================================
def W_N2(xN2, WAr=26.4, WN2=34.8):
    xN2 = np.asarray(xN2, dtype=float)
    return 1.0 / ((1.0 - xN2) / WAr + xN2 / WN2)


def apply_w_scaling(df, conc_col="N2 concentration (%)"):
    df = df.copy(deep=True)
    y_cols, err_cols = infer_pressure_cols(df)
    if not y_cols:
        return df
    w = W_N2(df[conc_col].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w)[:, None]
    df.loc[:, y_cols + err_cols] = df[y_cols + err_cols].to_numpy(dtype=float) * factor
    return df


def drop_pure_n2_points(df, conc_col="N2 concentration (%)"):
    return df[(df[conc_col] != 100) & (df[conc_col] != 150)].copy()


def call_read_experimental(uncertainty_mode):
    aliases = {"systematic": ["sistematic"], "sistematic": ["systematic"], "stadistic": ["statistic", "statistical"]}
    last_error = None
    for mode in [uncertainty_mode] + aliases.get(uncertainty_mode, []):
        try:
            read_experimental(
                str(archivo_entrada),
                yields,
                presiones,
                str(DATA_DIR_EXP),
                concentraciones_reales=None,
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_experimental(uncertainty_mode="all", w_scaled=True, drop_pure=True):
    call_read_experimental(uncertainty_mode)
    df = pd.read_csv(DATA_DIR_EXP / "yield_N2.csv")
    if drop_pure:
        df = drop_pure_n2_points(df)
    if w_scaled:
        df = apply_w_scaling(df)
    return {"vis": df}


def run_fit(exp_dict, x_start):
    return fitParameters(
        equations,
        {"vis": exp_dict["vis"].fillna(0)},
        degrad_data,
        x0=x_start,
        bounds=bounds,
        is_infrared=True,
        fixed_idx=fixed_idx,
        fixed_error=fixed_error,
        verbose=0,
    )

# =========================================================
# CENTRAL PARAMETERS FROM PRIMARY FIT
# =========================================================
par_primary = read_primary_parameters(DATA_DIR_PAR / "ArN2_primary.csv")
if len(par_primary) != len(x0):
    raise ValueError("ArN2_primary.csv no tiene la longitud esperada para el modelo UV.")

print("=" * 60)
print("Optimal Ar-N2 UV line loaded from:", DATA_DIR_PAR / "ArN2_primary.csv")
print(par_primary)
print("=" * 60)

# =========================================================
# TOYS
# =========================================================
exp_stat = load_experimental("stadistic")
exp_sys = load_experimental("sistematic")
bar_cols, err_cols = infer_pressure_cols(exp_stat["vis"])

stat_params, stat_failures = fit_toy_parameters(
    N_STAT_TOYS,
    lambda rng: build_statistical_toy_dict(exp_stat, bar_cols, err_cols, rng),
    run_fit,
    par_primary,
    seed=STAT_SEED,
)
syst_params, syst_failures = fit_toy_parameters(
    N_SYST_TOYS,
    lambda rng: build_correlated_systematic_toy_dict(exp_stat, exp_sys, bar_cols, err_cols, rng,
                                                      group_map={"vis": "N2_UV"}),
    run_fit,
    par_primary,
    seed=SYST_SEED,
)

print(f"Statistical toys accepted: {len(stat_params)} / {N_STAT_TOYS}; failed: {stat_failures}")
print(f"Systematic toys accepted:   {len(syst_params)} / {N_SYST_TOYS}; failed: {syst_failures}")

export_parameter_products(
    DATA_DIR_BANDS,
    TEX_DIR,
    "ArN2_primary_UV",
    names_csv,
    names_tex,
    par_primary,
    stat_params,
    syst_params,
    caption=(r"Parámetros del ajuste primario UV en Ar--N$_2$. La línea central procede "
             r"de data/Parameters/ArN2_primary.csv; las incertidumbres son de pseudoexperimentos."),
    label="tab:ArN2_UV_toy_uncertainties",
)

# =========================================================
# PLOT
# =========================================================
yield_plot = load_experimental("all")["vis"]
yield_plot.loc[yield_plot["N2 concentration (%)"] <= 0, "N2 concentration (%)"] = 1e-6

fN2 = np.logspace(-4, 0, 1000)
scale_to_ph_mev = 1000.0 / par_primary[0]

def model_total(par):
    return theory_yield_N2_uv(par, degrad_data, fN2, plot_pressure)

y0 = model_total(par_primary)
stat_curves = curves_from_parameters(stat_params, model_total)
syst_curves = curves_from_parameters(syst_params, model_total)
y_low_stat, y_up_stat = percentile_curve_band(stat_curves, y0)
y_low_syst, y_up_syst = percentile_curve_band(syst_curves, y0)

save_band_csv(
    DATA_DIR_BANDS / "ArN2_UV_primary_band_1bar.csv",
    fN2 * 100,
    y0 * scale_to_ph_mev,
    y_low_stat * scale_to_ph_mev,
    y_up_stat * scale_to_ph_mev,
    y_low_syst * scale_to_ph_mev,
    y_up_syst * scale_to_ph_mev,
    metadata={"gas": "ArN2", "channel": "UV", "pressure_bar": 1.0, "scale": "ph_per_MeV"},
)

colors = plt.get_cmap("viridis")(np.linspace(0.18, 0.82, 4))
col = f"{plot_pressure:.1f}bar"
err_col = f"Err {plot_pressure:.1f}bar"

con_uv_cf4_macfly = [100.0]
y_uv_cf4_macfly = [94]
y_err_uv_cf4_macfly = [94 * 14e-2]
con_uv_cf4_morii = [100.0]
y_uv_cf4_morii = [145]
y_err_uv_cf4_morii = [2]
con_uv_cf4_lehaut = [100.0]
y_uv_cf4_lehaut = [96]
y_err_uv_cf4_lehaut = [40]


fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.fill_between(fN2 * 100, y_low_syst * scale_to_ph_mev, y_up_syst * scale_to_ph_mev,
                alpha=0.2, color=colors[0], label="Sistemático")
ax.fill_between(fN2 * 100, y_low_stat * scale_to_ph_mev, y_up_stat * scale_to_ph_mev,
                alpha=0.3, color=colors[0], label="Estadístico")
ax.plot(fN2 * 100, y0 * scale_to_ph_mev, lw=2.0, color=colors[0], label="Primary fit")
ax.errorbar(yield_plot["N2 concentration (%)"], yield_plot[col] * scale_to_ph_mev,
            yerr=yield_plot[err_col] * scale_to_ph_mev,
            marker="o", linestyle="none", ms=4, color=colors[0], ecolor=colors[0], capsize=2,
            label=f"X-ray {plot_pressure:g} bar")
ax.errorbar(con_uv_cf4_macfly, y_uv_cf4_macfly, yerr=y_err_uv_cf4_macfly,
            marker="v", linestyle="none", ms=5, color=colors[2], ecolor=colors[2], capsize=2,
            label="MacFly Collaboration")
ax.errorbar(con_uv_cf4_morii, y_uv_cf4_morii, yerr=y_err_uv_cf4_morii,
            marker="o", linestyle="none", ms=5, color=colors[3], ecolor=colors[3], capsize=2,
            label=r"$\alpha$ Morii")
ax.errorbar(con_uv_cf4_lehaut, y_uv_cf4_lehaut, yerr=y_err_uv_cf4_lehaut,
            marker="s", linestyle="none", ms=5, color=colors[1], ecolor=colors[1], capsize=2,
            label=r"$\alpha$ Lehaut")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(5e-2, 110)
ax.set_ylim(5e1, 3500)
ax.grid(False)
ax.set_xlabel(r"N$_2$ concentration [\%]")
ax.set_ylabel("ph/MeV")
ax.set_title(r"Primary Ar--N$_2$ UV yield (300--420 nm)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ArN2_bands_toy_stat_syst.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved ArN2 UV bands in:", DATA_DIR_BANDS)
