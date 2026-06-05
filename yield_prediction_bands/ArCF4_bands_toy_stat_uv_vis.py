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

from ArCF4 import theory_yield_vis, theory_yield_uv  # noqa: E402
from read_experimental import read_experimental  # noqa: E402
from fiting import fitParameters  # noqa: E402

# =========================================================
# CONFIG
# =========================================================
DATA_DIR_EXP = DATA_DIR / "Experimental" / "ArCF4"
DATA_DIR_DEGRAD = DATA_DIR / "Primary_DegradData"
DATA_DIR_PAR = DATA_DIR / "Parameters"
DATA_DIR_BANDS = DATA_DIR / "sistematic_stadistic_data"

PLOTS_DIR = ensure_dir(BASE_DIR / "plots")
TEX_DIR = ensure_dir(BASE_DIR / "tex_param")

archivo_entrada = DATA_DIR_EXP / "CF4_primary_data_final.pkl"
yields = ["vis", "UV"]
presiones = [1, 2, 2.5, 3, 4, 5]
output_dir = DATA_DIR_EXP

N_STAT_TOYS = int(os.environ.get("N_STAT_TOYS", "150"))
N_SYST_TOYS = int(os.environ.get("N_SYST_TOYS", "150"))
STAT_SEED = 12345
SYST_SEED = 54321

# =========================================================
# DATA / MODEL
# =========================================================
degrad_data = pd.read_csv(DATA_DIR_DEGRAD / "ArCF4.csv")

cf4_pct_w = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])


def W_CF4(f):
    return np.interp(np.asarray(f, dtype=float), cf4_pct_w, ion_pot)


x0_default = np.array([
    0.14,
    0.10, 0.99, 3, 0.037 * 3,
    1.0, 0.065, 0.48, 50.10, 0.37,
    0.00001,
], dtype=float)

lower = [
    0.0,
    0.0, 0.0, 0.0, 0.0,
    0.00, 0.065, 0.0, 50, 0.0,
    0.0,
]

upper = [
    0.99,
    1.0, 1.0, 10000.0, 10000.0,
    10000.0, 0.066, 1.0, 50.2, 1.0,
    0.0001,
]

bounds = (lower, upper)
fixed_idx = [6, 8]
fixed_values = [0.065, 50.05]
fixed_error = 0.01

equations = {
    "vis": theory_yield_vis,
    "uv": theory_yield_uv,
}

names_tex = [
    r"$N_{\mathrm{norm}}$",
    r"$P_{\mathrm{CF}_3}^{\mathrm{vis}}$",
    r"$P_{\mathrm{Ar}^{**}}$",
    r"$K_{\mathrm{Ar}^{**},Q(\mathrm{Ar})}$",
    r"$K_{\mathrm{Ar}^{**},Q(\mathrm{CF}_4)}$",
    r"$1/(\tau_{\mathrm{disc}}K_{\mathrm{relax}})$",
    r"$\tau_{\mathrm{UV}}K_{\mathrm{CF}_4,Q(\mathrm{CF}_4)}$",
    r"$P_{\mathrm{CF}_4}^{\mathrm{dir}}$",
    r"$K_{\mathrm{Ar}^{++},Q(\mathrm{CF}_4)}$",
    r"$P_{\mathrm{Ar}^{++}}$",
    r"$P_{\mathrm{CF}_3}^{\mathrm{UV}}$",
]

names_csv = [
    "Nnorm",
    "PCF3dir_vis",
    "PAr_dblestar",
    "KAr_dblestar_QAr",
    "KAr_dblestar_QCF4",
    "inv_tauDisc_Krelax",
    "tauUv_KCF4_QCF4",
    "PCF4dir",
    "KArpp_QCF4",
    "PArpp",
    "PCF3dir_uv",
]

# =========================================================
# HELPERS
# =========================================================
def call_read_experimental(uncertainty_mode):
    aliases = {
        "systematic": ["sistematic"],
        "sistematic": ["systematic"],
        "statistic": ["stadistic", "statistical"],
        "stadistic": ["statistic", "statistical"],
        "statistical": ["stadistic", "statistic"],
    }
    last_error = None
    for mode in [uncertainty_mode] + aliases.get(uncertainty_mode, []):
        try:
            read_experimental(
                str(archivo_entrada),
                yields,
                presiones,
                str(output_dir),
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_experimental(uncertainty_mode="all", w_scaled=True):
    call_read_experimental(uncertainty_mode)
    df_uv = pd.read_csv(DATA_DIR_EXP / "UV.csv")
    df_vis = pd.read_csv(DATA_DIR_EXP / "vis.csv")

    df_uv.loc[0, "fCF4"] = 0.001
    df_vis = df_vis.fillna(0)

    if w_scaled:
        for df in (df_uv, df_vis):
            y_cols, err_cols = infer_pressure_cols(df)
            w = W_CF4(df["fCF4"].to_numpy(dtype=float) / 100.0)
            factor = (1.0 / w)[:, None]
            df.loc[:, y_cols + err_cols] = df[y_cols + err_cols].to_numpy(dtype=float) * factor

    return {"uv": df_uv, "vis": df_vis}


def make_experimental_data(exp_dict):
    return {"vis": exp_dict["vis"].fillna(0), "uv": exp_dict["uv"].fillna(0)}


def run_fit(exp_dict, x_start):
    return fitParameters(
        equations,
        make_experimental_data(exp_dict),
        degrad_data,
        x0=x_start,
        bounds=bounds,
        fixed_idx=fixed_idx,
        fixed_values=fixed_values,
        fixed_error=fixed_error,
        verbose=0,
    )


def fit_runner(toy_data, x_start):
    return run_fit(toy_data, x_start)


def toy_builder_stat(exp_stat, bar_cols, err_cols):
    return lambda rng: build_statistical_toy_dict(exp_stat, bar_cols, err_cols, rng)


def toy_builder_syst(exp_stat, exp_sys, bar_cols, err_cols):
    # One nuisance for UV and one for VIS.
    return lambda rng: build_correlated_systematic_toy_dict(
        exp_stat,
        exp_sys,
        bar_cols,
        err_cols,
        rng,
        group_map={"uv": "uv", "vis": "vis"},
    )


def envelope_from_toys(params, model, nominal_curve):
    curves = curves_from_parameters(params, model)
    return percentile_curve_band(curves, nominal_curve)


def plot_band(ax, x_percent, y0, stat_low, stat_high, syst_low, syst_high, *, color, label):
    ax.fill_between(x_percent, syst_low, syst_high, alpha=0.24, color=color, label="Sistemático" if label else None)
    ax.fill_between(x_percent, stat_low, stat_high, alpha=0.22, color=color, label="Estadístico" if label else None)
    ax.plot(x_percent, y0, lw=2.0, color=color, label=label)


# =========================================================
# CENTRAL PARAMETERS FROM PRIMARY FIT
# =========================================================
primary_parameter_path = DATA_DIR_PAR / "ArCF4_primary.csv"
par_primary = read_primary_parameters(primary_parameter_path)
if len(par_primary) != len(x0_default):
    raise ValueError("ArCF4_primary.csv no tiene la longitud esperada para el modelo UV/VIS.")

print("=" * 60)
print("Optimal fit line loaded from:", primary_parameter_path)
print(par_primary)
print("=" * 60)

# =========================================================
# TOY FITS
# =========================================================
exp_stat = load_experimental("stadistic", w_scaled=True)
exp_sys = load_experimental("sistematic", w_scaled=True)
exp_plot = load_experimental("all", w_scaled=True)
bar_cols, err_cols = infer_pressure_cols(exp_stat["uv"])

stat_params, stat_failures = fit_toy_parameters(
    N_STAT_TOYS,
    toy_builder_stat(exp_stat, bar_cols, err_cols),
    fit_runner,
    par_primary,
    seed=STAT_SEED,
)
syst_params, syst_failures = fit_toy_parameters(
    N_SYST_TOYS,
    toy_builder_syst(exp_stat, exp_sys, bar_cols, err_cols),
    fit_runner,
    par_primary,
    seed=SYST_SEED,
)

print(f"Statistical toys accepted: {len(stat_params)} / {N_STAT_TOYS}; failed: {stat_failures}")
print(f"Systematic toys accepted:   {len(syst_params)} / {N_SYST_TOYS}; failed: {syst_failures}")

export_parameter_products(
    DATA_DIR_BANDS,
    TEX_DIR,
    "ArCF4_primary_UV_VIS",
    names_csv,
    names_tex,
    par_primary,
    stat_params,
    syst_params,
    caption=(
        r"Parámetros del ajuste primario Ar--CF$_4$ UV/VIS. "
        r"La columna central procede de data/Parameters/ArCF4_primary.csv; "
        r"las incertidumbres estadística y sistemática proceden de pseudoexperimentos."
    ),
    label="tab:ArCF4_UV_VIS_toy_uncertainties",
)

# =========================================================
# PLOTS
# =========================================================
yield_uv = exp_plot["uv"]
yield_vis = exp_plot["vis"]
norm = par_primary[0]
scale_to_ph_mev = 1000.0 / norm
colors = plt.get_cmap("viridis")(np.linspace(0.18, 0.82, 4))

# UV, ph/MeV
fCF4_uv = np.logspace(-5, 0, 600)

def model_uv(par):
    return theory_yield_uv(par, degrad_data, fCF4_uv, 1)

y0_uv = model_uv(par_primary)
y_low_stat_uv, y_up_stat_uv = envelope_from_toys(stat_params, model_uv, y0_uv)
y_low_syst_uv, y_up_syst_uv = envelope_from_toys(syst_params, model_uv, y0_uv)

save_band_csv(
    DATA_DIR_BANDS / "ArCF4_UV_primary_band_1bar.csv",
    fCF4_uv * 100,
    y0_uv * scale_to_ph_mev,
    y_low_stat_uv * scale_to_ph_mev,
    y_up_stat_uv * scale_to_ph_mev,
    y_low_syst_uv * scale_to_ph_mev,
    y_up_syst_uv * scale_to_ph_mev,
    metadata={"gas": "ArCF4", "channel": "UV", "pressure_bar": 1.0, "scale": "ph_per_MeV"},
)

con_uv_cf4_morozov = [100.0]
y_uv_cf4_morozov = [2175]
y_err_uv_cf4_morozov = [2600 - 2175]
cf4_pct_lit = np.array([0.15, 0.35, 0.50, 1.00, 2.00, 6.00, 11.00])
uv_E100 = np.array([358.9, 350.8, 292.2, 209.4, 227.8, 245.5, 263.4])
uv_E100_err = np.array([3.0, 3.9, 0.2, 2.0, 10.9, 2.4, 4.3])
Ar_third_continuum_ph_MeV = np.array([2.7e3])
Ar_third_continuum_err_ph_MeV = np.array([0.14e3])

fig, ax = plt.subplots(figsize=(6.4, 4.3))
plot_band(
    ax,
    fCF4_uv * 100,
    y0_uv * scale_to_ph_mev,
    y_low_stat_uv * scale_to_ph_mev,
    y_up_stat_uv * scale_to_ph_mev,
    y_low_syst_uv * scale_to_ph_mev,
    y_up_syst_uv * scale_to_ph_mev,
    color=colors[0],
    label="Primary fit",
)
ax.errorbar(yield_uv["fCF4"], yield_uv["1.0bar"] * scale_to_ph_mev,
            yerr=yield_uv["Err 1.0bar"] * scale_to_ph_mev,
            marker="o", linestyle="none", ms=4, color=colors[0], ecolor=colors[0], capsize=2,
            label="X-ray (220--400 nm)")

ax.errorbar(con_uv_cf4_morozov, y_uv_cf4_morozov, yerr=y_err_uv_cf4_morozov,
            marker="x", linestyle="none", ms=5, color=colors[2], ecolor=colors[2], capsize=2,
            label=r"$\alpha$ Morozov")
ax.errorbar(0.001, Ar_third_continuum_ph_MeV, yerr=Ar_third_continuum_err_ph_MeV,
            marker="v", linestyle="none", ms=5, color=colors[3], ecolor=colors[3], capsize=2,
            label=r"$\alpha$ Santorelli")
ax.errorbar(cf4_pct_lit, uv_E100, yerr=uv_E100_err, marker="o", linestyle="none", ms=4,
            color=colors[1], ecolor=colors[1], capsize=2, label=r"$\alpha$ P. Amedo")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(False)
ax.set_xlabel(r"CF$_4$ concentration [\%]")
ax.set_ylabel("ph/MeV")
ax.set_title(r"Ar--CF$_4$ UV yield (220--400 nm)")
ax.legend(ncol=1)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ArCF4_bands_uv_toy_stat_syst.pdf", bbox_inches="tight")
plt.close(fig)

# Visible, ph/MeV
fCF4_vis = np.logspace(-3, 0, 600)

def model_vis(par):
    return theory_yield_vis(par, degrad_data, fCF4_vis, 1)

y0_vis = model_vis(par_primary)
y_low_stat_vis, y_up_stat_vis = envelope_from_toys(stat_params, model_vis, y0_vis)
y_low_syst_vis, y_up_syst_vis = envelope_from_toys(syst_params, model_vis, y0_vis)

save_band_csv(
    DATA_DIR_BANDS / "ArCF4_VIS_primary_band_1bar.csv",
    fCF4_vis * 100,
    y0_vis * scale_to_ph_mev,
    y_low_stat_vis * scale_to_ph_mev,
    y_up_stat_vis * scale_to_ph_mev,
    y_low_syst_vis * scale_to_ph_mev,
    y_up_syst_vis * scale_to_ph_mev,
    metadata={"gas": "ArCF4", "channel": "VIS", "pressure_bar": 1.0, "scale": "ph_per_MeV"},
)

cf4_red_E100 = [0.2, 0.4, 0.7, 1.0, 2.0, 7.0, 10.0]
y_red_E100 = [450, 500, 600, 1150, 1300, 1850, 1900]
yerr_red_E100 = [60, 60, 60, 90, 100, 120, 120]
vis_cf4_red_E100 = [100.0]
vis_y_red_E100 = [1184.7]
vis_yerr_red_E100 = [47]
vis2_cf4_red_E100 = [100.0]
vis2_y_red_E100 = [695]
vis2_yerr_red_E100 = [827 - 695]

fig, ax = plt.subplots(figsize=(6.4, 4.3))
plot_band(
    ax,
    fCF4_vis * 100,
    y0_vis * scale_to_ph_mev,
    y_low_stat_vis * scale_to_ph_mev,
    y_up_stat_vis * scale_to_ph_mev,
    y_low_syst_vis * scale_to_ph_mev,
    y_up_syst_vis * scale_to_ph_mev,
    color=colors[0],
    label="Primary fit",
)
ax.errorbar(yield_vis["fCF4"], yield_vis["1.0bar"] * scale_to_ph_mev,
            yerr=yield_vis["Err 1.0bar"] * scale_to_ph_mev,
            marker="o", linestyle="none", ms=4, color=colors[0], ecolor=colors[0], capsize=2,
            label="X-ray")
ax.errorbar(cf4_red_E100, y_red_E100, yerr=yerr_red_E100, marker="o", linestyle="none", ms=4,
            color=colors[1], ecolor=colors[1], capsize=2, label=r"$\alpha$ P. Amedo")
ax.errorbar(vis2_cf4_red_E100, vis2_y_red_E100, yerr=vis2_yerr_red_E100,
            marker="x", linestyle="none", ms=5, color=colors[2], ecolor=colors[2], capsize=2,
            label=r"$\alpha$ Morozov")
ax.errorbar(vis_cf4_red_E100, vis_y_red_E100, yerr=vis_yerr_red_E100,
            marker="v", linestyle="none", ms=5, color=colors[3], ecolor=colors[3], capsize=2,
            label=r"$\alpha$ Lehaut")
ax.set_xscale("log")
ax.grid(False)
ax.set_xlabel(r"CF$_4$ concentration [\%]")
ax.set_ylabel("ph/MeV")
ax.set_title(r"Ar--CF$_4$ visible yield (400--700 nm)")
ax.legend(ncol=1)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ArCF4_bands_vis_toy_stat_syst.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved UV/VIS bands in:", DATA_DIR_BANDS)
