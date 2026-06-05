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
    pressure_cols,
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

from ArCF4_infrarred import (  # noqa: E402
    theory_yield_ArCF4_Ir_696,
    theory_yield_ArCF4_Ir_727,
    theory_yield_ArCF4_Ir_750,
    theory_yield_ArCF4_Ir_763,
    theory_yield_ArCF4_Ir_772,
    theory_yield_ArCF4_Ir_794,
)
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
lines = ["696", "727", "750", "763", "772", "794"]
FIT_PRESSURES = [1, 2, 3]
PLOT_PRESSURES = [1, 2, 3, 4, 5]
FIT_BAR_COLS, FIT_ERR_COLS = pressure_cols(FIT_PRESSURES)
PLOT_BAR_COLS, PLOT_ERR_COLS = pressure_cols(PLOT_PRESSURES)

N_STAT_TOYS = int(os.environ.get("N_STAT_TOYS", "150"))
N_SYST_TOYS = int(os.environ.get("N_SYST_TOYS", "150"))
STAT_SEED = 314159
SYST_SEED = 271828
TAU_FIXED_ERROR = 0.1
FIXED_IDX = [1, 5, 9, 13, 17, 21]

# =========================================================
# W-VALUE Ar/CF4
# =========================================================
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])


def W_CF4(f):
    return np.interp(np.asarray(f, dtype=float), cf4_pct, ion_pot)


# =========================================================
# FIT SETUP
# =========================================================
x0_semifixed = np.array([
    0.0, 28.3, 0.0, 0.0,
    0.0, 28.3, 0.0, 0.0,
    0.0, 21.7, 0.0, 0.0,
    0.0, 29.4, 0.0, 0.0,
    0.0, 28.3, 0.0, 0.0,
    0.0, 29.3, 0.0, 0.0,
], dtype=float)

lower_semifixed = x0_semifixed * 0.999999999999999
upper_semifixed = x0_semifixed * 1.000000000000001

lower = np.array([
    0.0, 0.0, 0.01, 0.0,
    0.0, 0.0, 0.00, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
], dtype=float) + lower_semifixed

x0 = np.array([
    0.02, 0.0, 1.0, 1.0,
    0.02, 0.0, 1.0, 1.0,
    0.02, 0.0, 1.0, 1.0,
    0.02, 0.0, 1.0, 1.0,
    0.02, 0.0, 1.0, 1.0,
    0.02, 0.0, 1.0, 1.0,
], dtype=float) + x0_semifixed

upper = np.array([
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
], dtype=float) + upper_semifixed

bounds = (list(lower), list(upper))

equations = {
    "696": theory_yield_ArCF4_Ir_696,
    "727": theory_yield_ArCF4_Ir_727,
    "750": theory_yield_ArCF4_Ir_750,
    "763": theory_yield_ArCF4_Ir_763,
    "772": theory_yield_ArCF4_Ir_772,
    "794": theory_yield_ArCF4_Ir_794,
}

names_tex = [
    r"$P_{\mathrm{Ar}^*\,696}$", r"$\tau_{696}$", r"$K_{\mathrm{Ar},696}$", r"$K_{\mathrm{CF}_4,696}$",
    r"$P_{\mathrm{Ar}^*\,727}$", r"$\tau_{727}$", r"$K_{\mathrm{Ar},727}$", r"$K_{\mathrm{CF}_4,727}$",
    r"$P_{\mathrm{Ar}^*\,750}$", r"$\tau_{750}$", r"$K_{\mathrm{Ar},750}$", r"$K_{\mathrm{CF}_4,750}$",
    r"$P_{\mathrm{Ar}^*\,764}$", r"$\tau_{764}$", r"$K_{\mathrm{Ar},764}$", r"$K_{\mathrm{CF}_4,764}$",
    r"$P_{\mathrm{Ar}^*\,772}$", r"$\tau_{772}$", r"$K_{\mathrm{Ar},772}$", r"$K_{\mathrm{CF}_4,772}$",
    r"$P_{\mathrm{Ar}^*\,794}$", r"$\tau_{794}$", r"$K_{\mathrm{Ar},794}$", r"$K_{\mathrm{CF}_4,794}$",
]

names_csv = [
    "PAr_star_696", "tau_Ar_696", "K_Ar_Q_Ar_696", "K_Ar_Q_CF4_696",
    "PAr_star_727", "tau_Ar_727", "K_Ar_Q_Ar_727", "K_Ar_Q_CF4_727",
    "PAr_star_750", "tau_Ar_750", "K_Ar_Q_Ar_750", "K_Ar_Q_CF4_750",
    "PAr_star_764", "tau_Ar_764", "K_Ar_Q_Ar_764", "K_Ar_Q_CF4_764",
    "PAr_star_772", "tau_Ar_772", "K_Ar_Q_Ar_772", "K_Ar_Q_CF4_772",
    "PAr_star_794", "tau_Ar_794", "K_Ar_Q_Ar_794", "K_Ar_Q_CF4_794",
]

# =========================================================
# HELPERS
# =========================================================
def apply_global_threshold(df, conc_col="fCF4", is_727=False, pressures=None, threshold=0.0):
    if pressures is None:
        pressures = FIT_PRESSURES
    bar_cols, err_cols = pressure_cols(pressures)
    df_low = df[df[conc_col] < 11].copy()
    if is_727:
        df_low = df[df[conc_col] < 6].copy()
    mask = df_low[bar_cols] >= threshold
    df_low.loc[:, bar_cols] = df_low[bar_cols].where(mask)
    for bar, err in zip(bar_cols, err_cols):
        df_low.loc[:, err] = df_low[err].where(mask[bar])
    return df_low[[conc_col] + bar_cols + err_cols]


def call_read_experimental(uncertainty_mode, pressures=PLOT_PRESSURES):
    aliases = {"systematic": ["sistematic"], "sistematic": ["systematic"], "stadistic": ["statistic", "statistical"]}
    last_error = None
    for mode in [uncertainty_mode] + aliases.get(uncertainty_mode, []):
        try:
            read_experimental(
                str(archivo_entrada),
                lines,
                pressures,
                str(DATA_DIR_EXP),
                concentraciones_reales=None,
                uncertainty_mode=mode,
                yield_mode="ir",
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def read_and_scale_ir_data(uncertainty_mode="all"):
    call_read_experimental(uncertainty_mode, pressures=PLOT_PRESSURES)
    exp = {line: pd.read_csv(DATA_DIR_EXP / f"{line}.csv") for line in lines}
    w_cf4 = W_CF4(exp["696"]["fCF4"].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w_cf4)[:, None]
    y_cols = PLOT_BAR_COLS + PLOT_ERR_COLS
    for line in lines:
        exp[line].loc[:, y_cols] = exp[line][y_cols].to_numpy(dtype=float) * factor
    return exp


def load_ir_experimental(uncertainty_mode="all", pressures=None):
    if pressures is None:
        pressures = FIT_PRESSURES
    exp_raw = read_and_scale_ir_data(uncertainty_mode)
    out = {}
    for line in lines:
        df = apply_global_threshold(exp_raw[line], pressures=pressures, is_727=(line == "727"))
        df.loc[df.index[0], "fCF4"] = 0.001
        out[line] = df
    return out


def make_experimental_data(exp_dict):
    return {line: exp_dict[line].fillna(0) for line in lines}


def run_fit(exp_dict, x_start):
    return fitParameters(
        equations,
        make_experimental_data(exp_dict),
        degrad_data,
        x0=x_start,
        bounds=bounds,
        is_infrared=True,
        fixed_idx=FIXED_IDX,
        fixed_error=TAU_FIXED_ERROR,
        verbose=0,
    )


def total_ir_model(par, fCF4, pressure, normCF4):
    total = sum(func(par, degrad_data, fCF4, pressure) for func in equations.values())
    return total * (1000.0 / normCF4)


def build_total_ir_experimental(exp_dict, pressure, normCF4):
    col = f"{float(pressure):.1f}bar"
    err_col = f"Err {col}"
    conc_col = "fCF4"
    merged = None
    for line in lines:
        df = exp_dict[line][[conc_col, col, err_col]].copy()
        df = df.rename(columns={col: f"y_{line}", err_col: f"e_{line}"})
        merged = df if merged is None else pd.merge(merged, df, on=conc_col, how="inner")
    if merged is None or merged.empty:
        return np.array([]), np.array([]), np.array([])
    x_percent = merged[conc_col].to_numpy(dtype=float)
    y_total = sum(merged[f"y_{line}"].to_numpy(dtype=float) for line in lines)
    err2_total = sum(merged[f"e_{line}"].to_numpy(dtype=float) ** 2 for line in lines)
    factor = 1000.0 / normCF4
    return x_percent, y_total * factor, np.sqrt(err2_total) * factor

# =========================================================
# CENTRAL PARAMETERS FROM PRIMARY FIT
# =========================================================
degrad_data = pd.read_csv(DATA_DIR_DEGRAD / "ArCF4_IR.csv")
par_primary = read_primary_parameters(DATA_DIR_PAR / "ArCF4_IR_primary.csv")
normCF4 = read_primary_parameters(DATA_DIR_PAR / "ArCF4_primary.csv")[0]

print("=" * 60)
print("Optimal IR line loaded from:", DATA_DIR_PAR / "ArCF4_IR_primary.csv")
print(par_primary)
print("=" * 60)

# =========================================================
# TOYS
# =========================================================
exp_stat = load_ir_experimental("stadistic", pressures=FIT_PRESSURES)
exp_sys = load_ir_experimental("sistematic", pressures=FIT_PRESSURES)

stat_params, stat_failures = fit_toy_parameters(
    N_STAT_TOYS,
    lambda rng: build_statistical_toy_dict(exp_stat, FIT_BAR_COLS, FIT_ERR_COLS, rng),
    run_fit,
    par_primary,
    seed=STAT_SEED,
)
syst_params, syst_failures = fit_toy_parameters(
    N_SYST_TOYS,
    lambda rng: build_correlated_systematic_toy_dict(exp_stat, exp_sys, FIT_BAR_COLS, FIT_ERR_COLS, rng,
                                                      group_map={line: line for line in lines}),
    run_fit,
    par_primary,
    seed=SYST_SEED,
)

print(f"Statistical toys accepted: {len(stat_params)} / {N_STAT_TOYS}; failed: {stat_failures}")
print(f"Systematic toys accepted:   {len(syst_params)} / {N_SYST_TOYS}; failed: {syst_failures}")

export_parameter_products(
    DATA_DIR_BANDS,
    TEX_DIR,
    "ArCF4_IR_primary",
    names_csv,
    names_tex,
    par_primary,
    stat_params,
    syst_params,
    caption=(r"Parámetros del ajuste primario IR en Ar--CF$_4$. La línea central procede "
             r"de data/Parameters/ArCF4_IR_primary.csv; las incertidumbres son de toys "
             r"estadísticos independientes y sistemáticos correlacionados por línea."),
    label="tab:ArCF4_IR_toy_uncertainties",
)

# =========================================================
# PLOT TOTAL IR
# =========================================================
exp_plot = load_ir_experimental("all", pressures=PLOT_PRESSURES)
fCF4_grid = np.logspace(-5, 0, 1000)
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(PLOT_PRESSURES)))

fig, ax = plt.subplots(figsize=(6.5, 4.4))

for i, pressure in enumerate(PLOT_PRESSURES):
    color = colors[i]

    def model_total(par, pressure=pressure):
        return total_ir_model(par, fCF4_grid, pressure, normCF4)

    y0 = model_total(par_primary)
    stat_curves = curves_from_parameters(stat_params, model_total)
    syst_curves = curves_from_parameters(syst_params, model_total)
    y_low_stat, y_up_stat = percentile_curve_band(stat_curves, y0)
    y_low_syst, y_up_syst = percentile_curve_band(syst_curves, y0)

    save_band_csv(
        DATA_DIR_BANDS / f"ArCF4_IR_primary_total_band_{pressure:g}bar.csv",
        fCF4_grid * 100,
        y0,
        y_low_stat,
        y_up_stat,
        y_low_syst,
        y_up_syst,
        metadata={"gas": "ArCF4", "channel": "IR_total", "pressure_bar": float(pressure), "scale": "ph_per_MeV"},
    )

    ax.fill_between(fCF4_grid * 100, y_low_syst, y_up_syst, color=color, alpha=0.24,
                    label="Sistemático" if i == 0 else None)
    ax.fill_between(fCF4_grid * 100, y_low_stat, y_up_stat, color=color, alpha=0.22,
                    label="Estadístico" if i == 0 else None)
    ax.plot(fCF4_grid * 100, y0, lw=2.0, color=color, label=f"{pressure:g} bar")

    x_exp, y_exp, yerr_exp = build_total_ir_experimental(exp_plot, pressure, normCF4)
    ax.errorbar(x_exp, y_exp, yerr=yerr_exp, marker="o", linestyle="none", ms=4,
                color=color, ecolor=color, elinewidth=1, capsize=2)

ax.set_xlim(1e-2,2e1)
ax.set_xscale("log")
ax.grid(False)
ax.set_xlabel(r"CF$_4$ concentration [\%]")
ax.set_ylabel("ph/MeV")
ax.set_title(r"Primary Ar--CF$_4$ IR yield (680--800 nm)")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ArCF4_IR_total_bands_toy_stat_syst.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved ArCF4 IR bands in:", DATA_DIR_BANDS)
