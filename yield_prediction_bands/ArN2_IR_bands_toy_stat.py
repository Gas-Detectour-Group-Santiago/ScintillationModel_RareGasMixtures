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

from ArN2_infrarred import (  # noqa: E402
    theory_yield_ArN2_Ir_696,
    theory_yield_ArN2_Ir_727,
    theory_yield_ArN2_Ir_750,
    theory_yield_ArN2_Ir_763,
    theory_yield_ArN2_Ir_772,
)
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
lines = ["696", "727", "750", "763", "772"]
PLOT_PRESSURES = [1, 2, 3, 4, 5]
BAR_COLS, ERR_COLS = pressure_cols(PLOT_PRESSURES)

N_STAT_TOYS = int(os.environ.get("N_STAT_TOYS", "150"))
N_SYST_TOYS = int(os.environ.get("N_SYST_TOYS", "150"))
STAT_SEED = 223344
SYST_SEED = 443322
fixed_idx = [1, 5, 9, 13, 17]
fixed_error = 0.1

# =========================================================
# DATA / MODEL
# =========================================================
degrad_data = pd.read_csv(DATA_DIR_DEGRAD / "ArN2_IR.csv")
if "Ar_794" not in degrad_data.columns:
    degrad_data["Ar_794"] = 0.0

# =========================================================
# HELPERS
# =========================================================
def W_N2(xN2, WAr=26.4, WN2=34.8):
    xN2 = np.asarray(xN2, dtype=float)
    return 1.0 / ((1.0 - xN2) / WAr + xN2 / WN2)


def apply_w_scaling(df, conc_col="N2 concentration (%)"):
    df = df.copy(deep=True)
    cols = [c for c in BAR_COLS + ERR_COLS if c in df.columns]
    if not cols:
        return df
    w = W_N2(df[conc_col].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w)[:, None]
    df.loc[:, cols] = df[cols].to_numpy(dtype=float) * factor
    return df


def apply_global_threshold(df, conc_col="N2 concentration (%)", is_727=False):
    df_ref_50 = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()
    threshold_50 = df_ref_50[BAR_COLS].max().max() if not df_ref_50.empty else 0.0
    threshold_100 = df_ref_100[BAR_COLS].max().max() if not df_ref_100.empty else 0.0
    threshold = min(threshold_50, threshold_100)

    df_low = df[df[conc_col] < 50].copy()
    if is_727:
        df_low = df[df[conc_col] < 5].copy()

    mask = df_low[BAR_COLS] >= threshold
    df_low.loc[:, BAR_COLS] = df_low[BAR_COLS].where(mask)
    for bar, err in zip(BAR_COLS, ERR_COLS):
        df_low.loc[:, err] = df_low[err].where(mask[bar])
    return df_low[[conc_col] + BAR_COLS + ERR_COLS]


def call_read_experimental(uncertainty_mode):
    aliases = {"systematic": ["sistematic"], "sistematic": ["systematic"], "stadistic": ["statistic", "statistical"]}
    last_error = None
    for mode in [uncertainty_mode] + aliases.get(uncertainty_mode, []):
        try:
            read_experimental(
                str(archivo_entrada),
                lines,
                PLOT_PRESSURES,
                str(DATA_DIR_EXP),
                concentraciones_reales=None,
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_ir_experimental(uncertainty_mode="all"):
    call_read_experimental(uncertainty_mode)
    raw = {line: pd.read_csv(DATA_DIR_EXP / f"{line}.csv") for line in lines}
    out = {}
    for line, df in raw.items():
        df = apply_w_scaling(df)
        out[line] = apply_global_threshold(df, is_727=(line == "727"))
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
        fixed_idx=fixed_idx,
        fixed_error=fixed_error,
        verbose=0,
    )


def total_ir_model(par, fN2, pressure, norm_uv):
    total = sum(func(par, degrad_data, fN2, pressure) for func in equations.values())
    return total * 1000.0 / norm_uv


def build_total_ir_experimental(exp_dict, pressure, norm_uv):
    col = f"{float(pressure):.1f}bar"
    err_col = f"Err {col}"
    conc_col = "N2 concentration (%)"
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
    factor = 1000.0 / norm_uv
    return x_percent, y_total * factor, np.sqrt(err2_total) * factor

# =========================================================
# FIT PARAMETERS
# =========================================================
x0_semifixed = np.array([
    0.0, 28.3, 0.0, 0.0,
    0.0, 28.3, 0.0, 0.0,
    0.0, 21.7, 0.0, 0.0,
    0.0, 29.4, 0.0, 0.0,
    0.0, 28.3, 0.0, 0.0,
], dtype=float)

lower_semifixed = x0_semifixed * 0.999999999999999
upper_semifixed = x0_semifixed * 1.000000000000001

lower = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
], dtype=float) + lower_semifixed

x0 = np.array([
    0.0159, 0.0, 1.0, 1.0,
    0.0159, 0.0, 1.0, 1.0,
    0.0159, 0.0, 1.0, 1.0,
    0.0159, 0.0, 1.0, 1.0,
    0.0159, 0.0, 1.0, 1.0,
], dtype=float) + x0_semifixed

upper = np.array([
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
    0.02, 0.0, 1000.0, 1000.0,
], dtype=float) + upper_semifixed

bounds = (list(lower), list(upper))

equations = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}

names_tex = [
    r"$P_{\mathrm{Ar}^*\,696}$", r"$\tau_{696}$", r"$K_{\mathrm{Ar},696}$", r"$K_{\mathrm{N}_2,696}$",
    r"$P_{\mathrm{Ar}^*\,727}$", r"$\tau_{727}$", r"$K_{\mathrm{Ar},727}$", r"$K_{\mathrm{N}_2,727}$",
    r"$P_{\mathrm{Ar}^*\,750}$", r"$\tau_{750}$", r"$K_{\mathrm{Ar},750}$", r"$K_{\mathrm{N}_2,750}$",
    r"$P_{\mathrm{Ar}^*\,764}$", r"$\tau_{764}$", r"$K_{\mathrm{Ar},764}$", r"$K_{\mathrm{N}_2,764}$",
    r"$P_{\mathrm{Ar}^*\,772}$", r"$\tau_{772}$", r"$K_{\mathrm{Ar},772}$", r"$K_{\mathrm{N}_2,772}$",
]

names_csv = [
    "PAr_star_696", "tau_N2_696", "K_Ar_Q_Ar_696", "K_Ar_Q_N2_696",
    "PAr_star_727", "tau_N2_727", "K_Ar_Q_Ar_727", "K_Ar_Q_N2_727",
    "PAr_star_750", "tau_N2_750", "K_Ar_Q_Ar_750", "K_Ar_Q_N2_750",
    "PAr_star_764", "tau_N2_764", "K_Ar_Q_Ar_764", "K_Ar_Q_N2_764",
    "PAr_star_772", "tau_N2_772", "K_Ar_Q_Ar_772", "K_Ar_Q_N2_772",
]

# =========================================================
# CENTRAL PARAMETERS FROM PRIMARY FIT
# =========================================================
par_primary = read_primary_parameters(DATA_DIR_PAR / "ArN2_IR_primary.csv")
norm_uv = read_primary_parameters(DATA_DIR_PAR / "ArN2_primary.csv")[0]

print("=" * 60)
print("Optimal Ar-N2 IR line loaded from:", DATA_DIR_PAR / "ArN2_IR_primary.csv")
print(par_primary)
print("=" * 60)

# =========================================================
# TOYS
# =========================================================
exp_stat = load_ir_experimental("stadistic")
exp_sys = load_ir_experimental("sistematic")

stat_params, stat_failures = fit_toy_parameters(
    N_STAT_TOYS,
    lambda rng: build_statistical_toy_dict(exp_stat, BAR_COLS, ERR_COLS, rng),
    run_fit,
    par_primary,
    seed=STAT_SEED,
)
syst_params, syst_failures = fit_toy_parameters(
    N_SYST_TOYS,
    lambda rng: build_correlated_systematic_toy_dict(exp_stat, exp_sys, BAR_COLS, ERR_COLS, rng,
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
    "ArN2_IR_primary",
    names_csv,
    names_tex,
    par_primary,
    stat_params,
    syst_params,
    caption=(r"Parámetros del ajuste primario IR en Ar--N$_2$. La línea central procede "
             r"de data/Parameters/ArN2_IR_primary.csv; las incertidumbres son de toys "
             r"estadísticos independientes y sistemáticos correlacionados por línea."),
    label="tab:ArN2_IR_toy_uncertainties",
)

# =========================================================
# PLOT TOTAL IR
# =========================================================
exp_plot = load_ir_experimental("all")
fN2 = np.logspace(-4, 0, 1000)
plot_pressures = PLOT_PRESSURES
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(plot_pressures)))

fig, ax = plt.subplots(figsize=(6.5, 4.4))

for i, pressure in enumerate(plot_pressures):
    color = colors[i]

    def model_total(par, pressure=pressure):
        return total_ir_model(par, fN2, pressure, norm_uv)

    y0 = model_total(par_primary)
    stat_curves = curves_from_parameters(stat_params, model_total)
    syst_curves = curves_from_parameters(syst_params, model_total)
    y_low_stat, y_up_stat = percentile_curve_band(stat_curves, y0)
    y_low_syst, y_up_syst = percentile_curve_band(syst_curves, y0)

    save_band_csv(
        DATA_DIR_BANDS / f"ArN2_IR_primary_total_band_{pressure:g}bar.csv",
        fN2 * 100,
        y0,
        y_low_stat,
        y_up_stat,
        y_low_syst,
        y_up_syst,
        metadata={"gas": "ArN2", "channel": "IR_total", "pressure_bar": float(pressure), "scale": "ph_per_MeV"},
    )

    ax.fill_between(fN2 * 100, y_low_syst, y_up_syst, color=color, alpha=0.24,
                    label="Sistemático" if i == 0 else None)
    ax.fill_between(fN2 * 100, y_low_stat, y_up_stat, color=color, alpha=0.22,
                    label="Estadístico" if i == 0 else None)
    ax.plot(fN2 * 100, y0, lw=2.0, color=color, label=f"{pressure:g} bar")

    x_exp, y_exp, yerr_exp = build_total_ir_experimental(exp_plot, pressure, norm_uv)
    ax.errorbar(x_exp, y_exp, yerr=yerr_exp, marker="o", linestyle="none", ms=4,
                color=color, ecolor=color, elinewidth=1, capsize=2)

ax.set_xlim(1e-2,2e1)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(False)
ax.set_xlabel(r"N$_2$ concentration [\%]")
ax.set_ylabel("ph/MeV")
ax.set_title(r"Primary Ar--N$_2$ IR yield (680--785 nm)")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "ArN2_IR_total_bands_toy_stat_syst.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved ArN2 IR bands in:", DATA_DIR_BANDS)
