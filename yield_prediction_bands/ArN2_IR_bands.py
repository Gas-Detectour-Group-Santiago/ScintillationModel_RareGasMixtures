import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(__file__)

models_dir = os.path.abspath(os.path.join(BASE_DIR, "../models"))
data_dir   = os.path.abspath(os.path.join(BASE_DIR, "../data"))
fit_dir    = os.path.abspath(os.path.join(BASE_DIR, "../primary_fits"))

sys.path.append(models_dir)
sys.path.append(data_dir)
sys.path.append(fit_dir)

from ArN2_infrarred import *
from read_experimental import read_experimental
from fiting import fitParameters

# =========================================================
# HELPERS
# =========================================================
def apply_global_threshold(df, conc_col="N2 concentration (%)", is_727=False):
    bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
    err_cols = [f"Err {c}" for c in bar_cols]

    df_ref_50  = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    threshold_50  = df_ref_50[bar_cols].max().max()
    threshold_100 = df_ref_100[bar_cols].max().max()
    threshold = min(threshold_50, threshold_100)

    df_low = df[df[conc_col] < 50].copy()
    if is_727:
        df_low = df[df[conc_col] < 5].copy()

    mask = df_low[bar_cols] >= threshold
    df_low[bar_cols] = df_low[bar_cols].where(mask)

    for bar, err in zip(bar_cols, err_cols):
        df_low[err] = df_low[err].where(mask[bar])

    return df_low, threshold


def build_shifted_dict(exp_dict, bar_cols, err_cols):
    exp_low = {}
    exp_up = {}

    for key, df in exp_dict.items():
        df_low = df.copy(deep=True)
        df_up = df.copy(deep=True)

        y = df[bar_cols].to_numpy(dtype=float)
        e = df[err_cols].to_numpy(dtype=float)

        mask = ~np.isnan(y)

        y_low = np.where(mask, y - e, np.nan)
        y_up  = np.where(mask, y + e, np.nan)

        df_low.loc[:, bar_cols] = y_low
        df_up.loc[:, bar_cols]  = y_up

        exp_low[key] = df_low.fillna(0)
        exp_up[key]  = df_up.fillna(0)

    return exp_low, exp_up


def build_full_covariance(pcov, npar, free_idx):
    pcov = np.asarray(pcov, dtype=float)

    if pcov.shape == (npar, npar):
        return pcov

    if pcov.shape == (len(free_idx), len(free_idx)):
        full = np.zeros((npar, npar), dtype=float)
        full[np.ix_(free_idx, free_idx)] = pcov
        return full

    raise ValueError(
        f"Forma inesperada de pcov: {pcov.shape}. "
        f"Esperaba {(npar, npar)} o {(len(free_idx), len(free_idx))}."
    )


def statistical_band(model_func, par_natural, cov_full):
    y0 = model_func(par_natural)

    npts = len(y0)
    npar = len(par_natural)
    G = np.zeros((npts, npar), dtype=float)

    for j in range(npar):
        dp = np.zeros_like(par_natural)
        h = 1e-6 * max(abs(par_natural[j]), 1.0)
        dp[j] = h

        y_plus  = model_func(par_natural + dp)
        y_minus = model_func(par_natural - dp)

        G[:, j] = (y_plus - y_minus) / (2 * h)

    var_y = np.einsum("ij,jk,ik->i", G, cov_full, G)
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))

    return y0, y0 - sigma_y, y0 + sigma_y


def total_ir_model(par, degrad_data, fN2, pressure, norm_uv):
    total = (
        theory_yield_ArN2_Ir_696(par, degrad_data, fN2, pressure)
        + theory_yield_ArN2_Ir_727(par, degrad_data, fN2, pressure)
        + theory_yield_ArN2_Ir_750(par, degrad_data, fN2, pressure)
        + theory_yield_ArN2_Ir_763(par, degrad_data, fN2, pressure)
        + theory_yield_ArN2_Ir_772(par, degrad_data, fN2, pressure)
    )

    factor = 1000 / (norm_uv)
    return total * factor


def build_total_ir_experimental(exp_dict, pressure, norm_uv):
    """
    Construye los puntos experimentales totales IR (suma de 696, 727, 750, 763, 772)
    para una presión dada, ya escalados a ph/MeV.

    Solo conserva concentraciones donde existen las 5 líneas.
    """

    col = f"{pressure:.1f}bar"
    err_col = f"Err {pressure:.1f}bar"
    conc_col = "N2 concentration (%)"
    lines = ["696", "727", "750", "763", "772"]

    merged = None

    for line in lines:
        df = exp_dict[line][[conc_col, col, err_col]].copy()

        df = df.rename(columns={
            col: f"y_{line}",
            err_col: f"e_{line}"
        })

        if merged is None:
            merged = df
        else:
            merged = pd.merge(
                merged,
                df,
                on=conc_col,
                how="inner"   # solo concentraciones comunes a todas las líneas
            )

    if merged is None or merged.empty:
        return np.array([]), np.array([]), np.array([])

    x_percent = merged[conc_col].to_numpy(dtype=float)

    y_total = np.zeros(len(merged), dtype=float)
    err2_total = np.zeros(len(merged), dtype=float)

    for line in lines:
        y_total += merged[f"y_{line}"].to_numpy(dtype=float)
        err2_total += merged[f"e_{line}"].to_numpy(dtype=float) ** 2

    err_total = np.sqrt(err2_total)

    factor_pts = 1000 / norm_uv

    y_total_scaled = y_total * factor_pts
    err_total_scaled = err_total * factor_pts

    return x_percent, y_total_scaled, err_total_scaled

# =========================================================
# CONFIG
# =========================================================
DATA_DIR_EXP    = "../data/Experimental/ArN2/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR    = "../data/Parameters"

archivo_entrada = "../data/Experimental/ArN2/N2_primary_data_final.pkl"
yields = ["696", "727", "750", "763", "772"]
presiones = [1, 2, 3, 4, 5]
concentraciones_reales = None
output_dir = "../data/Experimental/ArN2/"

bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
err_cols = [f"Err {c}" for c in bar_cols]

fixed_idx = [1, 5, 9, 13, 17]
fixed_error = 0.1

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.15, 0.85, len(presiones)))

# =========================================================
# CARGA DE DATOS
# =========================================================
degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2_IR.csv"))
norm_uv = pd.read_csv(
    os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv")
)["parameter"].to_numpy()[0]


def load_ir_experimental(uncertainty_mode="all"):
    read_experimental(
        archivo_entrada,
        yields,
        presiones,
        output_dir,
        concentraciones_reales=concentraciones_reales,
        uncertainty_mode=uncertainty_mode
    )

    df_696 = pd.read_csv(os.path.join(DATA_DIR_EXP, "696.csv"))
    df_727 = pd.read_csv(os.path.join(DATA_DIR_EXP, "727.csv"))
    df_750 = pd.read_csv(os.path.join(DATA_DIR_EXP, "750.csv"))
    df_763 = pd.read_csv(os.path.join(DATA_DIR_EXP, "763.csv"))
    df_772 = pd.read_csv(os.path.join(DATA_DIR_EXP, "772.csv"))

    df_696, _ = apply_global_threshold(df_696)
    df_727, _ = apply_global_threshold(df_727, is_727=True)
    df_750, _ = apply_global_threshold(df_750)
    df_763, _ = apply_global_threshold(df_763)
    df_772, _ = apply_global_threshold(df_772)

    return {
        "696": df_696,
        "727": df_727,
        "750": df_750,
        "763": df_763,
        "772": df_772,
    }


# =========================================================
# PARÁMETROS DEL AJUSTE
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
    0.25, 0.0, 1.0, 1.0,
    0.25, 0.0, 1.0, 1.0,
    0.25, 0.0, 1.0, 1.0,
    0.25, 0.0, 1.0, 1.0,
    0.25, 0.0, 1.0, 1.0,
], dtype=float) + x0_semifixed

upper = np.array([
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
], dtype=float) + upper_semifixed

bounds = (list(lower), list(upper))

equations = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}

# =========================================================
# AJUSTE NOMINAL
# =========================================================
exp_nom = load_ir_experimental(uncertainty_mode="all")

experimental_data = {
    "696": exp_nom["696"].fillna(0),
    "727": exp_nom["727"].fillna(0),
    "750": exp_nom["750"].fillna(0),
    "763": exp_nom["763"].fillna(0),
    "772": exp_nom["772"].fillna(0),
}

popt = fitParameters(
    equations,
    experimental_data,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error
)

par_natural = popt.x.copy()
npar_full = len(par_natural)
free_idx = [i for i in range(npar_full) if i not in fixed_idx]

free_idx = [0,2,3,4,6,7,8,10,11,12,14,15,16,18,19]
cov_full = build_full_covariance(popt.pcov, npar_full, free_idx)

print("=" * 60)
print("Parámetros globales:\n", popt.x)
if hasattr(popt, "chi2"):
    print(f"Chi2 (real): {popt.chi2}")
if hasattr(popt, "dof"):
    print(f"Grados de libertad: {popt.dof}")
if hasattr(popt, "chi2_red"):
    print(f"Chi2 reducido: {popt.chi2_red}")
print("=" * 60)

# =========================================================
# REFITS SISTEMÁTICOS LOW / UP
# =========================================================
exp_sys = load_ir_experimental(uncertainty_mode="systematic")
experimental_data_low, experimental_data_up = build_shifted_dict(
    exp_sys, bar_cols, err_cols
)

popt_low = fitParameters(
    equations,
    experimental_data_low,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error
)

popt_up = fitParameters(
    equations,
    experimental_data_up,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error
)

par_low = popt_low.x.copy()
par_up  = popt_up.x.copy()

# =========================================================
# PLOT TOTAL IR BANDS
# =========================================================
os.makedirs("plots", exist_ok=True)

fN2 = np.logspace(-4, 0, 1000)
plot_pressures = [1, 2, 3, 4, 5]

plt.figure(figsize=(6.2, 4.2))

for i, pressure in enumerate(plot_pressures):
    color = colors[i]

    def model_total(par):
        return total_ir_model(par, degrad_data, fN2, pressure, norm_uv)

    y0, y_low_stat, y_up_stat = statistical_band(
        model_total, par_natural, cov_full
    )

    y_low_sys = model_total(par_low)
    y_up_sys  = model_total(par_up)

    ymin = np.minimum.reduce([y0, y_low_sys, y_up_sys])
    ymax = np.maximum.reduce([y0, y_low_sys, y_up_sys])

    plt.fill_between(
        fN2 * 100,
        ymin,
        ymax,
        alpha=0.28,
        color=color,
        label="Sistemático" if i == 0 else None
    )

    plt.fill_between(
        fN2 * 100,
        y_low_stat,
        y_up_stat,
        alpha=0.08,
        color=color,
        label="Estadístico" if i == 0 else None
    )

    plt.plot(
        fN2 * 100,
        y0,
        lw=2,
        color=color,
        label=f"{pressure} bar"
    )

    x_exp, y_exp, yerr_exp = build_total_ir_experimental(
        exp_nom, pressure, norm_uv
    )

    plt.errorbar(
        x_exp,
        y_exp,
        yerr=yerr_exp,
        marker="o",
        linestyle="none",
        ms=4,
        color=color,
        ecolor=color,
        elinewidth=1,
        capsize=2,
       # label=f"{pressure} bar data"
    )

plt.xscale("log")
# plt.yscale("log")

plt.grid(True, which="major", alpha=0.3)
plt.grid(True, which="minor", alpha=0.08)

plt.xlabel("N$_2$ concentration [\%]")
plt.ylabel("ph/MeV")
plt.title("Primary ArN$_2$ IR (680-785 nm) yield with bands")
plt.legend()
plt.tight_layout()

plt.savefig("plots/ArN2_IR_total_bands.pdf", dpi=300, bbox_inches="tight")
plt.show()