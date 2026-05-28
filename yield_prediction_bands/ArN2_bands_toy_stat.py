import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["grid"])

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(__file__)

models_dir = os.path.abspath(os.path.join(BASE_DIR, "../models"))
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../data"))
fit_dir = os.path.abspath(os.path.join(BASE_DIR, "../primary_fits"))

sys.path.append(models_dir)
sys.path.append(data_dir)
sys.path.append(fit_dir)

from ArN2 import *
from read_experimental import read_experimental
from fiting import fitParameters

# =========================================================
# CONFIG
# =========================================================
archivo_entrada = "../data/Experimental/ArN2/N2_primary_data_final.pkl"
output_dir_exp = "../data/Experimental/ArN2/"
data_dir_degrad = "../data/Primary_DegradData"
data_dir_par = "../data/Parameters"

plot_pressure = 1.0
yields = ["yield_N2"]
presiones = [1, 2, 2.5, 3, 4, 5]

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.15, 0.85, 6))

# Statistical band from independent pseudo-experiments.
# Do not use coherent +/- shifts for statistical errors and do not propagate the
# ill-conditioned covariance directly.
N_STAT_TOYS = 120
STAT_PERCENTILES = (16.0, 84.0)
RANDOM_SEED = 123456

# The primary Ar-N2 fit is now tied to the Ar-CF4 absolute normalization.
# Keep a fallback value so the script remains explicit and reproducible.
try:
    parameter_data_ArCF4 = pd.read_csv(
        os.path.join(data_dir_par, "ArCF4_primary.csv")
    )["parameter"].to_numpy(dtype=float)
    norm_uv = float(parameter_data_ArCF4[0])
except Exception:
    norm_uv = 0.0044564

# =========================================================
# DATA
# =========================================================
degrad_data = pd.read_csv(os.path.join(data_dir_degrad, "ArN2.csv"))

# =========================================================
# MODEL / FIT PARAMETERS
# Same configuration as the updated ArN2_fit.py.
# =========================================================
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

lower_og = np.array([
    0.0,
    0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0,
], dtype=float)

x0_og = np.array([
    norm_uv,
    0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0,
], dtype=float)

upper_og = np.array([
    1.0,
    0.7,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 1.0,
], dtype=float)

x0 = x0_og + x0_semifixed
bounds = (list(lower_og + lower_semifixed), list(upper_og + upper_semifixed))

equations = {"vis": theory_yield_N2_uv}
fixed_idx = [0, 2]
fixed_error = 0.376

# =========================================================
# HELPERS
# =========================================================
def W_N2(xN2, WAr=26.4, WN2=34.8):
    xN2 = np.asarray(xN2, dtype=float)
    return 1.0 / ((1.0 - xN2) / WAr + xN2 / WN2)


def pressure_columns(df):
    y_cols = []
    err_cols = []
    for col in df.columns:
        if col.endswith("bar") and not col.startswith("Err "):
            err = f"Err {col}"
            if err in df.columns:
                y_cols.append(col)
                err_cols.append(err)
    return y_cols, err_cols


def apply_w_scaling(df, conc_col="N2 concentration (%)"):
    df = df.copy(deep=True)
    y_cols, err_cols = pressure_columns(df)
    if not y_cols:
        return df

    w = W_N2(df[conc_col].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w)[:, None]
    df.loc[:, y_cols + err_cols] = df[y_cols + err_cols].to_numpy(dtype=float) * factor
    return df


def drop_pure_n2_points(df, conc_col="N2 concentration (%)"):
    df = df.copy(deep=True)
    return df[(df[conc_col] != 100) & (df[conc_col] != 150)].copy()


def call_read_experimental(uncertainty_mode):
    # Your codebase historically used both "sistematic" and "systematic".
    # Prefer the project spelling, but keep a fallback for older branches.
    modes_to_try = [uncertainty_mode]
    if uncertainty_mode == "sistematic":
        modes_to_try.append("systematic")
    elif uncertainty_mode == "systematic":
        modes_to_try.append("sistematic")

    last_error = None
    for mode in modes_to_try:
        try:
            read_experimental(
                archivo_entrada,
                yields,
                presiones,
                output_dir_exp,
                concentraciones_reales=None,
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_experimental(uncertainty_mode="all", w_scaled=True, drop_pure=True):
    call_read_experimental(uncertainty_mode)
    df = pd.read_csv(os.path.join(output_dir_exp, "yield_N2.csv"))
    if drop_pure:
        df = drop_pure_n2_points(df)
    if w_scaled:
        df = apply_w_scaling(df)
    return df


def build_full_covariance(pcov, npar, free_idx):
    pcov = np.asarray(pcov, dtype=float)
    if pcov.shape == (npar, npar):
        return pcov
    if pcov.shape == (len(free_idx), len(free_idx)):
        full = np.zeros((npar, npar), dtype=float)
        full[np.ix_(free_idx, free_idx)] = pcov
        return full
    raise ValueError(
        f"Forma inesperada de la covarianza: {pcov.shape}; "
        f"esperaba {(npar, npar)} o {(len(free_idx), len(free_idx))}."
    )


def covariance_from_result(popt, fixed_idx):
    npar = len(popt.x)
    free_idx = [i for i in range(npar) if i not in fixed_idx]

    if hasattr(popt, "pcov"):
        return build_full_covariance(popt.pcov, npar, free_idx)

    J = popt.jac
    m, p = J.shape
    s2 = 2.0 * popt.cost / max(m - p, 1)
    pcov = s2 * np.linalg.pinv(J.T @ J)
    return build_full_covariance(pcov, npar, free_idx)


def statistical_band(model_func, par_natural, cov_full):
    y0 = np.asarray(model_func(par_natural), dtype=float)

    npts = len(y0)
    npar = len(par_natural)
    G = np.zeros((npts, npar), dtype=float)

    for j in range(npar):
        dp = np.zeros_like(par_natural)
        h = 1e-6 * max(abs(par_natural[j]), 1.0)
        dp[j] = h

        y_plus = np.asarray(model_func(par_natural + dp), dtype=float)
        y_minus = np.asarray(model_func(par_natural - dp), dtype=float)
        G[:, j] = (y_plus - y_minus) / (2.0 * h)

    var_y = np.einsum("ij,jk,ik->i", G, cov_full, G)
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))
    return y0, y0 - sigma_y, y0 + sigma_y


def envelope_from_nominal_up_down(y0, y_low, y_up):
    ymin = np.minimum.reduce([y0, y_low, y_up])
    ymax = np.maximum.reduce([y0, y_low, y_up])
    return ymin, ymax



def make_statistical_toy_dataframe(df, rng):
    """
    Independent statistical toy for one dataframe.

    Statistical uncertainties are point-to-point, so each valid cell is moved
    independently.  This is the key difference with a systematic band, where a
    coherent up/down displacement is acceptable.
    """
    toy = df.copy(deep=True)
    y_cols, err_cols = pressure_columns(toy)

    y = toy[y_cols].to_numpy(dtype=float)
    e = toy[err_cols].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(e) & (e > 0.0)

    fluct = rng.normal(loc=0.0, scale=1.0, size=y.shape) * e
    toy.loc[:, y_cols] = np.where(mask, y + fluct, y)

    return toy.fillna(0)


def fit_statistical_toy_curves(df_stat, model_func, nominal_parameters):
    """
    Fits many independent statistical pseudo-experiments and returns the
    16--84 percentile envelope of the resulting curves.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    curves = []
    failures = 0

    for _ in range(N_STAT_TOYS):
        toy_df = make_statistical_toy_dataframe(df_stat, rng)
        try:
            toy_fit = fitParameters(
                equations,
                {"vis": toy_df},
                degrad_data,
                x0=nominal_parameters,
                bounds=bounds,
                is_infrared=True,
                fixed_idx=fixed_idx,
                fixed_error=fixed_error,
            )
            curves.append(np.asarray(model_func(toy_fit.x), dtype=float))
        except Exception:
            failures += 1

    if len(curves) == 0:
        y_nom = np.asarray(model_func(nominal_parameters), dtype=float)
        return y_nom.copy(), y_nom.copy(), failures

    curves = np.asarray(curves, dtype=float)
    y_low, y_up = np.nanpercentile(curves, STAT_PERCENTILES, axis=0)
    return y_low, y_up, failures

# =========================================================
# NOMINAL FIT
# =========================================================
yield_nominal = load_experimental(uncertainty_mode="all", w_scaled=True, drop_pure=True)
experimental_data = {"vis": yield_nominal}

popt = fitParameters(
    equations,
    experimental_data,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error,
)

par_natural = popt.x.copy()
cov_full = covariance_from_result(popt, fixed_idx)

if hasattr(popt, "chi2"):
    chi2 = popt.chi2
elif hasattr(popt, "cost"):
    chi2 = 2.0 * popt.cost
else:
    chi2 = np.nan

if hasattr(popt, "dof"):
    dof = popt.dof
elif hasattr(popt, "fun"):
    dof = len(popt.fun) - len(par_natural)
else:
    dof = np.nan

chi2_red = chi2 / dof if np.isfinite(chi2) and np.isfinite(dof) and dof != 0 else np.nan

print("=" * 60)
print("Parámetros globales:\n", par_natural)
print(f"Grados de libertad: {dof}")
print(f"Chi2 (real): {chi2}")
print(f"Chi2 reducido: {chi2_red}")
print("=" * 60)

# =========================================================
# SYSTEMATIC LOW / UP REFITS
# =========================================================
yield_sys = load_experimental(uncertainty_mode="sistematic", w_scaled=True, drop_pure=True)
y_cols, err_cols = pressure_columns(yield_sys)

err_sys = yield_sys[err_cols].to_numpy(dtype=float)
yield_low = yield_sys.copy(deep=True)
yield_up = yield_sys.copy(deep=True)
yield_low.loc[:, y_cols] = yield_low[y_cols].to_numpy(dtype=float) - err_sys
yield_up.loc[:, y_cols] = yield_up[y_cols].to_numpy(dtype=float) + err_sys

popt_low = fitParameters(
    equations,
    {"vis": yield_low},
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error,
)

popt_up = fitParameters(
    equations,
    {"vis": yield_up},
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error,
)

par_low = popt_low.x.copy()
par_up = popt_up.x.copy()

# =========================================================
# DATA FOR PLOTS
# =========================================================
yield_plot = load_experimental(uncertainty_mode="all", w_scaled=True, drop_pure=True)
yield_plot.loc[yield_plot["N2 concentration (%)"] <= 0, "N2 concentration (%)"] = 1e-6

# =========================================================
# CURVES AND BANDS
# =========================================================
fN2 = np.logspace(-4, 0, 1000)
scale_to_ph_mev = 1000.0 / norm_uv


def model_total(par):
    return theory_yield_N2_uv(par, degrad_data, fN2, plot_pressure)


y0 = model_total(par_natural)

# Statistical band: independent pseudo-experiments.
yield_stat = load_experimental(uncertainty_mode="stadistic", w_scaled=True, drop_pure=True)
y_low_stat, y_up_stat, n_stat_failed = fit_statistical_toy_curves(
    yield_stat,
    model_total,
    par_natural,
)

# Systematic band: coherent low/up refits.
y_low_sys = model_total(par_low)
y_up_sys = model_total(par_up)
y_sys_min, y_sys_max = envelope_from_nominal_up_down(y0, y_low_sys, y_up_sys)



# =========================================================
# PLOT: PH/MEV SCALE
# =========================================================

col = f"{plot_pressure:.1f}bar"
err_col = f"Err {plot_pressure:.1f}bar"

con_uv_cf4_macfly = [100.0]
y_uv_cf4_macfly   = [94]
y_err_uv_cf4_macfly= [94 * 14e-2] 

con_uv_cf4_morii = [100.0]
y_uv_cf4_morii   = [145]
y_err_uv_cf4_morii= [2]

plt.figure(figsize=(6.5, 4.5))
plt.fill_between(
    fN2 * 100,
    y_sys_min * scale_to_ph_mev,
    y_sys_max * scale_to_ph_mev,
    alpha=0.30,
    label="Sistemático",
    color=colors[0],
)
plt.fill_between(
    fN2 * 100,
    y_low_stat * scale_to_ph_mev,
    y_up_stat * scale_to_ph_mev,
    alpha=0.22,
    label="Estadístico",
    color=colors[1],
    zorder=3,
)
plt.plot(fN2 * 100, y_low_stat * scale_to_ph_mev, lw=0.7, ls="--", color=colors[1], zorder=4)
plt.plot(fN2 * 100, y_up_stat * scale_to_ph_mev, lw=0.7, ls="--", color=colors[1], zorder=4)
plt.plot(fN2 * 100, y0 * scale_to_ph_mev, lw=2, label="Ajuste nominal", color=colors[0])
plt.errorbar(
    yield_plot["N2 concentration (%)"],
    yield_plot[col] * scale_to_ph_mev,
    yerr=yield_plot[err_col] * scale_to_ph_mev,
    marker="o",
    linestyle="none",
    ms=4,
    color=colors[0],
    ecolor=colors[0],
    elinewidth=1,
    capsize=2,
    label=f"X-ray {plot_pressure:g} bar",
)


plt.errorbar(con_uv_cf4_macfly,
             y_uv_cf4_macfly,
             yerr=y_err_uv_cf4_macfly,
            marker="v",
            linestyle="none",
            ms=5,
            color=colors[1],
            ecolor=colors[1],
            elinewidth=1,
            capsize=2,
            label="MacFly Colaboration")




plt.errorbar(con_uv_cf4_morii,
             y_uv_cf4_morii,
             yerr=y_err_uv_cf4_morii,
            marker="o",
            linestyle="none",
            ms=5,
            color=colors[2],
            ecolor=colors[2],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$ Morii")



plt.xscale("log")
plt.yscale("log")
plt.xlim(5e-2, 110)
plt.ylim(5e1, 3500)
plt.xlabel("N$_2$ concentration [$\\%$]")
plt.ylabel("ph/MeV")
plt.title("Primary Ar-N$_2$ UV (300--420 nm) ")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArN2_bands.pdf", dpi=300, bbox_inches="tight")
plt.show()


print("STAT toy failures:", n_stat_failed, "/", N_STAT_TOYS)
print("STAT toy min/max:", np.nanmin(y_up_stat - y_low_stat), np.nanmax(y_up_stat - y_low_stat))
