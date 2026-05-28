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
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../data"))
fit_dir = os.path.abspath(os.path.join(BASE_DIR, "../primary_fits"))

sys.path.append(models_dir)
sys.path.append(data_dir)
sys.path.append(fit_dir)

from ArN2_infrarred import *
from read_experimental import read_experimental
from fiting import fitParameters

# =========================================================
# CONFIG
# =========================================================
DATA_DIR_EXP = "../data/Experimental/ArN2/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

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

N_STAT_TOYS = 120
STAT_PERCENTILES = (16.0, 84.0)
RANDOM_SEED = 223344

# =========================================================
# DATA
# =========================================================
degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2_IR.csv"))
norm_uv = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_primary.csv"))["parameter"].to_numpy(dtype=float)[0]

# =========================================================
# HELPERS
# =========================================================
def W_N2(xN2, WAr=26.4, WN2=34.8):
    xN2 = np.asarray(xN2, dtype=float)
    return 1.0 / ((1.0 - xN2) / WAr + xN2 / WN2)


def apply_w_scaling(df, conc_col="N2 concentration (%)"):
    """Apply the same 1/W(f) preprocessing used in ArN2_IR_fit.py."""
    df = df.copy(deep=True)
    cols = [c for c in bar_cols + err_cols if c in df.columns]
    if not cols:
        return df

    w = W_N2(df[conc_col].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w)[:, None]
    df.loc[:, cols] = df[cols].to_numpy(dtype=float) * factor
    return df


def apply_global_threshold(df, conc_col="N2 concentration (%)", is_727=False):
    df_ref_50 = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    threshold_50 = df_ref_50[bar_cols].max().max()
    threshold_100 = df_ref_100[bar_cols].max().max()
    threshold = min(threshold_50, threshold_100)

    df_low = df[df[conc_col] < 50].copy()
    if is_727:
        df_low = df[df[conc_col] < 5].copy()

    mask = df_low[bar_cols] >= threshold
    df_low.loc[:, bar_cols] = df_low[bar_cols].where(mask)

    for bar, err in zip(bar_cols, err_cols):
        df_low.loc[:, err] = df_low[err].where(mask[bar])

    return df_low, threshold


def call_read_experimental(uncertainty_mode):
    modes_to_try = [uncertainty_mode]
    if uncertainty_mode == "sistematic":
        modes_to_try.append("systematic")
    elif uncertainty_mode == "systematic":
        modes_to_try.append("sistematic")
    elif uncertainty_mode == "stadistic":
        modes_to_try.extend(["statistic", "statistical", "estadistico", "estadistic"])
    elif uncertainty_mode in ("statistic", "statistical", "estadistico", "estadistic"):
        modes_to_try.append("stadistic")

    last_error = None
    for mode in modes_to_try:
        try:
            read_experimental(
                archivo_entrada,
                yields,
                presiones,
                output_dir,
                concentraciones_reales=concentraciones_reales,
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_ir_experimental(uncertainty_mode="all"):
    call_read_experimental(uncertainty_mode)

    df_696 = pd.read_csv(os.path.join(DATA_DIR_EXP, "696.csv"))
    df_727 = pd.read_csv(os.path.join(DATA_DIR_EXP, "727.csv"))
    df_750 = pd.read_csv(os.path.join(DATA_DIR_EXP, "750.csv"))
    df_763 = pd.read_csv(os.path.join(DATA_DIR_EXP, "763.csv"))
    df_772 = pd.read_csv(os.path.join(DATA_DIR_EXP, "772.csv"))

    # Critical update: the IR fit first divides both yields and errors by W(f).
    df_696 = apply_w_scaling(df_696)
    df_727 = apply_w_scaling(df_727)
    df_750 = apply_w_scaling(df_750)
    df_763 = apply_w_scaling(df_763)
    df_772 = apply_w_scaling(df_772)

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


def build_shifted_dict(exp_dict, bar_cols, err_cols):
    exp_low = {}
    exp_up = {}

    for key, df in exp_dict.items():
        df_low = df.copy(deep=True)
        df_up = df.copy(deep=True)

        y = df[bar_cols].to_numpy(dtype=float)
        e = df[err_cols].to_numpy(dtype=float)
        mask = ~np.isnan(y)

        df_low.loc[:, bar_cols] = np.where(mask, y - e, np.nan)
        df_up.loc[:, bar_cols] = np.where(mask, y + e, np.nan)

        exp_low[key] = df_low.fillna(0)
        exp_up[key] = df_up.fillna(0)

    return exp_low, exp_up



def build_statistical_toy_dict(exp_dict, bar_cols, err_cols, rng):
    """Independent point-to-point statistical toy for all IR lines."""
    toy = {}
    for key, df in exp_dict.items():
        df_toy = df.copy(deep=True)
        y = df_toy[bar_cols].to_numpy(dtype=float)
        e = df_toy[err_cols].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(e) & (e > 0.0)
        fluct = rng.normal(0.0, 1.0, size=y.shape) * e
        df_toy.loc[:, bar_cols] = np.where(mask, y + fluct, y)
        toy[key] = df_toy.fillna(0)
    return toy


def fit_statistical_toys(exp_stat, nominal_parameters):
    rng = np.random.default_rng(RANDOM_SEED)
    pars = []
    failures = 0

    for _ in range(N_STAT_TOYS):
        toy_data = build_statistical_toy_dict(exp_stat, bar_cols, err_cols, rng)
        try:
            toy_fit = fitParameters(
                equations,
                toy_data,
                degrad_data,
                x0=nominal_parameters,
                bounds=bounds,
                is_infrared=True,
                fixed_idx=fixed_idx,
                fixed_error=fixed_error,
            )
            pars.append(toy_fit.x.copy())
        except Exception:
            failures += 1

    return np.asarray(pars, dtype=float), failures



def covariance_from_result(popt, fixed_idx):
    """
    Devuelve (cov_reduced, free_idx), es decir, la covarianza solo en el
    subespacio de parámetros libres. Esto replica la lógica de las bands de
    Ar-CF4: no propagamos parámetros fijos ni multiplicamos derivadas NaN por
    covarianzas nulas.
    """
    npar = len(popt.x)
    free_idx = [i for i in range(npar) if i not in fixed_idx]
    nfree = len(free_idx)

    def _clean_cov(cov):
        cov = np.asarray(cov, dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.T)
        return cov

    pcov = getattr(popt, "pcov", None)
    if pcov is not None:
        pcov = np.asarray(pcov, dtype=float)
        if pcov.shape == (npar, npar):
            cov_red = pcov[np.ix_(free_idx, free_idx)]
            cov_red = _clean_cov(cov_red)
            if np.any(np.diag(cov_red) > 0):
                return cov_red, free_idx
        elif pcov.shape == (nfree, nfree):
            cov_red = _clean_cov(pcov)
            if np.any(np.diag(cov_red) > 0):
                return cov_red, free_idx

    if not hasattr(popt, "jac"):
        raise ValueError("No hay ni popt.pcov útil ni popt.jac para construir la covarianza estadística.")

    J = np.asarray(popt.jac, dtype=float)
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    if J.shape[1] == npar:
        J_red = J[:, free_idx]
    elif J.shape[1] == nfree:
        J_red = J
    else:
        raise ValueError(
            f"No puedo mapear la Jacobiana: J.shape={J.shape}, npar={npar}, nfree={nfree}."
        )

    m, p = J_red.shape
    if hasattr(popt, "chi2") and hasattr(popt, "dof") and popt.dof > 0:
        s2 = popt.chi2 / popt.dof
    else:
        s2 = 2.0 * popt.cost / max(m - p, 1)

    cov_red = s2 * np.linalg.pinv(J_red.T @ J_red, rcond=1e-12)
    cov_red = _clean_cov(cov_red)
    return cov_red, free_idx


def _eval_model_finite(model_func, par):
    y = np.asarray(model_func(par), dtype=float)
    if y.ndim != 1:
        y = np.ravel(y)
    return y


def statistical_band(model_func, par_natural, cov_red, free_idx, bounds=None):
    """
    Propagación estadística real por derivadas numéricas en los parámetros libres.
    No se ensancha artificialmente la banda. Para parámetros pegados a bounds se
    usa derivada one-sided para no evaluar el modelo fuera de la región física.
    """
    par_natural = np.asarray(par_natural, dtype=float)
    y0 = _eval_model_finite(model_func, par_natural)

    npts = len(y0)
    G = np.zeros((npts, len(free_idx)), dtype=float)

    if bounds is not None:
        lower_b = np.asarray(bounds[0], dtype=float)
        upper_b = np.asarray(bounds[1], dtype=float)
    else:
        lower_b = np.full_like(par_natural, -np.inf)
        upper_b = np.full_like(par_natural, np.inf)

    for k, j in enumerate(free_idx):
        h0 = 1e-5 * max(abs(par_natural[j]), 1.0)
        span = upper_b[j] - lower_b[j]
        if np.isfinite(span) and span > 0:
            h0 = min(h0, 1e-3 * span)
        h0 = max(h0, 1e-10)

        p_plus = par_natural.copy()
        p_minus = par_natural.copy()

        can_plus = par_natural[j] + h0 <= upper_b[j]
        can_minus = par_natural[j] - h0 >= lower_b[j]

        if can_plus and can_minus:
            p_plus[j] += h0
            p_minus[j] -= h0
            y_plus = _eval_model_finite(model_func, p_plus)
            y_minus = _eval_model_finite(model_func, p_minus)
            deriv = (y_plus - y_minus) / (2.0 * h0)
        elif can_plus:
            p_plus[j] += h0
            y_plus = _eval_model_finite(model_func, p_plus)
            deriv = (y_plus - y0) / h0
        elif can_minus:
            p_minus[j] -= h0
            y_minus = _eval_model_finite(model_func, p_minus)
            deriv = (y0 - y_minus) / h0
        else:
            deriv = np.zeros_like(y0)

        G[:, k] = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)

    cov_red = np.asarray(cov_red, dtype=float)
    cov_red = np.nan_to_num(cov_red, nan=0.0, posinf=0.0, neginf=0.0)
    cov_red = 0.5 * (cov_red + cov_red.T)

    var_y = np.einsum("ij,jk,ik->i", G, cov_red, G)
    var_y = np.nan_to_num(var_y, nan=0.0, posinf=0.0, neginf=0.0)
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
    return total * 1000.0 / norm_uv


def build_total_ir_experimental(exp_dict, pressure, norm_uv):
    col = f"{pressure:.1f}bar"
    err_col = f"Err {pressure:.1f}bar"
    conc_col = "N2 concentration (%)"
    lines = ["696", "727", "750", "763", "772"]

    merged = None
    for line in lines:
        df = exp_dict[line][[conc_col, col, err_col]].copy()
        df = df.rename(columns={col: f"y_{line}", err_col: f"e_{line}"})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=conc_col, how="inner")

    if merged is None or merged.empty:
        return np.array([]), np.array([]), np.array([])

    x_percent = merged[conc_col].to_numpy(dtype=float)
    y_total = np.zeros(len(merged), dtype=float)
    err2_total = np.zeros(len(merged), dtype=float)

    for line in lines:
        y_total += merged[f"y_{line}"].to_numpy(dtype=float)
        err2_total += merged[f"e_{line}"].to_numpy(dtype=float) ** 2

    err_total = np.sqrt(err2_total)
    factor_pts = 1000.0 / norm_uv
    return x_percent, y_total * factor_pts, err_total * factor_pts

# =========================================================
# FIT PARAMETERS
# Same configuration as the updated ArN2_IR_fit.py.
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

# =========================================================
# NOMINAL FIT
# =========================================================
exp_nom = load_ir_experimental(uncertainty_mode="all")
experimental_data = {key: df.fillna(0) for key, df in exp_nom.items()}

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
cov_red, free_idx = covariance_from_result(popt, fixed_idx)

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
# SYSTEMATIC LOW / UP REFITS
# =========================================================
exp_sys = load_ir_experimental(uncertainty_mode="sistematic")
experimental_data_low, experimental_data_up = build_shifted_dict(exp_sys, bar_cols, err_cols)

popt_low = fitParameters(
    equations,
    experimental_data_low,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error,
)

popt_up = fitParameters(
    equations,
    experimental_data_up,
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
# STATISTICAL TOYS
# =========================================================
# Statistical errors are point-to-point.  We therefore build independent
# pseudo-experiments instead of shifting all points coherently up/down.
exp_stat = load_ir_experimental(uncertainty_mode="stadistic")
par_stat_toys, n_stat_failed = fit_statistical_toys(exp_stat, par_natural)

# =========================================================
# PLOT TOTAL IR BANDS
# =========================================================
os.makedirs("plots", exist_ok=True)

fN2 = np.logspace(-4, 0, 1000)
plot_pressures = [1, 2, 3, 4, 5]

plt.figure(figsize=(6.2, 4.2))

for i, pressure in enumerate(plot_pressures):
    color = colors[i]

    def model_total(par, pressure=pressure):
        return total_ir_model(par, degrad_data, fN2, pressure, norm_uv)

    y0 = model_total(par_natural)
    y_low_sys = model_total(par_low)
    y_up_sys = model_total(par_up)
    if len(par_stat_toys) > 0:
        stat_curves = np.asarray([model_total(par) for par in par_stat_toys], dtype=float)
        y_stat_min, y_stat_max = np.nanpercentile(stat_curves, STAT_PERCENTILES, axis=0)
    else:
        y_stat_min = y0.copy()
        y_stat_max = y0.copy()

    ymin = np.minimum.reduce([y0, y_low_sys, y_up_sys])
    ymax = np.maximum.reduce([y0, y_low_sys, y_up_sys])

    plt.fill_between(
        fN2 * 100,
        ymin,
        ymax,
        alpha=0.28,
        color=color,
        label="Sistemático" if i == 0 else None,
    )
    plt.fill_between(
        fN2 * 100,
        y_stat_min,
        y_stat_max,
        alpha=0.22,
        color=color,
        label="Estadístico" if i == 0 else None,
        zorder=3,
    )
    plt.plot(fN2 * 100, y_stat_min, lw=0.6, ls="--", color=color, zorder=4)
    plt.plot(fN2 * 100, y_stat_max, lw=0.6, ls="--", color=color, zorder=4)
    plt.plot(fN2 * 100, y0, lw=2, color=color, label=f"{pressure} bar", zorder=5)

    x_exp, y_exp, yerr_exp = build_total_ir_experimental(exp_nom, pressure, norm_uv)
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
    )

plt.xscale("log")
plt.grid(True, which="major", alpha=0.3)
plt.grid(True, which="minor", alpha=0.08)
plt.xlabel("N$_2$ concentration [$\\%$]")
plt.ylabel("ph/MeV")
plt.title("Primary Ar-N$_2$ IR (680--785 nm) yield with bands")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArN2_IR_total_bands.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("STAT toy failures IR:", n_stat_failed, "/", N_STAT_TOYS)
print("Banda estadística IR calculada con pseudo-experimentos independientes; no se ha ensanchado artificialmente.")
