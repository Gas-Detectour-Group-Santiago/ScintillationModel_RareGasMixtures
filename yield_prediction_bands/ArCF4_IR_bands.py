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

from ArCF4_infrarred import *
from read_experimental import read_experimental
from fiting import fitParameters


# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización, según la columna Ar/CF4
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

energy_X_ray_CF4 = 15


def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W


# =========================================================
# HELPERS
# =========================================================

def apply_global_threshold(df, conc_col="fCF4", is_727=False, force_error=None):
    """
    Aplica el corte en concentración y la máscara de threshold.

    Si force_error is None:
        conserva los errores originales del dataframe.

    Si force_error tiene un valor, por ejemplo 0.0009:
        fuerza todas las incertidumbres válidas a ese valor.

    Esto permite separar:
        - errores usados para el ajuste
        - errores sistemáticos usados para desplazar
        - errores "all" usados para representar
    """

    bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
    err_cols = [f"Err {c}" for c in bar_cols]

    # 1) Región de referencia
    df_ref_10  = df[df[conc_col] == 10].copy()
    df_ref_20  = df[df[conc_col] == 20].copy()
    df_ref_50  = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    # 2) Threshold global del dataframe
    threshold_10 = df_ref_10[bar_cols].max().max()
    threshold_20 = df_ref_20[bar_cols].max().max()
    threshold_50 = df_ref_50[bar_cols].max().max()
    threshold_100 = df_ref_100[bar_cols].max().max()

    threshold = 0
    # threshold = min(threshold_20, threshold_50, threshold_100)

    # 3) Región de baja concentración usada en el ajuste
    df_low = df[df[conc_col] < 11].copy()

    if is_727:
        df_low = df[df[conc_col] < 6].copy()

    # 4) Máscara celda a celda
    mask = df_low[bar_cols] >= threshold

    # 5) Aplicar máscara a los yields
    df_low[bar_cols] = df_low[bar_cols].where(mask)

    # 6) Aplicar máscara a los errores
    for bar, err in zip(bar_cols, err_cols):

        # Solo fuerzo el error cuando lo pido explícitamente.
        # Para systematic y all de representación se conservan los errores reales.
        if force_error is not None:
            df_low[err] = force_error

        df_low[err] = df_low[err].where(mask[bar])

    df_low["fCF4"] *= 1

    return df_low, threshold


def build_shifted_dict(exp_dict, bar_cols, err_cols, fit_error=None):
    """
    Construye dos diccionarios:
        - exp_low: yields desplazados como y - e
        - exp_up:  yields desplazados como y + e

    Los desplazamientos se hacen usando las incertidumbres presentes
    en exp_dict. Por tanto, si exp_dict viene de uncertainty_mode="systematic",
    los shifts son sistemáticos.

    Si fit_error no es None, después del shift se fuerzan los errores usados
    para el ajuste a ese valor. En tu caso, fit_error=0.0009.
    """

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

        # Importante:
        # El desplazamiento ya se hizo con la incertidumbre sistemática.
        # Pero el refit se pesa con la incertidumbre fija 0.0009.
        if fit_error is not None:
            for j, err in enumerate(err_cols):
                df_low.loc[:, err] = np.where(mask[:, j], fit_error, np.nan)
                df_up.loc[:, err]  = np.where(mask[:, j], fit_error, np.nan)

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


def total_ir_model(par, degrad_data, fN2, pressure, normCF4):
    total = (
        theory_yield_ArCF4_Ir_696(par, degrad_data, fN2, pressure)
        + theory_yield_ArCF4_Ir_727(par, degrad_data, fN2, pressure)
        + theory_yield_ArCF4_Ir_750(par, degrad_data, fN2, pressure)
        + theory_yield_ArCF4_Ir_763(par, degrad_data, fN2, pressure)
        + theory_yield_ArCF4_Ir_772(par, degrad_data, fN2, pressure)
        + theory_yield_ArCF4_Ir_794(par, degrad_data, fN2, pressure)
    )

    factor = 1000 / normCF4
    return total * factor


def build_total_ir_experimental(exp_dict, pressure, normCF4):
    """
    Construye los puntos experimentales totales IR:

        696 + 727 + 750 + 763 + 772 + 794

    para una presión dada, escalados a ph/MeV.

    Solo conserva concentraciones comunes a todas las líneas.

    El error total se calcula sumando en cuadratura los errores de cada línea.
    Por tanto, si exp_dict viene de uncertainty_mode="all", las barras del plot
    serán las incertidumbres totales.
    """

    col = f"{pressure:.1f}bar"
    err_col = f"Err {pressure:.1f}bar"
    conc_col = "fCF4"
    lines = ["696", "727", "750", "763", "772", "794"]

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
                how="inner"
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

    factor_pts = 1000 / normCF4

    y_total_scaled = y_total * factor_pts
    err_total_scaled = err_total * factor_pts

    return x_percent, y_total_scaled, err_total_scaled


def make_experimental_data(exp_dict):
    """
    Convierte el diccionario de dataframes en el formato usado por fitParameters,
    rellenando los NaN con 0 como en tu script original.
    """

    return {
        "696": exp_dict["696"].fillna(0),
        "727": exp_dict["727"].fillna(0),
        "750": exp_dict["750"].fillna(0),
        "763": exp_dict["763"].fillna(0),
        "772": exp_dict["772"].fillna(0),
        "794": exp_dict["794"].fillna(0),
    }


# =========================================================
# CONFIG
# =========================================================

DATA_DIR_EXP    = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR    = "../data/Parameters"

archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"

yields = ["696", "727", "750", "763", "772", "794"]
presiones = [1, 2, 3, 4, 5]
concentraciones_reales = None
output_dir = "../data/Experimental/ArCF4/"

bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
err_cols = [f"Err {c}" for c in bar_cols]

fixed_idx = [1, 5, 9, 13, 17]
fixed_error = 0.1

FIT_ERROR = 0.0009

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.15, 0.85, len(presiones)))


# =========================================================
# CARGA DE DATOS
# =========================================================

degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4_IR.csv"))

parameter_data_ArCF4 = pd.read_csv(
    os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv")
)["parameter"].to_numpy()

normCF4 = parameter_data_ArCF4[0].copy()


def load_ir_experimental(uncertainty_mode="all", force_error=None):
    """
    Carga los datos IR para Ar/CF4.

    uncertainty_mode controla lo que devuelve read_experimental:

        uncertainty_mode="all"
            errores totales

        uncertainty_mode="systematic"
            errores sistemáticos

    force_error controla qué errores quedan finalmente en los dataframes:

        force_error=None
            conserva los errores dados por read_experimental

        force_error=0.0009
            fuerza todos los errores válidos a 0.0009

    Por tanto:

        load_ir_experimental("all", force_error=0.0009)
            -> para ajuste nominal

        load_ir_experimental("systematic", force_error=None)
            -> para construir shifts sistemáticos

        load_ir_experimental("all", force_error=None)
            -> para dibujar barras de error totales
    """

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
    df_794 = pd.read_csv(os.path.join(DATA_DIR_EXP, "794.csv"))

    w_cf4 = W_CF4(df_696["fCF4"].to_numpy() / 100)

    y_cols = [
        "1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar",
        "Err 1.0bar", "Err 2.0bar", "Err 3.0bar",
        "Err 4.0bar", "Err 5.0bar"
    ]

    factor = (1 / w_cf4)[:, None]

    df_696[y_cols] = df_696[y_cols].to_numpy() * factor
    df_727[y_cols] = df_727[y_cols].to_numpy() * factor
    df_750[y_cols] = df_750[y_cols].to_numpy() * factor
    df_763[y_cols] = df_763[y_cols].to_numpy() * factor
    df_772[y_cols] = df_772[y_cols].to_numpy() * factor
    df_794[y_cols] = df_794[y_cols].to_numpy() * factor

    df_696, _ = apply_global_threshold(df_696, force_error=force_error)
    df_727, _ = apply_global_threshold(df_727, force_error=force_error)
    df_750, _ = apply_global_threshold(df_750, force_error=force_error)
    df_763, _ = apply_global_threshold(df_763, force_error=force_error)
    df_772, _ = apply_global_threshold(df_772, force_error=force_error)
    df_794, _ = apply_global_threshold(df_794, force_error=force_error)

    return {
        "696": df_696,
        "727": df_727,
        "750": df_750,
        "763": df_763,
        "772": df_772,
        "794": df_794,
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
    0.0, 29.3, 0.0, 0.0,
], dtype=float)

lower_semifixed = x0_semifixed * 0.999999999999999
upper_semifixed = x0_semifixed * 1.000000000000001

lower = np.array([
    0.0, 0.0, 0.0, 0.0,
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
    0.25, 0.0, 1.0, 1.0,
], dtype=float) + x0_semifixed

upper = np.array([
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
    0.25, 0.0, 1000.0, 1000.0,
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


# =========================================================
# AJUSTE NOMINAL
# =========================================================

# Valores centrales de "all", pero errores usados en el ajuste = 0.0009
exp_fit = load_ir_experimental(
    uncertainty_mode="all",
    force_error=FIT_ERROR
)

experimental_data = make_experimental_data(exp_fit)

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

# Índices libres. Mantengo tu lista explícita.
free_idx = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]

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

# Estos datos conservan las incertidumbres sistemáticas reales.
# Se usarán solamente para construir y +/- sigma_syst.
exp_sys = load_ir_experimental(
    uncertainty_mode="systematic",
    force_error=None
)

# Construimos shifts usando la incertidumbre sistemática,
# pero el refit de cada shift se pesa con error fijo 0.0009.
experimental_data_low, experimental_data_up = build_shifted_dict(
    exp_sys,
    bar_cols,
    err_cols,
    fit_error=FIT_ERROR
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
# DATOS PARA REPRESENTACIÓN
# =========================================================

# Estos datos son exclusivamente para dibujar los puntos experimentales.
# Conservan las incertidumbres "all" reales.
exp_plot = load_ir_experimental(
    uncertainty_mode="all",
    force_error=None
)


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
        return total_ir_model(par, degrad_data, fN2, pressure, normCF4)

    # Banda estadística: propagación de pcov del ajuste nominal.
    # Ojo: esta pcov viene de un ajuste ponderado con error fijo 0.0009.
    y0, y_low_stat, y_up_stat = statistical_band(
        model_total,
        par_natural,
        cov_full
    )

    # Banda sistemática: diferencia entre refits con datasets desplazados
    # por la incertidumbre sistemática real.
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

    # Puntos experimentales con incertidumbre "all" real.
    x_exp, y_exp, yerr_exp = build_total_ir_experimental(
        exp_plot,
        pressure,
        normCF4
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

plt.xlabel("CF$_4$ concentration [\%]")
plt.ylabel("ph/MeV")
plt.title("Primary ArCF$_4$ IR (680-785 nm) yield with bands")
plt.legend()
plt.tight_layout()

plt.savefig("plots/ArCF4_IR_total_bands.pdf", dpi=300, bbox_inches="tight")
plt.show()