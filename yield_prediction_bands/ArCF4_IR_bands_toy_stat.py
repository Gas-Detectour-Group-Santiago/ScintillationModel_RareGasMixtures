import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

from ArCF4_infrarred import *
from read_experimental import read_experimental
from fiting import fitParameters
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure


# =========================================================
# CONFIG GENERAL
# =========================================================
DATA_DIR_EXP = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"
yields = ["696", "727", "750", "763", "772", "794"]
concentraciones_reales = None
output_dir = "../data/Experimental/ArCF4/"

# El ajuste usa solo 1, 2 y 3 bar.
FIT_PRESSURES = [1, 2, 3]

# La representación muestra también 4 y 5 bar.
PLOT_PRESSURES = [1, 2, 3, 4, 5]

FIT_BAR_COLS = [f"{p:.1f}bar" for p in FIT_PRESSURES]
FIT_ERR_COLS = [f"Err {c}" for c in FIT_BAR_COLS]

PLOT_BAR_COLS = [f"{p:.1f}bar" for p in PLOT_PRESSURES]
PLOT_ERR_COLS = [f"Err {c}" for c in PLOT_BAR_COLS]

TAU_FIXED_ERROR = 0.1

# Statistical band: independent pseudo-experiments.
# This avoids the huge, ill-conditioned covariance bands and avoids coherent
# +/- shifts, which are systematic-like rather than statistical.
N_STAT_TOYS = 120
STAT_PERCENTILES = (16.0, 84.0)
RANDOM_SEED = 314159

# Índices de tau: segundo parámetro de cada línea.
FIXED_IDX = [1, 5, 9, 13, 17, 21]

os.makedirs("plots/ArCF4_IR", exist_ok=True)
os.makedirs("tex_param", exist_ok=True)
os.makedirs(DATA_DIR_PAR, exist_ok=True)


# =========================================================
# W-VALUE Ar/CF4
# =========================================================
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])


def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    return np.interp(f_cf4, cf4_pct, ion_pot)


# =========================================================
# HELPERS
# =========================================================
def pressure_cols(pressures):
    bars = [f"{p:.1f}bar" for p in pressures]
    errs = [f"Err {c}" for c in bars]
    return bars, errs


def apply_global_threshold(
    df,
    conc_col="fCF4",
    is_727=False,
    force_error=None,
    pressures=None,
    threshold=0.0,
):
    """
    Aplica el corte de concentración y conserva solo las columnas de presión indicadas.

    - Para el fit se llama con pressures=[1, 2, 3].
    - Para representar se llama con pressures=[1, 2, 3, 4, 5].

    Así 4 y 5 bar no entran nunca en fitParameters, pero sí pueden dibujarse.
    Si force_error=None, se conservan los errores reales leídos del archivo experimental.
    """
    if pressures is None:
        pressures = FIT_PRESSURES

    bar_cols, err_cols = pressure_cols(pressures)

    # Región de baja concentración usada en el fit/plot.
    df_low = df[df[conc_col] < 11].copy()
    if is_727:
        df_low = df[df[conc_col] < 6].copy()

    mask = df_low[bar_cols] >= threshold

    df_low[bar_cols] = df_low[bar_cols].where(mask)

    for bar, err in zip(bar_cols, err_cols):
        if force_error is not None:
            df_low[err] = force_error
        df_low[err] = df_low[err].where(mask[bar])

    # Devuelve solo concentración + presiones seleccionadas.
    return df_low[[conc_col] + bar_cols + err_cols], threshold


def read_and_scale_ir_data(uncertainty_mode="all"):
    """Lee los CSV generados por read_experimental y normaliza yields y errores por 1/W."""
    read_experimental(
        archivo_entrada,
        yields,
        PLOT_PRESSURES,
        output_dir,
        concentraciones_reales=concentraciones_reales,
        uncertainty_mode=uncertainty_mode,
        yield_mode="ir",
    )

    exp = {
        line: pd.read_csv(os.path.join(DATA_DIR_EXP, f"{line}.csv"))
        for line in yields
    }

    w_cf4 = W_CF4(exp["696"]["fCF4"].to_numpy(dtype=float) / 100.0)
    factor = (1.0 / w_cf4)[:, None]

    y_cols_full = PLOT_BAR_COLS + PLOT_ERR_COLS

    for line in yields:
        exp[line][y_cols_full] = exp[line][y_cols_full].to_numpy(dtype=float) * factor

    return exp

def load_ir_experimental(uncertainty_mode="all", force_error=None, pressures=None):
    """
    Devuelve el diccionario de dataframes listo para fit o plot.

    pressures=[1,2,3]     -> dataset de ajuste.
    pressures=[1,2,3,4,5] -> dataset de representación.
    """
    if pressures is None:
        pressures = FIT_PRESSURES

    exp_raw = read_and_scale_ir_data(uncertainty_mode=uncertainty_mode)

    exp_out = {}
    for line in yields:
        df_line, _ = apply_global_threshold(
            exp_raw[line],
            force_error=force_error,
            pressures=pressures,
            is_727=(line == "727"),
        )

        # Equivalente a yield_XXX_ir_n.loc[0, "fCF4"] = 0.001
        df_line.loc[df_line.index[0], "fCF4"] = 0.001

        exp_out[line] = df_line

    return exp_out


def make_experimental_data(exp_dict):
    return {line: exp_dict[line].fillna(0) for line in yields}


def build_systematic_shifted_dict(exp_central, exp_sys, bar_cols, err_cols):
    """
    Construye datasets desplazados para estimar la banda sistemática.

    - exp_central contiene los valores centrales y los errores reales usados en el ajuste
      nominal. En este script se carga con uncertainty_mode="stadistic".
    - exp_sys contiene las incertidumbres sistemáticas reales. Se usa solo para desplazar
      los puntos como y - sigma_sys e y + sigma_sys.

    Los errores que quedan en los datasets low/up son los errores reales de exp_central.
    Es decir: no se fuerza ningún 0.0009/0.009 ni ningún error artificial.
    """
    exp_low = {}
    exp_up = {}

    for key in exp_central:
        df_c = exp_central[key].copy(deep=True)
        df_s = exp_sys[key].copy(deep=True)

        df_low = df_c.copy(deep=True)
        df_up = df_c.copy(deep=True)

        y = df_c[bar_cols].to_numpy(dtype=float)
        e_sys = df_s[err_cols].to_numpy(dtype=float)
        e_fit = df_c[err_cols].to_numpy(dtype=float)

        mask = ~np.isnan(y)

        df_low.loc[:, bar_cols] = np.where(mask, y - e_sys, np.nan)
        df_up.loc[:, bar_cols] = np.where(mask, y + e_sys, np.nan)

        # Mantener los errores reales del ajuste nominal para ponderar los refits.
        df_low.loc[:, err_cols] = np.where(mask, e_fit, np.nan)
        df_up.loc[:, err_cols] = np.where(mask, e_fit, np.nan)

        exp_low[key] = df_low.fillna(0)
        exp_up[key] = df_up.fillna(0)

    return exp_low, exp_up



def build_statistical_toy_dict(exp_central, bar_cols, err_cols, rng):
    """
    Builds one statistical pseudo-experiment.

    Each point is fluctuated independently by its statistical uncertainty.  This
    is the correct statistical analogue of the systematic low/up refits above;
    shifting every point coherently would be a systematic variation, not a
    statistical one.
    """
    toy = {}

    for key, df in exp_central.items():
        df_toy = df.copy(deep=True)
        y = df_toy[bar_cols].to_numpy(dtype=float)
        e = df_toy[err_cols].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(e) & (e > 0.0)
        fluct = rng.normal(0.0, 1.0, size=y.shape) * e
        df_toy.loc[:, bar_cols] = np.where(mask, y + fluct, y)
        toy[key] = df_toy.fillna(0)

    return toy


def fit_statistical_toys(exp_central, nominal_parameters):
    rng = np.random.default_rng(RANDOM_SEED)
    pars = []
    failures = 0

    for _ in range(N_STAT_TOYS):
        toy_data = build_statistical_toy_dict(exp_central, FIT_BAR_COLS, FIT_ERR_COLS, rng)
        try:
            toy_fit = fitParameters(
                equations,
                make_experimental_data(toy_data),
                degrad_data,
                x0=nominal_parameters,
                bounds=bounds,
                is_infrared=True,
                fixed_idx=FIXED_IDX,
                fixed_error=TAU_FIXED_ERROR,
            )
            pars.append(toy_fit.x.copy())
        except Exception:
            failures += 1

    return np.asarray(pars, dtype=float), failures

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

        y_plus = model_func(par_natural + dp)
        y_minus = model_func(par_natural - dp)
        G[:, j] = (y_plus - y_minus) / (2.0 * h)

    var_y = np.einsum("ij,jk,ik->i", G, cov_full, G)
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))

    return y0, y0 - sigma_y, y0 + sigma_y


def total_ir_model(par, degrad_data, fCF4, pressure, normCF4):
    total = (
        theory_yield_ArCF4_Ir_696(par, degrad_data, fCF4, pressure)
        + theory_yield_ArCF4_Ir_727(par, degrad_data, fCF4, pressure)
        + theory_yield_ArCF4_Ir_750(par, degrad_data, fCF4, pressure)
        + theory_yield_ArCF4_Ir_763(par, degrad_data, fCF4, pressure)
        + theory_yield_ArCF4_Ir_772(par, degrad_data, fCF4, pressure)
        + theory_yield_ArCF4_Ir_794(par, degrad_data, fCF4, pressure)
    )

    return total * (1000.0 / normCF4)


def build_total_ir_experimental(exp_dict, pressure, normCF4):
    col = f"{pressure:.1f}bar"
    err_col = f"Err {pressure:.1f}bar"
    conc_col = "fCF4"

    merged = None

    for line in yields:
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

    for line in yields:
        y_total += merged[f"y_{line}"].to_numpy(dtype=float)
        err2_total += merged[f"e_{line}"].to_numpy(dtype=float) ** 2

    factor_pts = 1000.0 / normCF4
    return x_percent, y_total * factor_pts, np.sqrt(err2_total) * factor_pts


# =========================================================
# DATOS DEGRAD Y NORMALIZACIÓN GLOBAL
# =========================================================
degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4_IR.csv"))

parameter_data_ArCF4 = pd.read_csv(
    os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv")
)["parameter"].to_numpy()

normCF4 = parameter_data_ArCF4[0].copy()


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

# Vuelvo a la configuración del segundo script.
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


# =========================================================
# AJUSTE NOMINAL: SOLO 1, 2 Y 3 BAR, CON ERRORES REALES
# =========================================================
exp_fit = load_ir_experimental(
    uncertainty_mode="stadistic",
    force_error=None,
    pressures=FIT_PRESSURES,
)

experimental_data = make_experimental_data(exp_fit)

popt = fitParameters(
    equations,
    experimental_data,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=FIXED_IDX,
    fixed_error=TAU_FIXED_ERROR,
)

par_natural = popt.x.copy()
npar_full = len(par_natural)
free_idx = [i for i in range(npar_full) if i not in FIXED_IDX]
cov_full = build_full_covariance(popt.pcov, npar_full, free_idx)

# Keep this only for the correlation matrix diagnostic.
cov_theta = cov_full
chi2 = getattr(popt, "chi2", np.nan)
dof = getattr(popt, "dof", np.nan)
chi2_red = getattr(popt, "chi2_red", np.nan)

print("=" * 60)
print("Parámetros globales:\n", popt.x)
print(f"Grados de libertad: {dof}")
print(f"Chi2 (real): {chi2}")
print(f"Chi2 reducido: {chi2_red}")
print("=" * 60)


# =========================================================
# REFITS SISTEMÁTICOS: SOLO 1, 2 Y 3 BAR, LOW/UP CON ERROR SISTEMÁTICO
# =========================================================
exp_sys_fit = load_ir_experimental(
    uncertainty_mode="sistematic",
    force_error=None,
    pressures=FIT_PRESSURES,
)

experimental_data_low, experimental_data_up = build_systematic_shifted_dict(
    exp_fit,
    exp_sys_fit,
    FIT_BAR_COLS,
    FIT_ERR_COLS,
)

popt_low = fitParameters(
    equations,
    experimental_data_low,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=FIXED_IDX,
    fixed_error=TAU_FIXED_ERROR,
)

popt_up = fitParameters(
    equations,
    experimental_data_up,
    degrad_data,
    x0=x0,
    bounds=bounds,
    is_infrared=True,
    fixed_idx=FIXED_IDX,
    fixed_error=TAU_FIXED_ERROR,
)

par_low = popt_low.x.copy()
par_up = popt_up.x.copy()


# =========================================================
# STATISTICAL TOYS: SOLO 1, 2 Y 3 BAR
# =========================================================
# The nominal fit already uses uncertainty_mode="stadistic", so exp_fit
# contains the central values and point-to-point statistical errors.
par_stat_toys, n_stat_failed = fit_statistical_toys(exp_fit, par_natural)


# =========================================================
# DATOS PARA REPRESENTAR: 1, 2, 3, 4 Y 5 BAR
# =========================================================
exp_plot = load_ir_experimental(
    uncertainty_mode="all",
    force_error=None,
    pressures=PLOT_PRESSURES,
)


# =========================================================
# PLOTS INDIVIDUALES POR LÍNEA: MODELO 1-5 BAR, FIT 1-3 BAR
# =========================================================
for name in equations:
    concentrations = np.logspace(-6, 0, 1000)

    plot_fit_vs_experiment_by_pressure(
        df_exp=exp_plot[name],
        theory_func=equations[name],
        fit_params=popt.x,
        degrad_data=degrad_data,
        concentration_grid=concentrations,
        pressures=PLOT_PRESSURES,
        x_col="fCF4",
        x_plot_factor=100,
        min_positive_x=1e-3,
        title=f"Primary ArCF$_4$ IR ({name} nm) Yield fit",
        xlabel="Concentration of CF$_4$ [$\\%$]",
        ylabel="Normalized Yield",
        xlim=(0.001 * 0.9, 100 * 1.1),
        ylim=(0.000001, 0.005),
        xscale="log",
        yscale="log",
        cmap="viridis",
        darken_factor=-0.15,
        legend=True,
        legend_kwargs={"ncol": 2, "fontsize": 9},
        output=f"plots/ArCF4_IR/ArCF4_global_{name}_fit123_plot12345.pdf",
        show=False,
        activate_components=False,
    )


# =========================================================
# PLOT TOTAL IR CON BANDAS: REPRESENTA 1, 2, 3, 4 Y 5 BAR
# =========================================================
fCF4_grid = np.logspace(-5, 0, 1000)

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.15, 0.85, len(PLOT_PRESSURES)))

plt.figure(figsize=(6.2, 4.2))

for i, pressure in enumerate(PLOT_PRESSURES):
    color = colors[i]

    def model_total(par):
        return total_ir_model(par, degrad_data, fCF4_grid, pressure, normCF4)

    y0 = model_total(par_natural)

    if len(par_stat_toys) > 0:
        stat_curves = np.asarray([model_total(par) for par in par_stat_toys], dtype=float)
        y_low_stat, y_up_stat = np.nanpercentile(stat_curves, STAT_PERCENTILES, axis=0)
    else:
        y_low_stat = y0.copy()
        y_up_stat = y0.copy()

    y_low_sys = model_total(par_low)
    y_up_sys = model_total(par_up)

    ymin = np.minimum.reduce([y0, y_low_sys, y_up_sys])
    ymax = np.maximum.reduce([y0, y_low_sys, y_up_sys])

    plt.fill_between(
        fCF4_grid * 100,
        ymin,
        ymax,
        alpha=0.28,
        color=color,
        label="Sistemático" if i == 0 else None,
    )

    plt.fill_between(
        fCF4_grid * 100,
        y_low_stat,
        y_up_stat,
        alpha=0.20,
        color=color,
        label="Estadístico" if i == 0 else None,
        zorder=3,
    )
    plt.plot(fCF4_grid * 100, y_low_stat, lw=0.6, ls="--", color=color, zorder=4)
    plt.plot(fCF4_grid * 100, y_up_stat, lw=0.6, ls="--", color=color, zorder=4)

    print(
        f"{pressure:g} bar | STAT width min/max = "
        f"{np.nanmin(y_up_stat - y_low_stat):.4g} / {np.nanmax(y_up_stat - y_low_stat):.4g}; "
        f"SYS width min/max = {np.nanmin(ymax - ymin):.4g} / {np.nanmax(ymax - ymin):.4g}"
    )

    plt.plot(
        fCF4_grid * 100,
        y0,
        lw=2,
        color=color,
        label=f"{pressure} bar",
    )

    x_exp, y_exp, yerr_exp = build_total_ir_experimental(
        exp_plot,
        pressure,
        normCF4,
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
    )

plt.xscale("log")
# plt.grid(True, which="major", alpha=0.3)
# plt.grid(True, which="minor", alpha=0.08)
plt.grid(False)
plt.xlabel("CF$_4$ concentration [$\\%$]")
plt.ylabel("ph/MeV")
plt.title("Primary ArCF$_4$ IR (680-785 nm) yield with bands")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_IR_total_bands.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("STAT toy failures ArCF4 IR:", n_stat_failed, "/", N_STAT_TOYS)


# =========================================================
# EXPORT CSV / LATEX
# =========================================================
names_tex = [
    "$P_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 696 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 696 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 727 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 727 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 750 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 750 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 764 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 764 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 772 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 772 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 794 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 794 \\mathrm{nm}}$",
]

names_csv = [
    "PAr_star_696", "tau_Ar_696", "K_Ar_Q_Ar_696", "K_Ar_Q_CF4_696",
    "PAr_star_727", "tau_Ar_727", "K_Ar_Q_Ar_727", "K_Ar_Q_CF4_727",
    "PAr_star_750", "tau_Ar_750", "K_Ar_Q_Ar_750", "K_Ar_Q_CF4_750",
    "PAr_star_764", "tau_Ar_764", "K_Ar_Q_Ar_764", "K_Ar_Q_CF4_764",
    "PAr_star_772", "tau_Ar_772", "K_Ar_Q_Ar_772", "K_Ar_Q_CF4_772",
    "PAr_star_794", "tau_Ar_794", "K_Ar_Q_Ar_794", "K_Ar_Q_CF4_794",
]

export_to_csv("../data/Parameters/ArCF4_IR_primary.csv", popt, names_csv)

export_fit_table_latex(
    result=popt,
    names=names_tex,
    filename="tex_param/ArCF4_IR_param.tex",
    caption="Parámetros obtenidos del ajuste del centelleo IR en Ar--CF$_4$.",
    label="tab:ArCF4_IR_fit",
    sigfigs=4,
)


# =========================================================
# MATRIZ DE CORRELACIÓN
# =========================================================
diag = np.sqrt(np.diag(cov_theta))
outer = np.outer(diag, diag)

with np.errstate(divide="ignore", invalid="ignore"):
    corr = cov_theta / outer

corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
corr = np.clip(corr, -1, 1)

corr_df = pd.DataFrame(corr, columns=names_tex, index=names_tex)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Correlación"},
)
plt.title("Matriz de Correlación de Parámetros Ajustados", fontsize=14)
plt.tight_layout()
plt.savefig("plots/ArCF4_IR/ArCF4_IR_CorrelationMatrix_GlobalFit.pdf", dpi=300)
