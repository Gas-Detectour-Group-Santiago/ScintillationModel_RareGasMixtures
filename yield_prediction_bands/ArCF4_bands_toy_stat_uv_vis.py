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

from ArCF4 import *
from read_experimental import read_experimental
from fiting import fitParameters

# =========================================================
# CONFIG
# =========================================================
DATA_DIR_EXP = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"
yields = ["vis", "UV"]
presiones = [1, 2, 2.5, 3, 4, 5]
output_dir = "../data/Experimental/ArCF4/"

N_STAT_TOYS = 120
STAT_PERCENTILES = [16, 84]
RNG_SEED = 12345

os.makedirs("plots", exist_ok=True)

# =========================================================
# DATA / MODEL
# =========================================================
degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4.csv"))

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, 10))

# % de CF4 en Ar
cf4_pct_w = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])


def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    return np.interp(f_cf4, cf4_pct_w, ion_pot)


# =========================================================
# FIT CONFIGURATION
# Same physics/bounds as the uploaded ArCF4_bands(1).py,
# but the statistical band is now obtained with toys.
# =========================================================
x0 = np.array([
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
fixed_idx = [6, 8, 10]
fixed_values = [0.065, 50.05, 0.0001]
fixed_error = 0.01

equations = {
    "vis": theory_yield_vis,
    "uv": theory_yield_uv,
}


# =========================================================
# HELPERS
# =========================================================
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


def call_read_experimental(uncertainty_mode):
    """Keep compatibility with both project spellings."""
    modes_to_try = [uncertainty_mode]
    aliases = {
        "systematic": ["sistematic"],
        "sistematic": ["systematic"],
        "statistic": ["stadistic", "statistical"],
        "stadistic": ["statistic", "statistical"],
        "statistical": ["stadistic", "statistic"],
    }
    modes_to_try.extend(aliases.get(uncertainty_mode, []))

    last_error = None
    for mode in modes_to_try:
        try:
            read_experimental(
                archivo_entrada,
                yields,
                presiones,
                output_dir,
                uncertainty_mode=mode,
            )
            return
        except Exception as exc:
            last_error = exc
    raise last_error


def load_experimental(uncertainty_mode="all", w_scaled=True):
    call_read_experimental(uncertainty_mode)

    df_uv = pd.read_csv(os.path.join(DATA_DIR_EXP, "UV.csv"))
    df_vis = pd.read_csv(os.path.join(DATA_DIR_EXP, "vis.csv"))

    # Keep the same low-concentration UV convention as your original script.
    df_uv.loc[0, "fCF4"] = 0.001
    df_vis = df_vis.fillna(0)

    if w_scaled:
        for df in (df_uv, df_vis):
            y_cols, err_cols = pressure_columns(df)
            w = W_CF4(df["fCF4"].to_numpy(dtype=float) / 100.0)
            factor = (1.0 / w)[:, None]
            df.loc[:, y_cols + err_cols] = df[y_cols + err_cols].to_numpy(dtype=float) * factor

    return {"uv": df_uv, "vis": df_vis}


def make_experimental_data(exp_dict):
    return {
        "vis": exp_dict["vis"].fillna(0),
        "uv": exp_dict["uv"].fillna(0),
    }


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
    )


def build_shifted_dict(exp_dict, sign=+1):
    """Coherent up/down shifts. Used for the systematic envelope."""
    out = {}
    for key, df in exp_dict.items():
        shifted = df.copy(deep=True)
        y_cols, err_cols = pressure_columns(shifted)
        y = shifted[y_cols].to_numpy(dtype=float)
        e = shifted[err_cols].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        shifted.loc[:, y_cols] = np.where(mask, y + sign * e, np.nan)
        out[key] = shifted.fillna(0)
    return out


def build_statistical_toy_dict(exp_stat, rng):
    """Independent point-by-point statistical fluctuations."""
    out = {}
    for key, df in exp_stat.items():
        toy = df.copy(deep=True)
        y_cols, err_cols = pressure_columns(toy)
        y = toy[y_cols].to_numpy(dtype=float)
        e = toy[err_cols].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        fluct = rng.normal(loc=0.0, scale=np.nan_to_num(e, nan=0.0))
        y_toy = np.where(mask, y + fluct, np.nan)
        # Photon yields should not become negative in the refit dataset.
        y_toy = np.where(mask, np.maximum(y_toy, 0.0), np.nan)
        toy.loc[:, y_cols] = y_toy
        out[key] = toy.fillna(0)
    return out


def fit_statistical_toys(exp_stat, nominal_parameters):
    rng = np.random.default_rng(RNG_SEED)
    toy_parameters = []
    n_failed = 0

    for itoy in range(N_STAT_TOYS):
        toy_data = build_statistical_toy_dict(exp_stat, rng)
        try:
            toy_fit = run_fit(toy_data, nominal_parameters)
            if np.all(np.isfinite(toy_fit.x)):
                toy_parameters.append(toy_fit.x.copy())
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if len(toy_parameters) == 0:
        raise RuntimeError("All statistical toys failed; no statistical band can be built.")

    return np.asarray(toy_parameters, dtype=float), n_failed


def percentile_band_from_toys(toy_parameters, model_func):
    curves = np.asarray([model_func(par) for par in toy_parameters], dtype=float)
    y_low, y_up = np.nanpercentile(curves, STAT_PERCENTILES, axis=0)
    return y_low, y_up


def envelope_from_nominal_up_down(y0, y_low, y_up):
    ymin = np.minimum.reduce([y0, y_low, y_up])
    ymax = np.maximum.reduce([y0, y_low, y_up])
    return ymin, ymax


def print_band_width(label, y_low_stat, y_up_stat, y_sys_min, y_sys_max, scale=1.0):
    stat_width = (np.asarray(y_up_stat) - np.asarray(y_low_stat)) * scale
    sys_width = (np.asarray(y_sys_max) - np.asarray(y_sys_min)) * scale
    print(
        f"{label} | STAT width min/max = "
        f"{np.nanmin(stat_width):.6g} / {np.nanmax(stat_width):.6g}; "
        f"SYS width min/max = {np.nanmin(sys_width):.6g} / {np.nanmax(sys_width):.6g}"
    )


# =========================================================
# NOMINAL FIT
# =========================================================
exp_nominal = load_experimental("all", w_scaled=True)
popt = run_fit(exp_nominal, x0)
par_natural = popt.x.copy()

chi2 = 2.0 * popt.cost if hasattr(popt, "cost") else getattr(popt, "chi2", np.nan)
dof = len(popt.fun) - len(par_natural) if hasattr(popt, "fun") else getattr(popt, "dof", np.nan)
chi2_red = chi2 / dof if np.isfinite(chi2) and np.isfinite(dof) and dof != 0 else np.nan

print("=" * 60)
print("Parámetros globales:", par_natural)
print(f"Chi2 (real): {chi2}")
print(f"Grados de libertad: {dof}")
print(f"Chi2 reducido: {chi2_red}")
print("=" * 60)

# =========================================================
# SYSTEMATIC REFITS: coherent +/- systematic shifts
# =========================================================
exp_sys = load_experimental("systematic", w_scaled=True)
popt_low = run_fit(build_shifted_dict(exp_sys, sign=-1), par_natural)
popt_up = run_fit(build_shifted_dict(exp_sys, sign=+1), par_natural)
par_low = popt_low.x.copy()
par_up = popt_up.x.copy()

# =========================================================
# STATISTICAL TOYS: independent point-by-point fluctuations
# =========================================================
exp_stat = load_experimental("stadistic", w_scaled=True)
par_stat_toys, n_failed = fit_statistical_toys(exp_stat, par_natural)
print(f"Statistical toys accepted: {len(par_stat_toys)} / {N_STAT_TOYS}; failed: {n_failed}")

# =========================================================
# DATA FOR PLOTS
# =========================================================
exp_plot = load_experimental("all", w_scaled=True)
yield_uv = exp_plot["uv"]
yield_vis = exp_plot["vis"]

norm = par_natural[0]
scale_to_ph_mev = 1000.0 / norm

# =========================================================
# PLOTS: NORMALIZED UV/VIS BANDS
# =========================================================
# UV normalized
fCF4_uv_norm = np.logspace(-5, 0, 100)

def model_uv_norm(par):
    return theory_yield_uv(par, degrad_data, fCF4_uv_norm, 1)

y0_uv = model_uv_norm(par_natural)
y_low_sys_uv = model_uv_norm(par_low)
y_up_sys_uv = model_uv_norm(par_up)
y_sys_min_uv, y_sys_max_uv = envelope_from_nominal_up_down(y0_uv, y_low_sys_uv, y_up_sys_uv)
y_low_stat_uv, y_up_stat_uv = percentile_band_from_toys(par_stat_toys, model_uv_norm)
print_band_width("UV normalized", y_low_stat_uv, y_up_stat_uv, y_sys_min_uv, y_sys_max_uv)

plt.figure(figsize=(6, 4))
plt.fill_between(fCF4_uv_norm * 100, y_sys_min_uv, y_sys_max_uv, alpha=0.30, label="Sistemático")
plt.fill_between(fCF4_uv_norm * 100, y_low_stat_uv, y_up_stat_uv, alpha=0.30, label="Estadístico")
plt.plot(fCF4_uv_norm * 100, y0_uv, lw=2, label="Ajuste nominal")
plt.errorbar(yield_uv["fCF4"], yield_uv["1.0bar"], yerr=yield_uv["Err 1.0bar"], label="Data", fmt=".r")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("CF$_4$ concentration $\\%$")
plt.ylabel("Arb.")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/ArCF4_bands_uv_normalized_toy_stat.pdf", dpi=300, bbox_inches="tight")
plt.show()

# VIS normalized
fCF4_vis_norm = np.logspace(-3, 0, 100)

def model_vis_norm(par):
    return theory_yield_vis(par, degrad_data, fCF4_vis_norm, 1)

y0_vis = model_vis_norm(par_natural)
y_low_sys_vis = model_vis_norm(par_low)
y_up_sys_vis = model_vis_norm(par_up)
y_sys_min_vis, y_sys_max_vis = envelope_from_nominal_up_down(y0_vis, y_low_sys_vis, y_up_sys_vis)
y_low_stat_vis, y_up_stat_vis = percentile_band_from_toys(par_stat_toys, model_vis_norm)
print_band_width("VIS normalized", y_low_stat_vis, y_up_stat_vis, y_sys_min_vis, y_sys_max_vis)

plt.figure(figsize=(6, 4))
plt.fill_between(fCF4_vis_norm * 100, y_sys_min_vis, y_sys_max_vis, alpha=0.30, label="Sistemático")
plt.fill_between(fCF4_vis_norm * 100, y_low_stat_vis, y_up_stat_vis, alpha=0.30, label="Estadístico")
plt.plot(fCF4_vis_norm * 100, y0_vis, lw=2, label="Ajuste nominal")
plt.errorbar(yield_vis["fCF4"], yield_vis["1.0bar"], yerr=yield_vis["Err 1.0bar"], label="Data", fmt=".r")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("CF$_4$ concentration $\\%$")
plt.ylabel("Arb.")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/ArCF4_bands_vis_normalized_toy_stat.pdf", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# EXTERNAL DATA FOR PH/MEV PLOTS
# =========================================================
con_uv_cf4_morozov = [100.0]
y_uv_cf4_morozov = [2175]
y_err_uv_cf4_morozov = [2600 - 2175]

cf4_pct_lit = np.array([0.15, 0.35, 0.50, 1.00, 2.00, 6.00, 11.00])
uv_E100 = np.array([358.9, 350.8, 292.2, 209.4, 227.8, 245.5, 263.4])
uv_E100_err = np.array([3.0, 3.9, 0.2, 2.0, 10.9, 2.4, 4.3])

Ar_third_continuum_ph_MeV = np.array([2.7e3])
Ar_third_continuum_err_ph_MeV = np.array([0.14e3])

cf4_red_E100 = [0.2, 0.4, 0.7, 1.0, 2.0, 7.0, 10.0]
y_red_E100 = [450, 500, 600, 1150, 1300, 1850, 1900]
yerr_red_E100 = [60, 60, 60, 90, 100, 120, 120]

vis_cf4_red_E100 = [100.0]
vis_y_red_E100 = [1184.7]
vis_yerr_red_E100 = [47]

vis2_cf4_red_E100 = [100.0]
vis2_y_red_E100 = [695]
vis2_yerr_red_E100 = [827 - 695]

# =========================================================
# PH/MEV UV BAND
# =========================================================
fCF4_uv = np.logspace(-5, 0, 100)

def model_uv(par):
    return theory_yield_uv(par, degrad_data, fCF4_uv, 1)

y0_uv = model_uv(par_natural)
y_low_sys_uv = model_uv(par_low)
y_up_sys_uv = model_uv(par_up)
y_sys_min_uv, y_sys_max_uv = envelope_from_nominal_up_down(y0_uv, y_low_sys_uv, y_up_sys_uv)
y_low_stat_uv, y_up_stat_uv = percentile_band_from_toys(par_stat_toys, model_uv)
print_band_width("UV ph/MeV", y_low_stat_uv, y_up_stat_uv, y_sys_min_uv, y_sys_max_uv, scale_to_ph_mev)

plt.figure(figsize=(6, 4))
plt.fill_between(fCF4_uv * 100, y_sys_min_uv * scale_to_ph_mev, y_sys_max_uv * scale_to_ph_mev,
                 alpha=0.30, label="Sistemático", color=colors[2])
plt.fill_between(fCF4_uv * 100, y_low_stat_uv * scale_to_ph_mev, y_up_stat_uv * scale_to_ph_mev,
                 alpha=0.30, label="Estadístico", color=colors[0])
plt.plot(fCF4_uv * 100, y0_uv * scale_to_ph_mev, lw=2, label="Ajuste nominal", color=colors[2])
plt.errorbar(yield_uv["fCF4"], yield_uv["1.0bar"] * scale_to_ph_mev,
             yerr=yield_uv["Err 1.0bar"] * scale_to_ph_mev,
             marker="o", linestyle="none", label="X-ray (220-400 nm)", ms=4,
             color=colors[2], ecolor=colors[2], elinewidth=1, capsize=2)
plt.errorbar(con_uv_cf4_morozov, y_uv_cf4_morozov, yerr=y_err_uv_cf4_morozov,
             marker="x", linestyle="none", ms=5, color=colors[6], ecolor=colors[6],
             elinewidth=1, capsize=2, label="$\\alpha$'s Morozov")
plt.errorbar(0.001, Ar_third_continuum_ph_MeV, yerr=Ar_third_continuum_err_ph_MeV,
             marker="v", linestyle="none", ms=5, color=colors[7], ecolor=colors[7],
             elinewidth=1, capsize=2, label="$\\alpha$'s Santorelli (160-325 nm)")
plt.errorbar(cf4_pct_lit, uv_E100, yerr=uv_E100_err, ms=4, marker="o",
             linestyle="none", color=colors[5], ecolor=colors[5], elinewidth=1,
             capsize=2, label="$\\alpha$'s P. Amedo (250-400 nm)")
plt.xscale("log")
plt.yscale("log")
plt.grid(False)
plt.xlabel("CF$_4$ concentration $\\%$")
plt.title("Ar-CF$_4$ UV (220-400 nm)")
plt.ylabel("ph/MeV")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_uv_toy_stat.pdf", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# PH/MEV VISIBLE BAND
# =========================================================
fCF4_vis = np.logspace(-3, 0, 100)

def model_vis(par):
    return theory_yield_vis(par, degrad_data, fCF4_vis, 1)

y0_vis = model_vis(par_natural)
y_low_sys_vis = model_vis(par_low)
y_up_sys_vis = model_vis(par_up)
y_sys_min_vis, y_sys_max_vis = envelope_from_nominal_up_down(y0_vis, y_low_sys_vis, y_up_sys_vis)
y_low_stat_vis, y_up_stat_vis = percentile_band_from_toys(par_stat_toys, model_vis)
print_band_width("VIS ph/MeV", y_low_stat_vis, y_up_stat_vis, y_sys_min_vis, y_sys_max_vis, scale_to_ph_mev)

plt.figure(figsize=(6, 4))
plt.fill_between(fCF4_vis * 100, y_sys_min_vis * scale_to_ph_mev, y_sys_max_vis * scale_to_ph_mev,
                 alpha=0.30, label="Sistemático", color=colors[2])
plt.fill_between(fCF4_vis * 100, y_low_stat_vis * scale_to_ph_mev, y_up_stat_vis * scale_to_ph_mev,
                 alpha=0.30, label="Estadístico", color=colors[0])
plt.plot(fCF4_vis * 100, y0_vis * scale_to_ph_mev, lw=2, label="Ajuste nominal", color=colors[2])
plt.errorbar(yield_vis["fCF4"], yield_vis["1.0bar"] * scale_to_ph_mev,
             yerr=yield_vis["Err 1.0bar"] * scale_to_ph_mev,
             marker="o", linestyle="none", label="X-ray", ms=4,
             color=colors[2], ecolor=colors[2], elinewidth=1, capsize=2)
plt.errorbar(cf4_red_E100, y_red_E100, yerr=yerr_red_E100, ms=4, marker="o",
             linestyle="none", color=colors[5], ecolor=colors[5], elinewidth=1,
             capsize=2, label="$\\alpha$'s P. Amedo")
plt.errorbar(vis2_cf4_red_E100, vis2_y_red_E100, yerr=vis2_yerr_red_E100,
             marker="x", linestyle="none", ms=5, color=colors[6], ecolor=colors[6],
             elinewidth=1, capsize=2, label="$\\alpha$'s Morozov")
plt.errorbar(vis_cf4_red_E100, vis_y_red_E100, yerr=vis_yerr_red_E100,
             marker="v", linestyle="none", ms=5, color=colors[7], ecolor=colors[7],
             elinewidth=1, capsize=2, label="$\\alpha$'s Lehaut")
plt.xscale("log")
# plt.yscale("log")
plt.grid(False)
plt.title("Ar-CF$_4$ Visible (400-700 nm)")
plt.xlabel("CF$_4$ concentration $\\%$")
plt.ylabel("ph/MeV")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_vis_toy_stat.pdf", dpi=300, bbox_inches="tight")
plt.show()
