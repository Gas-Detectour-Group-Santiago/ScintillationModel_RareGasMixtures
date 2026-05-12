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

from ArN2 import *
from read_experimental import read_experimental
from fiting import fitParameters

# =========================================================
# CONFIG
# =========================================================
archivo_entrada = "../data/Experimental/ArN2/N2_primary_data_final.pkl"
output_dir_exp  = "../data/Experimental/ArN2/"
data_dir_degrad = "../data/Primary_DegradData"
data_dir_par = "../data/Parameters"

parameter_data_ArCF4 = pd.read_csv(os.path.join(data_dir_par, "ArCF4_primary.csv"))["parameter"].to_numpy()

plot_pressure = 1
yields = ["yield_N2"]
presiones = [1, 2, 2.5, 3, 4, 5]

# mismo esquema de color que en Ar-CF4
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, 10))

norm = parameter_data_ArCF4[0]

# =========================================================
# CARGA DE DEGRAD
# =========================================================
degrad_data = pd.read_csv(os.path.join(data_dir_degrad, "ArN2.csv"))

# =========================================================
# PARÁMETROS INICIALES
# =========================================================
to_m3 = 2.69 * 10**25 * 10**(-9)

tau_N2        = 1e2 / np.mean(np.array([2.6, 2.07, 3.3, 2.5, 2.74, 2.66]))
K_N2_Q_N2     = to_m3 * 1e-17 * np.mean(np.array([0.71, 1.12, 1, 1.4]))
K_N2_Q_Ar     = to_m3 * 1e-19 * np.mean(np.array([5.6, 8.6]))

K_ArMeta_Q_N2c = to_m3 * 1e-17 * np.mean(np.array([3.2, 3.0, 1.1]))
K_ArMeta_Q_N2b = to_m3 * 1e-17 * np.mean(np.array([0.16]))
K_ArMeta_Q_2Ar = 1e-9 * np.mean(np.array([7.93e6]))

K_ArRes_Q_N2c  = to_m3 * 1e-17 * np.mean(np.array([1.5, 3.6]))
K_ArRes_Q_N2b  = to_m3 * 1e-17 * np.mean(np.array([1.5, 0]))
K_ArRes_Q_2Ar  = 1e-9 * np.mean(np.array([9.24e5]))

x0_semifixed = np.array([
    0.0,
    0.0,
    tau_N2, K_N2_Q_N2, K_N2_Q_Ar,
    K_ArMeta_Q_N2c, K_ArMeta_Q_N2b, K_ArMeta_Q_2Ar,
    K_ArRes_Q_N2c, K_ArRes_Q_N2b, K_ArRes_Q_2Ar,
    0.0,0.0
], dtype=float)

lower_semifixed = x0_semifixed * 0.33
upper_semifixed = x0_semifixed * 3.0

lower_og = np.array([
    0.0,
    0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0,0.0
], dtype=float)

x0_og = np.array([
    0.25,
    0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0,0.0
], dtype=float)

upper_og = np.array([
    1.0,
    1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.00001,0.00001
], dtype=float)

x0 = x0_og + x0_semifixed
bounds = (list(lower_og + lower_semifixed), list(upper_og + upper_semifixed))

equations = {
    "vis": theory_yield_N2_uv
}

fixed_idx = [2,11,12]
fixed_error = [0.376]

# =========================================================
# HELPERS
# =========================================================
def load_experimental(uncertainty_mode="all"):
    read_experimental(
        archivo_entrada,
        yields,
        presiones,
        output_dir_exp,
        concentraciones_reales=None,
        uncertainty_mode=uncertainty_mode
    )
    return pd.read_csv(os.path.join(output_dir_exp, "yield_N2.csv"))


def statistical_band(model_func, par_natural, cov_theta, free_idx):
    y0 = model_func(par_natural)

    npts = len(y0)
    npar = len(par_natural)
    G = np.zeros((npts, npar))

    for j in range(npar):
        dp = np.zeros_like(par_natural)
        h = 1e-6 * max(abs(par_natural[j]), 1.0)
        dp[j] = h

        y_plus = model_func(par_natural + dp)
        y_minus = model_func(par_natural - dp)

        G[:, j] = (y_plus - y_minus) / (2 * h)

    G_red = G[:, free_idx]
    var_y = np.einsum("ij,jk,ik->i", G_red, cov_theta, G_red)
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))

    return y0, y0 - sigma_y, y0 + sigma_y


def envelope_from_nominal_up_down(y0, y_low, y_up):
    ymin = np.minimum.reduce([y0, y_low, y_up])
    ymax = np.maximum.reduce([y0, y_low, y_up])
    return ymin, ymax


# =========================================================
# AJUSTE NOMINAL
# =========================================================
yield_nominal = load_experimental(uncertainty_mode="all")

def W_N2(xN2, WAr=26.4, WN2=34.8):
    return 1.0 / ((1.0-xN2)/WAr + xN2/WN2)


w_n2 =  W_N2(yield_nominal["N2 concentration (%)"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar",  "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_n2)[:, None]

yield_nominal[y_cols]  = yield_nominal[y_cols].to_numpy() * factor


experimental_data = {"vis": yield_nominal}

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

N_res = popt.fun.size
N_par = popt.x.size
dof   = N_res - N_par
chi2  = 2 * popt.cost
chi2_red = chi2 / dof

print("=" * 60)
print("Parámetros globales:\n", popt.x)
print(f"Grados de libertad: {dof}")
print(f"Chi2 (real): {chi2}")
print(f"Chi2 reducido: {chi2_red}")
print("=" * 60)

# =========================================================
# COVARIANZA ESTADÍSTICA
# =========================================================
J = popt.jac
m, p = J.shape
s2 = 2 * popt.cost / (m - p)
cov_theta = s2 * np.linalg.inv(J.T @ J)

npar_full = len(par_natural)
free_idx = [i for i in range(npar_full) if i not in fixed_idx]

if len(free_idx) != cov_theta.shape[0]:
    raise ValueError(
        f"len(free_idx)={len(free_idx)} pero cov_theta.shape={cov_theta.shape}. "
        "Revisa cómo devuelve jac la función fitParameters."
    )

# =========================================================
# AJUSTES LOW / UP PARA ENVOLVENTE SISTEMÁTICA
# =========================================================
yield_sys = load_experimental(uncertainty_mode="systematic")

w_n2 =  W_N2(yield_sys["N2 concentration (%)"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar",  "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_n2)[:, None]
yield_sys[y_cols]  = yield_sys[y_cols].to_numpy() * factor


y_cols = ["1.0bar", "2.0bar", "2.5bar", "3.0bar", "4.0bar", "5.0bar"]
err_cols = ["Err 1.0bar", "Err 2.0bar", "Err 2.5bar", "Err 3.0bar", "Err 4.0bar", "Err 5.0bar"]

err_sys = yield_sys[err_cols].to_numpy()

yield_low = yield_sys.copy(deep=True)
yield_up  = yield_sys.copy(deep=True)

yield_low.loc[:, y_cols] = yield_low.loc[:, y_cols].to_numpy() - err_sys
yield_up.loc[:, y_cols]  = yield_up.loc[:, y_cols].to_numpy() + err_sys

experimental_data_low = {"vis": yield_low}
experimental_data_up  = {"vis": yield_up}

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
# DATOS EXPERIMENTALES PARA PLOT
# =========================================================
yield_plot = load_experimental(uncertainty_mode="all")

w_n2 =  W_N2(yield_plot["N2 concentration (%)"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar",  "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_n2)[:, None]
yield_plot[y_cols]  = yield_plot[y_cols].to_numpy() * factor

yield_plot.loc[yield_plot["N2 concentration (%)"] <= 0, "N2 concentration (%)"] = 1e-6

# =========================================================
# CURVAS Y BANDAS
# =========================================================
fN2 = np.logspace(-4, 0, 1000)

factor = 1000/norm
factor2= 1000/norm


def model_total(par):
    return theory_yield_N2_uv(par, degrad_data, fN2, plot_pressure)

y0, y_low_stat, y_up_stat = statistical_band(
    model_total, par_natural, cov_theta, free_idx
)

y_low_sys = model_total(par_low)
y_up_sys  = model_total(par_up)

# envolvente como en el esquema nuevo de Ar-CF4
y_sys_min, y_sys_max = envelope_from_nominal_up_down(y0, y_low_sys, y_up_sys)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(6.5, 4.5))

plt.fill_between(
    fN2 * 100,
    y_sys_min,
    y_sys_max,
    alpha=0.30,
    label="Sistemático",
    color=colors[2]
)

plt.fill_between(
    fN2 * 100,
    y_low_stat,
    y_up_stat,
    alpha=0.30,
    label="Estadístico",
    color=colors[0]
)

plt.plot(
    fN2 * 100,
    y0,
    lw=2,
    label="Ajuste nominal",
    color=colors[2]
)

plt.errorbar(
    yield_plot["N2 concentration (%)"],
    yield_plot["1.0bar"],
    yerr=yield_plot["Err 1.0bar"],
    marker="o",
    linestyle="none",
    ms=4,
    color=colors[2],
    ecolor=colors[2],
    elinewidth=1,
    capsize=2,
    label="Datos 1 bar"
)

plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-2, 100)
plt.xlabel("N$_2$ concentration [%]")
plt.ylabel("Normalized yield")
plt.legend()
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/ArN2_bands_1bar.pdf", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Ph/MeV
# =========================================================
plt.figure(figsize=(6.5, 4.5))

plt.fill_between(
    fN2 * 100,
    y_sys_min*factor,
    y_sys_max*factor,
    alpha=0.30,
    label="Sistemático",
    color=colors[2]
)

plt.fill_between(
    fN2 * 100,
    y_low_stat*factor,
    y_up_stat*factor,
    alpha=0.30,
    label="Estadístico",
    color=colors[0]
)

plt.plot(
    fN2 * 100,
    y0*factor,
    lw=2,
    label="Ajuste nominal",
    color=colors[2]
)

plt.errorbar(
    yield_plot["N2 concentration (%)"],
    yield_plot["1.0bar"]*factor2,
    yerr=yield_plot["Err 1.0bar"]*factor2,
    marker="o",
    linestyle="none",
    ms=4,
    color=colors[2],
    ecolor=colors[2],
    elinewidth=1,
    capsize=2,
    label="Datos 1 bar"
)

plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-2, 100)
plt.xlabel("N$_2$ concentration [$\%$]")
plt.ylabel("Normalized yield")
plt.title("Ar-N$_2$ UV (300-400 nm)")
plt.ylim(100,5000)
plt.grid(False)
plt.legend()
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/ArN2_bands_1bar.pdf", dpi=300, bbox_inches="tight")
plt.show()