import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots

plt.style.use(["science", "grid"])

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
fit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../primary_fits'))

sys.path.append(models_dir)
from ArCF4 import *

sys.path.append(data_dir)
from read_Degrad import read_degrad
from read_experimental import read_experimental

sys.path.append(fit_dir)
from fiting import fitParameters, fitParameters_lmfit, fitParameters_minimize
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure

####################################################################################
# =================== INFORMACION EXTRA ========================
####################################################################################

DATA_DIR = "../data/Experimental/ArCF4/"
yield_uv  = pd.read_csv(os.path.join(DATA_DIR, "UV.csv"))
yield_vis = pd.read_csv(os.path.join(DATA_DIR, "vis.csv"))

DATA_DIR = "../data/Primary_DegradData"
degrad_data = pd.read_csv(os.path.join(DATA_DIR, "ArCF4.csv"))

names_csv = [
    "Nnorm",
    "PCF3dir vis$",
    "PAr**",
    "KAr**QAr",
    "KAr**QCF4",
    "1/tauDiscKrelax",
    "tauUvKCF4QCF4",
    "PCF4dir",
    "KAr++QCF4",
    "PAr++",
    "PCF3dir uv$",
]

names_tex = [
    "$N_{\\text{norm}}$",
    "$P_{\\mathrm{CF_3}}|_{\\mathrm{vis,dir}}$",
    "$P_{\\mathrm{Ar}^{**}} $",
    "${K_{\\mathrm{Ar^{**},Q(Ar)}}}$ [ns]",
    "${K_{\\mathrm{Ar^{**},Q(CF_4)}}}$ [ns]",
    "$1 / {\\tau_{\\mathrm{dis}} K_{\\mathrm{relax}}}$",
    "$\\tau_{\mathrm{uv}} K_{\mathrm{CF_4^{+,*}Q(CF_4)}}$",
    "$P_{\\mathrm{CF_4^{+,*}}}|_{\\mathrm{dir}}$",
    "$K_{\\mathrm{Ar^{++},Q(CF_4)}}$ [ns]",
    "$P_{\\mathrm{Ar}^{++}}$",
    "$P_{\\mathrm{CF_3}}|_{\\mathrm{uv,dir}}$",
]

archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"
yields = ["vis", "UV"]
presiones = [1, 2, 2.5, 3, 4, 5]
output_dir = "../data/Experimental/ArCF4/"

DATA_DIR = "../data/Parameters"
parameter_CF4 = pd.read_csv(os.path.join(DATA_DIR, "ArCF4_primary.csv"))["parameter"].to_numpy()


cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, 10))

####################################################################################

# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W

####################################################################################
# =================== DATA PARA AJUSTE  ========================
####################################################################################

x0 = np.array([
    0.14,
    0.10, 0.99, 3, 0.037 * 3,
    1.0, 0.065, 0.48, 50.10, 0.37,
    0.00001
])

lower = [
    0.0,
    0.0, 0.0, 0.0, 0.0,
    0.00, 0.065, 0.0, 50, 0.0,
    0.0
]

upper = [
    0.99,
    1.0, 1.0, 10000.0, 10000.0,
    10000.0, 0.066, 1.0, 50.2, 1.0,
    0.0001
]

bounds = (lower, upper)

equations = {
    "vis": theory_yield_vis,
    "uv": theory_yield_uv
}

yield_uv.loc[0, "fCF4"] = 0.001
yield_vis.loc[0, :].drop(['Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar'])
yield_vis = yield_vis.fillna(0)

w_cf4 =  W_CF4(yield_vis["fCF4"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar", "2.5bar", "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_cf4)[:, None]

yield_vis[y_cols]  = yield_vis[y_cols].to_numpy() * factor
yield_uv[y_cols]  = yield_uv[y_cols].to_numpy() * factor

experimental_data = {
    "vis": yield_vis,
    "uv": yield_uv
}


####################################################################################
# =================== AJUSTE NOMINAL ========================
####################################################################################


x0 = np.array([0.14,
               0.10, 0.99, 3, 0.037*3,
               1.0 ,0.065, 0.48, 50.10, 0.37,
               0.00001])
lower = [0.0,
         0.0, 0.0, 0.0, 0.0,
         0.00, 0.065, 0.0, 50, 0.0,
         0.0]

upper = [0.99, 
         1.0, 1.0, 10000.0, 10000.0,
         10000.0, 0.066, 1.0, 50.2, 1.0, 
         0.0001]


bounds = (lower, upper)

fixed_idx = [6, 8, 10]
fixed_error = 0.01

popt = fitParameters(
    equations,
    experimental_data,
    degrad_data,
    x0=x0,
    bounds=bounds,
    fixed_idx=fixed_idx,
    fixed_values = [0.065, 50.05, 0.0001],
    fixed_error=fixed_error
)

J = popt.jac
m, p = J.shape
s2 = 2 * popt.cost / (m - p)
chi2 = 2 * popt.cost
N_res = popt.fun.size
N_par = popt.x.size
dof = N_res - N_par
chi2_red = chi2 / dof

print("=" * 60)
print("Parámetros globales:", popt.x)
print(f"Chi2 (real): {chi2}")
print(f"Grados de libertad: {dof}")
print(f"Chi2 reducido: {chi2_red}")
print("=" * 60)

cov_theta = s2 * np.linalg.inv(J.T @ J)

popt_og = popt
par_natural = popt.x.copy()


####################################################################################
# =================== AJUSTES LOW & UP  ========================
####################################################################################

read_experimental(archivo_entrada, yields, presiones, output_dir, uncertainty_mode="systematic")

DATA_DIR = "../data/Experimental/ArCF4/"
yield_uv = pd.read_csv(os.path.join(DATA_DIR, "UV.csv"))
yield_vis = pd.read_csv(os.path.join(DATA_DIR, "vis.csv"))


w_cf4 =  W_CF4(yield_vis["fCF4"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar", "2.5bar", "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_cf4)[:, None]

yield_vis[y_cols]  = yield_vis[y_cols].to_numpy() * factor
yield_uv[y_cols]  = yield_uv[y_cols].to_numpy() * factor


y_cols = ['1.0bar', '2.0bar', '2.5bar', '3.0bar', '4.0bar', '5.0bar']
err_cols = ['Err 1.0bar', 'Err 2.0bar', 'Err 2.5bar', 'Err 3.0bar', 'Err 4.0bar', 'Err 5.0bar']

err_uv = yield_uv[err_cols].to_numpy()
err_vis = yield_vis[err_cols].to_numpy()

yield_uv_up = yield_uv.copy(deep=True)
yield_uv_low = yield_uv.copy(deep=True)

yield_vis_up = yield_vis.copy(deep=True)
print("Parámetros globales:", popt.x)
yield_vis_low = yield_vis.copy(deep=True)

yield_uv_up.loc[:, y_cols] = yield_uv_up.loc[:, y_cols].to_numpy() + err_uv
yield_uv_low.loc[:, y_cols] = yield_uv_low.loc[:, y_cols].to_numpy() - err_uv

yield_vis_up.loc[:, y_cols] = yield_vis_up.loc[:, y_cols].to_numpy() + err_vis
yield_vis_low.loc[:, y_cols] = yield_vis_low.loc[:, y_cols].to_numpy() - err_vis

experimental_data_up = {
    "vis": yield_vis_up,
    "uv": yield_uv_up
}

experimental_data_low = {
    "vis": yield_vis_low,
    "uv": yield_uv_low
}

popt_low = fitParameters(
    equations,
    experimental_data_low,
    degrad_data,
    x0=x0,
    bounds=bounds,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error
)
par_low = popt_low.x.copy()

popt_up = fitParameters(
    equations,
    experimental_data_up,
    degrad_data,
    x0=x0,
    bounds=bounds,
    fixed_idx=fixed_idx,
    fixed_error=fixed_error
)
par_up = popt_up.x.copy()

sigma_par_sis = np.abs(par_low - par_up) / 2
sigma_par_stat = np.sqrt(np.diag(cov_theta))

####################################################################################
# =================== FUNCION AUXILIAR PARA BANDAS ========================
####################################################################################

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
    var_y = np.einsum('ij,jk,ik->i', G_red, cov_theta, G_red)
    sigma_y = np.sqrt(np.maximum(var_y, 0))

    return y0, y0 - sigma_y, y0 + sigma_y

####################################################################################
# =================== PRIMARIO BANDAS ========================
####################################################################################

read_experimental(archivo_entrada, yields, presiones, output_dir, uncertainty_mode="all")

DATA_DIR = "../data/Experimental/ArCF4/"
yield_uv = pd.read_csv(os.path.join(DATA_DIR, "UV.csv"))
yield_vis = pd.read_csv(os.path.join(DATA_DIR, "vis.csv"))


w_cf4 =  W_CF4(yield_vis["fCF4"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar", "2.5bar", "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_cf4)[:, None]

yield_vis[y_cols]  = yield_vis[y_cols].to_numpy() * factor
yield_uv[y_cols]  = yield_uv[y_cols].to_numpy() * factor

yield_uv.loc[0, "fCF4"] = 0.001

free_idx = [0, 1, 2, 3, 4, 5, 7, 9]

os.makedirs("plots", exist_ok=True)

# =================== ULTRAVIOLETA BANDAS ========================

fCF4 = np.logspace(-5, 0, 100)

def model_uv(par):
    return theory_yield_uv(par, degrad_data, fCF4, 1)

y0_uv, y_low_stat_uv, y_up_stat_uv = statistical_band(
    model_uv, par_natural, cov_theta, free_idx
)

y_low_sys_uv = theory_yield_uv(par_low, degrad_data, fCF4, 1)
y_up_sys_uv = theory_yield_uv(par_up, degrad_data, fCF4, 1)
y0_uv = theory_yield_uv(par_natural, degrad_data, fCF4, 1)

plt.figure(figsize=(6, 4))
ymin = np.minimum(np.minimum(y0_uv, y_up_sys_uv), y_low_sys_uv)
ymax = np.maximum(np.maximum(y0_uv, y_up_sys_uv), y_low_sys_uv)

plt.fill_between(
    fCF4 * 100,
    ymin,
    ymax,
    alpha=0.3,
    label="Sistemático"
)

plt.fill_between(
    fCF4 * 100,
    y_low_stat_uv,
    y_up_stat_uv,
    alpha=0.3,
    label="Estadístico"
)

plt.plot(
    fCF4 * 100,
    y0_uv,
    lw=2,
    label="Ajuste nominal"
)

plt.errorbar(
    yield_uv["fCF4"],
    yield_uv["1.0bar"],
    yerr=yield_uv["Err 1.0bar"],
    label="Data",
    fmt=".r"
)


plt.xscale("log")
plt.yscale("log")
plt.xlabel("CF$_4$ concentration $\%$")
plt.ylabel("Arb. ")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_uv.pdf")
plt.show()

# =================== VISIBLE BANDAS ========================

fCF4 = np.logspace(-3, 0, 100)

def model_vis(par):
    return theory_yield_vis(par, degrad_data, fCF4, 1)

y0_vis, y_low_stat_vis, y_up_stat_vis = statistical_band(
    model_vis, par_natural, cov_theta, free_idx
)

y_low_sys_vis = theory_yield_vis(par_low, degrad_data, fCF4, 1)
y_up_sys_vis = theory_yield_vis(par_up, degrad_data, fCF4, 1)

plt.figure(figsize=(6, 4))
plt.fill_between(
    fCF4 * 100,
    y_low_sys_vis,
    y_up_sys_vis,
    alpha=0.3,
    label="Sistemático"
)

plt.fill_between(
    fCF4 * 100,
    y_low_stat_vis,
    y_up_stat_vis,
    alpha=0.3,
    label="Estadístico"
)

plt.plot(
    fCF4 * 100,
    y0_vis,
    lw=2,
    label="Ajuste nominal"
)

plt.errorbar(
    yield_vis["fCF4"],
    yield_vis["1.0bar"],
    yerr=yield_vis["Err 1.0bar"],
    label="Data",
    fmt=".r"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("CF$_4$ concentration $\%$")
plt.ylabel("Arb.")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_vis.pdf")
plt.show()


####################################################################################
# =================== PH per MeV ========================
####################################################################################

con_uv_cf4_morozov = [100.0]
y_uv_cf4_morozov   = [2175]
y_err_uv_cf4_morozov= [2600-2175]


read_experimental(archivo_entrada, yields, presiones, output_dir, uncertainty_mode="all")

DATA_DIR = "../data/Experimental/ArCF4/"
yield_uv = pd.read_csv(os.path.join(DATA_DIR, "UV.csv"))
yield_vis = pd.read_csv(os.path.join(DATA_DIR, "vis.csv"))

yield_vis[y_cols]  = yield_vis[y_cols].to_numpy() * factor
yield_uv[y_cols]  = yield_uv[y_cols].to_numpy() * factor

yield_uv.loc[0, "fCF4"] = 0.001

free_idx = [0, 1, 2, 3, 4, 5, 7, 9]

os.makedirs("plots", exist_ok=True)

norm = par_natural[0]


# =================== ULTRAVIOLETA BANDAS ========================

fCF4 = np.logspace(-5, 0, 100)

factor  = 1000 / norm 
factor2 = 1000 / norm 

def model_uv(par):
    return theory_yield_uv(par, degrad_data, fCF4, 1)

y0_uv, y_low_stat_uv, y_up_stat_uv = statistical_band(
    model_uv, par_natural, cov_theta, free_idx
)

y_low_sys_uv = theory_yield_uv(par_low, degrad_data, fCF4, 1)
y_up_sys_uv = theory_yield_uv(par_up, degrad_data, fCF4, 1)
y0_uv = theory_yield_uv(par_natural, degrad_data, fCF4, 1)

plt.figure(figsize=(6, 4))
ymin = np.minimum(np.minimum(y0_uv, y_up_sys_uv), y_low_sys_uv)
ymax = np.maximum(np.maximum(y0_uv, y_up_sys_uv), y_low_sys_uv)

plt.fill_between(
    fCF4 * 100,
    ymin*factor,
    ymax*factor,
    alpha=0.3,
    label="Sistemático",
    color=colors[2]
)

plt.fill_between(
    fCF4 * 100,
    y_low_stat_uv*factor,
    y_up_stat_uv*factor,
    alpha=0.3,
    label="Estadístico",
    color=colors[0]
)

plt.plot(
    fCF4 * 100,
    y0_uv*factor,
    lw=2,
    label="Ajuste nominal",
    color=colors[2]
)

plt.errorbar(
    yield_uv["fCF4"],
    yield_uv["1.0bar"]*factor2,
    yerr=yield_uv["Err 1.0bar"]*factor2,
    marker="o",
    linestyle="none",
    label="X-ray",
    ms=4,
    color=colors[2],
    ecolor=colors[2],
    elinewidth=1,
    capsize=2,
)


plt.errorbar(con_uv_cf4_morozov,
             y_uv_cf4_morozov,
             yerr=y_err_uv_cf4_morozov,
            marker="x",
            linestyle="none",
            ms=5,
            color=colors[6],
            ecolor=colors[6],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s Morozov")

plt.xscale("log")
# plt.yscale("log")
plt.grid(False)
plt.xlabel("CF$_4$ concentration $\%$")
plt.title("Ar-CF$_4$ UV (220-400 nm)")
plt.ylabel("ph/MeV")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_uv.pdf")
plt.show()

# =================== VISIBLE BANDAS ========================


cf4_red_E0 = [0.2, 0.4, 0.7, 1.0, 2.0, 7.0, 10.0]
y_red_E0   = [500, 700, 1050, 1450, 1950, 2400, 2550]
yerr_red_E0= [70, 70, 80, 100, 120, 160, 170]

# [400–750] nm, E = 100 V/cm (rojo abierto)
cf4_red_E100 = [0.2, 0.4, 0.7, 1.0, 2.0, 7.0, 10.0]
y_red_E100   = [450, 500, 600, 1150, 1300, 1850, 1900]
yerr_red_E100= [60, 60, 60, 90, 100, 120, 120]


vis_cf4_red_E100 = [100.0]
vis_y_red_E100   = [1184.7]
vis_yerr_red_E100= [47]


vis2_cf4_red_E100 = [100.0]
vis2_y_red_E100= [695]
vis2_yerr_red_E100= [827-695]



fCF4 = np.logspace(-3, 0, 100)

def model_vis(par):
    return theory_yield_vis(par, degrad_data, fCF4, 1)

y0_vis, y_low_stat_vis, y_up_stat_vis = statistical_band(
    model_vis, par_natural, cov_theta, free_idx
)


y_low_sys_vis = theory_yield_vis(par_low, degrad_data, fCF4, 1)
y_up_sys_vis = theory_yield_vis(par_up, degrad_data, fCF4, 1)
y0_vis = theory_yield_vis(par_natural, degrad_data, fCF4, 1)

plt.figure(figsize=(6, 4))
ymin = np.minimum(np.minimum(y0_vis, y_up_sys_vis), y_low_sys_vis)
ymax = np.maximum(np.maximum(y0_vis, y_up_sys_vis), y_low_sys_vis)

plt.figure(figsize=(6, 4))


plt.fill_between(
    fCF4 * 100,
    ymin*factor,
    ymax*factor,
    alpha=0.3,
    label="Sistemático",
    color=colors[2],
)


plt.fill_between(
    fCF4 * 100,
    y_low_stat_vis*factor,
    y_up_stat_vis*factor,
    alpha=0.3,
    label="Estadístico",
    color=colors[0],
)

plt.plot(
    fCF4 * 100,
    y0_vis*factor,
    lw=2,
    label="Ajuste nominal",
    color=colors[2],
)

plt.errorbar(
    yield_vis["fCF4"],
    yield_vis["1.0bar"]*factor2,
    yerr=yield_vis["Err 1.0bar"]*factor2,
    marker="o",
    linestyle="none",
    label="X-ray",
    ms=4,
    color=colors[2],
    ecolor=colors[2],
    elinewidth=1,
    capsize=2,
    
)

plt.errorbar(cf4_red_E100,
            y_red_E100,
            yerr=yerr_red_E100,
            ms=4,
            marker="o",
            linestyle="none",
            color=colors[5],
            ecolor=colors[5],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s P. Amedo")

plt.errorbar(vis2_cf4_red_E100,
             vis2_y_red_E100,
             yerr=vis2_yerr_red_E100,
            marker="x",
            linestyle="none",
            ms=5,
            color=colors[6],
            ecolor=colors[6],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s Morozov")


plt.errorbar(vis_cf4_red_E100,
             vis_y_red_E100,
             yerr=vis_yerr_red_E100,
            marker="v",
            linestyle="none",
            ms=5,
            color=colors[7],
            ecolor=colors[7],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s Lehaut")





plt.xscale("log")
# plt.yscale("log")
plt.grid(False)
plt.title("Ar-CF$_4$ Visible (400-700 nm)")
plt.xlabel("CF$_4$ concentration $\%$")
plt.ylabel("ph/MeV")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ArCF4_bands_vis.pdf")
plt.show()
