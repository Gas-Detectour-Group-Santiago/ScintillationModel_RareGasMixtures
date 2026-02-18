import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from ScintillationClass import Scintillation
from ArCF4_Completed import *
from ArCF4_PabloModel import *


#######################################################################
# ======================= 1) LECTURA DE DATOS =========================
#######################################################################

DATA_DIR = "../pickle_data"

yield_uv  = pd.read_pickle(os.path.join(DATA_DIR, "yield_uv_cal.pkl"))
yield_vis = pd.read_pickle(os.path.join(DATA_DIR, "yield_vis_cal.pkl"))

PCF3        = pd.read_pickle(os.path.join(DATA_DIR, "poblations_CF3.pkl"))
PCF4        = pd.read_pickle(os.path.join(DATA_DIR, "poblations_CF4.pkl"))
PArDbleStar = pd.read_pickle(os.path.join(DATA_DIR, "poblations_Ar_dbleStar.pkl"))
PAr3rd      = pd.read_pickle(os.path.join(DATA_DIR, "poblations_Ar_3rd.pkl"))

# ------------------------------------------------------------
# Preprocesado
# ------------------------------------------------------------
fCF4_real   = yield_uv["fCF4 real"].to_numpy()
fCF4        = PCF3["fCF4"]

PCF3        = PCF3[["CF3 >11.5","Err CF3 >11.5"]]
PCF4        = PCF4[["CF4 all","Err CF4 all"]]
PArDbleStar = PArDbleStar[["Ar** all","Err Ar** all"]]
PAr3rd      = PAr3rd[["Ar3rd all","Err Ar3rd all"]]

# Diccionario Yields
yields = {
    "fCF4": fCF4_real,
    "sCF4": yield_uv["Err fCF4 real"].to_numpy(),
    "vis": yield_vis.drop(columns=["fCF4", "fCF4 real", "Err fCF4 real"]),
    "uv":  yield_uv.drop(columns=["fCF4", "fCF4 real", "Err fCF4 real"]),
}


# Diccionario poblaciones de degradación
poblation_degrad_data = {
    "fCF4": fCF4.to_numpy(),
    "CF3": PCF3,
    "Ar dbleStar": PArDbleStar,
    "CF4": PCF4,
    "Ar 3rd": PAr3rd,
}

# Diccionario modelos físicos
scintillation_theory_models = {
    "CF3 dir":          Pgamma_CF3dir,
    "CF3 Ar dbleStar":  Pgamma_CF3ArDbleStar,
    "CF4 dir":          Pgamma_CF4dir,
    "CF4 Ar 3rd":       Pgamma_CF4Ar3rd,
    "Ar 3rd":           Pgamma_Ar3rd
}

#######################################################################
# =========== 2) CONSTRUCCIÓN DEL OBJETO PRINCIPAL ====================
#######################################################################

ArCF4 = Scintillation(
    yields=yields,
    poblation_degrad=poblation_degrad_data,
    scintillation_models=scintillation_theory_models
)

# Se puede comentar
ArCF4.plotPoblationInterpolation("CF3", savefig="InterpolacionPoblationCF3.pdf")


#######################################################################
# ======================== 3) DEFINO TEORÍA ===========================
#######################################################################

scintillation = {
    "vis": theory_yield_vis,
    "uv": theory_yield_uv_noP,
}

ArCF4.buildYieldFunctionsFromRaw(scintillation)

#######################################################################
# ======================== 4) AJUSTAMOS ===========================
#######################################################################

x0 = np.array([1,
               0.11352059, 0.00156568 ,0.03740022, 
               0.01793004 ,0.1, 0.31565123, 0.83056742])
lower = [0.0,
         0.0, 0.0, 0.0,  
         0.0, 0.07, 0.0, 0.0]
upper = [1.0, 
         1.0, 1.0, 10.0, 
         10.0, 10.0, 1.0, 100.0]





bounds=(lower, upper)

popt = ArCF4.fitParametersGlobalRaw_residuals(bands=["vis", "uv"], x0=x0, bounds=bounds)

ArCF4.exportParamsToCSV(
    archive="Parametros_Globales.csv",
    names=["N","P1", "P2", "P3", "K1", "K2", "K3", "K4"],
    band="global"
)

names = [
    "$N_{\\text{norm}}$",
    "$f_{\\mathrm{CF_3^*(2A_1')}}$",
    "$f_{\\mathrm{Ar^{**}}} P_{\\mathrm{CF_3^*(2A_1')}} $",
    "$\\frac{K_{\\mathrm{Ar^{**},Q(Ar)}}}{K_{\\mathrm{Ar^{**},Q(CF_4)}}}$",
    "$\\frac{\\tau_1}{K_{\\mathrm{cool}}}$",
    "$\\frac{K_{\\mathrm{scint}}}{\\tau_2}$",
    "$f_{\\mathrm{CF_4^{+}}}$",
    "$K_{\\mathrm{Ar^{++},Q(CF_4)}}$ [ns]"
]

ArCF4.exportParamsToTeX(
    archive="params_global.tex",
    names=names,
    band="global",
    caption="Parámetros ajustados del modelo global de centelleo",
    label="tab:scint_global_params",
    precision=3
)




J = popt.jac
m, p = J.shape
s2 = 2 * popt.cost / (m - p)
cov_theta = s2 * np.linalg.inv(J.T @ J)

print("Parámetros globales:", popt.x)# ===================== χ² ===========================

chi2 = 2 * popt.cost
N_res = popt.fun.size
N_par = popt.x.size
dof   = N_res - N_par
chi2_red = chi2 / dof

print(f"Chi2 (real): {chi2}")
print(f"Grados de libertad: {dof}")
print(f"Chi2 reducido: {chi2_red}")

print("cov_theta",cov_theta)


print("Total de residuos usados =", N_res)
print("Puntos de datos por banda:")

for band in ["vis", "uv"]:
    dfY = ArCF4.yields[band]
    cols_phys = [c for c in dfY.columns if ("err" not in c.lower() and "fcf4" not in c.lower())]

    print(f"  Banda {band}:")
    for col in cols_phys:
        print(f"    {col}: {len(dfY[col])} puntos")
        
#######################################################################
# ======================== 5) GRAFICAMOS ===========================
#######################################################################

ArCF4.choosePlotNormalization("vis", mode="handle_global")
ArCF4.choosePlotNormalization("uv", mode="handle_global")

ArCF4.enableExperimentalData("vis", 1.0)
ArCF4.enableTeoCurve("vis", 1.0)

ArCF4.enableExperimentalData("uv", 1.0)
ArCF4.enableTeoCurve("uv", 1.0)


#ArCF4.enableExperimentalData("vis", 3.0)
#ArCF4.enableTeoCurve("vis", 3.0)

#ArCF4.enableExperimentalData("uv", 3.0)
#ArCF4.enableTeoCurve("uv", 3.0)

ArCF4.enableExperimentalData("vis", 4.0)
ArCF4.enableTeoCurve("vis", 4.0)

ArCF4.enableExperimentalData("uv", 4.0)
ArCF4.enableTeoCurve("uv", 4.0)


#ArCF4.enableExperimentalData("vis", 5.0)
#ArCF4.enableTeoCurve("vis", 5.0)

#ArCF4.enableExperimentalData("uv", 5.0)
#ArCF4.enableTeoCurve("uv", 5.0)

ArCF4.plotTeoCurve("vis", savefig="Ajuste_VIS_GlobalNorm.pdf")
ArCF4.plotTeoCurve("uv",  savefig="Ajuste_UV_GlobalNorm.pdf")

#######################################################################
# =================== 6) MATRIZ DE CORRELACIÓN ========================
#######################################################################


# Construimos matriz de correlación a partir de covarianzas
diag = np.sqrt(np.diag(cov_theta))
outer = np.outer(diag, diag)
corr = cov_theta / outer

# Seguridad numérica
corr = np.clip(corr, -1, 1)

# Etiquetas de parámetros
param_labels = ["N", "P1", "P2", "P3", "K1", "K2", "K3", "K4"]

# DataFrame para seaborn
corr_df = pd.DataFrame(corr, columns=param_labels, index=param_labels)

# --- Plot estilo seaborn ---
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
    cbar_kws={"label": "Correlación"}
)
plt.title("Matriz de Correlación de Parámetros Ajustados", fontsize=14)
plt.tight_layout()

plt.savefig("CorrelationMatrix_GlobalFit.pdf", dpi=300)