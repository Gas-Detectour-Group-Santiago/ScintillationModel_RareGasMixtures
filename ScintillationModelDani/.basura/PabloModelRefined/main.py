import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from ScintillationClass import *
from ArCF4_Completed import *
from ArCF4_PabloModel import *

#######################################################################
# ======================= 1) LECTURA DE DATOS =========================
#######################################################################

DATA_DIR = "../pickle_data"

yield_uv  = pd.read_pickle(os.path.join(DATA_DIR, "yield_uv.pkl"))
yield_vis = pd.read_pickle(os.path.join(DATA_DIR, "yield_vis.pkl"))

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

scintillation_vis = {
    "CF3 dir": ["Probabilidad"],
    "CF3 Ar dbleStar": ["Probabilidad"]
}

scintillation_uv = {
    "CF4 dir": ["Relajacion", "Centelleo"],
    "CF4 Ar 3rd": [""],
    "Ar 3rd": [""],
}

scintillation = {
    "vis": scintillation_vis,
    "uv": scintillation_uv,
}

ArCF4.build_theory_functions(scintillation)

#######################################################################
# =================== 4) AJUSTE DEL VISIBLE (VIS) =====================
#######################################################################

x0_vis = [0.10826166, 0.19710833]
params_vis = ArCF4.fitParamtersWithNormalization("vis", x0_vis, n0=1.0, idx_ref=-1)
print("\n=== Parámetros ajustados VIS ===")
print(params_vis)

ArCF4.choosePlotNormalization("vis", mode="index", idx_ref=-1)
ArCF4.enableExperimentalData("vis", 1.0)
ArCF4.enableExperimentalData("vis", 4.0)
ArCF4.enableTeoCurve("vis", 1.0)

ArCF4.plotTeoCurve("vis", savefig="AjusteVis.pdf")


#######################################################################
# =================== 5) AJUSTE DEL ULTRAVIOLETA (UV) =================
#######################################################################

x0_uv = [0.43153474, 2.30165488]
params_uv = ArCF4.fitParamtersWithNormalization("uv", x0_uv, n0=1.0, idx_ref=-1)
print("\n=== Parámetros ajustados UV ===")
print(params_uv)

ArCF4.choosePlotNormalization("uv", mode="index", idx_ref=-1)
ArCF4.enableExperimentalData("uv", 1.0)
ArCF4.enableExperimentalData("uv", 4.0)
ArCF4.enableTeoCurve("uv", 1.0)
ArCF4.enableTeoCurve("uv", 4.0)

ArCF4.plotTeoCurve("uv", savefig="AjusteUV.pdf")

#######################################################################
# ======= 6) PLOT GLOBAL VIS + UV NORMALIZADOS CONJUNTAMENTE ==========
#######################################################################

#ArCF4.set_manual_parameters("vis", [0.15,0.3,0.43153474, 2.30165488])




ArCF4.choosePlotNormalization(
    band="vis",
    mode="global",
    idx_ref=-1,
    global_bands=["vis", "uv"]
)

ArCF4.choosePlotNormalization(
    band="uv",
    mode="global",
    idx_ref=-1,
    global_bands=["vis", "uv"]
)

# 2) Activar datos experimentales y teoría (VIS + UV)
ArCF4.enableExperimentalData("vis", 1.0)
ArCF4.enableTeoCurve("vis", 1.0)


ArCF4.enableExperimentalData("uv", 1.0)
ArCF4.enableExperimentalData("uv", 1.0)

# 3) Plot VIS con normalización global
ArCF4.plotTeoCurve(
    band="vis",
    savefig="Ajuste_VIS_GlobalNorm.pdf"
)

# 4) Plot UV con normalización global
ArCF4.plotTeoCurve(
    band="uv",
    savefig="Ajuste_UV_GlobalNorm.pdf"
)