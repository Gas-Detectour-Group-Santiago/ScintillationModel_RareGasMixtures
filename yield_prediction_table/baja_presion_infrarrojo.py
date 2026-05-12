import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots
from matplotlib import colors as mcolors

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArN2 import *
from ArN2_infrarred import *
from ArCF4 import *
from ArCF4_infrarred import *


def W_ArN2(xN2, WAr=26.4, WN2=34.8):
    return 1.0 / ((1.0-xN2)/WAr + xN2/WN2)


cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])


def W_ArCF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W

#####################################################
###### Traemos los datos anteriormente generados 

DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

degrad_data_ArN2 = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2.csv"))
degrad_data_ArN2_IR = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2_IR.csv"))
degrad_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4.csv"))
degrad_data_ArCF4_IR = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4_IR.csv"))

parameter_data_ArN2 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_primary.csv"))["parameter"].to_numpy()
parameter_data_ArN2_IR = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_IR_primary.csv"))["parameter"].to_numpy()
parameter_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()
parameter_data_ArCF4_IR = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv"))["parameter"].to_numpy()

normN2  = parameter_data_ArN2[0].copy()
normCF4 = parameter_data_ArCF4[0].copy()

Xray_energy_cf4 = 0.015 # MeV
Xray_energy_n2  = 0.012 # MeV

#################

def calcula_yields(pressure,Norm,Xray_energy_cf4=0.015,Xray_energy_n2=0.012):

    W_ArCF4_0 = W_ArCF4(0)
    W_ArN2_0 = W_ArN2(0)
    W_ArCF4_100 = W_ArCF4(1)
    W_ArN2_100 = W_ArN2(1)

    f0 = 0.00001
    f1 = 1

    y_cf4_ir  = theory_yield_ArCF4_Ir_696(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm
    y_cf4_ir += theory_yield_ArCF4_Ir_727(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm
    y_cf4_ir += theory_yield_ArCF4_Ir_750(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm
    y_cf4_ir += theory_yield_ArCF4_Ir_763(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm
    y_cf4_ir += theory_yield_ArCF4_Ir_772(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm
    y_cf4_ir += theory_yield_ArCF4_Ir_794(parameter_data_ArCF4_IR, degrad_data_ArCF4_IR, f0, pressure) * 1000 / Norm

    
    y_n2_ir = theory_yield_ArN2_Ir_696(parameter_data_ArN2_IR, degrad_data_ArN2_IR, f0, pressure)  * 1000 / Norm
    y_n2_ir += theory_yield_ArN2_Ir_727(parameter_data_ArN2_IR, degrad_data_ArN2_IR, f0, pressure) * 1000 / Norm
    y_n2_ir += theory_yield_ArN2_Ir_750(parameter_data_ArN2_IR, degrad_data_ArN2_IR, f0, pressure) * 1000 / Norm
    y_n2_ir += theory_yield_ArN2_Ir_763(parameter_data_ArN2_IR, degrad_data_ArN2_IR, f0, pressure) * 1000 / Norm
    y_n2_ir += theory_yield_ArN2_Ir_772(parameter_data_ArN2_IR, degrad_data_ArN2_IR, f0, pressure) * 1000 / Norm

    return np.array([y_cf4_ir])

############

names_tex = np.array([
    "0.01mbar",
    "0.1mbar",
    "1mbar",
    "2mbar",
    "5mbar",
    "10mbar",
])



############

x0 = calcula_yields(0.00001,normCF4)
x0 = np.append(x0,calcula_yields(0.0001,normCF4))
x0 = np.append(x0,calcula_yields(0.001,normCF4))
x0 = np.append(x0,calcula_yields(0.002,normCF4))
x0 = np.append(x0,calcula_yields(0.005,normCF4))
x0 = np.append(x0,calcula_yields(0.010,normCF4))

x1 = calcula_yields(0.00001,normN2)
x1 = np.append(x1,calcula_yields(0.0001,normN2))
x1 = np.append(x1,calcula_yields(0.001,normN2))
x1 = np.append(x1,calcula_yields(0.002,normN2))
x1 = np.append(x1,calcula_yields(0.005,normN2))
x1 = np.append(x1,calcula_yields(0.010,normN2))

print(f"Norm CF4 = {normCF4:.2f}")
print(f"Norm N2 = {normN2:.2f}")
print("=="*30)


df = pd.DataFrame({"Parameter"  :names_tex,
                   "Norm ArCF4" :x0,
                   "Norm ArN2"  :x1})

df.style.hide(axis="index").format(
    lambda x: f"$\\num{{{x:.3e}}}$" if isinstance(x, (int, float)) else x
    ).to_latex(
    "tex_table/selected_prediction_baja_presion.tex",
    caption="Prediction for absolute X-ray IR (680-800nm) primary scintillation in pure Argon (all data in ph/MeV).",
    label="tab:prediction_baja_presion",
    siunitx=True,
    hrules=True
)
