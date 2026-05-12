import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib import colors as mcolors
import scienceplots

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4_infrarred import *


#####################################################
###### Traemos los datos anteriormente generados 

DATA_DIR_EXP = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"


degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4_IR.csv"))

parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv"))["parameter"].to_numpy()
parameter_data_CF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()


######################################33

x_data = [0.01]
y_data = [17000] # A mi me da 136 ahora mismo!!
sy_data = [4000]

Norm = parameter_data_CF4[0]
print(Norm)

fN2 = np.logspace(-4,0,1000)
pressure = [1,2,3,4,5]

plt.figure(figsize=(6,4))
plt.style.use(['science','grid'])

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

for i in range(len(pressure)):

    yield_teo =  theory_yield_ArCF4_Ir_696(parameter_data, degrad_data, fN2, pressure[i]) * 1000  / Norm
    yield_teo += theory_yield_ArCF4_Ir_727(parameter_data, degrad_data, fN2, pressure[i]) * 1000/ Norm
    yield_teo += theory_yield_ArCF4_Ir_750(parameter_data, degrad_data, fN2, pressure[i]) * 1000/ Norm
    yield_teo += theory_yield_ArCF4_Ir_763(parameter_data, degrad_data, fN2, pressure[i]) * 1000/ Norm
    yield_teo += theory_yield_ArCF4_Ir_772(parameter_data, degrad_data, fN2, pressure[i]) * 1000/ Norm
    yield_teo += theory_yield_ArCF4_Ir_794(parameter_data, degrad_data, fN2, pressure[i]) * 1000/ Norm

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        linewidth = 2,
        label=f"{pressure[i]} bar prediction"
    )


# plt.errorbar(x_data,
#             y_data,
#             yerr = sy_data,
#             marker="o",
#             linestyle="none",
#             label="1 bar",
#             ms=4,
#             color=colors[0],
#             ecolor=colors[0],
#             elinewidth=1,
#             capsize=2,)

plt.xscale("log")
#plt.yscale("log")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)

plt.ylabel("ph/MeV")
plt.xlabel("CF$_4$ concentration [\%]")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)
plt.title("Primary Ar--CF$_4$ IR (680-800nm) yield prediction")
plt.legend()
plt.savefig("plots/ArCF4_IR_primary.pdf")