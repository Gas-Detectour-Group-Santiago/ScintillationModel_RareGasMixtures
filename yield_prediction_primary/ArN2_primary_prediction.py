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


#####################################################
###### Traemos los datos anteriormente generados 

DATA_DIR_EXP = "../data/Experimental/ArN2/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR =  "../data/Parameters"

yield_N2_uv  = pd.read_csv(os.path.join(DATA_DIR_EXP, "yield_N2.csv"))

degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2.csv"))

parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_primary.csv"))["parameter"].to_numpy()
print(parameter_data)
parameter_data[0]  = 1
print(parameter_data)



######################################33

x_data = [100,100,100]
y_data = [96,141.1,146.1]
sy_data = [40,2.1,2.1]

fN2 = np.logspace(-4,0,1000)
pressure = [1,2,3,4,5]


plt.figure(figsize=(6,4))
plt.style.use(['science','grid'])

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

for i in range(len(pressure)):
    yield_teo = theory_yield_N2_uv(parameter_data, degrad_data, fN2, pressure[i]) * 1000
    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        linewidth = 2,
        label=f"{pressure[i]} bar prediction"
    )

plt.errorbar(x_data,
            y_data,
            yerr = sy_data,
            marker="o",
            linestyle="none",
            label="1 bar",
            ms=4,
            color=colors[0],
            ecolor=colors[0],
            elinewidth=1,
            capsize=2,)

plt.xscale("log")
plt.ylabel("ph/MeV")
plt.title("Primary ArN$_2$ UV yield prediction")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)
plt.xlabel("N$_2$ concentration [\%]")
#plt.yscale("log")
plt.legend(loc="upper left")
plt.savefig("plots/ArN2_primary.pdf")