import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots
plt.style.use('grid')

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4 import *


######################################33

DATA_DIR_EXP = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

yield_N2_uv  = pd.read_csv(os.path.join(DATA_DIR_EXP, "vis.csv"))

degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4.csv"))

parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()
print(parameter_data)
parameter_data[0] = 1 

######################################33


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

 


cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100])/100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

def ion_potential(f):
    f_cf4 = np.asarray(f, dtype=float)
    W=np.interp(f_cf4,cf4_pct,ion_pot)
    return W


######################################33

fCF4 = np.logspace(-3,0,1000)
pressure = [1,2,3,4,5]

plt.figure(figsize=(6,4))
plt.style.use(['science','grid'])

print(parameter_data)


cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

for i in range(len(pressure)):
    yield_teo = (theory_yield_vis(parameter_data,degrad_data,fCF4,pressure[i])) * 1000
    plt.plot(
        fCF4 * 100,
        yield_teo,
        color=colors[i],
        linewidth = 2,
        label=f"{pressure[i]} bar prediction"
    )


plt.errorbar(cf4_red_E100,
            y_red_E100,
            yerr=yerr_red_E100,
            ms=4,
            marker="o",
            linestyle="none",
            color=colors[0],
            ecolor=colors[0],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s P. Amedo")


plt.errorbar(vis_cf4_red_E100,
             vis_y_red_E100,
             yerr=vis_yerr_red_E100,
            marker="v",
            linestyle="none",
            ms=5,
            color=colors[0],
            ecolor=colors[0],
            elinewidth=1,
            capsize=2,
            label="$\\alpha$'s Lehaut")

plt.xscale("log")
#plt.yscale("log")
plt.ylabel("ph/MeV")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)
plt.xlabel(r"CF$_4$ concentration [\%]")
plt.title("Primary Ar-CF4 visible yield prediction")
plt.legend()
plt.savefig("plots/ArCF4_vis_primary.pdf")


######################################33

fCF4 = np.logspace(-3,0,1000)
pressure = [1,2,3,4,5]



plt.figure(figsize=(6,4))
plt.style.use(['science','grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size": 11})          # specify font size here


print(parameter_data)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

for i in range(len(pressure)):
    yield_teo = (theory_yield_uv(parameter_data,degrad_data,fCF4,pressure[i])) * (1/0.015) * ion_potential(fCF4)
    plt.plot(
        fCF4 * 100,
        yield_teo,
        color=colors[i],
        label=f"{pressure[i]} bar prediction",
        linewidth = 2
    )




plt.xscale("log")
#plt.yscale("log")
plt.ylabel("ph/MeV")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)
plt.title("Primary Ar-CF4 UV yield prediction")
plt.xlabel("CF$_4$ concentration [\%]")
plt.legend()
plt.savefig("plots/ArCF4_uv_primary.pdf")
