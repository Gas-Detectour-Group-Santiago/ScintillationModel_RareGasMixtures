import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import dill 
import scienceplots

plt.style.use(['science', 'grid'])
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArN2 import *
from ArN2_infrarred import *


######################################33

DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2.csv"))
degrad_data_IR = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArN2_IR.csv"))


parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_primary.csv"))["parameter"].to_numpy()
parameter_data_IR = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArN2_IR_primary.csv"))["parameter"].to_numpy()

print(parameter_data)
print(parameter_data_IR)

norm = parameter_data[0].copy()
parameter_data[0] = 1

######################################

def gaussiana(x,mu,sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

######################################33

with open("../data/Experimental/ArN2/N2_primary_data_final.pkl", "rb") as f:
        df = dill.load(f)


spectrum = df.loc[0,"mean_spectrum"]

print(parameter_data)
print(norm)


cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
pressure = [1]
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))
concentrations = [0.1,1,10,100]

wavelength = np.linspace(300,800,2000)

equations = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}


# =========================================================
# PRIMERA PASADA: calcular todos los espectros y el ymax global
# =========================================================
all_spectra = []
global_ymax = 0

for con in concentrations:
    spectra_con = []
    factor = (1/0.012) * W_ArN2(con/100)


    for pres in pressure:
        yield_N2 = theory_yield_N2_uv(
            parameter_data, degrad_data, np.array([con/100]), pres
        )
   
        yield_total = 0.13 * factor * yield_N2[0] * gaussiana(wavelength, 310, 3)
        yield_total += 0.42 * factor * yield_N2[0] * gaussiana(wavelength, 335, 2.5)
        yield_total += 0.3 * factor * yield_N2[0] * gaussiana(wavelength, 355, 2.5)
        yield_total += 0.1 * factor * yield_N2[0] * gaussiana(wavelength, 378, 2.5)
        yield_total += 0.05* factor * yield_N2[0] * gaussiana(wavelength, 403, 2.5)

        for name, yield_IR in equations.items():
            yield_ir = yield_IR(
                parameter_data_IR, degrad_data_IR, np.array([con/100]), pres
            )
            yield_total += (factor/norm) * yield_ir[0] * gaussiana(wavelength, float(name), 2.8)

        spectra_con.append((pres, yield_total))
        global_ymax = max(global_ymax, np.max(yield_total))

    all_spectra.append((con, spectra_con))

# =========================================================
# SEGUNDA PASADA: dibujar una única figura con 4 paneles
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
axs = axs.ravel()

for ax, (con, spectra_con) in zip(axs, all_spectra):

    mask1 = df["N2 concentration (%)"] == con 

    for k, (pres, yield_total) in enumerate(spectra_con):
        ax.plot(
            wavelength,
            yield_total,
            color="blue",
            label=f"Pred. {pres:.1f} bar"
        )

        mask2 = df["P (bar)"] == pres   
        dic = df[(mask1 & mask2)].iloc[0]["mean_spectrum"]

        wavelen   = np.array(dic["wavelength"])
        intensity = np.array(dic["intensity"])

        if k==0:
            factor = (np.max(yield_total))/(np.max(intensity))

        ax.plot(
            wavelen,
            factor*intensity,
            color="red",
            label=f"Exp. {pres:.1f} bar"
        )


    ax.set_title(f"{con:.1f} $\%$ N$_2$")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel("ph/MeV/nm")
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.08)
    ax.set_ylim(0, 1.5 * global_ymax)
    ax.legend(ncol=2,loc="upper right")

fig.suptitle(r"Primary Ar-N$_2$ Spectra Prediction", fontsize=14)
fig.tight_layout()
fig.savefig("plots_ArN2/ArN2_concentration_comparation.pdf", dpi=300, bbox_inches="tight")
plt.show()