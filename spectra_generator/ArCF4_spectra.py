import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4 import *
from ArCF4_infrarred import *

######################################33

DATA_DIR_EXP = "../data/Experimental/ArCF4/"
DATA_DIR_DEGRAD = "../data/Primary_DegradData"
DATA_DIR_PAR = "../data/Parameters"

yield_N2_uv  = pd.read_csv(os.path.join(DATA_DIR_EXP, "vis.csv"))

degrad_data = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4.csv"))
degrad_data_IR = pd.read_csv(os.path.join(DATA_DIR_DEGRAD, "ArCF4_IR.csv"))

parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()
parameter_data_IR = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv"))["parameter"].to_numpy()

norm = parameter_data[0].copy()
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



cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100])/100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

def ion_potential(f):
    f_cf4 = np.asarray(f, dtype=float)
    W=np.interp(f_cf4,cf4_pct,ion_pot)
    return W

def gaussiana(x,mu,sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

######################################33

fCF4 = np.logspace(-3,0,1000)


equations = {
    "696": theory_yield_ArCF4_Ir_696,
    "727": theory_yield_ArCF4_Ir_727,
    "750": theory_yield_ArCF4_Ir_750,
    "763": theory_yield_ArCF4_Ir_763,
    "772": theory_yield_ArCF4_Ir_772,
    "794": theory_yield_ArCF4_Ir_794,
}


print(parameter_data)


cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
pressure = [1,2,3,4,5,10]#,2,3,4,5]
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))
concentrations = [0.1,1,10,100]

wavelength = np.linspace(200,800,2000)

for i,pres in enumerate(pressure):
    plt.figure(figsize=(6,4))
    for j,con in enumerate(concentrations):

        yield_vis = (theory_yield_vis(parameter_data,degrad_data,np.array([con/100]),pres)) * (1/0.015) * ion_potential(con/100)

        yield_uv, yield_cf4, yield_ArDbleStar, yield_cf3_uv = theory_yield_uv(
            parameter_data, degrad_data, np.array([con/100]), pres, activate_components=True
        )

        factor = (1/0.015) * ion_potential(con/100)

        yield_uv *= factor
        yield_cf4 *= factor
        yield_ArDbleStar *= factor
        yield_cf3_uv *= factor
                
        print(yield_vis)
        print(yield_uv)

        plt.style.use(['science','grid'])
        plt.grid(True, which='major', alpha=0.3)
        plt.grid(True, which='minor', alpha=0.08)

        yield_vis = yield_vis[0]*gaussiana(wavelength,630,40) 
        yield_cf4_230 = (0.75/1.85) *  yield_cf4[0]*gaussiana(wavelength,230,20)
        yield_cf4_290 = (1.0/1.85) * yield_cf4[0]*gaussiana(wavelength,290,20)
        yield_cf4_364 = (0.1/1.85) *  yield_cf4[0]*gaussiana(wavelength,364,40)
        yield_cf4 = yield_cf4_230 + yield_cf4_290 + yield_cf4_364
        yield_arDbleStar = yield_ArDbleStar[0]*gaussiana(wavelength,220,60)
        yield_CF3 = yield_cf3_uv[0]*gaussiana(wavelength,245,60)
        
        yield_total = yield_vis + yield_cf4 + yield_arDbleStar + yield_CF3

        for name, yield_IR in equations.items():
            yield_ir = yield_IR(
                parameter_data_IR, degrad_data_IR, np.array([con/100]), pres
            )
            yield_total += (factor/norm) * yield_ir[0] * gaussiana(wavelength, float(name), 2.5)


        plt.plot(wavelength,yield_total,label=f"{con:.1f} $\%$ CF$_4$")

        plt.ylabel("ph/MeV/nm")
        plt.xlabel("$\lambda$ [nm]")
        plt.title(f"Primary Ar-CF4 Spectra Prediction {pres:.1f} bar")
        plt.legend()
        plt.legend()
        plt.savefig(f"plots_ArCF4/ArCF4_{pres:.1f}bar.pdf")

#####



# Guardamos todo primero
all_spectra = []
global_ymax = 0

for con in concentrations:
    spectra_con = []

    for pres in pressure:
        yield_vis = theory_yield_vis(
            parameter_data, degrad_data, np.array([con/100]), pres
        ) * (1/0.015) * ion_potential(con/100)

        yield_uv, yield_cf4, yield_ArDbleStar, yield_cf3_uv = theory_yield_uv(
            parameter_data, degrad_data, np.array([con/100]), pres, activate_components=True
        )

        factor = (1/0.015) * ion_potential(con/100)

        yield_uv *= factor
        yield_cf4 *= factor
        yield_ArDbleStar *= factor
        yield_cf3_uv *= factor

        yield_vis_spec = yield_vis[0] * gaussiana(wavelength, 630, 40)

        yield_cf4_230 = (0.8/1.85) * yield_cf4[0] * gaussiana(wavelength, 230, 20)
        yield_cf4_290 = (0.95/1.85) * yield_cf4[0] * gaussiana(wavelength, 290, 20)
        yield_cf4_364 = (0.10/1.85) * yield_cf4[0] * gaussiana(wavelength, 364, 40)
        yield_cf4_spec = yield_cf4_230 + yield_cf4_290 + yield_cf4_364

        yield_arDbleStar_spec = yield_ArDbleStar[0] * gaussiana(wavelength, 220, 60)
        yield_CF3_spec = yield_cf3_uv[0] * gaussiana(wavelength, 245, 60)


        yield_total = (
            yield_vis_spec
            + yield_cf4_spec
            + yield_arDbleStar_spec
            + yield_CF3_spec
        )


        for name, yield_IR in equations.items():
            yield_ir = yield_IR(
                parameter_data_IR, degrad_data_IR, np.array([con/100]), pres
            )
            yield_total += (factor/norm) * yield_ir[0] * gaussiana(wavelength, float(name), 2.5)


        spectra_con.append((pres, yield_total))
        global_ymax = max(global_ymax, np.max(yield_total))

    all_spectra.append((con, spectra_con))


# Figura con 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
axs = axs.ravel()

for ax, (con, spectra_con) in zip(axs, all_spectra):
    for k, (pres, yield_total) in enumerate(spectra_con):
        ax.plot(
            wavelength,
            yield_total,
            color=colors[k],
            label=f"{pres:.1f} bar"
        )

    ax.set_title(f"Ar {100-con:.1f} $\%$ CF$_4$ {con:.1f} $\%$ ")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel("ph/MeV/nm")
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.08)
    ax.set_ylim(0, global_ymax * 1.05)
    ax.legend(ncols=2,loc="upper right")

fig.suptitle("Primary Ar-CF$_4$ Spectra Prediction", fontsize=14)
fig.tight_layout()
fig.savefig("plots_ArCF4/ArCF4_concentration.pdf", dpi=300, bbox_inches="tight")
plt.show()