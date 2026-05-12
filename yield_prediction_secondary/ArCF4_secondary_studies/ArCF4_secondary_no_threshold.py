import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use("grid")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4 import *
from read_Degrad import read_degrad
from read_experimental import read_experimental
from read_Root import export_hlevels_to_csv,read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


# ============================================================
# RUTAS
# ============================================================
folder_path = "../../data/Secondary_GarfieldData/ArCF4/root"
table_path = "../../data/Secondary_GarfieldData/levels/ArCF4_level_data.csv"

csv_folder = "../../data/Secondary_GarfieldData/ArCF4/csv"
populations_dir = "../../data/Secondary_GarfieldData/ArCF4/populations"
plots_dir = "plots"

os.makedirs(populations_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

export_hlevels_to_csv(
    folder_path,
    table_path,
    object_name="hLevels",
    argon_update=True
)

# ============================================================
# 2) LEER GANANCIAS ne y ni
#    IMPORTANTE: usar el mismo gas_concentration que luego en
#    read_garfield_csv_folder para que el merge sea consistente
# ============================================================
summary = read_data_per_primary_electron(
    folder_path,
    gas_concentration="cf4"
)


# ============================================================
# 3) CONFIGURACIÓN DE POBLACIONES
# ============================================================
config = pd.DataFrame({
    "CF4": {
        "name principal": "ION",
        "gas": "CF4",
        "energy low": 15.5,
        "energy up": 16,
        "name output": "CF4",
        "type": "ionisation"
    },
    "Ar**": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 0,
        "energy up": 100,
        "name output": "Ar_dbleStar",
        "type": "excitation"
    },
    "CF3": {
        "name principal": "NEUTRAL DISS",
        "gas": "CF4",
        "energy low": 0,
        "energy up": 100,
        "name output": "CF3",
        "type": "inelastic"
    },
    "Ar3rd": {
        "name principal": "IONISATION",
        "gas": "Ar",
        "energy low": 40,
        "energy up": 120,
        "name output": "Ar_3rd",
        "type": "ionisation"
    }
})


# ============================================================
# 5) POBLACIONES NORMALIZADAS POR ne
# ============================================================
garfield_norm_ne = read_garfield_csv_folder(
    folder_path=csv_folder,
    dataframe=config,
    output_dir=populations_dir,
    output_general_name=os.path.join(populations_dir, "ArCF4_secondary"),
    gas_concentration="cf4",
    gain_summary=summary,
    normalized="ne"
)

# ============================================================
# 6) CARGA DE DATOS PARA EL MODELO
# ============================================================
DATA_DIR_EXP = "../../data/Experimental/ArCF4/"
DATA_DIR_GARFIELD = populations_dir
DATA_DIR_PAR = "../../data/Parameters"

garfield_data = pd.read_csv(os.path.join(DATA_DIR_GARFIELD, "ArCF4_secondary.csv"))
garfield_data["concentration"] = garfield_data["concentration"] / 100.0

parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()

print("parameter_data original:")
print(parameter_data)

parameter_data[0] = 1
parameter_data[-1] = 0.01 # 50 #

print("parameter_data modificado:")
print(parameter_data)


# ============================================================
# 7) MALLA DE CONCENTRACIONES Y CAMPOS
# ============================================================
fN2 = np.logspace(-3, 0, 1000)
gaps = [0.05, 0.57, 0.57]
npe = [1000, 50, 50]
pressure = [1, 0.050, 0.025]

# ============================================================
# 8) YIELD VISIBLE
# ============================================================
plt.figure(figsize=(6, 4))
plt.style.use("grid")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

for i, gap in enumerate(gaps):
    subset = garfield_data[garfield_data["gap_mm"] == gap].copy()

    yield_teo = (
        theory_yield_vis(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * ion_potential(fN2)
    )

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{gap} mm prediction {pressure[i]:.3f} bar"
    )
 
plt.errorbar(
    [100, 67, 10, 5],
    [0.09335376, 0.2802068, 0.38966203, 0.38287151],
    yerr=np.array([0.1, 0.3, 0.39, 0.38]) * 0.25,
    fmt=".",
    color = colors[0]
)
plt.errorbar(
    [20],
    [0.12],
    yerr=np.array([0.12]) * 0.25,
    fmt=".",
    color = colors[1]

)

plt.errorbar(
    [100],
    [0.1],
    yerr=np.array([0.1]) * 0.25,
    fmt=".",
    color = colors[2]

)

#plt.title("1 bar 100 gain secondary visible yield prediction for Ar/CF4 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlabel("CF$_4$ concentration [\%]")
plt.xlim(1, 110)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_vis_secondary_no_threshold.pdf"))



# ============================================================
# 9) YIELD UV
# ============================================================
plt.figure(figsize=(6, 4))
plt.style.use("grid")
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.08)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

for i, gap in enumerate(gaps):
    subset = garfield_data[garfield_data["gap_mm"] == gap].copy()

    yield_teo_uv = (
        theory_yield_uv(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * ion_potential(fN2)
    )
    yield_teo_vis = (
        theory_yield_vis(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * ion_potential(fN2)
    )

    yield_teo = 0.8*yield_teo_uv 

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{gap} mm prediction {pressure[i]:.3f} bar"
    )

plt.errorbar(
    [100, 67, 10, 5],
    [0.03942121448304051, 0.044033601820777875, 0.08455236611728804, 0.06737771706391489],
    yerr=np.array([0.04, 0.045, 0.085, 0.068]) * 0.25,
    fmt=".",
    color = colors[0]

)
plt.errorbar(
    [20],
    [0.028],
    yerr=np.array([0.028]) * 0.25,
    fmt=".",
    color = colors[1]

)

plt.errorbar(
    [100],
    [0.14],
    yerr=np.array([0.17]) * 0.25,
    fmt=".",
    color = colors[2]

)

#plt.title("1 bar 100 gain secondary UV yield prediction for Ar/CF4 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlim(1, 110)
plt.ylim(0.01, 0.19)
plt.xlabel("CF$_4$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_uv_secondary_no_threshold.pdf"))
