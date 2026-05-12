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
folder_path = "../../data/Secondary_GarfieldData/HeCF4/root"
table_path = "../../data/Secondary_GarfieldData/levels/HeCF4_level_data.csv"

csv_folder = "../../data/Secondary_GarfieldData/HeCF4/csv"
populations_dir = "../../data/Secondary_GarfieldData/HeCF4/populations"
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
        "energy low": 11.80,
        "energy up": 100,
        "name output": "Ar_dbleStar",
        "type": "excitation"
    },
    "CF3": {
        "name principal": "NEUTRAL DISS",
        "gas": "CF4",
        "energy low": 15.62,
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
    output_general_name=os.path.join(populations_dir, "HeCF4_secondary"),
    gas_concentration="cf4",
    gain_summary=summary,
    normalized="ni"
)

# ============================================================
# 6) CARGA DE DATOS PARA EL MODELO
# ============================================================
DATA_DIR_EXP = "../../data/Experimental/HeCF4/"
DATA_DIR_GARFIELD = populations_dir
DATA_DIR_PAR = "../../data/Parameters"

garfield_data = pd.read_csv(os.path.join(DATA_DIR_GARFIELD, "HeCF4_secondary.csv"))
garfield_data["concentration"] = garfield_data["concentration"] / 100.0



parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_secondary.csv"))["parameter"].to_numpy()

print("parameter_data original:")
print(parameter_data)

parameter_data[0] = 1
parameter_data[1] = 0.39
parameter_data[2] = 0.39
parameter_data[5] *= 1 # 50 #
parameter_data[6] *= 1 # 600 # 500 # 50 #
parameter_data[7] *= 0.2
parameter_data[-1] = 0 # 0.2 # 1 #

print("parameter_data modificado:")
print(parameter_data)


# ============================================================
# 7) MALLA DE CONCENTRACIONES Y CAMPOS
# ============================================================
fN2 = np.logspace(-3, 0, 1000)
gaps = [0.05, 0.57]
npe = [1000, 50]
electricField = [55,0]
pressure = [1, 0.300]

# =========================
# He-CF4 LIP
# =========================

concentration_cf4_LIP = np.array([20, 40, 100])

he_cf4_conditions_LIP = np.array([
    60,
    75,
    95
])

he_cf4_vis_LIP = np.array([
    0.05981728,
    0.0633149,
    0.09335376
])

he_cf4_vis_err_LIP = np.array([
    0.016221195011451566,
    0.023434172057910808,
    0.02018412139031937
])


he_cf4_UV_LIP = np.array([
    0.0586689111317149,
    0.12606696047521188,
    0.03942121448304051
])

he_cf4_UV_err_LIP = np.array([
    0.014241770732051665,
    0.04213797898747448,
    0.00859011077564899
])



# =========================
# He-CF4 Florian
# =========================

concentration_cf4_Florian = np.array([20])

pressure_cf4_Florian = np.array([0.3])

he_cf4_conditions_Florian = np.array([
    12.105
])

he_cf4_vis_Florian = np.array([
    0.03459508998978698
])

he_cf4_vis_err_Florian = np.array([
    0.007479843426044407
])


he_cf4_UV_Florian = np.array([
    0.0709633243946791
])

he_cf4_UV_err_Florian = np.array([
    0.015463319066967298
])



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
    mask1 = garfield_data["gap_mm"] == gap
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = garfield_data["pressure"] == pressure[i]
    subset = garfield_data[(mask1 & mask2 & mask3)].copy()

    yield_teo = (
        theory_yield_vis(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * 15
    )

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{gap} mm prediction {pressure[i]:.3f} bar"
    )
 
plt.errorbar(
    concentration_cf4_LIP,
    he_cf4_vis_LIP,
    yerr=he_cf4_vis_err_LIP,
    fmt=".",
    color = colors[0]
)

plt.errorbar(
    concentration_cf4_Florian,
    he_cf4_vis_Florian,
    yerr=he_cf4_vis_err_Florian,
    fmt=".",
    color = colors[1]
)


#plt.title("1 bar 100 gain secondary visible yield prediction for Ar/CF4 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlabel("CF$_4$ concentration [\%]")
plt.title("He-CF$_4$ Vis (400-720 nm)")
plt.xlim(10, 110)
plt.ylim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "HeCF4_vis_secondary_threshold.pdf"))



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

par_7_og = parameter_data[7] 

for i, gap in enumerate(gaps):
    
    mask1 = garfield_data["gap_mm"] == gap
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = garfield_data["pressure"] == pressure[i]
    subset = garfield_data[(mask1 & mask2 & mask3)].copy()
    
    #parameter_data[7] = par_7_og/pressure[i]

    yield_teo_uv = (
        theory_yield_uv(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * 15
    )

    yield_teo_vis = (
        theory_yield_vis(parameter_data, subset, fN2, pressure[i])
        / npe[i]
        * 15
    )
    

    yield_teo = 0.8*(yield_teo_uv + 0.25 * yield_teo_vis)

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{gap} mm prediction {pressure[i]:.3f} bar"
    )

plt.errorbar(
    concentration_cf4_LIP,
    he_cf4_UV_LIP,
    yerr=he_cf4_UV_err_LIP,
    fmt=".",
    color = colors[0]
)

plt.errorbar(
    concentration_cf4_Florian,
    he_cf4_UV_Florian,
    yerr=he_cf4_UV_err_Florian,
    fmt=".",
    color = colors[1]
)


#plt.title("1 bar 100 gain secondary UV yield prediction for Ar/CF4 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlim(10, 110)
plt.ylim(0, 0.2)
plt.xlabel("CF$_4$ concentration [\%]")
plt.title("He-CF$_4$ UV (220-400 nm)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "HeCF4_uv_secondary_threshold.pdf"))
