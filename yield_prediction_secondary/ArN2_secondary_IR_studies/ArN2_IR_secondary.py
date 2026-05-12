import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("default")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

sys.path.append(models_dir)
from ArN2_infrarred import *


sys.path.append(data_dir)
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


# ============================================================
# RUTAS
# ============================================================
folder_path = "../../data/Secondary_GarfieldData/ArN2/root"
table_path = "../../data/Secondary_GarfieldData/levels/ArN2_level_data.csv"

csv_folder = "../../data/Secondary_GarfieldData/ArN2/csv"
populations_dir = "../../data/Secondary_GarfieldData/ArN2/populations"
plots_dir = "plots"


os.makedirs(populations_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# ============================================================
# 1) EXPORTAR hLevels A CSV
# ============================================================
export_hlevels_to_csv(
    folder_path,
    table_path,
    object_name="hLevels",
    argon_update=True
)

# ============================================================
# 2) LEER GANANCIAS ne y ni
#    IMPORTANTE: usar el mismo gas_concentration en ambos pasos
# ============================================================
summary = read_data_per_primary_electron(
    folder_path,
    gas_concentration="n2"
)
print(summary)


# ============================================================
# 3) CONFIGURACIÓN DE POBLACIONES
# ============================================================
config = pd.DataFrame({
    "Ar_696": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.32,
        "energy up": 100,
        "name output": "Ar_696",
        "type": "excitation"
    },
    "Ar_727": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.32,
        "energy up": 100,
        "name output": "Ar_727",
        "type": "excitation"
    },
    "Ar_750": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.17,
        "energy up": 100,
        "name output": "Ar_750",
        "type": "excitation"
    },
    "Ar_763": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.17,
        "energy up": 1000,
        "name output": "Ar_763",
        "type": "excitation"
    },
    "Ar_772": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.32,
        "energy up": 100,
        "name output": "Ar_772",
        "type": "excitation"
    },
    "Ar_794": {
        "name principal": "EXC",
        "gas": "Ar",
        "energy low": 13.28,
        "energy up": 100,
        "name output": "Ar_794",
        "type": "excitation"
    },
})

garfield = read_garfield_csv_folder(
    folder_path=csv_folder,
    dataframe=config,
    output_dir=populations_dir,
    output_general_name=os.path.join(populations_dir, "ArN2_IR_secondary"),
    gas_concentration="n2",
    gain_summary=summary,
    normalized="ne"
)


# ============================================================
# 4) CARGA DE DATOS PARA EL MODELO
# ============================================================
DATA_DIR_EXP = "../../data/Experimental/ArN2/"
DATA_DIR_GARFIELD = populations_dir
DATA_DIR_PAR = "../../data/Parameters"


garfield_data = pd.read_csv(os.path.join(DATA_DIR_GARFIELD, "ArN2_IR_secondary.csv"))
garfield_data["concentration"] = garfield_data["concentration"] / 100.0

parameter_data = pd.read_csv(
    os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv")
)["parameter"].to_numpy()

pressure = 1
fN2 = np.logspace(-3, 0, 1000)
electric_fields = [70, 80, 90]


# ============================================================
# 5) PREDICCIÓN IR
# ============================================================
plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(electric_fields)))

for i, electric_field in enumerate(electric_fields):
    subset = garfield_data[garfield_data["electric_field"] == electric_field].copy()

    yield_teo = theory_yield_ArN2_Ir_696(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_727(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_750(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_763(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_772(parameter_data, subset, fN2, pressure) * 12 / 100

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{electric_field} kV/cm prediction"
    )
    

plt.title("1 bar 10k gain secondary IR (680-785nm) Yield Prediction for Ar/N2 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlabel("N$_2$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArN2_IR_secondary.pdf"))


# ============================================================
# 6) PREDICCIÓN IR per BAND
# ============================================================
plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(electric_fields)))

for i, electric_field in enumerate(electric_fields):
    subset = garfield_data[garfield_data["electric_field"] == electric_field].copy()


    yield_teo = theory_yield_ArN2_Ir_696(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_727(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_750(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_763(parameter_data, subset, fN2, pressure) * 12 / 100
    yield_teo += theory_yield_ArN2_Ir_772(parameter_data, subset, fN2, pressure) * 12 / 100

    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{electric_field} kV/cm prediction"
    )

plt.title("1 bar 10k gain secondary IR (680-785nm) Yield Prediction for Ar/N2 mixture")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlabel("N$_2$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArN2_IR_secondary_per_band.pdf"))
