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
from ArCF4_infrarred import *


sys.path.append(data_dir)
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
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
    gas_concentration="cf4"
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
# 4) CARGA DE DATOS PARA EL MODELO
# ============================================================

DATA_DIR_EXP = "../../data/Experimental/ArCF4/"
DATA_DIR_GARFIELD = populations_dir
DATA_DIR_PAR = "../../data/Parameters"

garfield_data = pd.read_csv(os.path.join(DATA_DIR_GARFIELD, "ArCF4_secondary.csv"))
garfield_data["concentration"] = garfield_data["concentration"] / 100.0



parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv"))["parameter"].to_numpy()

parameter_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_secondary_2_0.csv"))["parameter"].to_numpy()

norm1 = parameter_data_ArCF4[0]

parameter_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()

norm2 = parameter_data_ArCF4[0]

# ============================================================
# 7) MALLA DE CONCENTRACIONES Y CAMPOS
# ============================================================
fN2 = np.logspace(-3, 0, 1000)
gaps = [0.05, 0.57, 0.57]
npe = [1000, 50, 50]
electricField = [60, 5, 5]
pressure = [1, 1, 0.025]

norm = [norm1, norm1, norm1]


# ============================================================
# 5) EXPERIMENTAL DATA
# ============================================================


concentration_cf4 = np.array([5, 10, 67])

ar_cf4_696_ir_LIP = np.array([
    0.0007423954814679462,
    0.00016606742822049213,
    5.7976067957635574e-05
])

ar_cf4_696_err_ir_LIP = np.array([
    0.00014864734147261504,
    3.337848055232784e-05,
    1.1643395769579259e-05
])


ar_cf4_727_ir_LIP = np.array([
    0.0,
    0.00023289581724378054,
    0.00013834145604106908
])

ar_cf4_727_err_ir_LIP = np.array([
    0.0,
    4.681055514551999e-05,
    2.778326265939665e-05
])
ar_cf4_750_ir_LIP = np.array([
    0.030352019422880832,
    0.01646440994283698,
    0.004112455696967686
])

ar_cf4_750_err_ir_LIP = np.array([
    0.006077282402925308,
    0.003309240065745319,
    0.0008259088784642111
])


ar_cf4_763_ir_LIP = np.array([
    0.028478502055388058,
    0.010520420255841482,
    0.0009514346999852515
])

ar_cf4_763_err_ir_LIP = np.array([
    0.0057021543440504,
    0.0021145365269683238,
    0.00019107764895222087
])


ar_cf4_772_ir_LIP = np.array([
    0.009542209996882418,
    0.00406238907466698,
    0.0003514776387269656
])

ar_cf4_772_err_ir_LIP = np.array([
    0.0019106045001854228,
    0.0008165139677163302,
    7.058763031059062e-05
])


ar_cf4_794_ir_LIP = np.array([
    0.0032255060792284705,
    0.0,
    0.00010916911326505648
])

ar_cf4_794_err_ir_LIP = np.array([
    0.0006458321953051536,
    0.0,
    2.1924549841632906e-05
])

concentration_cf4_Florian = np.array([20])

# ============================================================
# Florian th-GEM, Ar/CF4 80/20, 1 bar
# ============================================================

ar_cf4_696_ir_Florian_1bar = np.array([
    0.0012121110778869518
])

ar_cf4_696_err_ir_Florian_1bar = np.array([
    0.00026032890460820274
])


ar_cf4_727_ir_Florian_1bar = np.array([
    0.0009963113528044523
])

ar_cf4_727_err_ir_Florian_1bar = np.array([
    0.0002139809196170797
])


ar_cf4_750_ir_Florian_1bar = np.array([
    0.010641617291858389
])

ar_cf4_750_err_ir_Florian_1bar = np.array([
    0.0022855335813600896
])


ar_cf4_763_ir_Florian_1bar = np.array([
    0.005277639438703508
])

ar_cf4_763_err_ir_Florian_1bar = np.array([
    0.0011334952044080517
])


ar_cf4_772_ir_Florian_1bar = np.array([
    0.0018895075477094276
])

ar_cf4_772_err_ir_Florian_1bar = np.array([
    0.0004058154727878853
])


ar_cf4_794_ir_Florian_1bar = np.array([
    0.0
])

ar_cf4_794_err_ir_Florian_1bar = np.array([
    0.0
])


# ============================================================
# Florian th-GEM, Ar/CF4 80/20, 50 mbar
# ============================================================

ar_cf4_696_ir_Florian_50mbar = np.array([
    0.002246119186430594
])

ar_cf4_696_err_ir_Florian_50mbar = np.array([
    0.00048240607489727035
])


ar_cf4_727_ir_Florian_50mbar = np.array([
    0.0005887938444242574
])

ar_cf4_727_err_ir_Florian_50mbar = np.array([
    0.0001264571039365711
])


ar_cf4_750_ir_Florian_50mbar = np.array([
    0.057088337541668865
])

ar_cf4_750_err_ir_Florian_50mbar = np.array([
    0.012261041623375114
])


ar_cf4_763_ir_Florian_50mbar = np.array([
    0.02364076551779444
])

ar_cf4_763_err_ir_Florian_50mbar = np.array([
    0.005077401488711407
])


ar_cf4_772_ir_Florian_50mbar = np.array([
    0.00668799699933377
])

ar_cf4_772_err_ir_Florian_50mbar = np.array([
    0.0014364021289985191
])


ar_cf4_794_ir_Florian_50mbar = np.array([
    0.004140203178868282
])

ar_cf4_794_err_ir_Florian_50mbar = np.array([
    0.0008892044450984726
])
ar_cf4_sum_ir_LIP = (
    ar_cf4_696_ir_LIP
    + ar_cf4_727_ir_LIP
    + ar_cf4_750_ir_LIP
    + ar_cf4_763_ir_LIP
    + ar_cf4_772_ir_LIP
    + ar_cf4_794_ir_LIP
)

ar_cf4_sum_err_ir_LIP = np.sqrt(
    ar_cf4_696_err_ir_LIP**2
    + ar_cf4_727_err_ir_LIP**2
    + ar_cf4_750_err_ir_LIP**2
    + ar_cf4_763_err_ir_LIP**2
    + ar_cf4_772_err_ir_LIP**2
    + ar_cf4_794_err_ir_LIP**2
)

ar_cf4_sum_ir_Florian_1bar = (
    ar_cf4_696_ir_Florian_1bar
    + ar_cf4_727_ir_Florian_1bar
    + ar_cf4_750_ir_Florian_1bar
    + ar_cf4_763_ir_Florian_1bar
    + ar_cf4_772_ir_Florian_1bar
    + ar_cf4_794_ir_Florian_1bar
)* 25e-1

ar_cf4_sum_err_ir_Florian_1bar = np.sqrt(
    ar_cf4_696_err_ir_Florian_1bar**2
    + ar_cf4_727_err_ir_Florian_1bar**2
    + ar_cf4_750_err_ir_Florian_1bar**2
    + ar_cf4_763_err_ir_Florian_1bar**2
    + ar_cf4_772_err_ir_Florian_1bar**2
    + ar_cf4_794_err_ir_Florian_1bar**2
)* 25e-1


ar_cf4_sum_ir_Florian_50mbar = (
    ar_cf4_696_ir_Florian_50mbar
    + ar_cf4_727_ir_Florian_50mbar
    + ar_cf4_750_ir_Florian_50mbar
    + ar_cf4_763_ir_Florian_50mbar
    + ar_cf4_772_ir_Florian_50mbar
    + ar_cf4_794_ir_Florian_50mbar
) * 25e-1

ar_cf4_sum_err_ir_Florian_50mbar = np.sqrt(
    ar_cf4_696_err_ir_Florian_50mbar**2
    + ar_cf4_727_err_ir_Florian_50mbar**2
    + ar_cf4_750_err_ir_Florian_50mbar**2
    + ar_cf4_763_err_ir_Florian_50mbar**2
    + ar_cf4_772_err_ir_Florian_50mbar**2
    + ar_cf4_794_err_ir_Florian_50mbar**2
) * 25e-1
# ============================================================
# 5) PREDICCIÓN IR
# ============================================================
plt.figure(figsize=(6, 4))

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

plt.grid(False)
# plt.grid(True, which='major', alpha=0.3)
# plt.grid(True, which='minor', alpha=0.08)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, 3))

for i, gap in enumerate(gaps):
    mask1 = garfield_data["gap_mm"] == gap
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 =  np.isclose(garfield_data["pressure"],pressure[i],atol=0.05)
    subset = garfield_data[(mask1 & mask2 & mask3)].copy()
  
    yield_teo = theory_yield_ArCF4_Ir_696(parameter_data, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo += theory_yield_ArCF4_Ir_727(parameter_data, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo += theory_yield_ArCF4_Ir_750(parameter_data, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo += theory_yield_ArCF4_Ir_763(parameter_data, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo += theory_yield_ArCF4_Ir_772(parameter_data, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]



    plt.plot(
        fN2 * 100,
        yield_teo,
        color=colors[i],
        label=f"{gap} mm {pressure[i]:.3f} bar"
    )
    
plt.errorbar(
        concentration_cf4,
        ar_cf4_sum_ir_LIP,
        yerr= ar_cf4_sum_err_ir_LIP,
        marker="o",
        linestyle="none",
        ms=5,
        color=colors[0],
        elinewidth=1,
        capsize=2,
        label = "1 bar GEM LIP"
)
label=f"{gap} mm prediction {pressure[i]:.3f} bar"

plt.errorbar(
        [20],
        ar_cf4_sum_ir_Florian_1bar,
        yerr= ar_cf4_sum_err_ir_Florian_1bar,
        marker="o",
        linestyle="none",
        ms=5,
        color=colors[1],
        elinewidth=1,
        capsize=2,
        label = "1 bar th-GEM Florian"
)


plt.errorbar(
        [20],
        ar_cf4_sum_ir_Florian_50mbar,
        yerr= ar_cf4_sum_err_ir_Florian_50mbar,
        marker="o",
        linestyle="none",
        ms=5,
        color=colors[2],
        elinewidth=1,
        capsize=2,
        label = "50 mbar th-GEM Florian"
)

plt.title("NIR (680-800nm) Yield Prediction for Ar/CF mixture")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e0,8e1)
plt.ylim(1e-4,2e0)
plt.ylabel("ph/e$^-$")
plt.xlabel("CF$_4$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_IR_secondary.pdf"))


# ============================================================
# 6) PREDICCIÓN IR per BAND
# ============================================================
