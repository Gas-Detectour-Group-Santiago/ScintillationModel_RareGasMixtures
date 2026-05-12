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
folder_path = "../../data/Secondary_GarfieldData/CF4/root"
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
    output_general_name=os.path.join(populations_dir, "ArCF4_secondary"),
    gas_concentration="cf4",
    gain_summary=summary,
    normalized="ni"
)

# ============================================================
# 6) CARGA DE DATOS PARA EL MODELO
# ============================================================
DATA_DIR_EXP = "../../data/Experimental/ArCF4/"
DATA_DIR_GARFIELD = populations_dir
DATA_DIR_PAR = "../../data/Parameters"

garfield_data = pd.read_csv(os.path.join(DATA_DIR_GARFIELD, "ArCF4_secondary.csv"))
garfield_data["concentration"] = garfield_data["concentration"] / 100.0



parameter_data = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_secondary.csv"))["parameter"].to_numpy()

print("parameter_data original:")
print(parameter_data)

parameter_data[0] = 1
parameter_data[1] = 0.39
parameter_data[2] = 0.39
parameter_data[5] *= 1 # 50 #
parameter_data[6] *= 1 # 600 # 500 # 50 #
parameter_data[7] *= 0.01
parameter_data[-1] = 0.2 # 1 #

print("parameter_data modificado:")
print(parameter_data)


# ============================================================
# 7) MALLA DE CONCENTRACIONES Y CAMPOS
# ============================================================
fN2 = np.logspace(-3, 0, 1000)
gaps = [0.05, 0.57, 0.57, 0.57]
npe = [1000, 50, 50, 50]
electricField = [60, 0, 0, 0]
pressure = [1, 0.050, 0.025, 1]
# ============================================================
# 10) YIELD UV FRENTE A PRESIÓN PARA CF4 PURO
# ============================================================

yields_pure_CF4 = {
    0.025: 0.1367619130620444,
    0.05: 0.07782489159147364,
    0.1: 0.05014074226650811,
    0.2: 0.03902250674735871,
    0.5: 0.03591879466086511,
    0.8: 0.03243891687489655,
    1.0: 0.03942121448304051
}

# Presiones experimentales ordenadas
pressures_exp = np.array(sorted(yields_pure_CF4.keys()), dtype=float)
yield_exp = np.array([yields_pure_CF4[p] for p in pressures_exp])

# Si quieres mantener el mismo criterio de incertidumbre del 25 %
yield_exp_err = 0.25 * yield_exp


# ------------------------------------------------------------
# Configuración del cálculo teórico
# ------------------------------------------------------------
# CF4 puro: concentración = 100 % = 1.0
fCF4_pure = np.array([1.0])

# Uso el caso que ya tenías asociado al punto de CF4 puro a 1 bar:
# gap = 0.05 mm, presión = 1 bar, npe = 1000
gap_pressure_scan = 0.05
npe_pressure_scan = 1000
electric_field_min_pressure_scan = 60

par_7_og = parameter_data[7]

pressures_theory = []
yield_theory = []

for p in pressures_exp:

    mask_gap = garfield_data["gap_mm"] == gap_pressure_scan

    # Mejor no usar atol=0.026 aquí, porque 0.025 y 0.05 bar se pueden mezclar.
    mask_pressure = np.isclose(
        garfield_data["pressure"],
        p,
        rtol=1e-3,
        atol=1e-5
    )

    mask_field = garfield_data["electric_field"] > electric_field_min_pressure_scan

    subset = garfield_data[(mask_gap & mask_pressure & mask_field)].copy()

    if subset.empty:
        print(f"[AVISO] No hay datos Garfield para p = {p} bar, gap = {gap_pressure_scan} mm")
        continue

    # IMPORTANTE:
    # pressure entra como escalar, no como array.
    parameter_data[7] = par_7_og / p

    y_uv = (
        theory_yield_uv(parameter_data, subset, fCF4_pure, p)
        / npe_pressure_scan
        * 15
    )

    # theory_yield_uv devuelve array porque fCF4_pure es array.
    # Como solo tiene un punto, tomamos el primero.
    pressures_theory.append(p)
    yield_theory.append(np.asarray(y_uv).ravel()[0])


# Restauramos el parámetro original
parameter_data[7] = par_7_og

pressures_theory = np.array(pressures_theory)
yield_theory = np.array(yield_theory)


# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(6, 4))
plt.style.use("grid")
plt.grid(True, which="major", alpha=0.3)
plt.grid(True, which="minor", alpha=0.08)

plt.errorbar(
    pressures_exp,
    yield_exp,
    yerr=yield_exp_err,
    fmt=".",
    capsize=3,
    label="Experimental CF$_4$ puro"
)

plt.plot(
    pressures_theory,
    yield_theory,
    marker="o",
    label="Predicción UV CF$_4$ puro"
)

plt.xscale("log")
plt.xlabel("Pressure [bar]")
plt.ylabel("UV yield [ph/e$^-$]")
plt.title("UV yield vs pressure for pure CF$_4$")

plt.legend(loc="best")
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, "ArCF4_uv_yield_vs_pressure_CF4_pure.pdf"))
plt.show()