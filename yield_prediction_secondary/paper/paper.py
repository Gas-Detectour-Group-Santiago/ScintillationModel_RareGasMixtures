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
from ArCF4_infrarred import *
from ArCF4 import *

sys.path.append(data_dir)
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


# ============================================================
# RUTAS
# ============================================================
folder_path = "../../data/Secondary_GarfieldData/Paper/root"
table_path = "../../data/Secondary_GarfieldData/levels/ArCF4_level_data.csv"

csv_folder = "../../data/Secondary_GarfieldData/Paper/csv"
populations_dir = "../../data/Secondary_GarfieldData/Paper/populations"
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
    },

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



parameter_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_secondary.csv"))["parameter"].to_numpy()
nor = parameter_data_ArCF4[0]

parameter_data_ArCF4[0] = 1
parameter_data_ArCF4[1] = 0.39
parameter_data_ArCF4[2] = 0.39
parameter_data_ArCF4[5] *= 1 # 50 #
parameter_data_ArCF4[6] *= 1 # 600 # 500 # 50 #
parameter_data_ArCF4[7] *= 0.01
parameter_data_ArCF4[-1] = 0.2 # 1 #

parameter_data_IR = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv"))["parameter"].to_numpy()

# parameter_data_ArCF4 = pd.read_csv(os.path.join(DATA_DIR_PAR, "ArCF4_primary.csv"))["parameter"].to_numpy()


# ============================================================
# 1.3) PREDICCIÓN VIS y VIS+NIR vs ELECTRIC FIELD (1% CF4)
# ============================================================

fCF4_fixed = 0.01   # 1% de CF4

fN2 = np.logspace(-3, 0, 1000)
pressure = [0.050,1,10]
npe = [100,100,100]

norm_factor = nor

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

plt.grid(False)

for i, p in enumerate(pressure):

    mask_p = garfield_data["pressure"] == p
    mask_c = np.isclose(garfield_data["concentration"], fCF4_fixed)

    subset_all = garfield_data[(mask_p & mask_c)].copy()

    print(f"\nPressure = {p} bar")
    print(f"N puntos = {len(subset_all)}")
    print("Gaps disponibles:", np.sort(subset_all["gap_mm"].unique()))

    if subset_all.empty:
        print(f"WARNING: no hay datos para pressure={p} y CF4={fCF4_fixed}")
        continue

    efields = np.sort(subset_all["electric_field"].unique())

    efields_plot = []
    yield_vis_list = []
    yield_visnir_list = []

    for E in efields:

        subset = subset_all[np.isclose(subset_all["electric_field"], E)].copy()

        npes = npe[i]

        yield_teo_IR = (
            theory_yield_ArCF4_Ir_696(parameter_data_IR, subset, fCF4_fixed, p) * 15 / npes / norm_factor
            + theory_yield_ArCF4_Ir_727(parameter_data_IR, subset, fCF4_fixed, p) * 15 / npes / norm_factor
            + theory_yield_ArCF4_Ir_750(parameter_data_IR, subset, fCF4_fixed, p) * 15 / npes / norm_factor
            + theory_yield_ArCF4_Ir_763(parameter_data_IR, subset, fCF4_fixed, p) * 15 / npes / norm_factor
            + theory_yield_ArCF4_Ir_772(parameter_data_IR, subset, fCF4_fixed, p) * 15 / npes / norm_factor
        )

        yield_teo_vis = (
            theory_yield_vis(parameter_data_ArCF4, subset, fCF4_fixed, p)
            * 15 / npes
        )

        yield_teo_IR = float(np.squeeze(np.asarray(yield_teo_IR)))
        yield_teo_vis = float(np.squeeze(np.asarray(yield_teo_vis)))

        efields_plot.append(E)
        yield_vis_list.append(yield_teo_vis)
        yield_visnir_list.append(yield_teo_vis + yield_teo_IR)

    plt.plot(
        efields_plot,
        yield_visnir_list,
        color=colors[i],
        label=f"400-800 nm, 1$\\%$ CF$_4$, {p} bar"
    )

    plt.plot(
        efields_plot,
        yield_vis_list,
        color=colors[i],
        linestyle="--",
        label=f"400-720 nm, 1$\\%$ CF$_4$, {p} bar"
    )

plt.title("Secondary Yield Prediction for Ar/CF$_4$ mixture, 1$\\%$ CF$_4$ \\& 100 Gain")
plt.xlabel("Electric field [kV/cm]")
plt.ylabel("ph/e$^-$")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_gem_vsE_1percentCF4.pdf"))
plt.show()

# ============================================================
# 1.1) MALLA DE CONCENTRACIONES Y CAMPOS
# ===========================================================


cf4_ArCF4_pct = np.array([
    5.0,
    10.0,
    67.0,
])

nph_per_e_ArCF4_gain100 = np.array([
    0.56,   # Ar + 5% CF4, extrapolado
    0.52,   # Ar + 10% CF4, extrapolado suave
    0.30,   # Ar + 67% CF4, prácticamente plateau
])

nph_per_e_ArCF4_gain100_err = np.array([
    0.03,
    0.02,
    0.01,
])

fN2 = np.logspace(-3, 0, 1000)
gaps = [0.05] * 3
npe = [100] * 3
electricField = [0] * 3
pressure = [0.2, 1, 10]

norm =  [nor] * len(gaps)

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

plt.grid(False)
# plt.grid(True, which='major', alpha=0.3)
# plt.grid(True, which='minor', alpha=0.08)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

for i, gap in enumerate(gaps):
    mask1 = garfield_data["gap_mm"] == gap
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = garfield_data["pressure"] == pressure[i]
    subset = garfield_data[(mask1 & mask2 & mask3)].copy()
  
    yield_teo_IR = theory_yield_ArCF4_Ir_696(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_727(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_750(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_763(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_772(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]

    yield_teo_vis = (
        theory_yield_vis(parameter_data_ArCF4, subset, fN2, pressure[i])
        / npe[i]
        * 15
    )

    plt.plot(
        fN2 * 100,
        yield_teo_vis+yield_teo_IR,
        color=colors[i],
        label=f"400-800 nm {pressure[i]} bar"
    )


    plt.plot(
        fN2 * 100,
        yield_teo_vis,
        color=colors[i],
        linestyle="--",
        label=f"400-720 nm {pressure[i]} bar"
    )

plt.errorbar(
        cf4_ArCF4_pct,
        nph_per_e_ArCF4_gain100,
        yerr=nph_per_e_ArCF4_gain100_err,
        color=colors[1],
        linestyle=None,
        label=f"LIP"
    )

plt.title("Secondary Yield Prediction for Ar/CF mixture GEM 100 gain")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
plt.xlabel("CF$_4$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_gem.pdf"))


# ============================================================
# 1.2) MALLA DE CONCENTRACIONES Y CAMPOS
# ============================================================

fN2 = np.logspace(-3, 0, 1000)
gaps = [0.57] * 3
npe = [100] * 3
electricField = [0] * 3
pressure = [0.050,1,10]

norm =  [nor] * len(gaps)

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

plt.grid(False)
# plt.grid(True, which='major', alpha=0.3)
# plt.grid(True, which='minor', alpha=0.08)

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

for i, gap in enumerate(gaps):
    mask1 = garfield_data["gap_mm"] == gap
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = mask2
    if pressure[i] == 100: 
        fN2 = np.logspace(-3, -0.8, 1000)
        mask3 = garfield_data["concentration"] <= 1
    mask4 = garfield_data["pressure"] == pressure[i]
    subset = garfield_data[(mask1 & mask2 & mask3 & mask4)].copy()
  
    yield_teo_IR = theory_yield_ArCF4_Ir_696(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_727(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_750(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_763(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]
    yield_teo_IR += theory_yield_ArCF4_Ir_772(parameter_data_IR, subset, fN2, pressure[i]) * 15 / npe[i] / norm[i]

    yield_teo_vis = (
        theory_yield_vis(parameter_data_ArCF4, subset, fN2, pressure[i])
        / npe[i]
        * 15
    )

    plt.plot(
        fN2 * 100,
        yield_teo_vis+yield_teo_IR,
        color=colors[i],
        label=f"400-800 nm {pressure[i]} bar"
    )


    plt.plot(
        fN2 * 100,
        yield_teo_vis,
        color=colors[i],
        linestyle="--",
        label=f"400-720 nm {pressure[i]} bar"
    )
    
    
plt.title("Secondary Yield Prediction for Ar/CF mixture thGEM 100 gain")
# plt.yscale("log")
plt.xscale("log")
plt.ylabel("ph/e$^-$")
# plt.ylim(0.005,5)
plt.xlabel("CF$_4$ concentration [\%]")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_th_gem.pdf"))
# ============================================================
# FUNCIÓN AUXILIAR PARA CALCULAR ni-/ni+
# Aproximación usada:
#     ni- ≈ ni - ne
#     ni-/ni+ ≈ (ni - ne) / ni
# ============================================================
def compute_ni_minus_over_ni_plus(df, xcol):
    """
    Agrupa por xcol, promedia ne y ni, y calcula:

        ratio = (ni - ne) / ni

    Devuelve:
        xcol, ne, ni, ni_minus_est, ratio
    """

    tmp = df.groupby(xcol, as_index=False)[["ne", "ni"]].mean()

    # evitar división por cero
    tmp = tmp[tmp["ni"] != 0].copy()

    tmp["ni_minus_est"] = tmp["ni"] - tmp["ne"]
    tmp["ratio"] = tmp["ni_minus_est"] / tmp["ni"]

    return tmp.sort_values(xcol)


# ============================================================
# 1) ni-/ni+ vs ELECTRIC FIELD
#    1% CF4
# ============================================================

fCF4_fixed = 0.01   # 1% de CF4
pressure = [0.050, 1, 10]

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure)))

plt.grid(False)

for i, p in enumerate(pressure):

    mask_p = np.isclose(garfield_data["pressure"], p)
    mask_c = np.isclose(garfield_data["concentration"], fCF4_fixed)

    subset_all = garfield_data[(mask_p & mask_c)].copy()

    print(f"\n[ni-/ni+ vs E] Pressure = {p} bar")
    print(f"N puntos = {len(subset_all)}")

    if subset_all.empty:
        print(f"WARNING: no hay datos para pressure={p} y CF4={fCF4_fixed}")
        continue

    ratio_data = compute_ni_minus_over_ni_plus(subset_all, "electric_field")

    if ratio_data.empty:
        print(f"WARNING: no hay datos válidos de ni-/ni+ para pressure={p}")
        continue

    plt.plot(
        ratio_data["electric_field"],
        ratio_data["ratio"],
        color=colors[i],
        marker="o",
        label=f"1$\\%$ CF$_4$, {p} bar"
    )

plt.title("Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$, 1$\\%$ CF$_4$")
plt.xlabel("Electric field [kV/cm]")
plt.ylabel("$n_{i^-}/n_{i^+}$")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_niMinus_over_niPlus_vsE_1percentCF4.pdf"))
plt.show()


# ============================================================
# 2) ni-/ni+ vs CF4 concentration
#    GEM, gap = 0.05 mm
# ============================================================

gaps = [0.05] * 3
pressure = [0.2, 1, 10]
electricField = [0] * 3

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

plt.grid(False)

for i, gap in enumerate(gaps):

    mask1 = np.isclose(garfield_data["gap_mm"], gap)
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = np.isclose(garfield_data["pressure"], pressure[i])

    subset = garfield_data[(mask1 & mask2 & mask3)].copy()

    print(f"\n[ni-/ni+ GEM] gap = {gap} mm, pressure = {pressure[i]} bar")
    print(f"N puntos = {len(subset)}")

    if subset.empty:
        print(f"WARNING: no hay datos para gap={gap} y pressure={pressure[i]}")
        continue

    ratio_data = compute_ni_minus_over_ni_plus(subset, "concentration")

    if ratio_data.empty:
        print(f"WARNING: no hay datos válidos de ni-/ni+ para pressure={pressure[i]}")
        continue

    plt.plot(
        ratio_data["concentration"] * 100,
        ratio_data["ratio"],
        color=colors[i],
        marker="o",
        label=f"{pressure[i]} bar"
    )

plt.title("Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$ GEM")
plt.xscale("log")
plt.xlabel("CF$_4$ concentration [\\%]")
plt.ylabel("$n_{i^-}/n_{i^+}$")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_gem_niMinus_over_niPlus.pdf"))
plt.show()


# ============================================================
# 3) ni-/ni+ vs CF4 concentration
#    thGEM, gap = 0.57 mm
# ============================================================

gaps = [0.57] * 3
pressure = [0.050, 1, 10]
electricField = [0] * 3

plt.figure(figsize=(6, 4))
plt.style.use("science")

cmap = "viridis"
cmap_obj = plt.get_cmap(cmap)
colors = cmap_obj(np.linspace(0.15, 0.85, len(gaps)))

plt.grid(False)

for i, gap in enumerate(gaps):

    mask1 = np.isclose(garfield_data["gap_mm"], gap)
    mask2 = garfield_data["electric_field"] > electricField[i]
    mask3 = np.isclose(garfield_data["pressure"], pressure[i])

    subset = garfield_data[(mask1 & mask2 & mask3)].copy()

    print(f"\n[ni-/ni+ thGEM] gap = {gap} mm, pressure = {pressure[i]} bar")
    print(f"N puntos = {len(subset)}")

    if subset.empty:
        print(f"WARNING: no hay datos para gap={gap} y pressure={pressure[i]}")
        continue

    ratio_data = compute_ni_minus_over_ni_plus(subset, "concentration")

    if ratio_data.empty:
        print(f"WARNING: no hay datos válidos de ni-/ni+ para pressure={pressure[i]}")
        continue

    plt.plot(
        ratio_data["concentration"] * 100,
        ratio_data["ratio"],
        color=colors[i],
        marker="o",
        label=f"{pressure[i]} bar"
    )

plt.title("Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$ thGEM")
plt.xscale("log")
plt.xlabel("CF$_4$ concentration [\\%]")
plt.ylabel("$n_{i^-}/n_{i^+}$")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ArCF4_thgem_niMinus_over_niPlus.pdf"))
plt.show()


# ============================================================
# 4) OPCIONAL: guardar también la columna en garfield_data
# ============================================================

garfield_data["ni_minus_est"] = garfield_data["ni"] - garfield_data["ne"]
garfield_data["niMinus_over_niPlus"] = garfield_data["ni_minus_est"] / garfield_data["ni"]