#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("grid")

# ============================================================
# PATHS
# Guarda este script en la misma carpeta desde la que ejecutas
# ArCF4_secondary_threshold.py y HeCF4_secondary_threshold.py
# ============================================================
BASE_DIR = os.path.dirname(__file__)

models_dir = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../../data"))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4 import theory_yield_vis, theory_yield_uv
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


# ============================================================
# CONFIG
# ============================================================
plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

output_pdf = os.path.join(
    plots_dir,
    "LIP_UV.pdf"
)

required_cols = [
    "concentration", "gap_mm", "electric_field", "pressure",
    "CF3", "Ar_dbleStar", "CF4", "Ar_3rd",
]

fCF4_grid = np.logspace(-3, 0, 1000)

LIP_GAP_MM = 0.05
LIP_PRESSURE_BAR = 1.0
LIP_NPE = 1000
ENERGY_FACTOR = 15.0


thresholds = pd.read_csv("../../data/Thresholds/ArCF4.csv")

# Casos que quieres comparar.
# 1_0 y 2_0 se usan como los límites de la banda.
# 1_3 o 1_5 se usan como curva central "optimal fit".
CASE_CONFIGS = {
    "1_0": {
        "EAr**": float(thresholds["E_th_Ar**"]),
        "ECF3": float(thresholds["E_th_CF3"]),
        "csv": "ArCF4_secondary_1_0.csv",
        "PCF3*": 1,
        "PAr**": 1,
        "PCF4": 1/7,
    },
    "1_3": {
        "EAr**": float(thresholds["E_th_Ar**"]),
        "ECF3": float(thresholds["E_th_CF3"]),
        "csv": "ArCF4_secondary_1_3.csv",
        "PCF3*": 1,
        "PAr**": 1,
        "PCF4": 1/7,
    },
    "1_5": {
        "EAr**": float(thresholds["E_th_Ar**"]),
        "ECF3": float(thresholds["E_th_CF3"]),
        "csv": "ArCF4_secondary_1_5.csv",
        "PCF3*": 1,
        "PAr**": 1,
        "PCF4": 1/7,
    },
    "2_0": {
        "EAr**": float(thresholds["E_th_Ar**"]),
        "ECF3": float(thresholds["E_th_CF3"]),
        "csv": "ArCF4_secondary_2_0.csv",
        "PCF3*": 1,
        "PAr**": 1,
        "PCF4": 1,
    },
}

BAND_LOW_CASE = "1_0"
BAND_HIGH_CASE = "2_0"

# Elige una de estas dos líneas y comenta la otra:
OPTIMAL_CASE = "1_3"
# OPTIMAL_CASE = "1_5"


# ============================================================
# DATOS EXPERIMENTALES LIP, GEM 0.050 mm
# ============================================================
# ---------------- Ar-CF4 ----------------
ar_cf4_conc = np.array([100, 67, 10, 5], dtype=float)

ar_cf4_vis = np.array([
    0.09335376,
    0.2802068,
    0.38966203,
    0.38287151,
], dtype=float)

ar_cf4_vis_err = np.array([
    0.1,
    0.3,
    0.39,
    0.38,
], dtype=float) * 0.25

ar_cf4_uv = np.array([
    0.03942121448304051,
    0.044033601820777875,
    0.08455236611728804,
    0.06737771706391489,
], dtype=float)

ar_cf4_uv_err = np.array([
    0.04,
    0.045,
    0.085,
    0.068,
], dtype=float) * 0.25


# ---------------- He-CF4 ----------------
he_cf4_conc = np.array([20, 40, 100], dtype=float)

he_cf4_vis = np.array([
    0.05981728,
    0.0633149,
    0.09335376,
], dtype=float)

he_cf4_vis_err = np.array([
    0.016221195011451566,
    0.023434172057910808,
    0.02018412139031937,
], dtype=float)

he_cf4_uv = np.array([
    0.0586689111317149,
    0.12606696047521188,
    0.03942121448304051,
], dtype=float)

he_cf4_uv_err = np.array([
    0.014241770732051665,
    0.04213797898747448,
    0.00859011077564899,
], dtype=float)


# ============================================================
# GARFIELD POPULATIONS
# ============================================================
def secondary_population_config(EAr,ECF3):
    """
    Misma configuración de poblaciones que en tus scripts secundarios.
    Las columnas generadas tienen que incluir:
        CF3, Ar_dbleStar, CF4, Ar_3rd
    """
    return pd.DataFrame({
        "CF4": {
            "name principal": "ION",
            "gas": "CF4",
            "energy low": 15.5,
            "energy up": 16,
            "name output": "CF4",
            "type": "ionisation",
        },
        "Ar**": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": EAr * 0.99,
            "energy up": 100,
            "name output": "Ar_dbleStar",
            "type": "excitation",
        },
        "CF3": {
            "name principal": "NEUTRAL DISS",
            "gas": "CF4",
            "energy low": ECF3 * 0.99,
            "energy up": 100,
            "name output": "CF3",
            "type": "inelastic",
        },
        "Ar3rd": {
            "name principal": "IONISATION",
            "gas": "Ar",
            "energy low": 40,
            "energy up": 120,
            "name output": "Ar_3rd",
            "type": "ionisation",
        },
    })


def build_secondary_population(gas_name, case_name, EAr = 12.9, ECF3=15.88):
    """
    Genera y lee el csv de poblaciones secundarias para un gas y un caso.
    Se añade case_name al nombre de salida para no sobrescribir los cuatro casos.
    """
    folder_path = os.path.join(
        data_dir,
        "Secondary_GarfieldData",
        gas_name,
        "root",
    )

    table_path = os.path.join(
        data_dir,
        "Secondary_GarfieldData",
        "levels",
        f"{gas_name}_level_data.csv",
    )

    csv_folder = os.path.join(
        data_dir,
        "Secondary_GarfieldData",
        gas_name,
        "csv",
    )

    populations_dir = os.path.join(
        data_dir,
        "Secondary_GarfieldData",
        gas_name,
        "populations",
    )

    os.makedirs(populations_dir, exist_ok=True)

    output_general_name = os.path.join(
        populations_dir,
        f"{gas_name}_secondary_{case_name}",
    )

    output_csv = output_general_name + ".csv"

    export_hlevels_to_csv(
        folder_path,
        table_path,
        object_name="hLevels",
        argon_update=True,
    )

    summary = read_data_per_primary_electron(
        folder_path,
        gas_concentration="cf4",
    )

    read_garfield_csv_folder(
        folder_path=csv_folder,
        dataframe=secondary_population_config(EAr = EAr, ECF3=ECF3),
        output_dir=populations_dir,
        output_general_name=output_general_name,
        gas_concentration="cf4",
        gain_summary=summary,
        normalized="ni",
    )

    df = pd.read_csv(output_csv)
    df["concentration"] = df["concentration"] / 100.0

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{gas_name}, case {case_name}: faltan columnas {missing} en {output_csv}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    return df


def select_lip_subset(df, gas_name, case_name, efield_min):
    """
    Selecciona la condición LIP usada para las curvas teóricas:
        GEM 0.050 mm, 1 bar, campo por encima del umbral usado en tus scripts.
    """
    mask = (
        np.isclose(df["gap_mm"], LIP_GAP_MM)
        & (df["electric_field"] > efield_min)
        & np.isclose(df["pressure"], LIP_PRESSURE_BAR, atol=0.026)
    )

    subset = df[mask].copy()

    if subset.empty:
        raise ValueError(
            f"{gas_name}, case {case_name}: subset vacío para gap={LIP_GAP_MM} mm, "
            f"E > {efield_min}, p={LIP_PRESSURE_BAR} bar.\n"
            f"Valores gap disponibles: {np.sort(df['gap_mm'].unique())}\n"
            f"Valores pressure disponibles: {np.sort(df['pressure'].unique())}\n"
            f"Rango E disponible: {df['electric_field'].min()} - {df['electric_field'].max()}"
        )

    return subset


# ============================================================
# PARAMETERS
# ============================================================
def load_parameters(csv_name, PCF3, PAr, PCF4):
    par = pd.read_csv(
        os.path.join(data_dir, "Parameters", csv_name)
    )["parameter"].to_numpy(dtype=float)

    par[0] = 1
    par[1] *= PCF3
    par[2] *= PAr
    par[7] *= PCF4

    print(par)
    return par


def compute_theory(par, subset, pressure):
    y_vis = (
        theory_yield_vis(par, subset, fCF4_grid, pressure)
        / LIP_NPE
        * ENERGY_FACTOR
    )

    y_uv_raw = (
        theory_yield_uv(par, subset, fCF4_grid, pressure)
        / LIP_NPE
        * ENERGY_FACTOR
    )

    y_uv = y_uv_raw 

    return np.asarray(y_vis, dtype=float), np.asarray(y_uv, dtype=float)


def build_theory_family(gas_name, efield_min):
    """
    Devuelve un diccionario del tipo:

        theory[case_name]["VIS"]
        theory[case_name]["UV"]

    con case_name = 1_0, 1_3, 1_5, 2_0.
    """
    theory = {}

    for case_name, cfg in CASE_CONFIGS.items():
        df = build_secondary_population(
            gas_name=gas_name,
            case_name=case_name,
            EAr=cfg["EAr**"],
            ECF3=cfg["ECF3"],
        )

        subset = select_lip_subset(
            df=df,
            gas_name=gas_name,
            case_name=case_name,
            efield_min=efield_min,
        )

        par = load_parameters(
            csv_name=cfg["csv"],
            PCF3=cfg["PCF3*"],
            PAr=cfg["PAr**"],
            PCF4=cfg["PCF4"],
        )

        y_vis, y_uv = compute_theory(
            par=par,
            subset=subset,
            pressure=LIP_PRESSURE_BAR,
        )

        theory[case_name] = {
            "VIS": y_vis,
            "UV": y_uv,
        }

    return theory


def positive_array(y):
    """Evita problemas en escala log si aparece algún cero, negativo o NaN."""
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y) & (y > 0), y, np.nan)


def band_limits(y1, y2):
    y1 = positive_array(y1)
    y2 = positive_array(y2)
    y_low = np.fmin(y1, y2)
    y_high = np.fmax(y1, y2)
    return y_low, y_high


def pretty_case(case_name):
    return case_name.replace("_", ".")


# ============================================================
# BUILD DATA + THEORY
# ============================================================
ar_theory = build_theory_family(
    gas_name="ArCF4",
    efield_min=60.0,
)

he_theory = build_theory_family(
    gas_name="HeCF4",
    efield_min=55.0,
)


# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 5.0))


cmap_obj = plt.get_cmap("viridis")
colors = cmap_obj(np.linspace(0.12, 0.88, 4))

x_theory = fCF4_grid * 100.0

series = [
    {
        "name": r"Ar-CF$_4$ VIS",
        "gas_theory": ar_theory,
        "channel": "VIS",
        "color": colors[0],
        "marker": "o",
        "linestyle": "-",
        "xexp": ar_cf4_conc,
        "yexp": ar_cf4_vis,
        "yerr": ar_cf4_vis_err,
    },
    {
        "name": r"Ar-CF$_4$ UV",
        "gas_theory": ar_theory,
        "channel": "UV",
        "color": colors[1],
        "marker": "s",
        "linestyle": "--",
        "xexp": ar_cf4_conc,
        "yexp": ar_cf4_uv,
        "yerr": ar_cf4_uv_err,
    },
    {
        "name": r"He-CF$_4$ VIS",
        "gas_theory": he_theory,
        "channel": "VIS",
        "color": colors[2],
        "marker": "^",
        "linestyle": "-.",
        "xexp": he_cf4_conc,
        "yexp": he_cf4_vis,
        "yerr": he_cf4_vis_err,
    },
    {
        "name": r"He-CF$_4$ UV",
        "gas_theory": he_theory,
        "channel": "UV",
        "color": colors[3],
        "marker": "D",
        "linestyle": ":",
        "xexp": he_cf4_conc,
        "yexp": he_cf4_uv,
        "yerr": he_cf4_uv_err,
    },
]


for i in [1,3]:
    s = series[i]
    channel = s["channel"]
    theory = s["gas_theory"]

    y_low, y_high = band_limits(
        theory[BAND_LOW_CASE][channel],
        theory[BAND_HIGH_CASE][channel],
    )

    y_opt = positive_array(theory[OPTIMAL_CASE][channel])

    # Banda: 1_0 -- 2_0
    ax.fill_between(
        x_theory,
        y_low,
        y_high,
        color=s["color"],
        alpha=0.16,
        linewidth=0.0,
        label=(
            s["name"]
            + rf" range {pretty_case(BAND_LOW_CASE)}--{pretty_case(BAND_HIGH_CASE)}"
        ),
    )

    # Bordes suaves de la banda, por si quieres ver exactamente los límites.
    ax.plot(
        x_theory,
        positive_array(theory[BAND_LOW_CASE][channel]),
        color=s["color"],
        linestyle=":",
        lw=0.9,
        alpha=0.45,
    )
    ax.plot(
        x_theory,
        positive_array(theory[BAND_HIGH_CASE][channel]),
        color=s["color"],
        linestyle=":",
        lw=0.9,
        alpha=0.45,
    )

    # Curva central: cambia OPTIMAL_CASE arriba entre 1_3 y 1_5.
    ax.plot(
        x_theory,
        y_opt,
        color=s["color"],
        linestyle=s["linestyle"],
        lw=2.4,
        label=s["name"] + rf" optimal fit {pretty_case(OPTIMAL_CASE)}",
    )

    ax.errorbar(
        s["xexp"],
        s["yexp"],
        yerr=s["yerr"],
        fmt=s["marker"],
        ms=5.5,
        color=s["color"],
        ecolor=s["color"],
        capsize=3,
        elinewidth=1.1,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linestyle="none",
        label=s["name"] + " exp. LIP, GEM 0.050 mm",
    )


# ax.grid(True, which="major", alpha=0.3)
# ax.grid(True, which="minor", alpha=0.08)
ax.grid(False)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"CF$_4$ concentration [\%]")
ax.set_ylabel(r"ph/e$^-$")
ax.set_title(r"LIP GEM 0.050 mm: Ar-CF$_4$ and He-CF$_4$ UV")

ax.set_xlim(4, 120)
ax.set_ylim(5e-3, 0.6)

ax.legend(loc="lower left", fontsize=7.2, frameon=True, ncol=2)

plt.tight_layout()
plt.savefig(output_pdf, bbox_inches="tight")
print(f"Saved: {output_pdf}")
plt.show()
