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
    "HeCF4_VIS_comparation.pdf"
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

normalization = "ne"

# ============================================================
# DATOS EXPERIMENTALES LIP, GEM 0.050 mm
# ============================================================
# ---------------- Ar-CF4 ----------------

type_of_data = "normalized"

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

#
rate = 2.42
#

he_cf4_conc_1bar_Florian = np.array([100, 20], dtype=float)
he_cf4_vis_1bar_Florian = np.array([0.3542 * 187/530,  
                                    0.3542 * 45/530,
                                    # 0.3542 * 530/530
                                ]
                                ,dtype=float) * rate
he_cf4_vis_err_1bar_Florian = np.array([0.3542 * 187/530,
                                        0.3542 * 45/530,
                                ],dtype=float) * rate * 0.25


he_cf4_conc_300mbar_Florian = np.array([100, 20], dtype=float)
he_cf4_vis_300mbar_Florian = np.array([0.3542 * 100/530,  
                                       0.3542 * 45/530,
                                ]
                                ,dtype=float) * rate
he_cf4_vis_err_300mbar_Florian = np.array([0.3542 * 100/530,  
                                       0.3542 * 45/530,
                                ],dtype=float) * rate * 0.25

if type_of_data == "normalized":
    # ============================================================
    # RATE
    # ============================================================
    rate = 2.42


    # ============================================================
    # Ar-CF4 LIP, GEM 0.050 mm, 1 bar
    # ============================================================

    ar_cf4_conc = np.array([5, 10, 67, 100], dtype=float)

    ar_cf4_efield_LIP = np.array([65, 78, 88, 95], dtype=float)

    ar_cf4_vis = np.array([
        0.38287151,
        0.38966203,
        0.2802068,
        0.09335376,
    ], dtype=float)

    ar_cf4_vis_err = np.array([
        0.11515380887765425,
        0.1103105398012753,
        0.06028075287069421,
        0.02004987990487861,
    ], dtype=float)


    ar_cf4_uv = np.array([
        0.06737771706391489,
        0.08455236611728804,
        0.044033601820777875,
        0.03942121448304051,
    ], dtype=float)

    ar_cf4_uv_err = np.array([
        0.013490812870036886,
        0.016994479521607656,
        0.00884331537513073,
        0.00859011077564899,
    ], dtype=float)


    # ============================================================
    # Ar-CF4 LIP IR, GEM 0.050 mm, 1 bar
    # ============================================================

    ar_cf4_conc_ir_LIP = np.array([5, 10, 67], dtype=float)

    ar_cf4_696_ir_LIP = np.array([
        0.0007423954814679462,
        0.00016606742822049213,
        5.7976067957635574e-05,
    ], dtype=float)

    ar_cf4_696_err_ir_LIP = np.array([
        0.00014864734147261504,
        3.337848055232784e-05,
        1.1643395769579259e-05,
    ], dtype=float)


    ar_cf4_727_ir_LIP = np.array([
        0.0,
        0.00023289581724378054,
        0.00013834145604106908,
    ], dtype=float)

    ar_cf4_727_err_ir_LIP = np.array([
        0.0,
        4.681055514551999e-05,
        2.778326265939665e-05,
    ], dtype=float)


    ar_cf4_750_ir_LIP = np.array([
        0.030352019422880832,
        0.01646440994283698,
        0.004112455696967686,
    ], dtype=float)

    ar_cf4_750_err_ir_LIP = np.array([
        0.006077282402925308,
        0.003309240065745319,
        0.0008259088784642111,
    ], dtype=float)


    ar_cf4_763_ir_LIP = np.array([
        0.028478502055388058,
        0.010520420255841482,
        0.0009514346999852515,
    ], dtype=float)

    ar_cf4_763_err_ir_LIP = np.array([
        0.0057021543440504,
        0.0021145365269683238,
        0.00019107764895222087,
    ], dtype=float)


    ar_cf4_772_ir_LIP = np.array([
        0.009542209996882418,
        0.00406238907466698,
        0.0003514776387269656,
    ], dtype=float)

    ar_cf4_772_err_ir_LIP = np.array([
        0.0019106045001854228,
        0.0008165139677163302,
        7.058763031059062e-05,
    ], dtype=float)


    ar_cf4_794_ir_LIP = np.array([
        0.0032255060792284705,
        0.0,
        0.00010916911326505648,
    ], dtype=float)

    ar_cf4_794_err_ir_LIP = np.array([
        0.0006458321953051536,
        0.0,
        2.1924549841632906e-05,
    ], dtype=float)


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


    # ============================================================
    # Ar-CF4 Florian, th-GEM, 1 bar
    # ============================================================

    ar_cf4_conc_1bar_Florian = np.array([20, 100], dtype=float)

    ar_cf4_efield_1bar_Florian = np.array([29, 43], dtype=float)

    ar_cf4_vis_1bar_Florian = np.array([
        0.3542035420581855,
        0.09335376,
    ], dtype=float) * rate

    ar_cf4_vis_err_1bar_Florian = np.array([
        0.07607389619126878,
        0.02004987990487861,
    ], dtype=float) * rate


    ar_cf4_uv_1bar_Florian = np.array([
        0.04599727051007351,
        0.03942121448304051,
    ], dtype=float) * rate

    ar_cf4_uv_err_1bar_Florian = np.array([
        0.010023071441114785,
        0.00859011077564899,
    ], dtype=float) * rate


    # ============================================================
    # Ar-CF4 Florian, th-GEM, 50 mbar
    # ============================================================

    ar_cf4_conc_50mbar_Florian = np.array([20], dtype=float)

    ar_cf4_efield_50mbar_Florian = np.array([8.421], dtype=float)

    ar_cf4_vis_50mbar_Florian = np.array([
        0.1057123306627957,
    ], dtype=float) * rate

    ar_cf4_vis_err_50mbar_Florian = np.array([
        0.022704169293163653,
    ], dtype=float) * rate


    ar_cf4_uv_50mbar_Florian = np.array([
        0.02769685258037305,
    ], dtype=float) * rate

    ar_cf4_uv_err_50mbar_Florian = np.array([
        0.0060353044654314165,
    ], dtype=float) * rate


    # ============================================================
    # Ar-CF4 Florian, th-GEM, 25 mbar, pure CF4
    # ============================================================

    ar_cf4_conc_25mbar_Florian = np.array([100], dtype=float)

    ar_cf4_efield_25mbar_Florian = np.array([10], dtype=float)

    ar_cf4_vis_25mbar_Florian = np.array([
        0.04953590534192422,
    ], dtype=float) * rate

    ar_cf4_vis_err_25mbar_Florian = np.array([
        0.010638981979885468,
    ], dtype=float) * rate


    ar_cf4_uv_25mbar_Florian = np.array([
        0.1367619130620444,
    ], dtype=float) * rate

    ar_cf4_uv_err_25mbar_Florian = np.array([
        0.029801212329418493,
    ], dtype=float) * rate


    # ============================================================
    # Ar-CF4 Florian IR, th-GEM, 1 bar
    # ============================================================

    ar_cf4_conc_ir_1bar_Florian = np.array([20], dtype=float)

    ar_cf4_696_ir_Florian_1bar = np.array([
        0.0012121110778869518,
    ], dtype=float) * rate

    ar_cf4_696_err_ir_Florian_1bar = np.array([
        0.00026032890460820274,
    ], dtype=float) * rate


    ar_cf4_727_ir_Florian_1bar = np.array([
        0.0009963113528044523,
    ], dtype=float) * rate

    ar_cf4_727_err_ir_Florian_1bar = np.array([
        0.0002139809196170797,
    ], dtype=float) * rate


    ar_cf4_750_ir_Florian_1bar = np.array([
        0.010641617291858389,
    ], dtype=float) * rate

    ar_cf4_750_err_ir_Florian_1bar = np.array([
        0.0022855335813600896,
    ], dtype=float) * rate


    ar_cf4_763_ir_Florian_1bar = np.array([
        0.005277639438703508,
    ], dtype=float) * rate

    ar_cf4_763_err_ir_Florian_1bar = np.array([
        0.0011334952044080517,
    ], dtype=float) * rate


    ar_cf4_772_ir_Florian_1bar = np.array([
        0.0018895075477094276,
    ], dtype=float) * rate

    ar_cf4_772_err_ir_Florian_1bar = np.array([
        0.0004058154727878853,
    ], dtype=float) * rate


    ar_cf4_794_ir_Florian_1bar = np.array([
        0.0,
    ], dtype=float) * rate

    ar_cf4_794_err_ir_Florian_1bar = np.array([
        0.0,
    ], dtype=float) * rate


    ar_cf4_sum_ir_Florian_1bar = (
        ar_cf4_696_ir_Florian_1bar
        + ar_cf4_727_ir_Florian_1bar
        + ar_cf4_750_ir_Florian_1bar
        + ar_cf4_763_ir_Florian_1bar
        + ar_cf4_772_ir_Florian_1bar
        + ar_cf4_794_ir_Florian_1bar
    )

    ar_cf4_sum_err_ir_Florian_1bar = np.sqrt(
        ar_cf4_696_err_ir_Florian_1bar**2
        + ar_cf4_727_err_ir_Florian_1bar**2
        + ar_cf4_750_err_ir_Florian_1bar**2
        + ar_cf4_763_err_ir_Florian_1bar**2
        + ar_cf4_772_err_ir_Florian_1bar**2
        + ar_cf4_794_err_ir_Florian_1bar**2
    )


    # ============================================================
    # Ar-CF4 Florian IR, th-GEM, 50 mbar
    # ============================================================

    ar_cf4_conc_ir_50mbar_Florian = np.array([20], dtype=float)

    ar_cf4_696_ir_Florian_50mbar = np.array([
        0.002246119186430594,
    ], dtype=float) * rate

    ar_cf4_696_err_ir_Florian_50mbar = np.array([
        0.00048240607489727035,
    ], dtype=float) * rate


    ar_cf4_727_ir_Florian_50mbar = np.array([
        0.0005887938444242574,
    ], dtype=float) * rate

    ar_cf4_727_err_ir_Florian_50mbar = np.array([
        0.0001264571039365711,
    ], dtype=float) * rate


    ar_cf4_750_ir_Florian_50mbar = np.array([
        0.057088337541668865,
    ], dtype=float) * rate

    ar_cf4_750_err_ir_Florian_50mbar = np.array([
        0.012261041623375114,
    ], dtype=float) * rate


    ar_cf4_763_ir_Florian_50mbar = np.array([
        0.02364076551779444,
    ], dtype=float) * rate

    ar_cf4_763_err_ir_Florian_50mbar = np.array([
        0.005077401488711407,
    ], dtype=float) * rate


    ar_cf4_772_ir_Florian_50mbar = np.array([
        0.00668799699933377,
    ], dtype=float) * rate

    ar_cf4_772_err_ir_Florian_50mbar = np.array([
        0.0014364021289985191,
    ], dtype=float) * rate


    ar_cf4_794_ir_Florian_50mbar = np.array([
        0.004140203178868282,
    ], dtype=float) * rate

    ar_cf4_794_err_ir_Florian_50mbar = np.array([
        0.0008892044450984726,
    ], dtype=float) * rate


    ar_cf4_sum_ir_Florian_50mbar = (
        ar_cf4_696_ir_Florian_50mbar
        + ar_cf4_727_ir_Florian_50mbar
        + ar_cf4_750_ir_Florian_50mbar
        + ar_cf4_763_ir_Florian_50mbar
        + ar_cf4_772_ir_Florian_50mbar
        + ar_cf4_794_ir_Florian_50mbar
    )

    ar_cf4_sum_err_ir_Florian_50mbar = np.sqrt(
        ar_cf4_696_err_ir_Florian_50mbar**2
        + ar_cf4_727_err_ir_Florian_50mbar**2
        + ar_cf4_750_err_ir_Florian_50mbar**2
        + ar_cf4_763_err_ir_Florian_50mbar**2
        + ar_cf4_772_err_ir_Florian_50mbar**2
        + ar_cf4_794_err_ir_Florian_50mbar**2
    )


    # ============================================================
    # He-CF4 LIP, GEM 0.050 mm, 1 bar
    # ============================================================

    he_cf4_conc = np.array([20, 40, 100], dtype=float)

    he_cf4_efield_LIP = np.array([60, 75, 95], dtype=float)

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
    # He-CF4 Florian, th-GEM, 300 mbar
    # ============================================================

    he_cf4_conc_300mbar_Florian = np.array([20], dtype=float)

    he_cf4_efield_300mbar_Florian = np.array([12.105], dtype=float)

    he_cf4_vis_300mbar_Florian = np.array([
        0.03459508998978698,
    ], dtype=float) * rate

    he_cf4_vis_err_300mbar_Florian = np.array([
        0.007479843426044407,
    ], dtype=float) * rate


    he_cf4_uv_300mbar_Florian = np.array([
        0.0709633243946791,
    ], dtype=float) * rate

    he_cf4_uv_err_300mbar_Florian = np.array([
        0.015463319066967298,
    ], dtype=float) * rate    


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
        normalized=normalization,
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


def select_lip_subset(df, gas_name, case_name, pressure, efield_min, efield_max, gap):
    """
    Selecciona la condición LIP usada para las curvas teóricas:
        GEM 0.050 mm, 1 bar, campo por encima del umbral usado en tus scripts.
    """
    mask = (
        np.isclose(df["gap_mm"], gap,  atol=0.005)
        & (df["electric_field"] < efield_max)
        & (df["electric_field"] > efield_min)
        & np.isclose(df["pressure"], pressure, atol=0.026)
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
    
    print(subset)

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

    return par


def compute_theory(par, subset, pressure, npe):
    y_vis = (
        theory_yield_vis(par, subset, fCF4_grid, pressure)
        / npe
        * ENERGY_FACTOR
    )

    y_uv_raw = (
        theory_yield_uv(par, subset, fCF4_grid, pressure)
        / npe
        * ENERGY_FACTOR
    )

    y_uv = y_uv_raw 

    return np.asarray(y_vis, dtype=float), np.asarray(y_uv, dtype=float)


def build_theory_family(gas_name,pressure, efield_min, efield_max, gap, npe):
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
            pressure=pressure,
            efield_min=efield_min,
            efield_max=efield_max,
            gap = gap,
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
            npe = npe
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
he_theory_1bar_LIP = build_theory_family(
    gas_name="HeCF4",
    pressure=1,
    efield_min=10.0,
    efield_max=300.0,
    gap = 0.050,
    npe = 1000
)

he_theory_1bar_Florian = build_theory_family(
    gas_name="HeCF4",
    pressure=1,
    efield_min=2.0,
    efield_max=50.0,
    gap = 0.57,
    npe = 50
)


he_theory_300mbar_Florian = build_theory_family(
    gas_name="HeCF4",
    pressure=0.300,
    efield_min=2,
    efield_max=25.0,
    gap = 0.57,
    npe = 50
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
        "name": r"1 bar GEM LIP",
        "gas_theory": he_theory_1bar_LIP,
        "channel": "VIS",
        "color": colors[0],
        "marker": "o",
        "linestyle": "-",
        "xexp": he_cf4_conc,
        "yexp": he_cf4_vis,
        "yerr": he_cf4_vis_err,
    },
    {
        "name": r"1 bar th-GEM Florian",
        "gas_theory": he_theory_1bar_Florian,
        "channel": "VIS",
        "color": colors[1],
        "marker": "s",
        "linestyle": "--",
        "xexp": he_cf4_conc_1bar_Florian,
        "yexp": he_cf4_vis_1bar_Florian,
        "yerr": he_cf4_vis_err_1bar_Florian,
    },
    {
        "name": r"300 mbar th-GEM Florian",
        "gas_theory": he_theory_300mbar_Florian,
        "channel": "VIS",
        "color": colors[2],
        "marker": "^",
        "linestyle": "-.",
        "xexp": he_cf4_conc_300mbar_Florian,
        "yexp": he_cf4_vis_300mbar_Florian,
        "yerr": he_cf4_vis_err_300mbar_Florian,
    },
    # {
    #     "name": r"He-CF$_4$ UV",
    #     "gas_theory": ar_theory_25mbar_Florian,
    #     "channel": "UV",
    #     "color": colors[3],
    #     "marker": "D",
    #     "linestyle": ":",
    #     "xexp": ar_cf4_conc_25mbar_Florian,
    #     "yexp": ar_cf4_vis_25mbar_Florian,
    #     "yerr": ar_cf4_vis_err_25mbar_Florian,
    # },

]


for i in [0,1,2]:
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
            # + rf" range {pretty_case(BAND_LOW_CASE)}--{pretty_case(BAND_HIGH_CASE)}"
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
        label=s["name"] # + rf" optimal fit {pretty_case(OPTIMAL_CASE)}",
    )
    if "LIP" in s["name"]:
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
            label=s["name"]  # + " exp. LIP, GEM 0.050 mm",
        )   

    else:
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
            markerfacecolor=s["color"],
            markeredgewidth=1.2,
            linestyle="none",
            label=s["name"]  # + " exp. LIP, GEM 0.050 mm",
        )   


# ax.grid(True, which="major", alpha=0.3)
# ax.grid(True, which="minor", alpha=0.08)
ax.grid(False)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"CF$_4$ concentration [\%]")
ax.set_ylabel(r"ph/e$^-$")
ax.set_title(r"He-CF$_4$ VIS (400-720 nm)")

ax.set_xlim(10, 110)
ax.set_ylim(3e-2, 8e-1)

ax.legend(loc="upper left", fontsize=11.2, frameon=True, ncol=2)

plt.tight_layout()
plt.savefig(output_pdf, bbox_inches="tight")
print(f"Saved: {output_pdf}")
plt.show()
