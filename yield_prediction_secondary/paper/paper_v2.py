#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper_v2.py
------------
Versión Ar/CF4 del script de paper con:

  - Selector global de canal espectral:
        SPECTRAL_MODE = "visible"     -> 400--720 nm
        SPECTRAL_MODE = "visible_ir"  -> 400--800 nm
        SPECTRAL_MODE = "ir"          -> 720--800 nm aprox.

  - Banda teórica tipo 1_0--2_0 para la contribución visible,
    siguiendo la lógica del script LIP VIS/UV.

  - Curva central configurable con OPTIMAL_CASE = "1_3" o "1_5".

  - Solo Ar/CF4. No se genera nada de He/CF4.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401. Registra el estilo "science".


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
BASE_DIR = os.path.dirname(__file__)

models_dir = os.path.abspath(os.path.join(BASE_DIR, "../../models"))
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../../data"))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArCF4_infrarred import *  # noqa: F401,F403
from ArCF4 import theory_yield_vis
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


# ============================================================
# SELECTOR PRINCIPAL
# ============================================================
# Cambia solo esta línea para elegir qué se representa en las figuras principales.
# Opciones válidas:
#   "visible"     -> visible solo, 400--720 nm
#   "visible_ir"  -> visible + infrarrojo, 400--800 nm
#   "ir"          -> infrarrojo solo, aprox. 720--800 nm
SPECTRAL_MODE = "visible_ir"

# Curva central dentro de la banda. Elige una y comenta la otra si quieres.
OPTIMAL_CASE = "1_3"
# OPTIMAL_CASE = "1_5"

BAND_LOW_CASE = "1_0"
BAND_HIGH_CASE = "2_0"

# Si True, recalcula los CSV de poblaciones cada vez.
# Si False, reutiliza los CSV existentes si ya están generados.
REBUILD_POPULATIONS = True

# Normalización de las poblaciones secundarias. En el paper original estaba en "ne".
POPULATION_NORMALIZATION = "ne"

# Figuras a producir.
RUN_YIELD_VS_E = True
RUN_GEM_CONCENTRATION_SCAN = True
RUN_THGEM_CONCENTRATION_SCAN = True
RUN_NEGATIVE_ION_RATIO_PLOTS = True

# Detalles gráficos.
PLOT_BAND_EDGES = True
SHOW_LIP_POINTS_IN_GEM_SCAN = True
USE_LOG_Y_FOR_MAIN_YIELD_PLOTS = False

# En el script original se declaraba Ar_794, pero no se sumaba en el IR total.
# Lo dejo desactivado para respetar la definición original.
INCLUDE_IR_794 = False

VALID_SPECTRAL_MODES = {"visible", "visible_ir", "ir"}
if SPECTRAL_MODE not in VALID_SPECTRAL_MODES:
    raise ValueError(
        f"SPECTRAL_MODE={SPECTRAL_MODE!r} no es válido. "
        f"Usa una de estas opciones: {sorted(VALID_SPECTRAL_MODES)}"
    )

if OPTIMAL_CASE not in {"1_3", "1_5"}:
    raise ValueError("OPTIMAL_CASE debe ser '1_3' o '1_5'.")


# ============================================================
# RUTAS
# ============================================================
folder_path = os.path.join(data_dir, "Secondary_GarfieldData", "Paper", "root")
table_path = os.path.join(data_dir, "Secondary_GarfieldData", "levels", "ArCF4_level_data.csv")
csv_folder = os.path.join(data_dir, "Secondary_GarfieldData", "Paper", "csv")
populations_dir = os.path.join(data_dir, "Secondary_GarfieldData", "Paper", "populations")
plots_dir = os.path.join(BASE_DIR, "plots")

os.makedirs(populations_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

DATA_DIR_PAR = os.path.join(data_dir, "Parameters")


# ============================================================
# CONSTANTES DEL MODELO
# ============================================================
ENERGY_FACTOR = 15.0
fCF4_GRID = np.logspace(-3, 0, 1000)

CHANNEL_LABEL = {
    "visible": r"400--720 nm",
    "visible_ir": r"400--800 nm",
    "ir": r"720--800 nm",
}[SPECTRAL_MODE]

CHANNEL_FILE_TAG = {
    "visible": "VIS",
    "visible_ir": "VIS_IR",
    "ir": "IR",
}[SPECTRAL_MODE]

# Casos para la banda visible.
# 1_0 y 2_0 son los bordes; 1_3 o 1_5 es la curva central.
CASE_CONFIGS = {
    "1_0": {
        "ECF3": 15.63,
        "csv": "ArCF4_secondary.csv",
        "Pscint": 0.23,
        "PCF4": 1 / 7,
    },
    "1_3": {
        "ECF3": 15.88,
        "csv": "ArCF4_secondary_1_3.csv",
        "Pscint": 0.35,
        "PCF4": 1 / 4,
    },
    "1_5": {
        "ECF3": 16.13,
        "csv": "ArCF4_secondary_1_5.csv",
        "Pscint": 0.42,
        "PCF4": 1 / 7,
    },
    "2_0": {
        "ECF3": 16.38,
        "csv": "ArCF4_secondary_2_0.csv",
        "Pscint": 0.49,
        "PCF4": 1,
    },
}

REQUIRED_COLS = [
    "concentration", "gap_mm", "electric_field", "pressure",
    "CF3", "Ar_dbleStar", "CF4", "Ar_3rd",
    "Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772",
]

if INCLUDE_IR_794:
    REQUIRED_COLS.append("Ar_794")


# ============================================================
# DATOS EXPERIMENTALES LIP DEL SCRIPT ORIGINAL, GEM 100 GAIN
# ============================================================
cf4_ArCF4_pct = np.array([5.0, 10.0, 67.0], dtype=float)

nph_per_e_ArCF4_gain100 = np.array([
    0.56,   # Ar + 5% CF4, extrapolado
    0.52,   # Ar + 10% CF4, extrapolado suave
    0.30,   # Ar + 67% CF4, prácticamente plateau
], dtype=float)

nph_per_e_ArCF4_gain100_err = np.array([
    0.03,
    0.02,
    0.01,
], dtype=float)


# ============================================================
# UTILIDADES
# ============================================================
def use_paper_style():
    try:
        plt.style.use("science")
    except OSError:
        plt.style.use("default")


def pretty_case(case_name):
    return case_name.replace("_", ".")


def positive_array(y):
    """Evita problemas en escala log si aparecen ceros, negativos o NaN."""
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y) & (y > 0), y, np.nan)


def band_limits(y1, y2):
    y1 = positive_array(y1)
    y2 = positive_array(y2)
    return np.fmin(y1, y2), np.fmax(y1, y2)


def scalarize(y):
    arr = np.squeeze(np.asarray(y, dtype=float))
    if arr.ndim != 0:
        arr = np.ravel(arr)
        if arr.size != 1:
            raise ValueError(f"Se esperaba un escalar y se obtuvo un array de tamaño {arr.size}.")
        return float(arr[0])
    return float(arr)


def secondary_population_config(ECF3):
    """
    Configuración de poblaciones para Ar/CF4.

    Incluye las columnas visibles usadas por theory_yield_vis y las líneas IR
    usadas por ArCF4_infrarred.py. La parte visible sigue la lógica del
    script de bandas LIP: ECF3 cambia con cada caso 1_0, 1_3, 1_5, 2_0.
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
            "energy low": 12.90,
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
        "Ar_696": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.32,
            "energy up": 100,
            "name output": "Ar_696",
            "type": "excitation",
        },
        "Ar_727": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.32,
            "energy up": 100,
            "name output": "Ar_727",
            "type": "excitation",
        },
        "Ar_750": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.17,
            "energy up": 100,
            "name output": "Ar_750",
            "type": "excitation",
        },
        "Ar_763": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.17,
            "energy up": 1000,
            "name output": "Ar_763",
            "type": "excitation",
        },
        "Ar_772": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.32,
            "energy up": 100,
            "name output": "Ar_772",
            "type": "excitation",
        },
        "Ar_794": {
            "name principal": "EXC",
            "gas": "Ar",
            "energy low": 13.28,
            "energy up": 100,
            "name output": "Ar_794",
            "type": "excitation",
        },
    })


def build_secondary_population(case_name, cfg, gain_summary):
    output_general_name = os.path.join(populations_dir, f"ArCF4_secondary_{case_name}")
    output_csv = output_general_name + ".csv"

    if REBUILD_POPULATIONS or not os.path.exists(output_csv):
        read_garfield_csv_folder(
            folder_path=csv_folder,
            dataframe=secondary_population_config(ECF3=cfg["ECF3"]),
            output_dir=populations_dir,
            output_general_name=output_general_name,
            gas_concentration="cf4",
            gain_summary=gain_summary,
            normalized=POPULATION_NORMALIZATION,
        )

    df = pd.read_csv(output_csv)
    df["concentration"] = df["concentration"] / 100.0

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"case {case_name}: faltan columnas {missing} en {output_csv}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    return df


def load_visible_parameters(case_name, cfg):
    par_path = os.path.join(DATA_DIR_PAR, cfg["csv"])
    par = pd.read_csv(par_path)["parameter"].to_numpy(dtype=float)

    # El primer parámetro se guarda como normalización auxiliar para el IR,
    # como hacía el script original.
    norm_factor = float(par[0])

    # Misma lógica del script de bandas VIS/UV.
    par[0] = 1.0
    par[1] = cfg["Pscint"]
    par[2] = cfg["Pscint"]
    par[5] *= 1.0
    par[6] *= 1.0
    par[7] *= cfg["PCF4"]
    par[-1] = 0.0

    return par, norm_factor


def load_ir_parameters():
    return pd.read_csv(
        os.path.join(DATA_DIR_PAR, "ArCF4_IR_primary.csv")
    )["parameter"].to_numpy(dtype=float)


def compute_ir_yield(parameter_data_IR, subset, fCF4, pressure, npe, norm_factor):
    y_ir = (
        theory_yield_ArCF4_Ir_696(parameter_data_IR, subset, fCF4, pressure)
        + theory_yield_ArCF4_Ir_727(parameter_data_IR, subset, fCF4, pressure)
        + theory_yield_ArCF4_Ir_750(parameter_data_IR, subset, fCF4, pressure)
        + theory_yield_ArCF4_Ir_763(parameter_data_IR, subset, fCF4, pressure)
        + theory_yield_ArCF4_Ir_772(parameter_data_IR, subset, fCF4, pressure)
    )

    if INCLUDE_IR_794:
        y_ir = y_ir + theory_yield_ArCF4_Ir_794(parameter_data_IR, subset, fCF4, pressure)

    return np.asarray(y_ir, dtype=float) * ENERGY_FACTOR / npe / 0.004364145384539131 # norm_factor


def compute_visible_yield(parameter_data_visible, subset, fCF4, pressure, npe):
    y_vis = theory_yield_vis(parameter_data_visible, subset, fCF4, pressure)
    return np.asarray(y_vis, dtype=float) * ENERGY_FACTOR / npe


def combine_channel(y_vis, y_ir):
    if SPECTRAL_MODE == "visible":
        return y_vis
    if SPECTRAL_MODE == "visible_ir":
        return y_vis + y_ir
    if SPECTRAL_MODE == "ir":
        return y_ir
    raise RuntimeError("SPECTRAL_MODE inválido.")


def compute_selected_yield(case_model, parameter_data_IR, subset, fCF4, pressure, npe):
    y_vis = compute_visible_yield(
        parameter_data_visible=case_model["visible_par"],
        subset=subset,
        fCF4=fCF4,
        pressure=pressure,
        npe=npe,
    )
    y_ir = compute_ir_yield(
        parameter_data_IR=parameter_data_IR,
        subset=subset,
        fCF4=fCF4,
        pressure=pressure,
        npe=npe,
        norm_factor=case_model["norm_factor"],
    )
    return combine_channel(y_vis, y_ir)


def select_subset(df, gap_mm=None, pressure=None, electric_field_min=None,
                  concentration=None, electric_field=None):
    mask = np.ones(len(df), dtype=bool)

    if gap_mm is not None:
        mask &= np.isclose(df["gap_mm"], gap_mm)
    if pressure is not None:
        mask &= np.isclose(df["pressure"], pressure, atol=0.026)
    if electric_field_min is not None:
        mask &= df["electric_field"] > electric_field_min
    if concentration is not None:
        mask &= np.isclose(df["concentration"], concentration)
    if electric_field is not None:
        mask &= np.isclose(df["electric_field"], electric_field)

    return df[mask].copy()


def build_case_models():
    print("Exportando hLevels...")
    export_hlevels_to_csv(
        folder_path,
        table_path,
        object_name="hLevels",
        argon_update=True,
    )

    print("Leyendo ganancias ne/ni...")
    summary = read_data_per_primary_electron(
        folder_path,
        gas_concentration="cf4",
    )
    print(summary)

    parameter_data_IR = load_ir_parameters()
    case_models = {}

    for case_name, cfg in CASE_CONFIGS.items():
        print(f"\nPreparando caso {case_name}...")
        df = build_secondary_population(
            case_name=case_name,
            cfg=cfg,
            gain_summary=summary,
        )
        visible_par, norm_factor = load_visible_parameters(case_name, cfg)
        case_models[case_name] = {
            "df": df,
            "visible_par": visible_par,
            "norm_factor": norm_factor,
        }

    return case_models, parameter_data_IR


def plot_band_and_optimal(ax, x, y_low_case, y_high_case, y_opt,
                          color, label_base, linestyle="-", marker=None):
    y_low, y_high = band_limits(y_low_case, y_high_case)
    y_opt = positive_array(y_opt)

    # Para IR puro la banda suele colapsar casi sobre una única curva. Aun así,
    # se dibuja si los extremos no son idénticos.
    if np.nanmax(np.abs(y_high - y_low)) > 0:
        ax.fill_between(
            x,
            y_low,
            y_high,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            label=(
                label_base
                + rf" band {pretty_case(BAND_LOW_CASE)}--{pretty_case(BAND_HIGH_CASE)}"
            ),
        )

        if PLOT_BAND_EDGES:
            ax.plot(
                x,
                positive_array(y_low_case),
                color=color,
                linestyle=":",
                lw=0.9,
                alpha=0.45,
            )
            ax.plot(
                x,
                positive_array(y_high_case),
                color=color,
                linestyle=":",
                lw=0.9,
                alpha=0.45,
            )

    plot_kwargs = {
        "color": color,
        "linestyle": linestyle,
        "lw": 2.3,
        "label": label_base + rf" optimal {pretty_case(OPTIMAL_CASE)}",
    }
    if marker is not None:
        plot_kwargs["marker"] = marker
        plot_kwargs["ms"] = 4.5

    ax.plot(x, y_opt, **plot_kwargs)


def plot_yield_vs_e(case_models, parameter_data_IR):
    fCF4_fixed = 0.01
    pressure_values = [0.050, 1.0, 10.0]
    npe_values = [100, 100, 100]

    use_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    cmap_obj = plt.get_cmap("viridis")
    colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure_values)))

    for i, pressure in enumerate(pressure_values):
        opt_df = case_models[OPTIMAL_CASE]["df"]
        opt_subset_all = select_subset(
            opt_df,
            pressure=pressure,
            concentration=fCF4_fixed,
        )

        print(f"\n[vs E] pressure = {pressure} bar")
        print(f"N puntos optimal = {len(opt_subset_all)}")
        if not opt_subset_all.empty:
            print("Gaps disponibles:", np.sort(opt_subset_all["gap_mm"].unique()))

        if opt_subset_all.empty:
            print(f"WARNING: no hay datos para p={pressure} bar y CF4={fCF4_fixed}")
            continue

        efields = np.sort(opt_subset_all["electric_field"].unique())
        y_by_case = {case_name: [] for case_name in CASE_CONFIGS}
        x_efields = []

        for electric_field in efields:
            case_values = {}
            skip_this_field = False

            for case_name, case_model in case_models.items():
                subset = select_subset(
                    case_model["df"],
                    pressure=pressure,
                    concentration=fCF4_fixed,
                    electric_field=electric_field,
                )
                if subset.empty:
                    skip_this_field = True
                    break

                y = compute_selected_yield(
                    case_model=case_model,
                    parameter_data_IR=parameter_data_IR,
                    subset=subset,
                    fCF4=fCF4_fixed,
                    pressure=pressure,
                    npe=npe_values[i],
                )
                case_values[case_name] = scalarize(y)

            if skip_this_field:
                continue

            x_efields.append(electric_field)
            for case_name in CASE_CONFIGS:
                y_by_case[case_name].append(case_values[case_name])

        if not x_efields:
            print(f"WARNING: no hay campos comunes para p={pressure} bar")
            continue

        x_efields = np.asarray(x_efields, dtype=float)
        for case_name in CASE_CONFIGS:
            y_by_case[case_name] = np.asarray(y_by_case[case_name], dtype=float)

        plot_band_and_optimal(
            ax=ax,
            x=x_efields,
            y_low_case=y_by_case[BAND_LOW_CASE],
            y_high_case=y_by_case[BAND_HIGH_CASE],
            y_opt=y_by_case[OPTIMAL_CASE],
            color=colors[i],
            label_base=rf"{CHANNEL_LABEL}, 1$\%$ CF$_4$, {pressure:g} bar",
            linestyle="-",
        )

    ax.set_title(rf"Secondary yield for Ar/CF$_4$, 1$\%$ CF$_4$, {CHANNEL_LABEL}")
    ax.set_xlabel(r"Electric field [kV/cm]")
    ax.set_ylabel(r"ph/e$^-$")
    ax.set_xscale("log")
    if USE_LOG_Y_FOR_MAIN_YIELD_PLOTS:
        ax.set_yscale("log")
    ax.grid(False)
    ax.legend(fontsize=7.5, frameon=True)
    fig.tight_layout()

    out = os.path.join(
        plots_dir,
        f"paper_v2_ArCF4_{CHANNEL_FILE_TAG}_vsE_1percentCF4.pdf",
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def plot_concentration_scan(case_models, parameter_data_IR, *,
                            gap_values, pressure_values, electric_field_min_values,
                            npe_values, output_basename, title, show_lip_points=False):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    cmap_obj = plt.get_cmap("viridis")
    colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure_values)))

    x_theory = fCF4_GRID * 100.0

    for i, pressure in enumerate(pressure_values):
        y_by_case = {}

        for case_name, case_model in case_models.items():
            subset = select_subset(
                case_model["df"],
                gap_mm=gap_values[i],
                pressure=pressure,
                electric_field_min=electric_field_min_values[i],
            )

            print(
                f"\n[{output_basename}] case={case_name}, gap={gap_values[i]} mm, "
                f"p={pressure} bar, N={len(subset)}"
            )

            if subset.empty:
                raise ValueError(
                    f"Subset vacío para {output_basename}, case={case_name}, "
                    f"gap={gap_values[i]} mm, p={pressure} bar, "
                    f"E>{electric_field_min_values[i]}."
                )

            y_by_case[case_name] = compute_selected_yield(
                case_model=case_model,
                parameter_data_IR=parameter_data_IR,
                subset=subset,
                fCF4=fCF4_GRID,
                pressure=pressure,
                npe=npe_values[i],
            )

        plot_band_and_optimal(
            ax=ax,
            x=x_theory,
            y_low_case=y_by_case[BAND_LOW_CASE],
            y_high_case=y_by_case[BAND_HIGH_CASE],
            y_opt=y_by_case[OPTIMAL_CASE],
            color=colors[i],
            label_base=rf"{CHANNEL_LABEL}, {pressure:g} bar",
            linestyle="-",
        )

    if show_lip_points and SHOW_LIP_POINTS_IN_GEM_SCAN and SPECTRAL_MODE != "ir":
        ax.errorbar(
            cf4_ArCF4_pct,
            nph_per_e_ArCF4_gain100,
            yerr=nph_per_e_ArCF4_gain100_err,
            fmt="o",
            ms=5.5,
            color="black",
            ecolor="black",
            capsize=3,
            elinewidth=1.1,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            label="LIP exp., GEM 100 gain",
        )

    ax.set_title(title)
    ax.set_xscale("log")
    if USE_LOG_Y_FOR_MAIN_YIELD_PLOTS:
        ax.set_yscale("log")
    ax.set_ylabel(r"ph/e$^-$")
    ax.set_xlabel(r"CF$_4$ concentration [\%]")
    ax.grid(False)
    ax.legend(fontsize=7.2, frameon=True)
    fig.tight_layout()

    out = os.path.join(
        plots_dir,
        f"paper_v2_ArCF4_{output_basename}_{CHANNEL_FILE_TAG}.pdf",
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def compute_ni_minus_over_ni_plus(df, xcol):
    """
    Aproximación:
        n_i- ≈ n_i - n_e
        n_i-/n_i+ ≈ (n_i - n_e) / n_i
    """
    tmp = df.groupby(xcol, as_index=False)[["ne", "ni"]].mean()
    tmp = tmp[tmp["ni"] != 0].copy()
    tmp["ni_minus_est"] = tmp["ni"] - tmp["ne"]
    tmp["ratio"] = tmp["ni_minus_est"] / tmp["ni"]
    return tmp.sort_values(xcol)


def plot_negative_ion_ratios(garfield_data):
    pressure_values = [0.050, 1.0, 10.0]
    fCF4_fixed = 0.01

    use_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    cmap_obj = plt.get_cmap("viridis")
    colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure_values)))

    for i, pressure in enumerate(pressure_values):
        subset_all = select_subset(
            garfield_data,
            pressure=pressure,
            concentration=fCF4_fixed,
        )

        print(f"\n[ni-/ni+ vs E] pressure = {pressure} bar, N={len(subset_all)}")
        if subset_all.empty:
            continue

        ratio_data = compute_ni_minus_over_ni_plus(subset_all, "electric_field")
        if ratio_data.empty:
            continue

        ax.plot(
            ratio_data["electric_field"],
            ratio_data["ratio"],
            color=colors[i],
            marker="o",
            label=rf"1$\%$ CF$_4$, {pressure:g} bar",
        )

    ax.set_title(r"Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$, 1$\%$ CF$_4$")
    ax.set_xlabel(r"Electric field [kV/cm]")
    ax.set_ylabel(r"$n_{i^-}/n_{i^+}$")
    ax.set_xscale("log")
    ax.grid(False)
    ax.legend(frameon=True)
    fig.tight_layout()
    out = os.path.join(plots_dir, "paper_v2_ArCF4_niMinus_over_niPlus_vsE_1percentCF4.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()

    # GEM: gap = 0.05 mm
    plot_negative_ion_ratio_vs_concentration(
        garfield_data,
        gap_values=[0.05, 0.05, 0.05],
        pressure_values=[0.2, 1.0, 10.0],
        electric_field_min_values=[0.0, 0.0, 0.0],
        output_name="paper_v2_ArCF4_gem_niMinus_over_niPlus.pdf",
        title=r"Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$ GEM",
    )

    # thGEM: gap = 0.57 mm
    plot_negative_ion_ratio_vs_concentration(
        garfield_data,
        gap_values=[0.57, 0.57, 0.57],
        pressure_values=[0.050, 1.0, 10.0],
        electric_field_min_values=[0.0, 0.0, 0.0],
        output_name="paper_v2_ArCF4_thgem_niMinus_over_niPlus.pdf",
        title=r"Estimated $n_{i^-}/n_{i^+}$ for Ar/CF$_4$ thGEM",
    )


def plot_negative_ion_ratio_vs_concentration(garfield_data, *, gap_values,
                                             pressure_values,
                                             electric_field_min_values,
                                             output_name, title):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    cmap_obj = plt.get_cmap("viridis")
    colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure_values)))

    for i, pressure in enumerate(pressure_values):
        subset = select_subset(
            garfield_data,
            gap_mm=gap_values[i],
            pressure=pressure,
            electric_field_min=electric_field_min_values[i],
        )

        print(
            f"\n[ni-/ni+ concentration] gap={gap_values[i]} mm, "
            f"p={pressure} bar, N={len(subset)}"
        )

        if subset.empty:
            continue

        ratio_data = compute_ni_minus_over_ni_plus(subset, "concentration")
        if ratio_data.empty:
            continue

        ax.plot(
            ratio_data["concentration"] * 100.0,
            ratio_data["ratio"],
            color=colors[i],
            marker="o",
            label=rf"{pressure:g} bar",
        )

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel(r"CF$_4$ concentration [\%]")
    ax.set_ylabel(r"$n_{i^-}/n_{i^+}$")
    ax.grid(False)
    ax.legend(frameon=True)
    fig.tight_layout()
    out = os.path.join(plots_dir, output_name)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def main():
    print(f"Modo espectral activo: {SPECTRAL_MODE} ({CHANNEL_LABEL})")
    print(f"Banda: {BAND_LOW_CASE}--{BAND_HIGH_CASE}; optimal: {OPTIMAL_CASE}")

    case_models, parameter_data_IR = build_case_models()

    if RUN_YIELD_VS_E:
        plot_yield_vs_e(case_models, parameter_data_IR)

    if RUN_GEM_CONCENTRATION_SCAN:
        plot_concentration_scan(
            case_models,
            parameter_data_IR,
            gap_values=[0.05, 0.05, 0.05],
            pressure_values=[0.2, 1.0, 10.0],
            electric_field_min_values=[0.0, 0.0, 0.0],
            npe_values=[100, 100, 100],
            output_basename="gem",
            title=rf"Secondary yield prediction for Ar/CF$_4$ GEM, {CHANNEL_LABEL}",
            show_lip_points=False,
        )

    if RUN_THGEM_CONCENTRATION_SCAN:
        plot_concentration_scan(
            case_models,
            parameter_data_IR,
            gap_values=[0.57, 0.57, 0.57],
            pressure_values=[0.050, 1.0, 10.0],
            electric_field_min_values=[0.0, 0.0, 0.0],
            npe_values=[100, 100, 100],
            output_basename="thgem",
            title=rf"Secondary yield prediction for Ar/CF$_4$ thGEM, {CHANNEL_LABEL}",
            show_lip_points=False,
        )

    if RUN_NEGATIVE_ION_RATIO_PLOTS:
        # Usamos el caso central como dataset representativo para las gráficas ne/ni.
        garfield_data = case_models[OPTIMAL_CASE]["df"].copy()
        garfield_data["ni_minus_est"] = garfield_data["ni"] - garfield_data["ne"]
        garfield_data["niMinus_over_niPlus"] = garfield_data["ni_minus_est"] / garfield_data["ni"]
        plot_negative_ion_ratios(garfield_data)


if __name__ == "__main__":
    main()
