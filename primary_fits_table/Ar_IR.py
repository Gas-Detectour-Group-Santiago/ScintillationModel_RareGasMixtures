import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciencePlots debe importarse antes de módulos que hagan plt.style.use(["science"])
try:
    import scienceplots  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Falta el paquete SciencePlots. Instálalo con `pip install SciencePlots` "
        "o ejecuta este script en el mismo entorno donde ya lo usabas."
    ) from exc

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(__file__)
models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
data_dir = os.path.abspath(os.path.join(BASE_DIR, '../data'))
fit_dir = os.path.abspath(os.path.join(BASE_DIR, '../primary_fits'))

sys.path.append(models_dir)
sys.path.append(data_dir)
sys.path.append(fit_dir)
from fiting import fitParameters, fitParameters_lmfit, fitParameters_minimize
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure

# =========================================================
# Imports: models and utilities
# =========================================================
from ArCF4_infrarred import (
    theory_yield_ArCF4_Ir_696,
    theory_yield_ArCF4_Ir_727,
    theory_yield_ArCF4_Ir_750,
    theory_yield_ArCF4_Ir_763,
    theory_yield_ArCF4_Ir_772,
    theory_yield_ArCF4_Ir_794,
)
from ArN2_infrarred import (
    theory_yield_ArN2_Ir_696,
    theory_yield_ArN2_Ir_727,
    theory_yield_ArN2_Ir_750,
    theory_yield_ArN2_Ir_763,
    theory_yield_ArN2_Ir_772,
)

from read_Degrad import read_degrad
from read_experimental_ArCF4_IR import read_experimental as read_experimental_cf4_ir
from read_experimental import read_experimental as read_experimental_n2_ir
from fiting import fitParameters
from parameter_export import export_fit_table_latex


# =========================================================
# Threshold helpers
# =========================================================
def apply_global_threshold_cf4(df, conc_col="fCF4", is_727=False):
    bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
    err_cols = [f"Err {c}" for c in bar_cols]

    df_ref_10 = df[df[conc_col] == 10].copy()
    df_ref_20 = df[df[conc_col] == 20].copy()
    df_ref_50 = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    threshold_10 = df_ref_10[bar_cols].max().max() if not df_ref_10.empty else 0.0
    threshold_20 = df_ref_20[bar_cols].max().max() if not df_ref_20.empty else 0.0
    threshold_50 = df_ref_50[bar_cols].max().max() if not df_ref_50.empty else 0.0
    threshold_100 = df_ref_100[bar_cols].max().max() if not df_ref_100.empty else 0.0

    # Igual que en tu script original: threshold efectivo nulo
    _ = (threshold_10, threshold_20, threshold_50, threshold_100)
    threshold = 0.0

    df_low = df[df[conc_col] < 11].copy()
    if is_727:
        df_low = df[df[conc_col] < 6].copy()

    mask = df_low[bar_cols] >= threshold
    df_low[bar_cols] = df_low[bar_cols].where(mask)

    for bar, err in zip(bar_cols, err_cols):
        df_low[err] = 0.0009
        df_low[err] = df_low[err].where(mask[bar])

    return df_low, threshold


def apply_global_threshold_n2(df, conc_col="N2 concentration (%)", is_727=False):
    bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
    err_cols = [f"Err {c}" for c in bar_cols]

    df_ref_50 = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    threshold_50 = df_ref_50[bar_cols].max().max() if not df_ref_50.empty else 0.0
    threshold_100 = df_ref_100[bar_cols].max().max() if not df_ref_100.empty else 0.0
    threshold = min(threshold_50, threshold_100)

    df_low = df[df[conc_col] < 50].copy()
    if is_727:
        df_low = df[df[conc_col] < 5].copy()

    mask = df_low[bar_cols] >= threshold
    df_low[bar_cols] = df_low[bar_cols].where(mask)

    for bar, err in zip(bar_cols, err_cols):
        df_low[err] = df_low[err].where(mask[bar])

    return df_low, threshold


# =========================================================
# CF4 IR fit
# =========================================================
def fit_cf4_ir():
    archivo_entrada = np.array([
        "/output_99.9Ar_0.1CF4.txt",
        "/output_99.8Ar_0.2CF4.txt",
        "/output_99.5Ar_0.5CF4.txt",
        "/output_99Ar_1CF4.txt",
        "/output_98Ar_2CF4.txt",
        "/output_95Ar_5CF4.txt",
        "/output_90Ar_10CF4.txt",
        "/output_80Ar_20CF4.txt",
        "/output_50Ar_50CF4.txt",
        "/output_PureCF4.txt",
    ])

    archivo_salida_1 = np.array([
        "/ar_degrad_output_99.9Ar_0.1CF4.csv",
        "/ar_degrad_output_99.8Ar_0.2CF4.csv",
        "/ar_degrad_output_99.5Ar_0.5CF4.csv",
        "/ar_degrad_output_99Ar_1CF4.csv",
        "/ar_degrad_output_98Ar_2CF4.csv",
        "/ar_degrad_output_95Ar_5CF4.csv",
        "/ar_degrad_output_90Ar_10CF4.csv",
        "/ar_degrad_output_80Ar_20CF4.csv",
        "/ar_degrad_output_50Ar_50CF4.csv",
        "/ar_degrad_output_PureCF4.csv",
    ])

    archivo_salida_2 = np.array([
        "/cf4_degrad_output_99.9Ar_0.1CF4.csv",
        "/cf4_degrad_output_99.8Ar_0.2CF4.csv",
        "/cf4_degrad_output_99.5Ar_0.5CF4.csv",
        "/cf4_degrad_output_99Ar_1CF4.csv",
        "/cf4_degrad_output_98Ar_2CF4.csv",
        "/cf4_degrad_output_95Ar_5CF4.csv",
        "/cf4_degrad_output_90Ar_10CF4.csv",
        "/cf4_degrad_output_80Ar_20CF4.csv",
        "/cf4_degrad_output_50Ar_50CF4.csv",
        "/cf4_degrad_output_PureCF4.csv",
    ])

    prefijo_txt = "../data/Primary_DegradData/ArCF4/txt"
    archivo_entrada = np.char.add(prefijo_txt, archivo_entrada)

    prefijo_csv = "../data/Primary_DegradData/ArCF4/csv"
    archivo_salida_1 = np.char.add(prefijo_csv, archivo_salida_1)
    archivo_salida_2 = np.char.add(prefijo_csv, archivo_salida_2)

    gas1 = "ARGON"
    gas2 = "CF4"
    concentration = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])

    dataframe = pd.DataFrame(
        {
            "Ar* 696": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_696"],
            "Ar* 727": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_727"],
            "Ar* 750": [["EXC"], "ARGON", 13.47, 13.47 + 10, "Ar_750"],
            "Ar* 763": [["EXC"], "ARGON", 13.17, 13.17 + 10, "Ar_763"],
            "Ar* 772": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_772"],
            "Ar* 794": [["EXC"], "ARGON", 13.28, 13.28 + 10, "Ar_794"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )

    output_dir = "../data/Primary_DegradData/ArCF4/"
    output_general_name = "../data/Primary_DegradData/ArCF4_IR"
    read_degrad(
        archivo_entrada,
        archivo_salida_1,
        archivo_salida_2,
        gas1,
        gas2,
        concentration,
        dataframe,
        output_dir,
        output_general_name,
    )

    archivo_entrada_exp = "../data/Experimental/ArCF4/IR_yields.pkl"
    yields = ["696", "727", "750", "763", "772", "794"]
    presiones = [1, 2, 3, 4, 5]
    read_experimental_cf4_ir(
        archivo_entrada_exp,
        yields,
        presiones,
        "../data/Experimental/ArCF4/",
        concentraciones_reales=None,
        no_sistematic=False,
    )

    data_dir_exp = "../data/Experimental/ArCF4/"
    y696 = pd.read_csv(os.path.join(data_dir_exp, "696.csv"))
    y727 = pd.read_csv(os.path.join(data_dir_exp, "727.csv"))
    y750 = pd.read_csv(os.path.join(data_dir_exp, "750.csv"))
    y763 = pd.read_csv(os.path.join(data_dir_exp, "763.csv"))
    y772 = pd.read_csv(os.path.join(data_dir_exp, "772.csv"))
    y794 = pd.read_csv(os.path.join(data_dir_exp, "794.csv"))

    y696_n, _ = apply_global_threshold_cf4(y696)
    y727_n, _ = apply_global_threshold_cf4(y727)
    y750_n, _ = apply_global_threshold_cf4(y750)
    y763_n, _ = apply_global_threshold_cf4(y763)
    y772_n, _ = apply_global_threshold_cf4(y772)
    y794_n, _ = apply_global_threshold_cf4(y794)

    degrad_data = pd.read_csv(os.path.join("../data/Primary_DegradData", "ArCF4_IR.csv"))

    x0_semifixed = np.array([
        0.0, 28.3, 0.0, 0.0,
        0.0, 28.3, 0.0, 0.0,
        0.0, 21.7, 0.0, 0.0,
        0.0, 29.4, 0.0, 0.0,
        0.0, 28.3, 0.0, 0.0,
        0.0, 29.3, 0.0, 0.0,
    ])

    lower_semifixed = x0_semifixed * 0.999999999999999
    upper_semifixed = x0_semifixed * 1.000000000000001

    lower = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]) + lower_semifixed

    x0 = np.array([
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
    ]) + x0_semifixed

    upper = np.array([
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
    ]) + upper_semifixed

    bounds = (list(lower), list(upper))

    equations = {
        "696": theory_yield_ArCF4_Ir_696,
        "727": theory_yield_ArCF4_Ir_727,
        "750": theory_yield_ArCF4_Ir_750,
        "763": theory_yield_ArCF4_Ir_763,
        "772": theory_yield_ArCF4_Ir_772,
        "794": theory_yield_ArCF4_Ir_794,
    }

    experimental_data = {
        "696": y696_n.fillna(0),
        "727": y727_n.fillna(0),
        "750": y750_n.fillna(0),
        "763": y763_n.fillna(0),
        "772": y772_n.fillna(0),
        "794": y794_n.fillna(0),
    }

    popt = fitParameters(
        equations,
        experimental_data,
        degrad_data,
        x0=x0,
        bounds=bounds,
        is_infrared=True,
        fixed_idx=[1, 5, 9, 13, 17, 21],
        fixed_error=0.1,
    )

    chi2 = getattr(popt, "chi2", np.nan)
    dof = getattr(popt, "dof", np.nan)
    chi2_red = getattr(popt, "chi2_red", np.nan)

    print("=" * 60)
    print("Ar--CF4 IR")
    print("Parámetros globales:\n", popt.x)
    print(f"Grados de libertad: {dof}")
    print(f"Chi2 (real): {chi2}")
    print(f"Chi2 reducido: {chi2_red}")
    print("=" * 60)

    return popt


# =========================================================
# N2 IR fit
# =========================================================
def fit_n2_ir():
    archivo_entrada = np.array([
        "/output_Argon_0.1_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_0.5_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_1.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_5.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_10.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_20.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_Argon_50.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
        "/output_100.0N2_E_0.0Vcmbar_P_1bar_12keV.txt",
    ])

    archivo_salida_1 = np.array([
        "/ar_degrad_output_99.9Ar_0.1N2.csv",
        "/ar_degrad_output_99.5Ar_0.5N2.csv",
        "/ar_degrad_output_99Ar_1N2.csv",
        "/ar_degrad_output_95Ar_5N2.csv",
        "/ar_degrad_output_90Ar_1N2.csv",
        "/ar_degrad_output_80Ar_20N2.csv",
        "/ar_degrad_output_50Ar_50N2.csv",
        "/ar_degrad_output_PureN2.csv",
    ])

    archivo_salida_2 = np.array([
        "/n2_degrad_output_99.9Ar_0.1N2.csv",
        "/n2_degrad_output_99.5Ar_0.5N2.csv",
        "/n2_degrad_output_99Ar_1N2.csv",
        "/n2_degrad_output_95Ar_5N2.csv",
        "/n2_degrad_output_90Ar_10N2.csv",
        "/n2_degrad_output_80Ar_20N2.csv",
        "/n2_degrad_output_50Ar_50N2.csv",
        "/n2_degrad_output_PureN2.csv",
    ])

    prefijo_txt = "../data/Primary_DegradData/ArN2/txt"
    archivo_entrada = np.char.add(prefijo_txt, archivo_entrada)

    prefijo_csv = "../data/Primary_DegradData/ArN2/csv"
    archivo_salida_1 = np.char.add(prefijo_csv, archivo_salida_1)
    archivo_salida_2 = np.char.add(prefijo_csv, archivo_salida_2)

    gas1 = "ARGON"
    gas2 = "NITROGEN"
    concentration = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])

    dataframe = pd.DataFrame(
        {
            "Ar* 696": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_696"],
            "Ar* 727": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_727"],
            "Ar* 750": [["EXC"], "ARGON", 13.47, 13.47 + 10, "Ar_750"],
            "Ar* 763": [["EXC"], "ARGON", 13.17, 13.17 + 10, "Ar_763"],
            "Ar* 772": [["EXC"], "ARGON", 13.32, 13.32 + 10, "Ar_772"],
            # Aunque Ar--N2 no se ajuste a 794 nm, las funciones de modelo
            # leen internamente la columna Ar_794. La generamos para evitar
            # KeyError y su contribución queda efectivamente nula en la comparación.
            "Ar* 794": [["EXC"], "ARGON", 13.28, 13.28 + 10, "Ar_794"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )

    output_dir = "../data/Primary_DegradData/ArN2/"
    output_general_name = "../data/Primary_DegradData/ArN2_IR"
    read_degrad(
        archivo_entrada,
        archivo_salida_1,
        archivo_salida_2,
        gas1,
        gas2,
        concentration,
        dataframe,
        output_dir,
        output_general_name,
    )

    archivo_entrada_exp = "../data/Experimental/ArN2/N2_primary_data_final.pkl"
    yields = ["696", "727", "750", "763", "772"]
    presiones = [1, 2, 3, 4, 5]
    read_experimental_n2_ir(
        archivo_entrada_exp,
        yields,
        presiones,
        "../data/Experimental/ArN2/",
        concentraciones_reales=None,
        uncertainty_mode="all",
    )

    data_dir_exp = "../data/Experimental/ArN2/"
    y696 = pd.read_csv(os.path.join(data_dir_exp, "696.csv"))
    y727 = pd.read_csv(os.path.join(data_dir_exp, "727.csv"))
    y750 = pd.read_csv(os.path.join(data_dir_exp, "750.csv"))
    y763 = pd.read_csv(os.path.join(data_dir_exp, "763.csv"))
    y772 = pd.read_csv(os.path.join(data_dir_exp, "772.csv"))

    y696_n, _ = apply_global_threshold_n2(y696)
    y727_n, _ = apply_global_threshold_n2(y727, is_727=True)
    y750_n, _ = apply_global_threshold_n2(y750)
    y763_n, _ = apply_global_threshold_n2(y763)
    y772_n, _ = apply_global_threshold_n2(y772)

    degrad_data = pd.read_csv(os.path.join("../data/Primary_DegradData", "ArN2_IR.csv"))

    # Salvaguarda: algunas versiones del CSV pueden no traer Ar_794.
    # El modelo Ar--N2 la consulta internamente, así que la añadimos a cero.
    if "Ar_794" not in degrad_data.columns:
        degrad_data["Ar_794"] = 0.0

    x0_semifixed = np.array([
        0.0, 28.3, 0.0, 0.0,
        0.0, 28.3, 0.0, 0.0,
        0.0, 21.7, 0.0, 0.0,
        0.0, 29.4, 0.0, 0.0,
        0.0, 28.3, 0.0, 0.0,
    ])

    lower_semifixed = x0_semifixed * 0.999999999999999
    upper_semifixed = x0_semifixed * 1.000000000000001

    lower = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]) + lower_semifixed

    x0 = np.array([
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
        0.25, 0.0, 1.0, 1.0,
    ]) + x0_semifixed

    upper = np.array([
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
        0.25, 0.0, 1000.0, 1000.0,
    ]) + upper_semifixed

    bounds = (list(lower), list(upper))

    equations = {
        "696": theory_yield_ArN2_Ir_696,
        "727": theory_yield_ArN2_Ir_727,
        "750": theory_yield_ArN2_Ir_750,
        "763": theory_yield_ArN2_Ir_763,
        "772": theory_yield_ArN2_Ir_772,
    }

    experimental_data = {
        "696": y696_n.fillna(0),
        "727": y727_n.fillna(0),
        "750": y750_n.fillna(0),
        "763": y763_n.fillna(0),
        "772": y772_n.fillna(0),
    }

    popt = fitParameters(
        equations,
        experimental_data,
        degrad_data,
        x0=x0,
        bounds=bounds,
        is_infrared=True,
        fixed_idx=[1, 5, 9, 13, 17],
        fixed_error=0.1,
    )

    chi2 = getattr(popt, "chi2", np.nan)
    dof = getattr(popt, "dof", np.nan)
    chi2_red = getattr(popt, "chi2_red", np.nan)

    print("=" * 60)
    print("Ar--N2 IR")
    print("Parámetros globales:\n", popt.x)
    print(f"Grados de libertad: {dof}")
    print(f"Chi2 (real): {chi2}")
    print(f"Chi2 reducido: {chi2_red}")
    print("=" * 60)

    return popt


# =========================================================
# Main: fit both and export one compact comparison table
# =========================================================
def main():
    popt_cf4 = fit_cf4_ir()
    popt_n2 = fit_n2_ir()

    # Solo comparamos la parte común 696, 727, 750, 764 y 772 nm.
    # Se deja fuera 794 nm porque solo existe en el ajuste Ar--CF4.
    x_cf4_common = np.asarray(popt_cf4.x[:20], dtype=float)
    x_n2_common = np.asarray(popt_n2.x, dtype=float)

    names_tex_common = [
        "$P_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
        "$\\tau_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 696 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 696 \\mathrm{nm}}$",

        "$P_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
        "$\\tau_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 727 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 727 \\mathrm{nm}}$",

        "$P_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
        "$\\tau_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 750 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 750 \\mathrm{nm}}$",

        "$P_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
        "$\\tau_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 764 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 764 \\mathrm{nm}}$",

        "$P_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
        "$\\tau_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 772 \\mathrm{nm}}$",
        "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 772 \\mathrm{nm}}$",

        # "$P_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
        # "$\\tau_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
        # "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 794 \\mathrm{nm}}$",
        # "$K_{\\mathrm{Ar}^*, Q(\\mathrm{add.}) \\ 794 \\mathrm{nm}}$",
    ]

    export_fit_table_latex(
        results=[x_cf4_common, x_n2_common],
        names=names_tex_common,
        filename="tex_param/IR_common_comparison_table.tex",
        caption=(
            "Parámetros ajustados del centelleo IR primario en Ar--CF$_4$ y Ar--N$_2$ "
            "para las longitudes de onda comunes (696, 727, 750, 764 y 772 nm). "
            "Las incertidumbres mostradas se han fijado temporalmente al 20\\% relativo "
            "para compactar la comparación."
        ),
        label="tab:IR_common_comparison",
        column_names=["Ar--CF$_4$", "Ar--N$_2$"],
        units=None,
        err_sigfigs=2,
        rel_sigfigs=2,
        show_relative_error=False,
        relative_incertainty=0.2,
    )

    print("Tabla exportada en: tex_param/IR_common_comparison_table.tex")


if __name__ == "__main__":
    main()
