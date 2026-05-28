import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid"])
except Exception:
    plt.style.use("default")


# ============================================================
# USER CONFIGURATION
# ============================================================
# CF4 concentration as fraction: 0.10 = 10% CF4, 0.05 = 5% CF4, etc.
SELECTED_CF4_CONCENTRATION = 0.10

# Pressure used to select the Garfield secondary cumulative.
# If the Garfield table has no pressure column, this filter is skipped.
SELECTED_PRESSURE_BAR = 1.0

# Gap used for the Garfield secondary cumulative.
# If the Garfield table has no gap column, this filter is skipped.
SELECTED_GAP_MM = 0.05

# Optional electric-field selection. Leave as None to average over all rows
# above MIN_ELECTRIC_FIELD after concentration/pressure/gap selection.
SELECTED_ELECTRIC_FIELD = None
MIN_ELECTRIC_FIELD = 60.0

# Normalisation used by read_garfield_csv_folder: normally "ne" in your script.
NORMALIZATION = "ne"

# If True, rebuilds hLevels -> ArCF4_level_data.csv from the ROOT files.
# If the level CSV already exists, you can set this to False to save time.
REBUILD_LEVEL_TABLE = True

# If True, append the mathematical endpoint cumulative=0 at the upper energy.
FORCE_ZERO_ENDPOINT = True

# Save only PDF figures.
PLOTS_DIR = "plots"


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(__file__)

models_dir = os.path.abspath(os.path.join(BASE_DIR, "../models"))
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../data"))

sys.path.append(models_dir)
sys.path.append(data_dir)

from read_Degrad import read_degrad
from read_Root import export_hlevels_to_csv, read_data_per_primary_electron
from read_secondary import read_garfield_csv_folder


PRIMARY_TXT_DIR = "../data/Primary_DegradData/ArCF4/txt"
PRIMARY_TMP_CSV_DIR = "../data/Primary_DegradData/ArCF4/cumulative_tmp_csv"
PRIMARY_OUTPUT_DIR = "../data/Primary_DegradData/ArCF4"
PRIMARY_TMP_GENERAL = "../data/Primary_DegradData/ArCF4_cumulative_tmp"

SECONDARY_ROOT_DIR = "../data/Secondary_GarfieldData/ArCF4/root"
SECONDARY_CSV_DIR = "../data/Secondary_GarfieldData/ArCF4/csv"
SECONDARY_LEVELS_CSV = "../data/Secondary_GarfieldData/levels/ArCF4_level_data.csv"
SECONDARY_POPULATIONS_DIR = "../data/Secondary_GarfieldData/ArCF4/populations"
SECONDARY_TMP_GENERAL = os.path.join(SECONDARY_POPULATIONS_DIR, "ArCF4_cumulative_tmp")

yaxis_log = True

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PRIMARY_TMP_CSV_DIR, exist_ok=True)
os.makedirs(SECONDARY_POPULATIONS_DIR, exist_ok=True)


# ============================================================
# PRIMARY DEGRAD FILE LISTS
# ============================================================
INPUT_TXT_NAMES = np.array([
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

AR_OUTPUT_NAMES = np.array([
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

CF4_OUTPUT_NAMES = np.array([
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

archivo_entrada = np.char.add(PRIMARY_TXT_DIR, INPUT_TXT_NAMES)
archivo_salida_1 = np.char.add(PRIMARY_TMP_CSV_DIR, AR_OUTPUT_NAMES)
archivo_salida_2 = np.char.add(PRIMARY_TMP_CSV_DIR, CF4_OUTPUT_NAMES)

GAS1 = "ARGON"
GAS2 = "CF4"
CONCENTRATIONS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])


# ============================================================
# SPECIES CONFIGURATION
# ============================================================
SPECIES = {
    "Ar**": {
        "output": "Ar_dbleStar",
        "level_selector": lambda levels: levels[levels["state_name"].str.contains("EXC", na=False)].copy(),
        "primary_gas": "ARGON",
        "primary_name": ["EXC"],
        "primary_energy_up": 100.0,
        "secondary_gas": "Ar",
        "secondary_name": "EXC",
        "secondary_type": "excitation",
        "secondary_energy_up": 100.0,
        "label": r"Ar$^{**}$",
        "file_tag": "Ar_dbleStar",
    },
    "CF3": {
        "output": "CF3",
        "level_selector": lambda levels: levels[levels["state_name"].str.contains("NEUTRAL", na=False)].copy(),
        "primary_gas": "CF4",
        "primary_name": ["NEUTRAL DISS"],
        "primary_energy_up": 100.0,
        "secondary_gas": "CF4",
        "secondary_name": "NEUTRAL DISS",
        "secondary_type": "inelastic",
        "secondary_energy_up": 1100.0,
        "label": r"CF$_3^*$",
        "file_tag": "CF3",
    },
}


# ============================================================
# SMALL HELPERS
# ============================================================
def concentration_title(fcf4):
    ar = 100.0 * (1.0 - fcf4)
    cf4 = 100.0 * fcf4
    return rf"Ar/CF$_4$ = {ar:g}/{cf4:g}"


def concentration_tag(fcf4):
    return f"cf4_{100.0 * fcf4:g}pct".replace(".", "p")


def get_nearest_value(df, column, target, label, atol=1e-12):
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        raise ValueError(f"No numeric values found in column {column!r} while selecting {label}.")

    df_valid = df.loc[valid].copy()
    values = values.loc[valid]

    exact = np.isclose(values.to_numpy(dtype=float), target, rtol=0.0, atol=atol)
    if exact.any():
        return df_valid.loc[exact]

    nearest = values.iloc[np.argmin(np.abs(values.to_numpy(dtype=float) - target))]
    print(f"[WARN] No exact {label}={target}. Using nearest available value: {nearest}.")
    return df_valid.loc[np.isclose(values.to_numpy(dtype=float), nearest, rtol=0.0, atol=atol)]


def find_first_column(df, candidates):
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def normalize_concentration_column(df):
    df = df.copy()
    conc_col = find_first_column(df, ["concentration", "cf4", "fCF4", "f_cf4"])
    if conc_col is None:
        raise KeyError("Could not find a concentration column in the Garfield cumulative table.")

    df[conc_col] = pd.to_numeric(df[conc_col], errors="coerce")
    if df[conc_col].max(skipna=True) > 1.5:
        df[conc_col] = df[conc_col] / 100.0
    return df, conc_col


def reduce_population(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Selected rows do not contain a valid population value.")
    return float(np.mean(values))


def safe_ratio(numerator, denominator):
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


# ============================================================
# CONFIG BUILDERS
# ============================================================
def make_primary_config(species_key, threshold_eV):
    """Configuration passed to read_degrad for a given primary threshold."""
    if species_key == "Ar**":
        ar_low = threshold_eV * 0.999
        cf3_low = 0.0
    elif species_key == "CF3":
        ar_low = 0.0
        cf3_low = threshold_eV * 0.999
    else:
        raise KeyError(species_key)

    return pd.DataFrame(
        {
            "CF4": [["ION CF3 +"], "CF4", 0.0, 100.0, "CF4"],
            "Ar**": [["EXC"], "ARGON", ar_low, 100.0, "Ar_dbleStar"],
            "CF3": [["NEUTRAL DISS"], "CF4", cf3_low, 100.0, "CF3"],
            "Ar3rd": [["CHARGE STATE =2"], "ARGON", 0.0, 100.0, "Ar_3rd"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )


def make_secondary_config(species_key, threshold_eV):
    """Configuration passed to read_garfield_csv_folder for a given secondary threshold."""
    if species_key == "Ar**":
        ar_low = threshold_eV * 0.999
        cf3_low = 0.0
    elif species_key == "CF3":
        ar_low = 0.0
        cf3_low = threshold_eV * 0.999
    else:
        raise KeyError(species_key)

    return pd.DataFrame(
        {
            "CF4": {
                "name principal": "ION",
                "gas": "CF4",
                "energy low": 14.0,
                "energy up": 20.0,
                "name output": "CF4",
                "type": "ionisation",
            },
            "Ar**": {
                "name principal": "EXC",
                "gas": "Ar",
                "energy low": ar_low,
                "energy up": 100.0,
                "name output": "Ar_dbleStar",
                "type": "excitation",
            },
            "CF3": {
                "name principal": "NEUTRAL DISS",
                "gas": "CF4",
                "energy low": cf3_low,
                "energy up": 1100.0,
                "name output": "CF3",
                "type": "inelastic",
            },
            "Ar3rd": {
                "name principal": "IONISATION",
                "gas": "Ar",
                "energy low": 40.0,
                "energy up": 120.0,
                "name output": "Ar_3rd",
                "type": "ionisation",
            },
        }
    )


# ============================================================
# PRIMARY AND SECONDARY CUMULATIVES
# ============================================================
def primary_population(species_key, threshold_eV, selected_concentration):
    cfg = make_primary_config(species_key, threshold_eV)

    read_degrad(
        archivo_entrada,
        archivo_salida_1,
        archivo_salida_2,
        GAS1,
        GAS2,
        CONCENTRATIONS,
        cfg,
        PRIMARY_OUTPUT_DIR,
        PRIMARY_TMP_GENERAL,
    )

    path = PRIMARY_TMP_GENERAL + ".csv"
    df = pd.read_csv(path)
    rows = get_nearest_value(df, "concentration", selected_concentration, "CF4 concentration")
    output_col = SPECIES[species_key]["output"]
    return reduce_population(rows[output_col])


def prepare_secondary_inputs():
    if REBUILD_LEVEL_TABLE or not os.path.exists(SECONDARY_LEVELS_CSV):
        export_hlevels_to_csv(
            SECONDARY_ROOT_DIR,
            SECONDARY_LEVELS_CSV,
            object_name="hLevels",
            argon_update=True,
        )

    summary = read_data_per_primary_electron(
        SECONDARY_ROOT_DIR,
        gas_concentration="cf4",
    )
    return summary


def filter_secondary_rows(df, selected_concentration, selected_pressure, selected_gap):
    df, conc_col = normalize_concentration_column(df)
    selected = get_nearest_value(df, conc_col, selected_concentration, "CF4 concentration")

    pressure_col = find_first_column(selected, ["pressure", "pressure_bar", "p_bar", "p"])
    if pressure_col is not None:
        selected = get_nearest_value(selected, pressure_col, selected_pressure, "pressure [bar]", atol=1e-9)
    else:
        print("[WARN] No pressure column found in Garfield table. Skipping pressure filter.")

    gap_col = find_first_column(selected, ["gap_mm", "gap", "gap [mm]"])
    if gap_col is not None:
        selected = get_nearest_value(selected, gap_col, selected_gap, "gap [mm]", atol=1e-9)
    else:
        print("[WARN] No gap column found in Garfield table. Skipping gap filter.")

    efield_col = find_first_column(
        selected,
        ["electric_field", "electric field", "electric_field_kv_cm", "e_field", "E"],
    )
    if efield_col is not None:
        selected[efield_col] = pd.to_numeric(selected[efield_col], errors="coerce")
        if SELECTED_ELECTRIC_FIELD is not None:
            selected = get_nearest_value(
                selected,
                efield_col,
                SELECTED_ELECTRIC_FIELD,
                "electric field",
                atol=1e-9,
            )
        else:
            selected = selected[selected[efield_col] > MIN_ELECTRIC_FIELD].copy()
            if selected.empty:
                raise ValueError(
                    f"No Garfield rows left after requiring {efield_col} > {MIN_ELECTRIC_FIELD}."
                )
    else:
        print("[WARN] No electric-field column found in Garfield table. Skipping E-field filter.")

    return selected


def secondary_population(species_key, threshold_eV, selected_concentration, selected_pressure, selected_gap, summary):
    cfg = make_secondary_config(species_key, threshold_eV)

    read_garfield_csv_folder(
        folder_path=SECONDARY_CSV_DIR,
        dataframe=cfg,
        output_dir=SECONDARY_POPULATIONS_DIR,
        output_general_name=SECONDARY_TMP_GENERAL,
        gas_concentration="cf4",
        gain_summary=summary,
        normalized=NORMALIZATION,
    )

    df = pd.read_csv(SECONDARY_TMP_GENERAL + ".csv")
    rows = filter_secondary_rows(df, selected_concentration, selected_pressure, selected_gap)
    output_col = SPECIES[species_key]["output"]
    return reduce_population(rows[output_col])


def load_energy_grid(species_key):
    levels = pd.read_csv(SECONDARY_LEVELS_CSV)
    levels = SPECIES[species_key]["level_selector"](levels)
    energies = pd.to_numeric(levels["energy_eV"], errors="coerce").dropna().to_numpy(dtype=float)
    energies = np.unique(np.sort(energies))
    return energies


def build_cumulative(species_key, summary):
    energies = load_energy_grid(species_key)
    if energies.size == 0:
        raise ValueError(f"No level energies found for {species_key}.")

    x_values = [0.0]

    primary_baseline = primary_population(species_key, 0.0, SELECTED_CF4_CONCENTRATION)
    secondary_baseline = secondary_population(
        species_key,
        0.0,
        SELECTED_CF4_CONCENTRATION,
        SELECTED_PRESSURE_BAR,
        SELECTED_GAP_MM,
        summary,
    )

    primary_cum = [1.0]
    secondary_cum = [1.0]

    for energy in energies:
        p = primary_population(species_key, energy, SELECTED_CF4_CONCENTRATION)
        s = secondary_population(
            species_key,
            energy,
            SELECTED_CF4_CONCENTRATION,
            SELECTED_PRESSURE_BAR,
            SELECTED_GAP_MM,
            summary,
        )

        x_values.append(float(energy))
        primary_cum.append(safe_ratio(p, primary_baseline))
        secondary_cum.append(safe_ratio(s, secondary_baseline))

    if FORCE_ZERO_ENDPOINT:
        endpoint = max(
            SPECIES[species_key]["primary_energy_up"],
            SPECIES[species_key]["secondary_energy_up"],
            float(np.max(energies)) * 1.05,
        )
        x_values.append(endpoint)
        primary_cum.append(0.0)
        secondary_cum.append(0.0)

    out = pd.DataFrame(
        {
            "energy_eV": x_values,
            "primary_cumulative": primary_cum,
            "secondary_cumulative": secondary_cum,
        }
    )
    out = out.sort_values("energy_eV").reset_index(drop=True)
    return out


# ============================================================
# PLOTTING
# ============================================================
def plot_cumulative(species_key, cumulative, yaxis_log = False):
    info = SPECIES[species_key]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        cumulative["energy_eV"],
        cumulative["primary_cumulative"],
        marker="o",
        lw=1.6,
        ms=3.5,
        label="Primary / Degrad",
    )
    ax.plot(
        cumulative["energy_eV"],
        cumulative["secondary_cumulative"],
        marker="s",
        lw=1.6,
        ms=3.5,
        ls="--",
        label="Secondary / Garfield++",
    )

    ax.set_xlabel(r"Threshold energy $E_{\mathrm{thr}}$ [eV]")
    ax.set_ylabel(r"Cumulative fraction $N(E \geq E_{\mathrm{thr}})/N(E \geq 0)$")
    ax.set_ylim(-0.05, 1.05)
    if species_key == "Ar**": 
        ax.set_xlim(11.5,15.7)
        if yaxis_log: 
            ax.set_yscale("log")    
            ax.set_ylim(1e-2, 1.05)
    else: 
        ax.set_xlim(11.5,21)
        if yaxis_log: 
            ax.set_yscale("log")    
            ax.set_ylim(1e-3, 1.05)


    title = (
        rf"{info['label']} cumulative threshold scan; "
        rf"{concentration_title(SELECTED_CF4_CONCENTRATION)}, "
        rf"$P={SELECTED_PRESSURE_BAR:g}$ bar"
    )
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.08)

    fig.tight_layout()

    pressure_tag = f"{SELECTED_PRESSURE_BAR:g}".replace(".", "p")
    out_name = (
        f"ArCF4_cumulative_{info['file_tag']}_"
        f"{concentration_tag(SELECTED_CF4_CONCENTRATION)}_"
        f"{pressure_tag}bar.pdf"
    )
    out_path = os.path.join(PLOTS_DIR, out_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

    csv_path = out_path.replace(".pdf", ".csv")
    cumulative.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    summary = prepare_secondary_inputs()

    for species_key in ["Ar**", "CF3"]:
        cumulative = build_cumulative(species_key, summary)
        plot_cumulative(species_key, cumulative, yaxis_log=yaxis_log)


if __name__ == "__main__":
    main()
