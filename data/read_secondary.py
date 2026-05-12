import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(["science"])


def _normalise_gas_name(name):
    if pd.isna(name):
        return None

    s = str(name).strip().lower()
    mapping = {
        "ar": "ar",
        "argon": "ar",
        "cf4": "cf4",
        "n2": "n2",
        "nitrogen": "n2",
        "co2": "co2",
        "xe": "xe",
        "ne": "ne",
        "he": "he",
        "ch4": "ch4",
        "ic4h10": "ic4h10",
        "c2h6": "c2h6",
        "c3h8": "c3h8",
    }
    return mapping.get(s, s)


def _parse_csv_metadata_from_name(filename, gas_concentration=None):
    """
    Extrae metadata del nombre del CSV.

    Añade:
        concentration = concentración del gas pedido en gas_concentration

    Ejemplo:
        ar_90.0_n2_10.0_60.0kVcm_1.000bar_0.1071mm_100npe.csv
    """
    stem = Path(filename).stem
    tokens = stem.split("_")

    gas_fractions = {}
    electric_field = np.nan
    pressure = np.nan
    gap_mm = np.nan

    i = 0
    while i < len(tokens) - 1:
        try:
            frac = float(tokens[i + 1])
            gas = _normalise_gas_name(tokens[i])
            if gas is not None:
                gas_fractions[gas] = frac
            i += 2
            continue
        except ValueError:
            pass

        token = tokens[i].lower()

        if token.endswith("kvcm"):
            try:
                electric_field = float(token.replace("kvcm", ""))
            except ValueError:
                pass

        elif token.endswith("bar"):
            try:
                pressure = float(token.replace("bar", ""))
            except ValueError:
                pass

        elif token.endswith("mm"):
            try:
                gap_mm = float(token.replace("mm", ""))
            except ValueError:
                pass

        i += 1

    concentration = np.nan
    if gas_concentration is not None:
        gas_concentration = _normalise_gas_name(gas_concentration)
        concentration = gas_fractions.get(gas_concentration, np.nan)

    return {
        "concentration": concentration,
        "electric_field": electric_field,
        "pressure": pressure,
        "gap_mm": gap_mm,
        "gas_fractions": gas_fractions
    }


def _merge_gain_summary(population_gen, gain_summary):
    """
    Añade columnas ne y ni a population_gen cruzando por:
        concentration, electric_field, pressure, gap_mm
    """
    population_gen = population_gen.copy()

    if gain_summary is None:
        population_gen["ne"] = np.nan
        population_gen["ni"] = np.nan
        return population_gen

    if isinstance(gain_summary, (str, Path)):
        gain_summary_df = pd.read_csv(gain_summary)
    elif isinstance(gain_summary, pd.DataFrame):
        gain_summary_df = gain_summary.copy()
    else:
        raise TypeError("gain_summary debe ser None, una ruta CSV o un pandas.DataFrame")

    required_cols = {"electric_field", "pressure", "gap", "concentration", "ne_mean", "ni_mean"}
    missing = required_cols - set(gain_summary_df.columns)
    if missing:
        raise ValueError(
            f"gain_summary debe contener las columnas {required_cols}. Faltan: {missing}"
        )

    gain_summary_df = gain_summary_df.rename(columns={"gap": "gap_mm"}).copy()

    merge_cols = ["concentration", "electric_field", "pressure", "gap_mm"]

    for col in merge_cols:
        population_gen[col] = pd.to_numeric(population_gen[col], errors="coerce").round(6)
        gain_summary_df[col] = pd.to_numeric(gain_summary_df[col], errors="coerce").round(6)

    gains_small = (
        gain_summary_df[merge_cols + ["ne_mean", "ni_mean"]]
        .drop_duplicates(subset=merge_cols)
        .rename(columns={"ne_mean": "ne", "ni_mean": "ni"})
    )

    merged = population_gen.merge(gains_small, how="left", on=merge_cols)

    missing_match = merged["ne"].isna() | merged["ni"].isna()
    if missing_match.any():
        print("[AVISO] Filas sin match en gain_summary:")
        print(merged.loc[missing_match, merge_cols])

    return merged


def read_garfield_csv_folder(
    folder_path,
    dataframe,
    output_dir,
    output_general_name,
    gas_concentration,
    use_poisson_error=True,
    gain_summary=None,
    normalized=None
):
    """
    Lee todos los CSV de una carpeta generados a partir de Garfield++ y crea
    tablas resumen.

    Parámetros
    ----------
    gas_concentration : str
        Gas cuya concentración se va a usar para la columna 'concentration'.
        Ejemplos: 'cf4', 'ar', 'n2'
    normalized : None, 'ne', 'ni'
        Si se indica, divide las poblaciones y errores por esa ganancia.
    """
    folder = Path(folder_path)
    output_dir = Path(output_dir)

    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta válida: {folder}")

    if normalized not in (None, "ne", "ni"):
        raise ValueError("normalized debe ser None, 'ne' o 'ni'")

    gas_concentration = _normalise_gas_name(gas_concentration)
    if gas_concentration is None:
        raise ValueError("gas_concentration debe ser un string válido")

    output_dir.mkdir(exist_ok=True, parents=True)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No se encontraron CSVs en: {folder}")
        return pd.DataFrame()

    rows_meta = []
    for csv_file in csv_files:
        meta = _parse_csv_metadata_from_name(
            csv_file.name,
            gas_concentration=gas_concentration
        )
        rows_meta.append({
            "file": str(csv_file),
            "concentration": meta["concentration"],
            "electric_field": meta["electric_field"],
            "gap_mm": meta["gap_mm"],
            "pressure": meta["pressure"]
        })

    population_gen = pd.DataFrame(rows_meta)
    population_gen = _merge_gain_summary(population_gen, gain_summary)

    for nombre_col in dataframe.columns:
        base_cols = ["concentration", "electric_field", "gap_mm", "pressure", "ne", "ni"]
        population = population_gen[base_cols].copy()

        name_of_state = dataframe.loc["name principal", nombre_col]
        gas = _normalise_gas_name(dataframe.loc["gas", nombre_col])
        energy_upper_limit = dataframe.loc["energy up", nombre_col]
        energy_lower_limit = dataframe.loc["energy low", nombre_col]
        name_of_output = dataframe.loc["name output", nombre_col]

        process_type = None
        if "type" in dataframe.index:
            process_type = dataframe.loc["type", nombre_col]

        if isinstance(name_of_state, str):
            name_of_state = [name_of_state]
        elif pd.isna(name_of_state):
            name_of_state = []

        for i, csv_file in enumerate(population_gen["file"]):
            df = pd.read_csv(csv_file)

            expected_cols = {"gas", "state_name", "energy_eV", "n_events"}
            missing_cols = expected_cols - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"El archivo {csv_file} no contiene las columnas necesarias: {missing_cols}"
                )

            df = df.copy()
            df["gas"] = df["gas"].apply(_normalise_gas_name)

            df_main_gas = df.loc[df["gas"] == gas, :].copy()
            mask = pd.Series(True, index=df_main_gas.index)

            for token in name_of_state:
                mask &= df_main_gas["state_name"].fillna("").str.contains(
                    str(token), case=False, na=False
                )

            if pd.notna(energy_lower_limit):
                mask &= pd.to_numeric(df_main_gas["energy_eV"], errors="coerce").fillna(-np.inf) > energy_lower_limit

            if pd.notna(energy_upper_limit):
                mask &= pd.to_numeric(df_main_gas["energy_eV"], errors="coerce").fillna(np.inf) < energy_upper_limit

            if (
                process_type is not None
                and pd.notna(process_type)
                and "type" in df_main_gas.columns
            ):
                mask &= df_main_gas["type"].fillna("").str.lower() == str(process_type).lower()

            total_events = df_main_gas.loc[mask, "n_events"].fillna(0).sum()
            err = np.sqrt(total_events) if use_poisson_error else np.nan

            if normalized is not None:
                norm_value = population.loc[i, normalized]
                if pd.notna(norm_value) and norm_value != 0:
                    total_events = total_events / norm_value
                    if use_poisson_error:
                        err = err / norm_value
                else:
                    total_events = np.nan
                    if use_poisson_error:
                        err = np.nan

            population.loc[i, name_of_output] = total_events
            population_gen.loc[i, name_of_output] = total_events

            if use_poisson_error:
                population.loc[i, "Err" + name_of_output] = err
                population_gen.loc[i, "Err" + name_of_output] = err

        cols_no_fill = {"concentration", "electric_field", "gap_mm", "pressure", "ne", "ni"}
        cols_fill = [c for c in population.columns if c not in cols_no_fill]
        population[cols_fill] = population[cols_fill].fillna(0)

        population.to_csv(output_dir / f"{name_of_output}.csv", index=False)
        print(f"✅ Guardado: {name_of_output}.csv")

    population_gen = population_gen.drop(columns=["file"])

    cols_no_fill = {"concentration", "electric_field", "gap_mm", "pressure", "ne", "ni"}
    cols_fill = [c for c in population_gen.columns if c not in cols_no_fill]
    population_gen[cols_fill] = population_gen[cols_fill].fillna(0)

    sort_cols = [c for c in ["concentration", "electric_field", "gap_mm", "pressure"] if c in population_gen.columns]
    if sort_cols:
        population_gen = population_gen.sort_values(sort_cols).reset_index(drop=True)

    if not str(output_general_name).endswith(".csv"):
        output_general_name = f"{output_general_name}.csv"

    population_gen.to_csv(output_general_name, index=False)
    print(f"✅ Guardado: {output_general_name}")

    return population_gen