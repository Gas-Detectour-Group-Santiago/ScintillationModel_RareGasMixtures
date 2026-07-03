from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

"""
Analysis_primary_degrad.py
--------------------------
Script autónomo para leer las salidas TXT de Degrad y convertirlas en CSVs
directamente utilizables con pandas.read_csv / Typst load-csv.

No depende de que otro script llame a funciones auxiliares. Basta con ejecutar:

    python Analysis_primary_degrad.py

Las configuraciones importantes están al principio del archivo para poder
cambiar umbrales, poblaciones o rutas sin tocar la lógica de lectura.
"""


ROOT_DIR = Path(__file__).resolve().parent
PRIMARY_DIR = ROOT_DIR / "Primary_DegradData"


# =============================================================================
# CONFIGURACIÓN FÍSICA / DE SELECCIÓN
# =============================================================================

# Umbrales editables.
# En ArCF4 se usa el umbral físico que queremos para las poblaciones Ar* / CF3*.
# ArN2 mantiene la separación meta/resonante/doble excitación alrededor de
# 11.6--11.7 eV.
E_TH_AR_CF4 = 12.9
E_TH_AR_N2 = 11.7
E_TH_CF3 = 12.9


def degrad_config_arcf4() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CF4": [["ION CF3 +"], "CF4", 15.0, 100.0, "CF4"],
            "Ar**": [["EXC"], "ARGON", E_TH_AR_CF4, 100.0, "Ar_dbleStar"],
            "CF3": [["NEUTRAL DISS"], "CF4", E_TH_CF3, 100.0, "CF3"],
            "Ar3rd": [["CHARGE STATE ="], "ARGON", 40.0, 100.0, "Ar_3rd"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )


def degrad_config_arn2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ar Meta": [["EXC"], "ARGON", 0.0, 11.6, "Ar_meta"],
            "Ar Res": [["EXC"], "ARGON", 11.6, 11.7, "Ar_res"],
            "Ar**": [["EXC"], "ARGON", E_TH_AR_N2, 100.0, "Ar_dbleStar"],
            "N2*": [["C 3PI"], "NITROGEN", 11.0, 15.5, "N2_star"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )


def degrad_config_ar2nd() -> pd.DataFrame:
    """Selection used only by the Ar second-continuum model.

    It deliberately does not replace the historical ``Ar_dbleStar`` column used
    by the Ar--CF4 UV/VIS fit.  The second continuum needs the full atomic
    precursor family, so it stores Ar_meta, Ar_res, Ar_dbleStar and their sum in
    dedicated ``*_Ar2nd.csv`` files.
    """
    return pd.DataFrame(
        {
            "Ar Meta": [["EXC"], "ARGON", 0.0, 11.6, "Ar_meta"],
            "Ar Res": [["EXC"], "ARGON", 11.6, 11.7, "Ar_res"],
            "Ar**": [["EXC"], "ARGON", E_TH_AR_N2, 100.0, "Ar_dbleStar"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )


def degrad_config_ir() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ar* 696": [["EXC"], "ARGON", 13.32, 13.32 + 10.0, "Ar_696"],
            "Ar* 727": [["EXC"], "ARGON", 13.32, 13.32 + 10.0, "Ar_727"],
            "Ar* 750": [["EXC"], "ARGON", 13.47, 13.47 + 10.0, "Ar_750"],
            "Ar* 763": [["EXC"], "ARGON", 13.17, 13.17 + 10.0, "Ar_763"],
            "Ar* 772": [["EXC"], "ARGON", 13.32, 13.32 + 10.0, "Ar_772"],
            "Ar* 794": [["EXC"], "ARGON", 13.50, 13.50 + 10.0, "Ar_794"],
        },
        index=["name principal", "gas", "energy low", "energy up", "name output"],
    )


@dataclass(frozen=True)
class DegradRunConfig:
    name: str
    txt_dir: Path
    gas1: str
    gas2: str
    concentration_gas: str
    dataframe: pd.DataFrame
    raw_csv_dir: Path
    population_dir: Path
    output_general_csv: Path


RUNS: tuple[DegradRunConfig, ...] = (
    DegradRunConfig(
        name="ArCF4_primary",
        txt_dir=PRIMARY_DIR / "ArCF4" / "txt",
        gas1="ARGON",
        gas2="CF4",
        concentration_gas="CF4",
        dataframe=degrad_config_arcf4(),
        raw_csv_dir=PRIMARY_DIR / "ArCF4" / "csv",
        population_dir=PRIMARY_DIR / "ArCF4",
        output_general_csv=PRIMARY_DIR / "ArCF4.csv",
    ),
    DegradRunConfig(
        name="ArCF4_Ar2nd",
        txt_dir=PRIMARY_DIR / "ArCF4" / "txt",
        gas1="ARGON",
        gas2="CF4",
        concentration_gas="CF4",
        dataframe=degrad_config_ar2nd(),
        raw_csv_dir=PRIMARY_DIR / "ArCF4" / "csv",
        population_dir=PRIMARY_DIR / "ArCF4_Ar2nd",
        output_general_csv=PRIMARY_DIR / "ArCF4_Ar2nd.csv",
    ),
    DegradRunConfig(
        name="ArCF4_IR",
        txt_dir=PRIMARY_DIR / "ArCF4" / "txt",
        gas1="ARGON",
        gas2="CF4",
        concentration_gas="CF4",
        dataframe=degrad_config_ir(),
        raw_csv_dir=PRIMARY_DIR / "ArCF4" / "csv",
        population_dir=PRIMARY_DIR / "ArCF4",
        output_general_csv=PRIMARY_DIR / "ArCF4_IR.csv",
    ),
    DegradRunConfig(
        name="ArN2_primary",
        txt_dir=PRIMARY_DIR / "ArN2" / "txt",
        gas1="ARGON",
        gas2="NITROGEN",
        concentration_gas="N2",
        dataframe=degrad_config_arn2(),
        raw_csv_dir=PRIMARY_DIR / "ArN2" / "csv",
        population_dir=PRIMARY_DIR / "ArN2",
        output_general_csv=PRIMARY_DIR / "ArN2.csv",
    ),
    DegradRunConfig(
        name="ArN2_Ar2nd",
        txt_dir=PRIMARY_DIR / "ArN2" / "txt",
        gas1="ARGON",
        gas2="NITROGEN",
        concentration_gas="N2",
        dataframe=degrad_config_ar2nd(),
        raw_csv_dir=PRIMARY_DIR / "ArN2" / "csv",
        population_dir=PRIMARY_DIR / "ArN2_Ar2nd",
        output_general_csv=PRIMARY_DIR / "ArN2_Ar2nd.csv",
    ),
    DegradRunConfig(
        name="ArN2_IR",
        txt_dir=PRIMARY_DIR / "ArN2" / "txt",
        gas1="ARGON",
        gas2="NITROGEN",
        concentration_gas="N2",
        dataframe=degrad_config_ir(),
        raw_csv_dir=PRIMARY_DIR / "ArN2" / "csv",
        population_dir=PRIMARY_DIR / "ArN2",
        output_general_csv=PRIMARY_DIR / "ArN2_IR.csv",
    ),
)


# =============================================================================
# LECTURA DEGRAD
# =============================================================================


def normalise_gas_name(name: str) -> str:
    s = str(name).strip().upper()
    aliases = {
        "AR": "ARGON",
        "ARGON": "ARGON",
        "CF4": "CF4",
        "N2": "NITROGEN",
        "NITROGEN": "NITROGEN",
        "HE": "HELIUM",
        "HELIUM": "HELIUM",
    }
    return aliases.get(s, s)


def safe_output_stem(txt_path: Path) -> str:
    stem = txt_path.stem
    if stem.startswith("output_"):
        stem = stem[len("output_") :]
    return stem


def gas_prefix(gas: str) -> str:
    gas = normalise_gas_name(gas)
    return {
        "ARGON": "ar",
        "CF4": "cf4",
        "NITROGEN": "n2",
        "HELIUM": "he",
    }.get(gas, gas.lower())


def parse_concentration_from_filename(path: Path, concentration_gas: str) -> float:
    """
    Devuelve la fracción 0..1 del gas de interés a partir del nombre del TXT.

    Ejemplos:
        output_95Ar_5CF4.txt  -> 0.05 para CF4
        output_PureCF4.txt    -> 1.00 para CF4
        output_100.0N2_...txt -> 1.00 para N2
    """
    stem = path.stem
    target = concentration_gas.strip().lower()
    target_aliases = {
        "argon": ("ar", "argon"),
        "ar": ("ar", "argon"),
        "cf4": ("cf4",),
        "n2": ("n2", "nitrogen"),
        "nitrogen": ("n2", "nitrogen"),
        "he": ("he", "helium"),
        "helium": ("he", "helium"),
    }.get(target, (target,))

    pure_match = re.search(r"Pure([A-Za-z0-9]+)", stem, flags=re.IGNORECASE)
    if pure_match:
        gas = pure_match.group(1).lower()
        return 1.0 if gas in target_aliases else 0.0

    # Formatos tipo 95Ar_5CF4, 99.5Ar_0.5CF4, 100.0N2.
    for value, gas in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z][A-Za-z0-9]*)", stem):
        if gas.lower() in target_aliases:
            return float(value) / 100.0

    # Formatos tipo Argon_0.1_N2_E_... donde el valor aparece antes del gas.
    tokens = re.split(r"[_\s]+", stem)
    for i, token in enumerate(tokens):
        if token.lower() in target_aliases and i > 0:
            try:
                return float(tokens[i - 1]) / 100.0
            except ValueError:
                pass

    raise ValueError(f"No pude extraer concentración de {concentration_gas!r} en {path.name}")


def split_by_gas(block_text: str) -> list[tuple[str, str]]:
    headers = list(
        re.finditer(
            r"^\s*(?P<gas>[A-Z0-9]+)(?:\s+\d{4})?\s+ANISOTROPIC[^\n]*\n[-]{8,}\s*",
            block_text,
            flags=re.M,
        )
    )

    parts: list[tuple[str, str]] = []
    for i, header in enumerate(headers):
        gas = header.group("gas").strip()
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(block_text)
        parts.append((gas, block_text[start:end]))

    return parts


def parse_degrad_lines(gas: str, gas_block: str) -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*(?P<proc>.+?)"
        r"(?:\s+(?:E(?:LEVEL|LOSS)=\s*(?P<energy>-?\d*\.?\d+(?:D[+-]?\d+)?)))?"
        r"\s+(?P<value>-?\d*\.?\d+)\s*\+\-\s*(?P<err>-?\d*\.?\d+)\s*%",
        flags=re.M,
    )

    rows = []
    for match in pattern.finditer(gas_block):
        proc = re.sub(r"\s{2,}", " ", match.group("proc").strip())
        energy = match.group("energy")
        rows.append(
            {
                "Gas": gas,
                "Proceso": proc,
                "Energia": float(energy.replace("D", "E")) if energy else np.nan,
                "Eventos": float(match.group("value")),
                "Error%": float(match.group("err")),
            }
        )

    return pd.DataFrame(rows)


def read_degrad_txt(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = "NUMBER OF COLLISIONS PER EVENT FOR EACH GAS"
    idx = text.find(title)

    if idx == -1:
        raise ValueError(f"No encontré el bloque de colisiones en {path}")

    frames = [parse_degrad_lines(gas, block) for gas, block in split_by_gas(text[idx:])]
    if not frames:
        raise ValueError(f"No pude separar gases en {path}")

    return pd.concat(frames, ignore_index=True)


def save_split_raw_csvs(df: pd.DataFrame, raw_csv_dir: Path, txt_path: Path, gas1: str, gas2: str) -> None:
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_output_stem(txt_path)

    for gas in (gas1, gas2):
        gas_norm = normalise_gas_name(gas)
        out = raw_csv_dir / f"{gas_prefix(gas_norm)}_degrad_output_{stem}.csv"
        df.loc[df["Gas"].map(normalise_gas_name) == gas_norm].to_csv(out, index=False)


def select_population(df: pd.DataFrame, name_tokens, gas: str, energy_low: float, energy_up: float) -> tuple[float, float]:
    if isinstance(name_tokens, str):
        name_tokens = [name_tokens]

    gas_norm = normalise_gas_name(gas)
    df_gas = df.loc[df["Gas"].map(normalise_gas_name) == gas_norm].copy()

    mask = pd.Series(True, index=df_gas.index)
    for token in name_tokens:
        mask &= df_gas["Proceso"].fillna("").str.contains(str(token), case=False, regex=False)

    energy = pd.to_numeric(df_gas["Energia"], errors="coerce")
    mask &= energy.ge(float(energy_low)) & energy.lt(float(energy_up))

    selected = df_gas.loc[mask]
    value = selected["Eventos"].sum()
    err = np.sqrt(((selected["Eventos"] * selected["Error%"]) ** 2).sum()) / 100.0
    return float(value), float(err)


AR2ND_PRECURSOR_COLUMNS = ("Ar_meta", "Ar_res", "Ar_dbleStar")


def add_ar2nd_precursor_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Add the explicit precursor population used by the Ar second continuum."""
    out = df.copy()
    if not all(col in out.columns for col in AR2ND_PRECURSOR_COLUMNS):
        return out

    out["Ar_2nd_precursor"] = out[list(AR2ND_PRECURSOR_COLUMNS)].sum(axis=1)
    err_cols = [f"Err{col}" for col in AR2ND_PRECURSOR_COLUMNS]
    if all(col in out.columns for col in err_cols):
        out["ErrAr_2nd_precursor"] = np.sqrt(np.square(out[err_cols]).sum(axis=1))
    return out


def analyse_degrad_run(config: DegradRunConfig) -> pd.DataFrame:
    if not config.txt_dir.is_dir():
        raise NotADirectoryError(f"No existe la carpeta TXT: {config.txt_dir}")

    txt_files = sorted(config.txt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No hay TXT de Degrad en {config.txt_dir}")

    rows = []

    for txt_path in txt_files:
        concentration = parse_concentration_from_filename(txt_path, config.concentration_gas)
        df = read_degrad_txt(txt_path)
        save_split_raw_csvs(df, config.raw_csv_dir, txt_path, config.gas1, config.gas2)

        row = {"concentration": concentration}
        for col in config.dataframe.columns:
            name_tokens = config.dataframe.loc["name principal", col]
            gas = config.dataframe.loc["gas", col]
            energy_low = config.dataframe.loc["energy low", col]
            energy_up = config.dataframe.loc["energy up", col]
            out_name = config.dataframe.loc["name output", col]

            value, err = select_population(df, name_tokens, gas, energy_low, energy_up)
            row[out_name] = value
            row[f"Err{out_name}"] = err

        rows.append(row)

    population_gen = pd.DataFrame(rows).sort_values("concentration").reset_index(drop=True)
    population_gen = population_gen.fillna(0)
    if config.name.endswith("_Ar2nd"):
        population_gen = add_ar2nd_precursor_sum(population_gen)

    config.population_dir.mkdir(parents=True, exist_ok=True)
    config.output_general_csv.parent.mkdir(parents=True, exist_ok=True)

    for col in config.dataframe.columns:
        out_name = config.dataframe.loc["name output", col]
        one = population_gen[["concentration", out_name, f"Err{out_name}"]].copy()
        one.to_csv(config.population_dir / f"{out_name}.csv", index=False)

    if config.name.endswith("_Ar2nd") and "Ar_2nd_precursor" in population_gen.columns:
        one = population_gen[["concentration", "Ar_2nd_precursor", "ErrAr_2nd_precursor"]].copy()
        one.to_csv(config.population_dir / "Ar_2nd_precursor.csv", index=False)

    population_gen.to_csv(config.output_general_csv, index=False)
    print(f"✅ {config.name}: {config.output_general_csv.relative_to(ROOT_DIR)}")
    return population_gen


def main() -> None:
    for config in RUNS:
        analyse_degrad_run(config)


if __name__ == "__main__":
    main()
