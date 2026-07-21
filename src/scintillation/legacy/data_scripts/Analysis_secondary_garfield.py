from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

"""
Analysis_secondary_garfield.py
------------------------------
Script autónomo para leer ROOTs de Garfield++ y exportar CSVs listos para
pandas.read_csv / Typst load-csv.

Hace en una sola pasada:
    1. Lee hLevels de cada ROOT.
    2. Cruza los niveles con las tablas Secondary_GarfieldData/levels/*.csv.
    3. Lee dataPerPrimaryElectron para ne/ni.
    4. Extrae npe del nombre del ROOT y, si hace falta, de títulos internos.
    5. Agrupa únicamente por mezcla gaseosa: ArCF4, ArN2, HeCF4.

Por defecto NO dibuja distribuciones de ganancia. Para activarlas:

    PRINT_GAIN_DISTRIBUTIONS = True

o llama a analyse_secondary_run(..., print_gain_distributions=True).
"""


ROOT_DIR = Path(__file__).resolve().parent
SECONDARY_DIR = ROOT_DIR / "Secondary_GarfieldData"
ARCF4_PAPER_DIR = SECONDARY_DIR / "ArCF4_paper"

PRINT_GAIN_DISTRIBUTIONS = False
SAVE_LEVEL_CSVS = False
SAVE_SPLIT_POPULATIONS = False
NORMALIZED_BY: str | None = None # "ne"  # None, "ne" o "ni"
USE_POISSON_ERROR = True

E_TH_AR_CF4 = 12.0
E_TH_CF3 = 12.9
E_TH_AR_N2 = 11.7


# =============================================================================
# CONFIGURACIONES DE POBLACIÓN
# =============================================================================


def _population_config(mixture_id: str) -> pd.DataFrame:
    import sys

    project_root = ROOT_DIR.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scintillation.populations import rules_as_legacy_dataframe

    return rules_as_legacy_dataframe(project_root, mixture_id)


def config_arcf4_secondary() -> pd.DataFrame:
    return _population_config("ArCF4")


def config_arn2_secondary() -> pd.DataFrame:
    return _population_config("ArN2")


@dataclass(frozen=True)
class SecondaryRunConfig:
    mixture: str
    root_dirs: tuple[Path, ...]
    level_table: Path
    gas_concentration: str
    dataframe: pd.DataFrame
    level_csv_dir: Path
    population_dir: Path
    output_general_csv: Path




def _reference_root_dirs(reference_dir: Path, legacy_root_dir: Path | None = None) -> tuple[Path, ...]:
    """Possible ROOT locations for one self-contained reference folder.

    Preferred layout:
        data/Secondary_GarfieldData/ArCF4_paper/<reference>/root/*.root

    For transition, roots placed directly in <reference>/ or in the old
    data/Secondary_GarfieldData/Paper/root/<reference>/ folder are also read.
    Outputs are always written to the new ArCF4_paper/<reference>/csv and
    ArCF4_paper/<reference>/populations folders.
    """
    candidates = [reference_dir / "root", reference_dir]
    if legacy_root_dir is not None:
        candidates.append(legacy_root_dir)
    return tuple(candidates)


def arcf4_paper_run(reference_name: str, *, legacy_root_name: str | None = None) -> SecondaryRunConfig:
    reference_dir = ARCF4_PAPER_DIR / reference_name
    legacy_root_dir = None
    if legacy_root_name is not None:
        legacy_root_dir = SECONDARY_DIR / "Paper" / "root" / legacy_root_name
    return SecondaryRunConfig(
        mixture=f"ArCF4_paper/{reference_name}",
        root_dirs=_reference_root_dirs(reference_dir, legacy_root_dir),
        level_table=SECONDARY_DIR / "levels" / "ArCF4_level_data.csv",
        gas_concentration="cf4",
        dataframe=config_arcf4_secondary(),
        level_csv_dir=reference_dir / "csv",
        population_dir=reference_dir / "populations",
        output_general_csv=reference_dir / "populations" / "ArCF4_secondary.csv",
    )


ARCF4_PAPER_RUNS: tuple[SecondaryRunConfig, ...] = (
    arcf4_paper_run("electricField", legacy_root_name="electricField"),
    arcf4_paper_run("gem_200mbar", legacy_root_name="gem_200mbar"),
    arcf4_paper_run("gem_1bar", legacy_root_name="gem_1bar"),
    arcf4_paper_run("gem_10bar", legacy_root_name="gem_10bar"),
    arcf4_paper_run("thgem_50mbar", legacy_root_name="thgem_50mbar"),
    arcf4_paper_run("thgem_1bar", legacy_root_name="thgem_1bar"),
    arcf4_paper_run("thgem_10bar", legacy_root_name="thgem_10bar"),
)


RUNS: tuple[SecondaryRunConfig, ...] = (
    SecondaryRunConfig(
        mixture="ArCF4",
        root_dirs=(SECONDARY_DIR / "ArCF4" / "root",),
        level_table=SECONDARY_DIR / "levels" / "ArCF4_level_data.csv",
        gas_concentration="cf4",
        dataframe=config_arcf4_secondary(),
        level_csv_dir=SECONDARY_DIR / "ArCF4" / "csv",
        population_dir=SECONDARY_DIR / "ArCF4" / "populations",
        output_general_csv=SECONDARY_DIR / "ArCF4" / "populations" / "ArCF4_secondary.csv",
    ),
    SecondaryRunConfig(
        mixture="ArN2",
        root_dirs=(SECONDARY_DIR / "ArN2" / "root",),
        level_table=SECONDARY_DIR / "levels" / "ArN2_level_data.csv",
        gas_concentration="n2",
        dataframe=config_arn2_secondary(),
        level_csv_dir=SECONDARY_DIR / "ArN2" / "csv",
        population_dir=SECONDARY_DIR / "ArN2" / "populations",
        output_general_csv=SECONDARY_DIR / "ArN2" / "populations" / "ArN2_secondary.csv",
    ),
    SecondaryRunConfig(
        mixture="HeCF4",
        root_dirs=(SECONDARY_DIR / "HeCF4" / "root",),
        level_table=SECONDARY_DIR / "levels" / "HeCF4_level_data.csv",
        gas_concentration="cf4",
        dataframe=config_arcf4_secondary(),
        level_csv_dir=SECONDARY_DIR / "HeCF4" / "csv",
        population_dir=SECONDARY_DIR / "HeCF4" / "populations",
        output_general_csv=SECONDARY_DIR / "HeCF4" / "populations" / "HeCF4_secondary.csv",
    ),
)


def _registered_secondary_runs() -> tuple[SecondaryRunConfig, ...]:
    project_root=Path(os.environ.get("SCINTILLATION_ROOT",ROOT_DIR.parent))
    path=project_root/"config"/"secondary_inputs.csv"
    if not path.exists():
        return (*RUNS,*ARCF4_PAPER_RUNS)
    frame=pd.read_csv(path)
    frame=frame.loc[frame["enabled"].astype(str).str.lower().isin({"1","true","yes","on"})]
    runs=[]
    for row in frame.itertuples(index=False):
        output_base=SECONDARY_DIR/str(row.output_subdir)
        runs.append(SecondaryRunConfig(
            mixture=str(row.run_id), root_dirs=(SECONDARY_DIR/str(row.raw_subdir),),
            level_table=SECONDARY_DIR/"levels"/str(row.level_table),
            gas_concentration=str(row.gas_concentration),
            dataframe=_population_config(str(row.population_rules_mixture)),
            level_csv_dir=output_base/"csv", population_dir=output_base/"populations",
            output_general_csv=output_base/"populations"/f"{row.mixture_id}_secondary.csv",
        ))
    return tuple(runs)

REGISTERED_RUNS = _registered_secondary_runs()


# =============================================================================
# UTILIDADES
# =============================================================================


def import_uproot():
    try:
        import uproot  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Falta uproot. Instálalo en tu entorno con: pip install uproot"
        ) from exc
    return uproot


def normalise_gas_name(name) -> str | None:
    if pd.isna(name):
        return None

    s = str(name).strip().lower()
    mapping = {
        "ar": "ar",
        "argon": "ar",
        "cf4": "cf4",
        "n2": "n2",
        "nitrogen": "n2",
        "he": "he",
        "helium": "he",
        "co2": "co2",
        "xe": "xe",
        "ne": "ne",
        "ch4": "ch4",
        "ic4h10": "ic4h10",
        "c2h6": "c2h6",
        "c3h8": "c3h8",
    }
    return mapping.get(s, s)


def parse_root_metadata(root_file: Path, gas_concentration: str | None = None) -> dict:
    stem = root_file.stem
    tokens = stem.split("_")

    gas_fractions: dict[str, float] = {}
    electric_field = np.nan
    pressure = np.nan
    gap_mm = np.nan
    npe = np.nan

    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            gas = normalise_gas_name(tokens[i])
            try:
                frac = float(tokens[i + 1])
                if gas is not None:
                    gas_fractions[gas] = frac
                i += 2
                continue
            except ValueError:
                pass

        token = tokens[i].strip().lower()

        if token.endswith("kvcm"):
            electric_field = float(token.replace("kvcm", ""))
        elif token.endswith("bar"):
            pressure = float(token.replace("bar", ""))
        elif token.endswith("mm"):
            gap_mm = float(token.replace("mm", ""))
        elif token.endswith("npe"):
            npe = float(token.replace("npe", ""))

        i += 1

    gas_concentration_norm = normalise_gas_name(gas_concentration) if gas_concentration else None
    concentration = gas_fractions.get(gas_concentration_norm, np.nan) if gas_concentration_norm else np.nan

    gas_items = list(gas_fractions.items())

    return {
        "file": root_file.name,
        "gas_mixture": mixture_name_from_fractions(gas_fractions),
        "gas1": gas_items[0][0] if len(gas_items) >= 1 else pd.NA,
        "concentration_gas_1": gas_items[0][1] if len(gas_items) >= 1 else np.nan,
        "gas2": gas_items[1][0] if len(gas_items) >= 2 else pd.NA,
        "concentration_gas_2": gas_items[1][1] if len(gas_items) >= 2 else np.nan,
        "concentration": concentration,
        "electric_field": electric_field,
        "pressure": pressure,
        "gap_mm": gap_mm,
        "npe": npe,
        "gas_fractions": gas_fractions,
    }


def mixture_name_from_fractions(gas_fractions: dict[str, float]) -> str:
    active = [gas for gas, frac in gas_fractions.items() if frac > 0]
    if not active:
        return ""
    preferred = {"ar": "Ar", "cf4": "CF4", "n2": "N2", "he": "He"}
    return "".join(preferred.get(g, g.upper()) for g in active)


def active_gases_from_metadata(meta: dict) -> list[str]:
    return [gas for gas, frac in meta["gas_fractions"].items() if frac > 0]


def read_level_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la tabla de niveles: {path}")

    df = pd.read_csv(path)
    required = {"level", "gas", "state_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} no contiene columnas obligatorias: {missing}")

    df = df.copy()
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df = df.dropna(subset=["level"]).copy()
    df["level"] = df["level"].astype(int)
    df["_gas_norm"] = df["gas"].apply(normalise_gas_name)
    return df


def mapping_for_file(level_table: pd.DataFrame, active_gases: list[str]) -> pd.DataFrame:
    mapping = level_table.copy()
    if active_gases:
        mapping = mapping.loc[mapping["_gas_norm"].isin(active_gases)].copy()

    mapping = mapping.sort_values("level").reset_index(drop=True)

    # Para gases puros Garfield suele numerar únicamente los niveles de ese gas.
    if len(active_gases) == 1:
        mapping = mapping.loc[mapping["_gas_norm"] == active_gases[0]].copy()
        mapping = mapping.sort_values("level").reset_index(drop=True)
        mapping["level"] = np.arange(len(mapping), dtype=int)

    return mapping


def hist_values(root_handle, object_name: str = "hLevels") -> np.ndarray:
    keys = root_handle.keys(cycle=False)
    if object_name not in keys:
        raise KeyError(f"No existe {object_name!r}. Objetos disponibles: {keys}")
    h = root_handle[object_name]
    return np.asarray(h.values(), dtype=float)


def hlevels_to_dataframe(root_handle, meta: dict, level_table: pd.DataFrame) -> pd.DataFrame:
    values = hist_values(root_handle, "hLevels")
    mapping = mapping_for_file(level_table, active_gases_from_metadata(meta))

    df = pd.DataFrame({"level": np.arange(len(values), dtype=int), "n_events": values})
    df = df.merge(mapping, how="left", on="level")

    if not mapping.empty:
        max_level = int(max(df["level"].max(), mapping["level"].max()))
        df = df.set_index("level").reindex(range(max_level + 1)).reset_index()
        df["n_events"] = df["n_events"].fillna(0)

        meta_cols = [c for c in mapping.columns if c != "level"]
        meta_df = mapping[["level"] + meta_cols].drop_duplicates("level")
        df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
        df = df.merge(meta_df, how="left", on="level")

    if "_gas_norm" in df.columns:
        df = df.drop(columns="_gas_norm")

    preferred = ["level", "gas", "state_name", "type", "energy_eV", "n_events"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def extract_npe_from_titles(root_handle, fallback=np.nan) -> float:
    pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*npe", flags=re.IGNORECASE)

    for key in root_handle.keys(cycle=False):
        try:
            obj = root_handle[key]
            title = getattr(obj, "title", None)
            if callable(title):
                title = title()
            if title:
                match = pattern.search(str(title))
                if match:
                    return float(match.group(1))
        except Exception:
            continue

    return fallback


def read_gain_summary(root_handle) -> dict:
    out = {
        "ne": np.nan,
        "ne_std": np.nan,
        "ni": np.nan,
        "ni_std": np.nan,
        "n_entries": 0,
    }

    if "dataPerPrimaryElectron" not in root_handle:
        return out

    tree = root_handle["dataPerPrimaryElectron"]
    branches = set(tree.keys())
    needed = {"nElectrons", "nIons"}
    if not needed.issubset(branches):
        return out

    arrays = tree.arrays(["nElectrons", "nIons"], library="np")
    ne = np.asarray(arrays["nElectrons"], dtype=float)
    ni = np.asarray(arrays["nIons"], dtype=float)

    out["ne"] = float(np.mean(ne)) if len(ne) else np.nan
    out["ne_std"] = float(np.std(ne, ddof=1)) if len(ne) > 1 else 0.0
    out["ni"] = float(np.mean(ni)) if len(ni) else np.nan
    out["ni_std"] = float(np.std(ni, ddof=1)) if len(ni) > 1 else 0.0
    out["n_entries"] = int(len(ne))
    return out


def maybe_plot_gain_distribution(root_file: Path, ne_values: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[AVISO] matplotlib no está instalado; no se dibujan distribuciones.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ne_values, bins="auto", edgecolor="white")
    ax.set_xlabel("Electrons generated per primary electron")
    ax.set_ylabel("Frequency")
    ax.set_title(root_file.stem)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{root_file.stem}_ne.pdf")
    plt.close(fig)


def select_population(df: pd.DataFrame, name_tokens, gas: str, energy_low: float, energy_up: float) -> float:
    gas_norm = normalise_gas_name(gas)
    if isinstance(name_tokens, str):
        name_tokens = [name_tokens]

    work = df.copy()
    work["gas_norm"] = work["gas"].apply(normalise_gas_name)
    df_gas = work.loc[work["gas_norm"] == gas_norm].copy()

    mask = pd.Series(True, index=df_gas.index)
    for token in name_tokens:
        mask &= df_gas["state_name"].fillna("").str.contains(str(token), case=False, regex=False)

    energy = pd.to_numeric(df_gas["energy_eV"], errors="coerce")
    mask &= energy.ge(float(energy_low)) & energy.lt(float(energy_up))

    return float(df_gas.loc[mask, "n_events"].fillna(0).sum())


def unique_csv_path(out_dir: Path, stem: str) -> Path:
    candidate = out_dir / f"{stem}.csv"
    if not candidate.exists():
        return candidate

    i = 2
    while True:
        candidate = out_dir / f"{stem}_{i}.csv"
        if not candidate.exists():
            return candidate
        i += 1


def analyse_root_file(
    root_file: Path,
    level_table: pd.DataFrame,
    dataframe: pd.DataFrame,
    gas_concentration: str,
    level_csv_dir: Path,
    save_level_csvs: bool,
    normalized_by: str | None,
    use_poisson_error: bool,
    print_gain_distributions: bool,
) -> dict:
    uproot = import_uproot()
    meta = parse_root_metadata(root_file, gas_concentration=gas_concentration)

    with uproot.open(root_file) as f:
        if pd.isna(meta["npe"]):
            meta["npe"] = extract_npe_from_titles(f, fallback=np.nan)

        level_df = hlevels_to_dataframe(f, meta, level_table)
        gain = read_gain_summary(f)

        if print_gain_distributions and "dataPerPrimaryElectron" in f:
            tree = f["dataPerPrimaryElectron"]
            if "nElectrons" in set(tree.keys()):
                ne_values = np.asarray(tree.arrays(["nElectrons"], library="np")["nElectrons"], dtype=float)
                maybe_plot_gain_distribution(root_file, ne_values, root_file.parent.parent / "gain_distribution")

    if save_level_csvs:
        level_csv_dir.mkdir(parents=True, exist_ok=True)
        level_csv = unique_csv_path(level_csv_dir, root_file.stem)
        level_df.to_csv(level_csv, index=False)

    row = {
        "file": meta["file"],
        "gas_mixture": meta["gas_mixture"],
        "gas1": meta["gas1"],
        "concentration_gas_1": meta["concentration_gas_1"],
        "gas2": meta["gas2"],
        "concentration_gas_2": meta["concentration_gas_2"],
        "concentration": meta["concentration"],
        "electric_field": meta["electric_field"],
        "gap_mm": meta["gap_mm"],
        "pressure": meta["pressure"],
        "npe": meta["npe"],
        **gain,
    }

    if normalized_by not in (None, "ne", "ni"):
        raise ValueError("normalized_by debe ser None, 'ne' o 'ni'")

    norm_value = row.get(normalized_by, np.nan) if normalized_by else None

    for col in dataframe.columns:
        name_tokens = dataframe.loc["name principal", col]
        gas = dataframe.loc["gas", col]
        energy_low = dataframe.loc["energy low", col]
        energy_up = dataframe.loc["energy up", col]
        out_name = dataframe.loc["name output", col]

        raw_total = select_population(level_df, name_tokens, gas, energy_low, energy_up)
        err = np.sqrt(raw_total) if use_poisson_error else np.nan

        value = raw_total
        if normalized_by is not None:
            if pd.notna(norm_value) and norm_value != 0:
                value = raw_total / float(norm_value)
                err = err / float(norm_value) if use_poisson_error else np.nan
            else:
                value = np.nan
                err = np.nan

        row[out_name] = value
        if use_poisson_error:
            row[f"Err{out_name}"] = err

    return row


def analyse_secondary_run(
    config: SecondaryRunConfig,
    print_gain_distributions: bool = PRINT_GAIN_DISTRIBUTIONS,
    save_level_csvs: bool = SAVE_LEVEL_CSVS,
    normalized_by: str | None = NORMALIZED_BY,
    use_poisson_error: bool = USE_POISSON_ERROR,
) -> pd.DataFrame:
    root_files: list[Path] = []
    for root_dir in config.root_dirs:
        if root_dir.is_dir():
            root_files.extend(sorted(root_dir.glob("*.root")))

    if not root_files:
        raise FileNotFoundError(f"No encontré ROOTs para {config.mixture}: {config.root_dirs}")

    level_table = read_level_table(config.level_table)
    rows = []

    for root_file in root_files:
        try:
            row = analyse_root_file(
                root_file=root_file,
                level_table=level_table,
                dataframe=config.dataframe,
                gas_concentration=config.gas_concentration,
                level_csv_dir=config.level_csv_dir,
                save_level_csvs=save_level_csvs,
                normalized_by=normalized_by,
                use_poisson_error=use_poisson_error,
                print_gain_distributions=print_gain_distributions,
            )
            rows.append(row)
            print(f"✅ {config.mixture}: {root_file.name}")
        except Exception as exc:
            print(f"[ERROR] {config.mixture}: no pude procesar {root_file.name}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    no_fill = {
        "file",
        "gas_mixture",
        "gas1",
        "gas2",
        "concentration_gas_1",
        "concentration_gas_2",
        "concentration",
        "electric_field",
        "gap_mm",
        "pressure",
        "npe",
        "ne",
        "ne_std",
        "ni",
        "ni_std",
        "n_entries",
    }
    fill_cols = [c for c in df.columns if c not in no_fill]
    df[fill_cols] = df[fill_cols].fillna(0)

    sort_cols = [c for c in ["concentration", "pressure", "gap_mm", "electric_field", "npe"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    config.population_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_general_csv, index=False)
    if {"Ar_1s4_1s5", "Ar_1s2_1s3", "Ar_dbleStar"}.issubset(df.columns):
        config.output_general_csv.with_suffix(config.output_general_csv.suffix + ".ar2nd_v2").write_text(
            "Ar2nd bins: [11.5,11.7), [11.7,12.0), [12.0,100.0) eV\n",
            encoding="utf-8",
        )

    base_cols = [
        "concentration",
        "electric_field",
        "gap_mm",
        "pressure",
        "npe",
        "ne",
        "ni",
        "ne_std",
        "ni_std",
        "n_entries",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    if SAVE_SPLIT_POPULATIONS:
        for col in config.dataframe.columns:
            out_name = config.dataframe.loc["name output", col]
            cols = base_cols + [out_name]
            err_col = f"Err{out_name}"
            if err_col in df.columns:
                cols.append(err_col)
            df[cols].to_csv(config.population_dir / f"{out_name}.csv", index=False)

    print(f"📄 Guardado resumen: {config.output_general_csv.relative_to(ROOT_DIR)}")
    return df


def main() -> None:
    for config in REGISTERED_RUNS:
        try:
            analyse_secondary_run(config)
        except FileNotFoundError as exc:
            print(f"[skip] {config.mixture}: {exc}")


if __name__ == "__main__":
    main()
