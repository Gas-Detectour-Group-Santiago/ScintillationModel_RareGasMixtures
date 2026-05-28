from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import scienceplots


def _get_active_gases_from_filename(root_file):
    """
    Extrae pares gas-fracción del nombre del archivo.
    Ejemplo:
        ar_100.0_n2_0.0_60.0kVcm_... -> ["ar"]
        ar_90.0_n2_10.0_60.0kVcm_... -> ["ar", "n2"]

    Devuelve la lista de gases con fracción > 0.
    """
    tokens = root_file.stem.lower().split("_")
    active_gases = []

    i = 0
    while i < len(tokens) - 1:
        gas = _normalise_gas_name(tokens[i])

        try:
            frac = float(tokens[i + 1])
            if frac > 0 and gas is not None:
                active_gases.append(gas)
            i += 2
        except ValueError:
            i += 1

    return active_gases


def _apply_argon_gap_to_mapping(df):
    """
    Inserta hueco 3..7 y desplaza +5 todos los niveles > 2.
    Se usa cuando hay que corregir la ausencia de los estados de Ar.
    """
    df = df.copy()
    df.loc[df["level"] > 2, "level"] += 5

    gap_rows = []
    cols = list(df.columns)
    for lev in range(3, 8):
        row = {col: pd.NA for col in cols}
        row["level"] = lev
        gap_rows.append(row)

    gap_df = pd.DataFrame(gap_rows, columns=cols)
    df = pd.concat([df, gap_df], ignore_index=True)
    df = df.sort_values("level").reset_index(drop=True)
    return df

def _apply_he_gap_to_mapping(df):
    """
    Inserta hueco en el nivel 2 y desplaza +1 todos los niveles > 1.
    Se usa cuando hay que corregir la ausencia de un estado de He.

    Es decir:
        level 0 -> 0
        level 1 -> 1
        level 2 -> hueco
        level 2 original -> 3
        level 3 original -> 4
        ...
    """
    df = df.copy()

    # Desplazar todos los niveles posteriores al 1
    df.loc[df["level"] > 1, "level"] += 1

    # Crear fila hueca en level = 2
    cols = list(df.columns)
    gap_row = {col: pd.NA for col in cols}
    gap_row["level"] = 2

    gap_df = pd.DataFrame([gap_row], columns=cols)

    df = pd.concat([df, gap_df], ignore_index=True)
    df = df.sort_values("level").reset_index(drop=True)

    return df
def _build_mapping_table_for_file(table_df, active_gases, argon_update, helium_update=False):
    """
    Construye la tabla de mapeo que se usará para un archivo concreto.

    Reglas:
    - Gas puro:
        * filtra al gas activo
        * renumera desde 0
        * si es Ar y argon_update=True, inserta hueco 3..7
        * si es He y helium_update=True, inserta hueco 2
    - Mezcla:
        * filtra solo los gases activos
        * conserva numeración global de mezcla
        * si hay Ar y argon_update=True, inserta hueco 3..7 globalmente
        * si hay He y helium_update=True, inserta hueco 2 globalmente
    """
    base = table_df.copy()

    if active_gases:
        base = base[base["_gas_norm"].isin(active_gases)].copy()

    base = base.sort_values("level").reset_index(drop=True)

    # Gas puro
    if len(active_gases) == 1:
        pure_gas = active_gases[0]

        base = base[base["_gas_norm"] == pure_gas].copy()
        base = base.sort_values("level").reset_index(drop=True)

        # Renumerar desde 0 para gas puro
        base["level"] = np.arange(len(base), dtype=int)

        if pure_gas == "he" and helium_update:
            base = _apply_he_gap_to_mapping(base)

        if pure_gas == "ar" and argon_update:
            base = _apply_argon_gap_to_mapping(base)

        return base

    # Mezcla
    if len(active_gases) >= 2:
        if "he" in active_gases and helium_update:
            base = _apply_he_gap_to_mapping(base)

        if "ar" in active_gases and argon_update:
            base = _apply_argon_gap_to_mapping(base)

        return base

    # Caso residual
    return base


def export_hlevels_to_csv(
    folder_path,
    table_path,
    object_name="hLevels",
    argon_update=True
):
    """
    Lee el histograma hLevels de todos los ROOT de una carpeta, lo cruza con una
    tabla externa con el mapeo de niveles, y exporta un CSV enriquecido.

    Parámetros
    ----------
    folder_path : str o Path
        Carpeta donde están los archivos .root
    table_path : str o Path
        Ruta al CSV/tabla con columnas al menos:
            level, gas, state_name
        Opcionalmente puede tener también:
            type, energy_eV
    object_name : str
        Nombre del histograma ROOT a leer
    argon_update : bool
        Si True:
        - en mezclas con Ar: inserta hueco global 3..7
        - en Ar puro: renumera desde 0 e inserta hueco 3..7
    """
    argon_update = False
    folder = Path(folder_path)
    table_path = Path(table_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta válida: {folder}")

    if not table_path.is_file():
        raise FileNotFoundError(f"No existe la tabla de niveles: {table_path}")

    table_df = pd.read_csv(table_path)

    required_cols = {"level", "gas", "state_name"}
    missing_cols = required_cols - set(table_df.columns)
    if missing_cols:
        raise ValueError(
            f"La tabla {table_path} debe contener las columnas {required_cols}. "
            f"Faltan: {missing_cols}"
        )

    table_df = table_df.copy()
    table_df["level"] = pd.to_numeric(table_df["level"], errors="coerce")
    table_df = table_df.dropna(subset=["level"]).copy()
    table_df["level"] = table_df["level"].astype(int)
    table_df["_gas_norm"] = table_df["gas"].apply(_normalise_gas_name)

    root_files = sorted(folder.glob("*.root"))
    if not root_files:
        print(f"No se encontraron archivos .root en: {folder}")
        return []

    out_dir = (folder.parent / "csv").resolve()
    out_dir.mkdir(exist_ok=True)

    generated_csvs = []

    for root_file in root_files:
        try:
            with uproot.open(root_file) as f:
                keys = f.keys(cycle=False)

                if object_name not in keys:
                    print(f"[AVISO] '{object_name}' no existe en {root_file.name}")
                    continue

                h = f[object_name]
                values = h.values()

                active_gases = _get_active_gases_from_filename(root_file)

                mapping_df = _build_mapping_table_for_file(
                    table_df=table_df,
                    active_gases=active_gases,
                    argon_update=argon_update
                )

                hist_levels = np.arange(len(values), dtype=int)

                df = pd.DataFrame({
                    "level": hist_levels,
                    "n_events": values
                })

                # Cruce con tabla de niveles
                df = df.merge(mapping_df, how="left", on="level")

                # Asegurar niveles insertados aunque no estén en el histograma
                if not mapping_df.empty:
                    max_level = int(max(df["level"].max(), mapping_df["level"].max()))
                else:
                    max_level = int(df["level"].max())

                df = (
                    df.set_index("level")
                      .reindex(range(0, max_level + 1))
                      .reset_index()
                      .rename(columns={"index": "level"})
                )

                df["n_events"] = df["n_events"].fillna(0)

                # Reinyectar metadata del mapping
                meta_cols = [c for c in mapping_df.columns if c != "level"]
                if meta_cols:
                    meta_df = mapping_df[["level"] + meta_cols].drop_duplicates("level")
                    df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
                    df = df.merge(meta_df, how="left", on="level")

                # Si hay gases activos, vaciar gases no activos
                if active_gases and "_gas_norm" in df.columns:
                    mask_other_gas = df["_gas_norm"].notna() & (~df["_gas_norm"].isin(active_gases))
                    for col in ["gas", "state_name", "type", "energy_eV"]:
                        if col in df.columns:
                            df.loc[mask_other_gas, col] = pd.NA

                if "_gas_norm" in df.columns:
                    df = df.drop(columns=["_gas_norm"])

                preferred_order = [
                    "level",
                    "gas",
                    "state_name",
                    "type",
                    "energy_eV",
                    "n_events"
                ]
                final_cols = [c for c in preferred_order if c in df.columns] + \
                             [c for c in df.columns if c not in preferred_order]
                df = df[final_cols]

                csv_path = out_dir / f"{root_file.stem}.csv"
                df.to_csv(csv_path, index=False)
                generated_csvs.append(csv_path)

                if len(active_gases) == 1:
                    print(f"[OK] CSV generado: {csv_path.name} | gas puro: {active_gases[0]}")
                else:
                    print(f"[OK] CSV generado: {csv_path.name} | gases activos: {active_gases}")

        except Exception as e:
            print(f"[ERROR] No se pudo interpretar '{object_name}' en {root_file.name}: {e}")

    return generated_csvs

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


def _extract_metadata_from_filename(root_file, gas_concentration=None):
    """
    Extrae del nombre del archivo:
        - electric_field
        - pressure
        - gap
        - gas1
        - concentration_gas_1
        - gas2
        - concentration_gas_2
        - concentration   <-- concentración del gas pedido en gas_concentration

    Ejemplo:
        ar_90.0_n2_10.0_60.0kVcm_1.0bar_128um.root
    """
    stem = Path(root_file).stem
    tokens = stem.split("_")

    meta = {
        "electric_field": pd.NA,
        "pressure": pd.NA,
        "gap": pd.NA,
        "gas1": pd.NA,
        "concentration_gas_1": pd.NA,
        "gas2": pd.NA,
        "concentration_gas_2": pd.NA,
        "concentration": pd.NA,
    }

    gas_pairs = []
    i = 0
    while i < len(tokens) - 1:
        gas = _normalise_gas_name(tokens[i])
        try:
            frac = float(tokens[i + 1])
            if gas is not None:
                gas_pairs.append((gas, frac))
            i += 2
            continue
        except ValueError:
            pass

        tok = tokens[i].strip().lower()

        m_field = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(kvcm|kv/cm|vcm|v/cm)$", tok, flags=re.IGNORECASE)
        if m_field and pd.isna(meta["electric_field"]):
            meta["electric_field"] = float(m_field.group(1))
            i += 1
            continue

        m_pressure = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(bar|mbar|atm|torr)$", tok, flags=re.IGNORECASE)
        if m_pressure and pd.isna(meta["pressure"]):
            meta["pressure"] = float(m_pressure.group(1))
            i += 1
            continue

        m_gap = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(nm|um|mm|cm)$", tok, flags=re.IGNORECASE)
        if m_gap and pd.isna(meta["gap"]):
            meta["gap"] = float(m_gap.group(1))
            i += 1
            continue

        i += 1

    if len(gas_pairs) >= 1:
        meta["gas1"] = gas_pairs[0][0]
        meta["concentration_gas_1"] = gas_pairs[0][1]

    if len(gas_pairs) >= 2:
        meta["gas2"] = gas_pairs[1][0]
        meta["concentration_gas_2"] = gas_pairs[1][1]

    # Concentración del gas elegido explícitamente
    if gas_concentration is not None:
        gas_concentration = _normalise_gas_name(gas_concentration)
        concentration = pd.NA
        for gas, frac in gas_pairs:
            if gas == gas_concentration:
                concentration = frac
                break
        meta["concentration"] = concentration

    return meta


def read_data_per_primary_electron(
    folder_path,
    tree_name="dataPerPrimaryElectron",
    gas_concentration=None
):
    """
    Lee el árbol 'dataPerPrimaryElectron' de todos los ROOT de una carpeta,
    extrae nElectrons y nIons, calcula medias y desviaciones estándar, y además
    extrae metadata del nombre del archivo.

    Si gas_concentration se especifica, añade una columna 'concentration'
    con la fracción del gas pedido.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta válida: {folder}")

    root_files = sorted(folder.glob("*.root"))
    if not root_files:
        print(f"No se encontraron archivos .root en: {folder}")
        return pd.DataFrame()

    plot_dir = (folder / ".." / "gain_distribution").resolve()
    plot_dir.mkdir(exist_ok=True)

    summary_rows = []

    for root_file in root_files:
        try:
            with uproot.open(root_file) as f:
                if tree_name not in f:
                    print(f"[AVISO] '{tree_name}' no existe en {root_file.name}")
                    continue

                tree = f[tree_name]
                available_branches = tree.keys()

                if "nElectrons" not in available_branches or "nIons" not in available_branches:
                    print(
                        f"[AVISO] En {root_file.name} faltan ramas. "
                        f"Disponibles: {available_branches}"
                    )
                    continue

                arrays = tree.arrays(["nElectrons", "nIons"], library="np")
                ne = np.asarray(arrays["nElectrons"], dtype=float)
                ni = np.asarray(arrays["nIons"], dtype=float)

                ne_mean = np.mean(ne)
                ne_std = np.std(ne, ddof=1) if len(ne) > 1 else 0.0

                ni_mean = np.mean(ni)
                ni_std = np.std(ni, ddof=1) if len(ni) > 1 else 0.0

                meta = _extract_metadata_from_filename(
                    root_file,
                    gas_concentration=gas_concentration
                )

                summary_rows.append({
                    "file": root_file.name,
                    "electric_field": meta["electric_field"],
                    "pressure": meta["pressure"],
                    "gap": meta["gap"],
                    "gas1": meta["gas1"],
                    "concentration_gas_1": meta["concentration_gas_1"],
                    "gas2": meta["gas2"],
                    "concentration_gas_2": meta["concentration_gas_2"],
                    "concentration": meta["concentration"],
                    "ne_mean": ne_mean,
                    "ne_std": ne_std,
                    "ni_mean": ni_mean,
                    "ni_std": ni_std,
                    "n_entries": len(ne)
                })

                # fig, ax = plt.subplots(figsize=(6, 4))

                # plt.style.use(["grid"])        
                # ax.grid(True, which='major', alpha=0.3)
                # ax.grid(True, which='minor', alpha=0.08)
                    
                
                # ax.hist(ne, bins="auto", edgecolor="white")
                # ax.set_xlabel("Electrons Generated per Primary Electron")
                # ax.set_ylabel("Frequency")
                # ax.set_title(
                #     f"Electron Gain {meta['gas1']} {meta["concentration_gas_1"]}$\%$ - {meta['gas2']} {meta["concentration_gas_2"]}$\%$ "
                #     f"{meta['pressure']} bar {meta['electric_field']} kV/cm"
                # )

                # plt.tight_layout()

                # ax.text(
                #     0.97, 0.97,
                #     rf"Mean = {ne_mean:.2f}",
                #     transform=ax.transAxes,
                #     fontsize=11,
                #     ha='right',
                #     va='top',
                #     color='black',
                #     bbox=dict(
                #         boxstyle='round,pad=0.3',
                #         facecolor='white',
                #         edgecolor='0.7',
                #         linewidth=1.0
                #     )
                # )

                # plt.savefig(plot_dir / f"{root_file.stem}_ne.pdf", dpi=200)

                print(f"[OK] Procesado: {root_file.name}")

        except Exception as e:
            print(f"[ERROR] No se pudo procesar {root_file.name}: {e}")

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        preferred_order = [
            "file",
            "electric_field",
            "pressure",
            "gap",
            "gas1",
            "concentration_gas_1",
            "gas2",
            "concentration_gas_2",
            "concentration",
            "ne_mean",
            "ne_std",
            "ni_mean",
            "ni_std",
            "n_entries",
        ]
        summary_df = summary_df[[c for c in preferred_order if c in summary_df.columns]]

        summary_csv = (folder / ".." / "dataPerPrimaryElectron_summary.csv").resolve()
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Resumen guardado en: {summary_csv}")

    return summary_df