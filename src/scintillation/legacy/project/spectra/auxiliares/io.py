from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import match_float, require_file


RAW_REQUIRED_COLUMNS = {
    "gas_mixture",
    "concentration_percent",
    "concentration_fraction",
    "pressure_bar",
    "spectrum_name",
    "spectrum_column",
    "wavelength_nm",
    "intensity_raw",
}


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}")


def _safe_dill_load(path: Path):
    """Load old experimental pickle files robustly."""
    path = Path(path)

    class CompatUnpicklerMixin:
        def find_class(self, module, name):  # pragma: no cover - depends on old pickle internals
            if module == "scipy.special._special_ufuncs":
                import scipy.special as sc

                if hasattr(sc, name):
                    return getattr(sc, name)
            return super().find_class(module, name)

    try:
        import dill

        class CompatUnpickler(CompatUnpicklerMixin, dill.Unpickler):
            pass

        with open(path, "rb") as handle:
            return CompatUnpickler(handle).load()
    except ModuleNotFoundError:
        import pickle

        class CompatUnpickler(CompatUnpicklerMixin, pickle.Unpickler):
            pass

        with open(path, "rb") as handle:
            return CompatUnpickler(handle).load()


def _spectrum_arrays_from_value(value) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(value, dict):
        if "wavelength" in value and "intensity" in value:
            return np.asarray(value["wavelength"], dtype=float), np.asarray(value["intensity"], dtype=float)
        if "wavelength_nm" in value and "intensity" in value:
            return np.asarray(value["wavelength_nm"], dtype=float), np.asarray(value["intensity"], dtype=float)
        if "x" in value and "y" in value:
            return np.asarray(value["x"], dtype=float), np.asarray(value["y"], dtype=float)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return np.asarray(value[0], dtype=float), np.asarray(value[1], dtype=float)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2 and 2 in arr.shape:
            if arr.shape[0] == 2:
                return arr[0], arr[1]
            return arr[:, 0], arr[:, 1]
    return None


def _get_spectrum_arrays(row: pd.Series, preferred_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    for column in preferred_columns:
        if column not in row.index:
            continue
        parsed = _spectrum_arrays_from_value(row[column])
        if parsed is None:
            continue
        wavelength, intensity = parsed
        if wavelength.size == intensity.size and wavelength.size > 0:
            return wavelength, intensity
    return None


def _raw_pickle_candidates(project_root: Path, gas: str) -> list[Path]:
    pickle_rel = cfg.RAW_PICKLE_FILES.get(gas)
    if pickle_rel is None:
        return []

    configured = Path(project_root) / pickle_rel
    base = configured.parent
    names = [configured.name]
    if configured.suffix == ".pkl":
        names.append(configured.stem)
    else:
        names.append(f"{configured.name}.pkl")

    if gas == "ArCF4":
        names.extend(["CF4_data", "CF4_data.pkl", "CF4_primary_data_final.pkl"])
    elif gas == "ArN2":
        names.extend(["N2_data", "N2_data.pkl", "N2_primary_data_final.pkl"])

    out: list[Path] = []
    seen: set[str] = set()
    for name in names:
        path = base / name
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _raw_pickle_to_long(project_root: Path, gas: str) -> pd.DataFrame | None:
    pickle_path = next((path for path in _raw_pickle_candidates(project_root, gas) if path.exists()), None)
    if pickle_path is None:
        return None

    loaded = _safe_dill_load(pickle_path)
    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(f"{pickle_path} no contiene un pandas.DataFrame, sino {type(loaded)!r}")

    df = loaded.copy()
    conc_col = _first_existing_column(df, tuple(cfg.RAW_CONCENTRATION_COLUMNS[gas]))
    pressure_col = _first_existing_column(df, tuple(cfg.RAW_PRESSURE_COLUMNS[gas]))

    rows: list[pd.DataFrame] = []
    for source_row, row in df.iterrows():
        concentration = float(row[conc_col])
        pressure = float(row[pressure_col])
        for output_column in cfg.SPECTRUM_COLUMNS:
            candidates = tuple(cfg.RAW_SPECTRUM_COLUMN_CANDIDATES.get(output_column, (output_column,)))
            arrays = _get_spectrum_arrays(row, candidates)
            if arrays is None:
                continue
            wavelength, intensity = arrays
            rows.append(
                pd.DataFrame(
                    {
                        "gas_mixture": gas,
                        "source_pickle": str(pickle_path),
                        "source_row": source_row,
                        "spectrum_name": f"{gas}_{concentration:g}pct_{pressure:g}bar_row{source_row}_{output_column}",
                        "spectrum_column": output_column,
                        "concentration_percent": concentration,
                        "concentration_fraction": concentration / 100.0,
                        "pressure_bar": pressure,
                        "point_index": np.arange(wavelength.size, dtype=int),
                        "wavelength_nm": wavelength,
                        "intensity_raw": intensity,
                    }
                )
            )

    if not rows:
        raise RuntimeError(f"No pude extraer espectros de {pickle_path}")

    return pd.concat(rows, ignore_index=True)


def _read_raw_long(project_root: Path, gas: str) -> pd.DataFrame:
    if getattr(cfg, "RAW_PREFER_PICKLES", False):
        from_pickle = _raw_pickle_to_long(project_root, gas)
        if from_pickle is not None:
            return from_pickle

    path = Path(project_root) / cfg.GASES[gas].raw_csv
    df = pd.read_csv(require_file(path))
    missing = sorted(RAW_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"{path} no tiene columnas esperadas: {missing}")
    return df


def read_raw_csv(project_root: Path, gas: str, *, aggregate: bool = True) -> pd.DataFrame:
    """
    Read raw spectra and collapse repeated spectra when requested.

    Preferred source: the original experimental pickles, because they contain the
    Genaro concentration grid (0, 0.1, 0.5, 1, 5, 10, 20, 50, 100).  The older
    data/Spectra CSVs are used only as fallback.
    """
    df = _read_raw_long(project_root, gas)
    missing = sorted(RAW_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"raw input for {gas} no tiene columnas esperadas: {missing}")

    df = df[(df["gas_mixture"] == gas) & df["spectrum_column"].isin(cfg.SPECTRUM_COLUMNS)].copy()
    if df.empty:
        return df

    df["concentration_percent"] = df["concentration_percent"].astype(float)
    df["concentration_fraction"] = df["concentration_fraction"].astype(float)
    df["pressure_bar"] = df["pressure_bar"].astype(float)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)
    df["intensity_raw"] = df["intensity_raw"].astype(float)
    df["wavelength_nm_rounded"] = df["wavelength_nm"].round(int(cfg.RAW_WAVELENGTH_ROUND_DECIMALS))

    if not aggregate or not cfg.RAW_AGGREGATE_REPLICATES:
        df["intensity_mean"] = df["intensity_raw"]
        df["intensity_std"] = 0.0
        df["intensity_sem"] = 0.0
        df["n_replicates"] = 1
        df["source_rows"] = df.get("source_row", pd.Series(index=df.index, dtype="object")).astype(str)
        return df.sort_values(
            ["concentration_percent", "pressure_bar", "spectrum_column", "spectrum_name", "wavelength_nm"]
        ).reset_index(drop=True)

    return aggregate_raw_spectra(df)


def aggregate_raw_spectra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average repeated raw spectra condition by condition.

    Output invariant: for each gas/concentration/pressure/spectrum_column and
    wavelength there is only one row. ``intensity_raw`` is kept as an alias of
    ``intensity_mean`` so all plotting/comparison code can use the same column.
    """
    group_cols = [
        "gas_mixture",
        "concentration_percent",
        "concentration_fraction",
        "pressure_bar",
        "spectrum_column",
        "wavelength_nm_rounded",
    ]

    replicate_column = "source_row" if "source_row" in df.columns else "spectrum_name"
    out = (
        df.groupby(group_cols, as_index=False, sort=False)
        .agg(
            wavelength_nm=("wavelength_nm", "mean"),
            intensity_mean=("intensity_raw", "mean"),
            intensity_std=("intensity_raw", "std"),
            n_points_averaged=("intensity_raw", "size"),
            n_replicates=(replicate_column, "nunique"),
        )
    )
    out["n_replicates"] = out["n_replicates"].astype(int)
    out["intensity_std"] = out["intensity_std"].fillna(0.0)
    n = out["n_points_averaged"].to_numpy(dtype=float)
    std = out["intensity_std"].to_numpy(dtype=float)
    out["intensity_sem"] = np.where(n > 0, std / np.sqrt(n), 0.0)
    out["intensity_raw"] = out["intensity_mean"]
    out = out.drop(columns=["wavelength_nm_rounded"])
    return out.sort_values(
        ["gas_mixture", "concentration_percent", "pressure_bar", "spectrum_column", "wavelength_nm"]
    ).reset_index(drop=True)


def select_raw_spectrum(
    raw_df: pd.DataFrame,
    *,
    gas: str,
    concentration_percent: float,
    pressure_bar: float,
    spectrum_column: str | None = None,
) -> pd.DataFrame:
    column = spectrum_column or cfg.RAW_PLOT_SPECTRUM_COLUMN
    sub = raw_df[
        (raw_df["gas_mixture"] == gas)
        & match_float(raw_df["concentration_percent"], concentration_percent)
        & match_float(raw_df["pressure_bar"], pressure_bar)
        & (raw_df["spectrum_column"] == column)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.sort_values("wavelength_nm").reset_index(drop=True)
