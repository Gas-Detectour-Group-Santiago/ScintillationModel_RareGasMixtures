from __future__ import annotations

import importlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

"""
Analysis_spectra.py
-------------------
Script autónomo para extraer los espectros experimentales raw de los pickles
de Ar--CF4 y Ar--N2 y guardarlos como CSVs directamente utilizables con
pandas.read_csv / Typst load-csv.

Uso:
    python Analysis_spectra.py

Este script solo prepara los datos raw. No genera figuras, no anota líneas y
no llama a spectra_generator.

Formato de salida:
    data/Spectra/ArCF4_raw_spectra.csv.gz
    data/Spectra/ArN2_raw_spectra.csv.gz
    data/Spectra/raw_spectra.csv.gz (optional)

Cada CSV está en formato largo:
    gas_mixture, concentration_percent, concentration_fraction, pressure_bar,
    spectrum_name, spectrum_column, wavelength_nm, intensity_raw, ...

Si una fila del pickle contiene varias columnas espectrales válidas
(`mean_spectrum`, `C1_spectrum`, `C2_spectrum`, etc.), se exportan todas y se
distinguen por `spectrum_name`/`spectrum_column`.
"""


ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_DIR = ROOT_DIR / "Experimental"
SPECTRA_DIR = ROOT_DIR / "Spectra"


@dataclass(frozen=True)
class SpectraRunConfig:
    name: str
    gas_mixture: str
    pickle_path: Path
    output_csv: Path
    concentration_columns: tuple[str, ...]
    pressure_columns: tuple[str, ...]
    spectrum_columns: tuple[str, ...]
    concentration_output_name: str


RUNS: tuple[SpectraRunConfig, ...] = (
    SpectraRunConfig(
        name="ArCF4_raw_spectra",
        gas_mixture="ArCF4",
        pickle_path=EXPERIMENTAL_DIR / "ArCF4" / "CF4_data.pkl",
        output_csv=SPECTRA_DIR / "ArCF4_raw_spectra.csv.gz",
        concentration_columns=("concentracion", "concentration_CF4", "CF4 concentration (%)", "fCF4"),
        pressure_columns=("presion", "P (bar)", "pressure", "pressure_bar"),
        spectrum_columns=("mean_spectrum", "C1_spectrum", "C2_spectrum", "data(norm)"),
        concentration_output_name="fCF4",
    ),
    SpectraRunConfig(
        name="ArN2_raw_spectra",
        gas_mixture="ArN2",
        pickle_path=EXPERIMENTAL_DIR / "ArN2" / "N2_data.pkl",
        output_csv=SPECTRA_DIR / "ArN2_raw_spectra.csv.gz",
        concentration_columns=("N2 concentration (%)", "concentration_N2", "concentracion", "fN2"),
        pressure_columns=("P (bar)", "presion", "pressure", "pressure_bar"),
        spectrum_columns=(
            "mean_spectrum",
            "C1_spectrum",
            "C2_spectrum",
            "C1",
            "C2",
            "spectrum_new_cal",
            "spectrum_old_cal",
        ),
        concentration_output_name="fN2",
    ),
)


# =============================================================================
# COMPATIBILIDAD PICKLE / DILL
# =============================================================================


def patch_scipy_special_for_old_pickles() -> None:
    """
    Algunos pickles antiguos guardados con dill buscan símbolos dentro de
    scipy.special._special_ufuncs. Si scipy está disponible, inyectamos aliases.
    """
    try:
        import scipy.special  # type: ignore

        special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")
    except Exception:
        return

    for name in ("erf", "erfc", "erfi", "gamma", "lgamma", "wofz"):
        if not hasattr(special_ufuncs, name) and hasattr(scipy.special, name):
            setattr(special_ufuncs, name, getattr(scipy.special, name))


def load_pickle(path: Path):
    patch_scipy_special_for_old_pickles()

    try:
        import dill  # type: ignore

        loader = dill
    except ModuleNotFoundError:
        loader = pickle

    with path.open("rb") as f:
        return loader.load(f)


# =============================================================================
# EXTRACCIÓN DE ESPECTROS
# =============================================================================


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No encontré {what}. Probé {list(candidates)}. Columnas: {list(df.columns)}")


def is_nan_like(value) -> bool:
    try:
        out = pd.isna(value)
        return bool(out) if isinstance(out, (bool, np.bool_)) else False
    except Exception:
        return False


def scalar_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def extract_spectrum_arrays(value) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Devuelve wavelength/intensity si `value` parece un espectro raw.

    Los pickles usados hasta ahora guardan dicts del tipo:
        {"wavelength": array, "intensity": array}

    También aceptamos pares/listas de dos arrays por robustez.
    """
    if value is None or is_nan_like(value):
        return None

    if isinstance(value, dict):
        wave = value.get("wavelength", value.get("lambda", value.get("wavelength_nm")))
        intensity = value.get("intensity", value.get("raw", value.get("counts")))
        if wave is None or intensity is None:
            return None
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        wave, intensity = value[0], value[1]
    else:
        return None

    try:
        wave_arr = np.asarray(wave, dtype=float).ravel()
        intensity_arr = np.asarray(intensity, dtype=float).ravel()
    except Exception:
        return None

    n = min(len(wave_arr), len(intensity_arr))
    if n < 2:
        return None

    wave_arr = wave_arr[:n]
    intensity_arr = intensity_arr[:n]
    finite = np.isfinite(wave_arr) & np.isfinite(intensity_arr)
    if finite.sum() < 2:
        return None

    return wave_arr[finite], intensity_arr[finite]


def available_spectra(row: pd.Series, spectrum_columns: Iterable[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    spectra: list[tuple[str, np.ndarray, np.ndarray]] = []
    for col in spectrum_columns:
        if col not in row.index:
            continue
        arrays = extract_spectrum_arrays(row[col])
        if arrays is None:
            continue
        wavelength, intensity = arrays
        spectra.append((col, wavelength, intensity))
    return spectra


def infer_concentration_is_fraction(column_name: str, values: pd.Series) -> bool:
    """Infer whether a concentration column is stored as fraction or percent.

    The experimental pickles used by the old spectra scripts store columns
    such as ``concentracion`` and ``N2 concentration (%)`` in percent, even for
    values below one, e.g. 0.1 means 0.1 %, not 10 %.  Only explicit fraction
    columns (``fCF4``, ``fN2`` or names containing ``fraction``) are converted
    from 0..1 to percent.
    """
    name = column_name.lower().replace(" ", "")
    if "%" in column_name or "concentracion" in name or "concentration(%)" in name:
        return False
    if name in {"fcf4", "fn2"} or "fraction" in name:
        finite = pd.to_numeric(values, errors="coerce").dropna()
        return bool(len(finite) > 0 and finite.max() <= 1.0)
    return False


def normalise_concentration_percent(value: float, *, is_fraction: bool) -> float:
    value = scalar_or_nan(value)
    if not np.isfinite(value):
        return np.nan
    return value * 100.0 if is_fraction else value


def spectrum_rows_for_run(config: SpectraRunConfig) -> pd.DataFrame:
    df = load_pickle(config.pickle_path)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    conc_col = find_first_column(df, config.concentration_columns, "columna de concentración")
    pressure_col = find_first_column(df, config.pressure_columns, "columna de presión")
    concentration_is_fraction = infer_concentration_is_fraction(conc_col, df[conc_col])

    rows: list[dict[str, object]] = []

    for row_index, row in df.reset_index(drop=False).iterrows():
        concentration_percent = normalise_concentration_percent(
            row[conc_col],
            is_fraction=concentration_is_fraction,
        )
        pressure_bar = scalar_or_nan(row[pressure_col])
        spectra = available_spectra(row, config.spectrum_columns)
        if not spectra:
            continue

        concentration_fraction = concentration_percent / 100.0 if np.isfinite(concentration_percent) else np.nan
        source_index = row.get("index", row_index)

        for spectrum_column, wavelength, intensity in spectra:
            order = np.argsort(wavelength)
            wavelength = wavelength[order]
            intensity = intensity[order]
            spectrum_name = f"{config.gas_mixture}_{concentration_percent:g}pct_{pressure_bar:g}bar_{spectrum_column}"

            for point_index, (w, y) in enumerate(zip(wavelength, intensity, strict=False)):
                rows.append(
                    {
                        "gas_mixture": config.gas_mixture,
                        "source_pickle": str(config.pickle_path.relative_to(ROOT_DIR)),
                        "source_row": source_index,
                        "spectrum_name": spectrum_name,
                        "spectrum_column": spectrum_column,
                        config.concentration_output_name: concentration_fraction,
                        "concentration_percent": concentration_percent,
                        "concentration_fraction": concentration_fraction,
                        "pressure_bar": pressure_bar,
                        "point_index": point_index,
                        "wavelength_nm": float(w),
                        "intensity_raw": float(y),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    sort_cols = ["gas_mixture", "concentration_percent", "pressure_bar", "spectrum_column", "wavelength_nm"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def analyse_spectra_run(config: SpectraRunConfig) -> pd.DataFrame:
    out = spectrum_rows_for_run(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.output_csv, index=False, compression="gzip")
    n_spectra = out["spectrum_name"].nunique() if "spectrum_name" in out.columns else 0
    print(f"✅ {config.name}: {config.output_csv.relative_to(ROOT_DIR)} ({n_spectra} espectros, {len(out)} puntos)")
    return out


def main() -> None:
    outputs = [analyse_spectra_run(config) for config in RUNS]
    if os.environ.get("SCINTILLATION_EXPORT_COMBINED_SPECTRA", "0").lower() in {"1", "true", "yes", "on"}:
        combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        combined_path = SPECTRA_DIR / "raw_spectra.csv.gz"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False, compression="gzip")
        n_spectra = combined["spectrum_name"].nunique() if "spectrum_name" in combined.columns else 0
        print(f"📄 Guardado combinado: {combined_path.relative_to(ROOT_DIR)} ({n_spectra} espectros, {len(combined)} puntos)")


if __name__ == "__main__":
    main()
