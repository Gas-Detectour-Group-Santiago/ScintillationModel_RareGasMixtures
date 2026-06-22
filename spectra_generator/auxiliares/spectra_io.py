from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_file(path: Path, hint: str | None = None) -> Path:
    if path.exists():
        return path
    msg = f"No encuentro {path}"
    if hint:
        msg += f"\n{hint}"
    raise FileNotFoundError(msg)


def read_raw_spectra_csv(path: Path) -> pd.DataFrame:
    require_file(
        path,
        "Ejecuta primero: python data/Analysis_spectra.py",
    )
    df = pd.read_csv(path)
    required = {
        "gas_mixture",
        "concentration_percent",
        "concentration_fraction",
        "pressure_bar",
        "spectrum_name",
        "spectrum_column",
        "wavelength_nm",
        "intensity_raw",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} no tiene columnas esperadas: {missing}")
    return df


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

