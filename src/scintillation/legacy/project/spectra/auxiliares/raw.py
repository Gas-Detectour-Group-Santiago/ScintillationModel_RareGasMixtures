from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectra import config as cfg
from .common import ensure_parent, match_float
from .io import read_raw_csv, select_raw_spectrum
from .plotting import plot_raw_mosaic


def raw_aggregated_dataframe(project_root: Path, gas: str) -> pd.DataFrame:
    """Return the processed raw spectra for all configured spectrum columns."""
    return read_raw_csv(project_root, gas, aggregate=True)


def raw_mosaic_dataframe(all_columns: pd.DataFrame, gas: str) -> pd.DataFrame:
    """
    Data used by the raw mosaic for one gas.

    The full aggregated CSV keeps mean_spectrum, C1_spectrum and C2_spectrum.
    The mosaic itself uses cfg.RAW_PLOT_SPECTRUM_COLUMN.
    """
    df = all_columns[all_columns["spectrum_column"] == cfg.RAW_PLOT_SPECTRUM_COLUMN].copy()
    conc_mask = pd.Series(False, index=df.index)
    for concentration in cfg.RAW_CONCENTRATIONS_PERCENT:
        conc_mask |= match_float(df["concentration_percent"], concentration)
    pressure_mask = pd.Series(False, index=df.index)
    for pressure in cfg.RAW_PRESSURES_BAR:
        pressure_mask |= match_float(df["pressure_bar"], pressure)
    return df[conc_mask & pressure_mask].copy().sort_values(
        ["concentration_percent", "pressure_bar", "spectrum_column", "wavelength_nm"]
    )


def raw_reference_dataframe(all_raw: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Fixed Ar--CF4 95/5 reference drawn over every raw mosaic."""
    arcf4 = all_raw.get("ArCF4")
    if arcf4 is None or arcf4.empty:
        return None
    return select_raw_spectrum(
        arcf4,
        gas="ArCF4",
        concentration_percent=cfg.RAW_REFERENCE_CONCENTRATION_PERCENT,
        pressure_bar=cfg.RAW_REFERENCE_PRESSURE_BAR,
        spectrum_column=cfg.RAW_PLOT_SPECTRUM_COLUMN,
    )


def run_raw_mosaics(project_root: Path, outdir: Path) -> dict[str, pd.DataFrame]:
    all_raw: dict[str, pd.DataFrame] = {}

    for gas in cfg.GASES:
        all_columns = raw_aggregated_dataframe(project_root, gas)
        if all_columns.empty:
            continue
        aggregated_csv = outdir / "csv" / f"{gas}_raw_spectra_aggregated_C1_C2_mean.csv"
        ensure_parent(aggregated_csv)
        all_columns.to_csv(aggregated_csv, index=False)
        print(f"[spectra] raw aggregated CSV: {aggregated_csv}")
        all_raw[gas] = all_columns

    ref = raw_reference_dataframe(all_raw)

    for gas, all_columns in all_raw.items():
        df = raw_mosaic_dataframe(all_columns, gas)
        if df.empty:
            print(f"[spectra] aviso: raw mosaic vacío para {gas}")
            continue
        csv_path = outdir / "csv" / f"{gas}_raw_{cfg.RAW_PLOT_SPECTRUM_COLUMN}_mosaic.csv"
        ensure_parent(csv_path)
        df.to_csv(csv_path, index=False)
        plot_raw_mosaic(outdir, gas, df, reference_raw=ref)

    return all_raw
