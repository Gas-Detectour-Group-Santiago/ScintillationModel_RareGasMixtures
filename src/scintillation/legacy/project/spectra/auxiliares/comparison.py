from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import (
    ensure_parent,
    generated_to_raw_factor,
    match_float,
    peak_height,
    raw_unit_factor,
    smooth,
)
from .io import read_raw_csv, select_raw_spectrum
from .plotting import plot_comparison_mosaic


def total_generated(generated: pd.DataFrame, gas: str, pressure: float, concentration: float) -> pd.DataFrame:
    return generated[
        (generated["gas_mixture"] == gas)
        & (generated["component"] == "total")
        & match_float(generated["pressure_bar"], pressure)
        & match_float(generated["concentration_percent"], concentration)
    ].sort_values("wavelength_nm").copy()


def unit_scaled_pair(
    project_root: Path,
    gas: str,
    concentration: float,
    raw_y: np.ndarray,
    gen_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = cfg.COMPARISON_UNIT_SCALING
    if mode == "raw_to_generated":
        return raw_y * raw_unit_factor(project_root, gas, concentration), gen_y, r"ph MeV$^{-1}$ nm$^{-1}$"
    if mode == "generated_to_raw":
        return raw_y, gen_y * generated_to_raw_factor(project_root, gas, concentration), r"raw-like ph e$^{-1}$ nm$^{-1}$"
    if mode == "none":
        return raw_y, gen_y, "scaled intensity"
    raise ValueError(f"COMPARISON_UNIT_SCALING no válido: {mode!r}")


def anchor_factor(
    project_root: Path,
    gas: str,
    pressure: float,
    raw_cache: dict[str, pd.DataFrame],
    generated_by_gas: dict[str, pd.DataFrame],
) -> float:
    if not cfg.ANCHOR_MATCH_ENABLED:
        return 1.0
    anchor = cfg.ANCHORS[gas]
    concentration = float(anchor["concentration_percent"])
    window = tuple(anchor["window_nm"])

    raw = select_raw_spectrum(raw_cache[gas], gas=gas, concentration_percent=concentration, pressure_bar=pressure, spectrum_column=cfg.COMPARISON_SPECTRUM_COLUMN)
    gen = total_generated(generated_by_gas[gas], gas, pressure, concentration)
    if raw.empty or gen.empty:
        print(f"[spectra] aviso: sin anchor {gas} {pressure:g} bar {concentration:g}%; escala=1")
        return 1.0

    raw_y = smooth(raw["intensity_raw"].to_numpy(dtype=float), window=cfg.RAW_SMOOTH_WINDOW)
    gen_y = gen["intensity_ph_MeV_nm"].to_numpy(dtype=float)
    raw_scaled, gen_scaled, _ = unit_scaled_pair(project_root, gas, concentration, raw_y, gen_y)
    raw_peak = peak_height(raw["wavelength_nm"].to_numpy(dtype=float), raw_scaled, window)
    gen_peak = peak_height(gen["wavelength_nm"].to_numpy(dtype=float), gen_scaled, window)
    if not np.isfinite(raw_peak) or not np.isfinite(gen_peak) or raw_peak <= 0 or gen_peak <= 0:
        print(f"[spectra] aviso: anchor inválido {gas} {pressure:g} bar; escala=1")
        return 1.0

    if cfg.ANCHOR_SCALE_SIDE == "raw":
        return gen_peak / raw_peak
    if cfg.ANCHOR_SCALE_SIDE == "generated":
        return raw_peak / gen_peak
    raise ValueError(f"ANCHOR_SCALE_SIDE no válido: {cfg.ANCHOR_SCALE_SIDE!r}")


def _generated_total_groups(frame: pd.DataFrame, gas: str) -> dict[tuple[float, float], pd.DataFrame]:
    selected = frame.loc[(frame["gas_mixture"].astype(str) == gas) & (frame["component"].astype(str) == "total"),
                         ["pressure_bar", "concentration_percent", "wavelength_nm", "intensity_ph_MeV_nm"]].copy()
    return {
        (float(concentration), float(pressure)): group.sort_values("wavelength_nm").reset_index(drop=True)
        for (concentration, pressure), group in selected.groupby(["concentration_percent", "pressure_bar"], sort=False)
    }


def _raw_groups(frame: pd.DataFrame, gas: str) -> dict[tuple[float, float], pd.DataFrame]:
    selected = frame
    if "gas_mixture" in selected:
        selected = selected.loc[selected["gas_mixture"].astype(str) == gas]
    if "spectrum_column" in selected:
        selected = selected.loc[selected["spectrum_column"].astype(str) == cfg.COMPARISON_SPECTRUM_COLUMN]
    selected = selected[["pressure_bar", "concentration_percent", "wavelength_nm", "intensity_raw"]].copy()
    return {
        (float(concentration), float(pressure)): group.sort_values("wavelength_nm").reset_index(drop=True)
        for (concentration, pressure), group in selected.groupby(["concentration_percent", "pressure_bar"], sort=False)
    }


def comparison_dataframe(
    project_root: Path,
    generated_by_gas: dict[str, pd.DataFrame],
    spec: dict,
) -> pd.DataFrame:
    raw_cache = {gas: read_raw_csv(project_root, gas) for gas in spec["gases"]}
    raw_index = {gas: _raw_groups(raw_cache[gas], gas) for gas in spec["gases"]}
    generated_index = {gas: _generated_total_groups(generated_by_gas[gas], gas) for gas in spec["gases"]}

    anchor_scales: dict[tuple[str, float], float] = {}
    for gas in spec["gases"]:
        anchor = cfg.ANCHORS[gas]
        anchor_concentration = float(anchor["concentration_percent"])
        window = tuple(anchor["window_nm"])
        for pressure in spec["pressures_bar"]:
            raw = raw_index[gas].get((anchor_concentration, float(pressure)), pd.DataFrame())
            gen = generated_index[gas].get((anchor_concentration, float(pressure)), pd.DataFrame())
            factor = 1.0
            if cfg.ANCHOR_MATCH_ENABLED and not raw.empty and not gen.empty:
                raw_y = smooth(raw["intensity_raw"].to_numpy(dtype=float), window=cfg.RAW_SMOOTH_WINDOW)
                gen_y = gen["intensity_ph_MeV_nm"].to_numpy(dtype=float)
                raw_scaled, gen_scaled, _ = unit_scaled_pair(project_root, gas, anchor_concentration, raw_y, gen_y)
                raw_peak = peak_height(raw["wavelength_nm"].to_numpy(dtype=float), raw_scaled, window)
                gen_peak = peak_height(gen["wavelength_nm"].to_numpy(dtype=float), gen_scaled, window)
                if np.isfinite(raw_peak) and np.isfinite(gen_peak) and raw_peak > 0 and gen_peak > 0:
                    factor = gen_peak / raw_peak if cfg.ANCHOR_SCALE_SIDE == "raw" else raw_peak / gen_peak
            anchor_scales[(gas, float(pressure))] = factor

    rows = []
    for concentration in spec["concentrations_percent"]:
        for gas in spec["gases"]:
            for pressure in spec["pressures_bar"]:
                gen = generated_index[gas].get((float(concentration), float(pressure)), pd.DataFrame())
                raw = raw_index[gas].get((float(concentration), float(pressure)), pd.DataFrame())
                if gen.empty and raw.empty:
                    continue

                gen_w = gen["wavelength_nm"].to_numpy(dtype=float) if not gen.empty else np.array([], dtype=float)
                gen_y = gen["intensity_ph_MeV_nm"].to_numpy(dtype=float) if not gen.empty else np.array([], dtype=float)
                raw_w = raw["wavelength_nm"].to_numpy(dtype=float) if not raw.empty else np.array([], dtype=float)
                raw_y = smooth(raw["intensity_raw"].to_numpy(dtype=float), window=cfg.RAW_SMOOTH_WINDOW) if not raw.empty else np.array([], dtype=float)

                if not raw.empty and not gen.empty:
                    raw_scaled, gen_scaled, unit_label = unit_scaled_pair(project_root, gas, concentration, raw_y, gen_y)
                elif not raw.empty:
                    raw_scaled = raw_y.copy()
                    gen_scaled = np.array([], dtype=float)
                    unit_label = "raw intensity"
                else:
                    raw_scaled = np.array([], dtype=float)
                    gen_scaled = gen_y.copy()
                    unit_label = r"ph MeV$^{-1}$ nm$^{-1}$"

                factor = anchor_scales.get((gas, pressure), 1.0)
                if cfg.ANCHOR_MATCH_ENABLED:
                    if cfg.ANCHOR_SCALE_SIDE == "raw":
                        raw_scaled = raw_scaled * factor
                    elif cfg.ANCHOR_SCALE_SIDE == "generated":
                        gen_scaled = gen_scaled * factor

                if not gen.empty:
                    rows.append(pd.DataFrame({
                        "comparison": spec["name"],
                        "gas_mixture": gas,
                        "pressure_bar": pressure,
                        "concentration_percent": concentration,
                        "source": "generated",
                        "wavelength_nm": gen_w,
                        "plot_intensity": gen_scaled,
                        "unit_label": unit_label,
                        "anchor_scale": factor,
                    }))
                if not raw.empty:
                    rows.append(pd.DataFrame({
                        "comparison": spec["name"],
                        "gas_mixture": gas,
                        "pressure_bar": pressure,
                        "concentration_percent": concentration,
                        "source": "raw",
                        "wavelength_nm": raw_w,
                        "plot_intensity": raw_scaled,
                        "unit_label": unit_label,
                        "anchor_scale": factor,
                    }))
    if not rows:
        raise RuntimeError(f"Comparación vacía: {spec['name']}")
    return pd.concat(rows, ignore_index=True)


def run_comparison_mosaics(project_root: Path, outdir: Path, generated_by_gas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for spec in cfg.COMPARISON_PLOTS:
        print(f"[spectra] comparison {spec['name']}")
        df = comparison_dataframe(project_root, generated_by_gas, spec)
        csv_path = outdir / "csv" / str(spec["output_csv"])
        ensure_parent(csv_path)
        df.to_csv(csv_path, index=False)
        plot_comparison_mosaic(outdir, df, spec)
        out[str(spec["name"])] = df
    return out
