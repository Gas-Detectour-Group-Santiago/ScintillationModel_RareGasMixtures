from __future__ import annotations

from typing import Any, Iterable

import numpy as np

EV_PER_MEV = 1.0e6


def safe_sum(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v):
            total += v
    return total


def ph_per_electron_to_ph_per_MeV(
    y_ph_per_electron: Any,
    additive_fraction: float,
    w_func,
) -> np.ndarray:
    """Convert ph/e-/nm into ph/MeV/nm using W(f) in eV/e-."""
    w_value = np.asarray(w_func(float(additive_fraction)), dtype=float)
    return np.asarray(y_ph_per_electron, dtype=float) * EV_PER_MEV / w_value


def spectrum_shape_to_ph_per_e_nm(
    wavelength_nm: Any,
    raw_intensity: Any,
    total_yield_ph_per_electron: float,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Normalise an arbitrary raw spectrum shape to ph/e-/nm."""
    wavelength = np.asarray(wavelength_nm, dtype=float)
    raw = np.asarray(raw_intensity, dtype=float)

    finite = np.isfinite(wavelength) & np.isfinite(raw)
    out = np.full_like(raw, np.nan, dtype=float)
    if finite.sum() < 2 or not np.isfinite(total_yield_ph_per_electron):
        return out

    y_for_area = raw[finite]
    if clip_negative:
        y_for_area = np.clip(y_for_area, 0.0, None)

    order = np.argsort(wavelength[finite])
    w_sorted = wavelength[finite][order]
    y_sorted = y_for_area[order]
    area = float(np.trapezoid(y_sorted, w_sorted))

    if area <= 0.0 or not np.isfinite(area):
        return out

    raw_scaled = np.clip(raw, 0.0, None) if clip_negative else raw
    out = raw_scaled * float(total_yield_ph_per_electron) / area
    return out


def spectrum_shape_to_ph_per_MeV_nm(
    wavelength_nm: Any,
    raw_intensity: Any,
    total_yield_ph_per_electron: float,
    additive_fraction: float,
    w_func,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Normalise a raw spectrum shape to ph/MeV/nm."""
    y_ph_e_nm = spectrum_shape_to_ph_per_e_nm(
        wavelength_nm,
        raw_intensity,
        total_yield_ph_per_electron,
        clip_negative=clip_negative,
    )
    return ph_per_electron_to_ph_per_MeV(y_ph_e_nm, additive_fraction, w_func)


def get_arcf4_total_yield_ph_per_electron(row) -> float:
    """Total Ar-CF4 experimental yield in ph/e- from one pickle row."""
    yz = row.get("yields_zonas", {})
    if not isinstance(yz, dict):
        return np.nan

    total = safe_sum([yz.get("UV"), yz.get("vis")])
    ir = yz.get("ir", {})
    if isinstance(ir, dict):
        total += safe_sum(ir.values())
    return total


def get_n2_total_yield_ph_per_electron(row, *, include_ir: bool = True) -> float:
    """Total Ar-N2 experimental yield in ph/e- from one pickle row."""
    peaks = row.get("yields_picos", {})
    if isinstance(peaks, dict) and len(peaks) > 0:
        if include_ir:
            return safe_sum(peaks.values())
        return safe_sum(v for k, v in peaks.items() if _safe_float(k) < 500.0)

    try:
        return float(row.get("yield_N2", np.nan))
    except Exception:
        return np.nan


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan
