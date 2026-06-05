"""
Utilities for building and comparing primary scintillation spectra.

Unit convention used here
-------------------------
The experimental integrated yields in the pickles/CSV files are treated as
photons per primary electron (ph/e-). The primary-fit CSVs were built by
multiplying those yields by 1/W(f), so the fitted model functions return
ph/eV. Therefore:

    model ph/eV/nm      -> ph/MeV/nm : multiply by 1e6
    exp. ph/e-/nm       -> ph/MeV/nm : multiply by 1e6 / W(f)

The raw experimental spectra stored in the pickle files are used as spectral
shapes. They are renormalised to the corresponding integrated yield before the
W(f) conversion is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

EV_PER_MEV = 1.0e6


def repo_root_from_script(script_file: str | Path) -> Path:
    """Return the repository root assuming scripts live in spectra_generator/."""
    return Path(script_file).resolve().parents[1]


def setup_science_style(use_grid: bool = False) -> None:
    """Apply a robust matplotlib style without failing if scienceplots is absent."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401  # registers the style
        plt.style.use(["science", "grid"] if use_grid else ["science"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
    })


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normalised Gaussian PDF, so integral over wavelength is one."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def weighted_gaussian_sum(
    wavelength: np.ndarray,
    total_yield: float,
    peaks: Iterable[tuple[float, float, float]],
) -> np.ndarray:
    """
    Spread an integrated yield over several normalised Gaussian bands.

    Parameters
    ----------
    wavelength:
        Wavelength grid in nm.
    total_yield:
        Integrated yield in the desired units.
    peaks:
        Iterable of (mu_nm, sigma_nm, relative_weight). Relative weights are
        automatically normalised to sum to one.
    """
    peaks = list(peaks)
    weight_sum = float(np.sum([p[2] for p in peaks]))
    if weight_sum <= 0:
        raise ValueError("The sum of peak weights must be positive.")

    out = np.zeros_like(np.asarray(wavelength, dtype=float), dtype=float)
    for mu, sigma, weight in peaks:
        out += (weight / weight_sum) * total_yield * gaussian_pdf(wavelength, mu, sigma)
    return out


def model_fit_unit_to_ph_per_MeV(y_ph_per_eV: Any) -> np.ndarray:
    """Convert model output from ph/eV to ph/MeV."""
    return np.asarray(y_ph_per_eV, dtype=float) * EV_PER_MEV


def ph_per_electron_to_ph_per_MeV(y_ph_per_electron: Any, additive_fraction: float, w_func) -> np.ndarray:
    """Convert ph/e- to ph/MeV using W(f) in eV/e-."""
    w_value = np.asarray(w_func(additive_fraction), dtype=float)
    return np.asarray(y_ph_per_electron, dtype=float) * EV_PER_MEV / w_value


def spectrum_shape_to_ph_per_MeV_nm(
    wavelength: Any,
    raw_intensity: Any,
    total_yield_ph_per_electron: float,
    additive_fraction: float,
    w_func,
) -> np.ndarray:
    """
    Convert a raw experimental spectrum shape to ph/MeV/nm.

    The raw spectrum is first normalised to unit area and then rescaled to the
    integrated experimental yield in ph/e-. This avoids relying on arbitrary
    normalisations stored in the pickle spectra.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    raw_intensity = np.asarray(raw_intensity, dtype=float)

    finite = np.isfinite(wavelength) & np.isfinite(raw_intensity)
    if finite.sum() < 2:
        return np.full_like(raw_intensity, np.nan, dtype=float)

    area = np.trapezoid(np.clip(raw_intensity[finite], 0.0, None), wavelength[finite])
    if area <= 0:
        return np.full_like(raw_intensity, np.nan, dtype=float)

    intensity_ph_per_e_nm = np.clip(raw_intensity, 0.0, None) * total_yield_ph_per_electron / area
    return ph_per_electron_to_ph_per_MeV(intensity_ph_per_e_nm, additive_fraction, w_func)


def safe_sum(values: Iterable[Any]) -> float:
    """Sum finite scalar values, ignoring None/NaN and nested non-scalars."""
    total = 0.0
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v):
            total += v
    return total


def get_arcf4_total_yield_ph_per_electron(row) -> float:
    """Total Ar-CF4 experimental yield in ph/e- from the pickle row."""
    yz = row.get("yields_zonas", {})
    if not isinstance(yz, dict):
        return np.nan

    total = safe_sum([yz.get("UV"), yz.get("vis")])
    ir = yz.get("ir", {})
    if isinstance(ir, dict):
        total += safe_sum(ir.values())
    return total


def get_n2_total_yield_ph_per_electron(row, include_ir: bool = True) -> float:
    """Total Ar-N2 experimental yield in ph/e- from the pickle row."""
    peaks = row.get("yields_picos", {})
    if isinstance(peaks, dict) and len(peaks) > 0:
        if include_ir:
            return safe_sum(peaks.values())
        return safe_sum(v for k, v in peaks.items() if float(k) < 500.0)

    # Fallback: UV/N2 yield only.
    try:
        return float(row.get("yield_N2", np.nan))
    except Exception:
        return np.nan


def get_spectrum_arrays(row, preferred_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Extract wavelength/intensity arrays from a pickle row."""
    for col in preferred_columns:
        if col not in row.index:
            continue
        dic = row[col]
        if isinstance(dic, dict) and "wavelength" in dic and "intensity" in dic:
            return np.asarray(dic["wavelength"], dtype=float), np.asarray(dic["intensity"], dtype=float)
    raise KeyError(f"No valid spectrum found. Tried columns: {preferred_columns}")


class CompatUnpicklerMixin:
    """Mixin used by make_compat_unpickler to tolerate scipy pickle moves."""

    def find_class(self, module, name):  # pragma: no cover - depends on pickle contents
        if module == "scipy.special._special_ufuncs":
            import scipy.special as sc
            if hasattr(sc, name):
                return getattr(sc, name)
        return super().find_class(module, name)


def safe_dill_load(path: str | Path):
    """Load old dill pickles robustly."""
    import dill

    class CompatUnpickler(CompatUnpicklerMixin, dill.Unpickler):
        pass

    with open(path, "rb") as f:
        return CompatUnpickler(f).load()
