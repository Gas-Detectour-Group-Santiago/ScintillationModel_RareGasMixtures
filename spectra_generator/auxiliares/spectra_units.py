from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .spectra_types import GaussianPeak


KEV_PER_MEV = 1.0e3


def project_root_from_file(path: str | Path) -> Path:
    start = Path(path).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "data").is_dir() and (parent / "models").is_dir():
            return parent
    raise RuntimeError(f"No encuentro la raíz del proyecto desde {start}")


def setup_science_style(use_grid: bool = False) -> None:
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ModuleNotFoundError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "axes.grid": use_grid,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.prop_cycle": plt.cycler(color=plt.get_cmap("viridis")(np.linspace(0.12, 0.88, 8))),
        }
    )


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def weighted_gaussian_sum(
    wavelength_nm: np.ndarray,
    total_yield: float,
    peaks: list[GaussianPeak],
) -> np.ndarray:
    weights = np.asarray([p.weight for p in peaks], dtype=float)
    weights = weights / np.sum(weights)

    out = np.zeros_like(wavelength_nm, dtype=float)
    for peak, weight in zip(peaks, weights, strict=False):
        out += float(total_yield) * float(weight) * gaussian_pdf(wavelength_nm, peak.center_nm, peak.sigma_nm)
    return out


def model_fit_unit_to_ph_per_MeV(values) -> np.ndarray:
    """
    Reproduce la conversión usada en los scripts de espectros anteriores.

    Los fits primarios exportan yields por keV; para las tablas/figuras de
    espectros se expresan como ph/MeV, por eso el factor es 1e3.
    """
    return np.asarray(values, dtype=float) * KEV_PER_MEV


def read_parameter_vector(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "parameter" in df.columns:
        return df["parameter"].to_numpy(dtype=float)
    if "value" in df.columns:
        return df["value"].to_numpy(dtype=float)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError(f"No encuentro columna numérica de parámetros en {path}")
    return numeric.iloc[:, 0].to_numpy(dtype=float)


def first_finite_max(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.nanmax(finite))


def match_float(series: pd.Series, value: float, atol: float = 1.0e-9) -> pd.Series:
    return np.isclose(series.astype(float), float(value), atol=atol, rtol=0.0)


def raw_to_ph_per_MeV_nm(raw_intensity: np.ndarray, additive_fraction: float, w_function, norm: float) -> np.ndarray:
    w_value = float(np.asarray(w_function(float(additive_fraction)), dtype=float))
    if not np.isfinite(w_value) or w_value <= 0.0:
        raise ValueError(f"W inválido para fracción {additive_fraction}: {w_value}")
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Normalización inválida: {norm}")
    return np.asarray(raw_intensity, dtype=float) * 1.0e6 / w_value / norm

