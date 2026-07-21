from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from spectra import config as cfg


@dataclass(frozen=True)
class Peak:
    center_nm: float
    sigma_nm: float
    weight: float = 1.0


CF4_UV_PEAKS = (
    Peak(235.0, 17.0, 0.55),
    Peak(290.0, 17.0, 0.75),
    Peak(364.0, 50.0, 0.35),
)

AR_3RD_UV_PEAKS = (
    Peak(176.0, 30.0, 1.0),
    Peak(188.0, 30.0, 1.0),
    Peak(199.0, 30.0, 1.0),
    Peak(212.0, 30.0, 1.0),
    Peak(225.0, 30.0, 1.0),
    Peak(245.0, 30.0, 1.0),
)

N2_SECOND_POSITIVE_PEAKS = (
    Peak(335.0, 3.75, 0.42),
    Peak(355.0, 3.75, 0.30),
    Peak(378.0, 3.75, 0.10),
    Peak(403.0, 3.75, 0.05),
)


def find_project_root(start: str | Path) -> Path:
    start = Path(start).resolve()
    candidates = [start if start.is_dir() else start.parent, *(start.parents)]
    for parent in candidates:
        if (parent / "data").is_dir() and (parent / "models").is_dir():
            return parent
    raise RuntimeError(f"No encuentro la raíz del proyecto desde {start}")


def output_dir(project_root: Path) -> Path:
    out = Path(project_root) / cfg.OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    return path


def add_models_to_path(project_root: Path) -> None:
    models_path = str(Path(project_root) / "models")
    if models_path not in sys.path:
        sys.path.insert(0, models_path)


def match_float(series: pd.Series, value: float, atol: float = 1.0e-9) -> pd.Series:
    return np.isclose(series.astype(float), float(value), rtol=0.0, atol=atol)


def gaussian_pdf(wavelength_nm: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(wavelength_nm, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def weighted_gaussian_sum(wavelength_nm: np.ndarray, total_yield: float, peaks: Iterable[Peak]) -> np.ndarray:
    peaks = tuple(peaks)
    weights = np.asarray([p.weight for p in peaks], dtype=float)
    weights = weights / np.sum(weights)
    out = np.zeros_like(wavelength_nm, dtype=float)
    for peak, weight in zip(peaks, weights, strict=False):
        out += float(total_yield) * float(weight) * gaussian_pdf(wavelength_nm, peak.center_nm, peak.sigma_nm)
    return out


def read_parameter_vector(path: Path) -> np.ndarray:
    df = pd.read_csv(require_file(path))
    if "parameter" in df.columns:
        return df["parameter"].to_numpy(dtype=float)
    if "value" in df.columns:
        return df["value"].to_numpy(dtype=float)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError(f"No encuentro columna numérica de parámetros en {path}")
    return numeric.iloc[:, 0].to_numpy(dtype=float)


def setup_plot_style() -> None:
    from plot_style import setup_style
    setup_style(grid=False, use_latex=False, context="spectra")


def colors(n: int):
    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(n, 2)))


def smooth(y: np.ndarray, window: int = 1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = int(window)
    if window <= 1 or y.size < 3:
        return y
    if window % 2 == 0:
        window += 1
    window = min(window, y.size if y.size % 2 else y.size - 1)
    if window < 3:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def peak_height(wavelength: np.ndarray, y: np.ndarray, window: tuple[float, float]) -> float:
    mask = np.isfinite(wavelength) & np.isfinite(y) & (wavelength >= window[0]) & (wavelength <= window[1])
    if mask.sum() == 0:
        return np.nan
    value = float(np.nanmax(np.clip(y[mask], 0.0, None)))
    return value if value > 0.0 else np.nan


def w_value(project_root: Path, gas: str, concentration_percent: float) -> float:
    add_models_to_path(project_root)
    f = float(concentration_percent) / 100.0
    if gas == "ArCF4":
        from ArCF4 import ion_potential

        return float(np.asarray(ion_potential(f), dtype=float))
    if gas == "ArN2":
        from ArN2 import W_ArN2

        return float(np.asarray(W_ArN2(f), dtype=float))
    raise ValueError(f"Gas no soportado: {gas}")


def raw_unit_factor(project_root: Path, gas: str, concentration_percent: float) -> float:
    w = w_value(project_root, gas, concentration_percent)
    norm = 1.0
    if cfg.USE_NNORM_IN_RAW_UNIT_SCALING:
        norm_path = Path(project_root) / cfg.GASES[gas].norm_parameter_csv
        norm = float(read_parameter_vector(norm_path)[0])
    if w <= 0.0 or norm == 0.0:
        raise ValueError(f"W o Nnorm inválido: gas={gas}, W={w}, Nnorm={norm}")
    return float(cfg.X_RAY_ENERGY_EV) / w / norm


def generated_to_raw_factor(project_root: Path, gas: str, concentration_percent: float) -> float:
    return 1.0 / raw_unit_factor(project_root, gas, concentration_percent)
