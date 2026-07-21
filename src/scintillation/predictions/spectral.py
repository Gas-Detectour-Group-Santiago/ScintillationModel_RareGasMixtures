from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .results import UncertaintyBand


@dataclass(frozen=True)
class SpectrumComponent:
    channel_id: str
    source: str
    wavelength_nm: np.ndarray
    spectral_yield: np.ndarray
    unit: str
    mixture_id: str = ""
    geometry: str | None = None
    bands: Mapping[str, UncertaintyBand] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_nm, dtype=float)
        values = np.asarray(self.spectral_yield, dtype=float)
        if wavelength.ndim != 1 or values.ndim != 1 or wavelength.shape != values.shape:
            raise ValueError("SpectrumComponent requires matching one-dimensional wavelength/yield arrays")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("Spectrum wavelength grid must be strictly increasing")
        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "spectral_yield", values)

    @property
    def integrated_yield(self) -> float:
        return float(np.trapezoid(self.spectral_yield, self.wavelength_nm))


@dataclass(frozen=True)
class SpectrumResult:
    wavelength_nm: np.ndarray
    spectral_yield: np.ndarray
    unit: str
    components: Sequence[SpectrumComponent]

    @classmethod
    def compose(cls, components: Sequence[SpectrumComponent], *, wavelength_nm: np.ndarray | None = None) -> "SpectrumResult":
        if not components:
            raise ValueError("At least one spectrum component is required")
        units = {component.unit for component in components}
        if len(units) != 1:
            raise ValueError(f"Cannot sum spectrum components with different units: {sorted(units)}")
        if wavelength_nm is None:
            wavelength_nm = np.unique(np.concatenate([c.wavelength_nm for c in components]))
        grid = np.asarray(wavelength_nm, dtype=float)
        total = np.zeros_like(grid)
        for component in components:
            total += np.interp(grid, component.wavelength_nm, component.spectral_yield, left=0.0, right=0.0)
        return cls(grid, total, next(iter(units)), tuple(components))
