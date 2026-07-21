from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np


class DomainStatus(str, Enum):
    SIMULATED = "simulated"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"


@dataclass(frozen=True)
class UncertaintyBand:
    lower: np.ndarray
    upper: np.ndarray
    source: str
    method: str = "toys"

    def __post_init__(self) -> None:
        low = np.asarray(self.lower, dtype=float)
        high = np.asarray(self.upper, dtype=float)
        if low.shape != high.shape:
            raise ValueError(f"Band {self.source!r}: lower/upper shapes differ")
        if np.any(low > high):
            raise ValueError(f"Band {self.source!r}: lower exceeds upper")
        object.__setattr__(self, "lower", low)
        object.__setattr__(self, "upper", high)


@dataclass(frozen=True)
class UncertaintyPolicy:
    displayed: tuple[str, ...] = ("stat", "syst", "total")
    total_components: tuple[str, ...] = ("stat", "syst")
    combination: str = "quadrature"

    def combine(self, central: np.ndarray, bands: Mapping[str, UncertaintyBand]) -> UncertaintyBand:
        central = np.asarray(central, dtype=float)
        selected = [bands[name] for name in self.total_components if name in bands]
        if not selected:
            return UncertaintyBand(central.copy(), central.copy(), "total", self.combination)
        down = [np.clip(central - band.lower, 0.0, None) for band in selected]
        up = [np.clip(band.upper - central, 0.0, None) for band in selected]
        if self.combination == "quadrature":
            down_total = np.sqrt(np.sum(np.square(down), axis=0))
            up_total = np.sqrt(np.sum(np.square(up), axis=0))
        elif self.combination == "envelope":
            down_total = np.max(np.stack(down), axis=0)
            up_total = np.max(np.stack(up), axis=0)
        else:
            raise ValueError(f"Unsupported uncertainty combination: {self.combination}")
        return UncertaintyBand(central - down_total, central + up_total, "total", self.combination)


@dataclass(frozen=True)
class PredictionResult:
    x: np.ndarray
    central: np.ndarray
    unit: str
    bands: Mapping[str, UncertaintyBand] = field(default_factory=dict)
    domain_status: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        central = np.asarray(self.central, dtype=float)
        if x.shape != central.shape:
            raise ValueError("Prediction x and central arrays must have the same shape")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "central", central)
        for name, band in self.bands.items():
            if band.lower.shape != central.shape:
                raise ValueError(f"Band {name!r} shape does not match central prediction")
        if self.domain_status is not None:
            status = np.asarray(self.domain_status, dtype=object)
            if status.shape != central.shape:
                raise ValueError("domain_status shape does not match central prediction")
            object.__setattr__(self, "domain_status", status)
