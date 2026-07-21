from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class DomainStatus(str, Enum):
    SIMULATED = "simulated"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"


@dataclass(frozen=True)
class ExtrapolationPolicy:
    variable: str
    method: str
    simulated_min: float
    simulated_max: float
    allow_below: bool = False
    allow_above: bool = False

    def classify(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        status = np.full(values.shape, DomainStatus.INTERPOLATED.value, dtype=object)
        outside = (values < self.simulated_min) | (values > self.simulated_max)
        status[outside] = DomainStatus.EXTRAPOLATED.value
        return status

    def validate(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if np.any(values < self.simulated_min) and not self.allow_below:
            raise ValueError(f"{self.variable}: extrapolation below {self.simulated_min} is disabled")
        if np.any(values > self.simulated_max) and not self.allow_above:
            raise ValueError(f"{self.variable}: extrapolation above {self.simulated_max} is disabled")
