from __future__ import annotations

PH_PER_KEV_TO_PH_PER_MEV = 1000.0
PH_PER_EV_TO_PH_PER_MEV = 1.0e6


def scalar(value) -> float:
    import numpy as np

    return float(np.ravel(np.asarray(value, dtype=float))[0])

