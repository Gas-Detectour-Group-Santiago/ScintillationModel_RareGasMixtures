"""Joint Ar--CF4 / Ar--N2 infrared model.

The same Ar-state optical weights and Ar self-quenching coefficients are used
for both mixtures.  Only the molecular-quenching coefficients are gas-specific.
The parameter vector contains five entries per line, in this order::

    PAr_star, tau_ns, K_Q_Ar, K_Q_CF4, K_Q_N2

for 696, 727, 750, 763 and 772 nm.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

IR_LINES = ("696", "727", "750", "763", "772")
TAUS_NS = {"696": 28.3, "727": 28.3, "750": 21.7, "763": 29.4, "772": 28.3}
ENERGY_XRAY_KEV = {"ArCF4": 15.0, "ArN2": 12.0}
Mixture = Literal["ArCF4", "ArN2"]


def parameter_offset(line: str) -> int:
    try:
        return 5 * IR_LINES.index(str(line))
    except ValueError as exc:
        raise KeyError(f"Unknown IR line {line!r}; expected one of {IR_LINES}.") from exc


def unpack_line_parameters(params, line: str) -> tuple[float, float, float, float, float]:
    x = np.asarray(params, dtype=float)
    i = parameter_offset(line)
    if x.size < i + 5:
        raise ValueError(f"Joint IR vector has {x.size} entries; at least {i + 5} are required.")
    return tuple(float(v) for v in x[i : i + 5])


def interpolate_population(degrad_data: pd.DataFrame, line: str, concentration):
    """PCHIP interpolation of the Degrad population at an exact concentration.

    The Degrad grids start at 0.1% admixture.  PCHIP therefore performs a very
    short extrapolation for the exact pure-Ar point f=0, instead of silently
    replacing an experimental concentration by a neighbouring grid point.
    """

    conc = pd.to_numeric(degrad_data["concentration"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(degrad_data[f"Ar_{line}"], errors="coerce").to_numpy(dtype=float)
    target = np.asarray(concentration, dtype=float)

    mask = np.isfinite(conc) & np.isfinite(values)
    conc = conc[mask]
    values = values[mask]
    if conc.size < 2:
        raise ValueError(f"Not enough Degrad points to interpolate Ar_{line}.")

    order = np.argsort(conc)
    conc = conc[order]
    values = values[order]
    conc, unique_idx = np.unique(conc, return_index=True)
    values = values[unique_idx]
    return PchipInterpolator(conc, values, extrapolate=True)(target)


def theory_yield_joint(
    params,
    degrad_data: pd.DataFrame,
    concentration,
    pressure_bar: float,
    *,
    mixture: Mixture,
    line: str,
):
    """Return the raw fitted yield for one line.

    Concentration is a fraction (0--1), pressure is in bar, tau is in ns and
    the effective K values use the same 1/ns/bar convention as the legacy IR
    models.  The result keeps the legacy ``per incident keV`` convention; the
    prediction layer converts it to ph/MeV with the selected Nnorm.
    """

    f = np.asarray(concentration, dtype=float)
    p = float(pressure_bar)
    P, tau, k_ar, k_cf4, k_n2 = unpack_line_parameters(params, line)
    k_mol = k_cf4 if mixture == "ArCF4" else k_n2

    population = interpolate_population(degrad_data, line, f)
    radiative = 1.0 / tau
    survival = P * radiative / (radiative + p * (1.0 - f) * k_ar + p * f * k_mol)
    return survival * population / ENERGY_XRAY_KEV[mixture]


def theory_yield_total(params, degrad_data, concentration, pressure_bar: float, *, mixture: Mixture):
    out = None
    for line in IR_LINES:
        value = np.asarray(
            theory_yield_joint(
                params,
                degrad_data,
                concentration,
                pressure_bar,
                mixture=mixture,
                line=line,
            ),
            dtype=float,
        )
        out = value if out is None else out + value
    return out


def _wrapper(mixture: Mixture, line: str):
    def evaluate(params, degrad_data, concentration, pressure_bar):
        return theory_yield_joint(
            params,
            degrad_data,
            concentration,
            pressure_bar,
            mixture=mixture,
            line=line,
        )

    evaluate.__name__ = f"theory_yield_ArJoint_{mixture}_{line}"
    return evaluate


for _mixture in ("ArCF4", "ArN2"):
    for _line in IR_LINES:
        globals()[f"theory_yield_ArJoint_{_mixture}_{_line}"] = _wrapper(_mixture, _line)
