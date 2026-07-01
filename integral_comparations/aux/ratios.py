from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .integrators import IntegralConfig, IntegralResult, integrate_spectrum
from .spectra_io import SCAN, SpectrumProvider, SpectrumSelector


@dataclass(frozen=True)
class IntegralDefinition:
    name: str
    selector: SpectrumSelector
    integral: IntegralConfig


@dataclass(frozen=True)
class RatioDefinition:
    name: str
    numerator: IntegralDefinition
    denominator: IntegralDefinition
    description: str = ""


@dataclass(frozen=True)
class ScanConfig:
    """Generic pressure/concentration scan.

    ``None`` for pressures/concentrations means: infer all available values from
    the numerator selector.
    """

    pressures_bar: tuple[float, ...] | None = None
    concentrations_percent: tuple[float, ...] | None = None
    concentration_range_percent: tuple[float, float] | None = None
    pressure_range_bar: tuple[float, float] | None = None


class RatioScanner:
    """Compute one or several integral ratios over concentration/pressure grids."""

    def __init__(self, provider: SpectrumProvider) -> None:
        self.provider = provider

    def compute(
        self,
        ratios: RatioDefinition | Sequence[RatioDefinition],
        scan: ScanConfig | None = None,
    ) -> pd.DataFrame:
        if isinstance(ratios, RatioDefinition):
            ratio_list = [ratios]
        else:
            ratio_list = list(ratios)
        scan = scan or ScanConfig()

        rows: list[dict[str, object]] = []
        for ratio in ratio_list:
            rows.extend(self._compute_one(ratio, scan))

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["ratio_name", "pressure_bar", "concentration_percent"]).reset_index(drop=True)

    def _compute_one(self, ratio: RatioDefinition, scan: ScanConfig) -> list[dict[str, object]]:
        grid = self._scan_grid(ratio, scan)
        rows: list[dict[str, object]] = []

        denominator_cache: dict[tuple[float, float], tuple[float, IntegralResult, SpectrumSelector]] = {}

        for pressure_bar, concentration_percent in grid:
            num_selector = ratio.numerator.selector.resolved(
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )
            den_selector = ratio.denominator.selector.resolved(
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )

            try:
                numerator_value, numerator_result = self._compute_integral(ratio.numerator, num_selector)
            except Exception as exc:
                rows.append(self._error_row(ratio, pressure_bar, concentration_percent, "numerator", exc))
                continue

            den_key = (float(den_selector.pressure_bar), float(den_selector.concentration_percent))
            if den_key not in denominator_cache:
                try:
                    denominator_value, denominator_result = self._compute_integral(ratio.denominator, den_selector)
                    denominator_cache[den_key] = (denominator_value, denominator_result, den_selector)
                except Exception as exc:
                    rows.append(self._error_row(ratio, pressure_bar, concentration_percent, "denominator", exc))
                    continue
            else:
                denominator_value, denominator_result, _ = denominator_cache[den_key]

            ratio_value, status, error_stage, error_message = _safe_ratio(
                numerator_value,
                denominator_value,
            )

            rows.append(
                {
                    "ratio_name": ratio.name,
                    "description": ratio.description,
                    "pressure_bar": pressure_bar,
                    "concentration_percent": concentration_percent,
                    "ratio": ratio_value,
                    "numerator_name": ratio.numerator.name,
                    "numerator_gas": num_selector.gas,
                    "numerator_pressure_bar": num_selector.pressure_bar,
                    "numerator_concentration_percent": num_selector.concentration_percent,
                    "numerator_spectrum_column": num_selector.spectrum_column,
                    "numerator_source": num_selector.source,
                    "numerator_unit": num_selector.unit,
                    "numerator_range_nm": _range_label(ratio.numerator.integral.wavelength_range_nm),
                    "numerator_method": numerator_result.method,
                    "numerator_integral": numerator_value,
                    "numerator_n_points": numerator_result.n_points,
                    "denominator_name": ratio.denominator.name,
                    "denominator_gas": den_selector.gas,
                    "denominator_pressure_bar": den_selector.pressure_bar,
                    "denominator_concentration_percent": den_selector.concentration_percent,
                    "denominator_spectrum_column": den_selector.spectrum_column,
                    "denominator_source": den_selector.source,
                    "denominator_unit": den_selector.unit,
                    "denominator_range_nm": _range_label(ratio.denominator.integral.wavelength_range_nm),
                    "denominator_method": denominator_result.method,
                    "denominator_integral": denominator_value,
                    "denominator_n_points": denominator_result.n_points,
                    "status": status,
                    "error_stage": error_stage,
                    "error_message": error_message,
                }
            )
        return rows

    def _compute_integral(
        self,
        definition: IntegralDefinition,
        selector: SpectrumSelector,
    ) -> tuple[float, IntegralResult]:
        spectrum = self.provider.load(selector)
        result = integrate_spectrum(spectrum.wavelength_nm, spectrum.intensity, definition.integral)
        return result.value, result

    def _scan_grid(self, ratio: RatioDefinition, scan: ScanConfig) -> list[tuple[float, float]]:
        # Available values are inferred from the numerator selector. This is the
        # most common use: numerator scans c/P, denominator is fixed or matched.
        num = ratio.numerator.selector
        available = self.provider.available_conditions(
            num.gas,
            source=num.source,
            spectrum_column=num.spectrum_column,
        )
        if available.empty:
            return []

        if scan.pressures_bar is None:
            pressures = np.sort(available["pressure_bar"].astype(float).unique())
        else:
            pressures = np.asarray(scan.pressures_bar, dtype=float)

        if scan.pressure_range_bar is not None:
            pmin, pmax = scan.pressure_range_bar
            pressures = pressures[(pressures >= pmin) & (pressures <= pmax)]

        grid: list[tuple[float, float]] = []
        for pressure in pressures:
            available_p = available[np.isclose(available["pressure_bar"].astype(float), pressure)]
            if scan.concentrations_percent is None:
                concentrations = np.sort(available_p["concentration_percent"].astype(float).unique())
            else:
                concentrations = np.asarray(scan.concentrations_percent, dtype=float)

            if scan.concentration_range_percent is not None:
                cmin, cmax = scan.concentration_range_percent
                concentrations = concentrations[(concentrations >= cmin) & (concentrations <= cmax)]

            for concentration in concentrations:
                grid.append((float(pressure), float(concentration)))
        return grid

    def _error_row(
        self,
        ratio: RatioDefinition,
        pressure_bar: float,
        concentration_percent: float,
        stage: str,
        exc: Exception,
    ) -> dict[str, object]:
        return {
            "ratio_name": ratio.name,
            "description": ratio.description,
            "pressure_bar": pressure_bar,
            "concentration_percent": concentration_percent,
            "ratio": np.nan,
            "numerator_name": ratio.numerator.name,
            "denominator_name": ratio.denominator.name,
            "status": "error",
            "error_stage": stage,
            "error_message": f"{type(exc).__name__}: {exc}",
        }


def _range_label(bounds: tuple[float, float]) -> str:
    return f"{bounds[0]:g}-{bounds[1]:g}"


def _safe_ratio(numerator_value: float, denominator_value: float) -> tuple[float, str, str, str]:
    if not np.isfinite(numerator_value):
        return np.nan, "invalid_ratio", "ratio", "non-finite numerator integral"
    if not np.isfinite(denominator_value):
        return np.nan, "invalid_ratio", "ratio", "non-finite denominator integral"
    if denominator_value == 0.0:
        return np.nan, "invalid_ratio", "ratio", "zero denominator integral"
    ratio_value = numerator_value / denominator_value
    if not np.isfinite(ratio_value):
        return np.nan, "invalid_ratio", "ratio", "non-finite ratio"
    return float(ratio_value), "ok", "", ""
