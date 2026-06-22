from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .spectra_types import GaussianPeak, GeneratedSpectraConfig
from .spectra_units import (
    gaussian_pdf,
    model_fit_unit_to_ph_per_MeV,
    read_parameter_vector,
    weighted_gaussian_sum,
)


CF4_UV_PEAKS = [
    GaussianPeak(235.0, 17.0, 0.55),
    GaussianPeak(290.0, 17.0, 0.75),
    GaussianPeak(364.0, 50.0, 0.35),
]

AR_3RD_UV_PEAKS = [
    GaussianPeak(176.0, 30.0, 1.0),
    GaussianPeak(188.0, 30.0, 1.0),
    GaussianPeak(199.0, 30.0, 1.0),
    GaussianPeak(212.0, 30.0, 1.0),
    GaussianPeak(225.0, 30.0, 1.0),
    GaussianPeak(245.0, 30.0, 1.0),
]

N2_SECOND_POSITIVE_PEAKS = [
    GaussianPeak(335.0, 3.75, 0.42),
    GaussianPeak(355.0, 3.75, 0.30),
    GaussianPeak(378.0, 3.75, 0.10),
    GaussianPeak(403.0, 3.75, 0.05),
]


def _prepend_once(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _component_rows(
    gas_mixture: str,
    pressure_bar: float,
    concentration_percent: float,
    wavelength_nm: np.ndarray,
    components: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    total = np.zeros_like(wavelength_nm, dtype=float)

    for component, intensity in components.items():
        intensity = np.asarray(intensity, dtype=float)
        total += intensity
        rows.append(
            pd.DataFrame(
                {
                    "gas_mixture": gas_mixture,
                    "pressure_bar": pressure_bar,
                    "concentration_percent": concentration_percent,
                    "concentration_fraction": concentration_percent / 100.0,
                    "component": component,
                    "wavelength_nm": wavelength_nm,
                    "intensity_ph_MeV_nm": intensity,
                }
            )
        )

    rows.append(
        pd.DataFrame(
            {
                "gas_mixture": gas_mixture,
                "pressure_bar": pressure_bar,
                "concentration_percent": concentration_percent,
                "concentration_fraction": concentration_percent / 100.0,
                "component": "total",
                "wavelength_nm": wavelength_nm,
                "intensity_ph_MeV_nm": total,
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


class GeneratedSpectraBuilder:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        _prepend_once(self.project_root / "models")

    def build(self, config: GeneratedSpectraConfig) -> pd.DataFrame:
        if config.gas_mixture == "ArCF4":
            return self._build_arcf4(config)
        if config.gas_mixture == "ArN2":
            return self._build_arn2(config)
        raise ValueError(f"Mezcla no soportada: {config.gas_mixture}")

    def _build_arcf4(self, config: GeneratedSpectraConfig) -> pd.DataFrame:
        from ArCF4 import theory_yield_uv, theory_yield_vis
        from ArCF4_infrarred import (
            theory_yield_ArCF4_Ir_696,
            theory_yield_ArCF4_Ir_727,
            theory_yield_ArCF4_Ir_750,
            theory_yield_ArCF4_Ir_763,
            theory_yield_ArCF4_Ir_772,
        )

        degrad = pd.read_csv(config.degrad_csv)
        degrad_ir = pd.read_csv(config.degrad_ir_csv)
        params = read_parameter_vector(config.parameter_csv)
        params_ir = read_parameter_vector(config.ir_parameter_csv)
        norm = read_parameter_vector(config.norm_parameter_csv)[0]
        wavelength = np.linspace(config.wavelength_min_nm, config.wavelength_max_nm, config.wavelength_points)

        ir_lines = {
            "ir_696": (696.0, theory_yield_ArCF4_Ir_696),
            "ir_727": (727.0, theory_yield_ArCF4_Ir_727),
            "ir_750": (750.0, theory_yield_ArCF4_Ir_750),
            "ir_763": (763.0, theory_yield_ArCF4_Ir_763),
            "ir_772": (772.0, theory_yield_ArCF4_Ir_772),
        }

        out = []
        for pressure in config.pressures_bar:
            for concentration in config.concentrations_percent:
                f_arr = np.asarray([concentration / 100.0], dtype=float)

                y_vis = model_fit_unit_to_ph_per_MeV(
                    theory_yield_vis(params, degrad, f_arr, pressure)
                )[0]

                _, y_cf4, y_ar_dblestar, y_cf3_uv = theory_yield_uv(
                    params,
                    degrad,
                    f_arr,
                    pressure,
                    activate_components=True,
                )

                components = {
                    "vis": y_vis * gaussian_pdf(wavelength, 630.0, 40.0),
                    "cf4_uv": model_fit_unit_to_ph_per_MeV(y_cf4)[0]
                    * weighted_gaussian_sum(wavelength, 1.0, CF4_UV_PEAKS),
                    "ar3rd_uv": model_fit_unit_to_ph_per_MeV(y_ar_dblestar)[0]
                    * weighted_gaussian_sum(wavelength, 1.0, AR_3RD_UV_PEAKS)
                    / 0.49,
                    "cf3_uv": model_fit_unit_to_ph_per_MeV(y_cf3_uv)[0] * gaussian_pdf(wavelength, 260.0, 60.0),
                }

                for component, (line_nm, func) in ir_lines.items():
                    y_ir = model_fit_unit_to_ph_per_MeV(func(params_ir, degrad_ir, f_arr, pressure))[0]
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.5)

                components = {k: v / norm for k, v in components.items()}
                out.append(_component_rows("ArCF4", pressure, concentration, wavelength, components))

        return pd.concat(out, ignore_index=True)

    def _build_arn2(self, config: GeneratedSpectraConfig) -> pd.DataFrame:
        from ArN2 import theory_yield_N2_uv
        from ArN2_infrarred import (
            theory_yield_ArN2_Ir_696,
            theory_yield_ArN2_Ir_727,
            theory_yield_ArN2_Ir_750,
            theory_yield_ArN2_Ir_763,
            theory_yield_ArN2_Ir_772,
        )

        degrad = pd.read_csv(config.degrad_csv)
        degrad_ir = pd.read_csv(config.degrad_ir_csv)
        params = read_parameter_vector(config.parameter_csv)
        params_ir = read_parameter_vector(config.ir_parameter_csv)
        norm = read_parameter_vector(config.norm_parameter_csv)[0]
        wavelength = np.linspace(config.wavelength_min_nm, config.wavelength_max_nm, config.wavelength_points)

        ir_lines = {
            "ir_696": (696.0, theory_yield_ArN2_Ir_696),
            "ir_727": (727.0, theory_yield_ArN2_Ir_727),
            "ir_750": (750.0, theory_yield_ArN2_Ir_750),
            "ir_763": (763.0, theory_yield_ArN2_Ir_763),
            "ir_772": (772.0, theory_yield_ArN2_Ir_772),
        }

        out = []
        for pressure in config.pressures_bar:
            for concentration in config.concentrations_percent:
                f_arr = np.asarray([concentration / 100.0], dtype=float)
                y_n2_uv = model_fit_unit_to_ph_per_MeV(
                    theory_yield_N2_uv(params, degrad, f_arr, pressure)
                )[0]

                components = {
                    "n2_uv": weighted_gaussian_sum(wavelength, y_n2_uv, N2_SECOND_POSITIVE_PEAKS),
                }

                for component, (line_nm, func) in ir_lines.items():
                    y_ir = model_fit_unit_to_ph_per_MeV(func(params_ir, degrad_ir, f_arr, pressure))[0]
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.8)

                components = {k: v / norm for k, v in components.items()}
                out.append(_component_rows("ArN2", pressure, concentration, wavelength, components))

        return pd.concat(out, ignore_index=True)

