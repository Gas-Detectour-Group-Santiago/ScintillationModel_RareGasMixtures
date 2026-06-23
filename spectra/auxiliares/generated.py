from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import (
    AR_3RD_UV_PEAKS,
    CF4_UV_PEAKS,
    N2_SECOND_POSITIVE_PEAKS,
    add_models_to_path,
    ensure_parent,
    gaussian_pdf,
    read_parameter_vector,
    require_file,
    weighted_gaussian_sum,
)
from .plotting import plot_generated_mosaic


class GeneratedBuilder:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        add_models_to_path(self.project_root)

    def build_gas(self, gas: str) -> pd.DataFrame:
        if gas == "ArCF4":
            return self._build_arcf4()
        if gas == "ArN2":
            return self._build_arn2()
        raise ValueError(f"Gas no soportado: {gas}")

    @staticmethod
    def _component_rows(
        gas: str,
        pressure_bar: float,
        concentration_percent: float,
        wavelength: np.ndarray,
        components: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        rows = []
        total = np.zeros_like(wavelength, dtype=float)
        for component, intensity in components.items():
            intensity = np.asarray(intensity, dtype=float)
            total += intensity
            rows.append(
                pd.DataFrame(
                    {
                        "gas_mixture": gas,
                        "pressure_bar": pressure_bar,
                        "concentration_percent": concentration_percent,
                        "concentration_fraction": concentration_percent / 100.0,
                        "component": component,
                        "wavelength_nm": wavelength,
                        "intensity_ph_MeV_nm": intensity,
                    }
                )
            )
        rows.append(
            pd.DataFrame(
                {
                    "gas_mixture": gas,
                    "pressure_bar": pressure_bar,
                    "concentration_percent": concentration_percent,
                    "concentration_fraction": concentration_percent / 100.0,
                    "component": "total",
                    "wavelength_nm": wavelength,
                    "intensity_ph_MeV_nm": total,
                }
            )
        )
        return pd.concat(rows, ignore_index=True)

    def _build_arcf4(self) -> pd.DataFrame:
        from ArCF4 import theory_yield_uv, theory_yield_vis
        from ArCF4_infrarred import (
            theory_yield_ArCF4_Ir_696,
            theory_yield_ArCF4_Ir_727,
            theory_yield_ArCF4_Ir_750,
            theory_yield_ArCF4_Ir_763,
            theory_yield_ArCF4_Ir_772,
        )

        files = cfg.GASES["ArCF4"]
        degrad = pd.read_csv(require_file(self.project_root / files.degrad_csv))
        degrad_ir = pd.read_csv(require_file(self.project_root / files.degrad_ir_csv))
        params = read_parameter_vector(self.project_root / files.parameter_csv)
        params_ir = read_parameter_vector(self.project_root / files.ir_parameter_csv)
        norm = float(read_parameter_vector(self.project_root / files.norm_parameter_csv)[0])
        wavelength = np.linspace(*cfg.WAVELENGTH_RANGE_GENERATED["ArCF4"], cfg.WAVELENGTH_POINTS)

        ir_lines = {
            "ir_696": (696.0, theory_yield_ArCF4_Ir_696),
            "ir_727": (727.0, theory_yield_ArCF4_Ir_727),
            "ir_750": (750.0, theory_yield_ArCF4_Ir_750),
            "ir_763": (763.0, theory_yield_ArCF4_Ir_763),
            "ir_772": (772.0, theory_yield_ArCF4_Ir_772),
        }
        rows = []
        for pressure in cfg.GENERATED_PRESSURES_BAR:
            for concentration in cfg.GENERATED_CONCENTRATIONS_PERCENT:
                f = np.asarray([float(concentration) / 100.0])
                y_vis = np.asarray(theory_yield_vis(params, degrad, f, pressure), dtype=float)[0] * 1.0e3
                _, y_cf4, y_ar3rd, y_cf3 = theory_yield_uv(params, degrad, f, pressure, activate_components=True)
                components = {
                    "vis": y_vis * gaussian_pdf(wavelength, 630.0, 40.0),
                    "cf4_uv": np.asarray(y_cf4, dtype=float)[0] * 1.0e3 * weighted_gaussian_sum(wavelength, 1.0, CF4_UV_PEAKS),
                    "ar3rd_uv": np.asarray(y_ar3rd, dtype=float)[0] * 1.0e3 * weighted_gaussian_sum(wavelength, 1.0, AR_3RD_UV_PEAKS) / 0.49,
                    "cf3_uv": np.asarray(y_cf3, dtype=float)[0] * 1.0e3 * gaussian_pdf(wavelength, 260.0, 60.0),
                }
                for component, (line_nm, func) in ir_lines.items():
                    y_ir = np.asarray(func(params_ir, degrad_ir, f, pressure), dtype=float)[0] * 1.0e3
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.5)
                components = {k: v / norm for k, v in components.items()}
                rows.append(self._component_rows("ArCF4", pressure, concentration, wavelength, components))
        return pd.concat(rows, ignore_index=True)

    def _build_arn2(self) -> pd.DataFrame:
        from ArN2 import theory_yield_N2_uv
        from ArN2_infrarred import (
            theory_yield_ArN2_Ir_696,
            theory_yield_ArN2_Ir_727,
            theory_yield_ArN2_Ir_750,
            theory_yield_ArN2_Ir_763,
            theory_yield_ArN2_Ir_772,
        )

        files = cfg.GASES["ArN2"]
        degrad = pd.read_csv(require_file(self.project_root / files.degrad_csv))
        degrad_ir = pd.read_csv(require_file(self.project_root / files.degrad_ir_csv))
        params = read_parameter_vector(self.project_root / files.parameter_csv)
        params_ir = read_parameter_vector(self.project_root / files.ir_parameter_csv)
        norm = float(read_parameter_vector(self.project_root / files.norm_parameter_csv)[0])
        wavelength = np.linspace(*cfg.WAVELENGTH_RANGE_GENERATED["ArN2"], cfg.WAVELENGTH_POINTS)

        ir_lines = {
            "ir_696": (696.0, theory_yield_ArN2_Ir_696),
            "ir_727": (727.0, theory_yield_ArN2_Ir_727),
            "ir_750": (750.0, theory_yield_ArN2_Ir_750),
            "ir_763": (763.0, theory_yield_ArN2_Ir_763),
            "ir_772": (772.0, theory_yield_ArN2_Ir_772),
        }
        rows = []
        for pressure in cfg.GENERATED_PRESSURES_BAR:
            for concentration in cfg.GENERATED_CONCENTRATIONS_PERCENT:
                f = np.asarray([float(concentration) / 100.0])
                y_n2_uv = np.asarray(theory_yield_N2_uv(params, degrad, f, pressure), dtype=float)[0] * 1.0e3
                components = {"n2_uv": weighted_gaussian_sum(wavelength, y_n2_uv, N2_SECOND_POSITIVE_PEAKS)}
                for component, (line_nm, func) in ir_lines.items():
                    y_ir = np.asarray(func(params_ir, degrad_ir, f, pressure), dtype=float)[0] * 1.0e3
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.8)
                components = {k: v / norm for k, v in components.items()}
                rows.append(self._component_rows("ArN2", pressure, concentration, wavelength, components))
        return pd.concat(rows, ignore_index=True)


def build_generated_spectra(project_root: Path, outdir: Path) -> dict[str, pd.DataFrame]:
    builder = GeneratedBuilder(project_root)
    generated = {}
    for gas in cfg.GASES:
        print(f"[spectra] generated {gas}")
        df = builder.build_gas(gas)
        path = outdir / "csv" / f"{gas}_generated_spectra.csv"
        ensure_parent(path)
        df.to_csv(path, index=False)
        print(f"[spectra] CSV: {path}")
        generated[gas] = df
    return generated


def run_generated_mosaics(project_root: Path, outdir: Path, generated_by_gas: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    generated_by_gas = generated_by_gas or build_generated_spectra(project_root, outdir)
    for gas, df in generated_by_gas.items():
        plot_generated_mosaic(outdir, gas, df)
    return generated_by_gas
