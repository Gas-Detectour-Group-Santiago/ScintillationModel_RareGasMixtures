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
from .plotting import (
    plot_generated_mosaic,
    plot_generated_mosaic_brokenx,
    plot_generated_mosaic_with_inset,
)


class GeneratedBuilder:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        add_models_to_path(self.project_root)
        self._ar2nd_pure_reference_cache: tuple[pd.DataFrame, str, float] | None = None

    def build_gas(self, gas: str, *, amplied: bool = False) -> pd.DataFrame:
        if gas == "ArCF4":
            return self._build_arcf4(amplied=amplied)
        if gas == "ArN2":
            return self._build_arn2(amplied=amplied)
        raise ValueError(f"Gas no soportado: {gas}")

    def _load_ar2nd_degrad(self, gas: str) -> pd.DataFrame:
        """Load the dedicated Ar2nd precursor CSV for one mixture family."""
        path = getattr(cfg, "AR2ND_DEGRAD_CSVS", {}).get(gas)
        if path is None:
            path = cfg.GASES[gas].degrad_csv
        return pd.read_csv(require_file(self.project_root / path))

    def _ar2nd_pure_reference(self) -> tuple[pd.DataFrame, str, float]:
        """Common pure-Ar input for the Ar second continuum.

        The 0% additive panel is pure Ar independently of the scan label.  The
        dedicated Ar2nd tables use the same precursor definition
        Ar(1s4,1s5) + Ar(1s2,1s3) + Ar**, so any remaining difference is explicit:
        X-ray energy and the selected singlet/triplet partition.
        """
        if self._ar2nd_pure_reference_cache is not None:
            return self._ar2nd_pure_reference_cache

        reference_gas = str(getattr(cfg, "AR2ND_PURE_REFERENCE_GAS", "ArN2"))
        degrad = self._load_ar2nd_degrad(reference_gas)
        if reference_gas == "ArN2":
            from ArN2 import energy_X_ray_N2

            out = (degrad, reference_gas, float(energy_X_ray_N2))
        elif reference_gas == "ArCF4":
            from ArCF4 import energy_X_ray_CF4

            out = (degrad, reference_gas, float(energy_X_ray_CF4))
        else:
            raise ValueError(f"Gas de referencia Ar2nd no soportado: {reference_gas}")
        self._ar2nd_pure_reference_cache = out
        return out

    def _ar2nd_inputs_for_condition(
        self,
        gas: str,
        concentration_percent: float,
        degrad: pd.DataFrame,
        energy_xray_kev: float,
    ) -> tuple[pd.DataFrame, str, float]:
        """Return Ar2nd inputs, forcing one common pure-Ar reference at 0%."""
        pure_limit = float(getattr(cfg, "AR2ND_PURE_REFERENCE_CONCENTRATION_PERCENT", 0.0))
        if (
            bool(getattr(cfg, "AR2ND_FORCE_COMMON_PURE_REFERENCE", False))
            and float(concentration_percent) <= pure_limit + 1.0e-12
        ):
            return self._ar2nd_pure_reference()
        return degrad, gas, float(energy_xray_kev)


    @staticmethod
    def _apply_primary_normalisation(components: dict[str, np.ndarray], norm: float) -> dict[str, np.ndarray]:
        """Apply primary Nnorm only to primary optical components.

        The Ar second continuum is already an absolute per-energy prediction,
        so it must not be divided by the fitted primary normalisation.
        """
        return {
            key: (value if key == "ar2_2nd_continium" else value / norm)
            for key, value in components.items()
        }

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

    def _build_arcf4(self, *, amplied: bool = False) -> pd.DataFrame:
        from ArCF4 import energy_X_ray_CF4, theory_yield_uv, theory_yield_vis
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
        wavelength_range = (
            cfg.WAVELENGTH_RANGE_GENERATED_AMPLIED["ArCF4"]
            if amplied
            else cfg.WAVELENGTH_RANGE_GENERATED["ArCF4"]
        )
        wavelength_points = cfg.WAVELENGTH_POINTS_AMPLIED if amplied else cfg.WAVELENGTH_POINTS
        wavelength = np.linspace(*wavelength_range, wavelength_points)

        ar2_params = None
        ar2_degrad = None
        if amplied:
            from Ar2nd_continium import read_ar2nd_parameters

            ar2_params = read_ar2nd_parameters(self.project_root / cfg.AR2ND_CONTINIUM_PARAMETER_CSV)
            ar2_params["triplet_weight"] = 1.0
            ar2_degrad = self._load_ar2nd_degrad("ArCF4")

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
                y_cf4 = np.asarray(y_cf4, dtype=float)[0]
                components = {
                    "vis": y_vis * gaussian_pdf(wavelength, 630.0, 40.0),
                    "cf4_uv": y_cf4 * 1.0e3 * weighted_gaussian_sum(wavelength, 1.0, CF4_UV_PEAKS),
                    "ar3rd_uv": np.asarray(y_ar3rd, dtype=float)[0] * 1.0e3 * weighted_gaussian_sum(wavelength, 1.0, AR_3RD_UV_PEAKS) / 0.49,
                    "cf3_uv": np.asarray(y_cf3, dtype=float)[0] * 1.0e3 * gaussian_pdf(wavelength, 260.0, 60.0),
                }
                if amplied and ar2_params is not None:
                    if bool(getattr(cfg, "AMPLIED_INCLUDE_AR2ND_CONTINIUM", False)):
                        from Ar2nd_continium import theory_yield_ar2nd_continium

                        ar2_input, ar2_gas_mixture, ar2_energy_xray = self._ar2nd_inputs_for_condition(
                            "ArCF4", concentration, ar2_degrad, float(energy_X_ray_CF4)
                        )

                        y_ar2nd = np.asarray(
                            theory_yield_ar2nd_continium(
                                ar2_params,
                                ar2_input,
                                f,
                                pressure,
                                gas_mixture=ar2_gas_mixture,
                                energy_xray_ev=ar2_energy_xray,
                            ),
                            dtype=float,
                        )[0]
                        components["ar2_2nd_continium"] = (
                            y_ar2nd
                            * 1.0e3
                            * gaussian_pdf(
                                wavelength,
                                ar2_params["lambda_Ar2nd_nm"],
                                ar2_params["sigma_Ar2nd_nm"],
                            )
                        )
                    components["cf4_vuv_155"] = (
                        ar2_params["Br_CF4_D_to_X"]
                        * y_cf4
                        * 1.0e3
                        * gaussian_pdf(
                            wavelength,
                            ar2_params["lambda_CF4_D_to_X_nm"],
                            ar2_params["sigma_CF4_D_to_X_nm"],
                        )
                    )
                for component, (line_nm, func) in ir_lines.items():
                    y_ir = np.asarray(func(params_ir, degrad_ir, f, pressure), dtype=float)[0] * 1.0e3
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.5)
                components = self._apply_primary_normalisation(components, norm)
                rows.append(self._component_rows("ArCF4", pressure, concentration, wavelength, components))
        return pd.concat(rows, ignore_index=True)

    def _build_arn2(self, *, amplied: bool = False) -> pd.DataFrame:
        from ArN2 import energy_X_ray_N2, theory_yield_N2_uv
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
        wavelength_range = (
            cfg.WAVELENGTH_RANGE_GENERATED_AMPLIED["ArN2"]
            if amplied
            else cfg.WAVELENGTH_RANGE_GENERATED["ArN2"]
        )
        wavelength_points = cfg.WAVELENGTH_POINTS_AMPLIED if amplied else cfg.WAVELENGTH_POINTS
        wavelength = np.linspace(*wavelength_range, wavelength_points)

        ar2_params = None
        ar2_degrad = None
        if amplied:
            from Ar2nd_continium import read_ar2nd_parameters

            ar2_params = read_ar2nd_parameters(self.project_root / cfg.AR2ND_CONTINIUM_PARAMETER_CSV)
            ar2_params["triplet_weight"] = 1.0
            ar2_degrad = self._load_ar2nd_degrad("ArN2")

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
                if amplied and ar2_params is not None and bool(getattr(cfg, "AMPLIED_INCLUDE_AR2ND_CONTINIUM", False)):
                    from Ar2nd_continium import theory_yield_ar2nd_continium

                    ar2_input, ar2_gas_mixture, ar2_energy_xray = self._ar2nd_inputs_for_condition(
                        "ArN2", concentration, ar2_degrad, float(energy_X_ray_N2)
                    )

                    y_ar2nd = np.asarray(
                        theory_yield_ar2nd_continium(
                            ar2_params,
                            ar2_input,
                            f,
                            pressure,
                            gas_mixture=ar2_gas_mixture,
                            energy_xray_ev=ar2_energy_xray,
                        ),
                        dtype=float,
                    )[0]
                    components["ar2_2nd_continium"] = (
                        y_ar2nd
                        * 1.0e3
                        * gaussian_pdf(
                            wavelength,
                            ar2_params["lambda_Ar2nd_nm"],
                            ar2_params["sigma_Ar2nd_nm"],
                        )
                    )
                for component, (line_nm, func) in ir_lines.items():
                    y_ir = np.asarray(func(params_ir, degrad_ir, f, pressure), dtype=float)[0] * 1.0e3
                    components[component] = y_ir * gaussian_pdf(wavelength, line_nm, 2.8)
                components = self._apply_primary_normalisation(components, norm)
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


def build_generated_amplied_spectra(project_root: Path, outdir: Path) -> dict[str, pd.DataFrame]:
    builder = GeneratedBuilder(project_root)
    generated = {}
    for gas in cfg.GASES:
        print(f"[spectra] generated amplied {gas}")
        df = builder.build_gas(gas, amplied=True)
        path = outdir / "csv" / f"{gas}_spectra_generated_amplied.csv"
        ensure_parent(path)
        df.to_csv(path, index=False)
        print(f"[spectra] CSV: {path}")
        generated[gas] = df
    return generated


def run_generated_mosaics(project_root: Path, outdir: Path, generated_by_gas: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    generated_by_gas = generated_by_gas or build_generated_spectra(project_root, outdir)
    for gas, df in generated_by_gas.items():
        plot_generated_mosaic(outdir, gas, df)

    amplied_by_gas = build_generated_amplied_spectra(project_root, outdir)
    for gas, df in amplied_by_gas.items():
        plot_generated_mosaic(
            outdir,
            gas,
            df,
            wavelength_range=cfg.WAVELENGTH_RANGE_GENERATED_AMPLIED[gas],
            output_stem=f"{gas}_spectra_generated_amplied",
            title=rf"{gas} generated spectra, extended VUV",
            log_y=bool(cfg.GENERATED_AMPLIED_LOG_SCALE),
            log_ymin=float(cfg.GENERATED_AMPLIED_LOG_YMIN),
            log_ymax_factor=float(cfg.GENERATED_AMPLIED_LOG_YMAX_FACTOR),
        )
        if bool(getattr(cfg, "GENERATED_AMPLIED_INSET_ENABLED", False)):
            plot_generated_mosaic_with_inset(
                outdir,
                gas,
                df,
                main_window=tuple(cfg.GENERATED_AMPLIED_MAIN_WINDOW[gas]),
                main_ylim_window=tuple(cfg.GENERATED_AMPLIED_MAIN_YLIM_WINDOW[gas]),
                inset_window=tuple(cfg.GENERATED_AMPLIED_VUV_WINDOW_NM),
                output_stem=f"{gas}_spectra_generated_amplied_inset",
                title=rf"{gas} generated spectra, extended VUV (main + inset)",
            )
        if bool(getattr(cfg, "GENERATED_AMPLIED_BROKENX_ENABLED", False)):
            plot_generated_mosaic_brokenx(
                outdir,
                gas,
                df,
                output_stem=f"{gas}_spectra_generated_amplied_brokenx",
                title=rf"{gas} generated spectra, extended VUV (broken x axis)",
                left_window=tuple(cfg.GENERATED_AMPLIED_VUV_WINDOW_NM),
                right_window=tuple(cfg.GENERATED_AMPLIED_MAIN_WINDOW[gas]),
            )
    return generated_by_gas
