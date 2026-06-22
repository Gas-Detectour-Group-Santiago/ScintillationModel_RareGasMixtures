from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .spectra_io import ensure_parent, read_raw_spectra_csv
from .spectra_models import GeneratedSpectraBuilder
from .spectra_plotter import (
    plot_comparison,
    plot_comparison_mosaic,
    plot_generated_spectra,
    plot_raw_spectra,
)
from .spectra_types import (
    ComparisonConfig,
    ComparisonCurveConfig,
    ComparisonMosaicConfig,
    GeneratedSpectraConfig,
    RawReferenceConfig,
    RawSpectraConfig,
)
from .spectra_units import match_float, raw_to_ph_per_MeV_nm, read_parameter_vector


class SpectraRunner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        models_path = str(self.project_root / "models")
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

    def run_raw(self, configs: list[RawSpectraConfig] | tuple[RawSpectraConfig, ...], make_plots: bool = True) -> None:
        for config in configs:
            print(f"[spectra_raw] {config.name}")
            df = self._filtered_raw(config)
            ensure_parent(config.output_csv)
            df.to_csv(config.output_csv, index=False)
            print(f"[spectra_raw] CSV: {config.output_csv}")
            if make_plots:
                reference_df = self._filtered_reference(config.reference) if config.reference is not None else None
                plot_raw_spectra(df, config, reference_df=reference_df)
                print(f"[spectra_raw] PDF: {config.output_pdf}")

    def run_generated(
        self,
        configs: list[GeneratedSpectraConfig] | tuple[GeneratedSpectraConfig, ...],
        make_plots: bool = True,
    ) -> None:
        builder = GeneratedSpectraBuilder(self.project_root)
        for config in configs:
            print(f"[spectra_generated] {config.name}")
            df = builder.build(config)
            ensure_parent(config.output_csv)
            df.to_csv(config.output_csv, index=False)
            print(f"[spectra_generated] CSV: {config.output_csv}")
            if make_plots:
                plot_generated_spectra(
                    df,
                    config.output_pdf,
                    config.output_summary_pdf,
                    config.title,
                    config.wavelength_range_nm,
                )
                print(f"[spectra_generated] PDF: {config.output_pdf.parent}")

    def run_comparison(
        self,
        configs: list[ComparisonConfig | ComparisonMosaicConfig] | tuple[ComparisonConfig | ComparisonMosaicConfig, ...],
        make_plots: bool = True,
    ) -> None:
        for config in configs:
            print(f"[spectra_comparation] {config.name}")
            if isinstance(config, ComparisonMosaicConfig):
                df = self._comparison_mosaic_dataframe(config)
            else:
                df = self._comparison_dataframe(config)
            ensure_parent(config.output_csv)
            df.to_csv(config.output_csv, index=False)
            print(f"[spectra_comparation] CSV: {config.output_csv}")
            if make_plots:
                if isinstance(config, ComparisonMosaicConfig):
                    plot_comparison_mosaic(df, config)
                else:
                    plot_comparison(
                        df,
                        config.output_pdf,
                        config.title or config.name,
                        config.wavelength_range_nm,
                    )
                print(f"[spectra_comparation] PDF: {config.output_pdf}")

    def _filtered_raw(self, config: RawSpectraConfig) -> pd.DataFrame:
        df = read_raw_spectra_csv(config.input_csv)
        df = df[df["gas_mixture"] == config.gas_mixture].copy()
        if config.spectrum_columns:
            df = df[df["spectrum_column"].isin(config.spectrum_columns)].copy()
        if config.concentrations_percent is not None:
            mask = np.zeros(len(df), dtype=bool)
            for concentration in config.concentrations_percent:
                mask |= match_float(df["concentration_percent"], concentration)
            filtered = df[mask].copy()
            self._warn_if_concentrations_look_shifted(config, df, filtered)
            df = filtered
        if config.pressures_bar is not None:
            mask = np.zeros(len(df), dtype=bool)
            for pressure in config.pressures_bar:
                mask |= match_float(df["pressure_bar"], pressure)
            df = df[mask].copy()
        return df.sort_values(
            ["gas_mixture", "concentration_percent", "pressure_bar", "spectrum_column", "wavelength_nm"]
        ).reset_index(drop=True)

    @staticmethod
    def _warn_if_concentrations_look_shifted(config: RawSpectraConfig, original: pd.DataFrame, filtered: pd.DataFrame) -> None:
        if config.concentrations_percent is None:
            return
        available = np.asarray(sorted(original["concentration_percent"].dropna().unique()), dtype=float)
        missing = [
            float(c)
            for c in config.concentrations_percent
            if not np.any(np.isclose(available, float(c), atol=1.0e-9, rtol=0.0))
        ]
        shifted = [c for c in missing if 0.0 < c <= 1.0 and np.any(np.isclose(available, c * 100.0, atol=1.0e-9, rtol=0.0))]
        if shifted:
            print(
                "[spectra_raw] AVISO: faltan concentraciones pequeñas "
                f"{shifted} en {config.input_csv.name}, pero existen multiplicadas por 100. "
                "Ese CSV probablemente se generó con una versión antigua de data/Analysis_spectra.py. "
                "Regenera con: python data/Analysis_spectra.py"
            )
        if filtered.empty:
            print(f"[spectra_raw] AVISO: el filtro de {config.name} dejó el CSV vacío.")

    def _filtered_reference(self, reference: RawReferenceConfig) -> pd.DataFrame:
        df = read_raw_spectra_csv(reference.raw_csv)
        df = df[df["gas_mixture"] == reference.gas_mixture].copy()
        df = df[df["spectrum_column"].isin(reference.spectrum_columns)].copy()
        df = df[
            match_float(df["concentration_percent"], reference.concentration_percent)
            & match_float(df["pressure_bar"], reference.pressure_bar)
        ].copy()
        if df.empty:
            raise RuntimeError(
                f"Referencia raw vacía: {reference.gas_mixture}, "
                f"{reference.concentration_percent:g}%, {reference.pressure_bar:g} bar"
            )
        first_spectrum = df["spectrum_name"].iloc[0]
        return df[df["spectrum_name"] == first_spectrum].sort_values("wavelength_nm").reset_index(drop=True)

    def _w_function(self, gas_mixture: str):
        if gas_mixture == "ArCF4":
            from ArCF4 import ion_potential

            return ion_potential
        if gas_mixture == "ArN2":
            from ArN2 import W_ArN2

            return W_ArN2
        raise ValueError(f"Mezcla no soportada: {gas_mixture}")

    def _comparison_dataframe(self, config: ComparisonConfig) -> pd.DataFrame:
        # Legacy single-gas comparison. Kept for compatibility; the configured
        # production plots use ComparisonMosaicConfig below.
        raw = read_raw_spectra_csv(config.raw_csv)
        generated = pd.read_csv(config.generated_csv)

        raw = raw[raw["gas_mixture"] == config.gas_mixture].copy()
        raw = raw[raw["spectrum_column"].isin(config.spectrum_columns)].copy()
        generated = generated[
            (generated["gas_mixture"] == config.gas_mixture) & (generated["component"] == "total")
        ].copy()

        w_func = self._w_function(config.gas_mixture)
        norm = read_parameter_vector(config.norm_parameter_csv)[0]

        rows: list[pd.DataFrame] = []
        for pressure in config.pressures_bar:
            for concentration in config.concentrations_percent:
                gen_mask = match_float(generated["pressure_bar"], pressure) & match_float(
                    generated["concentration_percent"], concentration
                )
                gen = generated[gen_mask].copy()
                if not gen.empty:
                    gen["kind"] = "model"
                    gen["curve_name"] = f"{config.gas_mixture}_model_{pressure:g}bar"
                    gen["label"] = rf"{config.gas_mixture} model, {pressure:g} bar"
                    gen["plot_intensity"] = gen["intensity_ph_MeV_nm"]
                    gen["plot_scale"] = 1.0
                    gen["normalisation_reference"] = str(config.norm_parameter_csv)
                    rows.append(
                        gen[
                            [
                                "gas_mixture",
                                "kind",
                                "curve_name",
                                "label",
                                "pressure_bar",
                                "concentration_percent",
                                "concentration_fraction",
                                "wavelength_nm",
                                "intensity_ph_MeV_nm",
                                "plot_intensity",
                                "plot_scale",
                                "normalisation_reference",
                            ]
                        ]
                    )

                raw_one = self._raw_one(config.gas_mixture, raw, concentration, pressure, config.spectrum_columns)
                if raw_one is None:
                    continue
                intensity_direct = raw_to_ph_per_MeV_nm(
                    raw_one["intensity_raw"].to_numpy(dtype=float),
                    concentration / 100.0,
                    w_func,
                    norm,
                )
                out = pd.DataFrame(
                    {
                        "gas_mixture": config.gas_mixture,
                        "kind": "experiment_raw_W_Nnorm",
                        "curve_name": f"{config.gas_mixture}_raw_{pressure:g}bar",
                        "label": rf"{config.gas_mixture} exp. raw/W/Nnorm, {pressure:g} bar",
                        "pressure_bar": pressure,
                        "concentration_percent": concentration,
                        "concentration_fraction": concentration / 100.0,
                        "wavelength_nm": raw_one["wavelength_nm"].to_numpy(dtype=float),
                        "intensity_raw_au": raw_one["intensity_raw"].to_numpy(dtype=float),
                        "intensity_raw_W_Nnorm": intensity_direct,
                        "intensity_ph_MeV_nm": intensity_direct,
                        "plot_intensity": intensity_direct / config.raw_plot_scale,
                        "plot_scale": config.raw_plot_scale,
                        "normalisation_reference": str(config.norm_parameter_csv),
                    }
                )
                rows.append(out)

        if not rows:
            raise RuntimeError(f"No se generó ninguna comparación para {config.name}")

        return pd.concat(rows, ignore_index=True).sort_values(
            ["gas_mixture", "concentration_percent", "pressure_bar", "kind", "wavelength_nm"]
        )

    def _comparison_mosaic_dataframe(self, config: ComparisonMosaicConfig) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        raw_cache: dict[Path, pd.DataFrame] = {}
        generated_cache: dict[Path, pd.DataFrame] = {}
        norm_cache: dict[Path, float] = {}

        for concentration in config.concentrations_percent:
            for curve_order, curve in enumerate(config.curves):
                if curve.kind == "model":
                    rows.append(self._model_curve_dataframe(config, curve, curve_order, concentration, generated_cache))
                elif curve.kind == "raw":
                    raw_df = self._raw_curve_dataframe(config, curve, curve_order, concentration, raw_cache, generated_cache, norm_cache)
                    if raw_df is not None:
                        rows.append(raw_df)
                else:
                    raise ValueError(f"kind no soportado: {curve.kind}")

        rows = [row for row in rows if row is not None and not row.empty]
        if not rows:
            raise RuntimeError(f"No se generó ninguna comparación para {config.name}")

        return pd.concat(rows, ignore_index=True).sort_values(
            ["comparison", "concentration_percent", "curve_order", "wavelength_nm"]
        ).reset_index(drop=True)

    def _model_curve_dataframe(
        self,
        config: ComparisonMosaicConfig,
        curve: ComparisonCurveConfig,
        curve_order: int,
        concentration: float,
        generated_cache: dict[Path, pd.DataFrame],
    ) -> pd.DataFrame:
        if curve.generated_csv is None:
            raise ValueError(f"La curva {curve.name} necesita generated_csv.")
        generated = generated_cache.setdefault(curve.generated_csv, pd.read_csv(curve.generated_csv))
        gen = self._generated_total(generated, curve.gas_mixture, curve.pressure_bar, concentration)
        if gen.empty:
            return pd.DataFrame()
        intensity = gen["intensity_ph_MeV_nm"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "comparison": config.name,
                "curve_name": curve.name,
                "curve_order": curve_order,
                "gas_mixture": curve.gas_mixture,
                "kind": "generated",
                "label": curve.label,
                "pressure_bar": curve.pressure_bar,
                "concentration_percent": concentration,
                "concentration_fraction": concentration / 100.0,
                "wavelength_nm": gen["wavelength_nm"].to_numpy(dtype=float),
                "intensity_ph_MeV_nm": intensity,
                "intensity_raw_au": np.nan,
                "intensity_raw_W_Nnorm": np.nan,
                "plot_intensity": intensity / curve.plot_scale,
                "plot_scale": curve.plot_scale,
                "color": curve.color,
                "linestyle": curve.linestyle,
                "linewidth": curve.linewidth,
                "alpha": curve.alpha,
                "normalisation_reference": "generated_model",
            }
        )

    def _raw_curve_dataframe(
        self,
        config: ComparisonMosaicConfig,
        curve: ComparisonCurveConfig,
        curve_order: int,
        concentration: float,
        raw_cache: dict[Path, pd.DataFrame],
        generated_cache: dict[Path, pd.DataFrame],
        norm_cache: dict[Path, float],
    ) -> pd.DataFrame | None:
        if curve.raw_csv is None or curve.norm_parameter_csv is None:
            raise ValueError(f"La curva {curve.name} necesita raw_csv y norm_parameter_csv.")
        raw = raw_cache.setdefault(curve.raw_csv, read_raw_spectra_csv(curve.raw_csv))
        raw = raw[(raw["gas_mixture"] == curve.gas_mixture) & raw["spectrum_column"].isin(curve.spectrum_columns)].copy()
        raw_one = self._raw_one(curve.gas_mixture, raw, concentration, curve.pressure_bar, curve.spectrum_columns)
        if raw_one is None:
            return None

        wavelength = raw_one["wavelength_nm"].to_numpy(dtype=float)
        raw_au = raw_one["intensity_raw"].to_numpy(dtype=float)
        raw_au = self._smooth(raw_au, curve.smooth_window)

        w_func = self._w_function(curve.gas_mixture)
        if curve.norm_parameter_csv not in norm_cache:
            norm_cache[curve.norm_parameter_csv] = float(read_parameter_vector(curve.norm_parameter_csv)[0])
        norm = norm_cache[curve.norm_parameter_csv]
        raw_direct = raw_to_ph_per_MeV_nm(raw_au, concentration / 100.0, w_func, norm)

        plot_intensity = raw_direct / curve.plot_scale
        normalisation_reference = f"direct_raw_times_1e6_over_W_over_Nnorm_div_{curve.plot_scale:g}"
        if curve.raw_normalisation == "area_to_generated":
            if curve.generated_csv is None:
                raise ValueError(f"La curva raw {curve.name} necesita generated_csv para area_to_generated.")
            generated = generated_cache.setdefault(curve.generated_csv, pd.read_csv(curve.generated_csv))
            gen = self._generated_total(generated, curve.gas_mixture, curve.pressure_bar, concentration)
            plot_intensity, scale_factor = self._area_match_raw_to_generated(wavelength, raw_au, gen)
            normalisation_reference = "raw_area_matched_to_generated_total"
        else:
            scale_factor = 1.0 / curve.plot_scale

        return pd.DataFrame(
            {
                "comparison": config.name,
                "curve_name": curve.name,
                "curve_order": curve_order,
                "gas_mixture": curve.gas_mixture,
                "kind": "raw_area_matched" if curve.raw_normalisation == "area_to_generated" else "raw_direct_W_Nnorm",
                "label": curve.label,
                "pressure_bar": curve.pressure_bar,
                "concentration_percent": concentration,
                "concentration_fraction": concentration / 100.0,
                "wavelength_nm": wavelength,
                "intensity_ph_MeV_nm": plot_intensity,
                "intensity_raw_au": raw_au,
                "intensity_raw_W_Nnorm": raw_direct,
                "plot_intensity": plot_intensity,
                "plot_scale": scale_factor,
                "color": curve.color,
                "linestyle": curve.linestyle,
                "linewidth": curve.linewidth,
                "alpha": curve.alpha,
                "normalisation_reference": normalisation_reference,
            }
        )

    @staticmethod
    def _generated_total(generated: pd.DataFrame, gas_mixture: str, pressure: float, concentration: float) -> pd.DataFrame:
        return generated[
            (generated["gas_mixture"] == gas_mixture)
            & (generated["component"] == "total")
            & match_float(generated["pressure_bar"], pressure)
            & match_float(generated["concentration_percent"], concentration)
        ].sort_values("wavelength_nm").copy()

    @staticmethod
    def _area_match_raw_to_generated(
        wavelength_raw: np.ndarray,
        raw_intensity: np.ndarray,
        generated: pd.DataFrame,
    ) -> tuple[np.ndarray, float]:
        if generated.empty:
            return np.full_like(raw_intensity, np.nan, dtype=float), np.nan
        w_gen = generated["wavelength_nm"].to_numpy(dtype=float)
        y_gen = generated["intensity_ph_MeV_nm"].to_numpy(dtype=float)
        w_min = max(float(np.nanmin(wavelength_raw)), float(np.nanmin(w_gen)))
        w_max = min(float(np.nanmax(wavelength_raw)), float(np.nanmax(w_gen)))
        mask_raw = np.isfinite(wavelength_raw) & np.isfinite(raw_intensity) & (wavelength_raw >= w_min) & (wavelength_raw <= w_max)
        mask_gen = np.isfinite(w_gen) & np.isfinite(y_gen) & (w_gen >= w_min) & (w_gen <= w_max)
        if mask_raw.sum() < 2 or mask_gen.sum() < 2:
            return np.full_like(raw_intensity, np.nan, dtype=float), np.nan
        y_raw_pos = np.clip(raw_intensity, 0.0, None)
        raw_area = float(np.trapezoid(y_raw_pos[mask_raw], wavelength_raw[mask_raw]))
        gen_area = float(np.trapezoid(np.clip(y_gen[mask_gen], 0.0, None), w_gen[mask_gen]))
        if raw_area <= 0.0 or gen_area < 0.0 or not np.isfinite(raw_area) or not np.isfinite(gen_area):
            return np.full_like(raw_intensity, np.nan, dtype=float), np.nan
        scale = gen_area / raw_area
        return y_raw_pos * scale, scale

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        window = int(window)
        if window <= 1 or values.size < 3:
            return values
        if window % 2 == 0:
            window += 1
        window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
        if window < 3:
            return values
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(values, kernel, mode="same")

    @staticmethod
    def _raw_one(
        gas_mixture: str,
        raw: pd.DataFrame,
        concentration: float,
        pressure: float,
        spectrum_columns,
    ) -> pd.DataFrame | None:
        sub = raw[
            (raw["gas_mixture"] == gas_mixture)
            & raw["spectrum_column"].isin(spectrum_columns)
            & match_float(raw["pressure_bar"], pressure)
            & match_float(raw["concentration_percent"], concentration)
        ].copy()
        if sub.empty:
            return None
        first_spectrum = sub["spectrum_name"].iloc[0]
        sub = sub[sub["spectrum_name"] == first_spectrum].copy()
        return sub.sort_values("wavelength_nm").reset_index(drop=True)
