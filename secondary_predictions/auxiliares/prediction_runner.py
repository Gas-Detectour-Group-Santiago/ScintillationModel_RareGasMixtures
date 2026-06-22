from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from itertools import product as iter_product
from time import perf_counter

import numpy as np
import pandas as pd

from .bands import asymmetric_errors, band_dataframe, combine_stat_syst, percentile_band
from .fit_products import FitProduct, FitProductStore
from .model_adapters import apply_normalization, apply_secondary_normalization, prepare_parameters
from .plotter import plot_band, plot_metadata_curves, plot_multi_band
from .prediction_types import (
    BandCurveConfig,
    CombinedBandCurveConfig,
    BandPlotConfig,
    ExperimentalOverlay,
    MetadataCurveConfig,
    MetadataPlotConfig,
    MultiBandPlotConfig,
    NormalizationConfig,
    OCWBandConfig,
    PredictionPoint,
)
from .tables import export_prediction_table


def _canonical_band_mode(mode: str | None) -> str:
    value = (mode or "sys_stat").strip().lower()
    for sep in ("+", "⊕", "&", "-", " "):
        value = value.replace(sep, "_")
    while "__" in value:
        value = value.replace("__", "_")
    value = value.strip("_")
    aliases = {
        "sys": "sys_stat",
        "stat_syst": "sys_stat",
        "syst_stat": "sys_stat",
        "sys_syst": "sys_stat",
        "sys_stat": "sys_stat",
        "sys_stat": "sys_stat",
        "systematic_statistical": "sys_stat",
        "ocw": "ocw_bands",
        "ocw_band": "ocw_bands",
        "ocw_bands": "ocw_bands",
        "both": "sum",
        "all": "sum",
        "sum": "sum",
    }
    return aliases.get(value, value)


def _mode_uses_ocw(mode: str | None) -> bool:
    return _canonical_band_mode(mode) in {"ocw_bands", "sum"}


def _mode_uses_sys_stat(mode: str | None) -> bool:
    return _canonical_band_mode(mode) in {"sys_stat", "sum"}


class PredictionRunner:
    def __init__(
        self,
        project_root: Path,
        adapters: dict[str, object],
        overlays: list[ExperimentalOverlay] | None = None,
        *,
        predictions_subdir: str | Path = "Predictions",
        log_prefix: str = "[secondary_predictions]",
    ):
        self.project_root = Path(project_root).resolve()
        self.adapters = adapters
        self.store = FitProductStore(self.project_root)
        self.overlays = {overlay.id: overlay for overlay in overlays or []}
        self.predictions_dir = self.project_root / "data" / Path(predictions_subdir)
        self.tables_dir = self.project_root / "data" / "Tables"
        self.log_prefix = log_prefix

        self._products: dict[str, FitProduct] = {}

    @staticmethod
    def _progress(iterable, *, total: int, desc: str):
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=total, desc=desc, unit="toy", dynamic_ncols=True)
        except ModuleNotFoundError:
            return iterable

    def product(self, fit_name: str) -> FitProduct:
        if fit_name not in self._products:
            self._products[fit_name] = self.store.load(fit_name)
        return self._products[fit_name]

    def reference_norm(self, normalization: NormalizationConfig, current_fit_name: str) -> float | None:
        if normalization.mode not in {"reference_norm", "secondary"}:
            return None
        if normalization.mode == "secondary" and not normalization.reference_fit_name:
            return None
        ref_name = normalization.reference_fit_name or current_fit_name
        return float(self.product(ref_name).central[0])

    def evaluate(self, fit_name: str, component: str, params: np.ndarray, concentration, pressure: float, normalization: NormalizationConfig):
        adapter = self.adapters[fit_name]
        params_eval = prepare_parameters(params, normalization)
        raw = adapter.evaluate_raw(params_eval, component, concentration, pressure)
        return apply_normalization(
            raw,
            params_eval,
            normalization,
            reference_norm=self.reference_norm(normalization, fit_name),
        )

    def evaluate_point(self, point: PredictionPoint, params: np.ndarray):
        return self.evaluate(
            point.fit_name,
            point.component,
            params,
            point.concentration,
            point.pressure,
            point.normalization,
        )

    def evaluate_point_samples(self, point: PredictionPoint, toy_params: np.ndarray) -> np.ndarray:
        if toy_params.ndim != 2 or toy_params.shape[0] == 0:
            return np.empty((0,), dtype=float)

        rows = []
        for params in self._progress(toy_params, total=len(toy_params), desc=f"{point.id} toys"):
            try:
                rows.append(float(np.ravel(self.evaluate_point(point, params))[0]))
            except Exception:
                rows.append(np.nan)
        return np.asarray(rows, dtype=float)

    def build_point_table(self, points: list[PredictionPoint]) -> pd.DataFrame:
        rows = []
        for point in points:
            product = self.product(point.fit_name)
            central = float(np.ravel(self.evaluate_point(point, product.central))[0])
            stat_samples = self.evaluate_point_samples(point, product.stat_toys)
            syst_samples = self.evaluate_point_samples(point, product.syst_toys)

            stat_minus, stat_plus = asymmetric_errors(np.asarray([central]), stat_samples[:, None] if stat_samples.size else np.empty((0, 1)))
            syst_minus, syst_plus = asymmetric_errors(np.asarray([central]), syst_samples[:, None] if syst_samples.size else np.empty((0, 1)))

            stat_minus_v = float(stat_minus[0]) if len(stat_minus) else np.nan
            stat_plus_v = float(stat_plus[0]) if len(stat_plus) else np.nan
            syst_minus_v = float(syst_minus[0]) if len(syst_minus) else np.nan
            syst_plus_v = float(syst_plus[0]) if len(syst_plus) else np.nan

            rows.append(
                {
                    "id": point.id,
                    "label": point.label,
                    "tex_label": point.label,
                    "gas": point.gas,
                    "channel": point.channel,
                    "fit_name": point.fit_name,
                    "component": point.component,
                    "concentration": point.concentration,
                    "pressure_bar": point.pressure,
                    "normalization_mode": point.normalization.mode,
                    "normalization_reference": point.normalization.reference_fit_name or "",
                    "unit": point.normalization.output_unit,
                    "value": central,
                    "stat_minus": stat_minus_v,
                    "stat_plus": stat_plus_v,
                    "syst_minus": syst_minus_v,
                    "syst_plus": syst_plus_v,
                    "total_minus": float(np.sqrt(np.nan_to_num(stat_minus_v) ** 2 + np.nan_to_num(syst_minus_v) ** 2)),
                    "total_plus": float(np.sqrt(np.nan_to_num(stat_plus_v) ** 2 + np.nan_to_num(syst_plus_v) ** 2)),
                    "note": point.note,
                }
            )
        return pd.DataFrame(rows)

    def export_point_table(self, df: pd.DataFrame, stem: str, *, caption: str, label: str) -> tuple[Path, Path]:
        csv_path = self.predictions_dir / f"{stem}.csv"
        tex_path = self.tables_dir / f"{stem}.tex"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        export_prediction_table(df, tex_path, caption=caption, label=label)
        return csv_path, tex_path

    def evaluate_curve(self, config: BandPlotConfig, params: np.ndarray) -> np.ndarray:
        if config.selection is None:
            return np.asarray(
                self.evaluate(
                    config.fit_name,
                    config.component,
                    params,
                    config.grid,
                    config.pressure,
                    config.normalization,
                ),
                dtype=float,
            )

        adapter = self.adapters[config.fit_name]
        if not hasattr(adapter, "evaluate_secondary_raw"):
            raise TypeError(f"El adapter {config.fit_name!r} no soporta selecciones Garfield.")

        params_eval = prepare_parameters(params, config.normalization)
        raw, metadata = adapter.evaluate_secondary_raw(
            params_eval,
            config.component,
            config.selection,
            config.grid,
            pressure=config.pressure,
            x_axis=config.x_axis,
        )
        return np.asarray(
            apply_secondary_normalization(
                raw,
                params_eval,
                config.normalization,
                metadata,
                normalize_by=config.selection.normalize_by,
                npe_column=config.selection.npe_column,
                reference_norm=self.reference_norm(config.normalization, config.fit_name),
            ),
            dtype=float,
        )

    def evaluate_curve_samples(self, config: BandPlotConfig, toys: np.ndarray) -> np.ndarray:
        if toys.ndim != 2 or toys.shape[0] == 0:
            return np.empty((0, len(config.grid)), dtype=float)

        rows = []
        for params in self._progress(toys, total=len(toys), desc=f"{config.id} toys"):
            try:
                rows.append(self.evaluate_curve(config, params))
            except Exception:
                rows.append(np.full(len(config.grid), np.nan, dtype=float))
        return np.asarray(rows, dtype=float)

    def _ocw_params_for_side(self, product: FitProduct, ocw: OCWBandConfig, side: str) -> np.ndarray:
        params = np.asarray(product.central, dtype=float).copy()
        name_to_index = {name: i for i, name in enumerate(product.parameter_names)}
        missing: list[str] = []

        for rule in ocw.rules:
            idx = name_to_index.get(rule.name)
            if idx is None:
                missing.append(rule.name)
                continue
            params[idx] = rule.apply(params[idx], side)

        if missing and ocw.strict:
            raise KeyError(
                f"La configuración OCW {ocw.id!r} contiene parámetros ausentes en {product.name!r}: {missing}. "
                f"Disponibles: {product.parameter_names}"
            )
        return params

    def _ocw_corner_parameters(self, product: FitProduct, ocw: OCWBandConfig) -> list[np.ndarray]:
        name_to_index = {name: i for i, name in enumerate(product.parameter_names)}
        active_rules = [rule for rule in ocw.rules if rule.name in name_to_index]
        missing = [rule.name for rule in ocw.rules if rule.name not in name_to_index]
        if missing and ocw.strict:
            raise KeyError(
                f"La configuración OCW {ocw.id!r} contiene parámetros ausentes en {product.name!r}: {missing}. "
                f"Disponibles: {product.parameter_names}"
            )
        if not active_rules:
            return [np.asarray(product.central, dtype=float).copy()]

        corners: list[np.ndarray] = []
        for sides in iter_product(("low", "high"), repeat=len(active_rules)):
            params = np.asarray(product.central, dtype=float).copy()
            for rule, side in zip(active_rules, sides):
                idx = name_to_index[rule.name]
                params[idx] = rule.apply(params[idx], side)
            corners.append(params)
        return corners

    def build_ocw_band(self, config: BandPlotConfig, product: FitProduct) -> dict[str, np.ndarray]:
        ocw = config.ocw_config
        if ocw is None:
            nan = np.full(len(config.grid), np.nan, dtype=float)
            return {"low": nan, "high": nan, "optimum": nan}

        optimum = self.evaluate_curve(config, self._ocw_params_for_side(product, ocw, "optimum"))

        if ocw.use_corners:
            samples = np.vstack([self.evaluate_curve(config, params) for params in self._ocw_corner_parameters(product, ocw)])
            low = np.nanmin(samples, axis=0)
            high = np.nanmax(samples, axis=0)
        else:
            low_candidate = self.evaluate_curve(config, self._ocw_params_for_side(product, ocw, "low"))
            high_candidate = self.evaluate_curve(config, self._ocw_params_for_side(product, ocw, "high"))
            low = np.minimum(low_candidate, high_candidate)
            high = np.maximum(low_candidate, high_candidate)

        return {"low": np.asarray(low, dtype=float), "high": np.asarray(high, dtype=float), "optimum": np.asarray(optimum, dtype=float)}

    def build_band(self, config: BandPlotConfig) -> pd.DataFrame:
        product = self.product(config.fit_name)
        central = self.evaluate_curve(config, product.central)

        use_sys_stat = _mode_uses_sys_stat(config.band_mode)
        use_ocw = _mode_uses_ocw(config.band_mode) and config.ocw_config is not None

        need_stat = use_sys_stat and (config.show_stat or config.show_total)
        need_syst = use_sys_stat and (config.show_syst or config.show_total)
        stat_samples = self.evaluate_curve_samples(config, product.stat_toys) if need_stat else np.empty((0, len(config.grid)), dtype=float)
        syst_samples = self.evaluate_curve_samples(config, product.syst_toys) if need_syst else np.empty((0, len(config.grid)), dtype=float)

        stat = percentile_band(stat_samples, central)
        syst = percentile_band(syst_samples, central)
        total = combine_stat_syst(central, stat, syst)
        df = band_dataframe(config.grid, central, stat, syst, total)
        df["fit_central"] = central
        df["band_mode"] = _canonical_band_mode(config.band_mode)

        if use_ocw:
            ocw = self.build_ocw_band(config, product)
            df["ocw_low"] = ocw["low"]
            df["ocw_high"] = ocw["high"]
            df["ocw_optimum"] = ocw["optimum"]
            df["ocw_config_id"] = config.ocw_config.id
            if use_sys_stat:
                self._shift_intervals_to_reference(df, ocw["optimum"])

        if config.selection is not None:
            adapter = self.adapters[config.fit_name]
            if hasattr(adapter, "secondary_metadata"):
                metadata = adapter.secondary_metadata(config.selection, config.grid, x_axis=config.x_axis)
                df["selection_id"] = str(metadata.get("selection_id", config.selection.id))
                df["n_selected_rows"] = int(metadata.get("n_selected_rows", 0))
                df["normalization_denominator"] = config.selection.normalize_by
                df["x_axis"] = str(metadata.get("x_axis", config.x_axis))
                df["x_column"] = str(metadata.get("x_column", config.x_axis))
                for key in ("x", "concentration", "concentration_percent", "pressure", "gap_mm", "electric_field", "npe", "ne", "ni", "gain"):
                    value = metadata.get(key)
                    if value is not None:
                        df[key] = np.asarray(value, dtype=float)
        if "x" not in df.columns:
            df["x"] = np.asarray(config.grid, dtype=float)
        if "x_axis" not in df.columns:
            df["x_axis"] = config.x_axis
        return df

    def export_band(self, df: pd.DataFrame, config: BandPlotConfig, *, make_plot: bool = True) -> tuple[Path, Path | None]:
        csv_path = self.predictions_dir / "Bands" / f"{config.id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        plot_path = config.output
        if plot_path is not None and make_plot:
            plot_band(df, config, output=plot_path, overlays=self.overlays)
        return csv_path, plot_path if make_plot else None

    def _curve_to_band_config(self, curve: BandCurveConfig, *, title: str = "", xlabel: str = r"Concentration [\%]", ylabel: str = r"Yield [ph/MeV]", xscale: str = "log", yscale: str = "log", xlim=None, ylim=None) -> BandPlotConfig:
        return curve.as_band_plot_config(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            output=None,
        )

    @staticmethod
    def _band_errors(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        central = df["central"].to_numpy(dtype=float)
        low_col = f"{prefix}_low"
        high_col = f"{prefix}_high"
        if low_col not in df.columns or high_col not in df.columns:
            zeros = np.zeros_like(central, dtype=float)
            return zeros, zeros
        low = df[low_col].to_numpy(dtype=float)
        high = df[high_col].to_numpy(dtype=float)
        minus = np.nan_to_num(central - low, nan=0.0, posinf=0.0, neginf=0.0)
        plus = np.nan_to_num(high - central, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(minus, 0.0, None), np.clip(plus, 0.0, None)

    @staticmethod
    def _shift_intervals_to_reference(df: pd.DataFrame, reference: np.ndarray, prefixes: tuple[str, ...] = ("stat", "syst", "total")) -> None:
        """Move existing asymmetric intervals so their width follows a new optimum line.

        The toy/stat/syst machinery estimates an asymmetric minus/plus width
        around the fit central value.  When an OCW optimum is selected, the
        physically plotted line is no longer the fit central curve, so the same
        widths must be carried to the OCW optimum instead of staying attached
        to the old central.
        """
        reference = np.asarray(reference, dtype=float)
        central = df["central"].to_numpy(dtype=float)
        for prefix in prefixes:
            low_col = f"{prefix}_low"
            high_col = f"{prefix}_high"
            if low_col not in df.columns or high_col not in df.columns:
                continue
            low = df[low_col].to_numpy(dtype=float)
            high = df[high_col].to_numpy(dtype=float)
            minus = np.clip(np.nan_to_num(central - low, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
            plus = np.clip(np.nan_to_num(high - central, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
            df[low_col] = reference - minus
            df[high_col] = reference + plus

    @staticmethod
    def _align_band_to_grid(df: pd.DataFrame, x_grid: np.ndarray) -> pd.DataFrame:
        x_grid = np.asarray(x_grid, dtype=float)
        x_col = "x" if "x" in df.columns else "concentration"
        if x_col not in df.columns:
            raise KeyError("El CSV de banda no contiene columna 'x' ni 'concentration'.")
        x = df[x_col].to_numpy(dtype=float)
        if len(x) == len(x_grid) and np.allclose(x, x_grid, rtol=1e-9, atol=1e-12, equal_nan=False):
            return df.reset_index(drop=True).copy()

        out = pd.DataFrame({"x": x_grid})
        if "x_axis" in df.columns:
            out["x_axis"] = str(df["x_axis"].iloc[0])
        order = np.argsort(x)
        x_sorted = x[order]
        for col in df.columns:
            if col in {"x"}:
                continue
            values_numeric = pd.to_numeric(df[col], errors="coerce")
            if values_numeric.notna().sum() == 0:
                if col in {"x_axis", "x_column", "selection_id", "normalization_denominator", "band_mode", "ocw_config_id", "combined_from", "combine_operation", "combine_uncertainty_mode"}:
                    out[col] = df[col].iloc[0]
                continue
            values = values_numeric.to_numpy(dtype=float)[order]
            finite = np.isfinite(x_sorted) & np.isfinite(values)
            if finite.sum() == 0:
                out[col] = np.nan
            elif finite.sum() == 1:
                out[col] = values[finite][0]
            else:
                out[col] = np.interp(x_grid, x_sorted[finite], values[finite], left=values[finite][0], right=values[finite][-1])
        if "concentration" not in out.columns and "concentration" in df.columns:
            out["concentration"] = np.interp(x_grid, x_sorted, df["concentration"].to_numpy(dtype=float)[order])
        return out


    def combine_curve_bands(self, curve: CombinedBandCurveConfig, child_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if curve.operation != "sum":
            raise ValueError(f"Operación de curva combinada no soportada: {curve.operation!r}")
        if curve.uncertainty_mode != "quadrature":
            raise ValueError(f"Modo de incertidumbre combinado no soportado: {curve.uncertainty_mode!r}")
        if not child_dfs:
            raise ValueError(f"La curva combinada {curve.id!r} no contiene curvas hijas.")

        x_grid = curve.grid
        if x_grid.size == 0:
            x_grid = child_dfs[0]["concentration"].to_numpy(dtype=float)
        aligned = [self._align_band_to_grid(df, x_grid) for df in child_dfs]

        central = np.zeros(len(x_grid), dtype=float)
        stat_minus2 = np.zeros(len(x_grid), dtype=float)
        stat_plus2 = np.zeros(len(x_grid), dtype=float)
        syst_minus2 = np.zeros(len(x_grid), dtype=float)
        syst_plus2 = np.zeros(len(x_grid), dtype=float)
        any_ocw = any({"ocw_low", "ocw_high", "ocw_optimum"}.issubset(df.columns) for df in aligned)
        ocw_low = np.zeros(len(x_grid), dtype=float)
        ocw_high = np.zeros(len(x_grid), dtype=float)
        ocw_optimum = np.zeros(len(x_grid), dtype=float)

        for df in aligned:
            central += df["central"].to_numpy(dtype=float)
            stat_minus, stat_plus = self._band_errors(df, "stat")
            syst_minus, syst_plus = self._band_errors(df, "syst")
            stat_minus2 += stat_minus**2
            stat_plus2 += stat_plus**2
            syst_minus2 += syst_minus**2
            syst_plus2 += syst_plus**2
            if any_ocw:
                base = df["central"].to_numpy(dtype=float)
                ocw_low += df["ocw_low"].to_numpy(dtype=float) if "ocw_low" in df.columns else base
                ocw_high += df["ocw_high"].to_numpy(dtype=float) if "ocw_high" in df.columns else base
                ocw_optimum += df["ocw_optimum"].to_numpy(dtype=float) if "ocw_optimum" in df.columns else base

        stat_minus = np.sqrt(stat_minus2)
        stat_plus = np.sqrt(stat_plus2)
        syst_minus = np.sqrt(syst_minus2)
        syst_plus = np.sqrt(syst_plus2)
        total_minus = np.sqrt(stat_minus**2 + syst_minus**2)
        total_plus = np.sqrt(stat_plus**2 + syst_plus**2)

        out = pd.DataFrame(
            {
                "x": x_grid,
                "central": central,
                "fit_central": central,
                "stat_low": central - stat_minus,
                "stat_high": central + stat_plus,
                "syst_low": central - syst_minus,
                "syst_high": central + syst_plus,
                "total_low": central - total_minus,
                "total_high": central + total_plus,
                "band_mode": _canonical_band_mode(curve.band_mode),
                "x_axis": curve.x_axis,
                "combined_from": "+".join(child.id for child in curve.curves),
                "combine_operation": curve.operation,
                "combine_uncertainty_mode": curve.uncertainty_mode,
            }
        )
        if any_ocw:
            out["ocw_low"] = np.minimum(ocw_low, ocw_high)
            out["ocw_high"] = np.maximum(ocw_low, ocw_high)
            out["ocw_optimum"] = ocw_optimum
            if _mode_uses_sys_stat(curve.band_mode):
                self._shift_intervals_to_reference(out, ocw_optimum)

        # Keep useful Garfield metadata from the first child. These columns should
        # be identical for VIS and IR when they share the same selection.
        metadata_cols = (
            "selection_id",
            "n_selected_rows",
            "x_column",
            "concentration",
            "normalization_denominator",
            "concentration_percent",
            "pressure",
            "gap_mm",
            "electric_field",
            "npe",
            "ne",
            "ni",
            "gain",
        )
        first = aligned[0]
        for col in metadata_cols:
            if col in first.columns and col not in out.columns:
                values = first[col]
                if len(values) == len(out):
                    out[col] = values.to_numpy()
                elif len(values) == 1:
                    out[col] = values.iloc[0]
        return out

    def build_or_load_curve_band(self, curve: BandCurveConfig | CombinedBandCurveConfig, *, overwrite: bool = False, title: str = "", xlabel: str = r"Concentration [\%]", ylabel: str = r"Yield [ph/MeV]", xscale: str = "log", yscale: str = "log", xlim=None, ylim=None) -> pd.DataFrame:
        csv_path = self.predictions_dir / "Bands" / f"{curve.id}.csv"
        if csv_path.exists() and not overwrite:
            print(f"{self.log_prefix} banda {curve.id}: usando caché {csv_path}")
            return pd.read_csv(csv_path)

        if isinstance(curve, CombinedBandCurveConfig):
            print(f"{self.log_prefix} banda combinada {curve.id}...")
            child_dfs: list[pd.DataFrame] = []
            for child in curve.curves:
                child_for_combination = replace(
                    child,
                    show_stat=child.show_stat or curve.show_stat or curve.show_total,
                    show_syst=child.show_syst or curve.show_syst or curve.show_total,
                    show_total=child.show_total or curve.show_total,
                    band_mode=curve.band_mode if _mode_uses_ocw(curve.band_mode) else child.band_mode,
                )
                child_dfs.append(
                    self.build_or_load_curve_band(
                        child_for_combination,
                        overwrite=overwrite,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        xscale=xscale,
                        yscale=yscale,
                        xlim=xlim,
                        ylim=ylim,
                    )
                )
            df = self.combine_curve_bands(curve, child_dfs)
        else:
            print(f"{self.log_prefix} banda {curve.id}...")
            band_config = self._curve_to_band_config(curve, title=title, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)
            df = self.build_band(band_config)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"{self.log_prefix} banda CSV: {csv_path}")
        return df


    @staticmethod
    def _metadata_expression(df: pd.DataFrame, y: str) -> np.ndarray:
        """Evaluate a simple metadata expression on selected Garfield rows.

        Supported special names:
          - ni_minus_ne_over_ni, (ni-ne)/ni, ni-ne/ni

        Any real CSV column name is also accepted (for example ``ne``, ``ni``,
        ``gap_mm``, ``electric_field`` or ``npe``).
        """
        name = str(y).strip()
        key = name.lower().replace(" ", "").replace("$", "")
        aliases = {
            "ni_minus_ne_over_ni",
            "(ni-ne)/ni",
            "ni-ne/ni",
            "niminusneoverni",
            "n_i-n_e/n_i",
            "(n_i-n_e)/n_i",
        }
        if key in aliases:
            if "ni" not in df.columns or "ne" not in df.columns:
                raise KeyError("Para calcular (ni-ne)/ni hacen falta las columnas 'ni' y 'ne'.")
            ni = pd.to_numeric(df["ni"], errors="coerce").to_numpy(dtype=float)
            ne = pd.to_numeric(df["ne"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (ni - ne) / ni
            out[~np.isfinite(out)] = np.nan
            return out

        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

        # Common aliases for quick manual switches in configs.py.
        alias_map = {
            "e": "electric_field",
            "field": "electric_field",
            "electric_field_kvcm": "electric_field",
            "gap": "gap_mm",
            "pressure_bar": "pressure",
            "cf4": "concentration",
            "concentration_percent": "concentration",
            "gain": "ne",
        }
        resolved = alias_map.get(key)
        if resolved and resolved in df.columns:
            return pd.to_numeric(df[resolved], errors="coerce").to_numpy(dtype=float)

        raise KeyError(f"No sé calcular y={y!r}. Columnas disponibles: {list(df.columns)}")

    @staticmethod
    def _metadata_x_values(df: pd.DataFrame, x_axis: str) -> np.ndarray:
        axis = str(x_axis or "concentration").strip()
        key = axis.lower().replace(" ", "")
        alias_map = {
            "e": "electric_field",
            "field": "electric_field",
            "electric_field_kvcm": "electric_field",
            "gap": "gap_mm",
            "pressure_bar": "pressure",
            "cf4": "concentration",
            "concentration_percent": "concentration",
            "gain": "ne",
        }
        column = axis if axis in df.columns else alias_map.get(key, axis)
        if column not in df.columns:
            raise KeyError(f"No existe el eje x {x_axis!r}. Columnas disponibles: {list(df.columns)}")
        x = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if column == "concentration":
            # Analysis_secondary_garfield stores gas concentration in percent.
            # If a future CSV stores fractions, convert them to percent for the
            # paper-style concentration plots.
            finite = x[np.isfinite(x)]
            if finite.size and np.nanmax(finite) <= 1.5:
                x = x * 100.0
        return x

    def build_metadata_curve(self, config: MetadataPlotConfig, curve: MetadataCurveConfig) -> pd.DataFrame:
        adapter = self.adapters[config.adapter_name]
        if not hasattr(adapter, "select"):
            raise TypeError(f"El adapter {config.adapter_name!r} no permite seleccionar Garfield metadata.")

        selected = adapter.select(curve.selection).copy()
        if selected.empty:
            raise ValueError(f"La selección metadata {curve.selection.id!r} no deja ninguna fila.")

        x = self._metadata_x_values(selected, config.x_axis)
        y = self._metadata_expression(selected, config.y)
        out = pd.DataFrame(
            {
                "curve_id": curve.id,
                "curve_label": curve.label,
                "selection_id": curve.selection.id,
                "x_axis": config.x_axis,
                "y_expression": config.y,
                "x": x,
                "y": y,
            }
        )

        for col in (
            "file",
            "gas_mixture",
            "concentration",
            "electric_field",
            "gap_mm",
            "pressure",
            "npe",
            "ne",
            "ni",
            "ne_std",
            "ni_std",
            "n_entries",
        ):
            if col in selected.columns:
                out[col] = selected[col].to_numpy()

        out = out[np.isfinite(out["x"].to_numpy(dtype=float)) & np.isfinite(out["y"].to_numpy(dtype=float))].copy()
        if out.empty:
            return out

        if config.group_duplicate_x:
            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
            agg = {col: "mean" for col in numeric_cols if col != "x"}
            agg.update({
                "curve_id": "first",
                "curve_label": "first",
                "selection_id": "first",
                "x_axis": "first",
                "y_expression": "first",
            })
            # Keep a small diagnostic count in case several ROOT rows map to
            # the same x value.
            grouped = out.groupby("x", as_index=False).agg(agg)
            counts = out.groupby("x").size().rename("n_rows_at_x").reset_index()
            out = grouped.merge(counts, on="x", how="left")

        return out.sort_values("x").reset_index(drop=True)

    def run_metadata_plots(
        self,
        configs: list[MetadataPlotConfig],
        *,
        overwrite: bool = True,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        out: dict[str, dict[str, pd.DataFrame]] = {}
        t0 = perf_counter()
        for config in configs:
            print(f"{self.log_prefix} metadata {config.id}...")
            curve_dfs: dict[str, pd.DataFrame] = {}
            rows: list[pd.DataFrame] = []
            for curve in config.curves:
                df_curve = self.build_metadata_curve(config, curve)
                curve_dfs[curve.id] = df_curve
                if not df_curve.empty:
                    rows.append(df_curve)
            combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            csv_path = self.predictions_dir / "Metadata" / f"{config.id}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not csv_path.exists():
                combined.to_csv(csv_path, index=False)
            print(f"{self.log_prefix} metadata CSV: {csv_path}")
            if config.output is not None:
                plot_metadata_curves(curve_dfs, config, output=config.output)
                print(f"{self.log_prefix} metadata plot: {config.output}")
            out[config.id] = curve_dfs
        print(f"{self.log_prefix} metadata terminada en {perf_counter() - t0:.1f} s")
        return out

    def run_points(self, points: list[PredictionPoint], stem: str, *, caption: str, label: str) -> pd.DataFrame:
        df = self.build_point_table(points)
        csv_path, tex_path = self.export_point_table(df, stem, caption=caption, label=label)
        print(f"{self.log_prefix} tabla CSV: {csv_path}")
        print(f"{self.log_prefix} tabla TeX: {tex_path}")
        return df

    def run_bands(
        self,
        configs: list[BandPlotConfig],
        *,
        make_plots: bool = True,
        overwrite: bool = False,
    ) -> dict[str, pd.DataFrame]:
        out = {}
        t0 = perf_counter()
        for config in configs:
            csv_path = self.predictions_dir / "Bands" / f"{config.id}.csv"
            if csv_path.exists() and not overwrite:
                print(f"{self.log_prefix} banda {config.id}: usando caché {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                print(f"{self.log_prefix} banda {config.id}...")
                df = self.build_band(config)
            csv_path, plot_path = self.export_band(df, config, make_plot=make_plots)
            print(f"{self.log_prefix} banda CSV: {csv_path}")
            if plot_path:
                print(f"{self.log_prefix} plot: {plot_path}")
            out[config.id] = df
        print(f"{self.log_prefix} bandas terminadas en {perf_counter() - t0:.1f} s")
        return out

    def run_multi_bands(
        self,
        configs: list[MultiBandPlotConfig],
        *,
        overwrite: bool = False,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        out: dict[str, dict[str, pd.DataFrame]] = {}
        t0 = perf_counter()
        for config in configs:
            print(f"{self.log_prefix} multibanda {config.id}...")
            band_dfs: dict[str, pd.DataFrame] = {}
            for curve in config.curves:
                band_dfs[curve.id] = self.build_or_load_curve_band(
                    curve,
                    overwrite=overwrite,
                    title=config.title,
                    xlabel=config.xlabel,
                    ylabel=config.ylabel,
                    xscale=config.xscale,
                    yscale=config.yscale,
                    xlim=config.xlim,
                    ylim=config.ylim,
                )
            if config.output is not None:
                plot_multi_band(band_dfs, config, output=config.output)
                print(f"{self.log_prefix} multibanda plot: {config.output}")
            out[config.id] = band_dfs
        print(f"{self.log_prefix} multibandas terminadas en {perf_counter() - t0:.1f} s")
        return out
