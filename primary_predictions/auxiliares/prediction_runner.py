from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .bands import asymmetric_errors, band_dataframe, combine_stat_syst, percentile_band
from .fit_products import FitProduct, FitProductStore
from .model_adapters import apply_normalization, prepare_parameters
from .plotter import plot_band, plot_multi_band
from .prediction_types import (
    BandCurveConfig,
    BandPlotConfig,
    ExperimentalOverlay,
    MultiBandPlotConfig,
    NormalizationConfig,
    PredictionPoint,
)
from .tables import export_normalization_comparison_table, export_prediction_table


class PredictionRunner:
    def __init__(self, project_root: Path, adapters: dict[str, object], overlays: list[ExperimentalOverlay] | None = None):
        self.project_root = Path(project_root).resolve()
        self.adapters = adapters
        self.store = FitProductStore(self.project_root)
        self.overlays = {overlay.id: overlay for overlay in overlays or []}
        self.predictions_dir = self.project_root / "data" / "Predictions"
        self.tables_dir = self.project_root / "data" / "Tables"

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
        if normalization.mode != "reference_norm":
            return None
        ref_name = normalization.reference_fit_name or current_fit_name
        return float(self.product(ref_name).central[0])

    def evaluate(self, fit_name: str, component: str, params: np.ndarray, concentration, pressure: float, normalization: NormalizationConfig):
        adapter = self.adapters[fit_name]
        params_eval = prepare_parameters(
            params,
            normalization,
            central_params=self.product(fit_name).central,
        )
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
                    "normalization_propagate_nnorm": bool(point.normalization.propagate_nnorm),
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


    def build_normalization_comparison_table(
        self,
        points: list[PredictionPoint],
        *,
        left_normalization: NormalizationConfig,
        right_normalization: NormalizationConfig,
        left_prefix: str = "arcf4_norm",
        right_prefix: str = "arn2_norm",
    ) -> pd.DataFrame:
        """Evaluate the same point list under two reference normalizations."""

        left_df = self.build_point_table([replace(point, normalization=left_normalization) for point in points])
        right_df = self.build_point_table([replace(point, normalization=right_normalization) for point in points])

        base_cols = [
            "id",
            "label",
            "tex_label",
            "gas",
            "channel",
            "fit_name",
            "component",
            "concentration",
            "pressure_bar",
            "unit",
            "note",
        ]
        value_cols = [
            "normalization_mode",
            "normalization_reference",
            "normalization_propagate_nnorm",
            "value",
            "stat_minus",
            "stat_plus",
            "syst_minus",
            "syst_plus",
            "total_minus",
            "total_plus",
        ]

        out = left_df[base_cols].copy()
        for col in value_cols:
            out[f"{col}_{left_prefix}"] = left_df[col].to_numpy()
            out[f"{col}_{right_prefix}"] = right_df[col].to_numpy()
        return out

    def export_normalization_comparison_table(
        self,
        df: pd.DataFrame,
        stem: str,
        *,
        caption: str,
        label: str,
        left_prefix: str = "arcf4_norm",
        right_prefix: str = "arn2_norm",
    ) -> tuple[Path, Path]:
        csv_path = self.predictions_dir / f"{stem}.csv"
        tex_path = self.tables_dir / f"{stem}.tex"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        export_normalization_comparison_table(
            df,
            tex_path,
            caption=caption,
            label=label,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
        )
        return csv_path, tex_path

    def run_normalization_comparison_points(
        self,
        points: list[PredictionPoint],
        stem: str,
        *,
        left_normalization: NormalizationConfig,
        right_normalization: NormalizationConfig,
        caption: str,
        label: str,
        left_prefix: str = "arcf4_norm",
        right_prefix: str = "arn2_norm",
    ) -> pd.DataFrame:
        df = self.build_normalization_comparison_table(
            points,
            left_normalization=left_normalization,
            right_normalization=right_normalization,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
        )
        csv_path, tex_path = self.export_normalization_comparison_table(
            df,
            stem,
            caption=caption,
            label=label,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
        )
        print(f"[primary_predictions] tabla comparación CSV: {csv_path}")
        print(f"[primary_predictions] tabla comparación TeX: {tex_path}")
        return df

    def evaluate_curve(self, config: BandPlotConfig, params: np.ndarray) -> np.ndarray:
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

    def build_band(self, config: BandPlotConfig) -> pd.DataFrame:
        product = self.product(config.fit_name)
        central = self.evaluate_curve(config, product.central)
        stat_samples = self.evaluate_curve_samples(config, product.stat_toys)
        syst_samples = self.evaluate_curve_samples(config, product.syst_toys)
        stat = percentile_band(stat_samples, central)
        syst = percentile_band(syst_samples, central)
        total = combine_stat_syst(central, stat, syst)
        return band_dataframe(config.grid, central, stat, syst, total)

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

    def build_or_load_curve_band(self, curve: BandCurveConfig, *, overwrite: bool = False, title: str = "", xlabel: str = r"Concentration [\%]", ylabel: str = r"Yield [ph/MeV]", xscale: str = "log", yscale: str = "log", xlim=None, ylim=None) -> pd.DataFrame:
        band_config = self._curve_to_band_config(curve, title=title, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)
        csv_path = self.predictions_dir / "Bands" / f"{curve.id}.csv"
        if csv_path.exists() and not overwrite:
            print(f"[primary_predictions] banda {curve.id}: usando caché {csv_path}")
            return pd.read_csv(csv_path)

        print(f"[primary_predictions] banda {curve.id}...")
        df = self.build_band(band_config)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[primary_predictions] banda CSV: {csv_path}")
        return df

    def run_points(self, points: list[PredictionPoint], stem: str, *, caption: str, label: str) -> pd.DataFrame:
        df = self.build_point_table(points)
        csv_path, tex_path = self.export_point_table(df, stem, caption=caption, label=label)
        print(f"[primary_predictions] tabla CSV: {csv_path}")
        print(f"[primary_predictions] tabla TeX: {tex_path}")
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
                print(f"[primary_predictions] banda {config.id}: usando caché {csv_path}")
                df = pd.read_csv(csv_path)
            else:
                print(f"[primary_predictions] banda {config.id}...")
                df = self.build_band(config)
            csv_path, plot_path = self.export_band(df, config, make_plot=make_plots)
            print(f"[primary_predictions] banda CSV: {csv_path}")
            if plot_path:
                print(f"[primary_predictions] plot: {plot_path}")
            out[config.id] = df
        print(f"[primary_predictions] bandas terminadas en {perf_counter() - t0:.1f} s")
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
            print(f"[primary_predictions] multibanda {config.id}...")
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
                print(f"[primary_predictions] multibanda plot: {config.output}")
            out[config.id] = band_dfs
        print(f"[primary_predictions] multibandas terminadas en {perf_counter() - t0:.1f} s")
        return out
