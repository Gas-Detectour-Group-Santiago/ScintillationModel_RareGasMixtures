from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .integrators import IntegralResult


@dataclass(frozen=True)
class RatioPlotConfig:
    title: str
    xlabel: str = "Concentration [%]"
    ylabel: str = "Integral ratio"
    pressures_bar: tuple[float, ...] | None = None
    concentration_range_percent: tuple[float, float] | None = None
    ratio_names: tuple[str, ...] | None = None
    xscale: str = "auto"
    yscale: str = "auto"
    marker: str = "o"
    linewidth: float = 2.0
    figsize: tuple[float, float] = (6.8, 4.8)
    legend_title: str = "Pressure"


@dataclass(frozen=True)
class RatioGridPlotConfig:
    title: str
    xlabel: str = "Concentration [%]"
    ylabel: str = "Integral ratio"
    pressures_bar: tuple[float, ...] | None = None
    concentration_range_percent: tuple[float, float] | None = None
    ratio_names: tuple[str, ...] | None = None
    ratio_titles: dict[str, str] | None = None
    xscale: str = "auto"
    yscale: str = "auto"
    marker: str = "o"
    linewidth: float = 1.8
    ncols: int = 3
    panel_size: tuple[float, float] = (4.2, 3.2)
    legend_title: str = "Pressure"
    sharey: bool = True


@dataclass(frozen=True)
class GaussianPlotConfig:
    title: str = "Gaussian fit diagnostic"
    xlabel: str = "Wavelength [nm]"
    ylabel: str = "Intensity"
    figsize: tuple[float, float] = (8.2, 5.3)
    show_full_spectrum: bool = True
    show_window_lines: bool = True
    show_peak_markers: bool = True
    show_baseline: bool = False


@dataclass(frozen=True)
class GaussianMosaicPanel:
    result: IntegralResult
    title: str


@dataclass(frozen=True)
class GaussianMosaicPlotConfig:
    title: str = "Gaussian fit diagnostics"
    xlabel: str = "Wavelength [nm]"
    ylabel: str = "Intensity"
    nrows: int = 3
    ncols: int = 3
    panel_size: tuple[float, float] = (4.0, 2.8)
    show_full_spectrum: bool = True
    show_window_lines: bool = True
    show_peak_markers: bool = True
    show_baseline: bool = False
    sharey: bool = False
    x_padding_nm: float = 18.0
    legend: bool = True


class RatioPlotter:
    def __init__(self, *, use_science_style: bool = True, use_grid: bool = False) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        if use_science_style:
            try:
                import scienceplots  # noqa: F401

                plt.style.use(["science", "grid"] if use_grid else ["science"])
            except Exception:
                plt.style.use("default")
        else:
            plt.style.use("default")
        plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "legend.frameon": False})

    def plot_by_pressure(self, results: pd.DataFrame, fig_path: str | Path, config: RatioPlotConfig) -> None:
        df = _filter_results(results, config)
        if df.empty:
            raise RuntimeError("No rows left to plot after applying RatioPlotConfig filters.")

        ratio_names = list(df["ratio_name"].dropna().unique())
        pressures = np.sort(df["pressure_bar"].astype(float).unique())

        fig, ax = self.plt.subplots(figsize=config.figsize)
        colors = self.plt.cm.viridis(np.linspace(0.08, 0.92, max(len(pressures), 1)))

        multiple_ratios = len(ratio_names) > 1
        for color, pressure in zip(colors, pressures, strict=False):
            subset_p = df[np.isclose(df["pressure_bar"].astype(float), pressure)]
            for ratio_name in ratio_names:
                subset = subset_p[subset_p["ratio_name"].eq(ratio_name)].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values("concentration_percent")
                x_values = pd.to_numeric(subset["concentration_percent"], errors="coerce")
                y_values = pd.to_numeric(subset["ratio"], errors="coerce")
                finite = np.isfinite(x_values) & np.isfinite(y_values)
                if not np.any(finite):
                    continue
                label = f"{pressure:g} bar" if not multiple_ratios else f"{ratio_name}, {pressure:g} bar"
                ax.plot(
                    x_values[finite],
                    y_values[finite],
                    marker=config.marker,
                    lw=config.linewidth,
                    color=color,
                    label=label,
                )

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.grid(False)
        _apply_scales(ax, df, config.xscale, config.yscale)
        ax.legend(title=config.legend_title, fontsize=8, title_fontsize=9, loc="best")
        fig.tight_layout()
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        self.plt.close(fig)

    def plot_grid_by_pressure(self, results: pd.DataFrame, fig_path: str | Path, config: RatioGridPlotConfig) -> None:
        df = _filter_results_grid(results, config)
        if df.empty:
            raise RuntimeError("No rows left to plot after applying RatioGridPlotConfig filters.")

        ratio_names = list(config.ratio_names) if config.ratio_names is not None else list(df["ratio_name"].dropna().unique())
        ratio_names = [name for name in ratio_names if name in set(df["ratio_name"])]
        if not ratio_names:
            raise RuntimeError("No requested ratio_names are present in the results.")

        pressures = np.sort(df["pressure_bar"].astype(float).unique())
        ncols = max(1, int(config.ncols))
        nrows = ceil(len(ratio_names) / ncols)
        figsize = (config.panel_size[0] * ncols, config.panel_size[1] * nrows)

        fig, axes = self.plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=config.sharey)
        axes_arr = np.atleast_1d(axes).ravel()
        colors = self.plt.cm.viridis(np.linspace(0.08, 0.92, max(len(pressures), 1)))

        handles = None
        labels = None

        for ax, ratio_name in zip(axes_arr, ratio_names, strict=False):
            subset_r = df[df["ratio_name"].eq(ratio_name)].copy()
            for color, pressure in zip(colors, pressures, strict=False):
                subset = subset_r[np.isclose(subset_r["pressure_bar"].astype(float), pressure)].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values("concentration_percent")
                x_values = pd.to_numeric(subset["concentration_percent"], errors="coerce")
                y_values = pd.to_numeric(subset["ratio"], errors="coerce")
                finite = np.isfinite(x_values) & np.isfinite(y_values)
                if not np.any(finite):
                    continue
                ax.plot(
                    x_values[finite],
                    y_values[finite],
                    marker=config.marker,
                    lw=config.linewidth,
                    color=color,
                    label=f"{pressure:g} bar",
                )
            title = (config.ratio_titles or {}).get(str(ratio_name), str(ratio_name))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(config.xlabel)
            ax.set_ylabel(config.ylabel)
            ax.grid(False)
            _apply_scales(ax, subset_r, config.xscale, config.yscale)
            handles, labels = ax.get_legend_handles_labels()

        for ax in axes_arr[len(ratio_names):]:
            ax.set_visible(False)

        if handles and labels:
            fig.legend(handles, labels, title=config.legend_title, loc="upper center", ncol=min(len(labels), 5), fontsize=8)

        fig.suptitle(config.title, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        self.plt.close(fig)

    def plot_each_ratio_by_pressure(
        self,
        results: pd.DataFrame,
        output_dir: str | Path,
        config: RatioPlotConfig,
        *,
        filename_prefix: str = "integral_ratio",
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for ratio_name in results["ratio_name"].dropna().unique():
            one = results[results["ratio_name"].eq(ratio_name)].copy()
            safe_name = _safe_filename(str(ratio_name))
            fig_path = output_dir / f"{filename_prefix}_{safe_name}.pdf"
            plot_config = RatioPlotConfig(
                title=config.title if len(results["ratio_name"].unique()) == 1 else f"{config.title}: {ratio_name}",
                xlabel=config.xlabel,
                ylabel=config.ylabel,
                pressures_bar=config.pressures_bar,
                concentration_range_percent=config.concentration_range_percent,
                ratio_names=(str(ratio_name),),
                xscale=config.xscale,
                yscale=config.yscale,
                marker=config.marker,
                linewidth=config.linewidth,
                figsize=config.figsize,
                legend_title=config.legend_title,
            )
            self.plot_by_pressure(one, fig_path, plot_config)
            paths.append(fig_path)
        return paths


class GaussianFitPlotter:
    def __init__(self, *, use_science_style: bool = True, use_grid: bool = False) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        if use_science_style:
            try:
                import scienceplots  # noqa: F401

                plt.style.use(["science", "grid"] if use_grid else ["science"])
            except Exception:
                plt.style.use("default")
        else:
            plt.style.use("default")
        plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "legend.frameon": False})

    def plot(self, result: IntegralResult, fig_path: str | Path, config: GaussianPlotConfig | None = None) -> None:
        config = config or GaussianPlotConfig()
        payload = result.plot_payload
        if payload is None:
            raise RuntimeError("IntegralResult has no plot_payload. This is only available for gaussian_fit results.")

        fig, ax = self.plt.subplots(figsize=config.figsize)
        self._draw_gaussian_fit(
            ax,
            result,
            show_full_spectrum=config.show_full_spectrum,
            show_window_lines=config.show_window_lines,
            show_peak_markers=config.show_peak_markers,
            show_baseline=config.show_baseline,
            detailed_component_labels=True,
        )
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.grid(False)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        self.plt.close(fig)

    def plot_mosaic(
        self,
        panels: Sequence[GaussianMosaicPanel],
        fig_path: str | Path,
        config: GaussianMosaicPlotConfig | None = None,
    ) -> None:
        config = config or GaussianMosaicPlotConfig()
        panels = list(panels)
        if not panels:
            raise RuntimeError("No Gaussian panels were provided for the mosaic.")

        nrows = max(1, int(config.nrows))
        ncols = max(1, int(config.ncols))
        capacity = nrows * ncols
        if len(panels) > capacity:
            panels = panels[:capacity]

        figsize = (config.panel_size[0] * ncols, config.panel_size[1] * nrows)
        fig, axes = self.plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=config.sharey)
        axes_arr = np.atleast_1d(axes).ravel()

        handles = None
        labels = None
        for idx, (ax, panel) in enumerate(zip(axes_arr, panels, strict=False)):
            self._draw_gaussian_fit(
                ax,
                panel.result,
                show_full_spectrum=config.show_full_spectrum,
                show_window_lines=config.show_window_lines,
                show_peak_markers=config.show_peak_markers,
                show_baseline=config.show_baseline,
                detailed_component_labels=False,
                label_curves=(idx == 0),
            )
            payload = panel.result.plot_payload or {}
            xmin = float(payload.get("xmin_nm", np.nan))
            xmax = float(payload.get("xmax_nm", np.nan))
            if np.isfinite(xmin) and np.isfinite(xmax):
                ax.set_xlim(xmin - config.x_padding_nm, xmax + config.x_padding_nm)
            ax.set_title(panel.title, fontsize=9)
            ax.set_xlabel(config.xlabel, fontsize=8)
            ax.set_ylabel(config.ylabel, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(False)
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()

        for ax in axes_arr[len(panels):]:
            ax.set_visible(False)

        if config.legend and handles and labels:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(labels), 5),
                fontsize=8,
                bbox_to_anchor=(0.5, 0.975),
            )
            rect = (0.0, 0.0, 1.0, 0.90)
            suptitle_y = 0.995
        else:
            rect = (0.0, 0.0, 1.0, 0.94)
            suptitle_y = 0.995

        fig.suptitle(config.title, y=suptitle_y)
        fig.tight_layout(rect=rect)
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        self.plt.close(fig)

    def _draw_gaussian_fit(
        self,
        ax,
        result: IntegralResult,
        *,
        show_full_spectrum: bool,
        show_window_lines: bool,
        show_peak_markers: bool,
        show_baseline: bool,
        detailed_component_labels: bool,
        label_curves: bool = True,
    ) -> None:
        payload = result.plot_payload
        if payload is None:
            raise RuntimeError("IntegralResult has no plot_payload. This is only available for gaussian_fit results.")

        full_x = np.asarray(payload["full_x_nm"], dtype=float)
        full_y = np.asarray(payload["full_y"], dtype=float)
        window_x = np.asarray(payload["window_x_nm"], dtype=float)
        window_y = np.asarray(payload["window_y"], dtype=float)
        fit_x = np.asarray(payload["fit_x_nm"], dtype=float)
        fit_total_y = np.asarray(payload["fit_total_y"], dtype=float)
        fit_baseline_y = np.asarray(payload["fit_baseline_y"], dtype=float) if show_baseline else None
        fit_component_ys = [np.asarray(arr, dtype=float) for arr in payload["fit_component_ys"]]
        peak_names = list(payload["peak_names"])
        component_integrals = list(payload["component_integrals"])
        centres_nm = list(payload["centres_nm"])
        sigmas_nm = list(payload["sigmas_nm"])
        xmin = float(payload["xmin_nm"])
        xmax = float(payload["xmax_nm"])
        total_integral = float(payload["integral_value"])

        no_label = "_nolegend_"
        if show_full_spectrum:
            ax.plot(full_x, full_y, lw=0.9, alpha=0.30, label="Spectrum" if label_curves else no_label)
        ax.plot(window_x, window_y, lw=1.25, label="Fit window" if label_curves else no_label)
        total_label = f"Total fit ({total_integral:.2e})" if detailed_component_labels else "Total fit"
        ax.plot(fit_x, fit_total_y, lw=1.65, label=total_label if label_curves else no_label, zorder=5)
        if show_baseline and fit_baseline_y is not None:
            ax.plot(fit_x, fit_baseline_y, lw=0.95, alpha=0.75, label="Offset" if label_curves else no_label, zorder=4)

        for idx, (comp_y, peak_name, integral_value, centre, sigma) in enumerate(
            zip(fit_component_ys, peak_names, component_integrals, centres_nm, sigmas_nm, strict=False)
        ):
            if detailed_component_labels:
                label = f"{peak_name}: μ={centre:.2f} nm, σ={sigma:.2f} nm, A={integral_value:.3e}"
            else:
                label = "Gaussian components" if idx == 0 else no_label
            ax.plot(fit_x, comp_y, lw=1.05, alpha=0.95, label=label if label_curves or detailed_component_labels else no_label, zorder=6)

        if show_peak_markers:
            ymax = float(np.nanmax(window_y)) if window_y.size else np.nan
            ymin = float(np.nanmin(window_y)) if window_y.size else np.nan
            text_y = ymin + 0.92 * (ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else None
            for idx, (peak_name, centre) in enumerate(zip(peak_names, centres_nm, strict=False)):
                label = "Peak centres" if idx == 0 and label_curves and not detailed_component_labels else "_nolegend_"
                ax.axvline(float(centre), lw=0.55, alpha=0.38, ls=":", label=label)
                if detailed_component_labels and text_y is not None:
                    ax.text(
                        float(centre),
                        text_y,
                        str(peak_name).replace("N2_", "").replace("N2plus_", "N2+ "),
                        rotation=90,
                        fontsize=6.5,
                        ha="center",
                        va="top",
                        alpha=0.75,
                    )

        if show_window_lines:
            ax.axvline(xmin, lw=0.8, alpha=0.55)
            ax.axvline(xmax, lw=0.8, alpha=0.55)


def _filter_results(results: pd.DataFrame, config: RatioPlotConfig) -> pd.DataFrame:
    df = results.copy()
    df = df[df.get("status", "ok").eq("ok") if "status" in df.columns else np.ones(len(df), dtype=bool)]
    if "ratio" in df.columns:
        ratio = pd.to_numeric(df["ratio"], errors="coerce")
        df = df[np.isfinite(ratio)]
    if config.ratio_names is not None:
        df = df[df["ratio_name"].isin(config.ratio_names)]
    if config.pressures_bar is not None:
        mask = np.zeros(len(df), dtype=bool)
        for pressure in config.pressures_bar:
            mask |= np.isclose(df["pressure_bar"].astype(float), float(pressure))
        df = df[mask]
    if config.concentration_range_percent is not None:
        cmin, cmax = config.concentration_range_percent
        c = df["concentration_percent"].astype(float)
        df = df[(c >= cmin) & (c <= cmax)]
    return df


def _filter_results_grid(results: pd.DataFrame, config: RatioGridPlotConfig) -> pd.DataFrame:
    df = results.copy()
    df = df[df.get("status", "ok").eq("ok") if "status" in df.columns else np.ones(len(df), dtype=bool)]
    if "ratio" in df.columns:
        ratio = pd.to_numeric(df["ratio"], errors="coerce")
        df = df[np.isfinite(ratio)]
    if config.ratio_names is not None:
        df = df[df["ratio_name"].isin(config.ratio_names)]
    if config.pressures_bar is not None:
        mask = np.zeros(len(df), dtype=bool)
        for pressure in config.pressures_bar:
            mask |= np.isclose(df["pressure_bar"].astype(float), float(pressure))
        df = df[mask]
    if config.concentration_range_percent is not None:
        cmin, cmax = config.concentration_range_percent
        c = df["concentration_percent"].astype(float)
        df = df[(c >= cmin) & (c <= cmax)]
    return df


def _apply_scales(ax, df: pd.DataFrame, xscale: str, yscale: str) -> None:
    if df.empty:
        return
    x = pd.to_numeric(df["concentration_percent"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["ratio"], errors="coerce").to_numpy()
    finite_x = x[np.isfinite(x) & (x > 0)]
    finite_y = y[np.isfinite(y) & (y > 0)]

    if xscale == "log" or (xscale == "auto" and finite_x.size and finite_x.max() / finite_x.min() > 5.0):
        ax.set_xscale("log")
    if yscale == "log" or (yscale == "auto" and finite_y.size and finite_y.max() / finite_y.min() > 5.0):
        ax.set_yscale("log")


def _safe_filename(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "ratio"
