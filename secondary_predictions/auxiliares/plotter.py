from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paper_overlays import (
    PAPER_PRIMARY_IDS,
    get_primary_paper_overlay_series,
    paper_colors,
    plot_primary_paper_overlays,
)
from .prediction_types import BandPlotConfig, BandCurveConfig, ExperimentalOverlay, MultiBandPlotConfig


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
        "sys_stat": "sys_stat",
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


def configure_matplotlib(plt) -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4.3),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": False,
        }
    )


def _load_overlay(overlay: ExperimentalOverlay) -> pd.DataFrame:
    df = pd.read_csv(overlay.csv_path)
    for col, expected in overlay.conditions.items():
        if col not in df.columns:
            continue
        if isinstance(expected, tuple) and len(expected) >= 2 and expected[0] == "isclose":
            atol = float(expected[2]) if len(expected) > 2 else 1e-8
            df = df[np.isclose(df[col].to_numpy(dtype=float), float(expected[1]), atol=atol)]
        else:
            df = df[df[col] == expected]
    return df


def _expanded_xlim(xlim: tuple[float, float], xscale: str, *, frac: float = 0.035) -> tuple[float, float]:
    lo, hi = map(float, xlim)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return xlim
    if xscale == "log" and lo > 0.0 and hi > 0.0:
        factor = (hi / lo) ** frac
        return lo / factor, hi * factor
    span = hi - lo
    return lo - frac * span, hi + frac * span


def _x_values(df: pd.DataFrame, x_plot_factor: float = 1.0) -> np.ndarray:
    if "x" in df.columns:
        return df["x"].to_numpy(dtype=float) * float(x_plot_factor)
    return df["concentration"].to_numpy(dtype=float) * float(x_plot_factor)


def plot_band(
    band_df: pd.DataFrame,
    config: BandPlotConfig,
    *,
    output: Path,
    overlays: dict[str, ExperimentalOverlay] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    is_paper_primary = config.id in PAPER_PRIMARY_IDS
    if is_paper_primary:
        colors = paper_colors(plt, 4)
        central_color = colors[0]
        stat_color = colors[0]
        syst_color = colors[0]
    else:
        central_color = cmap(0.12)
        stat_color = cmap(0.10)
        syst_color = cmap(0.14)
    total_color = cmap(0.88)

    x = _x_values(band_df, config.x_plot_factor)
    central = band_df["central"].to_numpy(dtype=float)
    band_mode = _canonical_band_mode(config.band_mode)
    use_ocw = _mode_uses_ocw(band_mode) and {"ocw_low", "ocw_high", "ocw_optimum"}.issubset(band_df.columns)
    use_sys_stat = _mode_uses_sys_stat(band_mode)

    if use_ocw:
        ax.fill_between(
            x,
            band_df["ocw_low"].to_numpy(dtype=float),
            band_df["ocw_high"].to_numpy(dtype=float),
            color=cmap(0.55),
            alpha=0.40,
            label="OCW",
            linewidth=0,
        )

    if use_sys_stat and (not is_paper_primary) and config.show_total and {"total_low", "total_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["total_low"].to_numpy(dtype=float),
            band_df["total_high"].to_numpy(dtype=float),
            color=total_color,
            alpha=0.20,
            label=r"stat. $\oplus$ syst.",
            linewidth=0,
        )
    if use_sys_stat and config.show_syst and {"syst_low", "syst_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["syst_low"].to_numpy(dtype=float),
            band_df["syst_high"].to_numpy(dtype=float),
            color=syst_color,
            alpha=0.45 if is_paper_primary else 0.45,
            label="Sistemático" if is_paper_primary else "syst.",
            linewidth=0,
        )
    if use_sys_stat and config.show_stat and {"stat_low", "stat_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["stat_low"].to_numpy(dtype=float),
            band_df["stat_high"].to_numpy(dtype=float),
            color=stat_color,
            alpha=0.2 if is_paper_primary else 0.2,
            label="Estadístico" if is_paper_primary else "stat.",
            linewidth=0,
        )

    line_y = band_df["ocw_optimum"].to_numpy(dtype=float) if use_ocw else central
    line_label = "Primary fit" if is_paper_primary else ("OCW optimum" if use_ocw else "central")
    ax.plot(x, line_y, color=central_color, lw=2.0 if is_paper_primary else 1.9, label=line_label)

    if overlays:
        for overlay_id in config.overlays:
            if overlay_id not in overlays:
                continue
            overlay = overlays[overlay_id]
            df = _load_overlay(overlay)
            if df.empty:
                continue
            yerr = df[overlay.yerr_col].to_numpy(dtype=float) if overlay.yerr_col and overlay.yerr_col in df.columns else None
            ax.errorbar(
                df[overlay.x_col].to_numpy(dtype=float),
                df[overlay.y_col].to_numpy(dtype=float),
                yerr=yerr,
                fmt=overlay.marker,
                ms=4,
                capsize=2,
                label=overlay.label or overlay.id,
                color=overlay.color,
            )

    if is_paper_primary:
        plot_primary_paper_overlays(ax, config, output, plt)

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    if config.xlim:
        ax.set_xlim(*_expanded_xlim(config.xlim, config.xscale))
    if config.ylim:
        ax.set_ylim(*config.ylim)
    if is_paper_primary and config.id == "ArCF4_IR_primary_total":
        ax.legend(ncol=2)
    else:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_multi_band(
    band_dfs: dict[str, pd.DataFrame],
    config: MultiBandPlotConfig,
    *,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(len(config.curves), 2)))

    curve_colors: dict[str, object] = {}

    for idx, curve in enumerate(config.curves):
        if curve.id not in band_dfs:
            continue
        df = band_dfs[curve.id]
        color = colors[idx]
        x_plot_factor = curve.x_plot_factor if curve.x_plot_factor is not None else 100.0
        x = _x_values(df, x_plot_factor)
        y = df["central"].to_numpy(dtype=float)
        band_mode = _canonical_band_mode(getattr(curve, "band_mode", "sys_stat"))
        use_ocw = _mode_uses_ocw(band_mode) and {"ocw_low", "ocw_high", "ocw_optimum"}.issubset(df.columns)
        use_sys_stat = _mode_uses_sys_stat(band_mode)

        if use_ocw:
            ax.fill_between(
                x,
                df["ocw_low"].to_numpy(dtype=float),
                df["ocw_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.40,
                linewidth=0,
                label=f"{curve.label} OCW",
            )

        if use_sys_stat and curve.show_total and {"total_low", "total_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["total_low"].to_numpy(dtype=float),
                df["total_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.20,
                linewidth=0,
                label=rf"{curve.label} stat. $\oplus$ syst.",
            )
        if use_sys_stat and curve.show_syst and {"syst_low", "syst_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["syst_low"].to_numpy(dtype=float),
                df["syst_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.40,
                linewidth=0,
                label=f"{curve.label} OCW",
            )
        if use_sys_stat and curve.show_stat and {"stat_low", "stat_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["stat_low"].to_numpy(dtype=float),
                df["stat_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.08,
                linewidth=0,
                label="_nolegend_",
            )

        line_y = df["ocw_optimum"].to_numpy(dtype=float) if use_ocw else y
        line_label = f"{curve.label} (OCW)" if use_ocw else curve.label
        ax.plot(x, line_y, color=color, lw=2.0, label=line_label)
        curve_colors[curve.id] = color

        if curve.paper_overlay_id and isinstance(curve, BandCurveConfig):
            overlay_config = curve.as_band_plot_config(output=output)
            overlay_config = BandPlotConfig(
                **{**overlay_config.__dict__, "id": curve.paper_overlay_id}
            )
            for item in get_primary_paper_overlay_series(overlay_config, output):
                ax.errorbar(
                    np.asarray(item["x"], dtype=float),
                    np.asarray(item["y"], dtype=float),
                    yerr=np.asarray(item["yerr"], dtype=float),
                    marker=str(item.get("marker", "o")),
                    linestyle="none",
                    ms=float(item.get("ms", 4)),
                    color=color,
                    ecolor=color,
                    capsize=2,
                    label=str(item.get("label", f"X-ray {curve.label}")),
                )

    for series in config.experimental_series:
        color = series.color
        if color is None and series.color_from_curve_id:
            color = curve_colors.get(series.color_from_curve_id)
        if color is None:
            color = "black"

        x = np.asarray(series.x, dtype=float)
        y = np.asarray(series.y, dtype=float)
        yerr = None if series.yerr is None else np.asarray(series.yerr, dtype=float)
        markerfacecolor = series.markerfacecolor
        if markerfacecolor is None:
            markerfacecolor = "white" if str(series.marker).endswith("open") else color
        markeredgecolor = series.markeredgecolor or color
        marker = series.marker.replace("_open", "")
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            linestyle=series.linestyle,
            marker=marker,
            ms=float(series.markersize),
            color=color,
            ecolor=color,
            capsize=float(series.capsize),
            alpha=float(series.alpha),
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markeredgewidth=float(series.markeredgewidth),
            label=series.label or "_nolegend_",
            zorder=4.0,
        )

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    if config.xlim:
        ax.set_xlim(*_expanded_xlim(config.xlim, config.xscale))
    if config.ylim:
        ax.set_ylim(*config.ylim)
    ax.legend(loc=config.legend_loc, ncol=config.legend_ncol)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

def plot_metadata_curves(
    curve_dfs: dict[str, pd.DataFrame],
    config: MetadataPlotConfig,
    *,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(len(config.curves), 2)))

    for idx, curve in enumerate(config.curves):
        df = curve_dfs.get(curve.id)
        if df is None or df.empty:
            continue
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.plot(
            x,
            y,
            marker=curve.marker or config.marker,
            linestyle=curve.linestyle or config.linestyle,
            lw=float(config.linewidth),
            ms=float(config.markersize),
            color=colors[idx],
            label=curve.label,
        )

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    if config.xlim:
        ax.set_xlim(*_expanded_xlim(config.xlim, config.xscale))
    if config.ylim:
        ax.set_ylim(*config.ylim)
    ax.legend(loc=config.legend_loc, ncol=config.legend_ncol)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

