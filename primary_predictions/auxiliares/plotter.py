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
from .prediction_types import BandPlotConfig, ExperimentalOverlay, MultiBandPlotConfig


def configure_matplotlib(plt) -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "grid", "no-latex"])
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
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.edgecolor": "0.50",
            "legend.framealpha": 0.78,
        }
    )


def _clean_axis_label(label: str) -> str:
    """Avoid displaying LaTeX escape characters when ``no-latex`` is active."""
    text = str(label)
    while r"\%" in text:
        text = text.replace(r"\%", "%")
    return text.replace("$%$", "%")


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

    x = band_df["concentration"].to_numpy(dtype=float) * config.x_plot_factor
    central = band_df["central"].to_numpy(dtype=float)

    if (not is_paper_primary) and config.show_total and {"total_low", "total_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["total_low"].to_numpy(dtype=float),
            band_df["total_high"].to_numpy(dtype=float),
            color=total_color,
            alpha=0.22,
            label=r"stat. $\oplus$ syst.",
            linewidth=0,
        )
    if config.show_syst and {"syst_low", "syst_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["syst_low"].to_numpy(dtype=float),
            band_df["syst_high"].to_numpy(dtype=float),
            color=syst_color,
            alpha=0.45 if is_paper_primary else 0.45,
            label="Sistemático" if is_paper_primary else "syst.",
            linewidth=0,
        )
    if config.show_stat and {"stat_low", "stat_high"}.issubset(band_df.columns):
        ax.fill_between(
            x,
            band_df["stat_low"].to_numpy(dtype=float),
            band_df["stat_high"].to_numpy(dtype=float),
            color=stat_color,
            alpha=0.2 if is_paper_primary else 0.2,
            label="Estadístico" if is_paper_primary else "stat.",
            linewidth=0,
        )

    ax.plot(x, central, color=central_color, lw=2.0 if is_paper_primary else 1.9, label="Primary fit" if is_paper_primary else "central")

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
    ax.set_xlabel(_clean_axis_label(config.xlabel))
    ax.set_ylabel(_clean_axis_label(config.ylabel))
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

    for idx, curve in enumerate(config.curves):
        if curve.id not in band_dfs:
            continue
        df = band_dfs[curve.id]
        color = colors[idx]
        x = df["concentration"].to_numpy(dtype=float) * curve.x_plot_factor
        y = df["central"].to_numpy(dtype=float)

        # Uncertainty fills are intentionally omitted from the legend to keep it readable.
        if curve.show_total and {"total_low", "total_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["total_low"].to_numpy(dtype=float),
                df["total_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.14,
                linewidth=0,
                label="_nolegend_",
            )
        if curve.show_syst and {"syst_low", "syst_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["syst_low"].to_numpy(dtype=float),
                df["syst_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.18,
                linewidth=0,
                label="_nolegend_",
            )
        if curve.show_stat and {"stat_low", "stat_high"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["stat_low"].to_numpy(dtype=float),
                df["stat_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.08,
                linewidth=0,
                label="_nolegend_",
            )

        ax.plot(x, y, color=color, lw=2.0, label=curve.label)

        if curve.paper_overlay_id:
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

    ax.set_title(config.title)
    ax.set_xlabel(_clean_axis_label(config.xlabel))
    ax.set_ylabel(_clean_axis_label(config.ylabel))
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    if config.xlim:
        ax.set_xlim(*_expanded_xlim(config.xlim, config.xscale))
    if config.ylim:
        ax.set_ylim(*config.ylim)
    legend_kwargs = {"loc": config.legend_loc, "ncol": config.legend_ncol, "frameon": True}
    if getattr(config, "legend_fontsize", None) is not None:
        legend_kwargs["fontsize"] = float(config.legend_fontsize)
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
