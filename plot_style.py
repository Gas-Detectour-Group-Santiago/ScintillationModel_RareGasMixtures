#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Matplotlib style for the TFM figures.

Change this file to update the visual language of the whole plotting stack:
primary fits, primary/secondary predictions, spectra mosaics, integral
comparisons and population histograms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User-editable paper style switches
# =============================================================================
PAPER_CMAP = os.environ.get("TFM_PLOT_CMAP", "viridis")
PAPER_GRID = os.environ.get("TFM_PLOT_GRID", "0").strip().lower() in {"1", "true", "yes", "on"}
PAPER_USE_LATEX = os.environ.get("TFM_PLOT_USE_LATEX", "0").strip().lower() in {"1", "true", "yes", "on"}

FIGSIZE_SINGLE = (7.2, 4.9)
FIGSIZE_WIDE = (7.8, 5.2)
FIGSIZE_METADATA = (7.4, 5.0)
FIGSIZE_HISTOGRAM = (7.8, 5.0)
FIGSIZE_MOSAIC_3X3 = (13.6, 9.4)
FIGSIZE_COMPARISON_MOSAIC_3X3 = (13.9, 9.6)
FIGSIZE_GAUSSIAN_PANEL = (4.35, 3.05)

FONT_SIZE_TITLE = 15
FONT_SIZE_LABEL = 15
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11
FONT_SIZE_LEGEND_TITLE = 11
FONT_SIZE_MOSAIC_TITLE = 12
FONT_SIZE_MOSAIC_LABEL = 12
FONT_SIZE_MOSAIC_TICK = 10
FONT_SIZE_MOSAIC_LEGEND = 8.8

LINEWIDTH_MAIN = 2.2
LINEWIDTH_SECONDARY = 1.8
LINEWIDTH_MOSAIC = 1.45
MARKERSIZE = 5.2
CAPSIZE = 2.5

LABEL_CONCENTRATION_CF4 = r"CF$_4$ concentration [\%]"
LABEL_CONCENTRATION_N2 = r"N$_2$ concentration [\%]"
LABEL_CONCENTRATION = r"Concentration [\%]"
LABEL_YIELD_MEV = r"Yield [ph MeV$^{-1}$]"
LABEL_YIELD_ARB = r"Yield [arb. units]"
LABEL_WAVELENGTH = r"Wavelength [nm]"
LABEL_SPECTRAL_YIELD = r"Spectral yield [ph MeV$^{-1}$ nm$^{-1}$]"
LABEL_RAW_INTENSITY = r"Raw intensity [arb. units]"


@dataclass(frozen=True)
class LegendStyle:
    fontsize: float = FONT_SIZE_LEGEND
    title_fontsize: float = FONT_SIZE_LEGEND_TITLE
    frameon: bool = False
    borderaxespad: float = 0.6
    handlelength: float = 1.8
    handletextpad: float = 0.45
    columnspacing: float = 0.9

    def as_kwargs(self, **overrides: Any) -> dict[str, Any]:
        out = self.__dict__.copy()
        out.update(overrides)
        return out


LEGEND = LegendStyle()
MOSAIC_LEGEND = LegendStyle(fontsize=FONT_SIZE_MOSAIC_LEGEND, title_fontsize=FONT_SIZE_MOSAIC_LEGEND)


def setup_style(*, grid: bool | None = None, use_latex: bool | None = None, context: str = "single") -> None:
    """Apply a consistent paper-like style to Matplotlib figures."""

    grid = PAPER_GRID if grid is None else bool(grid)
    use_latex = PAPER_USE_LATEX if use_latex is None else bool(use_latex)

    try:
        import scienceplots  # noqa: F401  # needed so Matplotlib sees the style

        styles = ["science"]
        if not use_latex:
            styles.append("no-latex")
        plt.style.use(styles)
    except Exception:
        plt.style.use("default")

    is_mosaic = context.lower() in {"mosaic", "spectra", "gaussian_mosaic"}
    label_size = FONT_SIZE_MOSAIC_LABEL if is_mosaic else FONT_SIZE_LABEL
    title_size = FONT_SIZE_MOSAIC_TITLE if is_mosaic else FONT_SIZE_TITLE
    tick_size = FONT_SIZE_MOSAIC_TICK if is_mosaic else FONT_SIZE_TICK
    legend_size = FONT_SIZE_MOSAIC_LEGEND if is_mosaic else FONT_SIZE_LEGEND

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "figure.figsize": FIGSIZE_SINGLE,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": grid,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
            "axes.labelsize": label_size,
            "axes.titlesize": title_size,
            "axes.linewidth": 0.95,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            "legend.frameon": False,
            "lines.linewidth": LINEWIDTH_MAIN,
            "lines.markersize": MARKERSIZE,
            "errorbar.capsize": CAPSIZE,
        }
    )


def cmap_name(cmap: str | None = None) -> str:
    return str(cmap or PAPER_CMAP)


def palette(n: int, *, cmap: str | None = None, start: float = 0.12, stop: float = 0.88) -> np.ndarray:
    """Return an evenly spaced palette from the global colormap."""

    cm = plt.get_cmap(cmap_name(cmap))
    return cm(np.linspace(float(start), float(stop), max(int(n), 2)))


def apply_axis_style(ax, *, legend: bool = False, legend_kwargs: dict[str, Any] | None = None) -> None:
    """Final small cleanups shared by all figures."""

    ax.tick_params(axis="both", which="both", top=True, right=False)
    ax.margins(x=0.015)
    if legend:
        kwargs = LEGEND.as_kwargs()
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def apply_mosaic_axis_style(ax, *, legend: bool = False, legend_kwargs: dict[str, Any] | None = None) -> None:
    ax.tick_params(axis="both", which="both", top=True, right=False, labelsize=FONT_SIZE_MOSAIC_TICK)
    ax.margins(x=0.01)
    if legend:
        kwargs = MOSAIC_LEGEND.as_kwargs()
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def safe_legend(ax, *, legend_kwargs: dict[str, Any] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels, strict=False) if l and not str(l).startswith("_")]
    if not keep:
        return
    handles, labels = zip(*keep, strict=False)
    kwargs = LEGEND.as_kwargs()
    if legend_kwargs:
        kwargs.update(legend_kwargs)
    ax.legend(handles, labels, **kwargs)


def safe_mosaic_legend(ax, *, legend_kwargs: dict[str, Any] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels, strict=False) if l and not str(l).startswith("_")]
    if not keep:
        return
    handles, labels = zip(*keep, strict=False)
    kwargs = MOSAIC_LEGEND.as_kwargs()
    if legend_kwargs:
        kwargs.update(legend_kwargs)
    ax.legend(handles, labels, **kwargs)
