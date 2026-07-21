from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_DEFAULTS: dict[str, Any] = {
    "font_family": "serif",
    "mathtext_fontset": "dejavuserif",
    "cmap": "viridis",
    "grid": False,
    "use_latex": False,
    "figure_single": [7.2, 4.9],
    "figure_wide": [7.8, 5.2],
    "font_size_title": 15.0,
    "font_size_label": 15.0,
    "font_size_tick": 12.0,
    "font_size_legend": 11.0,
    "font_size_legend_title": 11.0,
    "line_width_main": 2.2,
    "line_width_secondary": 1.8,
    "marker_size": 5.2,
    "cap_size": 2.5,
    "legend_frame": True,
    "legend_fancybox": True,
    "legend_alpha": 0.92,
    "legend_columns": 1,
    "axes_line_width": 1.0,
    "tick_major_width": 1.2,
    "tick_minor_width": 0.8,
    "tick_major_length": 5.0,
    "tick_minor_length": 3.0,
    "tick_direction": "in",
    "ticks_top": True,
    "ticks_right": False,
    "band_alpha": 0.20,
    "errorbar_line_width": 1.2,
    "cap_thickness": 1.2,
    "grid_alpha": 0.22,
    "grid_line_width": 0.55,
}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _preset_file() -> Path | None:
    explicit = os.environ.get("SCINTILLATION_STYLE_FILE")
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.exists() else None
    root_text = os.environ.get("SCINTILLATION_PROJECT_ROOT")
    if root_text:
        root = Path(root_text).expanduser().resolve()
    else:
        candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
        root = next((candidate for candidate in candidates if (candidate / "config" / "styles").exists()), Path.cwd().resolve())
    styles = root / "config" / "styles"
    preset = os.environ.get("SCINTILLATION_STYLE_PRESET", "").strip()
    if not preset:
        active = styles / "active.txt"
        if active.exists():
            preset = active.read_text(encoding="utf-8").strip()
    preset = preset or "tfm"
    candidate = styles / f"{preset}.json"
    return candidate if candidate.exists() else None


def _load_style() -> dict[str, Any]:
    style = dict(_DEFAULTS)
    path = _preset_file()
    if path is not None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                style.update(loaded)
        except Exception:
            pass
    # Backward-compatible environment overrides.
    style["cmap"] = os.environ.get("TFM_PLOT_CMAP", str(style["cmap"]))
    if "TFM_PLOT_GRID" in os.environ:
        style["grid"] = _truthy(os.environ["TFM_PLOT_GRID"])
    if "TFM_PLOT_USE_LATEX" in os.environ:
        style["use_latex"] = _truthy(os.environ["TFM_PLOT_USE_LATEX"])
    return style


_STYLE = _load_style()
PAPER_CMAP = str(_STYLE["cmap"])
PAPER_GRID = bool(_STYLE["grid"])
PAPER_USE_LATEX = bool(_STYLE["use_latex"])
FONT_FAMILY = str(_STYLE["font_family"])
MATHTEXT_FONTSET = str(_STYLE["mathtext_fontset"])
LEGEND_FRAME = bool(_STYLE["legend_frame"])
LEGEND_FANCYBOX = bool(_STYLE["legend_fancybox"])
LEGEND_ALPHA = float(_STYLE["legend_alpha"])
LEGEND_COLUMNS = max(1, int(_STYLE.get("legend_columns", 1)))

FIGSIZE_SINGLE = tuple(float(v) for v in _STYLE["figure_single"])
FIGSIZE_WIDE = tuple(float(v) for v in _STYLE["figure_wide"])
FIGSIZE_METADATA = FIGSIZE_SINGLE
FIGSIZE_HISTOGRAM = FIGSIZE_WIDE
FIGSIZE_MOSAIC_3X3 = (13.6, 9.4)
FIGSIZE_COMPARISON_MOSAIC_3X3 = (13.9, 9.6)
FIGSIZE_GAUSSIAN_PANEL = (4.35, 3.05)

FONT_SIZE_TITLE = float(_STYLE["font_size_title"])
FONT_SIZE_LABEL = float(_STYLE["font_size_label"])
FONT_SIZE_TICK = float(_STYLE["font_size_tick"])
FONT_SIZE_LEGEND = float(_STYLE["font_size_legend"])
FONT_SIZE_LEGEND_TITLE = float(_STYLE["font_size_legend_title"])
FONT_SIZE_MOSAIC_TITLE = max(8.0, 0.80 * FONT_SIZE_TITLE)
FONT_SIZE_MOSAIC_LABEL = max(8.0, 0.80 * FONT_SIZE_LABEL)
FONT_SIZE_MOSAIC_TICK = max(7.0, 0.83 * FONT_SIZE_TICK)
FONT_SIZE_MOSAIC_LEGEND = max(6.5, 0.80 * FONT_SIZE_LEGEND)
LINEWIDTH_MAIN = float(_STYLE["line_width_main"])
LINEWIDTH_SECONDARY = float(_STYLE["line_width_secondary"])
LINEWIDTH_MOSAIC = max(0.8, 0.66 * LINEWIDTH_MAIN)
MARKERSIZE = float(_STYLE["marker_size"])
CAPSIZE = float(_STYLE["cap_size"])
AXES_LINEWIDTH = float(_STYLE.get("axes_line_width", 1.0))
TICK_MAJOR_WIDTH = float(_STYLE.get("tick_major_width", 1.2))
TICK_MINOR_WIDTH = float(_STYLE.get("tick_minor_width", 0.8))
TICK_MAJOR_LENGTH = float(_STYLE.get("tick_major_length", 5.0))
TICK_MINOR_LENGTH = float(_STYLE.get("tick_minor_length", 3.0))
TICK_DIRECTION = str(_STYLE.get("tick_direction", "in"))
TICKS_TOP = bool(_STYLE.get("ticks_top", True))
TICKS_RIGHT = bool(_STYLE.get("ticks_right", False))
BAND_ALPHA = float(_STYLE.get("band_alpha", 0.20))
ERRORBAR_LINEWIDTH = float(_STYLE.get("errorbar_line_width", 1.2))
CAP_THICKNESS = float(_STYLE.get("cap_thickness", 1.2))
GRID_ALPHA = float(_STYLE.get("grid_alpha", 0.22))
GRID_LINEWIDTH = float(_STYLE.get("grid_line_width", 0.55))

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
    frameon: bool = LEGEND_FRAME
    fancybox: bool = LEGEND_FANCYBOX
    framealpha: float = LEGEND_ALPHA
    facecolor: str = "white"
    edgecolor: str = "0.45"
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

GEOMETRY_MARKERS = {"GEM": "o", "THGEM": "s", "UNIFORM": "D", "UNSPECIFIED": "^"}
PARTICLE_LINESTYLES = {"xray": "-", "electron": "--", "alpha": ":", "model": "-"}
SOURCE_FILLSTYLES = {"experiment": "none", "simulation": "full", "interpolated": "none", "extrapolated": "none"}


def setup_style(*, grid: bool | None = None, use_latex: bool | None = None, context: str = "single") -> None:
    grid = PAPER_GRID if grid is None else bool(grid)
    use_latex = PAPER_USE_LATEX if use_latex is None else bool(use_latex)
    try:
        import scienceplots  # noqa: F401
        styles = ["science"] + ([] if use_latex else ["no-latex"])
        plt.style.use(styles)
    except Exception:
        plt.style.use("default")
    is_mosaic = context.lower() in {"mosaic", "spectra", "gaussian_mosaic"}
    label = FONT_SIZE_MOSAIC_LABEL if is_mosaic else FONT_SIZE_LABEL
    title = FONT_SIZE_MOSAIC_TITLE if is_mosaic else FONT_SIZE_TITLE
    tick = FONT_SIZE_MOSAIC_TICK if is_mosaic else FONT_SIZE_TICK
    legend = FONT_SIZE_MOSAIC_LEGEND if is_mosaic else FONT_SIZE_LEGEND
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": MATHTEXT_FONTSET,
        "text.usetex": use_latex,
        "figure.figsize": FIGSIZE_SINGLE,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": grid,
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": GRID_LINEWIDTH,
        "axes.labelsize": label,
        "axes.titlesize": title,
        "axes.linewidth": AXES_LINEWIDTH,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "xtick.direction": TICK_DIRECTION,
        "ytick.direction": TICK_DIRECTION,
        "xtick.top": TICKS_TOP,
        "ytick.right": TICKS_RIGHT,
        "xtick.major.size": TICK_MAJOR_LENGTH,
        "ytick.major.size": TICK_MAJOR_LENGTH,
        "xtick.minor.size": TICK_MINOR_LENGTH,
        "ytick.minor.size": TICK_MINOR_LENGTH,
        "xtick.major.width": TICK_MAJOR_WIDTH,
        "ytick.major.width": TICK_MAJOR_WIDTH,
        "xtick.minor.width": TICK_MINOR_WIDTH,
        "ytick.minor.width": TICK_MINOR_WIDTH,
        "legend.fontsize": legend,
        "legend.title_fontsize": legend,
        "legend.frameon": LEGEND_FRAME,
        "legend.fancybox": LEGEND_FANCYBOX,
        "legend.framealpha": LEGEND_ALPHA,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.45",
        "lines.linewidth": LINEWIDTH_MAIN,
        "lines.markersize": MARKERSIZE,
        "errorbar.capsize": CAPSIZE,
    })


def cmap_name(cmap: str | None = None) -> str:
    return str(cmap or PAPER_CMAP)


def palette(n: int, *, cmap: str | None = None, start: float = 0.12, stop: float = 0.88) -> np.ndarray:
    return plt.get_cmap(cmap_name(cmap))(np.linspace(float(start), float(stop), max(int(n), 2)))


def marker_for_geometry(geometry: str) -> str:
    return GEOMETRY_MARKERS.get(str(geometry).upper(), "^")


def linestyle_for_particle(particle: str) -> str:
    return PARTICLE_LINESTYLES.get(str(particle).lower(), "-")


def apply_axis_style(ax, *, legend: bool = False, legend_kwargs: dict[str, Any] | None = None) -> None:
    ax.tick_params(axis="both", which="major", top=TICKS_TOP, right=TICKS_RIGHT,
                   direction=TICK_DIRECTION, width=TICK_MAJOR_WIDTH, length=TICK_MAJOR_LENGTH)
    ax.tick_params(axis="both", which="minor", top=TICKS_TOP, right=TICKS_RIGHT,
                   direction=TICK_DIRECTION, width=TICK_MINOR_WIDTH, length=TICK_MINOR_LENGTH)
    ax.margins(x=0.015)
    if legend:
        kwargs = LEGEND.as_kwargs(ncol=LEGEND_COLUMNS)
        kwargs.update(legend_kwargs or {})
        ax.legend(**kwargs)


def apply_mosaic_axis_style(ax, *, legend: bool = False, legend_kwargs: dict[str, Any] | None = None) -> None:
    ax.tick_params(axis="both", which="major", top=TICKS_TOP, right=TICKS_RIGHT,
                   direction=TICK_DIRECTION, width=TICK_MAJOR_WIDTH, length=TICK_MAJOR_LENGTH,
                   labelsize=FONT_SIZE_MOSAIC_TICK)
    ax.tick_params(axis="both", which="minor", top=TICKS_TOP, right=TICKS_RIGHT,
                   direction=TICK_DIRECTION, width=TICK_MINOR_WIDTH, length=TICK_MINOR_LENGTH)
    ax.margins(x=0.01)
    if legend:
        kwargs = MOSAIC_LEGEND.as_kwargs(ncol=LEGEND_COLUMNS)
        kwargs.update(legend_kwargs or {})
        ax.legend(**kwargs)


def safe_legend(ax, *, legend_kwargs: dict[str, Any] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, label) for h, label in zip(handles, labels, strict=False) if label and not str(label).startswith("_")]
    if not keep:
        return
    kept_handles, kept_labels = zip(*keep, strict=False)
    kwargs = LEGEND.as_kwargs(ncol=LEGEND_COLUMNS)
    kwargs.update(legend_kwargs or {})
    ax.legend(kept_handles, kept_labels, **kwargs)


def safe_mosaic_legend(ax, *, legend_kwargs: dict[str, Any] | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, label) for h, label in zip(handles, labels, strict=False) if label and not str(label).startswith("_")]
    if not keep:
        return
    kept_handles, kept_labels = zip(*keep, strict=False)
    kwargs = MOSAIC_LEGEND.as_kwargs(ncol=LEGEND_COLUMNS)
    kwargs.update(legend_kwargs or {})
    ax.legend(kept_handles, kept_labels, **kwargs)


def boxed_legend_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = LEGEND.as_kwargs(ncol=LEGEND_COLUMNS)
    kwargs.update(overrides)
    return kwargs
