from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_STYLE: dict[str, Any] = {
    "label": "TFM",
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

NUMERIC_FIELDS = {
    "font_size_title", "font_size_label", "font_size_tick",
    "font_size_legend", "font_size_legend_title", "line_width_main",
    "line_width_secondary", "marker_size", "cap_size", "legend_alpha",
    "axes_line_width", "tick_major_width", "tick_minor_width",
    "tick_major_length", "tick_minor_length", "band_alpha",
    "errorbar_line_width", "cap_thickness", "grid_alpha", "grid_line_width",
}


def styles_dir(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "config" / "styles"


def active_style_path(project_root: str | Path) -> Path:
    return styles_dir(project_root) / "active.txt"


def available_styles(project_root: str | Path) -> dict[str, Path]:
    directory = styles_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    return {path.stem: path for path in sorted(directory.glob("*.json"))}


def active_style_name(project_root: str | Path) -> str:
    path = active_style_path(project_root)
    if path.exists():
        name = path.read_text(encoding="utf-8").strip()
        if name in available_styles(project_root):
            return name
    return "tfm"


def set_active_style(project_root: str | Path, name: str) -> Path:
    styles = available_styles(project_root)
    if name not in styles:
        raise KeyError(f"Unknown style preset: {name}")
    path = active_style_path(project_root)
    path.write_text(f"{name}\n", encoding="utf-8")
    return path


def _validated_style(raw: Mapping[str, Any]) -> dict[str, Any]:
    style = deepcopy(DEFAULT_STYLE)
    style.update(dict(raw))
    for field in NUMERIC_FIELDS:
        style[field] = float(style[field])
    for field in ("figure_single", "figure_wide"):
        value = style[field]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{field} must contain [width, height]")
        style[field] = [float(value[0]), float(value[1])]
        if any(component <= 0 for component in style[field]):
            raise ValueError(f"{field} values must be positive")
    style["legend_columns"] = max(1, int(style["legend_columns"]))
    style["grid"] = bool(style["grid"])
    style["use_latex"] = bool(style["use_latex"])
    style["legend_frame"] = bool(style["legend_frame"])
    style["legend_fancybox"] = bool(style["legend_fancybox"])
    style["ticks_top"] = bool(style["ticks_top"])
    style["ticks_right"] = bool(style["ticks_right"])
    if style["tick_direction"] not in {"in", "out", "inout"}:
        raise ValueError("tick_direction must be in, out or inout")
    style["legend_alpha"] = min(1.0, max(0.0, float(style["legend_alpha"])))
    style["label"] = str(style["label"])
    style["font_family"] = str(style["font_family"])
    style["mathtext_fontset"] = str(style["mathtext_fontset"])
    style["cmap"] = str(style["cmap"])
    return style


def load_style(project_root: str | Path, name_or_path: str | Path | None = None) -> dict[str, Any]:
    if name_or_path is None:
        name_or_path = active_style_name(project_root)
    candidate = Path(name_or_path)
    if not candidate.is_absolute() and candidate.suffix.lower() != ".json":
        candidate = styles_dir(project_root) / f"{candidate}.json"
    elif not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    if not candidate.exists():
        return deepcopy(DEFAULT_STYLE)
    with candidate.open("r", encoding="utf-8") as handle:
        return _validated_style(json.load(handle))


def save_style(project_root: str | Path, name: str, values: Mapping[str, Any]) -> Path:
    safe_name = "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch in {"_", "-"})
    if not safe_name:
        raise ValueError("Style name cannot be empty")
    path = styles_dir(project_root) / f"{safe_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = _validated_style(values)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def style_environment(project_root: str | Path, name: str | None = None) -> dict[str, str]:
    selected = name or active_style_name(project_root)
    path = available_styles(project_root).get(selected)
    if path is None:
        selected = "tfm"
        path = available_styles(project_root).get(selected)
    environment = {
        "SCINTILLATION_PROJECT_ROOT": str(Path(project_root).resolve()),
        "SCINTILLATION_STYLE_PRESET": selected,
    }
    if path is not None:
        environment["SCINTILLATION_STYLE_FILE"] = str(path.resolve())
    return environment


def matplotlib_rc(style: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "font.family": style["font_family"],
        "mathtext.fontset": style["mathtext_fontset"],
        "text.usetex": bool(style["use_latex"]),
        "figure.figsize": tuple(style["figure_single"]),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": bool(style["grid"]),
        "grid.alpha": style["grid_alpha"],
        "grid.linewidth": style["grid_line_width"],
        "axes.labelsize": style["font_size_label"],
        "axes.titlesize": style["font_size_title"],
        "axes.linewidth": style["axes_line_width"],
        "xtick.labelsize": style["font_size_tick"],
        "ytick.labelsize": style["font_size_tick"],
        "xtick.direction": style["tick_direction"],
        "ytick.direction": style["tick_direction"],
        "xtick.top": bool(style["ticks_top"]),
        "ytick.right": bool(style["ticks_right"]),
        "xtick.major.size": style["tick_major_length"],
        "ytick.major.size": style["tick_major_length"],
        "xtick.minor.size": style["tick_minor_length"],
        "ytick.minor.size": style["tick_minor_length"],
        "xtick.major.width": style["tick_major_width"],
        "ytick.major.width": style["tick_major_width"],
        "xtick.minor.width": style["tick_minor_width"],
        "ytick.minor.width": style["tick_minor_width"],
        "legend.fontsize": style["font_size_legend"],
        "legend.title_fontsize": style["font_size_legend_title"],
        "legend.frameon": bool(style["legend_frame"]),
        "legend.fancybox": bool(style["legend_fancybox"]),
        "legend.framealpha": style["legend_alpha"],
        "lines.linewidth": style["line_width_main"],
        "lines.markersize": style["marker_size"],
        "errorbar.capsize": style["cap_size"],
    }


@contextmanager
def temporary_style(style: Mapping[str, Any]):
    """Apply one validated GUI style only inside a preview/render context."""
    import matplotlib.pyplot as plt

    validated = _validated_style(style)
    with plt.rc_context(matplotlib_rc(validated)):
        try:
            import scienceplots  # noqa: F401
            base = ["science"] + ([] if validated.get("use_latex") else ["no-latex"])
            with plt.style.context(base):
                plt.rcParams.update(matplotlib_rc(validated))
                yield
        except Exception:
            yield
