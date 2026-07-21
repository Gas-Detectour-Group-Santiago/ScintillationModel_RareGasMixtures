from __future__ import annotations

"""Declarative plot-recipe readers used by the legacy-compatible engines.

The physics remains in the model/fit modules.  These readers only describe
which data/model components are drawn and how the final figure is laid out.
Every production figure is therefore reproducible from ``config/plots/*.csv``.
"""

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "on", "active", "enabled"}
FALSE_VALUES = {"0", "false", "no", "off", "disabled", "inactive"}


def repository_root(project_root: str | Path | None = None) -> Path:
    explicit = os.environ.get("SCINTILLATION_ROOT") or os.environ.get("SCINTILLATION_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    start = Path(project_root or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "config" / "project.toml").is_file():
            if candidate.name == ".runtime" and (candidate.parent / "config" / "project.toml").is_file():
                return candidate.parent
            return candidate
    raise FileNotFoundError(f"Cannot locate repository root from {project_root!r}")


def config_path(family: str, project_root: str | Path | None = None) -> Path:
    return repository_root(project_root) / "config" / "plots" / f"{family}.csv"


def read_plot_rows(family: str, project_root: str | Path | None = None) -> pd.DataFrame:
    path = config_path(family, project_root)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, keep_default_na=False)
    if "enabled" in frame.columns:
        frame = frame.loc[frame["enabled"].map(as_bool)]
    return frame.reset_index(drop=True)


def as_bool(value: object, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def as_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return default if text.lower() == "nan" else text


def as_optional_text(value: object) -> str | None:
    text = as_text(value).strip()
    return text or None


def as_float(value: object, default: float | None = None) -> float | None:
    text = as_text(value).strip()
    if not text:
        return default
    return float(text)


def as_int(value: object, default: int | None = None) -> int | None:
    value_float = as_float(value, None)
    return default if value_float is None else int(value_float)


def split_values(value: object, *, cast: Callable[[str], Any] = str, sep: str = "|") -> tuple[Any, ...]:
    text = as_text(value).strip()
    if not text:
        return tuple()
    return tuple(cast(item.strip()) for item in text.split(sep) if item.strip())


def as_limits(row: Mapping[str, object], prefix: str) -> tuple[float, float] | None:
    low = as_float(row.get(f"{prefix}min"), None)
    high = as_float(row.get(f"{prefix}max"), None)
    return None if low is None or high is None else (low, high)


def grid_from_row(row: Mapping[str, object]) -> np.ndarray:
    kind = as_text(row.get("grid_scale"), "log").strip().lower()
    start = as_float(row.get("grid_min"), None)
    stop = as_float(row.get("grid_max"), None)
    points = as_int(row.get("grid_points"), 700)
    values = split_values(row.get("grid_values"), cast=float)
    if values:
        return np.asarray(values, dtype=float)
    if start is None or stop is None:
        raise ValueError(f"Recipe {row.get('plot_id', '')!r} requires grid_min/grid_max")
    if kind in {"log", "logspace"}:
        if start <= 0 or stop <= 0:
            raise ValueError("Logarithmic grids require positive bounds")
        return np.logspace(np.log10(start), np.log10(stop), int(points or 700))
    if kind in {"linear", "linspace"}:
        return np.linspace(start, stop, int(points or 700))
    raise ValueError(f"Unsupported grid_scale {kind!r}")


def output_path(row: Mapping[str, object], legacy_project_root: str | Path) -> Path:
    value = as_optional_text(row.get("output"))
    if not value:
        raise ValueError(f"Recipe {row.get('plot_id', '')!r} has no output")
    path = Path(value)
    return path if path.is_absolute() else Path(legacy_project_root) / path


def _normalization(normalization_id: str, lookup: Callable[[str], Any]) -> Any:
    return lookup(normalization_id or "own")


def load_fit_plot_specs(
    legacy_project_root: str | Path,
    fit_id: str,
    *,
    plot_spec_cls: type,
    fallback: Sequence[Any] = (),
) -> tuple[Any, ...]:
    frame = read_plot_rows("fits", legacy_project_root)
    if frame.empty:
        return tuple(fallback)
    rows = frame.loc[(frame["fit_id"] == fit_id) & (frame["plot_type"] == "fit")]
    if rows.empty:
        return tuple()
    out = []
    for _, row in rows.iterrows():
        legend_kwargs: dict[str, Any] = {}
        ncol = as_int(row.get("legend_ncol"), None)
        loc = as_optional_text(row.get("legend_loc"))
        if ncol is not None:
            legend_kwargs["ncol"] = ncol
        if loc:
            legend_kwargs["loc"] = loc
        line_labels = split_values(row.get("line_labels"), cast=str)
        out.append(
            plot_spec_cls(
                name=as_text(row.get("plot_id")),
                dataset_key=as_text(row.get("dataset_id")),
                theory_key=as_text(row.get("component")),
                pressures=split_values(row.get("pressures_bar"), cast=float),
                concentration_grid=grid_from_row(row),
                title=as_text(row.get("title")),
                xlabel=as_text(row.get("xlabel")),
                ylabel=as_text(row.get("ylabel")),
                output=output_path(row, legacy_project_root),
                x_col=as_text(row.get("x_column")),
                x_plot_factor=float(as_float(row.get("x_plot_factor"), 100.0)),
                min_positive_x=as_float(row.get("min_positive_x"), None),
                xlim=as_limits(row, "x"),
                ylim=as_limits(row, "y"),
                xscale=as_text(row.get("xscale"), "log"),
                yscale=as_text(row.get("yscale"), "log"),
                cmap=as_optional_text(row.get("cmap")) or "viridis",
                darken_factor=float(as_float(row.get("darken_factor"), -0.15)),
                legend_kwargs=legend_kwargs or None,
                label_mode=as_text(row.get("label_mode"), "legend"),
                activate_components=as_bool(row.get("show_components"), False),
                line_label_fmt=line_labels or None,
                show_secondary_yaxis=as_bool(row.get("show_secondary_yaxis"), False),
                show_only_fit_points=as_bool(row.get("show_only_fit_points"), True),
            )
        )
    return tuple(out)


def fit_correlation_enabled(legacy_project_root: str | Path, fit_id: str, default: bool = True) -> bool:
    frame = read_plot_rows("fits", legacy_project_root)
    if frame.empty:
        return default
    rows = frame.loc[(frame["fit_id"] == fit_id) & (frame["plot_type"] == "correlation")]
    return not rows.empty


def primary_rows(legacy_project_root: str | Path, group: str) -> pd.DataFrame:
    frame = read_plot_rows("primary", legacy_project_root)
    if frame.empty:
        return frame
    return frame.loc[frame["group"] == group].copy()


def load_primary_band_plots(
    legacy_project_root: str | Path,
    group: str,
    *,
    band_plot_cls: type,
    normalization_lookup: Callable[[str], Any],
) -> list[Any]:
    rows = primary_rows(legacy_project_root, group)
    rows = rows.loc[rows["plot_type"] == "band"] if not rows.empty else rows
    out: list[Any] = []
    for _, row in rows.iterrows():
        band_mode = as_text(row.get("bands"), "total")
        out.append(
            band_plot_cls(
                id=as_text(row.get("plot_id")),
                title=as_text(row.get("title")),
                fit_name=as_text(row.get("model_id")),
                component=as_text(row.get("component"), "total"),
                pressure=float(as_float(row.get("pressure_bar"), 1.0)),
                x_grid=grid_from_row(row),
                normalization=_normalization(as_text(row.get("normalization"), "own"), normalization_lookup),
                xlabel=as_text(row.get("xlabel")),
                ylabel=as_text(row.get("ylabel")),
                x_plot_factor=float(as_float(row.get("x_plot_factor"), 100.0)),
                xscale=as_text(row.get("xscale"), "log"),
                yscale=as_text(row.get("yscale"), "log"),
                xlim=as_limits(row, "x"),
                ylim=as_limits(row, "y"),
                output=output_path(row, legacy_project_root),
                show_stat=band_mode in {"stat", "stat_syst", "total", "all"},
                show_syst=band_mode in {"syst", "stat_syst", "total", "all"},
                show_total=band_mode in {"total", "all"},
                overlays=split_values(row.get("datasets"), cast=str),
                scale_xray_with_normalization=as_bool(row.get("scale_xray_with_normalization"), True),
            )
        )
    return out


def load_primary_multiband_plots(
    legacy_project_root: str | Path,
    group: str,
    *,
    curve_cls: type,
    plot_cls: type,
    normalization_lookup: Callable[[str], Any],
) -> list[Any]:
    rows = primary_rows(legacy_project_root, group)
    rows = rows.loc[rows["plot_type"] == "multiband"] if not rows.empty else rows
    plots: list[Any] = []
    for plot_id, group_rows in rows.groupby("plot_id", sort=False):
        first = group_rows.iloc[0]
        curves = []
        for _, row in group_rows.iterrows():
            bands = as_text(row.get("bands"), "total")
            curves.append(
                curve_cls(
                    id=as_text(row.get("series_id")),
                    label=as_text(row.get("label")),
                    fit_name=as_text(row.get("model_id")),
                    component=as_text(row.get("component"), "total"),
                    pressure=float(as_float(row.get("pressure_bar"), 1.0)),
                    x_grid=grid_from_row(row),
                    normalization=_normalization(as_text(row.get("normalization"), "own"), normalization_lookup),
                    x_plot_factor=float(as_float(row.get("x_plot_factor"), 100.0)),
                    show_stat=bands in {"stat", "stat_syst", "total", "all"},
                    show_syst=bands in {"syst", "stat_syst", "total", "all"},
                    show_total=bands in {"total", "all"},
                    paper_overlay_id=as_optional_text(row.get("dataset_group")),
                    scale_xray_with_normalization=as_bool(row.get("scale_xray_with_normalization"), True),
                )
            )
        plots.append(
            plot_cls(
                id=str(plot_id),
                title=as_text(first.get("title")),
                curves=tuple(curves),
                xlabel=as_text(first.get("xlabel")),
                ylabel=as_text(first.get("ylabel")),
                xscale=as_text(first.get("xscale"), "log"),
                yscale=as_text(first.get("yscale"), "log"),
                xlim=as_limits(first, "x"),
                ylim=as_limits(first, "y"),
                output=output_path(first, legacy_project_root),
                legend_loc=as_text(first.get("legend_loc"), "best"),
                legend_ncol=int(as_int(first.get("legend_ncol"), 2) or 2),
                legend_fontsize=as_float(first.get("legend_fontsize"), None),
            )
        )
    return plots


def load_secondary_selections(
    legacy_project_root: str | Path,
    *,
    selection_cls: type,
) -> dict[str, Any]:
    path = repository_root(legacy_project_root) / "config" / "secondary_selections.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    out: dict[str, Any] = {}
    for _, row in frame.iterrows():
        if "enabled" in frame.columns and not as_bool(row.get("enabled"), True):
            continue
        reference = as_optional_text(row.get("reference_dir"))
        extra_masks: dict[str, Any] = {}
        gas_values = split_values(row.get("gas_mixture_in"), cast=str)
        if gas_values:
            extra_masks["gas_mixture"] = {"in": gas_values}
        out[as_text(row.get("selection_id"))] = selection_cls(
            id=as_text(row.get("selection_id")),
            gas=as_text(row.get("gas")),
            reference_dir=None if reference is None else Path(legacy_project_root) / reference,
            population_csv=None,
            population_filename=as_text(row.get("population_filename"), "ArCF4_secondary.csv"),
            pressure=as_float(row.get("pressure_bar"), None),
            pressure_atol=float(as_float(row.get("pressure_atol"), 0.026)),
            pressure_min=as_float(row.get("pressure_min"), None),
            pressure_max=as_float(row.get("pressure_max"), None),
            gap_mm=as_float(row.get("gap_mm"), None),
            gap_atol=float(as_float(row.get("gap_atol"), 1e-6)),
            gap_min=as_float(row.get("gap_min"), None),
            gap_max=as_float(row.get("gap_max"), None),
            electric_field=as_float(row.get("field_kV_cm"), None),
            electric_field_atol=float(as_float(row.get("field_atol"), 1e-8)),
            electric_field_min=as_float(row.get("field_min"), None),
            electric_field_max=as_float(row.get("field_max"), None),
            concentration=as_float(row.get("concentration_percent"), None),
            concentration_atol=float(as_float(row.get("concentration_atol"), 1e-8)),
            concentration_min=as_float(row.get("concentration_min"), None),
            concentration_max=as_float(row.get("concentration_max"), None),
            npe=as_float(row.get("npe"), None),
            npe_atol=float(as_float(row.get("npe_atol"), 1e-8)),
            npe_min=as_float(row.get("npe_min"), None),
            npe_max=as_float(row.get("npe_max"), None),
            npe_column=as_text(row.get("npe_column"), "npe"),
            ne=as_float(row.get("ne"), None),
            ne_atol=float(as_float(row.get("ne_atol"), 1e-8)),
            ne_min=as_float(row.get("ne_min"), None),
            ne_max=as_float(row.get("ne_max"), None),
            ni=as_float(row.get("ni"), None),
            ni_atol=float(as_float(row.get("ni_atol"), 1e-8)),
            ni_min=as_float(row.get("ni_min"), None),
            ni_max=as_float(row.get("ni_max"), None),
            gain=as_float(row.get("gain"), None),
            gain_atol=float(as_float(row.get("gain_atol"), 1e-8)),
            gain_min=as_float(row.get("gain_min"), None),
            gain_max=as_float(row.get("gain_max"), None),
            gain_column=as_text(row.get("gain_column"), "gain"),
            normalize_by=as_text(row.get("normalize_by"), "ne"),
            masks={},
            extra_masks=extra_masks,
        )
    return out


def _secondary_curve(
    row: Mapping[str, object],
    *,
    legacy_project_root: str | Path,
    curve_cls: type,
    normalization_lookup: Callable[[str], Any],
    selections: Mapping[str, Any],
    ocw_lookup: Callable[[str], Any | None],
) -> Any:
    selection_id = as_text(row.get("selection_id"))
    if selection_id not in selections:
        raise KeyError(f"Unknown secondary selection {selection_id!r}")
    bands = as_text(row.get("bands"), "total")
    return curve_cls(
        id=as_text(row.get("series_id")),
        label=as_text(row.get("label")),
        fit_name=as_text(row.get("model_id")),
        component=as_text(row.get("component"), "total"),
        pressure=as_float(row.get("pressure_bar"), None),
        x_grid=grid_from_row(row),
        normalization=_normalization(as_text(row.get("normalization"), "secondary_ArCF4_ne"), normalization_lookup),
        selection=selections[selection_id],
        x_plot_factor=float(as_float(row.get("x_plot_factor"), 100.0)),
        x_axis=as_text(row.get("x"), "concentration"),
        show_stat=bands in {"stat", "stat_syst", "total", "all"},
        show_syst=bands in {"syst", "stat_syst", "total", "all"},
        show_total=bands in {"total", "all", "ocw", "stat_syst_ocw"},
        band_mode=as_text(row.get("band_mode"), "sys_stat"),
        ocw_config=ocw_lookup(as_text(row.get("ocw_id"))),
        paper_overlay_id=as_optional_text(row.get("dataset_group")),
        color_group=as_optional_text(row.get("color_group")),
        linestyle=as_text(row.get("linestyle"), "-"),
        linewidth=float(as_float(row.get("linewidth"), 2.0)),
    )


def _combined_component_specs(value: object) -> list[dict[str, str]]:
    """Parse ``model:component:band_mode:ocw`` entries separated by ``|``."""
    specs = []
    for item in split_values(value, cast=str):
        parts = item.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid combined component {item!r}")
        specs.append({
            "model_id": parts[0],
            "component": parts[1],
            "band_mode": parts[2] if len(parts) > 2 else "sys_stat",
            "ocw_id": parts[3] if len(parts) > 3 else "",
        })
    return specs


def load_experimental_dataset_registry(project_root: str | Path | None = None) -> pd.DataFrame:
    path = repository_root(project_root) / "config" / "experimental_datasets.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def experimental_series_from_id(
    dataset_id: str,
    *,
    legacy_project_root: str | Path,
    experimental_series_cls: type,
    color_from_curve_id: str | None = None,
) -> Any:
    registry = load_experimental_dataset_registry(legacy_project_root)
    rows = registry.loc[(registry["dataset_id"] == dataset_id) & registry["enabled"].map(as_bool)]
    if rows.empty:
        raise KeyError(f"Unknown experimental dataset {dataset_id!r}")
    spec = rows.iloc[0]
    path = repository_root(legacy_project_root) / as_text(spec.get("file"))
    data = pd.read_csv(path)
    x_col = as_text(spec.get("x_column"), "x")
    y_col = as_text(spec.get("y_column"), "y")
    err_col = as_optional_text(spec.get("yerr_column"))
    return experimental_series_cls(
        x=data[x_col].to_numpy(dtype=float),
        y=data[y_col].to_numpy(dtype=float),
        yerr=None if err_col is None else data[err_col].to_numpy(dtype=float),
        label=as_text(spec.get("label")),
        marker=as_text(spec.get("marker"), "o"),
        linestyle=as_text(spec.get("linestyle"), "none"),
        color=as_optional_text(spec.get("color")),
        color_from_curve_id=color_from_curve_id or as_optional_text(spec.get("color_from_series_id")),
        markerfacecolor="white" if as_text(spec.get("marker_fill"), "filled") == "open" else as_optional_text(spec.get("marker_facecolor")),
        markeredgecolor=as_optional_text(spec.get("marker_edgecolor")),
        markeredgewidth=float(as_float(spec.get("marker_edgewidth"), 1.2)),
        markersize=float(as_float(spec.get("markersize"), 6.0)),
        capsize=float(as_float(spec.get("capsize"), 2.5)),
        alpha=float(as_float(spec.get("alpha"), 1.0)),
        scale_group=as_optional_text(spec.get("scale_group")),
        scale_anchor=as_bool(spec.get("scale_anchor"), False),
        scale_anchor_x=as_float(spec.get("scale_anchor_x"), None),
        scale_anchor_curve_id=as_optional_text(spec.get("scale_anchor_curve_id")),
        scale_model_column=as_text(spec.get("scale_model_column"), "auto"),
    )


def load_secondary_multiband_plots(
    legacy_project_root: str | Path,
    group: str,
    *,
    curve_cls: type,
    combined_cls: type,
    plot_cls: type,
    selection_cls: type,
    experimental_series_cls: type,
    normalization_lookup: Callable[[str], Any],
    ocw_lookup: Callable[[str], Any | None],
) -> list[Any]:
    frame = read_plot_rows("secondary", legacy_project_root)
    if frame.empty:
        return []
    rows = frame.loc[(frame["group"] == group) & (frame["plot_type"] == "multiband")]
    selections = load_secondary_selections(legacy_project_root, selection_cls=selection_cls)
    plots: list[Any] = []
    for plot_id, plot_rows in rows.groupby("plot_id", sort=False):
        first = plot_rows.iloc[0]
        curves = []
        experimental = []
        for _, row in plot_rows.iterrows():
            kind = as_text(row.get("kind"), "model")
            if kind == "experimental":
                experimental.append(
                    experimental_series_from_id(
                        as_text(row.get("dataset_id")),
                        legacy_project_root=legacy_project_root,
                        experimental_series_cls=experimental_series_cls,
                        color_from_curve_id=as_optional_text(row.get("color_from_series_id")),
                    )
                )
                continue
            if kind == "combined":
                children = []
                for index, child in enumerate(_combined_component_specs(row.get("components"))):
                    child_row = dict(row)
                    child_row.update(child)
                    child_row["series_id"] = f"{as_text(row.get('series_id'))}__component_{index}"
                    children.append(
                        _secondary_curve(
                            child_row,
                            legacy_project_root=legacy_project_root,
                            curve_cls=curve_cls,
                            normalization_lookup=normalization_lookup,
                            selections=selections,
                            ocw_lookup=ocw_lookup,
                        )
                    )
                bands = as_text(row.get("bands"), "total")
                curves.append(
                    combined_cls(
                        id=as_text(row.get("series_id")),
                        label=as_text(row.get("label")),
                        curves=tuple(children),
                        operation=as_text(row.get("operation"), "sum"),
                        uncertainty_mode=as_text(row.get("uncertainty_mode"), "quadrature"),
                        x_plot_factor=as_float(row.get("x_plot_factor"), None),
                        x_axis=as_text(row.get("x"), "concentration"),
                        show_stat=bands in {"stat", "stat_syst", "total", "all"},
                        show_syst=bands in {"syst", "stat_syst", "total", "all"},
                        show_total=bands in {"total", "all", "ocw", "stat_syst_ocw"},
                        band_mode=as_text(row.get("band_mode"), "sys_stat"),
                        ocw_config=ocw_lookup(as_text(row.get("ocw_id"))),
                        paper_overlay_id=as_optional_text(row.get("dataset_group")),
                        color_group=as_optional_text(row.get("color_group")),
                        linestyle=as_text(row.get("linestyle"), "-"),
                        linewidth=float(as_float(row.get("linewidth"), 2.0)),
                    )
                )
            else:
                curves.append(
                    _secondary_curve(
                        row,
                        legacy_project_root=legacy_project_root,
                        curve_cls=curve_cls,
                        normalization_lookup=normalization_lookup,
                        selections=selections,
                        ocw_lookup=ocw_lookup,
                    )
                )
        plots.append(
            plot_cls(
                id=str(plot_id),
                title=as_text(first.get("title")),
                curves=tuple(curves),
                xlabel=as_text(first.get("xlabel")),
                ylabel=as_text(first.get("ylabel")),
                xscale=as_text(first.get("xscale"), "log"),
                yscale=as_text(first.get("yscale"), "log"),
                xlim=as_limits(first, "x"),
                ylim=as_limits(first, "y"),
                output=output_path(first, legacy_project_root),
                legend_loc=as_text(first.get("legend_loc"), "best"),
                legend_ncol=int(as_int(first.get("legend_ncol"), 2) or 2),
                legend_fontsize=as_float(first.get("legend_fontsize"), None),
                hide_ocw_legend=as_bool(first.get("hide_ocw_legend"), False),
                experimental_series=tuple(experimental),
            )
        )
    return plots


def load_secondary_metadata_plots(
    legacy_project_root: str | Path,
    group: str,
    *,
    curve_cls: type,
    plot_cls: type,
    selection_cls: type,
) -> list[Any]:
    frame = read_plot_rows("secondary", legacy_project_root)
    if frame.empty:
        return []
    rows = frame.loc[(frame["group"] == group) & (frame["plot_type"] == "metadata")]
    selections = load_secondary_selections(legacy_project_root, selection_cls=selection_cls)
    plots = []
    for plot_id, plot_rows in rows.groupby("plot_id", sort=False):
        first = plot_rows.iloc[0]
        curves = []
        for _, row in plot_rows.iterrows():
            selection_id = as_text(row.get("selection_id"))
            curves.append(
                curve_cls(
                    id=as_text(row.get("series_id")),
                    label=as_text(row.get("label")),
                    selection=selections[selection_id],
                    marker=as_text(row.get("marker"), "o"),
                    linestyle=as_text(row.get("linestyle"), "-"),
                )
            )
        plots.append(
            plot_cls(
                id=str(plot_id),
                title=as_text(first.get("title")),
                curves=tuple(curves),
                x_axis=as_text(first.get("x"), "concentration"),
                y=as_text(first.get("y"), "ni_minus_ne_over_ni"),
                adapter_name=as_text(first.get("model_id"), "ArCF4_primary"),
                xlabel=as_text(first.get("xlabel")),
                ylabel=as_text(first.get("ylabel")),
                xscale=as_text(first.get("xscale"), "log"),
                yscale=as_text(first.get("yscale"), "linear"),
                xlim=as_limits(first, "x"),
                ylim=as_limits(first, "y"),
                output=output_path(first, legacy_project_root),
                legend_loc=as_text(first.get("legend_loc"), "best"),
                legend_ncol=int(as_int(first.get("legend_ncol"), 1) or 1),
                legend_fontsize=as_float(first.get("legend_fontsize"), None),
                marker=as_text(first.get("marker"), "o"),
                linestyle=as_text(first.get("linestyle"), "-"),
                linewidth=float(as_float(first.get("linewidth"), 1.5)),
                markersize=float(as_float(first.get("markersize"), 4.8)),
                group_duplicate_x=as_bool(first.get("group_duplicate_x"), True),
            )
        )
    return plots


def spectra_rows(legacy_project_root: str | Path) -> pd.DataFrame:
    return read_plot_rows("spectra", legacy_project_root)


def active_spectrum_recipes(legacy_project_root: str | Path, kind: str | None = None) -> pd.DataFrame:
    frame = spectra_rows(legacy_project_root)
    if frame.empty or kind is None:
        return frame
    return frame.loc[frame["plot_type"] == kind].copy()


def plot_enabled(family: str, plot_id: str, project_root: str | Path | None = None, default: bool = True) -> bool:
    path = config_path(family, project_root)
    if not path.exists():
        return default
    frame = pd.read_csv(path, keep_default_na=False)
    if "plot_id" not in frame.columns:
        return default
    rows = frame.loc[frame["plot_id"] == plot_id]
    if rows.empty:
        return default
    return any(as_bool(value) for value in rows.get("enabled", pd.Series([True] * len(rows))))
