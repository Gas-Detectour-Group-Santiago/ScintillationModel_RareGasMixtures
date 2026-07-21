from __future__ import annotations

from pathlib import Path
import sys
import os

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from .prediction_types import BandPlotConfig


PAPER_PRIMARY_IDS = {
    "ArCF4_primary_uv",
    "ArCF4_primary_vis",
    "ArN2_primary_uv",
    "ArCF4_IR_primary_total",
    "ArN2_IR_primary_total",
}

ARCF4_PURE_ARGON_DISPLAY_PERCENT = 0.001
ARCF4_IR_LEGACY_PRESSURES = (1.0, 2.0, 3.0)
ARCF4_IR_DISCARDED_CONCENTRATIONS_PERCENT = (20.0, 50.0, 100.0)


def paper_colors(plt, n: int = 4):
    del plt
    try:
        from plot_style import palette

        return palette(n, start=0.18, stop=0.82)
    except Exception:  # pragma: no cover - standalone fallback
        import matplotlib.pyplot as _plt

        return _plt.get_cmap("viridis")(np.linspace(0.18, 0.82, n))


def find_project_root(path: Path) -> Path:
    p = Path(path).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").is_dir() and (parent / "models").is_dir():
            return parent
    return p.parents[1]


def pressure_label(p: float) -> str:
    return f"{float(p):g}bar"


def error_label(p: float) -> str:
    return f"Err {pressure_label(p)}"


def _display_cf4_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    x[x <= 0.0] = ARCF4_PURE_ARGON_DISPLAY_PERCENT
    return x


def _discarded_concentration_mask(x_percent: np.ndarray) -> np.ndarray:
    x_percent = np.asarray(x_percent, dtype=float)
    mask = np.zeros_like(x_percent, dtype=bool)
    for c in ARCF4_IR_DISCARDED_CONCENTRATIONS_PERCENT:
        mask |= np.isclose(x_percent, c, rtol=0.0, atol=1e-10)
    return mask


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _read_parameter_vector(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "parameter" in df.columns:
        return df["parameter"].to_numpy(dtype=float)
    if "value" in df.columns:
        return df["value"].to_numpy(dtype=float)
    df = pd.read_csv(path, index_col=0)
    if "parameter" in df.columns:
        return df["parameter"].to_numpy(dtype=float)
    if "value" in df.columns:
        return df["value"].to_numpy(dtype=float)
    return None


def _normalization_denominator(project_root: Path, config: BandPlotConfig) -> float | None:
    norm = config.normalization
    if norm.mode in {"as_fit", "set_norm_one"}:
        return 1.0
    if norm.mode == "fixed_norm":
        return norm.fixed_norm

    fit_name = config.fit_name
    if norm.mode == "reference_norm":
        fit_name = norm.reference_fit_name or fit_name
    elif norm.mode != "own_norm":
        return None

    params = _read_parameter_vector(project_root / "data" / "Parameters" / f"{fit_name}.csv")
    if params is None or len(params) == 0:
        return None
    return float(params[0])


def _w_arcf4(fraction):
    cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100
    ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])
    return np.interp(np.asarray(fraction, dtype=float), cf4_pct, ion_pot)


def _w_arn2(fraction, WAr=26.4, WN2=34.8):
    x = np.asarray(fraction, dtype=float)
    return 1.0 / ((1.0 - x) / WAr + x / WN2)


def _scaled_experimental_xy(
    project_root: Path,
    config: BandPlotConfig,
    *,
    csv_path: Path,
    x_col: str,
    pressure: float,
    w_func,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = project_root / csv_path
    if not path.exists() and "csv" in csv_path.parts:
        parts = list(csv_path.parts)
        parts.remove("csv")
        fallback = project_root / Path(*parts)
        if fallback.exists():
            path = fallback
    if not path.exists():
        return None

    denom = (
        _normalization_denominator(project_root, config)
        if config.scale_xray_with_normalization
        else 1.0
    )
    if denom is None or denom == 0:
        return None

    df = pd.read_csv(path)
    if x_col not in df.columns:
        return None

    pcol = _first_existing_col(df, [pressure_label(pressure), f"{pressure:.1f}bar"])
    ecol = _first_existing_col(df, [error_label(pressure), f"Err {pressure:.1f}bar"])
    if pcol is None:
        return None

    x = df[x_col].to_numpy(dtype=float)
    w = np.asarray(w_func(x * 0.01), dtype=float)
    scale = float(config.normalization.output_scale) / denom
    y = df[pcol].to_numpy(dtype=float) / w * scale
    yerr = df[ecol].to_numpy(dtype=float) / w * scale if ecol is not None else np.zeros_like(y)
    xsafe = _display_cf4_x(x)
    return xsafe, y, yerr


def _scaled_ir_total(project_root: Path, config: BandPlotConfig, *, gas: str, x_col: str, pressure: float, w_func):
    denom = (
        _normalization_denominator(project_root, config)
        if config.scale_xray_with_normalization
        else 1.0
    )
    if denom is None or denom == 0:
        return None

    base_candidates = [
        project_root / "data" / "Experimental" / gas / "csv",
        project_root / "data" / "Experimental" / gas,
    ]
    lines = ("696", "727", "750", "763", "772")
    x_ref = None
    y_total = None
    e2_total = None

    for line in lines:
        path = _first_existing_path([base / f"{line}.csv" for base in base_candidates])
        if path is None:
            return None
        df = pd.read_csv(path)
        if x_col not in df.columns:
            return None
        pcol = _first_existing_col(df, [pressure_label(pressure), f"{pressure:.1f}bar"])
        ecol = _first_existing_col(df, [error_label(pressure), f"Err {pressure:.1f}bar"])
        if pcol is None:
            return None
        x = df[x_col].to_numpy(dtype=float)
        w = np.asarray(w_func(x * 0.01), dtype=float)
        scale = float(config.normalization.output_scale) / denom
        y = df[pcol].to_numpy(dtype=float) / w * scale
        e = df[ecol].to_numpy(dtype=float) / w * scale if ecol is not None else np.zeros_like(y)
        x_ref = x if x_ref is None else x_ref
        y_total = y if y_total is None else y_total + y
        e2_total = e**2 if e2_total is None else e2_total + e**2

    if x_ref is None or y_total is None or e2_total is None:
        return None

    x_plot = _display_cf4_x(x_ref) if gas == "ArCF4" else np.where(
        x_ref <= 0,
        np.min(x_ref[x_ref > 0]) * 0.1 if np.any(x_ref > 0) else 1e-6,
        x_ref,
    )
    yerr_total = np.sqrt(e2_total)

    # Mismo criterio visual que en el ajuste IR, pero solo para ArCF4 y solo
    # para 1, 2 y 3 bar: 20/50/100 % no se dibujan, y se quitan puntos por
    # debajo del floor = min_p max(Y_20,Y_50,Y_100), calculado con p=1,2,3.
    if gas == "ArCF4" and any(np.isclose(float(pressure), p) for p in ARCF4_IR_LEGACY_PRESSURES):
        discarded = _discarded_concentration_mask(x_ref)

        maxima = []
        for p_floor in ARCF4_IR_LEGACY_PRESSURES:
            total_for_p = None
            for line in lines:
                path = _first_existing_path([base / f"{line}.csv" for base in base_candidates])
                if path is None:
                    continue
                df = pd.read_csv(path)
                pcol = _first_existing_col(df, [pressure_label(p_floor), f"{p_floor:.1f}bar"])
                if pcol is None or x_col not in df.columns:
                    continue
                xx = df[x_col].to_numpy(dtype=float)
                yy = df[pcol].to_numpy(dtype=float) / np.asarray(w_func(xx * 0.01), dtype=float) * scale
                total_for_p = yy if total_for_p is None else total_for_p + yy

            if total_for_p is not None:
                vals = total_for_p[discarded & np.isfinite(total_for_p) & (total_for_p > 0.0)]
                if vals.size:
                    maxima.append(float(np.nanmax(vals)))

        floor = float(np.nanmin(maxima)) if maxima else None
        mask = np.isfinite(x_plot) & np.isfinite(y_total) & np.isfinite(yerr_total) & (y_total > 0.0) & (yerr_total > 0.0)
        mask &= ~discarded
        if floor is not None and np.isfinite(floor):
            mask &= y_total >= floor
        return x_plot[mask], y_total[mask], yerr_total[mask]

    mask = np.isfinite(x_plot) & np.isfinite(y_total) & np.isfinite(yerr_total) & (y_total > 0.0) & (yerr_total > 0.0)
    return x_plot[mask], y_total[mask], yerr_total[mask]


def _err(ax, x, y, yerr, *, color, label, marker="o", ms=4):
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker=marker,
        linestyle="none",
        ms=ms,
        color=color,
        ecolor=color,
        capsize=2,
        label=label,
    )


def _configured_dataset_ids(config: BandPlotConfig, project_root: Path) -> tuple[str, ...]:
    if config.overlays:
        return tuple(str(value) for value in config.overlays)
    try:
        from scintillation.plotting.recipe_config import read_plot_rows, split_values

        rows = read_plot_rows("primary", project_root)
        selected = rows.loc[rows["plot_id"] == config.id]
        if selected.empty:
            return tuple()
        return split_values(selected.iloc[0].get("datasets"), cast=str)
    except Exception:
        return tuple()


def _registry_rows(project_root: Path, dataset_ids: tuple[str, ...]) -> pd.DataFrame:
    from scintillation.plotting.recipe_config import load_experimental_dataset_registry, as_bool

    registry = load_experimental_dataset_registry(project_root)
    if registry.empty:
        return registry
    registry = registry.loc[registry["dataset_id"].astype(str).isin(dataset_ids)]
    if "enabled" in registry:
        registry = registry.loc[registry["enabled"].map(as_bool)]
    order = {dataset_id: index for index, dataset_id in enumerate(dataset_ids)}
    return registry.assign(_order=registry["dataset_id"].map(order)).sort_values("_order")


def _series_color(spec: pd.Series):
    explicit = str(spec.get("color", "")).strip()
    if explicit:
        return explicit
    position = str(spec.get("color_position", "")).strip()
    if not position:
        return None
    try:
        from plot_style import PAPER_CMAP
        import matplotlib.pyplot as _plt
        return _plt.get_cmap(PAPER_CMAP)(float(position))
    except Exception:
        return None


def _static_dataset(repo_root: Path, spec: pd.Series):
    path = repo_root / str(spec["file"])
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    xcol = str(spec.get("x_column") or "x")
    ycol = str(spec.get("y_column") or "y")
    ecol = str(spec.get("yerr_column") or "").strip()
    if xcol not in frame or ycol not in frame:
        return None
    x = pd.to_numeric(frame[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(frame[ecol], errors="coerce").to_numpy(dtype=float) if ecol and ecol in frame else np.zeros_like(y)
    return x, y, e


def _xray_dataset(runtime_root: Path, repo_root: Path, config: BandPlotConfig, spec: pd.Series):
    path = repo_root / str(spec["file"])
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    gas = str(spec.get("gas", ""))
    xcol = str(spec.get("x_column") or ("fCF4" if gas == "ArCF4" else "fN2"))
    pcol = _first_existing_col(frame, [pressure_label(config.pressure), f"{config.pressure:.1f}bar"])
    ecol = _first_existing_col(frame, [error_label(config.pressure), f"Err {config.pressure:.1f}bar"])
    if xcol not in frame or pcol is None:
        return None
    denominator = (
        _normalization_denominator(runtime_root, config)
        if config.scale_xray_with_normalization
        else 1.0
    )
    if denominator is None or denominator == 0:
        return None
    x = pd.to_numeric(frame[xcol], errors="coerce").to_numpy(dtype=float)
    w = _w_arcf4(x * 0.01) if gas == "ArCF4" else _w_arn2(x * 0.01)
    scale = float(config.normalization.output_scale) / denominator
    y = pd.to_numeric(frame[pcol], errors="coerce").to_numpy(dtype=float) / w * scale
    e = pd.to_numeric(frame[ecol], errors="coerce").to_numpy(dtype=float) / w * scale if ecol else np.zeros_like(y)
    if gas == "ArCF4":
        x = _display_cf4_x(x)
    return x, y, e


def get_primary_paper_overlay_series(config: BandPlotConfig, output: Path) -> list[dict[str, object]]:
    """Load all experimental layers declared for this primary plot.

    Data values, labels, markers and stable colour positions live in
    ``config/experimental_datasets.csv``.  The plotting code only implements
    reusable transformations such as X-ray normalisation and the sum of the
    five measured IR lines.
    """
    runtime_root = find_project_root(output)
    repo_root = Path(os.environ.get("SCINTILLATION_ROOT", runtime_root)).resolve()
    dataset_ids = _configured_dataset_ids(config, repo_root)
    rows = _registry_rows(repo_root, dataset_ids)
    series: list[dict[str, object]] = []
    for _, spec in rows.iterrows():
        transform = str(spec.get("transform", "static") or "static")
        if transform == "static":
            values = _static_dataset(repo_root, spec)
        elif transform == "primary_xray":
            values = _xray_dataset(runtime_root, repo_root, config, spec)
        elif transform == "primary_ir_sum":
            gas = str(spec.get("gas", ""))
            values = _scaled_ir_total(
                runtime_root,
                config,
                gas=gas,
                x_col="fCF4" if gas == "ArCF4" else "fN2",
                pressure=config.pressure,
                w_func=_w_arcf4 if gas == "ArCF4" else _w_arn2,
            )
        else:
            raise ValueError(f"Unsupported primary experimental transform {transform!r}")
        if values is None:
            continue
        x, y, yerr = values
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
        series.append(
            {
                "x": np.asarray(x)[finite],
                "y": np.asarray(y)[finite],
                "yerr": np.asarray(yerr)[finite],
                "label": str(spec.get("label", spec.get("dataset_id", ""))),
                "marker": str(spec.get("marker", "o")),
                "ms": float(spec.get("markersize") or 5.0),
                "color": _series_color(spec),
                "marker_fill": str(spec.get("marker_fill", "filled")),
                "capsize": float(spec.get("capsize") or 2.5),
            }
        )
    return series


def plot_primary_paper_overlays(ax, config: BandPlotConfig, output: Path, plt) -> None:
    colors = paper_colors(plt, 4)
    for idx, item in enumerate(get_primary_paper_overlay_series(config, output)):
        color = item.get("color") or colors[min(idx, len(colors) - 1)]
        ax.errorbar(
            item["x"], item["y"], yerr=item["yerr"],
            marker=str(item.get("marker", "o")), linestyle="none",
            ms=float(item.get("ms", 4)), color=color, ecolor=color,
            markerfacecolor="white" if item.get("marker_fill") == "open" else color,
            capsize=float(item.get("capsize", 2.5)),
            label=str(item.get("label", "")),
        )
