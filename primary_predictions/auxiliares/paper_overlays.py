from __future__ import annotations

from pathlib import Path

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


def paper_colors(plt, n: int = 4):
    return plt.get_cmap("viridis")(np.linspace(0.18, 0.82, n))


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
    if not path.exists():
        return None

    denom = _normalization_denominator(project_root, config)
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
    xsafe = np.where(x <= 0, np.min(x[x > 0]) * 0.1 if np.any(x > 0) else 1e-6, x)
    return xsafe, y, yerr


def _scaled_ir_total(project_root: Path, config: BandPlotConfig, *, gas: str, x_col: str, pressure: float, w_func):
    denom = _normalization_denominator(project_root, config)
    if denom is None or denom == 0:
        return None

    base = project_root / "data" / "Experimental" / gas / "csv"
    lines = ("696", "727", "750", "763", "772")
    x_ref = None
    y_total = None
    e2_total = None

    for line in lines:
        path = base / f"{line}.csv"
        if not path.exists():
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
    xsafe = np.where(x_ref <= 0, np.min(x_ref[x_ref > 0]) * 0.1 if np.any(x_ref > 0) else 1e-6, x_ref)
    return xsafe, y_total, np.sqrt(e2_total)


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


def get_primary_paper_overlay_series(config: BandPlotConfig, output: Path) -> list[dict[str, object]]:
    """Return overlay series in a plotting-agnostic format.

    Each element contains x, y, yerr, label, marker and ms.
    """

    project_root = find_project_root(output)
    series: list[dict[str, object]] = []

    if config.id == "ArCF4_primary_uv":
        xy = _scaled_experimental_xy(
            project_root,
            config,
            csv_path=Path("data/Experimental/ArCF4/csv/UV.csv"),
            x_col="fCF4",
            pressure=config.pressure,
            w_func=_w_arcf4,
        )
        if xy:
            series.append(dict(x=xy[0], y=xy[1], yerr=xy[2], label="X-ray (220--400 nm)", marker="o", ms=4))
        series.append(dict(x=np.array([100.0]), y=np.array([2175.0]), yerr=np.array([2600 - 2175]), label=r"$\alpha$ Morozov", marker="x", ms=5))
        series.append(dict(x=np.array([0.001]), y=np.array([2.7e3]), yerr=np.array([0.14e3]), label=r"$\alpha$ Santorelli", marker="v", ms=5))
        series.append(
            dict(
                x=np.array([0.15, 0.35, 0.50, 1.00, 2.00, 6.00, 11.00]),
                y=np.array([358.9, 350.8, 292.2, 209.4, 227.8, 245.5, 263.4]),
                yerr=np.array([3.0, 3.9, 0.2, 2.0, 10.9, 2.4, 4.3]),
                label=r"$\alpha$ P. Amedo",
                marker="o",
                ms=4,
            )
        )
        return series

    if config.id == "ArCF4_primary_vis":
        xy = _scaled_experimental_xy(
            project_root,
            config,
            csv_path=Path("data/Experimental/ArCF4/csv/vis.csv"),
            x_col="fCF4",
            pressure=config.pressure,
            w_func=_w_arcf4,
        )
        if xy:
            series.append(dict(x=xy[0], y=xy[1], yerr=xy[2], label="X-ray", marker="o", ms=4))
        series.append(dict(x=np.array([0.2, 0.4, 0.7, 1.0, 2.0, 7.0, 10.0]), y=np.array([450, 500, 600, 1150, 1300, 1850, 1900]), yerr=np.array([60, 60, 60, 90, 100, 120, 120]), label=r"$\alpha$ P. Amedo", marker="o", ms=4))
        series.append(dict(x=np.array([100.0]), y=np.array([695.0]), yerr=np.array([827 - 695]), label=r"$\alpha$ Morozov", marker="x", ms=5))
        series.append(dict(x=np.array([100.0]), y=np.array([1184.7]), yerr=np.array([47.0]), label=r"$\alpha$ Lehaut", marker="v", ms=5))
        return series

    if config.id == "ArN2_primary_uv":
        xy = _scaled_experimental_xy(
            project_root,
            config,
            csv_path=Path("data/Experimental/ArN2/csv/yield_N2.csv"),
            x_col="fN2",
            pressure=config.pressure,
            w_func=_w_arn2,
        )
        if xy:
            x, y, yerr = xy
            mask = (x != 100) & (x != 150)
            series.append(dict(x=x[mask], y=y[mask], yerr=yerr[mask], label=f"X-ray {config.pressure:g} bar", marker="o", ms=4))
        series.append(dict(x=np.array([100.0]), y=np.array([94.0]), yerr=np.array([94 * 14e-2]), label="MacFly Collaboration", marker="v", ms=5))
        series.append(dict(x=np.array([100.0]), y=np.array([145.0]), yerr=np.array([2.0]), label=r"$\alpha$ Morii", marker="o", ms=5))
        series.append(dict(x=np.array([100.0]), y=np.array([96.0]), yerr=np.array([40.0]), label=r"$\alpha$ Lehaut", marker="s", ms=5))
        return series

    if config.id == "ArCF4_IR_primary_total":
        xy = _scaled_ir_total(project_root, config, gas="ArCF4", x_col="fCF4", pressure=config.pressure, w_func=_w_arcf4)
        if xy:
            series.append(dict(x=xy[0], y=xy[1], yerr=xy[2], label=f"X-ray {config.pressure:g} bar", marker="o", ms=4))
        return series

    if config.id == "ArN2_IR_primary_total":
        xy = _scaled_ir_total(project_root, config, gas="ArN2", x_col="fN2", pressure=config.pressure, w_func=_w_arn2)
        if xy:
            series.append(dict(x=xy[0], y=xy[1], yerr=xy[2], label=f"X-ray {config.pressure:g} bar", marker="o", ms=4))
        return series

    return series


def plot_primary_paper_overlays(ax, config: BandPlotConfig, output: Path, plt) -> None:
    colors = paper_colors(plt, 4)
    for idx, item in enumerate(get_primary_paper_overlay_series(config, output)):
        color = colors[min(idx, len(colors) - 1)]
        _err(
            ax,
            item["x"],
            item["y"],
            item["yerr"],
            marker=str(item.get("marker", "o")),
            ms=float(item.get("ms", 4)),
            color=color,
            label=str(item.get("label", "")),
        )
