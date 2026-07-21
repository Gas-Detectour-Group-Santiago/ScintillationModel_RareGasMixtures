from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .recipe_config import as_bool, as_float, as_int, as_text, split_values
from .style import setup_style, palette, LINEWIDTH_MAIN, LINEWIDTH_SECONDARY


def _values(row: Mapping[str, object], key: str, default: Sequence[float]) -> tuple[float, ...]:
    values = split_values(row.get(key), cast=float)
    return values or tuple(float(v) for v in default)


def _window(row: Mapping[str, object], lo_key: str, hi_key: str, default: tuple[float, float]) -> tuple[float, float]:
    lo = as_float(row.get(lo_key), default[0])
    hi = as_float(row.get(hi_key), default[1])
    assert lo is not None and hi is not None
    return float(lo), float(hi)


def _output(row: Mapping[str, object], project_root: Path) -> Path:
    path = Path(as_text(row.get("output")))
    if not path.is_absolute():
        path = project_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _panel_grid(row: Mapping[str, object], n_panels: int) -> tuple[int, int]:
    rows = as_int(row.get("mosaic_rows"), None)
    cols = as_int(row.get("mosaic_cols"), None)
    if rows and cols and rows * cols >= n_panels:
        return int(rows), int(cols)
    cols = max(1, int(np.ceil(np.sqrt(max(n_panels, 1)))))
    rows = max(1, int(np.ceil(max(n_panels, 1) / cols)))
    return rows, cols


def _panel_title(concentration: float) -> str:
    return f"{float(concentration):g}% additive"


def _common_ylim(frames: Sequence[pd.DataFrame], y: str, window: tuple[float, float] | None = None) -> tuple[float, float]:
    maxima: list[float] = []
    for frame in frames:
        one = frame
        if window is not None and "wavelength_nm" in one:
            one = one.loc[one["wavelength_nm"].between(*window)]
        values = pd.to_numeric(one.get(y, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values >= 0)]
        if values.size:
            maxima.append(float(values.max()))
    ymax = max(maxima, default=0.0)
    return 0.0, 1.12 * ymax if ymax > 0 else 1.0


def _figure_legend(fig, axes, *, ncol: int) -> None:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in np.asarray(axes, dtype=object).ravel():
        if ax is None:
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l, strict=False):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=max(1, int(ncol)), frameon=False)


def _filter_spectrum(frame: pd.DataFrame, *, gas: str, concentration: float, pressure: float,
                     components: tuple[str, ...], source_column: str | None = None) -> pd.DataFrame:
    one = frame.copy()
    if "gas_mixture" in one:
        one = one.loc[one["gas_mixture"].astype(str) == gas]
    one = one.loc[np.isclose(pd.to_numeric(one["concentration_percent"], errors="coerce"), concentration)]
    one = one.loc[np.isclose(pd.to_numeric(one["pressure_bar"], errors="coerce"), pressure)]
    if "component" in one and components:
        one = one.loc[one["component"].astype(str).isin(components)]
        if len(components) > 1:
            value_col = "intensity_ph_MeV_nm"
            one = one.groupby("wavelength_nm", as_index=False)[value_col].sum()
    if source_column and "spectrum_column" in one:
        one = one.loc[one["spectrum_column"].astype(str) == source_column]
    return one.sort_values("wavelength_nm")


def render_raw(row: Mapping[str, object], *, project_root: Path, frame: pd.DataFrame,
               reference: pd.DataFrame | None = None) -> Path:
    setup_style(context="mosaic")
    gas = as_text(row.get("gas"))
    concentrations = _values(row, "concentrations_percent", sorted(frame["concentration_percent"].dropna().unique()))
    pressures = _values(row, "pressures_bar", sorted(frame["pressure_bar"].dropna().unique()))
    nrows, ncols = _panel_grid(row, len(concentrations))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.05 * ncols, 3.55 * nrows), sharex=True, squeeze=False)
    axes = axs.ravel()
    colors = palette(len(pressures))
    window = _window(row, "wavelength_min_nm", "wavelength_max_nm", (180.0, 800.0))
    source_column = as_text(row.get("components"), "mean_spectrum")
    yframes = [frame]
    if reference is not None and not reference.empty:
        yframes.append(reference)
    common = _common_ylim(yframes, "intensity_raw", window) if as_bool(row.get("share_y"), True) else None

    for idx, ax in enumerate(axes):
        if idx >= len(concentrations):
            ax.axis("off")
            continue
        concentration = concentrations[idx]
        if reference is not None and not reference.empty:
            ref = reference.loc[reference["wavelength_nm"].between(*window)].sort_values("wavelength_nm")
            if not ref.empty:
                ax.fill_between(ref["wavelength_nm"], 0, ref["intensity_raw"], alpha=0.20,
                                label="Ar--CF$_4$ 95/5 reference")
        panel_max = 0.0
        for color, pressure in zip(colors, pressures, strict=False):
            one = _filter_spectrum(frame, gas=gas, concentration=concentration, pressure=pressure,
                                   components=(), source_column=source_column)
            if one.empty:
                continue
            y = pd.to_numeric(one["intensity_raw"], errors="coerce").to_numpy(dtype=float)
            panel_max = max(panel_max, float(np.nanmax(np.clip(y, 0, None))))
            ax.plot(one["wavelength_nm"], y, color=color, lw=LINEWIDTH_SECONDARY, label=f"{pressure:g} bar")
        ax.set_title(_panel_title(concentration))
        ax.set_xlim(*window)
        ax.set_ylim(*(common or (0, 1.12 * panel_max if panel_max > 0 else 1)))
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel("Raw intensity")
    fig.suptitle(as_text(row.get("title"), f"{gas} experimental spectra"), y=0.985)
    _figure_legend(fig, axes, ncol=min(len(pressures) + 1, 6))
    fig.subplots_adjust(top=0.87, wspace=0.18, hspace=0.28)
    out = _output(row, project_root)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra-recipes] {row.get('plot_id')}: {out}")
    return out


def _generated_groups(
    frame: pd.DataFrame,
    *,
    gas: str,
    components: tuple[str, ...],
) -> dict[tuple[float, float], pd.DataFrame]:
    """Index a large generated spectrum once instead of scanning it per panel."""
    columns = ["concentration_percent", "pressure_bar", "wavelength_nm", "intensity_ph_MeV_nm"]
    if "gas_mixture" in frame:
        selected = frame.loc[frame["gas_mixture"] == gas]
    else:
        selected = frame
    if "component" in selected and components:
        selected = selected.loc[selected["component"].isin(components)]
    selected = selected.loc[:, columns].copy()
    for column in ("concentration_percent", "pressure_bar", "wavelength_nm", "intensity_ph_MeV_nm"):
        if not pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=columns)
    # Summing by wavelength works for one or many selected components and makes
    # total/fast/slow/component combinations share the same rendering path.
    selected = (
        selected.groupby(["concentration_percent", "pressure_bar", "wavelength_nm"], as_index=False, sort=False)[
            "intensity_ph_MeV_nm"
        ].sum()
        .sort_values(["concentration_percent", "pressure_bar", "wavelength_nm"])
    )
    return {
        (float(concentration), float(pressure)): group[["wavelength_nm", "intensity_ph_MeV_nm"]].reset_index(drop=True)
        for (concentration, pressure), group in selected.groupby(["concentration_percent", "pressure_bar"], sort=False)
    }


def _plot_generated_panel(ax, groups: Mapping[tuple[float, float], pd.DataFrame], *, concentration: float,
                          pressures: tuple[float, ...], colors, window: tuple[float, float],
                          log_y: bool, linewidth: float = LINEWIDTH_SECONDARY) -> float:
    ymax = 0.0
    for color, pressure in zip(colors, pressures, strict=False):
        one = groups.get((float(concentration), float(pressure)))
        if one is None:
            continue
        one = one.loc[one["wavelength_nm"].between(*window)]
        if one.empty:
            continue
        y = pd.to_numeric(one["intensity_ph_MeV_nm"], errors="coerce").to_numpy(dtype=float)
        ymax = max(ymax, float(np.nanmax(np.clip(y, 0, None))))
        if log_y:
            y = np.where(y > 0, y, np.nan)
        ax.plot(one["wavelength_nm"], y, color=color, lw=linewidth, label=f"{pressure:g} bar")
    return ymax


def render_generated(row: Mapping[str, object], *, project_root: Path, frame: pd.DataFrame) -> Path:
    setup_style(context="mosaic")
    gas = as_text(row.get("gas"))
    concentrations = _values(row, "concentrations_percent", sorted(frame["concentration_percent"].dropna().unique()))
    pressures = _values(row, "pressures_bar", sorted(frame["pressure_bar"].dropna().unique()))
    components = split_values(row.get("components"), cast=str) or ("total",)
    nrows, ncols = _panel_grid(row, len(concentrations))
    window = _window(row, "wavelength_min_nm", "wavelength_max_nm", (100.0, 800.0))
    inset_window = _window(row, "inset_min_nm", "inset_max_nm", (110.0, 170.0))
    log_y = as_bool(row.get("log_y"), False)
    share_y = as_bool(row.get("share_y"), True)
    use_inset = as_bool(row.get("inset"), False)
    use_broken = as_bool(row.get("broken_x"), False)
    colors = palette(len(pressures))
    groups = _generated_groups(frame, gas=gas, components=components)

    if use_broken:
        fig = plt.figure(figsize=(5.6 * ncols, 3.65 * nrows))
        outer = fig.add_gridspec(nrows, ncols)
        legend_axes = []
        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            if idx >= len(concentrations):
                ax = fig.add_subplot(outer[r, c]); ax.axis("off"); continue
            inner = outer[r, c].subgridspec(1, 2, width_ratios=(1.0, 3.2), wspace=0.05)
            left = fig.add_subplot(inner[0, 0]); right = fig.add_subplot(inner[0, 1])
            concentration = concentrations[idx]
            _plot_generated_panel(left, groups, concentration=concentration, pressures=pressures,
                                  colors=colors, window=inset_window, log_y=log_y)
            _plot_generated_panel(right, groups, concentration=concentration, pressures=pressures,
                                  colors=colors, window=window, log_y=log_y)
            left.set_xlim(*inset_window); right.set_xlim(*window)
            left.spines["right"].set_visible(False); right.spines["left"].set_visible(False)
            right.tick_params(labelleft=False, left=False)
            right.set_title(_panel_title(concentration)); legend_axes.append(right)
            if r == nrows - 1: left.set_xlabel(r"$\lambda$ [nm]"); right.set_xlabel(r"$\lambda$ [nm]")
            if c == 0: left.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        axes_for_legend = legend_axes
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=(5.05 * ncols, 3.55 * nrows), sharex=True, squeeze=False)
        axes = axs.ravel(); axes_for_legend = axes
        if share_y and not log_y:
            selected_frames = [
                group.loc[group["wavelength_nm"].between(*window)]
                for (concentration, pressure), group in groups.items()
                if concentration in concentrations and pressure in pressures
            ]
            global_ylim = _common_ylim(selected_frames, "intensity_ph_MeV_nm")
        else:
            global_ylim = None
        for idx, ax in enumerate(axes):
            if idx >= len(concentrations):
                ax.axis("off"); continue
            concentration = concentrations[idx]
            ymax = _plot_generated_panel(ax, groups, concentration=concentration, pressures=pressures,
                                          colors=colors, window=window, log_y=log_y)
            ax.set_title(_panel_title(concentration)); ax.set_xlim(*window)
            if log_y:
                ax.set_yscale("log")
            elif global_ylim:
                ax.set_ylim(*global_ylim)
            else:
                ax.set_ylim(0, 1.12 * ymax if ymax > 0 else 1)
            if use_inset:
                width = f"{float(as_float(row.get('inset_width_percent'), 45.0))}%"
                height = f"{float(as_float(row.get('inset_height_percent'), 45.0))}%"
                ins = inset_axes(ax, width=width, height=height,
                                 loc=as_text(row.get("inset_location"), "upper right"), borderpad=0.9)
                _plot_generated_panel(ins, groups, concentration=concentration, pressures=pressures,
                                      colors=colors, window=inset_window, log_y=log_y,
                                      linewidth=max(0.8, LINEWIDTH_SECONDARY * 0.8))
                ins.set_xlim(*inset_window); ins.set_title("VUV", fontsize=8); ins.tick_params(labelsize=7)
            if idx // ncols == nrows - 1: ax.set_xlabel(r"$\lambda$ [nm]")
            if idx % ncols == 0: ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
    fig.suptitle(as_text(row.get("title"), f"{gas} generated spectra"), y=0.985)
    _figure_legend(fig, axes_for_legend, ncol=min(len(pressures), 6))
    fig.subplots_adjust(top=0.87, wspace=0.20, hspace=0.29)
    out = _output(row, project_root)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra-recipes] {row.get('plot_id')}: {out}")
    return out


def render_comparison(row: Mapping[str, object], *, project_root: Path, frame: pd.DataFrame) -> Path:
    setup_style(context="mosaic")
    concentrations = _values(row, "concentrations_percent", sorted(frame["concentration_percent"].dropna().unique()))
    pressures = _values(row, "pressures_bar", sorted(frame["pressure_bar"].dropna().unique()))
    gases = split_values(row.get("gas"), cast=str) or tuple(frame["gas_mixture"].astype(str).unique())
    nrows, ncols = _panel_grid(row, len(concentrations))
    window = _window(row, "wavelength_min_nm", "wavelength_max_nm", (180.0, 800.0))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.05 * ncols, 3.55 * nrows), sharex=True, squeeze=False)
    axes = axs.ravel()
    colors = palette(max(1, len(gases) * len(pressures)))
    color_map = {(gas, pressure): colors[i] for i, (gas, pressure) in enumerate((g, p) for g in gases for p in pressures)}
    common = _common_ylim([frame], "plot_intensity", window) if as_bool(row.get("share_y"), True) else None
    for idx, ax in enumerate(axes):
        if idx >= len(concentrations): ax.axis("off"); continue
        concentration = concentrations[idx]; ymax = 0.0
        for gas in gases:
            for pressure in pressures:
                for source in ("raw", "generated"):
                    one = frame.loc[(frame["gas_mixture"].astype(str) == gas)
                                    & np.isclose(frame["concentration_percent"], concentration)
                                    & np.isclose(frame["pressure_bar"], pressure)
                                    & (frame["source"].astype(str) == source)].sort_values("wavelength_nm")
                    one = one.loc[one["wavelength_nm"].between(*window)]
                    if one.empty: continue
                    y = pd.to_numeric(one["plot_intensity"], errors="coerce").to_numpy(dtype=float)
                    ymax = max(ymax, float(np.nanmax(np.clip(y, 0, None))))
                    raw = source == "raw"
                    ax.plot(one["wavelength_nm"], y, color=color_map[(gas, pressure)],
                            lw=LINEWIDTH_MAIN if raw else LINEWIDTH_SECONDARY,
                            alpha=0.40 if raw else 1.0,
                            label=f"{gas}, {pressure:g} bar, {'raw' if raw else 'prediction'}")
        ax.set_title(_panel_title(concentration)); ax.set_xlim(*window)
        ax.set_ylim(*(common or (0, 1.12 * ymax if ymax > 0 else 1)))
        if idx // ncols == nrows - 1: ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0: ax.set_ylabel(frame["unit_label"].dropna().iloc[0] if "unit_label" in frame and not frame.empty else "Scaled intensity")
    fig.suptitle(as_text(row.get("title"), "Raw spectra vs prediction"), y=0.985)
    _figure_legend(fig, axes, ncol=min(4, max(1, len(gases) * len(pressures))))
    fig.subplots_adjust(top=0.87, wspace=0.18, hspace=0.28)
    out = _output(row, project_root); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[spectra-recipes] {row.get('plot_id')}: {out}")
    return out
