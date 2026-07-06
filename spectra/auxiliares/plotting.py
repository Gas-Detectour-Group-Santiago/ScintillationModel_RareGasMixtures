from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import colors, ensure_parent, match_float, setup_plot_style


def _positive_ymax(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    finite = finite[finite >= 0.0]
    if finite.size == 0:
        return 0.0
    ymax = float(np.nanmax(finite))
    return ymax if np.isfinite(ymax) and ymax > 0.0 else 0.0


def _common_ylim_from_df(df: pd.DataFrame, y_col: str) -> tuple[float, float]:
    if df.empty or y_col not in df.columns:
        return (0.0, 1.0)
    ymax = _positive_ymax(df[y_col].to_numpy(dtype=float))
    return (0.0, 1.12 * ymax if ymax > 0.0 else 1.0)


def _window_mask(series: pd.Series, window: tuple[float, float]) -> pd.Series:
    lo, hi = float(window[0]), float(window[1])
    values = pd.to_numeric(series, errors="coerce")
    return (values >= lo) & (values <= hi)


def _common_ylim_from_df_window(
    df: pd.DataFrame,
    y_col: str,
    *,
    x_col: str = "wavelength_nm",
    window: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if window is None:
        return _common_ylim_from_df(df, y_col)
    if df.empty or y_col not in df.columns or x_col not in df.columns:
        return (0.0, 1.0)
    sub = df[_window_mask(df[x_col], window)]
    return _common_ylim_from_df(sub, y_col)


def _positive_ymax_from_df_window(
    df: pd.DataFrame,
    y_col: str,
    *,
    x_col: str = "wavelength_nm",
    window: tuple[float, float] | None = None,
) -> float:
    if df.empty or y_col not in df.columns:
        return 0.0
    sub = df if window is None else df[_window_mask(df[x_col], window)]
    return _positive_ymax(sub[y_col].to_numpy(dtype=float))


def _positive_y_for_log(values: np.ndarray, ymin: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite_positive.size == 0:
        arr[:] = np.nan
        return arr
    floor = float(ymin) if ymin is not None and np.isfinite(ymin) and ymin > 0.0 else float(np.nanmin(finite_positive))
    arr[~np.isfinite(arr) | (arr <= 0.0)] = np.nan
    arr[arr < floor] = np.nan
    return arr


def _common_log_ylim_from_df(df: pd.DataFrame, y_col: str, ymin: float, ymax_factor: float) -> tuple[float, float]:
    if df.empty or y_col not in df.columns:
        return (float(ymin), 1.0)
    arr = np.asarray(df[y_col], dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size == 0:
        return (float(ymin), 1.0)
    ymax = float(np.nanmax(positive))
    upper = max(float(ymin) * 10.0, ymax * float(ymax_factor))
    return (float(ymin), upper)


def _panel_title(concentration: float) -> str:
    return f"{float(concentration):g}% additive"


def _raw_suptitle(gas: str) -> str:
    titles = {
        "ArCF4": r"Experimental raw spectra of Ar--CF$_4$ mixtures",
        "ArN2": r"Experimental raw spectra of Ar--N$_2$ mixtures",
    }
    return titles.get(gas, rf"Experimental raw spectra of {gas}")


def _figure_legend(
    fig,
    axs,
    *,
    ncol: int = 3,
    y: float = 0.93,
    fontsize: float = 14.0,
    handlelength: float = 3.2,
    columnspacing: float = 1.9,
) -> None:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in np.asarray(axs, dtype=object).ravel():
        if ax is None:
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l, strict=False):
            if not label or label == "_nolegend_" or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=max(1, int(ncol)),
            frameon=False,
            fontsize=float(fontsize),
            columnspacing=float(columnspacing),
            handlelength=float(handlelength),
            borderaxespad=0.0,
        )


def _plot_windowed_curve(ax, sub: pd.DataFrame, *, color: str, label: str | None = None, lw: float = 1.45) -> float:
    one = sub.sort_values("wavelength_nm")
    if one.empty:
        return 0.0
    x = one["wavelength_nm"].to_numpy(dtype=float)
    y = one["intensity_ph_MeV_nm"].to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    ymax = float(np.nanmax(np.clip(finite, 0, None))) if finite.size else 0.0
    ax.plot(x, y, color=color, lw=float(lw), label=label)
    return ymax


def _draw_break_marks(ax_left, ax_right, d: float = 0.012) -> None:
    kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=0.9)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False, lw=0.9)
    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)


def plot_raw_mosaic(outdir: Path, gas: str, raw_out: pd.DataFrame, reference_raw: pd.DataFrame | None = None) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15.2, 10.4), sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()
    line_colors = colors(len(cfg.RAW_PRESSURES_BAR))

    y_source = raw_out
    if reference_raw is not None and not reference_raw.empty:
        y_source = pd.concat([raw_out, reference_raw], ignore_index=True)
    global_ylim = _common_ylim_from_df(y_source, "intensity_raw") if cfg.RAW_SHARE_YLIM else None

    for idx, ax in enumerate(axs):
        concentration = cfg.RAW_CONCENTRATIONS_PERCENT[idx]
        sub_c = raw_out[match_float(raw_out["concentration_percent"], concentration)]
        ymax = 0.0

        if reference_raw is not None and not reference_raw.empty:
            ref = reference_raw.sort_values("wavelength_nm")
            ref_x = ref["wavelength_nm"].to_numpy(dtype=float)
            ref_y = ref["intensity_raw"].to_numpy(dtype=float)
            finite = ref_y[np.isfinite(ref_y)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(np.clip(finite, 0, None))))
            ax.fill_between(
                ref_x,
                0.0,
                ref_y,
                color=cfg.RAW_REFERENCE_COLOR,
                alpha=float(cfg.RAW_REFERENCE_ALPHA),
                zorder=0,
                label=rf"Ar--CF$_4$ 95/5, {cfg.RAW_REFERENCE_PRESSURE_BAR:g} bar",
            )
            ax.plot(
                ref_x,
                ref_y,
                color=cfg.RAW_REFERENCE_COLOR,
                lw=0.9,
                alpha=0.55,
                zorder=1,
            )

        for color, pressure in zip(line_colors, cfg.RAW_PRESSURES_BAR, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)].sort_values("wavelength_nm")
            if sub.empty:
                continue
            y = sub["intensity_raw"].to_numpy(dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(np.clip(finite, 0, None))))
            ax.plot(sub["wavelength_nm"], y, color=color, lw=1.1, label=rf"{pressure:g} bar", zorder=3)

        ax.set_title(_panel_title(concentration))
        ax.set_xlim(*cfg.WAVELENGTH_RANGE_RAW_NM)
        ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel("raw intensity")

    fig.suptitle(_raw_suptitle(gas), y=0.972)
    _figure_legend(fig, axs, ncol=3, y=0.952, fontsize=14.0)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.075, top=0.865, wspace=0.17, hspace=0.27)
    pdf_path = outdir / "plots" / f"{gas}_raw_{cfg.RAW_PLOT_SPECTRUM_COLUMN}_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] raw PDF: {pdf_path}")
    return pdf_path


def plot_generated_mosaic(
    outdir: Path,
    gas: str,
    generated: pd.DataFrame,
    *,
    wavelength_range: tuple[float, float] | None = None,
    output_stem: str | None = None,
    title: str | None = None,
    log_y: bool = False,
    log_ymin: float | None = None,
    log_ymax_factor: float | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    total = generated[(generated["gas_mixture"] == gas) & (generated["component"] == "total")].copy()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15.2, 10.4), sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()
    line_colors = colors(len(cfg.GENERATED_PRESSURES_BAR))
    if log_y:
        ymin = float(log_ymin if log_ymin is not None else cfg.GENERATED_AMPLIED_LOG_YMIN)
        ymax_factor = float(log_ymax_factor if log_ymax_factor is not None else cfg.GENERATED_AMPLIED_LOG_YMAX_FACTOR)
        global_ylim = _common_log_ylim_from_df(total, "intensity_ph_MeV_nm", ymin, ymax_factor) if cfg.GENERATED_SHARE_YLIM else None
    else:
        ymin = 0.0
        ymax_factor = 1.12
        global_ylim = _common_ylim_from_df(total, "intensity_ph_MeV_nm") if cfg.GENERATED_SHARE_YLIM else None

    for idx, ax in enumerate(axs):
        if idx >= len(cfg.GENERATED_CONCENTRATIONS_PERCENT):
            ax.axis("off")
            continue
        concentration = cfg.GENERATED_CONCENTRATIONS_PERCENT[idx]
        sub_c = total[match_float(total["concentration_percent"], concentration)]
        ymax = 0.0
        for color, pressure in zip(line_colors, cfg.GENERATED_PRESSURES_BAR, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)].sort_values("wavelength_nm")
            if sub.empty:
                continue
            y = sub["intensity_ph_MeV_nm"].to_numpy(dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(np.clip(finite, 0, None))))
            plot_y = _positive_y_for_log(y, ymin) if log_y else y
            ax.plot(sub["wavelength_nm"], plot_y, color=color, lw=1.45, label=rf"{pressure:g} bar")
        ax.set_title(_panel_title(concentration))
        ax.set_xlim(*(wavelength_range or cfg.WAVELENGTH_RANGE_GENERATED[gas]))
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(*(global_ylim if global_ylim is not None else (ymin, max(ymin * 10.0, ymax * ymax_factor if ymax > 0 else 1.0))))
        else:
            ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")

    fig.suptitle(title or rf"{gas} generated spectra", y=0.985)
    _figure_legend(
        fig,
        axs,
        ncol=len(cfg.GENERATED_PRESSURES_BAR),
        y=0.958,
        fontsize=17.0,
        handlelength=3.6,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.075, top=0.84, wspace=0.17, hspace=0.27)
    stem = output_stem or f"{gas}_generated_concentrations"
    pdf_path = outdir / "plots" / f"{stem}_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] generated PDF: {pdf_path}")
    return pdf_path


def plot_generated_mosaic_with_inset(
    outdir: Path,
    gas: str,
    generated: pd.DataFrame,
    *,
    main_window: tuple[float, float] | None = None,
    main_ylim_window: tuple[float, float] | None = None,
    inset_window: tuple[float, float] | None = None,
    output_stem: str | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    setup_plot_style()
    total = generated[(generated["gas_mixture"] == gas) & (generated["component"] == "total")].copy()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15.4, 10.8), sharex=False, sharey=False)
    axs = np.asarray(axs).ravel()
    line_colors = colors(len(cfg.GENERATED_PRESSURES_BAR))
    main_window = tuple(main_window or cfg.GENERATED_AMPLIED_MAIN_WINDOW[gas])
    main_ylim_window = tuple(main_ylim_window or cfg.GENERATED_AMPLIED_MAIN_YLIM_WINDOW[gas])
    inset_window = tuple(inset_window or cfg.GENERATED_AMPLIED_VUV_WINDOW_NM)
    main_global_ymax = _positive_ymax_from_df_window(total, "intensity_ph_MeV_nm", window=main_ylim_window)
    inset_global_ymax = _positive_ymax_from_df_window(total, "intensity_ph_MeV_nm", window=inset_window)
    main_ylim = (0.0, float(cfg.GENERATED_AMPLIED_INSET_MAIN_YMAX_FACTOR) * main_global_ymax) if cfg.GENERATED_SHARE_YLIM and main_global_ymax > 0.0 else None
    inset_ylim = (0.0, float(cfg.GENERATED_AMPLIED_INSET_VUV_YMAX_FACTOR) * inset_global_ymax) if cfg.GENERATED_SHARE_YLIM and inset_global_ymax > 0.0 else None

    inset_axes_list: list[object] = []
    for idx, ax in enumerate(axs):
        if idx >= len(cfg.GENERATED_CONCENTRATIONS_PERCENT):
            ax.axis("off")
            continue
        concentration = cfg.GENERATED_CONCENTRATIONS_PERCENT[idx]
        sub_c = total[match_float(total["concentration_percent"], concentration)]
        panel_main_ymax = 0.0
        panel_inset_ymax = 0.0

        axins = inset_axes(
            ax,
            width=str(cfg.GENERATED_AMPLIED_INSET_WIDTH),
            height=str(cfg.GENERATED_AMPLIED_INSET_HEIGHT),
            loc=str(cfg.GENERATED_AMPLIED_INSET_LOC),
            borderpad=0.9,
        )
        inset_axes_list.append(axins)

        for color, pressure in zip(line_colors, cfg.GENERATED_PRESSURES_BAR, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)].sort_values("wavelength_nm")
            if sub.empty:
                continue
            sub_main = sub[_window_mask(sub["wavelength_nm"], main_window)]
            sub_inset = sub[_window_mask(sub["wavelength_nm"], inset_window)]
            panel_main_ymax = max(panel_main_ymax, _plot_windowed_curve(ax, sub_main, color=color, label=rf"{pressure:g} bar"))
            panel_inset_ymax = max(panel_inset_ymax, _plot_windowed_curve(axins, sub_inset, color=color))

        ax.set_title(_panel_title(concentration))
        ax.set_xlim(*main_window)
        ax.set_ylim(*(main_ylim if main_ylim is not None else (0.0, panel_main_ymax * 1.12 if panel_main_ymax > 0 else 1.0)))
        axins.set_xlim(*inset_window)
        axins.set_ylim(*(inset_ylim if inset_ylim is not None else (0.0, panel_inset_ymax * 1.12 if panel_inset_ymax > 0 else 1.0)))
        axins.set_title("VUV", fontsize=8, pad=1.5)
        axins.tick_params(axis="both", labelsize=7.5, pad=1.0)

        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")

    fig.suptitle(title or rf"{gas} generated spectra, extended VUV (main + inset)", y=0.985)
    _figure_legend(
        fig,
        axs,
        ncol=len(cfg.GENERATED_PRESSURES_BAR),
        y=0.958,
        fontsize=17.0,
        handlelength=3.6,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.075, top=0.84, wspace=0.17, hspace=0.27)
    stem = output_stem or f"{gas}_spectra_generated_amplied_inset"
    pdf_path = outdir / "plots" / f"{stem}_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] generated inset PDF: {pdf_path}")
    return pdf_path


def plot_generated_mosaic_brokenx(
    outdir: Path,
    gas: str,
    generated: pd.DataFrame,
    *,
    output_stem: str | None = None,
    title: str | None = None,
    left_window: tuple[float, float] | None = None,
    right_window: tuple[float, float] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    total = generated[(generated["gas_mixture"] == gas) & (generated["component"] == "total")].copy()
    nrows, ncols = 3, 3
    fig = plt.figure(figsize=(16.2, 10.9))
    outer = fig.add_gridspec(nrows, ncols)
    line_colors = colors(len(cfg.GENERATED_PRESSURES_BAR))
    left_window = tuple(left_window or cfg.GENERATED_AMPLIED_VUV_WINDOW_NM)
    right_window = tuple(right_window or cfg.GENERATED_AMPLIED_MAIN_WINDOW[gas])
    left_ylim = _common_ylim_from_df_window(total, "intensity_ph_MeV_nm", window=left_window) if cfg.GENERATED_SHARE_YLIM else None
    right_ylim = _common_ylim_from_df_window(total, "intensity_ph_MeV_nm", window=right_window) if cfg.GENERATED_SHARE_YLIM else None

    legend_axes: list[object] = []
    left_axes: list[object] = []
    right_axes: list[object] = []

    for idx, concentration in enumerate(cfg.GENERATED_CONCENTRATIONS_PERCENT):
        row, col = divmod(idx, ncols)
        inner = outer[row, col].subgridspec(
            1,
            2,
            width_ratios=tuple(cfg.GENERATED_AMPLIED_BROKENX_WIDTH_RATIOS),
            wspace=0.05,
        )
        ax_left = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])
        legend_axes.append(ax_right)
        left_axes.append(ax_left)
        right_axes.append(ax_right)

        sub_c = total[match_float(total["concentration_percent"], concentration)]
        panel_left_ymax = 0.0
        panel_right_ymax = 0.0
        for color, pressure in zip(line_colors, cfg.GENERATED_PRESSURES_BAR, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)].sort_values("wavelength_nm")
            if sub.empty:
                continue
            sub_left = sub[_window_mask(sub["wavelength_nm"], left_window)]
            sub_right = sub[_window_mask(sub["wavelength_nm"], right_window)]
            panel_left_ymax = max(panel_left_ymax, _plot_windowed_curve(ax_left, sub_left, color=color))
            panel_right_ymax = max(panel_right_ymax, _plot_windowed_curve(ax_right, sub_right, color=color, label=rf"{pressure:g} bar"))

        ax_left.set_xlim(*left_window)
        ax_right.set_xlim(*right_window)
        ax_left.set_ylim(*(left_ylim if left_ylim is not None else (0.0, panel_left_ymax * 1.12 if panel_left_ymax > 0 else 1.0)))
        ax_right.set_ylim(*(right_ylim if right_ylim is not None else (0.0, panel_right_ymax * 1.12 if panel_right_ymax > 0 else 1.0)))

        ax_right.set_title(_panel_title(concentration))
        ax_left.spines["right"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.tick_params(labelleft=False, left=False)
        _draw_break_marks(ax_left, ax_right)

        if row == nrows - 1:
            ax_left.set_xlabel(r"$\lambda$ [nm]")
            ax_right.set_xlabel(r"$\lambda$ [nm]")
        if col == 0:
            ax_left.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")

    fig.suptitle(title or rf"{gas} generated spectra, extended VUV (broken x axis)", y=0.985)
    _figure_legend(
        fig,
        legend_axes,
        ncol=len(cfg.GENERATED_PRESSURES_BAR),
        y=0.958,
        fontsize=17.0,
        handlelength=3.6,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.075, top=0.84, wspace=0.25, hspace=0.30)
    stem = output_stem or f"{gas}_spectra_generated_amplied_brokenx"
    pdf_path = outdir / "plots" / f"{stem}_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] generated broken-x PDF: {pdf_path}")
    return pdf_path


def plot_comparison_mosaic(outdir: Path, df: pd.DataFrame, spec: dict) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15.4, 10.6), sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()
    curve_colors = {
        ("ArCF4", 1): "tab:blue",
        ("ArCF4", 4): "tab:cyan",
        ("ArN2", 1): "tab:red",
        ("ArN2", 4): "tab:orange",
    }
    gas_labels = {"ArCF4": r"Ar--CF$_4$", "ArN2": r"Ar--N$_2$"}
    source_labels = {"raw": "raw", "generated": "prediction"}
    global_ylim = _common_ylim_from_df(df, "plot_intensity") if cfg.COMPARISON_SHARE_YLIM else None

    for idx, ax in enumerate(axs):
        if idx >= len(spec["concentrations_percent"]):
            ax.axis("off")
            continue
        concentration = spec["concentrations_percent"][idx]
        sub_c = df[match_float(df["concentration_percent"], concentration)]
        ymax = 0.0
        for gas in spec["gases"]:
            for pressure in spec["pressures_bar"]:
                for source in ("raw", "generated"):
                    one = sub_c[
                        (sub_c["gas_mixture"] == gas)
                        & match_float(sub_c["pressure_bar"], pressure)
                        & (sub_c["source"] == source)
                    ].sort_values("wavelength_nm")
                    if one.empty:
                        continue
                    y = one["plot_intensity"].to_numpy(dtype=float)
                    finite = y[np.isfinite(y)]
                    if finite.size:
                        ymax = max(ymax, float(np.nanmax(np.clip(finite, 0, None))))
                    label = rf"{gas_labels.get(gas, gas)}, {pressure:g} bar, {source_labels[source]}"
                    is_raw = source == "raw"
                    ax.plot(
                        one["wavelength_nm"],
                        y,
                        color=curve_colors.get((gas, int(pressure))),
                        linestyle="-",
                        lw=(float(cfg.COMPARISON_RAW_LINEWIDTH) if is_raw else float(cfg.COMPARISON_GENERATED_LINEWIDTH)),
                        alpha=(float(cfg.COMPARISON_RAW_ALPHA) if is_raw else float(cfg.COMPARISON_GENERATED_ALPHA)),
                        zorder=2 if is_raw else 4,
                        label=label,
                    )
        ax.set_title(_panel_title(concentration))
        ax.set_xlim(*cfg.WAVELENGTH_RANGE_COMPARISON_NM)
        ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ylabel = df["unit_label"].dropna().iloc[0] if "unit_label" in df.columns and not df.empty else "scaled intensity"
            ax.set_ylabel(ylabel)

    fig.suptitle(spec["title"], y=0.972)
    _figure_legend(fig, axs, ncol=4, y=0.952, fontsize=14.0, handlelength=3.5, columnspacing=2.2)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.075, top=0.86, wspace=0.17, hspace=0.27)
    pdf_path = outdir / "plots" / str(spec["output_pdf"])
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] comparison PDF: {pdf_path}")
    return pdf_path
