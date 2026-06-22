from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spectra_types import ComparisonMosaicConfig, RawSpectraConfig
from .spectra_units import first_finite_max, match_float, setup_science_style


def _colors(n: int):
    return plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(n, 2)))


def _finite_minmax(*arrays) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        finite = a[np.isfinite(a)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0
    both = np.concatenate(vals)
    return float(np.nanmin(both)), float(np.nanmax(both))


def _one_spectrum(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    first = df["spectrum_name"].iloc[0]
    return df[df["spectrum_name"] == first].sort_values("wavelength_nm").copy()


def _raw_ylim(df: pd.DataFrame, reference_df: pd.DataFrame | None, config: RawSpectraConfig) -> tuple[float, float]:
    arrays = [df["intensity_raw"].to_numpy(dtype=float)] if not df.empty else []
    if reference_df is not None and not reference_df.empty:
        arrays.append(reference_df["intensity_raw"].to_numpy(dtype=float))
    ymin, ymax = _finite_minmax(*arrays)
    if config.gas_mixture == "ArCF4":
        ymin = 0.0
    span = ymax - ymin
    if span <= 0:
        span = abs(ymax) if ymax else 1.0
    return ymin - (0.04 * span if ymin < 0 else 0.0), ymax + 0.08 * span


def _legend_if_needed(ax, *, fontsize: float = 7.0, ncols: int = 1, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    labels = [label for label in labels if label and not label.startswith("_")]
    if labels:
        ax.legend(fontsize=fontsize, loc=loc, frameon=True, ncol=ncols)


def plot_raw_spectra(df: pd.DataFrame, config: RawSpectraConfig, reference_df: pd.DataFrame | None = None) -> None:
    setup_science_style(use_grid=False)
    config.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    concentrations = list(config.concentrations_percent or sorted(df["concentration_percent"].dropna().unique()))
    pressures = list(config.pressures_bar or sorted(df["pressure_bar"].dropna().unique()))
    colors = _colors(len(pressures))

    nrows, ncols = config.mosaic_shape
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=config.figsize,
        sharex=True,
        sharey=config.share_y,
    )
    axs = np.asarray(axs).ravel()

    reference_df = _one_spectrum(reference_df)
    ylimits = _raw_ylim(df, reference_df, config) if config.share_y else None

    for idx, ax in enumerate(axs):
        if idx >= len(concentrations):
            ax.axis("off")
            continue

        concentration = concentrations[idx]
        sub_c = df[match_float(df["concentration_percent"], concentration)]

        for color, pressure in zip(colors, pressures, strict=False):
            sub_p = sub_c[match_float(sub_c["pressure_bar"], pressure)]
            if sub_p.empty:
                continue
            for j, (_spectrum_name, one) in enumerate(sub_p.groupby("spectrum_name", sort=False)):
                one = one.sort_values("wavelength_nm")
                ax.plot(
                    one["wavelength_nm"],
                    one["intensity_raw"],
                    color=color,
                    lw=0.78,
                    alpha=0.82,
                    solid_capstyle="round",
                    label=rf"{pressure:g} bar" if j == 0 else None,
                )

        if reference_df is not None and config.reference is not None:
            ref = config.reference
            if ref.fill:
                ax.fill_between(
                    reference_df["wavelength_nm"],
                    reference_df["intensity_raw"],
                    np.zeros(len(reference_df)),
                    color=ref.color,
                    alpha=ref.alpha,
                    linewidth=0.0,
                    label=ref.label,
                )
            ax.plot(
                reference_df["wavelength_nm"],
                reference_df["intensity_raw"],
                color=ref.color,
                lw=ref.linewidth,
                alpha=min(1.0, ref.alpha + 0.20),
            )

        title = rf"{concentration:g}\%" if config.show_percent_in_titles else rf"{concentration:g}"
        ax.set_title(title)
        ax.set_xlim(*config.wavelength_range_nm)
        if ylimits is not None:
            ax.set_ylim(*ylimits)
        elif not sub_c.empty:
            ymin, ymax = _raw_ylim(sub_c, reference_df, config)
            ax.set_ylim(ymin, ymax)
        ax.grid(False)
        _legend_if_needed(ax, fontsize=6.7, loc="upper right")

    for idx, ax in enumerate(axs[: len(concentrations)]):
        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.set_xlabel(config.xlabel)
        if col == 0:
            ax.set_ylabel(config.ylabel)

    if config.title:
        fig.suptitle(config.title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(config.output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_generated_spectra(
    df: pd.DataFrame,
    output_pdf: Path,
    output_summary_pdf: Path,
    title: str,
    wavelength_range_nm: tuple[float, float],
) -> None:
    setup_science_style(use_grid=False)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_summary_pdf.parent.mkdir(parents=True, exist_ok=True)

    total = df[df["component"] == "total"]
    pressures = sorted(total["pressure_bar"].unique())
    concentrations = sorted(total["concentration_percent"].unique())
    colors_con = _colors(len(concentrations))
    colors_pressure = _colors(len(pressures))

    global_ymax = first_finite_max(total["intensity_ph_MeV_nm"].to_numpy())

    for pressure in pressures:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        sub_p = total[match_float(total["pressure_bar"], pressure)]
        for color, concentration in zip(colors_con, concentrations, strict=False):
            sub = sub_p[match_float(sub_p["concentration_percent"], concentration)]
            if sub.empty:
                continue
            ax.plot(
                sub["wavelength_nm"],
                sub["intensity_ph_MeV_nm"],
                color=color,
                lw=2.0,
                label=rf"{concentration:g}\%",
            )
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_title(rf"{title}, {pressure:g} bar")
        ax.set_xlim(*wavelength_range_nm)
        ax.set_ylim(bottom=0)
        _legend_if_needed(ax, fontsize=8, ncols=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(output_pdf.parent / f"{output_pdf.stem}_{pressure:g}bar.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, concentration in zip(axs, concentrations, strict=False):
        sub_c = total[match_float(total["concentration_percent"], concentration)]
        for color, pressure in zip(colors_pressure, pressures, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)]
            if sub.empty:
                continue
            ax.plot(
                sub["wavelength_nm"],
                sub["intensity_ph_MeV_nm"],
                color=color,
                lw=1.5,
                label=rf"{pressure:g} bar",
            )
        ax.set_title(rf"{concentration:g}\%")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(*wavelength_range_nm)
        ax.set_ylim(0, 1.05 * global_ymax)
        _legend_if_needed(ax, fontsize=8, ncols=2, loc="upper left")

    for ax in axs[len(concentrations) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_summary_pdf, bbox_inches="tight")
    plt.close(fig)


def _comparison_title(config: ComparisonMosaicConfig, concentration: float) -> str:
    if config.name.startswith("ArCF4"):
        return rf"{concentration:g}\% CF$_4$"
    return rf"{concentration:g}\% additive"


def _robust_positive_ylim(lines_y: list[np.ndarray]) -> tuple[float, float]:
    vals = []
    for y in lines_y:
        arr = np.asarray(y, dtype=float)
        finite = arr[np.isfinite(arr)]
        finite = finite[finite >= 0]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    ymax = float(np.nanmax(allv))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    return 0.0, ymax * 1.12


def plot_comparison_mosaic(df: pd.DataFrame, config: ComparisonMosaicConfig) -> None:
    setup_science_style(use_grid=False)
    config.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    nrows, ncols = config.mosaic_shape
    fig, axs = plt.subplots(nrows, ncols, figsize=config.figsize, sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()

    for idx, ax in enumerate(axs):
        if idx >= len(config.concentrations_percent):
            ax.axis("off")
            continue

        concentration = config.concentrations_percent[idx]
        sub_c = df[match_float(df["concentration_percent"], concentration)]
        for curve in config.curves:
            one = sub_c[sub_c["curve_name"] == curve.name].sort_values("wavelength_nm")
            if one.empty:
                continue
            ax.plot(
                one["wavelength_nm"],
                one["plot_intensity"],
                color=curve.color,
                lw=curve.linewidth,
                ls=curve.linestyle,
                alpha=curve.alpha,
                label=curve.label,
            )

        ax.set_title(_comparison_title(config, concentration))
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_xlim(*config.wavelength_range_nm)
        if config.ylim is not None:
            ax.set_ylim(*config.ylim)
        else:
            ax.set_ylim(*_robust_positive_ylim([line.get_ydata() for line in ax.lines]))
        ax.grid(False)
        _legend_if_needed(ax, fontsize=7.0, ncols=2, loc="upper right")

    if config.title:
        fig.suptitle(config.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(config.output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, output_pdf: Path, title: str, wavelength_range_nm: tuple[float, float]) -> None:
    # Wrapper legacy: reconstruye un mosaico básico usando las curvas presentes.
    curves = sorted(df["curve_name"].unique()) if "curve_name" in df.columns else []
    setup_science_style(use_grid=False)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    concentrations = sorted(df["concentration_percent"].unique())
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()
    colors = _colors(max(len(curves), 2))

    for ax, concentration in zip(axs, concentrations, strict=False):
        sub_c = df[match_float(df["concentration_percent"], concentration)]
        for color, curve_name in zip(colors, curves, strict=False):
            one = sub_c[sub_c["curve_name"] == curve_name].sort_values("wavelength_nm")
            if one.empty:
                continue
            ax.plot(one["wavelength_nm"], one["plot_intensity"], color=color, lw=1.8, label=curve_name)
        ax.set_title(rf"{concentration:g}\%")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(*wavelength_range_nm)
        ax.set_ylim(*_robust_positive_ylim([line.get_ydata() for line in ax.lines]))
        _legend_if_needed(ax, fontsize=7)

    for ax in axs[len(concentrations) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
