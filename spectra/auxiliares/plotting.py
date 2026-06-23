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



def plot_raw_mosaic(outdir: Path, gas: str, raw_out: pd.DataFrame, reference_raw: pd.DataFrame | None = None) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(12.0, 8.0), sharex=True, sharey=False)
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

        for color, pressure in zip(line_colors, cfg.RAW_PRESSURES_BAR, strict=False):
            sub = sub_c[match_float(sub_c["pressure_bar"], pressure)].sort_values("wavelength_nm")
            if sub.empty:
                continue
            y = sub["intensity_raw"].to_numpy(dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(np.clip(finite, 0, None))))
            ax.plot(sub["wavelength_nm"], y, color=color, lw=1.05, label=rf"{pressure:g} bar", zorder=3)

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
                zorder=4,
                label=rf"Ar--CF$_4$ 95/5, {cfg.RAW_REFERENCE_PRESSURE_BAR:g} bar",
            )
            ax.plot(
                ref_x,
                ref_y,
                color=cfg.RAW_REFERENCE_COLOR,
                lw=1.15,
                alpha=0.95,
                zorder=5,
            )

        ax.set_title(rf"{concentration:g}\% additive")
        ax.set_xlim(*cfg.WAVELENGTH_RANGE_RAW_NM)
        ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        ax.legend(fontsize=6.2, loc="upper right")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel("raw intensity")

    fig.suptitle(rf"{gas} raw {cfg.RAW_PLOT_SPECTRUM_COLUMN}")
    fig.tight_layout()
    pdf_path = outdir / "plots" / f"{gas}_raw_{cfg.RAW_PLOT_SPECTRUM_COLUMN}_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] raw PDF: {pdf_path}")
    return pdf_path

def plot_generated_mosaic(outdir: Path, gas: str, generated: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    total = generated[(generated["gas_mixture"] == gas) & (generated["component"] == "total")].copy()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(12.0, 8.0), sharex=True, sharey=False)
    axs = np.asarray(axs).ravel()
    line_colors = colors(len(cfg.GENERATED_PRESSURES_BAR))
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
            ax.plot(sub["wavelength_nm"], y, color=color, lw=1.45, label=rf"{pressure:g} bar")
        ax.set_title(rf"{concentration:g}\% additive")
        ax.set_xlim(*cfg.WAVELENGTH_RANGE_GENERATED[gas])
        ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        ax.legend(fontsize=6.5, loc="upper right", ncols=2)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")

    fig.suptitle(rf"{gas} generated spectra")
    fig.tight_layout()
    pdf_path = outdir / "plots" / f"{gas}_generated_concentrations_mosaic_3x3.pdf"
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] generated PDF: {pdf_path}")
    return pdf_path

def plot_comparison_mosaic(outdir: Path, df: pd.DataFrame, spec: dict) -> Path:
    import matplotlib.pyplot as plt

    setup_plot_style()
    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(12.2, 8.2), sharex=True, sharey=False)
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
                # Draw raw first, thick and translucent, then the prediction as a
                # finer solid curve on top.  No dashed curves are used.
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
                        lw=(
                            float(cfg.COMPARISON_RAW_LINEWIDTH)
                            if is_raw
                            else float(cfg.COMPARISON_GENERATED_LINEWIDTH)
                        ),
                        alpha=(
                            float(cfg.COMPARISON_RAW_ALPHA)
                            if is_raw
                            else float(cfg.COMPARISON_GENERATED_ALPHA)
                        ),
                        zorder=2 if is_raw else 4,
                        label=label,
                    )
        ax.set_title(rf"{concentration:g}\% additive")
        ax.set_xlim(*cfg.WAVELENGTH_RANGE_COMPARISON_NM)
        ax.set_ylim(*(global_ylim if global_ylim is not None else (0, ymax * 1.12 if ymax > 0 else 1.0)))
        ax.legend(fontsize=5.8, loc="upper right")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\lambda$ [nm]")
        if idx % ncols == 0:
            ylabel = df["unit_label"].dropna().iloc[0] if "unit_label" in df.columns and not df.empty else "scaled intensity"
            ax.set_ylabel(ylabel)

    fig.suptitle(spec["title"])
    fig.tight_layout()
    pdf_path = outdir / "plots" / str(spec["output_pdf"])
    ensure_parent(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] comparison PDF: {pdf_path}")
    return pdf_path
