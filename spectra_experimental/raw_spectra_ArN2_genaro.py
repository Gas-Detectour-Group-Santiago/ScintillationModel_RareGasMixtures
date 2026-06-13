from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use("grid")

from spectra_units import (
    get_arcf4_total_yield_ph_per_electron,
    get_n2_total_yield_ph_per_electron,
    get_spectrum_arrays,
    repo_root_from_script,
    safe_dill_load,
    setup_science_style,
)

ROOT_DIR = repo_root_from_script(__file__)
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "spectra_generator"

sys.path.insert(0, str(MODELS_DIR))

setup_science_style(use_grid=False)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

CONCENTRATIONS_PERCENT = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
PRESSURES_BAR = [1.0, 2.0, 3.0, 4.0, 5.0]

# Fixed Ar/CF4 reference
ARCF4_REFERENCE_CONCENTRATION_PERCENT = 5.0   # 95/5
ARCF4_REFERENCE_PRESSURE_BAR = 1.0

# Spectrum columns
N2_SPECTRUM_COLUMNS = ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal")
CF4_SPECTRUM_COLUMNS = ("data(norm)",)

# Wavelength range shown
X_RANGE_NM = (180.0, 800.0)

# Plot appearance
FIGSIZE = (18.0, 8.0)
NROWS = 3
NCOLS = 3

# If True, all subplots share the same y limit
USE_GLOBAL_YLIM = True

# Fill style for Ar/CF4 reference
ARCF4_FILL_ALPHA = 0.35
ARCF4_FILL_COLOR = "magenta"

# Output
OUTPUT_PDF = OUT_DIR / "experimental_ArN2_grid_with_fixed_ArCF4_95_5_reference.pdf"


def spectrum_shape_to_ph_per_e_nm(
    wavelength: np.ndarray,
    raw_intensity: np.ndarray,
    total_yield_ph_per_electron: float,
) -> np.ndarray:
    """
    Convert a raw experimental spectrum shape into ph/e-/nm.

    The raw spectrum is treated only as a shape:
      1. clip negative values,
      2. normalize to unit area,
      3. rescale to the integrated experimental yield in ph/e-.

    No W-value conversion is applied here.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    raw_intensity = np.asarray(raw_intensity, dtype=float)

    finite = np.isfinite(wavelength) & np.isfinite(raw_intensity)
    if finite.sum() < 2:
        return np.full_like(raw_intensity, np.nan, dtype=float)

    wave = wavelength[finite]
    inten = np.clip(raw_intensity[finite], 0.0, None)

    area = np.trapezoid(inten, wave)
    if area <= 0.0 or not np.isfinite(area):
        return np.full_like(raw_intensity, np.nan, dtype=float)

    out = np.full_like(raw_intensity, np.nan, dtype=float)
    out[finite] = inten * total_yield_ph_per_electron / area
    return out


def get_cf4_experimental_spectrum(
    df_cf4: pd.DataFrame,
    concentration_percent: float,
    pressure_bar: float,
):
    mask = np.isclose(
        df_cf4["concentracion"].astype(float),
        concentration_percent,
    ) & np.isclose(
        df_cf4["presion"].astype(float),
        pressure_bar,
    )

    if not np.any(mask):
        return None

    row = df_cf4.loc[mask].iloc[0]

    wave, raw = get_spectrum_arrays(row, CF4_SPECTRUM_COLUMNS)
    total_yield_ph_e = get_arcf4_total_yield_ph_per_electron(row)

    y = spectrum_shape_to_ph_per_e_nm(
        wave,
        raw,
        total_yield_ph_e,
    )

    return wave, y


def get_n2_experimental_spectrum(
    df_n2: pd.DataFrame,
    concentration_percent: float,
    pressure_bar: float,
):
    mask = np.isclose(
        df_n2["N2 concentration (%)"].astype(float),
        concentration_percent,
    ) & np.isclose(
        df_n2["P (bar)"].astype(float),
        pressure_bar,
    )

    if not np.any(mask):
        return None

    row = df_n2.loc[mask].iloc[0]

    wave, raw = get_spectrum_arrays(row, N2_SPECTRUM_COLUMNS)

    # include_ir=True to keep the full experimental spectrum (UV + IR features)
    total_yield_ph_e = get_n2_total_yield_ph_per_electron(row, include_ir=True)

    y = spectrum_shape_to_ph_per_e_nm(
        wave,
        raw,
        total_yield_ph_e,
    )

    return wave, y


def compute_global_ymax(
    df_n2: pd.DataFrame,
    df_cf4: pd.DataFrame,
) -> float:
    ymax = 0.0

    # Fixed Ar/CF4 reference
    ref_cf4 = get_cf4_experimental_spectrum(
        df_cf4,
        concentration_percent=ARCF4_REFERENCE_CONCENTRATION_PERCENT,
        pressure_bar=ARCF4_REFERENCE_PRESSURE_BAR,
    )
    if ref_cf4 is not None:
        _, y_cf4 = ref_cf4
        if np.any(np.isfinite(y_cf4)):
            ymax = max(ymax, float(np.nanmax(y_cf4)))

    # All Ar-N2 curves used in the grid
    for concentration_percent in CONCENTRATIONS_PERCENT:
        for pressure_bar in PRESSURES_BAR:
            spec = get_n2_experimental_spectrum(
                df_n2,
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )
            if spec is None:
                continue
            _, y_n2 = spec
            if np.any(np.isfinite(y_n2)):
                ymax = max(ymax, float(np.nanmax(y_n2)))

    return ymax


def format_concentration_label(c: float) -> str:
    if np.isclose(c, round(c)):
        return f"{int(round(c))}%"
    return f"{c:g}%"


def main() -> None:
    df_n2 = safe_dill_load(
        DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl"
    )
    df_cf4 = safe_dill_load(
        DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl"
    )

    # Fixed Ar/CF4 reference
    ref_cf4 = get_cf4_experimental_spectrum(
        df_cf4,
        concentration_percent=ARCF4_REFERENCE_CONCENTRATION_PERCENT,
        pressure_bar=ARCF4_REFERENCE_PRESSURE_BAR,
    )

    if ref_cf4 is None:
        raise RuntimeError(
            "Could not find the fixed Ar/CF4 95/5 reference spectrum at 1 bar."
        )

    w_cf4_ref, y_cf4_ref = ref_cf4

    norm_max = max(y_cf4_ref)

    # Shared y-limit
    global_ymax = None
    if USE_GLOBAL_YLIM:
        global_ymax = compute_global_ymax(df_n2, df_cf4)
        if not np.isfinite(global_ymax) or global_ymax <= 0.0:
            global_ymax = None

    # Pressure colors
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(PRESSURES_BAR)))

    fig, axs = plt.subplots(
        NROWS,
        NCOLS,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )
    axs = axs.ravel()

    for i, concentration_percent in enumerate(CONCENTRATIONS_PERCENT):
        ax = axs[i]

        # Plot Ar-N2 curves for all pressures
        for pressure_bar, color in zip(PRESSURES_BAR, colors):
            spec_n2 = get_n2_experimental_spectrum(
                df_n2,
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )

            if spec_n2 is None:
                continue

            w_n2, y_n2 = spec_n2

            ax.plot(
                w_n2,
                y_n2/norm_max,
                lw=1.8,
                color=color,
                alpha=0.85,
                label=rf"{pressure_bar:g} bar",
                zorder=3,
            )

        # Fixed Ar/CF4 95/5 reference
        ax.fill_between(
            w_cf4_ref,
            0.0,
            y_cf4_ref/norm_max,
            color=ARCF4_FILL_COLOR,
            alpha=ARCF4_FILL_ALPHA,
            label=r"Ar/CF$_4$ 95/5",
            zorder=1,
        )

        ax.plot(
            w_cf4_ref,
            y_cf4_ref/norm_max,
            color=ARCF4_FILL_COLOR,
            lw=1.4,
            alpha=0.9,
            zorder=2,
        )

        ax.set_title(format_concentration_label(concentration_percent), fontsize=17)
        ax.set_xlim(*X_RANGE_NM)
        ax.grid(False)

        # if global_ymax is not None:
        #     ax.set_ylim(0.0, 1.12 * global_ymax)

        if i % NCOLS == 0:
            ax.set_ylabel(r"$y$ / $e^-$ (A.U)", fontsize=16)

        if i >= (NROWS - 1) * NCOLS:
            ax.set_xlabel("Wavelength (nm)", fontsize=16)

        ax.tick_params(axis="both", labelsize=12)

        ax.legend(
            fontsize=8,
            loc="upper center",
            frameon=True,
            ncols=1,
            handlelength=1.5,
            borderpad=0.3,
            labelspacing=0.25,
        )

    # If there are unused axes, turn them off
    for j in range(len(CONCENTRATIONS_PERCENT), len(axs)):
        axs[j].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()