from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    scienceplots = None

plt.style.use("grid")

from spectra_units import (
    get_arcf4_total_yield_ph_per_electron,
    get_spectrum_arrays,
    repo_root_from_script,
    safe_dill_load,
    setup_science_style,
)

ROOT_DIR = repo_root_from_script(__file__)
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "spectra_experimental"

sys.path.insert(0, str(MODELS_DIR))

setup_science_style(use_grid=False)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

CONCENTRATIONS_PERCENT = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
PRESSURES_BAR = [1.0, 2.0, 3.0, 4.0, 5.0]

# Spectrum columns
CF4_SPECTRUM_COLUMNS = ("data(norm)",)

# Wavelength range shown
X_RANGE_NM = (180.0, 800.0)

# Plot appearance
FIGSIZE = (18.0, 8.0)
NROWS = 3
NCOLS = 3

# If True, all subplots share the same y limit
USE_GLOBAL_YLIM = True

# Output
OUTPUT_PDF = OUT_DIR / "experimental_ArCF4_grid.pdf"


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


def compute_global_ymax(df_cf4: pd.DataFrame) -> float:
    ymax = 0.0

    for concentration_percent in CONCENTRATIONS_PERCENT:
        for pressure_bar in PRESSURES_BAR:
            spec = get_cf4_experimental_spectrum(
                df_cf4,
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )
            if spec is None:
                continue

            _, y_cf4 = spec
            if np.any(np.isfinite(y_cf4)):
                ymax = max(ymax, float(np.nanmax(y_cf4)))

    return ymax


def format_concentration_label(c: float) -> str:
    if np.isclose(c, round(c)):
        return f"{int(round(c))}%"
    return f"{c:g}%"


def main() -> None:
    df_cf4 = safe_dill_load(
        DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl"
    )

    norm_max = compute_global_ymax(df_cf4)
    if not np.isfinite(norm_max) or norm_max <= 0.0:
        raise RuntimeError("Could not compute a valid normalization from Ar/CF4 spectra.")

    # Shared y-limit
    global_ymax = None
    if USE_GLOBAL_YLIM:
        global_ymax = compute_global_ymax(df_cf4)
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

        # Plot Ar-CF4 curves for all pressures
        for pressure_bar, color in zip(PRESSURES_BAR, colors):
            spec_cf4 = get_cf4_experimental_spectrum(
                df_cf4,
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )

            if spec_cf4 is None:
                continue

            w_cf4, y_cf4 = spec_cf4

            ax.plot(
                w_cf4,
                y_cf4 / norm_max,
                lw=1.8,
                color=color,
                alpha=0.85,
                label=rf"{pressure_bar:g} bar",
                zorder=3,
            )

        ax.set_title(format_concentration_label(concentration_percent), fontsize=17)
        ax.set_xlim(*X_RANGE_NM)
        ax.grid(False)

        # Keep the same visual convention as the reference script: the global
        # y-limit is computed but not forced.
        # if global_ymax is not None:
        #     ax.set_ylim(0.0, 1.12 * global_ymax / norm_max)

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
