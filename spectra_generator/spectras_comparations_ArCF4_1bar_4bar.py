from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# scienceplots is nice to have, but the script should not die if it is absent.
try:
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    scienceplots = None


# -------------------------------------------------------------------------
# Robust path handling
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _prepend_existing_paths() -> None:
    """
    Add likely project/module folders to sys.path.

    This lets the script run from different folders as long as it is kept inside
    the repository, or inside a subfolder copied from the repository.
    """
    candidates: list[Path] = []

    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.extend(
            [
                parent,
                parent / "models",
                parent / "spectra_generator",
                parent / "spectra_experimental_annotated",
            ]
        )

    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_prepend_existing_paths()


from spectra_units import (  # noqa: E402
    get_arcf4_total_yield_ph_per_electron,
    get_spectrum_arrays,
    repo_root_from_script,
    safe_dill_load,
    setup_science_style,
    spectrum_shape_to_ph_per_MeV_nm,
)

from ArCF4_spectra import WAVELENGTH_NM as WAVELENGTH_CF4_NM  # noqa: E402
from ArCF4_spectra import arcf4_primary_spectrum_ph_per_MeV_nm  # noqa: E402


try:
    ROOT_DIR = repo_root_from_script(__file__)
except Exception:
    # Fallback: first parent containing data/Experimental or models.
    ROOT_DIR = SCRIPT_DIR
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (parent / "data" / "Experimental").exists() or (parent / "models").exists():
            ROOT_DIR = parent
            break


MODELS_DIR = ROOT_DIR / "models"
if MODELS_DIR.exists() and str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from ArCF4 import ion_potential  # noqa: E402


# -------------------------------------------------------------------------
# Output folders
# -------------------------------------------------------------------------

OUT_DIR = ROOT_DIR / "spectra_generator"
CSV_DIR = OUT_DIR / "spectra_csv" / "comparisons"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Style
# -------------------------------------------------------------------------

try:
    setup_science_style(use_grid=False)
except Exception:
    pass

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Four clean panels. You can add/remove concentrations here.
CONCENTRATIONS_PERCENT = [0.1, 1.0, 5.0, 100.0]

# Requested pressures.
PRESSURES_BAR = [1.0, 4.0]

# Reference used to normalise all experimental Ar-CF4 spectra.
REFERENCE_CF4_CONCENTRATION_PERCENT = 5.0  # Ar/CF4 = 95/5
REFERENCE_PRESSURE_BAR = 1.0
VISIBLE_RANGE_NM = (400.0, 800.0)

# Stored experimental spectrum columns.
CF4_SPECTRUM_COLUMNS = ("data(norm)",)

# Axes.
X_RANGE_NM = (200.0, 800.0)

# Output.
OUTPUT_PDF = OUT_DIR / "Comparation_ArCF4_ph_MeV_nm_1bar_4bar_solid_tones.pdf"
OUTPUT_CSV = CSV_DIR / "ArCF4_model_experiment_ph_MeV_nm_1bar_4bar_solid_tones.csv"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _data_root() -> Path:
    """
    Return the directory that contains Experimental/ArCF4/...

    You may override it with:
        export SPECTRA_DATA_DIR=/path/to/data
    or:
        export SPECTRA_DATA_DIR=/path/to/data/Experimental
    """
    env = os.environ.get("SPECTRA_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "ArCF4").exists():
            return p
        if (p / "Experimental").exists():
            return p / "Experimental"

    candidates = [
        ROOT_DIR / "data" / "Experimental",
        ROOT_DIR / "Experimental",
        SCRIPT_DIR / "data" / "Experimental",
        SCRIPT_DIR / "Experimental",
    ]

    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.extend(
            [
                parent / "data" / "Experimental",
                parent / "Experimental",
            ]
        )

    for candidate in candidates:
        if (candidate / "ArCF4").exists():
            return candidate

    # Keep the standard path in the error message if nothing was found.
    return ROOT_DIR / "data" / "Experimental"


def finite_max(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.nan
    return float(np.nanmax(y[mask]))


def format_concentration_label(c: float) -> str:
    if np.isclose(c, round(c)):
        return rf"{int(round(c))}\% CF$_4$"
    return rf"{c:g}\% CF$_4$"


def mix_with_white(color: str, amount: float = 0.58) -> tuple[float, float, float]:
    """
    Return a lighter version of a matplotlib color.

    amount = 0 keeps the original color.
    amount = 1 gives white.
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def darken_color(color: str, amount: float = 0.25) -> tuple[float, float, float]:
    """
    Return a darker version of a matplotlib color.

    amount = 0 keeps the original color.
    amount = 1 gives black.
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - amount))


def get_visible_scale_factor(
    wavelength_exp: np.ndarray,
    y_exp: np.ndarray,
    wavelength_model: np.ndarray,
    y_model: np.ndarray,
    visible_range_nm: tuple[float, float] = VISIBLE_RANGE_NM,
) -> float:
    """
    Compute one multiplicative factor for the experimental spectrum using
    the visible part only:

        scale * y_exp ~= y_model

    in least-squares sense.
    """
    wavelength_exp = np.asarray(wavelength_exp, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)
    wavelength_model = np.asarray(wavelength_model, dtype=float)
    y_model = np.asarray(y_model, dtype=float)

    wmin, wmax = visible_range_nm

    mask = (
        np.isfinite(wavelength_exp)
        & np.isfinite(y_exp)
        & (wavelength_exp >= wmin)
        & (wavelength_exp <= wmax)
        & (wavelength_exp >= np.nanmin(wavelength_model))
        & (wavelength_exp <= np.nanmax(wavelength_model))
    )

    if not np.any(mask):
        raise RuntimeError("No visible experimental points were found for scaling.")

    w_fit = wavelength_exp[mask]
    y_exp_fit = y_exp[mask]
    y_model_fit = np.interp(w_fit, wavelength_model, y_model)

    mask_fit = (
        np.isfinite(y_exp_fit)
        & np.isfinite(y_model_fit)
        & (y_exp_fit > 0.0)
        & (y_model_fit >= 0.0)
    )

    if not np.any(mask_fit):
        raise RuntimeError("The visible Ar-CF4 points are not usable for scaling.")

    y_exp_fit = y_exp_fit[mask_fit]
    y_model_fit = y_model_fit[mask_fit]

    denominator = np.sum(y_exp_fit**2)
    if denominator <= 0.0:
        raise RuntimeError("Invalid denominator while computing the scale factor.")

    scale = np.sum(y_exp_fit * y_model_fit) / denominator

    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Invalid scale factor obtained: {scale}")

    return float(scale)


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

    y = spectrum_shape_to_ph_per_MeV_nm(
        wave,
        raw,
        total_yield_ph_e,
        additive_fraction=concentration_percent / 100.0,
        w_func=ion_potential,
    )

    return wave, y


def compute_arcf4_95_5_visible_scale(df_cf4: pd.DataFrame) -> float:
    """
    Compute the global experimental normalisation factor from the visible
    Ar-CF4 spectrum at 95/5 and 1 bar.
    """
    exp_ref = get_cf4_experimental_spectrum(
        df_cf4,
        concentration_percent=REFERENCE_CF4_CONCENTRATION_PERCENT,
        pressure_bar=REFERENCE_PRESSURE_BAR,
    )

    if exp_ref is None:
        raise RuntimeError(
            "Could not find the reference Ar-CF4 95/5 experimental spectrum "
            f"at {REFERENCE_PRESSURE_BAR:g} bar."
        )

    wavelength_exp, y_exp = exp_ref

    y_model = arcf4_primary_spectrum_ph_per_MeV_nm(
        REFERENCE_CF4_CONCENTRATION_PERCENT,
        REFERENCE_PRESSURE_BAR,
    )

    return get_visible_scale_factor(
        wavelength_exp=wavelength_exp,
        y_exp=y_exp,
        wavelength_model=WAVELENGTH_CF4_NM,
        y_model=y_model,
        visible_range_nm=VISIBLE_RANGE_NM,
    )


def main() -> None:
    experimental_dir = _data_root()
    cf4_pkl = experimental_dir / "ArCF4" / "CF4_primary_data_final.pkl"

    if not cf4_pkl.exists():
        raise FileNotFoundError(
            "Could not find CF4_primary_data_final.pkl.\n"
            f"Tried: {cf4_pkl}\n"
            "Set SPECTRA_DATA_DIR to your data folder if needed, e.g.\n"
            "  export SPECTRA_DATA_DIR=/path/to/data"
        )

    df_cf4 = safe_dill_load(cf4_pkl)

    experimental_scale = compute_arcf4_95_5_visible_scale(df_cf4)

    print(
        "Using one global experimental scale factor from "
        f"Ar-CF4 95/5 visible spectrum at {REFERENCE_PRESSURE_BAR:g} bar:"
    )
    print(f"experimental_scale = {experimental_scale:.6g}")

    # One colour family per pressure:
    #   - light tone: experimental spectrum,
    #   - dark tone: predicted/model spectrum.
    pressure_base_colors = {
        1.0: "tab:blue",
        4.0: "tab:purple",
    }

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.0),
        sharex=True,
        sharey=False,
    )
    axs = axs.ravel()

    csv_rows: list[pd.DataFrame] = []

    for ax, con in zip(axs, CONCENTRATIONS_PERCENT):
        local_ymax = 0.0

        for pres in PRESSURES_BAR:
            base_color = pressure_base_colors.get(float(pres), "tab:gray")
            model_color = darken_color(base_color, amount=0.18)
            exp_color = mix_with_white(base_color, amount=0.58)

            # -------------------------
            # Ar-CF4 model
            # -------------------------
            y_cf4_model = arcf4_primary_spectrum_ph_per_MeV_nm(con, pres)
            local_ymax = max(local_ymax, finite_max(y_cf4_model))

            ax.plot(
                WAVELENGTH_CF4_NM,
                y_cf4_model,
                color=model_color,
                lw=2.45,
                ls="-",
                alpha=0.98,
                label=rf"predicted, {pres:g} bar",
                zorder=4,
            )

            csv_rows.append(
                pd.DataFrame(
                    {
                        "mixture": "ArCF4",
                        "kind": "model",
                        "concentration_percent": con,
                        "pressure_bar": pres,
                        "wavelength_nm": WAVELENGTH_CF4_NM,
                        "intensity_ph_MeV_nm": y_cf4_model,
                        "experimental_scale": 1.0,
                        "normalisation_reference": "none",
                    }
                )
            )

            # -------------------------
            # Ar-CF4 experiment
            # -------------------------
            exp_cf4 = get_cf4_experimental_spectrum(df_cf4, con, pres)

            if exp_cf4 is None:
                print(f"WARNING: no experimental Ar-CF4 spectrum for {con:g}% at {pres:g} bar.")
                continue

            w_cf4, y_cf4_exp = exp_cf4
            y_cf4_exp_scaled = y_cf4_exp * experimental_scale

            local_ymax = max(local_ymax, finite_max(y_cf4_exp_scaled))

            ax.plot(
                w_cf4,
                y_cf4_exp_scaled,
                color=exp_color,
                lw=2.05,
                ls="-",
                alpha=0.98,
                label=rf"experimental, {pres:g} bar",
                zorder=3,
            )

            csv_rows.append(
                pd.DataFrame(
                    {
                        "mixture": "ArCF4",
                        "kind": "experiment_scaled",
                        "concentration_percent": con,
                        "pressure_bar": pres,
                        "wavelength_nm": w_cf4,
                        "intensity_ph_MeV_nm": y_cf4_exp_scaled,
                        "experimental_scale": experimental_scale,
                        "normalisation_reference": "ArCF4_95_5_visible_1bar",
                    }
                )
            )

        ax.set_title(format_concentration_label(con))
        ax.set_xlim(*X_RANGE_NM)

        if local_ymax > 0.0 and np.isfinite(local_ymax):
            ax.set_ylim(0.0, 1.10 * local_ymax)

        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.grid(False)

        ax.legend(
            ncols=2,
            loc="upper right",
            fontsize=7.5,
            frameon=True,
            handlelength=2.0,
            columnspacing=0.7,
            borderpad=0.35,
            labelspacing=0.25,
        )

    for j in range(len(CONCENTRATIONS_PERCENT), len(axs)):
        axs[j].axis("off")

    fig.suptitle(
        r"Primary Ar--CF$_4$ spectra at 1 and 4 bar",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved comparison in {OUTPUT_PDF}")
    print(f"Saved comparison CSV in {OUTPUT_CSV}")
    print(
        "All experimental spectra were multiplied by the same factor obtained "
        "from the visible Ar-CF4 95/5 spectrum at 1 bar."
    )
    print("Model spectra are unchanged.")


if __name__ == "__main__":
    main()
