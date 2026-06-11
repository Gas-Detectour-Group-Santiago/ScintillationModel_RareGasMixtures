from __future__ import annotations

import sys

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
    spectrum_shape_to_ph_per_MeV_nm,
)
from ArCF4_spectra import WAVELENGTH_NM as WAVELENGTH_CF4_NM
from ArCF4_spectra import arcf4_primary_spectrum_ph_per_MeV_nm
from ArN2_spectra import WAVELENGTH_NM as WAVELENGTH_N2_NM
from ArN2_spectra import arn2_primary_spectrum_ph_per_MeV_nm

ROOT_DIR = repo_root_from_script(__file__)
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "spectra_generator"
CSV_DIR = ROOT_DIR / "spectra_generator" / "spectra_csv" / "comparisons"

sys.path.insert(0, str(MODELS_DIR))

from ArCF4 import ion_potential  # noqa: E402
from ArN2 import W_ArN2  # noqa: E402

setup_science_style(use_grid=False)
CSV_DIR.mkdir(parents=True, exist_ok=True)

PRESSURES_BAR = [1]
CONCENTRATIONS_PERCENT = [0.1, 1, 5, 100]

# Reference used to normalise all experimental spectra.
REFERENCE_CF4_CONCENTRATION_PERCENT = 5.0  # Ar/CF4 = 95/5
REFERENCE_PRESSURE_BAR = 1.0
VISIBLE_RANGE_NM = (400.0, 800.0)

# The stored spectra are used as shapes and are renormalised to the integrated
# ph/e- yields before applying 1e6/W(f).
N2_SPECTRUM_COLUMNS = ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal")
CF4_SPECTRUM_COLUMNS = ("data(norm)",)


def finite_max(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)

    if not np.any(mask):
        return np.nan

    return float(np.nanmax(y[mask]))


def get_visible_scale_factor(
    wavelength_exp: np.ndarray,
    y_exp: np.ndarray,
    wavelength_model: np.ndarray,
    y_model: np.ndarray,
    visible_range_nm: tuple[float, float] = VISIBLE_RANGE_NM,
) -> float:
    """
    Compute one multiplicative factor for the experimental spectrum using
    the visible part only.

    The factor is obtained by least squares:

        scale * y_exp ~= y_model

    Therefore:

        scale = sum(y_exp * y_model) / sum(y_exp^2)

    The model is interpolated at the experimental wavelengths.
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
        raise RuntimeError(
            "No visible experimental points were found to compute the "
            "Ar-CF4 95/5 normalisation factor."
        )

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
        raise RuntimeError(
            "The visible Ar-CF4 95/5 points are not usable for normalisation."
        )

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

    total_yield_ph_e = get_n2_total_yield_ph_per_electron(row, include_ir=True)

    y = spectrum_shape_to_ph_per_MeV_nm(
        wave,
        raw,
        total_yield_ph_e,
        additive_fraction=concentration_percent / 100.0,
        w_func=W_ArN2,
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

    scale = get_visible_scale_factor(
        wavelength_exp=wavelength_exp,
        y_exp=y_exp,
        wavelength_model=WAVELENGTH_CF4_NM,
        y_model=y_model,
        visible_range_nm=VISIBLE_RANGE_NM,
    )

    return scale


def main() -> None:
    df_n2 = safe_dill_load(DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl")
    df_cf4 = safe_dill_load(DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl")

    experimental_scale = compute_arcf4_95_5_visible_scale(df_cf4)

    print(
        "Using one global experimental scale factor from "
        f"Ar-CF4 95/5 visible spectrum at {REFERENCE_PRESSURE_BAR:g} bar:"
    )
    print(f"experimental_scale = {experimental_scale:.6g}")

    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=False)
    axs = axs.ravel()

    csv_rows = []

    for ax, con in zip(axs, CONCENTRATIONS_PERCENT):
        local_ymax = 0.0

        for pres in PRESSURES_BAR:
            # -------------------------
            # Ar-N2: model + experiment
            # -------------------------
            y_n2_model = arn2_primary_spectrum_ph_per_MeV_nm(con, pres)

            local_ymax = max(local_ymax, finite_max(y_n2_model))

            ax.plot(
                WAVELENGTH_N2_NM,
                y_n2_model,
                color="tab:red",
                lw=2.2,
                label=rf"Ar--N$_2$ model, {pres:g} bar",
            )

            csv_rows.append(
                pd.DataFrame(
                    {
                        "mixture": "ArN2",
                        "kind": "model",
                        "concentration_percent": con,
                        "pressure_bar": pres,
                        "wavelength_nm": WAVELENGTH_N2_NM,
                        "intensity_ph_MeV_nm": y_n2_model,
                        "experimental_scale": 1.0,
                        "normalisation_reference": "none",
                    }
                )
            )

            exp_n2 = get_n2_experimental_spectrum(df_n2, con, pres)

            if exp_n2 is not None:
                w_n2, y_n2_exp = exp_n2

                y_n2_exp_scaled = y_n2_exp * experimental_scale

                local_ymax = max(local_ymax, finite_max(y_n2_exp_scaled))

                ax.plot(
                    w_n2,
                    y_n2_exp_scaled,
                    color="tab:green",
                    lw=1.8,
                    alpha=0.9,
                    label=rf"Ar--N$_2$ exp. scaled, {pres:g} bar",
                )

                csv_rows.append(
                    pd.DataFrame(
                        {
                            "mixture": "ArN2",
                            "kind": "experiment_scaled",
                            "concentration_percent": con,
                            "pressure_bar": pres,
                            "wavelength_nm": w_n2,
                            "intensity_ph_MeV_nm": y_n2_exp_scaled,
                            "experimental_scale": experimental_scale,
                            "normalisation_reference": "ArCF4_95_5_visible_1bar",
                        }
                    )
                )

            # -------------------------
            # Ar-CF4: model + experiment
            # -------------------------
            y_cf4_model = arcf4_primary_spectrum_ph_per_MeV_nm(con, pres)

            local_ymax = max(local_ymax, finite_max(y_cf4_model))

            ax.plot(
                WAVELENGTH_CF4_NM,
                y_cf4_model,
                color="tab:blue",
                lw=2.2,
                label=rf"Ar--CF$_4$ model, {pres:g} bar",
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

            exp_cf4 = get_cf4_experimental_spectrum(df_cf4, con, pres)

            if exp_cf4 is not None:
                w_cf4, y_cf4_exp = exp_cf4

                y_cf4_exp_scaled = y_cf4_exp * experimental_scale

                local_ymax = max(local_ymax, finite_max(y_cf4_exp_scaled))

                ax.plot(
                    w_cf4,
                    y_cf4_exp_scaled,
                    color="tab:orange",
                    lw=1.8,
                    alpha=0.9,
                    label=rf"Ar--CF$_4$ exp. scaled, {pres:g} bar",
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

        ax.set_title(rf"{con:g}\% additive")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(200, 800)

        if local_ymax > 0.0 and np.isfinite(local_ymax):
            ax.set_ylim(0, 1.08 * local_ymax)

        ax.grid(False)
        ax.legend(ncols=2, loc="upper right", fontsize=7)

    fig.suptitle(
        r"Primary Ar--N$_2$ and Ar--CF$_4$ spectra, "
        r"normalised with Ar--CF$_4$ 95/5 visible factor",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "Comparation_ph_MeV_nm_ArCF4_95_5_visible_norm.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            CSV_DIR / "ArN2_ArCF4_model_experiment_ph_MeV_nm_ArCF4_95_5_visible_norm.csv",
            index=False,
        )

    print(
        f"Saved comparison in "
        f"{OUT_DIR / 'Comparation_ph_MeV_nm_ArCF4_95_5_visible_norm.pdf'}"
    )
    print(f"Saved comparison CSV in {CSV_DIR}")
    print(
        "All experimental spectra were multiplied by the same factor obtained "
        "from the visible Ar-CF4 95/5 spectrum at 1 bar."
    )
    print("Model spectra are unchanged.")


if __name__ == "__main__":
    main()