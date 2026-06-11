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
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Fixed Ar-CF4 reference: always 95/5 at 1 bar.
ARCF4_REFERENCE_CONCENTRATION_PERCENT = 5.0
ARCF4_REFERENCE_PRESSURE_BAR = 1.0

# Integration ranges.
ARCF4_VISIBLE_RANGE_NM = (500.0, 750.0)
ARN2_UV_RANGE_NM = (323.0, 450.0)

# If None, all available Ar-N2 pressures are used.
# You can replace it by, for example: [0.05, 0.1, 0.3, 0.5, 1.0]
PRESSURES_TO_PLOT = None

N2_SPECTRUM_COLUMNS = ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal")
CF4_SPECTRUM_COLUMNS = ("data(norm)",)

CSV_NAME = (
    "ArN2_UV323_450_over_ArCF4_95_5_1bar_VIS500_750_"
    "ratio_vs_concentration_by_pressure.csv"
)

FIG_NAME = (
    "ArN2_UV323_450_over_ArCF4_95_5_1bar_VIS500_750_"
    "ratio_vs_concentration_by_pressure.pdf"
)


def integrate_range(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    xmin: float,
    xmax: float,
) -> float:
    """
    Integrate intensity between xmin and xmax using trapezoidal integration.

    The exact boundaries xmin and xmax are inserted by interpolation when
    needed, avoiding edge biases from the discrete wavelength sampling.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    mask = np.isfinite(wavelength_nm) & np.isfinite(intensity)
    wavelength_nm = wavelength_nm[mask]
    intensity = intensity[mask]

    if wavelength_nm.size < 2:
        return np.nan

    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    intensity = intensity[order]

    wavelength_nm, unique_idx = np.unique(wavelength_nm, return_index=True)
    intensity = intensity[unique_idx]

    if xmax <= wavelength_nm[0] or xmin >= wavelength_nm[-1]:
        return np.nan

    xmin_eff = max(xmin, wavelength_nm[0])
    xmax_eff = min(xmax, wavelength_nm[-1])

    mask_range = (wavelength_nm >= xmin_eff) & (wavelength_nm <= xmax_eff)

    w_int = wavelength_nm[mask_range]
    y_int = intensity[mask_range]

    y_xmin = np.interp(xmin_eff, wavelength_nm, intensity)
    y_xmax = np.interp(xmax_eff, wavelength_nm, intensity)

    w_int = np.concatenate(([xmin_eff], w_int, [xmax_eff]))
    y_int = np.concatenate(([y_xmin], y_int, [y_xmax]))

    order = np.argsort(w_int)
    w_int = w_int[order]
    y_int = y_int[order]

    w_int, unique_idx = np.unique(w_int, return_index=True)
    y_int = y_int[unique_idx]

    return float(np.trapz(y_int, w_int))


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


def get_available_pressures_n2(df_n2: pd.DataFrame) -> np.ndarray:
    return np.sort(df_n2["P (bar)"].astype(float).unique())


def get_available_concentrations_n2(
    df_n2: pd.DataFrame,
    pressure_bar: float,
) -> np.ndarray:
    mask = np.isclose(df_n2["P (bar)"].astype(float), pressure_bar)

    return np.sort(
        df_n2.loc[mask, "N2 concentration (%)"].astype(float).unique()
    )


def get_arcf4_reference_integral(df_cf4: pd.DataFrame) -> float:
    """
    Fixed denominator:
    Ar-CF4 95/5 at 1 bar, integrated from 500 to 750 nm.
    """
    exp_cf4_ref = get_cf4_experimental_spectrum(
        df_cf4,
        concentration_percent=ARCF4_REFERENCE_CONCENTRATION_PERCENT,
        pressure_bar=ARCF4_REFERENCE_PRESSURE_BAR,
    )

    if exp_cf4_ref is None:
        raise RuntimeError(
            "Could not find the fixed Ar-CF4 reference spectrum: "
            f"{ARCF4_REFERENCE_CONCENTRATION_PERCENT:g}% CF4, "
            f"{ARCF4_REFERENCE_PRESSURE_BAR:g} bar."
        )

    w_cf4_ref, y_cf4_ref = exp_cf4_ref

    arcf4_reference_integral = integrate_range(
        w_cf4_ref,
        y_cf4_ref,
        xmin=ARCF4_VISIBLE_RANGE_NM[0],
        xmax=ARCF4_VISIBLE_RANGE_NM[1],
    )

    if (
        not np.isfinite(arcf4_reference_integral)
        or arcf4_reference_integral <= 0.0
    ):
        raise RuntimeError(
            "Invalid Ar-CF4 reference integral obtained for "
            f"{ARCF4_REFERENCE_CONCENTRATION_PERCENT:g}% CF4, "
            f"{ARCF4_REFERENCE_PRESSURE_BAR:g} bar."
        )

    return arcf4_reference_integral


def compute_integral_ratios(
    df_cf4: pd.DataFrame,
    df_n2: pd.DataFrame,
) -> pd.DataFrame:
    arcf4_reference_integral = get_arcf4_reference_integral(df_cf4)

    print(
        "\nFixed denominator:"
        f"\nAr-CF4 {ARCF4_REFERENCE_CONCENTRATION_PERCENT:g}% CF4, "
        f"{ARCF4_REFERENCE_PRESSURE_BAR:g} bar, "
        f"{ARCF4_VISIBLE_RANGE_NM[0]:g}-{ARCF4_VISIBLE_RANGE_NM[1]:g} nm"
        f"\nIntegral = {arcf4_reference_integral:.6e} ph/MeV\n"
    )

    if PRESSURES_TO_PLOT is None:
        pressures = get_available_pressures_n2(df_n2)
    else:
        pressures = np.asarray(PRESSURES_TO_PLOT, dtype=float)

    rows = []

    for pressure_bar in pressures:
        concentrations = get_available_concentrations_n2(df_n2, pressure_bar)

        if concentrations.size == 0:
            print(f"Skipping P = {pressure_bar:g} bar: no Ar-N2 concentrations.")
            continue

        for concentration_percent in concentrations:
            exp_n2 = get_n2_experimental_spectrum(
                df_n2,
                concentration_percent=concentration_percent,
                pressure_bar=pressure_bar,
            )

            if exp_n2 is None:
                print(
                    f"Skipping P = {pressure_bar:g} bar, "
                    f"c = {concentration_percent:g}%: no Ar-N2 spectrum."
                )
                continue

            w_n2, y_n2 = exp_n2

            arn2_uv_integral = integrate_range(
                w_n2,
                y_n2,
                xmin=ARN2_UV_RANGE_NM[0],
                xmax=ARN2_UV_RANGE_NM[1],
            )

            if not np.isfinite(arn2_uv_integral):
                print(
                    f"Skipping P = {pressure_bar:g} bar, "
                    f"c = {concentration_percent:g}%: invalid Ar-N2 integral."
                )
                continue

            ratio = arn2_uv_integral / arcf4_reference_integral

            rows.append(
                {
                    "pressure_bar": pressure_bar,
                    "concentration_percent": concentration_percent,
                    "ArCF4_reference_concentration_percent": (
                        ARCF4_REFERENCE_CONCENTRATION_PERCENT
                    ),
                    "ArCF4_reference_pressure_bar": ARCF4_REFERENCE_PRESSURE_BAR,
                    "ArCF4_visible_range_nm": (
                        f"{ARCF4_VISIBLE_RANGE_NM[0]:g}-{ARCF4_VISIBLE_RANGE_NM[1]:g}"
                    ),
                    "ArN2_uv_range_nm": (
                        f"{ARN2_UV_RANGE_NM[0]:g}-{ARN2_UV_RANGE_NM[1]:g}"
                    ),
                    "ArCF4_95_5_1bar_visible_integral_ph_MeV": (
                        arcf4_reference_integral
                    ),
                    "ArN2_uv_integral_ph_MeV": arn2_uv_integral,
                    "ratio_ArN2_UV_over_ArCF4_95_5_1bar_VIS": ratio,
                }
            )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No valid integral ratios could be computed.")

    results = results.sort_values(
        ["pressure_bar", "concentration_percent"]
    ).reset_index(drop=True)

    return results


def plot_ratio_vs_concentration_by_pressure(
    results: pd.DataFrame,
    fig_path,
) -> None:
    pressures = np.sort(results["pressure_bar"].unique())

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(pressures)))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    for pressure_bar, color in zip(pressures, colors):
        subset = results[np.isclose(results["pressure_bar"], pressure_bar)].copy()
        subset = subset.sort_values("concentration_percent")

        ax.plot(
            subset["concentration_percent"],
            subset["ratio_ArN2_UV_over_ArCF4_95_5_1bar_VIS"],
            marker="o",
            lw=2.0,
            color=color,
            label=rf"{pressure_bar:g} bar",
        )

    ax.set_xlabel(r"N$_2$ concentration [\%]")

    ax.set_ylabel(
        r"$\int_{323}^{450} I_{\mathrm{Ar-N_2}}(c,P)\,d\lambda$ / "
        r"$\int_{500}^{750} I_{\mathrm{Ar-CF_4}}(5\%,1\,\mathrm{bar})\,d\lambda$"
    )

    ax.set_title(
        r"Experimental Ar--N$_2$ UV relative to fixed Ar--CF$_4$ VIS reference"
    )

    ax.grid(False)

    concentrations = np.sort(results["concentration_percent"].unique())

    if concentrations.min() > 0.0 and concentrations.max() / concentrations.min() > 5.0:
        ax.set_xscale("log")        
        ax.set_yscale("log")


    ax.legend(
        title=r"$P_{\mathrm{Ar-N_2}}$",
        fontsize=8,
        title_fontsize=9,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_n2 = safe_dill_load(
        DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl"
    )
    df_cf4 = safe_dill_load(
        DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl"
    )

    results = compute_integral_ratios(df_cf4=df_cf4, df_n2=df_n2)

    csv_path = CSV_DIR / CSV_NAME
    fig_path = OUT_DIR / FIG_NAME

    results.to_csv(csv_path, index=False)

    print("\nIntegrated experimental spectra:")
    print(results.to_string(index=False))

    print("\nMain values:")
    for _, row in results.iterrows():
        print(
            f"P_N2 = {row['pressure_bar']:g} bar | "
            f"c_N2 = {row['concentration_percent']:g}% | "
            f"Ar-N2 UV 323-450 nm = "
            f"{row['ArN2_uv_integral_ph_MeV']:.6e} ph/MeV | "
            f"Ar-CF4 95/5, 1 bar, VIS 500-750 nm = "
            f"{row['ArCF4_95_5_1bar_visible_integral_ph_MeV']:.6e} ph/MeV | "
            f"ratio = "
            f"{row['ratio_ArN2_UV_over_ArCF4_95_5_1bar_VIS']:.6e}"
        )

    plot_ratio_vs_concentration_by_pressure(results, fig_path)

    print(f"\nSaved CSV in {csv_path}")
    print(f"Saved figure in {fig_path}")


if __name__ == "__main__":
    main()