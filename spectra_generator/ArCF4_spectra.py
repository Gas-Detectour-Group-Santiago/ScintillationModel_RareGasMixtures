from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
plt.style.use("grid")

from spectra_units import (
    gaussian_pdf,
    model_fit_unit_to_ph_per_MeV,
    repo_root_from_script,
    setup_science_style,
    weighted_gaussian_sum,
)

ROOT_DIR = repo_root_from_script(__file__)
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "spectra_generator" / "plots_ArCF4"
CSV_DIR = ROOT_DIR / "spectra_generator" / "spectra_csv" / "ArCF4"

sys.path.insert(0, str(MODELS_DIR))

from ArCF4 import ion_potential, theory_yield_uv, theory_yield_vis  # noqa: E402
from ArCF4_infrarred import (  # noqa: E402
    theory_yield_ArCF4_Ir_696,
    theory_yield_ArCF4_Ir_727,
    theory_yield_ArCF4_Ir_750,
    theory_yield_ArCF4_Ir_763,
    theory_yield_ArCF4_Ir_772,
    theory_yield_ArCF4_Ir_794,
)

setup_science_style(use_grid=False)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PRESSURES_BAR = [1, 2, 3, 4, 5, 10]
CONCENTRATIONS_PERCENT = [0.1, 1, 10, 100]
WAVELENGTH_NM = np.linspace(200.0, 800.0, 2000)

# Empirical spectral decomposition. The weights are relative and are normalised
# internally, so the integral of each component equals the model yield.
CF4_UV_PEAKS = [
    (230.0, 20.0, 0.75),
    (290.0, 20.0, 1.00),
    (364.0, 40.0, 0.10),
]

IR_LINES = {
    696.0: theory_yield_ArCF4_Ir_696,
    727.0: theory_yield_ArCF4_Ir_727,
    750.0: theory_yield_ArCF4_Ir_750,
    763.0: theory_yield_ArCF4_Ir_763,
    772.0: theory_yield_ArCF4_Ir_772,
    794.0: theory_yield_ArCF4_Ir_794,
}

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
degrad_data = pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArCF4.csv")
degrad_data_ir = pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArCF4_IR.csv")

params_uv_vis = pd.read_csv(DATA_DIR / "Parameters" / "ArCF4_primary.csv")["parameter"].to_numpy(dtype=float)
params_ir = pd.read_csv(DATA_DIR / "Parameters" / "ArCF4_IR_primary.csv")["parameter"].to_numpy(dtype=float)


def arcf4_primary_spectrum_ph_per_MeV_nm(concentration_percent: float, pressure_bar: float) -> np.ndarray:
    """
    Build the primary Ar-CF4 spectrum in ph/MeV/nm.

    The fitted primary model functions were adjusted to experimental yields after
    division by W(f), i.e. in ph/eV. Therefore the final conversion to ph/MeV is
    simply a factor 1e6. The fitted normalisation is kept; we do not set
    N_norm=1 and we do not use the old W/0.015 conversion.
    """
    f_cf4 = concentration_percent / 100.0
    f_arr = np.array([f_cf4], dtype=float)

    y_vis = model_fit_unit_to_ph_per_MeV(
        theory_yield_vis(params_uv_vis, degrad_data, f_arr, pressure_bar)
    )[0]

    _, y_cf4, y_ar_dblestar, y_cf3_uv = theory_yield_uv(
        params_uv_vis,
        degrad_data,
        f_arr,
        pressure_bar,
        activate_components=True,
    )
    y_cf4 = model_fit_unit_to_ph_per_MeV(y_cf4)[0]
    y_ar_dblestar = model_fit_unit_to_ph_per_MeV(y_ar_dblestar)[0]
    y_cf3_uv = model_fit_unit_to_ph_per_MeV(y_cf3_uv)[0]

    spectrum = np.zeros_like(WAVELENGTH_NM, dtype=float)
    spectrum += y_vis * gaussian_pdf(WAVELENGTH_NM, 630.0, 40.0)
    spectrum += weighted_gaussian_sum(WAVELENGTH_NM, y_cf4, CF4_UV_PEAKS)
    spectrum += y_ar_dblestar * gaussian_pdf(WAVELENGTH_NM, 220.0, 60.0)
    spectrum += y_cf3_uv * gaussian_pdf(WAVELENGTH_NM, 245.0, 60.0)

    for line_nm, func in IR_LINES.items():
        y_ir = model_fit_unit_to_ph_per_MeV(
            func(params_ir, degrad_data_ir, f_arr, pressure_bar)
        )[0]
        spectrum += y_ir * gaussian_pdf(WAVELENGTH_NM, line_nm, 2.5)

    return spectrum


def save_pressure_csv(pressure_bar: float, spectra: dict[float, np.ndarray]) -> None:
    df = pd.DataFrame({"wavelength_nm": WAVELENGTH_NM})
    for con, y in spectra.items():
        df[f"{con:g}_percent_CF4_ph_MeV_nm"] = y
    df.to_csv(CSV_DIR / f"ArCF4_{pressure_bar:g}bar_spectra_ph_MeV_nm.csv", index=False)


def main() -> None:
    cmap = plt.get_cmap("viridis")
    colors_con = cmap(np.linspace(0.15, 0.85, len(CONCENTRATIONS_PERCENT)))
    colors_pres = cmap(np.linspace(0.15, 0.85, len(PRESSURES_BAR)))

    all_spectra: dict[float, dict[float, np.ndarray]] = {}
    global_ymax = 0.0

    # One figure per pressure, curves for concentrations.
    for pressure_bar in PRESSURES_BAR:
        spectra_at_pressure = {}
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        for color, con in zip(colors_con, CONCENTRATIONS_PERCENT):
            y = arcf4_primary_spectrum_ph_per_MeV_nm(con, pressure_bar)
            spectra_at_pressure[con] = y
            global_ymax = max(global_ymax, float(np.nanmax(y)))
            ax.plot(WAVELENGTH_NM, y, color=color, lw=2, label=rf"{con:g}\% CF$_4$")

        save_pressure_csv(pressure_bar, spectra_at_pressure)
        all_spectra[pressure_bar] = spectra_at_pressure

        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_title(rf"Primary Ar--CF$_4$ spectrum, {pressure_bar:g} bar")
        ax.set_xlim(200, 800)
        ax.set_ylim(bottom=0)
        ax.legend(ncols=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ArCF4_{pressure_bar:g}bar_ph_MeV_nm.pdf", bbox_inches="tight")
        plt.close(fig)

    # Four-panel figure: one panel per concentration, pressure scan.
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, con in zip(axs, CONCENTRATIONS_PERCENT):
        for color, pressure_bar in zip(colors_pres, PRESSURES_BAR):
            y = all_spectra[pressure_bar][con]
            ax.plot(WAVELENGTH_NM, y, color=color, lw=2, label=rf"{pressure_bar:g} bar")

        ax.set_title(rf"Ar--CF$_4$, {con:g}\% CF$_4$")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(200, 800)
        ax.set_ylim(0, 1.05 * global_ymax)
        ax.legend(ncols=2, fontsize=8, loc="upper right")

    fig.suptitle(r"Primary Ar--CF$_4$ spectra", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ArCF4_concentration_ph_MeV_nm.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Ar-CF4 spectra in {OUT_DIR}")
    print(f"Saved Ar-CF4 CSV spectra in {CSV_DIR}")
    print("Unit conversion: model ph/eV -> ph/MeV with factor 1e6.")
    print("W_ArCF4(f) is used only for experimental ph/e- spectra in comparison scripts.")


if __name__ == "__main__":
    main()
