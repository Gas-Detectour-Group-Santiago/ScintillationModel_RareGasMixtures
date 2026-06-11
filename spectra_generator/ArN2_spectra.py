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
OUT_DIR = ROOT_DIR / "spectra_generator" / "plots_ArN2"
CSV_DIR = ROOT_DIR / "spectra_generator" / "spectra_csv" / "ArN2"

sys.path.insert(0, str(MODELS_DIR))

from ArN2 import W_ArN2, theory_yield_N2_uv  # noqa: E402
from ArN2_infrarred import (  # noqa: E402
    theory_yield_ArN2_Ir_696,
    theory_yield_ArN2_Ir_727,
    theory_yield_ArN2_Ir_750,
    theory_yield_ArN2_Ir_763,
    theory_yield_ArN2_Ir_772,
)

setup_science_style(use_grid=False)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PRESSURES_BAR = [1, 2, 3, 4, 5, 10]
CONCENTRATIONS_PERCENT = [0.1, 1, 10, 100]
WAVELENGTH_NM = np.linspace(300.0, 800.0, 2000)

N2_SECOND_POSITIVE_PEAKS = [
    (335.0, 3.75, 0.42),
    (355.0, 3.75, 0.30),
    (378.0, 3.75, 0.10),
    (403.0, 3.75, 0.05),
]

IR_LINES = {
    696.0: theory_yield_ArN2_Ir_696,
    727.0: theory_yield_ArN2_Ir_727,
    750.0: theory_yield_ArN2_Ir_750,
    763.0: theory_yield_ArN2_Ir_763,
    772.0: theory_yield_ArN2_Ir_772,
}


norm = pd.read_csv(DATA_DIR / "Parameters" / "ArCF4_primary.csv")["parameter"].to_numpy(dtype=float)[0]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
degrad_data = pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArN2.csv")
degrad_data_ir = pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv")

params_uv = pd.read_csv(DATA_DIR / "Parameters" / "ArN2_primary.csv")["parameter"].to_numpy(dtype=float)
params_ir = pd.read_csv(DATA_DIR / "Parameters" / "ArN2_IR_primary.csv")["parameter"].to_numpy(dtype=float)


def arn2_primary_spectrum_ph_per_MeV_nm(concentration_percent: float, pressure_bar: float) -> np.ndarray:
    """
    Build the primary Ar-N2 spectrum in ph/MeV/nm.

    The fitted model functions were adjusted to experimental yields after
    division by W(f), i.e. in ph/eV. Therefore the final conversion to ph/MeV is
    a factor 1e6. The W_ArN2(f) conversion is applied to experimental spectra in
    the comparison scripts, not here.
    """
    f_n2 = concentration_percent / 100.0
    f_arr = np.array([f_n2], dtype=float)

    y_n2_uv = model_fit_unit_to_ph_per_MeV(
        theory_yield_N2_uv(params_uv, degrad_data, f_arr, pressure_bar)
    )[0]

    spectrum = weighted_gaussian_sum(WAVELENGTH_NM, y_n2_uv, N2_SECOND_POSITIVE_PEAKS)

    for line_nm, func in IR_LINES.items():
        y_ir = model_fit_unit_to_ph_per_MeV(
            func(params_ir, degrad_data_ir, f_arr, pressure_bar)
        )[0]
        spectrum += y_ir * gaussian_pdf(WAVELENGTH_NM, line_nm, 2.8)

    return spectrum/norm


def save_pressure_csv(pressure_bar: float, spectra: dict[float, np.ndarray]) -> None:
    df = pd.DataFrame({"wavelength_nm": WAVELENGTH_NM})
    for con, y in spectra.items():
        df[f"{con:g}_percent_N2_ph_MeV_nm"] = y
    df.to_csv(CSV_DIR / f"ArN2_{pressure_bar:g}bar_spectra_ph_MeV_nm.csv", index=False)


def main() -> None:
    cmap = plt.get_cmap("viridis")
    colors_con = cmap(np.linspace(0.15, 0.85, len(CONCENTRATIONS_PERCENT)))
    colors_pres = cmap(np.linspace(0.15, 0.85, len(PRESSURES_BAR)))

    all_spectra: dict[float, dict[float, np.ndarray]] = {}
    global_ymax = 0.0

    for pressure_bar in PRESSURES_BAR:
        spectra_at_pressure = {}
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        for color, con in zip(colors_con, CONCENTRATIONS_PERCENT):
            y = arn2_primary_spectrum_ph_per_MeV_nm(con, pressure_bar)
            spectra_at_pressure[con] = y
            global_ymax = max(global_ymax, float(np.nanmax(y)))
            ax.plot(WAVELENGTH_NM, y, color=color, lw=2, label=rf"{con:g}\% N$_2$")

        save_pressure_csv(pressure_bar, spectra_at_pressure)
        all_spectra[pressure_bar] = spectra_at_pressure

        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_title(rf"Primary Ar--N$_2$ spectrum, {pressure_bar:g} bar")
        ax.set_xlim(300, 800)
        ax.set_ylim(bottom=0)
        ax.legend(ncols=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ArN2_{pressure_bar:g}bar_ph_MeV_nm.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, con in zip(axs, CONCENTRATIONS_PERCENT):
        for color, pressure_bar in zip(colors_pres, PRESSURES_BAR):
            y = all_spectra[pressure_bar][con]
            ax.plot(WAVELENGTH_NM, y, color=color, lw=2, label=rf"{pressure_bar:g} bar")

        ax.set_title(rf"Ar--N$_2$, {con:g}\% N$_2$")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(300, 800)
        ax.set_ylim(0, 1.05 * global_ymax)
        ax.grid(False)

        ax.legend(ncols=2, fontsize=8, loc="upper right")

    fig.suptitle(r"Primary Ar--N$_2$ spectra", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ArN2_concentration_ph_MeV_nm.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Ar-N2 spectra in {OUT_DIR}")
    print(f"Saved Ar-N2 CSV spectra in {CSV_DIR}")
    print("Unit conversion: model ph/eV -> ph/MeV with factor 1e6.")
    print("W_ArN2(f) is used only for experimental ph/e- spectra in comparison scripts.")


if __name__ == "__main__":
    main()
