from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# The stored spectra are used as shapes and are renormalised to the integrated
# ph/e- yields before applying 1e6/W(f).
N2_SPECTRUM_COLUMNS = ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal")
CF4_SPECTRUM_COLUMNS = ("data(norm)",)


def get_cf4_experimental_spectrum(df_cf4, concentration_percent: float, pressure_bar: float):
    mask = np.isclose(df_cf4["concentracion"].astype(float), concentration_percent) & np.isclose(
        df_cf4["presion"].astype(float), pressure_bar
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


def get_n2_experimental_spectrum(df_n2, concentration_percent: float, pressure_bar: float):
    mask = np.isclose(df_n2["N2 concentration (%)"].astype(float), concentration_percent) & np.isclose(
        df_n2["P (bar)"].astype(float), pressure_bar
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


def main() -> None:
    df_n2 = safe_dill_load(DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl")
    df_cf4 = safe_dill_load(DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl")

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
            local_ymax = max(local_ymax, float(np.nanmax(y_n2_model)))
            ax.plot(
                WAVELENGTH_N2_NM,
                y_n2_model,
                color="tab:red",
                lw=2.2,
                label=rf"Ar--N$_2$ model, {pres:g} bar",
            )
            csv_rows.append(pd.DataFrame({
                "mixture": "ArN2",
                "kind": "model",
                "concentration_percent": con,
                "pressure_bar": pres,
                "wavelength_nm": WAVELENGTH_N2_NM,
                "intensity_ph_MeV_nm": y_n2_model,
            }))

            exp_n2 = get_n2_experimental_spectrum(df_n2, con, pres)
            if exp_n2 is not None:
                w_n2, y_n2_exp = exp_n2
                local_ymax = max(local_ymax, float(np.nanmax(y_n2_exp)))
                ax.plot(
                    w_n2,
                    y_n2_exp,
                    color="tab:green",
                    lw=1.8,
                    alpha=0.9,
                    label=rf"Ar--N$_2$ exp., {pres:g} bar",
                )
                csv_rows.append(pd.DataFrame({
                    "mixture": "ArN2",
                    "kind": "experiment",
                    "concentration_percent": con,
                    "pressure_bar": pres,
                    "wavelength_nm": w_n2,
                    "intensity_ph_MeV_nm": y_n2_exp,
                }))

            # -------------------------
            # Ar-CF4: model + experiment
            # -------------------------
            y_cf4_model = arcf4_primary_spectrum_ph_per_MeV_nm(con, pres)
            local_ymax = max(local_ymax, float(np.nanmax(y_cf4_model)))
            ax.plot(
                WAVELENGTH_CF4_NM,
                y_cf4_model,
                color="tab:blue",
                lw=2.2,
                label=rf"Ar--CF$_4$ model, {pres:g} bar",
            )
            csv_rows.append(pd.DataFrame({
                "mixture": "ArCF4",
                "kind": "model",
                "concentration_percent": con,
                "pressure_bar": pres,
                "wavelength_nm": WAVELENGTH_CF4_NM,
                "intensity_ph_MeV_nm": y_cf4_model,
            }))

            exp_cf4 = get_cf4_experimental_spectrum(df_cf4, con, pres)
            if exp_cf4 is not None:
                w_cf4, y_cf4_exp = exp_cf4
                local_ymax = max(local_ymax, float(np.nanmax(y_cf4_exp)))
                ax.plot(
                    w_cf4,
                    y_cf4_exp,
                    color="tab:orange",
                    lw=1.8,
                    alpha=0.9,
                    label=rf"Ar--CF$_4$ exp., {pres:g} bar",
                )
                csv_rows.append(pd.DataFrame({
                    "mixture": "ArCF4",
                    "kind": "experiment",
                    "concentration_percent": con,
                    "pressure_bar": pres,
                    "wavelength_nm": w_cf4,
                    "intensity_ph_MeV_nm": y_cf4_exp,
                }))

        ax.set_title(rf"{con:g}\% additive")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(200, 800)
        ax.set_ylim(0, 1.08 * local_ymax)
        ax.legend(ncols=2, loc="upper right", fontsize=7)

    fig.suptitle(r"Primary Ar--N$_2$ and Ar--CF$_4$ spectra in physical units", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Comparation_ph_MeV_nm.pdf", bbox_inches="tight")
    plt.close(fig)

    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            CSV_DIR / "ArN2_ArCF4_model_experiment_ph_MeV_nm.csv",
            index=False,
        )

    print(f"Saved physical-unit comparison in {OUT_DIR / 'Comparation_ph_MeV_nm.pdf'}")
    print(f"Saved comparison CSV in {CSV_DIR}")
    print("Experimental spectra: raw shape -> integrated ph/e- -> ph/MeV/nm using 1e6/W(f).")
    print("Model spectra: fitted ph/eV -> ph/MeV/nm using 1e6.")


if __name__ == "__main__":
    main()
