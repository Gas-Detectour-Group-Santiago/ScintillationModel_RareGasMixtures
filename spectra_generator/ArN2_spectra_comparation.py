from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
plt.style.use("grid")

from spectra_units import (
    get_n2_total_yield_ph_per_electron,
    get_spectrum_arrays,
    repo_root_from_script,
    safe_dill_load,
    setup_science_style,
    spectrum_shape_to_ph_per_MeV_nm,
)
from ArN2_spectra import arn2_primary_spectrum_ph_per_MeV_nm, WAVELENGTH_NM

ROOT_DIR = repo_root_from_script(__file__)
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "spectra_generator" / "plots_ArN2"
CSV_DIR = ROOT_DIR / "spectra_generator" / "spectra_csv" / "ArN2"

sys.path.insert(0, str(MODELS_DIR))

from ArN2 import W_ArN2  # noqa: E402

setup_science_style(use_grid=False)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

PRESSURES_BAR = [1]
CONCENTRATIONS_PERCENT = [0.1, 1, 10, 100]
SPECTRUM_COLUMNS = ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal")


def main() -> None:
    df_exp = safe_dill_load(DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl")

    all_rows = []
    global_ymax = 0.0

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, con in zip(axs, CONCENTRATIONS_PERCENT):
        for pres in PRESSURES_BAR:
            y_theory = arn2_primary_spectrum_ph_per_MeV_nm(con, pres)
            global_ymax = max(global_ymax, float(np.nanmax(y_theory)))

            ax.plot(
                WAVELENGTH_NM,
                y_theory,
                color="tab:blue",
                lw=2.2,
                label=rf"Model {pres:g} bar",
            )

            mask = np.isclose(df_exp["N2 concentration (%)"].astype(float), con) & np.isclose(
                df_exp["P (bar)"].astype(float), pres
            )
            if not np.any(mask):
                continue

            row = df_exp.loc[mask].iloc[0]
            wave_exp, intensity_raw = get_spectrum_arrays(row, SPECTRUM_COLUMNS)
            total_yield_ph_e = get_n2_total_yield_ph_per_electron(row, include_ir=True)

            y_exp = spectrum_shape_to_ph_per_MeV_nm(
                wave_exp,
                intensity_raw,
                total_yield_ph_e,
                additive_fraction=con / 100.0,
                w_func=W_ArN2,
            )
            global_ymax = max(global_ymax, float(np.nanmax(y_exp)))

            ax.plot(
                wave_exp,
                y_exp,
                color="tab:red",
                lw=1.8,
                alpha=0.9,
                label=rf"Exp. {pres:g} bar",
            )

            tmp = pd.DataFrame({
                "mixture": "ArN2",
                "concentration_percent": con,
                "pressure_bar": pres,
                "wavelength_nm": wave_exp,
                "experimental_ph_MeV_nm": y_exp,
            })
            all_rows.append(tmp)

        ax.set_title(rf"{con:g}\% N$_2$")
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"ph MeV$^{-1}$ nm$^{-1}$")
        ax.set_xlim(300, 800)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(False)

    for ax in axs:
        ax.set_ylim(0, 1.05 * global_ymax)

    
    fig.suptitle(r"Primary Ar--N$_2$ spectra: model vs. experiment", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ArN2_concentration_comparation_ph_MeV_nm.pdf", bbox_inches="tight")
    plt.close(fig)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            CSV_DIR / "ArN2_experimental_spectra_ph_MeV_nm.csv",
            index=False,
        )

    print(f"Saved Ar-N2 comparison in {OUT_DIR}")
    print("Experimental spectra: raw shape -> integrated ph/e- -> ph/MeV/nm using 1e6/W_ArN2(f).")


if __name__ == "__main__":
    main()
