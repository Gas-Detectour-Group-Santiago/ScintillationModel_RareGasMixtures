from __future__ import annotations

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    load_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Ar--N$_2$, 99/1, 1 bar"
OUTPUT_NAME = "ArN2_9901_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (300.0, 820.0)

SOURCE = {
    "pkl_path": DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl",
    "concentration_percent": 1.0,
    "concentration_column": "N2 concentration (%)",
    "pressure_column": "P (bar)",
    "spectrum_columns": ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal"),
}

ANNOTATIONS = [
    {
        "x_guess_nm": 337.1,
        "label": "0 - 0",
        "dx_nm": 15.0,
        "dy_frac": 0.24,
        "window_nm": 5.0,
    },
    {
        "x_guess_nm": 357.7,
        "label": "0 - 1",
        "dx_nm": 20.0,
        "dy_frac": 0.22,
        "window_nm": 5.0,
    },
    {
        "x_guess_nm": 380.5,
        "label": "0 - 2",
        "dx_nm": 16.0,
        "dy_frac": 0.16,
        "window_nm": 5.0,
    },
    {
        "x_guess_nm": 380.5,
        "label": "N$_2$(C$^3\\Pi_u$) $\\rightarrow$ N$_2$(B$^3\\Pi_g$)",
        "dx_nm": 16.0,
        "dy_frac": 0.86,
        "window_nm": 5.0,
        "arrow": False,
    },
    {
        "x_guess_nm": 405.9,
        "label": "0 - 3",
        "dx_nm": 18.0,
        "dy_frac": 0.20,
        "window_nm": 5.0,
    },
    {
        "x_guess_nm": 750.0,
        "label": "Ar(5p)$\\rightarrow$ Ar(4s)",
        "dx_nm": -55.0,
        "dy_frac": 0.4,
        "window_nm": 5.0,
        "arrow": False,
    },
]


def main() -> None:
    setup_raw_spectrum_style()
    spectra = load_spectra(SOURCE, PRESSURES_BAR)

    plot_raw_spectrum(
        spectra=spectra,
        annotations=ANNOTATIONS,
        title=TITLE,
        x_range_nm=X_RANGE_NM,
        output_name=OUTPUT_NAME,
        pressures_bar=PRESSURES_BAR,
        pressure_styles=DEFAULT_PRESSURE_STYLES,
    )


if __name__ == "__main__":
    main()
