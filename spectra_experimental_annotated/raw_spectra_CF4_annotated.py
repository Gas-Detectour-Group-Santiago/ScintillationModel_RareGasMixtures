from __future__ import annotations

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    load_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Pure CF$_4$, 1 bar"
OUTPUT_NAME = "CF4_pure_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (200.0, 750.0)

SOURCE = {
    "pkl_path": DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl",
    "concentration_percent": 100.0,
    "concentration_column": "concentracion",
    "pressure_column": "presion",
    "spectrum_columns": ("data(norm)",),
}

ANNOTATIONS = [
    {
        "x_guess_nm": 235.0,
        "label": "CF$_4^{+*}$(C) $\\rightarrow$ CF$_4^{+}$(X)",
        "dx_nm": -5.0,
        "dy_frac": 0.52,
        "window_nm": 12.0,
    },
    {
        "x_guess_nm": 290.0,
        "label": "CF$_4^{+*}$(C) $\\rightarrow$ CF$_4^{+*}$(A)",
        "dx_nm": 5.0,
        "dy_frac": 0.15,
        "window_nm": 16.0,
    },
    {
        "x_guess_nm": 615.0,
        "label": "CF$_3^*$ (2A$_2$'') $\\rightarrow$ CF$_3^*$(1A$_2$'')",
        "dx_nm": -50.0,
        "dy_frac": 0.21,
        "window_nm": 10.0,
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
