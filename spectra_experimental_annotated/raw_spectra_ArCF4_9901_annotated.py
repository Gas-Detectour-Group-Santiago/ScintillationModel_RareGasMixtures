from __future__ import annotations

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    DEFAULT_SMOOTH_REGIONS,
    load_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Ar--CF$_4$, 99/1, 1 bar"
OUTPUT_NAME = "ArCF4_9901_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (180.0, 820.0)
smooth_spectra = True
smooth_regions = DEFAULT_SMOOTH_REGIONS

SOURCE = {
    "pkl_path": DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl",
    "concentration_percent": 1.0,
    "concentration_column": "concentracion",
    "pressure_column": "presion",
    "spectrum_columns": ("data(norm)",),
}

ANNOTATIONS = [
    {
        "x_guess_nm": 235.0,
        "label": "CF$_4^{+*}$(C,v) $\\rightarrow$ CF$_4^{+}$(X)",
        "dx_nm": -35.0,
        "dy_frac": 0.82,
        "window_nm": 12.0,
    },
    {
        "x_guess_nm": 290.0,
        "label": "CF$_4^{+*}$(C,v) $\\rightarrow$ CF$_4^{+*}$(A)",
        "dx_nm": 18.0,
        "dy_frac": 0.63,
        "window_nm": 16.0,
    },
    {
        "x_guess_nm": 400.0,
        "label": "CF$_4^{+*}$(D,v) $\\rightarrow$ CF$_4^{+*}$(C)",
        "dx_nm": -10,
        "dy_frac": 0.53,
        "window_nm": 16.0,
    },
    {
        "x_guess_nm": 615.0,
        "label": "CF$_3^*$ (2A$_2$'') $\\rightarrow$ CF$_3^*$(1A$_1$')",
        "dx_nm": -105.0,
        "dy_frac": 0.21,
        "window_nm": 10.0,
    },
    {
        "x_guess_nm": 750.0,
        "label": "Ar$^{*}$(4p)$\\rightarrow$ Ar$^{*}$(4s)",
        "dx_nm": -55.0,
        "dy_frac": 0.72,
        "window_nm": 5.0,
        "arrow": False,
    },
]


def main() -> None:
    setup_raw_spectrum_style()
    spectra = load_spectra(
        SOURCE,
        PRESSURES_BAR,
        smooth_spectra=smooth_spectra,
        smooth_regions=smooth_regions,
    )

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
