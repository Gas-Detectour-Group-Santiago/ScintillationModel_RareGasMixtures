from __future__ import annotations

from pathlib import Path

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    DEFAULT_SMOOTH_REGIONS,
    ROOT_DIR,
    load_csv_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Primary He--CF$_4$, 80/20, 1 bar"
OUTPUT_NAME = "HeCF4_8020_primary_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (210.0, 820.0)
smooth_spectra = True
smooth_regions = DEFAULT_SMOOTH_REGIONS

CSV_FILENAME = "HeCF4_8020_primario_1_bar_Florian.csv"

CSV_SOURCES = {
    1.0: {
        "csv_path": DATA_DIR / "Experimental" / "HeCF4" / CSV_FILENAME,
        "fallback_paths": [
            ROOT_DIR / CSV_FILENAME,
            Path(__file__).resolve().parent / CSV_FILENAME,
        ],
        "sep": ";",
        "decimal": ",",
        "header": None,
        "wavelength_column": 0,
        "intensity_column": 1,
    },
}

PRESSURE_STYLES = {
    1.0: {
        **DEFAULT_PRESSURE_STYLES[1.0],
        "label": "primary, 1 bar",
    },
}

ANNOTATIONS = [
    {
        "x_guess_nm": 235.0,
        "label": "CF$_4^{+*}$(C) $\\rightarrow$ CF$_4^{+}$(X)",
        "dx_nm": 0.0,
        "dy_frac": 0.62,
        "window_nm": 12.0,
    },
    {
        "x_guess_nm": 290.0,
        "label": "CF$_4^{+*}$(C) $\\rightarrow$ CF$_4^{+*}$(A)",
        "dx_nm": 68.0,
        "dy_frac": 0.13,
        "window_nm": 16.0,
    },
]


def main() -> None:
    setup_raw_spectrum_style()
    spectra = load_csv_spectra(
        CSV_SOURCES,
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
        pressure_styles=PRESSURE_STYLES,
    )


if __name__ == "__main__":
    main()
