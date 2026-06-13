from __future__ import annotations

from pathlib import Path

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    ROOT_DIR,
    load_csv_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Ar--CF$_4$, 99/1, 1 bar"
OUTPUT_NAME = "ArCF4_9901_secondary_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (180.0, 820.0)

CSV_FILENAME = "ArCF4_9901_1_bar_Sara_with_IR.csv"

CSV_SOURCES = {
    1.0: {
        "csv_path": DATA_DIR / "Experimental" / "ArCF4" / CSV_FILENAME,
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
        "label": "1 bar",
    },
}

ANNOTATIONS = [
    {
        "x_guess_nm": 260.0,
        "label": "CF$_3^{*}$(2A$_2$'') $\\rightarrow$  CF$_3^*$(1A$_2$'')",
        "dx_nm": -35.0,
        "dy_frac": 1.22,
        "window_nm": 12.0,
    },
    {
        "x_guess_nm": 312.0,
        "label": "OH$^{*}$(A$^{2} \\Sigma^{+}$) $\\rightarrow$ OH(X$^{2}\\Pi$)",
        "dx_nm": -5.0,
        "dy_frac": 0.32,
        "window_nm": 12.0,
    },
    {
        "x_guess_nm": 615.0,
        "label": "CF$_3^*$ (2A$_2$'') $\\rightarrow$ CF$_3^*$(1A$_1$')",
        "dx_nm": -105.0,
        "dy_frac": 0.31,
        "window_nm": 10.0,
    },
    {
        "x_guess_nm": 750.0,
        "label": "Ar(5p)$\\rightarrow$ Ar(4s)",
        "dx_nm": -55.0,
        "dy_frac": 0.32,
        "window_nm": 5.0,
        "arrow": False,
    },
]


def _convert_um_to_nm_if_needed(spectra):
    """
    Sara's 99/1 CSV stores wavelength in micrometres. Convert it to nm.
    The guard keeps the script safe if the file is later saved directly in nm.
    """
    converted = {}
    for pressure_bar, (wavelength, intensity) in spectra.items():
        if wavelength.size and wavelength.max() < 10.0:
            wavelength = wavelength * 1000.0
        converted[pressure_bar] = (wavelength, intensity)
    return converted


def main() -> None:
    setup_raw_spectrum_style()
    spectra = load_csv_spectra(CSV_SOURCES)
    spectra = _convert_um_to_nm_if_needed(spectra)

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
