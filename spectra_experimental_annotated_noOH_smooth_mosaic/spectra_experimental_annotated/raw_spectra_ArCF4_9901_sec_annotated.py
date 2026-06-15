from __future__ import annotations

from pathlib import Path

import numpy as np

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    DEFAULT_SMOOTH_WINDOW_NM,
    ROOT_DIR,
    load_csv_spectra,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
    smooth_intensity,
)


TITLE = "Ar--CF$_4$, 99/1, 1 bar"
OUTPUT_NAME = "ArCF4_9901_secondary_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (180.0, 820.0)
show_OH = False
smooth_spectra = True
smooth_window_nm = DEFAULT_SMOOTH_WINDOW_NM
OH_REMOVE_RANGE_NM = (298.0, 328.0)

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

ANNOTATIONS_WITHOUT_OH = [
    {
        "x_guess_nm": 260.0,
        "label": "CF$_3^{*}$(2A$_2$'') $\\rightarrow$  CF$_3^*$(1A$_2$'')",
        "dx_nm": -35.0,
        "dy_frac": 1.22,
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
        "label": "Ar$^{*}$(4p)$\\rightarrow$ Ar$^{*}$(4s)",
        "dx_nm": -55.0,
        "dy_frac": 0.32,
        "window_nm": 5.0,
        "arrow": False,
    },
]

OH_ANNOTATION = {
    "x_guess_nm": 312.0,
    "label": "OH$^{*}$(A$^{2} \\Sigma^{+}$) $\\rightarrow$ OH(X$^{2}\\Pi$)",
    "dx_nm": -5.0,
    "dy_frac": 0.32,
    "window_nm": 12.0,
}


def _annotations():
    if not show_OH:
        return ANNOTATIONS_WITHOUT_OH
    return [
        ANNOTATIONS_WITHOUT_OH[0],
        OH_ANNOTATION,
        *ANNOTATIONS_WITHOUT_OH[1:],
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


def _remove_oh_peak_if_needed(spectra):
    if show_OH:
        return spectra

    x_min, x_max = OH_REMOVE_RANGE_NM
    cleaned = {}

    for pressure_bar, (wavelength, intensity) in spectra.items():
        wavelength = wavelength.copy()
        intensity = intensity.copy()
        mask = (wavelength >= x_min) & (wavelength <= x_max)

        if np.count_nonzero(mask) == 0:
            cleaned[pressure_bar] = (wavelength, intensity)
            continue

        left = np.where(wavelength < x_min)[0]
        right = np.where(wavelength > x_max)[0]

        if left.size == 0 or right.size == 0:
            cleaned[pressure_bar] = (wavelength[~mask], intensity[~mask])
            continue

        i_left = left[-1]
        i_right = right[0]
        intensity[mask] = np.interp(
            wavelength[mask],
            [wavelength[i_left], wavelength[i_right]],
            [intensity[i_left], intensity[i_right]],
        )

        cleaned[pressure_bar] = (wavelength, intensity)

    return cleaned


def _smooth_spectra_if_needed(spectra):
    if not smooth_spectra:
        return spectra

    return {
        pressure_bar: (
            wavelength,
            smooth_intensity(
                wavelength,
                intensity,
                window_nm=smooth_window_nm,
            ),
        )
        for pressure_bar, (wavelength, intensity) in spectra.items()
    }


def main() -> None:
    setup_raw_spectrum_style()
    spectra = load_csv_spectra(CSV_SOURCES, smooth_spectra=False)
    spectra = _convert_um_to_nm_if_needed(spectra)
    spectra = _remove_oh_peak_if_needed(spectra)
    spectra = _smooth_spectra_if_needed(spectra)

    plot_raw_spectrum(
        spectra=spectra,
        annotations=_annotations(),
        title=TITLE,
        x_range_nm=X_RANGE_NM,
        output_name=OUTPUT_NAME,
        pressures_bar=PRESSURES_BAR,
        pressure_styles=PRESSURE_STYLES,
    )


if __name__ == "__main__":
    main()
