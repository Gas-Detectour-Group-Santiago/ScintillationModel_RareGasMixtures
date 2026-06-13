from __future__ import annotations

from spectra_annotate import (
    DATA_DIR,
    DEFAULT_PRESSURE_STYLES,
    load_spectra_with_fallback,
    plot_raw_spectrum,
    setup_raw_spectrum_style,
)


TITLE = "Pure Ar, 1 bar"
OUTPUT_NAME = "Ar_pure_raw_1bar.pdf"
PRESSURES_BAR = [1.0]
X_RANGE_NM = (160.0, 820.0)

SOURCES = [
    {
        "name": "ArCF4 file, 0% CF4",
        "pkl_path": DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl",
        "concentration_percent": 0.0,
        "concentration_column": "concentracion",
        "pressure_column": "presion",
        "spectrum_columns": ("data(norm)",),
    },
    {
        "name": "ArN2 file, 0% N2",
        "pkl_path": DATA_DIR / "Experimental" / "ArN2" / "N2_primary_data_final.pkl",
        "concentration_percent": 0.0,
        "concentration_column": "N2 concentration (%)",
        "pressure_column": "P (bar)",
        "spectrum_columns": ("mean_spectrum", "spectrum_new_cal", "spectrum_old_cal"),
    },
]

ANNOTATIONS = [
    {
        "x_guess_nm": 220.0,
        "label": "Ar 3rd emission (170-240 nm)",
        "dx_nm": 25.0,
        "dy_frac": 0.34,
        "window_nm": 45.0,
    },
    {
        "x_guess_nm": 750.0,
        "label": "Ar(5p)$\\rightarrow$ Ar(4s)",
        "dx_nm": -95.0,
        "dy_frac": 0.6,
        "window_nm": 5.0,
        "arrow": False,
    },
]


def main() -> None:
    setup_raw_spectrum_style()
    spectra, used_sources = load_spectra_with_fallback(SOURCES, PRESSURES_BAR)

    plot_raw_spectrum(
        spectra=spectra,
        annotations=ANNOTATIONS,
        title=TITLE,
        x_range_nm=X_RANGE_NM,
        output_name=OUTPUT_NAME,
        pressures_bar=PRESSURES_BAR,
        pressure_styles=DEFAULT_PRESSURE_STYLES,
    )

    print("Sources used:", ", ".join(sorted(set(used_sources))))


if __name__ == "__main__":
    main()
