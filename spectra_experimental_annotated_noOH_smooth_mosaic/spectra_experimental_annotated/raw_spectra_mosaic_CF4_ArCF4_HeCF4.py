from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectra_annotate import (
    DATA_DIR,
    OUT_DIR,
    ROOT_DIR,
    load_csv_spectra,
    load_spectra,
    setup_raw_spectrum_style,
    smooth_intensity,
)


PRESSURE_BAR = 1.0
X_RANGE_NM = (200.0, 820.0)
show_OH = False
smooth_spectra = True
smooth_window_nm = 7.0
normalize_panels = True
OH_REMOVE_RANGE_NM = (298.0, 328.0)


CF4_PRIMARY_SOURCE = {
    "pkl_path": DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl",
    "concentration_percent": 100.0,
    "concentration_column": "concentracion",
    "pressure_column": "presion",
    "spectrum_columns": ("data(norm)",),
}

ARCF4_9901_PRIMARY_SOURCE = {
    "pkl_path": DATA_DIR / "Experimental" / "ArCF4" / "CF4_primary_data_final.pkl",
    "concentration_percent": 1.0,
    "concentration_column": "concentracion",
    "pressure_column": "presion",
    "spectrum_columns": ("data(norm)",),
}


def csv_source(filename: str, subfolder: str) -> dict[float, dict]:
    return {
        PRESSURE_BAR: {
            "csv_path": DATA_DIR / "Experimental" / subfolder / filename,
            "fallback_paths": [
                ROOT_DIR / filename,
                Path(__file__).resolve().parent / filename,
            ],
            "sep": ";",
            "decimal": ",",
            "header": None,
            "wavelength_column": 0,
            "intensity_column": 1,
        },
    }


PANELS = [
    {
        "mixture": "Pure CF$_4$",
        "kind": "Primary",
        "source_type": "pkl",
        "source": CF4_PRIMARY_SOURCE,
        "color": "tab:blue",
    },
    {
        "mixture": "Ar--CF$_4$ 99/1",
        "kind": "Primary",
        "source_type": "pkl",
        "source": ARCF4_9901_PRIMARY_SOURCE,
        "color": "tab:purple",
    },
    {
        "mixture": "He--CF$_4$ 80/20",
        "kind": "Primary",
        "source_type": "csv",
        "source": csv_source("HeCF4_8020_primario_1_bar_Florian.csv", "HeCF4"),
        "color": "tab:green",
    },
    {
        "mixture": "Pure CF$_4$",
        "kind": "Secondary",
        "source_type": "csv",
        "source": csv_source("CF4_1_bar_Florian.csv", "ArCF4"),
        "color": "tab:blue",
    },
    {
        "mixture": "Ar--CF$_4$ 99/1",
        "kind": "Secondary",
        "source_type": "csv",
        "source": csv_source("ArCF4_9901_1_bar_Sara_with_IR.csv", "ArCF4"),
        "color": "tab:purple",
        "remove_oh": True,
    },
    {
        "mixture": "He--CF$_4$ 80/20",
        "kind": "Secondary",
        "source_type": "csv",
        "source": csv_source("HeCF4_8020_secundario_1_bar_Florian.csv", "HeCF4"),
        "color": "tab:green",
    },
]


def convert_um_to_nm_if_needed(spectra):
    converted = {}
    for pressure_bar, (wavelength, intensity) in spectra.items():
        wavelength = wavelength.copy()
        if wavelength.size and np.nanmax(wavelength) < 10.0:
            wavelength = wavelength * 1000.0
        converted[pressure_bar] = (wavelength, intensity)
    return converted


def remove_oh_peak_if_needed(spectra, *, remove_oh: bool = False):
    if show_OH or not remove_oh:
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


def smooth_spectra_if_needed(spectra):
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


def normalize_spectra_if_needed(spectra):
    if not normalize_panels:
        return spectra

    normalized = {}
    for pressure_bar, (wavelength, intensity) in spectra.items():
        ymax = float(np.nanmax(intensity))
        if np.isfinite(ymax) and ymax > 0.0:
            intensity = intensity / ymax
        normalized[pressure_bar] = (wavelength, intensity)
    return normalized


def load_panel_spectrum(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    if panel["source_type"] == "pkl":
        spectra = load_spectra(
            panel["source"],
            [PRESSURE_BAR],
            smooth_spectra=False,
            normalize_to_max=False,
        )
    elif panel["source_type"] == "csv":
        spectra = load_csv_spectra(
            panel["source"],
            smooth_spectra=False,
            normalize_to_max=False,
        )
    else:
        raise ValueError(f"Unknown source_type: {panel['source_type']!r}")

    spectra = convert_um_to_nm_if_needed(spectra)
    spectra = remove_oh_peak_if_needed(spectra, remove_oh=panel.get("remove_oh", False))
    spectra = smooth_spectra_if_needed(spectra)
    spectra = normalize_spectra_if_needed(spectra)

    return spectra[PRESSURE_BAR]


def plot_panel(ax, panel: dict) -> None:
    wavelength, intensity = load_panel_spectrum(panel)

    ax.plot(
        wavelength,
        intensity,
        color=panel["color"],
        lw=2.0,
    )

    ax.set_xlim(*X_RANGE_NM)
    ax.set_ylim(bottom=0.0)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"{panel['mixture']}\\n{panel['kind']}", fontsize=12)


def save_mosaic_2x3() -> Path:
    fig, axs = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True)

    for ax, panel in zip(axs.ravel(), PANELS):
        plot_panel(ax, panel)

    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\lambda$ [nm]")

    fig.suptitle("Experimental spectra at 1 bar", fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.86, wspace=0.18, hspace=0.38)

    outpath = OUT_DIR / "mosaic_CF4_ArCF4_HeCF4_2x3.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def save_mosaic_3x2() -> Path:
    order = [0, 3, 1, 4, 2, 5]
    fig, axs = plt.subplots(3, 2, figsize=(9.0, 8.6), sharex=True)

    for ax, idx in zip(axs.ravel(), order):
        plot_panel(ax, PANELS[idx])

    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\lambda$ [nm]")

    fig.suptitle("Experimental spectra at 1 bar", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.20, hspace=0.45)

    outpath = OUT_DIR / "mosaic_CF4_ArCF4_HeCF4_3x2.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def main() -> None:
    setup_raw_spectrum_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_2x3 = save_mosaic_2x3()
    out_3x2 = save_mosaic_3x2()

    print(f"Saved figure to {out_2x3}")
    print(f"Saved figure to {out_3x2}")


if __name__ == "__main__":
    main()
