from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import ensure_parent, match_float, setup_plot_style
from .io import read_raw_csv, select_raw_spectrum


PRESSURE_BAR = 1.0
ANNOTATION_COLOR = "0.15"
DEFAULT_SMOOTH_REGIONS = (
    (180.0, 650.0, 12.0),
    (650.0, 820.0, 1.0),
)


ARCF4_9505_ANNOTATIONS = [
    {"x_guess_nm": 235.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+}$(X)", "dx_nm": -35.0, "dy_frac": 0.32, "window_nm": 12.0},
    {"x_guess_nm": 290.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+*}$(A)", "dx_nm": 18.0, "dy_frac": 0.23, "window_nm": 16.0},
    {"x_guess_nm": 400.0, "label": r"CF$_4^{+*}$(D) $\rightarrow$ CF$_4^{+*}$(C)", "dx_nm": 18.0, "dy_frac": 0.23, "window_nm": 16.0},
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_1$')", "dx_nm": -105.0, "dy_frac": 0.31, "window_nm": 10.0},
    {"x_guess_nm": 750.0, "label": r"Ar$^{*}$(4p)$\rightarrow$ Ar$^{*}$(4s)", "dx_nm": -55.0, "dy_frac": 0.12, "window_nm": 5.0, "arrow": False},
]

ARCF4_9901_ANNOTATIONS = [
    {"x_guess_nm": 235.0, "label": r"CF$_4^{+*}$(C,v) $\rightarrow$ CF$_4^{+}$(X)", "dx_nm": -35.0, "dy_frac": 0.82, "window_nm": 12.0},
    {"x_guess_nm": 290.0, "label": r"CF$_4^{+*}$(C,v) $\rightarrow$ CF$_4^{+*}$(A)", "dx_nm": 18.0, "dy_frac": 0.63, "window_nm": 16.0},
    {"x_guess_nm": 400.0, "label": r"CF$_4^{+*}$(D,v) $\rightarrow$ CF$_4^{+*}$(C)", "dx_nm": -10.0, "dy_frac": 0.53, "window_nm": 16.0},
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_1$')", "dx_nm": -105.0, "dy_frac": 0.21, "window_nm": 10.0},
    {"x_guess_nm": 750.0, "label": r"Ar$^{*}$(4p)$\rightarrow$ Ar$^{*}$(4s)", "dx_nm": -55.0, "dy_frac": 0.72, "window_nm": 5.0, "arrow": False},
]

ARN2_9901_ANNOTATIONS = [
    {"x_guess_nm": 337.1, "label": "0 - 0", "dx_nm": 15.0, "dy_frac": 0.24, "window_nm": 5.0},
    {"x_guess_nm": 357.7, "label": "0 - 1", "dx_nm": 20.0, "dy_frac": 0.22, "window_nm": 5.0},
    {"x_guess_nm": 380.5, "label": "0 - 2", "dx_nm": 16.0, "dy_frac": 0.16, "window_nm": 5.0},
    {"x_guess_nm": 380.5, "label": r"N$_2$(C$^3\Pi_u$) $\rightarrow$ N$_2$(B$^3\Pi_g$)", "dx_nm": 16.0, "dy_frac": 0.86, "window_nm": 5.0, "arrow": False},
    {"x_guess_nm": 405.9, "label": "0 - 3", "dx_nm": 18.0, "dy_frac": 0.20, "window_nm": 5.0},
    {"x_guess_nm": 750.0, "label": r"Ar$^{*}$(4p)$\rightarrow$ Ar$^{*}$(4s)", "dx_nm": -55.0, "dy_frac": 0.40, "window_nm": 5.0, "arrow": False},
]

AR_PURE_ANNOTATIONS = [
    {"x_guess_nm": 220.0, "label": "Ar 3rd emission (170-240 nm)", "dx_nm": 25.0, "dy_frac": 0.34, "window_nm": 45.0},
    {"x_guess_nm": 750.0, "label": r"Ar$^{*}$(4p)$\rightarrow$ Ar$^{*}$(4s)", "dx_nm": -95.0, "dy_frac": 0.60, "window_nm": 5.0, "arrow": False},
]

CF4_PRIMARY_ANNOTATIONS = [
    {"x_guess_nm": 235.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+}$(X)", "dx_nm": -5.0, "dy_frac": 0.52, "window_nm": 12.0},
    {"x_guess_nm": 290.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+*}$(A)", "dx_nm": 5.0, "dy_frac": 0.15, "window_nm": 16.0},
    {"x_guess_nm": 400.0, "label": r"CF$_4^{+*}$(D) $\rightarrow$ CF$_4^{+*}$(C)", "dx_nm": 18.0, "dy_frac": 0.23, "window_nm": 16.0},
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_2$'')", "dx_nm": -50.0, "dy_frac": 0.21, "window_nm": 10.0},
]

CF4_SECONDARY_ANNOTATIONS = [
    {"x_guess_nm": 260.0, "label": r"CF$_3^{*}$(2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_2$'')", "dx_nm": -35.0, "dy_frac": 0.32, "window_nm": 12.0},
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_2$'')", "dx_nm": -50.0, "dy_frac": 0.21, "window_nm": 10.0},
]

ARCF4_SECONDARY_ANNOTATIONS = [
    {"x_guess_nm": 260.0, "label": r"CF$_3^{*}$(2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_2$'')", "dx_nm": -35.0, "dy_frac": 0.32, "window_nm": 12.0},
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_1$')", "dx_nm": -105.0, "dy_frac": 0.31, "window_nm": 10.0},
    {"x_guess_nm": 750.0, "label": r"Ar$^{*}$(4p)$\rightarrow$ Ar$^{*}$(4s)", "dx_nm": -55.0, "dy_frac": 0.12, "window_nm": 5.0, "arrow": False},
]

HECF4_PRIMARY_ANNOTATIONS = [
    {"x_guess_nm": 235.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+}$(X)", "dx_nm": 0.0, "dy_frac": 0.62, "window_nm": 12.0},
    {"x_guess_nm": 290.0, "label": r"CF$_4^{+*}$(C) $\rightarrow$ CF$_4^{+*}$(A)", "dx_nm": 68.0, "dy_frac": 0.13, "window_nm": 16.0},
]

HECF4_SECONDARY_ANNOTATIONS = [
    *HECF4_PRIMARY_ANNOTATIONS,
    {"x_guess_nm": 615.0, "label": r"CF$_3^*$ (2A$_2$'') $\rightarrow$ CF$_3^*$(1A$_2$'')", "dx_nm": -35.0, "dy_frac": 0.31, "window_nm": 10.0},
]


ANNOTATED_SPECS = (
    {
        "name": "ArCF4_9505_raw_1bar",
        "title": r"Ar--CF$_4$, 95/5, 1 bar",
        "source": {"type": "raw", "candidates": [("ArCF4", 5.0, PRESSURE_BAR)]},
        "annotations": ARCF4_9505_ANNOTATIONS,
        "x_range_nm": (180.0, 820.0),
        "output_pdf": "ArCF4_9505_raw_1bar.pdf",
    },
    {
        "name": "ArCF4_9901_raw_1bar",
        "title": r"Ar--CF$_4$, 99/1, 1 bar",
        "source": {"type": "raw", "candidates": [("ArCF4", 1.0, PRESSURE_BAR)]},
        "annotations": ARCF4_9901_ANNOTATIONS,
        "x_range_nm": (180.0, 820.0),
        "output_pdf": "ArCF4_9901_raw_1bar.pdf",
    },
    {
        "name": "ArN2_9901_raw_1bar",
        "title": r"Ar--N$_2$, 99/1, 1 bar",
        "source": {"type": "raw", "candidates": [("ArN2", 1.0, PRESSURE_BAR)]},
        "annotations": ARN2_9901_ANNOTATIONS,
        "x_range_nm": (300.0, 820.0),
        "output_pdf": "ArN2_9901_raw_1bar.pdf",
    },
    {
        "name": "Ar_pure_raw_1bar",
        "title": "Pure Ar, 1 bar",
        "source": {"type": "raw", "candidates": [("ArCF4", 0.0, PRESSURE_BAR), ("ArN2", 0.0, PRESSURE_BAR)]},
        "annotations": AR_PURE_ANNOTATIONS,
        "x_range_nm": (160.0, 820.0),
        "output_pdf": "Ar_pure_raw_1bar.pdf",
    },
    {
        "name": "CF4_pure_raw_1bar",
        "title": r"Pure CF$_4$, 1 bar",
        "source": {"type": "raw", "candidates": [("ArCF4", 100.0, PRESSURE_BAR)]},
        "annotations": CF4_PRIMARY_ANNOTATIONS,
        "x_range_nm": (200.0, 750.0),
        "output_pdf": "CF4_pure_raw_1bar.pdf",
    },
    {
        "name": "CF4_pure_secondary_raw_1bar",
        "title": r"Secondary pure CF$_4$, 1 bar",
        "source": {"type": "csv", "filename": "CF4_1_bar_Florian.csv"},
        "annotations": CF4_SECONDARY_ANNOTATIONS,
        "x_range_nm": (200.0, 820.0),
        "output_pdf": "CF4_pure_secondary_raw_1bar.pdf",
    },
    {
        "name": "ArCF4_9505_secondary_raw_1bar",
        "title": r"Secondary Ar--CF$_4$, 95/5, 1 bar",
        "source": {"type": "csv", "filename": "ArCF4_9505_1_bar_Sara_with_IR.csv"},
        "annotations": ARCF4_SECONDARY_ANNOTATIONS,
        "x_range_nm": (210.0, 820.0),
        "output_pdf": "ArCF4_9505_secondary_raw_1bar.pdf",
    },
    {
        "name": "ArCF4_9901_secondary_raw_1bar",
        "title": r"Secondary Ar--CF$_4$, 99/1, 1 bar",
        "source": {"type": "csv", "filename": "ArCF4_9901_1_bar_Sara_with_IR_NO_OH.csv"},
        "annotations": ARCF4_SECONDARY_ANNOTATIONS,
        "x_range_nm": (180.0, 820.0),
        "output_pdf": "ArCF4_9901_secondary_raw_1bar.pdf",
    },
    {
        "name": "HeCF4_8020_primary_raw_1bar",
        "title": r"Primary He--CF$_4$, 80/20, 1 bar",
        "source": {"type": "csv", "filename": "HeCF4_8020_primario_1_bar_Florian.csv"},
        "annotations": HECF4_PRIMARY_ANNOTATIONS,
        "x_range_nm": (210.0, 820.0),
        "output_pdf": "HeCF4_8020_primary_raw_1bar.pdf",
    },
    {
        "name": "HeCF4_8020_secondary_raw_1bar",
        "title": r"Secondary He--CF$_4$, 80/20, 1 bar",
        "source": {"type": "csv", "filename": "HeCF4_8020_secundario_1_bar_Florian.csv"},
        "annotations": HECF4_SECONDARY_ANNOTATIONS,
        "x_range_nm": (210.0, 820.0),
        "output_pdf": "HeCF4_8020_secondary_raw_1bar.pdf",
    },
)


def _configure_annotated_style() -> None:
    setup_plot_style()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "dejavuserif",
            "font.family": "serif",
            "axes.grid": False,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )


def _smooth_intensity(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    *,
    smooth_regions: tuple[tuple[float, float, float], ...] = DEFAULT_SMOOTH_REGIONS,
    polyorder: int = 2,
) -> np.ndarray:
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(intensity, dtype=float)
    y_out = y.copy()

    if y.size < 5:
        return y_out

    for x_min, x_max, window_nm in smooth_regions:
        mask = np.isfinite(x) & np.isfinite(y) & (x >= x_min) & (x < x_max)
        if window_nm <= 0.0 or np.count_nonzero(mask) < 5:
            continue

        dx = np.diff(np.sort(x[mask]))
        dx = dx[np.isfinite(dx) & (dx > 0.0)]
        if dx.size == 0:
            continue

        window_length = max(int(round(window_nm / float(np.median(dx)))), polyorder + 3, 5)
        if window_length % 2 == 0:
            window_length += 1
        max_odd = np.count_nonzero(mask) if np.count_nonzero(mask) % 2 else np.count_nonzero(mask) - 1
        window_length = min(window_length, max_odd)
        if window_length <= polyorder + 1 or window_length < 5:
            continue

        try:
            from scipy.signal import savgol_filter

            y_out[mask] = savgol_filter(
                y[mask],
                window_length=window_length,
                polyorder=min(polyorder, window_length - 2),
                mode="interp",
            )
        except Exception:
            kernel = np.ones(window_length, dtype=float) / float(window_length)
            padded = np.pad(y[mask], window_length // 2, mode="edge")
            y_out[mask] = np.convolve(padded, kernel, mode="valid")

    return y_out


def _preprocess_spectrum(wavelength: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(wavelength, dtype=float)
    y = np.clip(np.asarray(intensity, dtype=float), 0.0, None)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size and np.nanmax(x) < 10.0:
        x = x * 1000.0
    y = _smooth_intensity(x, y)
    return x, y


def _annotated_input_path(project_root: Path, filename: str) -> Path:
    candidates = [
        project_root / "data" / cfg.ANNOTATED_INPUT_DIRNAME / filename,
        project_root / "data" / "Experimental" / "ArCF4" / filename,
        project_root / "data" / "Experimental" / "HeCF4" / filename,
        project_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"No encuentro {filename}. Rutas probadas:\n{tried}")


def _read_csv_spectrum(project_root: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    path = _annotated_input_path(project_root, filename)
    df = pd.read_csv(path, sep=";", decimal=",", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path} debe tener al menos dos columnas: wavelength; intensity")
    wavelength = df.iloc[:, 0].to_numpy(dtype=float)
    intensity = df.iloc[:, 1].to_numpy(dtype=float)
    return _preprocess_spectrum(wavelength, intensity)


def _select_raw_candidate(
    project_root: Path,
    raw_cache: dict[str, pd.DataFrame],
    gas: str,
    concentration_percent: float,
    pressure_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    if gas not in raw_cache:
        raw_cache[gas] = read_raw_csv(project_root, gas)
    sub = select_raw_spectrum(
        raw_cache[gas],
        gas=gas,
        concentration_percent=concentration_percent,
        pressure_bar=pressure_bar,
        spectrum_column=cfg.RAW_PLOT_SPECTRUM_COLUMN,
    )
    if sub.empty:
        raise RuntimeError(f"Sin raw para {gas}, {concentration_percent:g}%, {pressure_bar:g} bar")
    return _preprocess_spectrum(
        sub["wavelength_nm"].to_numpy(dtype=float),
        sub["intensity_raw"].to_numpy(dtype=float),
    )


def _load_spectrum(project_root: Path, raw_cache: dict[str, pd.DataFrame], source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if source["type"] == "csv":
        return _read_csv_spectrum(project_root, str(source["filename"]))
    if source["type"] != "raw":
        raise ValueError(f"source type no soportado: {source['type']!r}")

    errors = []
    for gas, concentration, pressure in source["candidates"]:
        try:
            return _select_raw_candidate(project_root, raw_cache, gas, float(concentration), float(pressure))
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("No pude cargar ningún candidato raw:\n" + "\n".join(f"  - {e}" for e in errors))


def _find_peak_near(wavelength: np.ndarray, intensity: np.ndarray, x_guess_nm: float, window_nm: float) -> tuple[float, float]:
    mask = (
        np.isfinite(wavelength)
        & np.isfinite(intensity)
        & (wavelength >= x_guess_nm - window_nm)
        & (wavelength <= x_guess_nm + window_nm)
    )
    if np.any(mask):
        local_x = wavelength[mask]
        local_y = intensity[mask]
        idx = int(np.nanargmax(local_y))
        return float(local_x[idx]), float(local_y[idx])
    return float(x_guess_nm), float(np.interp(x_guess_nm, wavelength, intensity))


def _annotate_peak(ax, wavelength: np.ndarray, intensity: np.ndarray, annotation: dict[str, Any], x_range_nm: tuple[float, float]) -> None:
    x_peak, y_peak = _find_peak_near(
        wavelength,
        intensity,
        float(annotation["x_guess_nm"]),
        float(annotation.get("window_nm", 10.0)),
    )
    ymax = float(np.nanmax(intensity)) if intensity.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0.0:
        ymax = 1.0

    x_min, x_max = x_range_nm
    x_text = x_peak + float(annotation.get("dx_nm", 0.0))
    ha = annotation.get("ha", "left")
    margin_nm = float(annotation.get("margin_nm", 25.0))
    if x_text > x_max - margin_nm:
        x_text = x_peak - abs(float(annotation.get("dx_nm", 35.0)))
        ha = "right"
    elif x_text < x_min + margin_nm:
        x_text = x_peak + abs(float(annotation.get("dx_nm", 35.0)))
        ha = "left"
    y_text = y_peak + float(annotation.get("dy_frac", 0.10)) * ymax

    bbox_style = annotation.get(
        "bbox",
        {"boxstyle": "round,pad=0.2", "fc": "white", "ec": ANNOTATION_COLOR, "alpha": 0.85},
    )
    if annotation.get("arrow", True):
        ax.annotate(
            annotation["label"],
            xy=(x_peak, y_peak),
            xytext=(x_text, y_text),
            textcoords="data",
            color=annotation.get("color", ANNOTATION_COLOR),
            fontsize=int(annotation.get("fontsize", 13)),
            ha=ha,
            va=annotation.get("va", "bottom"),
            arrowprops=annotation.get(
                "arrowprops",
                {"arrowstyle": "->", "lw": 1.1, "color": annotation.get("color", ANNOTATION_COLOR), "shrinkA": 0, "shrinkB": 0},
            ),
            bbox=bbox_style,
            clip_on=False,
            annotation_clip=False,
        )
    else:
        ax.text(
            x_text,
            y_text,
            annotation["label"],
            color=annotation.get("color", ANNOTATION_COLOR),
            fontsize=int(annotation.get("fontsize", 13)),
            ha=ha,
            va=annotation.get("va", "bottom"),
            bbox=bbox_style,
            clip_on=False,
        )


def _plot_one_annotated(outdir: Path, spec: dict[str, Any], wavelength: np.ndarray, intensity: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=spec.get("figsize", (10.5, 4.8)))
    ax.plot(
        wavelength,
        intensity,
        color=spec.get("color", "tab:blue"),
        lw=float(spec.get("lw", 2.0)),
        alpha=float(spec.get("alpha", 1.0)),
        linestyle="-",
        label=spec.get("label", "1 bar"),
    )

    ymax = float(np.nanmax(intensity)) if intensity.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0.0:
        ymax = 1.0
    x_range_nm = tuple(spec["x_range_nm"])
    ax.set_xlim(*x_range_nm)
    ax.set_ylim(0.0, float(spec.get("y_margin_factor", 1.50)) * ymax)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(spec["title"])

    for annotation in spec.get("annotations", []):
        _annotate_peak(ax, wavelength, intensity, annotation, x_range_nm=x_range_nm)

    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.15, top=0.88)

    pdf_path = outdir / "plots" / "annotated" / str(spec["output_pdf"])
    ensure_parent(pdf_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[spectra] annotated PDF: {pdf_path}")
    return pdf_path


def run_annotated_figures(project_root: Path, outdir: Path) -> dict[str, Path]:
    _configure_annotated_style()
    raw_cache: dict[str, pd.DataFrame] = {}
    out: dict[str, Path] = {}

    for spec in ANNOTATED_SPECS:
        print(f"[spectra] annotated {spec['name']}")
        wavelength, intensity = _load_spectrum(project_root, raw_cache, spec["source"])
        csv_path = outdir / "csv" / "annotated" / f"{spec['name']}.csv"
        ensure_parent(csv_path)
        pd.DataFrame({"wavelength_nm": wavelength, "intensity_raw": intensity}).to_csv(csv_path, index=False)
        out[str(spec["name"])] = _plot_one_annotated(outdir, spec, wavelength, intensity)

    return out
