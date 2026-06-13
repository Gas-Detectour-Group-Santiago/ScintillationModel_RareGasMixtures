from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectra_units import (
    repo_root_from_script,
    safe_dill_load,
    setup_science_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = repo_root_from_script(__file__)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except Exception:
            key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            out.append(path.expanduser())
    return out


def _parents_with_cwd() -> list[Path]:
    cwd = Path.cwd().resolve()
    paths = [SCRIPT_DIR, *SCRIPT_DIR.parents, cwd, *cwd.parents]
    return _unique_paths(paths)


def find_data_dir() -> Path:
    """
    Locate the project data directory without relying on a fixed layout.

    You can force it with:
        export SPECTRA_DATA_DIR=/path/to/data
    """
    env_path = os.environ.get("SPECTRA_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    for base in _parents_with_cwd():
        candidates = [base / "data", base]
        for candidate in candidates:
            if (candidate / "Experimental").exists():
                return candidate

    return ROOT_DIR / "data"


DATA_DIR = find_data_dir()
OUT_DIR = Path(os.environ.get("SPECTRA_OUT_DIR", SCRIPT_DIR / "raw_spectra")).expanduser().resolve()


DEFAULT_PRESSURES_BAR = [1.0]
DEFAULT_FIGSIZE = (10.5, 4.8)
DEFAULT_PRESSURE_STYLES = {
    1.0: {"line": "tab:blue", "label": "1 bar"},
}
DEFAULT_ANNOTATION_COLOR = "0.15"


def setup_raw_spectrum_style() -> None:
    setup_science_style(use_grid=False)

    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
    })

    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_paths(path: str | Path, fallback_paths: list[str | Path] | None = None) -> list[Path]:
    raw = Path(path).expanduser()
    roots = _parents_with_cwd()

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, SCRIPT_DIR / raw])
        candidates.extend(base / raw for base in roots)

    if fallback_paths is not None:
        candidates.extend(Path(p).expanduser() for p in fallback_paths)

    name = raw.name
    candidates.extend([
        SCRIPT_DIR / name,
        Path.cwd() / name,
        DATA_DIR / name,
        DATA_DIR / "Experimental" / name,
        ROOT_DIR / name,
    ])

    parts = raw.parts
    if "data" in parts:
        i = parts.index("data")
        tail = Path(*parts[i + 1:])
        candidates.extend(base / "data" / tail for base in roots)
        candidates.append(DATA_DIR / tail)

    if "Experimental" in parts:
        i = parts.index("Experimental")
        tail = Path(*parts[i:])
        candidates.extend(base / "data" / tail for base in roots)
        candidates.append(DATA_DIR / Path(*parts[i + 1:]))

    return _unique_paths(candidates)


def resolve_existing_path(path: str | Path, fallback_paths: list[str | Path] | None = None) -> Path:
    candidates = _candidate_paths(path, fallback_paths)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Last-resort search by filename in the most plausible directories. This is
    # intentionally local, not a full filesystem search.
    filename = Path(path).name
    for base in _unique_paths([SCRIPT_DIR, Path.cwd(), DATA_DIR, ROOT_DIR]):
        if not base.exists() or not base.is_dir():
            continue
        try:
            for match in base.rglob(filename):
                if match.is_file():
                    return match.resolve()
        except PermissionError:
            continue

    tried = "\n".join(f"  - {candidate}" for candidate in candidates[:30])
    if len(candidates) > 30:
        tried += f"\n  ... and {len(candidates) - 30} more candidates"
    raise FileNotFoundError(
        "Could not find input spectrum file.\n"
        f"File requested: {path}\n"
        "Tip: keep the data/Experimental directory above this folder, or set "
        "SPECTRA_DATA_DIR=/path/to/data.\n"
        f"Tried:\n{tried}"
    )


def extract_spectrum_arrays_quiet(
    row: Any,
    preferred_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    for col in preferred_columns:
        if col not in row.index:
            continue

        value = row[col]
        if isinstance(value, dict) and "wavelength" in value and "intensity" in value:
            return (
                np.asarray(value["wavelength"], dtype=float),
                np.asarray(value["intensity"], dtype=float),
            )

    raise KeyError(f"No valid spectrum found. Tried columns: {preferred_columns}")


def preprocess_intensity(
    intensity: np.ndarray,
    *,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> np.ndarray:
    y = np.asarray(intensity, dtype=float).copy()

    if clip_negative_values:
        y = np.clip(y, 0.0, None)

    if normalize_to_max:
        ymax = np.nanmax(y)
        if np.isfinite(ymax) and ymax > 0.0:
            y = y / ymax

    return y


def get_spectrum_from_source(
    source_config: dict,
    pressure_bar: float,
    *,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    pkl_path = resolve_existing_path(
        source_config["pkl_path"],
        fallback_paths=source_config.get("fallback_paths"),
    )
    df = safe_dill_load(pkl_path)

    concentration_column = source_config["concentration_column"]
    pressure_column = source_config["pressure_column"]

    if concentration_column not in df.columns:
        raise KeyError(f"Column {concentration_column!r} not found in {pkl_path}.")
    if pressure_column not in df.columns:
        raise KeyError(f"Column {pressure_column!r} not found in {pkl_path}.")

    mask = np.isclose(
        df[concentration_column].astype(float),
        source_config["concentration_percent"],
    ) & np.isclose(
        df[pressure_column].astype(float),
        pressure_bar,
    )

    if not np.any(mask):
        return None

    row = df.loc[mask].iloc[0]
    wavelength, intensity = extract_spectrum_arrays_quiet(
        row,
        source_config["spectrum_columns"],
    )

    return (
        np.asarray(wavelength, dtype=float),
        preprocess_intensity(
            intensity,
            normalize_to_max=normalize_to_max,
            clip_negative_values=clip_negative_values,
        ),
    )


def load_spectra(
    source_config: dict,
    pressures_bar: list[float],
    *,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    spectra: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for pressure_bar in pressures_bar:
        spectrum = get_spectrum_from_source(
            source_config,
            pressure_bar,
            normalize_to_max=normalize_to_max,
            clip_negative_values=clip_negative_values,
        )

        if spectrum is None:
            raise RuntimeError(
                f"No spectrum found for concentration "
                f"{source_config['concentration_percent']:g}% at "
                f"{pressure_bar:g} bar."
            )

        spectra[pressure_bar] = spectrum

    return spectra


def load_spectra_with_fallback(
    source_configs: list[dict],
    pressures_bar: list[float],
    *,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], list[str]]:
    spectra: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    used_sources: list[str] = []
    skipped_sources: list[str] = []

    for pressure_bar in pressures_bar:
        result = None
        used_source = None

        for source_config in source_configs:
            try:
                result = get_spectrum_from_source(
                    source_config,
                    pressure_bar,
                    normalize_to_max=normalize_to_max,
                    clip_negative_values=clip_negative_values,
                )
            except FileNotFoundError as exc:
                skipped_sources.append(f"{source_config.get('name', 'unnamed source')}: {exc}")
                result = None

            if result is not None:
                used_source = source_config.get("name", "unnamed source")
                break

        if result is None:
            details = "\n".join(skipped_sources[-3:])
            raise RuntimeError(f"No spectrum found at {pressure_bar:g} bar.\n{details}")

        spectra[pressure_bar] = result
        if used_source is not None:
            used_sources.append(used_source)

    return spectra, used_sources


def load_csv_spectrum(
    csv_path: str | Path,
    *,
    fallback_paths: list[str | Path] | None = None,
    sep: str = ";",
    decimal: str = ",",
    header: int | None = None,
    wavelength_column: int | str = 0,
    intensity_column: int | str = 1,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    csv_path = resolve_existing_path(csv_path, fallback_paths)

    df = pd.read_csv(
        csv_path,
        sep=sep,
        decimal=decimal,
        header=header,
    )

    wavelength = df[wavelength_column].to_numpy(dtype=float)
    intensity = df[intensity_column].to_numpy(dtype=float)

    finite = np.isfinite(wavelength) & np.isfinite(intensity)
    wavelength = wavelength[finite]
    intensity = intensity[finite]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    intensity = intensity[order]

    return (
        wavelength,
        preprocess_intensity(
            intensity,
            normalize_to_max=normalize_to_max,
            clip_negative_values=clip_negative_values,
        ),
    )


def load_csv_spectra(
    csv_sources: dict[float, dict],
    *,
    normalize_to_max: bool = False,
    clip_negative_values: bool = True,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    spectra: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for pressure_bar, csv_config in csv_sources.items():
        spectra[pressure_bar] = load_csv_spectrum(
            csv_config["csv_path"],
            fallback_paths=csv_config.get("fallback_paths"),
            sep=csv_config.get("sep", ";"),
            decimal=csv_config.get("decimal", ","),
            header=csv_config.get("header", None),
            wavelength_column=csv_config.get("wavelength_column", 0),
            intensity_column=csv_config.get("intensity_column", 1),
            normalize_to_max=normalize_to_max,
            clip_negative_values=clip_negative_values,
        )

    return spectra


def find_peak_near(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    x_guess_nm: float,
    window_nm: float = 10.0,
) -> tuple[float, float]:
    mask = (
        np.isfinite(wavelength)
        & np.isfinite(intensity)
        & (wavelength >= x_guess_nm - window_nm)
        & (wavelength <= x_guess_nm + window_nm)
    )

    if not np.any(mask):
        return float(x_guess_nm), float(np.interp(x_guess_nm, wavelength, intensity))

    local_x = wavelength[mask]
    local_y = intensity[mask]
    idx = int(np.nanargmax(local_y))

    return float(local_x[idx]), float(local_y[idx])


def find_peak_across_spectra(
    spectra: dict[float, tuple[np.ndarray, np.ndarray]],
    x_guess_nm: float,
    window_nm: float = 10.0,
) -> tuple[float, float, float]:
    best_pressure = np.nan
    best_x = float(x_guess_nm)
    best_y = -np.inf

    for pressure_bar, (wavelength, intensity) in spectra.items():
        x_peak, y_peak = find_peak_near(wavelength, intensity, x_guess_nm, window_nm)

        if np.isfinite(y_peak) and y_peak > best_y:
            best_pressure = float(pressure_bar)
            best_x = x_peak
            best_y = y_peak

    if not np.isfinite(best_y):
        best_y = 0.0

    return best_x, float(best_y), best_pressure


def annotate_peak(
    ax,
    spectra: dict[float, tuple[np.ndarray, np.ndarray]],
    annotation: dict,
    x_range_nm: tuple[float, float],
    color: str = DEFAULT_ANNOTATION_COLOR,
    annotation_fontsize: int = 13,
) -> None:
    x_peak, y_peak, _pressure_bar = find_peak_across_spectra(
        spectra,
        x_guess_nm=annotation["x_guess_nm"],
        window_nm=annotation.get("window_nm", 10.0),
    )

    global_ymax = max(float(np.nanmax(y)) for _, y in spectra.values())

    x_min, x_max = x_range_nm
    x_text = x_peak + annotation.get("dx_nm", 0.0)

    margin_nm = annotation.get("margin_nm", 25.0)
    ha = annotation.get("ha", "left")

    if x_text > x_max - margin_nm:
        x_text = x_peak - abs(annotation.get("dx_nm", 35.0))
        ha = "right"
    elif x_text < x_min + margin_nm:
        x_text = x_peak + abs(annotation.get("dx_nm", 35.0))
        ha = "left"

    y_text = y_peak + annotation.get("dy_frac", 0.10) * global_ymax

    bbox_style = annotation.get("bbox", {
        "boxstyle": "round,pad=0.2",
        "fc": "white",
        "ec": color,
        "alpha": 0.85,
    })

    if annotation.get("arrow", True):
        ax.annotate(
            annotation["label"],
            xy=(x_peak, y_peak),
            xytext=(x_text, y_text),
            textcoords="data",
            color=color,
            fontsize=annotation.get("fontsize", annotation_fontsize),
            ha=ha,
            va=annotation.get("va", "bottom"),
            arrowprops=annotation.get("arrowprops", {
                "arrowstyle": "->",
                "lw": 1.1,
                "color": color,
                "shrinkA": 0,
                "shrinkB": 0,
            }),
            bbox=bbox_style,
            clip_on=False,
            annotation_clip=False,
        )
    else:
        ax.text(
            x_text,
            y_text,
            annotation["label"],
            color=color,
            fontsize=annotation.get("fontsize", annotation_fontsize),
            ha=ha,
            va=annotation.get("va", "bottom"),
            bbox=bbox_style,
            clip_on=False,
        )


def style_axis(
    ax,
    title: str,
    x_range_nm: tuple[float, float],
    global_ymax: float,
    *,
    y_margin_factor: float = 1.50,
) -> None:
    ax.set_xlim(*x_range_nm)
    ax.set_ylim(0.0, y_margin_factor * global_ymax)

    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title)


def plot_raw_spectrum(
    *,
    spectra: dict[float, tuple[np.ndarray, np.ndarray]],
    annotations: list[dict],
    title: str,
    x_range_nm: tuple[float, float],
    output_name: str,
    pressures_bar: list[float],
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pressure_styles: dict[float, dict] | None = None,
    annotation_color: str = DEFAULT_ANNOTATION_COLOR,
    legend_fontsize: int = 14,
    y_margin_factor: float = 1.50,
) -> Path:
    if pressure_styles is None:
        pressure_styles = DEFAULT_PRESSURE_STYLES

    missing = [p for p in pressures_bar if p not in spectra]
    if missing:
        raise KeyError(f"Missing spectra for pressures: {missing}")

    fig, ax = plt.subplots(figsize=figsize)

    for pressure_bar in pressures_bar:
        wavelength, intensity = spectra[pressure_bar]
        style = pressure_styles.get(pressure_bar, {"line": "black"})

        ax.plot(
            wavelength,
            intensity,
            lw=style.get("lw", 2.0),
            color=style.get("line", "black"),
            ls=style.get("ls", "-"),
            label=style.get("label", rf"{pressure_bar:g} bar"),
        )

    global_ymax = max(float(np.nanmax(y)) for _, y in spectra.values())
    if not np.isfinite(global_ymax) or global_ymax <= 0.0:
        global_ymax = 1.0

    style_axis(
        ax,
        title,
        x_range_nm,
        global_ymax,
        y_margin_factor=y_margin_factor,
    )

    for annotation in annotations:
        annotate_peak(
            ax,
            spectra,
            annotation,
            x_range_nm=x_range_nm,
            color=annotation.get("color", annotation_color),
        )

    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)

    fig.subplots_adjust(
        left=0.07,
        right=0.96,
        bottom=0.15,
        top=0.88,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUT_DIR / output_name

    # Avoid bbox_inches="tight": it can crop the axis/annotations.
    fig.savefig(outpath)
    plt.close(fig)

    print(f"Saved figure to {outpath}")
    return outpath
