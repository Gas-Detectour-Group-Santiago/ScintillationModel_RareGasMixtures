from __future__ import annotations

import importlib
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from .paths import find_repo_root
from .units import (
    get_arcf4_total_yield_ph_per_electron,
    get_n2_total_yield_ph_per_electron,
    spectrum_shape_to_ph_per_MeV_nm,
    spectrum_shape_to_ph_per_e_nm,
)

SpectrumUnit = Literal["raw", "unit_area", "max_norm", "ph_per_e_nm", "ph_per_MeV_nm"]
SpectrumSource = Literal["pickle", "raw_csv"]

SCAN = "scan"


@dataclass(frozen=True)
class SpectrumSelector:
    """Select one experimental spectrum.

    ``concentration_percent`` and ``pressure_bar`` can be numbers or the string
    ``"scan"``. The scanner replaces ``"scan"`` by the current scan value.
    """

    gas: Literal["ArCF4", "ArN2"]
    concentration_percent: float | str
    pressure_bar: float | str
    spectrum_column: str = "mean_spectrum"
    source: SpectrumSource = "pickle"
    unit: SpectrumUnit = "ph_per_MeV_nm"
    include_ir_yield: bool = True
    clip_negative: bool = True
    column_fallbacks: tuple[str, ...] = ()

    def resolved(self, *, concentration_percent: float, pressure_bar: float) -> "SpectrumSelector":
        conc = concentration_percent if self.concentration_percent == SCAN else self.concentration_percent
        pres = pressure_bar if self.pressure_bar == SCAN else self.pressure_bar
        return replace(self, concentration_percent=float(conc), pressure_bar=float(pres))

    @property
    def preferred_columns(self) -> tuple[str, ...]:
        cols = (self.spectrum_column, *self.column_fallbacks)
        # Keep order while removing duplicates.
        out: list[str] = []
        for col in cols:
            if col and col not in out:
                out.append(col)
        return tuple(out)


@dataclass(frozen=True)
class SpectrumData:
    selector: SpectrumSelector
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    raw_intensity: np.ndarray
    total_yield_ph_per_electron: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GasConfig:
    pickle_path: Path
    raw_csv_path: Path
    concentration_columns: tuple[str, ...]
    pressure_columns: tuple[str, ...]
    default_spectrum_columns: tuple[str, ...]
    w_module: str
    w_function: str


class SpectrumProvider:
    """Load experimental spectra from pickles or long raw-spectrum CSVs."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = find_repo_root(root_dir)
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.root_dir / "models"
        if str(self.models_dir) not in sys.path:
            sys.path.insert(0, str(self.models_dir))

        self.configs: dict[str, GasConfig] = {
            "ArCF4": GasConfig(
                pickle_path=self.data_dir / "Experimental" / "ArCF4" / "CF4_data.pkl",
                raw_csv_path=self.data_dir / "spectra" / "ArCF4_raw_spectra.csv",
                concentration_columns=("concentracion", "concentration_CF4", "CF4 concentration (%)", "fCF4"),
                pressure_columns=("presion", "P (bar)", "pressure", "pressure_bar"),
                default_spectrum_columns=("mean_spectrum", "C1_spectrum", "C2_spectrum", "data(norm)"),
                w_module="ArCF4",
                w_function="ion_potential",
            ),
            "ArN2": GasConfig(
                pickle_path=self.data_dir / "Experimental" / "ArN2" / "N2_data.pkl",
                raw_csv_path=self.data_dir / "spectra" / "ArN2_raw_spectra.csv",
                concentration_columns=("N2 concentration (%)", "concentration_N2", "concentracion", "fN2"),
                pressure_columns=("P (bar)", "presion", "pressure", "pressure_bar"),
                default_spectrum_columns=(
                    "mean_spectrum",
                    "C1_spectrum",
                    "C2_spectrum",
                    "C1",
                    "C2",
                    "spectrum_new_cal",
                    "spectrum_old_cal",
                ),
                w_module="ArN2",
                w_function="W_ArN2",
            ),
        }
        self._pickle_cache: dict[Path, pd.DataFrame] = {}
        self._raw_csv_cache: dict[Path, pd.DataFrame] = {}
        self._w_cache: dict[str, Any] = {}

    def load(self, selector: SpectrumSelector) -> SpectrumData:
        if selector.gas not in self.configs:
            raise ValueError(f"Unknown gas {selector.gas!r}. Expected one of {sorted(self.configs)}")
        if selector.concentration_percent == SCAN or selector.pressure_bar == SCAN:
            raise ValueError("Selector still contains 'scan'. Resolve it before loading.")

        if selector.source == "pickle":
            return self._load_from_pickle(selector)
        if selector.source == "raw_csv":
            return self._load_from_raw_csv(selector)
        raise ValueError(f"Unknown source {selector.source!r}")

    def available_conditions(
        self,
        gas: Literal["ArCF4", "ArN2"],
        *,
        source: SpectrumSource = "pickle",
        spectrum_column: str | None = None,
    ) -> pd.DataFrame:
        """Return available concentration/pressure/spectrum-column combinations."""
        if source == "raw_csv":
            config = self.configs[gas]
            df = self._read_raw_csv(config.raw_csv_path)
            cols = ["gas_mixture", "spectrum_column", "concentration_percent", "pressure_bar"]
            out = df[cols].drop_duplicates().copy()
            if spectrum_column is not None:
                out = out[out["spectrum_column"].eq(spectrum_column)]
            return out.sort_values(["pressure_bar", "concentration_percent", "spectrum_column"]).reset_index(drop=True)

        config = self.configs[gas]
        df = self._read_pickle(config.pickle_path)
        conc_col = find_first_column(df, config.concentration_columns, "concentration")
        pressure_col = find_first_column(df, config.pressure_columns, "pressure")
        concentration_is_fraction = infer_concentration_is_fraction(conc_col, df[conc_col])

        rows: list[dict[str, Any]] = []
        columns_to_check = (spectrum_column,) if spectrum_column else config.default_spectrum_columns
        for _, row in df.iterrows():
            concentration_percent = normalise_concentration_percent(row[conc_col], is_fraction=concentration_is_fraction)
            pressure_bar = scalar_or_nan(row[pressure_col])
            for col in columns_to_check:
                if col in row.index and extract_spectrum_arrays(row[col]) is not None:
                    rows.append(
                        {
                            "gas_mixture": gas,
                            "spectrum_column": col,
                            "concentration_percent": concentration_percent,
                            "pressure_bar": pressure_bar,
                        }
                    )
        return pd.DataFrame(rows).drop_duplicates().sort_values(["pressure_bar", "concentration_percent", "spectrum_column"]).reset_index(drop=True)

    def _load_from_pickle(self, selector: SpectrumSelector) -> SpectrumData:
        config = self.configs[selector.gas]
        df = self._read_pickle(config.pickle_path)

        conc_col = find_first_column(df, config.concentration_columns, "concentration")
        pressure_col = find_first_column(df, config.pressure_columns, "pressure")
        concentration_is_fraction = infer_concentration_is_fraction(conc_col, df[conc_col])

        concentration_values = df[conc_col].apply(lambda x: normalise_concentration_percent(x, is_fraction=concentration_is_fraction))
        pressure_values = pd.to_numeric(df[pressure_col], errors="coerce")

        mask = np.isclose(concentration_values.astype(float), float(selector.concentration_percent)) & np.isclose(
            pressure_values.astype(float), float(selector.pressure_bar)
        )
        if not np.any(mask):
            raise KeyError(
                f"No {selector.gas} spectrum at c={selector.concentration_percent:g}% "
                f"and P={selector.pressure_bar:g} bar in {config.pickle_path}"
            )

        row = df.loc[mask].iloc[0]
        wavelength_nm, raw = get_spectrum_arrays(row, selector.preferred_columns or config.default_spectrum_columns)
        intensity, total_yield = self._convert_units(selector, wavelength_nm, raw, row)

        return SpectrumData(
            selector=selector,
            wavelength_nm=wavelength_nm,
            intensity=intensity,
            raw_intensity=raw,
            total_yield_ph_per_electron=total_yield,
            metadata={
                "source": "pickle",
                "source_path": str(config.pickle_path.relative_to(self.root_dir)),
                "concentration_column": conc_col,
                "pressure_column": pressure_col,
            },
        )

    def _load_from_raw_csv(self, selector: SpectrumSelector) -> SpectrumData:
        config = self.configs[selector.gas]
        csv_path = self._resolve_raw_csv_path(config.raw_csv_path)
        df = self._read_raw_csv(csv_path)
        mask = (
            df["gas_mixture"].eq(selector.gas)
            & np.isclose(df["concentration_percent"].astype(float), float(selector.concentration_percent))
            & np.isclose(df["pressure_bar"].astype(float), float(selector.pressure_bar))
        )
        columns = selector.preferred_columns or config.default_spectrum_columns
        mask &= df["spectrum_column"].isin(columns)
        subset = df.loc[mask].copy()
        if subset.empty:
            raise KeyError(
                f"No {selector.gas} raw-csv spectrum at c={selector.concentration_percent:g}% "
                f"P={selector.pressure_bar:g} bar, columns={columns} in {config.raw_csv_path}"
            )

        # Respect preferred order if several columns match.
        first_col = next(col for col in columns if col in set(subset["spectrum_column"]))
        subset = subset[subset["spectrum_column"].eq(first_col)].sort_values("wavelength_nm")
        wavelength_nm = subset["wavelength_nm"].to_numpy(dtype=float)
        raw = subset["intensity_raw"].to_numpy(dtype=float)

        if selector.unit in {"ph_per_e_nm", "ph_per_MeV_nm"}:
            raise ValueError(
                "raw_csv source does not contain integrated yields. Use unit='raw', 'unit_area' or 'max_norm', "
                "or use source='pickle' for calibrated ph/e-/ph/MeV spectra."
            )

        intensity, total_yield = self._convert_units(selector, wavelength_nm, raw, row=None)
        return SpectrumData(
            selector=selector,
            wavelength_nm=wavelength_nm,
            intensity=intensity,
            raw_intensity=raw,
            total_yield_ph_per_electron=total_yield,
            metadata={
                "source": "raw_csv",
                "source_path": _safe_relative_path(csv_path, self.root_dir),
                "spectrum_column_used": first_col,
            },
        )

    def _convert_units(
        self,
        selector: SpectrumSelector,
        wavelength_nm: np.ndarray,
        raw: np.ndarray,
        row,
    ) -> tuple[np.ndarray, float]:
        raw = np.asarray(raw, dtype=float)
        if selector.clip_negative:
            raw_for_norm = np.clip(raw, 0.0, None)
        else:
            raw_for_norm = raw

        if selector.unit == "raw":
            return raw_for_norm, np.nan

        area = float(np.trapezoid(raw_for_norm[np.argsort(wavelength_nm)], np.sort(wavelength_nm)))
        if selector.unit == "unit_area":
            if area <= 0.0 or not np.isfinite(area):
                return np.full_like(raw, np.nan, dtype=float), np.nan
            return raw_for_norm / area, np.nan

        if selector.unit == "max_norm":
            ymax = float(np.nanmax(raw_for_norm)) if raw_for_norm.size else np.nan
            if ymax <= 0.0 or not np.isfinite(ymax):
                return np.full_like(raw, np.nan, dtype=float), np.nan
            return raw_for_norm / ymax, np.nan

        if row is None:
            raise ValueError(f"Unit {selector.unit!r} needs a pickle row with integrated yields.")

        if selector.gas == "ArCF4":
            total_yield = get_arcf4_total_yield_ph_per_electron(row)
        elif selector.gas == "ArN2":
            total_yield = get_n2_total_yield_ph_per_electron(row, include_ir=selector.include_ir_yield)
        else:  # pragma: no cover
            raise ValueError(selector.gas)

        if selector.unit == "ph_per_e_nm":
            return (
                spectrum_shape_to_ph_per_e_nm(
                    wavelength_nm,
                    raw,
                    total_yield,
                    clip_negative=selector.clip_negative,
                ),
                total_yield,
            )

        if selector.unit == "ph_per_MeV_nm":
            w_func = self._w_function(selector.gas)
            return (
                spectrum_shape_to_ph_per_MeV_nm(
                    wavelength_nm,
                    raw,
                    total_yield,
                    float(selector.concentration_percent) / 100.0,
                    w_func,
                    clip_negative=selector.clip_negative,
                ),
                total_yield,
            )

        raise ValueError(f"Unknown unit {selector.unit!r}")

    def _w_function(self, gas: str):
        if gas not in self._w_cache:
            config = self.configs[gas]
            module = importlib.import_module(config.w_module)
            self._w_cache[gas] = getattr(module, config.w_function)
        return self._w_cache[gas]

    def _read_pickle(self, path: Path) -> pd.DataFrame:
        path = path.resolve()
        if path not in self._pickle_cache:
            self._pickle_cache[path] = load_pickle(path)
        df = self._pickle_cache[path]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            self._pickle_cache[path] = df
        return df

    def _read_raw_csv(self, path: Path) -> pd.DataFrame:
        path = self._resolve_raw_csv_path(path)
        if path not in self._raw_csv_cache:
            df = pd.read_csv(path)
            required = {
                "gas_mixture",
                "spectrum_column",
                "concentration_percent",
                "pressure_bar",
                "wavelength_nm",
                "intensity_raw",
            }
            missing = sorted(required.difference(df.columns))
            if missing:
                raise KeyError(f"Raw spectra CSV {path} is missing required columns: {missing}")
            self._raw_csv_cache[path] = df
        return self._raw_csv_cache[path]

    def _resolve_raw_csv_path(self, path: Path) -> Path:
        """Resolve spectra CSVs exported by data/Analysis_spectra.py.

        Preferred layout is ``data/spectra``.  A fallback to the historical
        ``data/Spectra`` directory is kept so the comparison code can consume
        existing exported CSVs without touching ``data/Analysis_spectra.py``.
        """
        candidates = [path]
        if path.parent.name == "spectra":
            candidates.append(path.parent.parent / "Spectra" / path.name)
        elif path.parent.name == "Spectra":
            candidates.append(path.parent.parent / "spectra" / path.name)

        # Last fallback: combined long CSV if per-gas files are not present.
        candidates.append(path.parent / "raw_spectra.csv")
        if path.parent.name == "spectra":
            candidates.append(path.parent.parent / "Spectra" / "raw_spectra.csv")
        elif path.parent.name == "Spectra":
            candidates.append(path.parent.parent / "spectra" / "raw_spectra.csv")

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

        tried = ", ".join(str(candidate) for candidate in seen)
        raise FileNotFoundError(f"Could not find exported raw spectra CSV. Tried: {tried}")


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_pickle(path: Path):
    patch_scipy_special_for_old_pickles()
    try:
        import dill  # type: ignore

        loader = dill
    except ModuleNotFoundError:
        loader = pickle

    with path.open("rb") as f:
        return loader.load(f)


def patch_scipy_special_for_old_pickles() -> None:
    try:
        import scipy.special  # type: ignore

        special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")
    except Exception:
        return

    for name in ("erf", "erfc", "erfi", "gamma", "lgamma", "wofz"):
        if not hasattr(special_ufuncs, name) and hasattr(scipy.special, name):
            setattr(special_ufuncs, name, getattr(scipy.special, name))


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {what}. Tried {list(candidates)}. Columns: {list(df.columns)}")


def scalar_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def infer_concentration_is_fraction(column_name: str, values: pd.Series) -> bool:
    name = column_name.lower().replace(" ", "")
    if "%" in column_name or "concentracion" in name or "concentration(%)" in name:
        return False
    if name in {"fcf4", "fn2"} or "fraction" in name:
        finite = pd.to_numeric(values, errors="coerce").dropna()
        return bool(len(finite) > 0 and finite.max() <= 1.0)
    return False


def normalise_concentration_percent(value: Any, *, is_fraction: bool) -> float:
    value = scalar_or_nan(value)
    if not np.isfinite(value):
        return np.nan
    return value * 100.0 if is_fraction else value


def is_nan_like(value: Any) -> bool:
    try:
        out = pd.isna(value)
        return bool(out) if isinstance(out, (bool, np.bool_)) else False
    except Exception:
        return False


def extract_spectrum_arrays(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if value is None or is_nan_like(value):
        return None

    if isinstance(value, dict):
        wave = value.get("wavelength", value.get("lambda", value.get("wavelength_nm")))
        intensity = value.get("intensity", value.get("raw", value.get("counts")))
        if wave is None or intensity is None:
            return None
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        wave, intensity = value[0], value[1]
    else:
        return None

    try:
        w = np.asarray(wave, dtype=float).ravel()
        y = np.asarray(intensity, dtype=float).ravel()
    except Exception:
        return None

    n = min(w.size, y.size)
    if n < 2:
        return None

    finite = np.isfinite(w[:n]) & np.isfinite(y[:n])
    if finite.sum() < 2:
        return None
    return w[:n][finite], y[:n][finite]


def get_spectrum_arrays(row: pd.Series, preferred_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    for col in preferred_columns:
        if col not in row.index:
            continue
        arrays = extract_spectrum_arrays(row[col])
        if arrays is not None:
            return arrays
    raise KeyError(f"No valid spectrum found. Tried columns: {preferred_columns}")
