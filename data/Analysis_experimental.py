from __future__ import annotations

import importlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

"""
Analysis_experimental.py
------------------------
Script autónomo para leer los pickles experimentales y exportar CSVs listos
para pandas.read_csv / Typst load-csv.

Uso:
    python Analysis_experimental.py

Nota:
    Los pickles originales suelen necesitar dill. Si no está instalado, el
    script intenta usar pickle, pero para esos ficheros concretos puede hacer
    falta instalar dill en el entorno donde se ejecute.
"""


ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_DIR = ROOT_DIR / "Experimental"


# =============================================================================
# LIMPIEZA EXPERIMENTAL
# =============================================================================

# Los CSV originales/pickles no se modifican. Estas opciones solo afectan a los
# CSV exportados por este script. Son deliberadamente conservadoras: no hacen
# cortes físicos ni rechazos dependientes de modelo.
ABSOLUTE_UNCERTAINTIES = True
NEGATIVE_YIELDS_TO_NAN = True
FILL_MISSING_WITH_ZERO = False


def _to_numeric_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _clean_yield_and_errors(
    out: pd.DataFrame,
    *,
    value_col: str,
    err_col: str,
    err_stat_col: str | None = None,
    err_syst_col: str | None = None,
) -> None:
    """Clean one pressure column in-place for CSV export.

    - uncertainties are magnitudes, so Err/ErrStat/ErrSyst are converted to abs();
    - if both stat and syst errors exist, Err is recomputed as quadrature;
    - negative yields are marked as NaN, not converted to fake zeros;
    - errors associated with rejected/missing yields are also set to NaN.
    """

    if value_col not in out.columns:
        return

    out[value_col] = _to_numeric_array(out[value_col])

    error_cols = [c for c in (err_col, err_stat_col, err_syst_col) if c and c in out.columns]
    if ABSOLUTE_UNCERTAINTIES:
        for col in error_cols:
            out[col] = np.abs(_to_numeric_array(out[col]))

    if err_stat_col in out.columns and err_syst_col in out.columns:
        stat = np.abs(_to_numeric_array(out[err_stat_col]))
        syst = np.abs(_to_numeric_array(out[err_syst_col]))
        out[err_col] = np.sqrt(stat**2 + syst**2)
    elif err_col in out.columns:
        out[err_col] = np.abs(_to_numeric_array(out[err_col]))

    y = _to_numeric_array(out[value_col])
    bad = ~np.isfinite(y)
    if NEGATIVE_YIELDS_TO_NAN:
        bad |= y < 0.0

    out.loc[bad, value_col] = np.nan
    for col in error_cols:
        out.loc[bad, col] = np.nan
    if err_col in out.columns:
        out.loc[bad, err_col] = np.nan


def clean_exported_yield_table(out: pd.DataFrame, pressures: Iterable[float]) -> pd.DataFrame:
    cleaned = out.copy()
    for pressure in pressures:
        _clean_yield_and_errors(
            cleaned,
            value_col=pressure_label(pressure),
            err_col=err_label(pressure),
            err_stat_col=err_stat_label(pressure),
            err_syst_col=err_syst_label(pressure),
        )
    if FILL_MISSING_WITH_ZERO:
        cleaned = cleaned.fillna(0.0)
    return cleaned


# =============================================================================
# CONFIGURACIÓN
# =============================================================================


@dataclass(frozen=True)
class ExperimentalRunConfig:
    name: str
    pickle_path: Path
    output_dir: Path
    yields: tuple[object, ...]
    pressures: tuple[float, ...] | None = None
    output_concentration_name: str | None = None
    yield_mode: str = "auto"


RUNS: tuple[ExperimentalRunConfig, ...] = (
    ExperimentalRunConfig(
        name="ArCF4_UV_VIS",
        pickle_path=EXPERIMENTAL_DIR / "ArCF4" / "CF4_data.pkl", # _pure_CF4_nomalised.pkl",
        output_dir=EXPERIMENTAL_DIR / "ArCF4" / "csv",
        yields=("UV", "vis"),
        pressures=None,
        output_concentration_name="fCF4",
        yield_mode="direct",
    ),
    ExperimentalRunConfig(
        name="ArCF4_IR",
        pickle_path=EXPERIMENTAL_DIR / "ArCF4" / "CF4_data.pkl", # _pure_CF4_nomalised.pkl",
        output_dir=EXPERIMENTAL_DIR / "ArCF4" / "csv",
        yields=(696, 727, 750, 763, 772),
        pressures=None,
        output_concentration_name="fCF4",
        yield_mode="ir",
    ),
    ExperimentalRunConfig(
        name="ArN2_UV",
        pickle_path=EXPERIMENTAL_DIR / "ArN2" / "N2_data.pkl", # _pure_CF4_nomalised.pkl",
        output_dir=EXPERIMENTAL_DIR / "ArN2" / "csv",
        yields=("yield_N2",),
        pressures=None,
        output_concentration_name="fN2",
        yield_mode="auto",
    ),
    ExperimentalRunConfig(
        name="ArN2_IR",
        pickle_path=EXPERIMENTAL_DIR / "ArN2" / "N2_data.pkl", # _pure_CF4_nomalised.pkl",
        output_dir=EXPERIMENTAL_DIR / "ArN2" / "csv",
        yields=(696, 727, 750, 763, 772),
        pressures=None,
        output_concentration_name="fN2",
        yield_mode="ir",
    ),
)


# =============================================================================
# COMPATIBILIDAD PICKLE / DILL
# =============================================================================


def patch_scipy_special_for_old_pickles() -> None:
    """
    Algunos pickles guardados con versiones concretas de scipy buscan símbolos
    dentro de scipy.special._special_ufuncs. Si scipy existe, inyectamos aliases.
    Si no existe, no hacemos nada.
    """
    try:
        import scipy.special  # type: ignore

        special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")
    except Exception:
        return

    for name in ("erf", "erfc", "erfi", "gamma", "lgamma", "wofz"):
        if not hasattr(special_ufuncs, name) and hasattr(scipy.special, name):
            setattr(special_ufuncs, name, getattr(scipy.special, name))


def load_pickle(path: Path):
    patch_scipy_special_for_old_pickles()

    try:
        import dill  # type: ignore

        loader = dill
    except ModuleNotFoundError:
        loader = pickle

    with path.open("rb") as f:
        return loader.load(f)


# =============================================================================
# EXTRACCIÓN ROBUSTA DE COLUMNAS / DICCIONARIOS
# =============================================================================


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No encontré {what}. Probé {list(candidates)}. Columnas: {list(df.columns)}")


def is_nan_like(x) -> bool:
    try:
        out = pd.isna(x)
        return bool(out) if isinstance(out, (bool, np.bool_)) else False
    except Exception:
        return False


def is_missing_value(x) -> bool:
    return x is None or is_nan_like(x)


def extract_item(obj, key):
    if obj is None or is_nan_like(obj):
        return np.nan

    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        if isinstance(key, str) and key.strip().isdigit() and int(key.strip()) in obj:
            return obj[int(key.strip())]
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        return lower_map.get(str(key).lower(), np.nan)

    if hasattr(obj, "get") and not isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return obj.get(key, np.nan)
        except Exception:
            pass

    try:
        return obj[key]
    except Exception:
        pass

    if isinstance(key, str) and key.strip().isdigit():
        try:
            return obj[int(key.strip())]
        except Exception:
            pass

    return np.nan


def extract_from_nested_dict(obj, outer_key, inner_key):
    outer = extract_item(obj, outer_key)
    if is_missing_value(outer):
        return np.nan
    return extract_item(outer, inner_key)


def first_non_null(values):
    for value in values:
        if not is_missing_value(value):
            return value
    return None


def normalize_uncertainty_mode(mode: str) -> str:
    aliases = {
        "stat": "stadistic",
        "stats": "stadistic",
        "stadistic": "stadistic",
        "statistic": "stadistic",
        "statistical": "stadistic",
        "estadistico": "stadistic",
        "estadistica": "stadistic",
        "syst": "sistematic",
        "sis": "sistematic",
        "sistematic": "sistematic",
        "systematic": "sistematic",
        "sistematico": "sistematic",
        "sistematica": "sistematic",
        "all": "all",
        "combined": "all",
        "total": "all",
    }
    out = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    if out not in {"stadistic", "sistematic", "all"}:
        raise ValueError("uncertainty_mode debe ser 'stadistic', 'sistematic' o 'all'")
    return out


def normalize_yield_mode(mode: str) -> str:
    aliases = {
        "auto": "auto",
        "direct": "direct",
        "directo": "direct",
        "uv": "direct",
        "vis": "direct",
        "visible": "direct",
        "ir": "ir",
        "infrared": "ir",
        "infrarrojo": "ir",
    }
    out = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    if out not in {"auto", "direct", "ir"}:
        raise ValueError("yield_mode debe ser 'auto', 'direct' o 'ir'")
    return out


def normalize_yield_name_for_zonas(yield_name):
    yn = str(yield_name).strip().lower()
    if yn in {"yield_n2", "n2", "total_n2", "yield n2"}:
        return "uv"
    return yield_name


def error_candidates(mode: str, schema: str = "new", is_n2: bool = False) -> list[str]:
    if schema == "old":
        mapping = {
            "stadistic": ["u_yields_zonas_stat", "uyields_estadistico", "u_yields_estadistico"],
            "sistematic": [
                "u_yields_zonas_sis",
                "uyields_sistematico",
                "u_yields_sistematico",
                "uyields_systematic",
                "u_yields_systematic",
                "uyields_cal",
            ],
            "all": ["u_yields_zonas"],
        }
        return mapping[mode]

    if is_n2:
        mapping = {
            "stadistic": ["u_yield_n2_estadistico"],
            "sistematic": ["u_yield_n2_sistematico", "u_yield_n2_systematic", "u_yield_n2_cal"],
            "all": ["u_yield_n2_combined"],
        }
        return mapping[mode]

    mapping = {
        "stadistic": ["u_yields_estadistico"],
        "sistematic": ["u_yields_sistematico", "u_yields_systematic", "u_yields_cal", "u_yields_picos_cal"],
        "all": ["u_yields_picos"],
    }
    return mapping[mode]


def extract_yield_from_zonas(obj, yield_name, yield_mode: str):
    yield_mode = normalize_yield_mode(yield_mode)

    if yield_mode == "ir":
        return extract_from_nested_dict(obj, "ir", yield_name)
    if yield_mode == "direct":
        return extract_item(obj, yield_name)

    direct = extract_item(obj, yield_name)
    if not is_missing_value(direct):
        return direct
    return extract_from_nested_dict(obj, "ir", yield_name)


def get_series_for_yield(
    df_pressure: pd.DataFrame,
    conc_col: str,
    yield_name,
    uncertainty_mode: str,
    yield_mode: str,
) -> tuple[pd.Series, pd.Series]:
    uncertainty_mode = normalize_uncertainty_mode(uncertainty_mode)
    yield_mode = normalize_yield_mode(yield_mode)

    if "yields_zonas" in df_pressure.columns:
        zone_yield_name = normalize_yield_name_for_zonas(yield_name)
        value_col = "yields_zonas"
        err_col = find_first_column(
            df_pressure,
            error_candidates(uncertainty_mode, schema="old"),
            f"error {uncertainty_mode} esquema viejo",
        )

        s = df_pressure.set_index(conc_col)[value_col].apply(
            lambda d: extract_yield_from_zonas(d, zone_yield_name, yield_mode)
        )
        err = df_pressure.set_index(conc_col)[err_col].apply(
            lambda d: extract_yield_from_zonas(d, zone_yield_name, yield_mode)
        )

        if s.isna().all():
            example = first_non_null(df_pressure[value_col].tolist())
            raise KeyError(f"No aparece el yield {yield_name!r}. Ejemplo de claves: {example}")

        return s, err

    yn = str(yield_name).strip().lower()
    if yn in {"yield_n2", "n2", "total_n2", "yield n2"}:
        value_col = find_first_column(df_pressure, ["yield_N2"], "yield_N2")
        err_col = find_first_column(
            df_pressure,
            error_candidates(uncertainty_mode, schema="new", is_n2=True),
            f"error {uncertainty_mode} para yield_N2",
        )
        return df_pressure.set_index(conc_col)[value_col], df_pressure.set_index(conc_col)[err_col]

    value_col = find_first_column(df_pressure, ["yields_picos"], "yields_picos")
    err_col = find_first_column(
        df_pressure,
        error_candidates(uncertainty_mode, schema="new", is_n2=False),
        f"error {uncertainty_mode} para yields_picos",
    )
    s = df_pressure.set_index(conc_col)[value_col].apply(lambda d: extract_item(d, yield_name))
    err = df_pressure.set_index(conc_col)[err_col].apply(lambda d: extract_item(d, yield_name))
    return s, err


def detect_pressure_column(df: pd.DataFrame) -> str:
    return find_first_column(df, ["presion", "presiones", "P (bar)", "pressure", "Pressure"], "presión")


def detect_concentration_column(df: pd.DataFrame) -> str:
    return find_first_column(
        df,
        [
            "concentracion",
            "concentraciones",
            "N2 concentration (%)",
            "CF4 concentration (%)",
            "concentration_N2",
            "concentration_CF4",
            "concentration",
            "fN2",
            "fCF4",
        ],
        "concentración",
    )


def pressure_label(p: float) -> str:
    return f"{float(p):g}bar"


def err_label(p: float) -> str:
    return f"Err {pressure_label(p)}"


def err_stat_label(p: float) -> str:
    return f"ErrStat {pressure_label(p)}"


def err_syst_label(p: float) -> str:
    return f"ErrSyst {pressure_label(p)}"


def analyse_experimental_run(config: ExperimentalRunConfig) -> dict[str, pd.DataFrame]:
    if not config.pickle_path.is_file():
        raise FileNotFoundError(f"No existe {config.pickle_path}")

    df = load_pickle(config.pickle_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{config.pickle_path} no contiene un pandas.DataFrame, sino {type(df)}")

    pressure_col = detect_pressure_column(df)
    conc_col = detect_concentration_column(df)

    pressures = config.pressures
    if pressures is None:
        pressures = tuple(sorted(pd.to_numeric(df[pressure_col], errors="coerce").dropna().unique()))

    df_ref = df.loc[pd.to_numeric(df[pressure_col], errors="coerce") == float(pressures[0])].copy()
    concentrations = df_ref[conc_col].to_numpy()

    try:
        target_index = np.round(np.asarray(concentrations, dtype=float), 8)
        numeric_index = True
    except Exception:
        target_index = concentrations
        numeric_index = False

    output_conc_name = config.output_concentration_name or conc_col
    config.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, pd.DataFrame] = {}

    for yield_name in config.yields:
        out = pd.DataFrame({output_conc_name: concentrations})

        for pressure in pressures:
            df_pressure = df.loc[pd.to_numeric(df[pressure_col], errors="coerce") == float(pressure)].copy()

            col = pressure_label(pressure)
            ecol = err_label(pressure)
            estat = err_stat_label(pressure)
            esyst = err_syst_label(pressure)

            if df_pressure.empty:
                out[col] = np.nan
                out[ecol] = np.nan
                out[estat] = np.nan
                out[esyst] = np.nan
                continue

            series, err_all = get_series_for_yield(
                df_pressure=df_pressure,
                conc_col=conc_col,
                yield_name=yield_name,
                uncertainty_mode="all",
                yield_mode=config.yield_mode,
            )
            _, err_stat = get_series_for_yield(
                df_pressure=df_pressure,
                conc_col=conc_col,
                yield_name=yield_name,
                uncertainty_mode="stadistic",
                yield_mode=config.yield_mode,
            )
            _, err_syst = get_series_for_yield(
                df_pressure=df_pressure,
                conc_col=conc_col,
                yield_name=yield_name,
                uncertainty_mode="sistematic",
                yield_mode=config.yield_mode,
            )

            if numeric_index:
                series.index = np.round(series.index.astype(float), 8)
                err_all.index = np.round(err_all.index.astype(float), 8)
                err_stat.index = np.round(err_stat.index.astype(float), 8)
                err_syst.index = np.round(err_syst.index.astype(float), 8)

            out[col] = pd.Series(series).reindex(target_index).to_numpy()
            out[ecol] = pd.Series(err_all).reindex(target_index).to_numpy()
            out[estat] = pd.Series(err_stat).reindex(target_index).to_numpy()
            out[esyst] = pd.Series(err_syst).reindex(target_index).to_numpy()

        out = clean_exported_yield_table(out, pressures)
        safe_name = str(yield_name).replace("/", "_")
        out_path = config.output_dir / f"{safe_name}.csv"
        out.to_csv(out_path, index=False)
        outputs[safe_name] = out
        print(f"✅ {config.name}: {out_path.relative_to(ROOT_DIR)}")

    return outputs


def main() -> None:
    for config in RUNS:
        analyse_experimental_run(config)


if __name__ == "__main__":
    main()
