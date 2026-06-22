from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .fit_types import DatasetSpec


def project_root_from_file(path: str | Path) -> Path:
    p = Path(path).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").is_dir() and (parent / "primary_fits").is_dir():
            return parent
    return p.parents[1]


def ensure_project_paths(project_root: Path) -> None:
    for folder in ("models", "data", "primary_fits"):
        path = str(project_root / folder)
        if path not in sys.path:
            sys.path.insert(0, path)


def pressure_label(p: float) -> str:
    return f"{float(p):g}bar"


def error_label(p: float) -> str:
    return f"Err {pressure_label(p)}"


def stat_error_label(p: float) -> str:
    return f"ErrStat {pressure_label(p)}"


def syst_error_label(p: float) -> str:
    return f"ErrSyst {pressure_label(p)}"


def _canonicalize_pressure_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        s = str(col).strip()

        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*bar$", s)
        if m:
            rename[col] = pressure_label(float(m.group(1)))
            continue

        m = re.match(r"^Err\s+([0-9]+(?:\.[0-9]+)?)\s*bar$", s)
        if m:
            rename[col] = error_label(float(m.group(1)))
            continue

        m = re.match(
            r"^Err(?:Stat|_stat| stat|Statistical| statistical)\s+([0-9]+(?:\.[0-9]+)?)\s*bar$",
            s,
            flags=re.IGNORECASE,
        )
        if m:
            rename[col] = stat_error_label(float(m.group(1)))
            continue

        m = re.match(
            r"^Err(?:Syst|Sys|_syst|_sys| syst| sys|Systematic| systematic)\s+([0-9]+(?:\.[0-9]+)?)\s*bar$",
            s,
            flags=re.IGNORECASE,
        )
        if m:
            rename[col] = syst_error_label(float(m.group(1)))

    return df.rename(columns=rename)


def _apply_concentration_window(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    max_percent = getattr(spec, "max_concentration_percent", None)
    if max_percent is None:
        return df

    out = df.copy()
    x = pd.to_numeric(out[spec.x_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & (x < float(max_percent))
    return out.loc[mask].reset_index(drop=True)


def _selected_columns(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    cols: list[str] = []
    if spec.x_col in df.columns:
        cols.append(spec.x_col)

    for pressure in spec.pressures:
        for col in (
            pressure_label(pressure),
            error_label(pressure),
            stat_error_label(pressure),
            syst_error_label(pressure),
        ):
            if col in df.columns and col not in cols:
                cols.append(col)

    if spec.keep_columns is not None:
        for col in spec.keep_columns:
            if col in df.columns and col not in cols:
                cols.append(col)

    return cols


def _read_csv_dataset(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el CSV experimental {path}. "
            "Ejecuta antes data/Analysis_experimental.py o indica la ruta correcta en DatasetSpec."
        )

    df = _canonicalize_pressure_columns(pd.read_csv(path))
    if spec.x_col not in df.columns:
        raise KeyError(f"{path} no contiene la columna x_col={spec.x_col!r}. Columnas: {list(df.columns)}")

    df = _apply_concentration_window(df, spec)

    for pressure in spec.pressures:
        col = pressure_label(pressure)
        err = error_label(pressure)
        err_stat = stat_error_label(pressure)
        err_syst = syst_error_label(pressure)

        if col not in df.columns:
            raise KeyError(f"{path} no contiene la columna de presión {col!r}.")
        if err_stat not in df.columns:
            raise KeyError(f"{path} no contiene la columna de error estadístico {err_stat!r}.")
        if err_syst not in df.columns:
            raise KeyError(f"{path} no contiene la columna de error sistemático {err_syst!r}.")

        # Las columnas de error son incertidumbres. Si el CSV conserva un
        # desplazamiento sistemático con signo, el ajuste usa su módulo.
        df[err_stat] = np.abs(pd.to_numeric(df[err_stat], errors="coerce").to_numpy(dtype=float))
        df[err_syst] = np.abs(pd.to_numeric(df[err_syst], errors="coerce").to_numpy(dtype=float))

        if err in df.columns:
            df[err] = np.abs(pd.to_numeric(df[err], errors="coerce").to_numpy(dtype=float))
        else:
            df[err] = np.sqrt(
                df[err_stat].to_numpy(dtype=float) ** 2
                + df[err_syst].to_numpy(dtype=float) ** 2
            )

    return df.reset_index(drop=True)


def apply_w_scaling(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    if spec.w_function is None:
        return df

    out = df.copy()
    x = out[spec.output_concentration_name].to_numpy(dtype=float) * spec.w_input_scale
    factor = (1.0 / np.asarray(spec.w_function(x), dtype=float))[:, None]

    cols = []
    for pressure in spec.pressures:
        cols.append(pressure_label(pressure))
        cols.append(error_label(pressure))
        cols.append(stat_error_label(pressure))
        cols.append(syst_error_label(pressure))
    cols = [c for c in cols if c in out.columns]

    if cols:
        out[cols] = out[cols].to_numpy(dtype=float) * factor
    return out


def load_dataset_triplet(project_root: Path, spec: DatasetSpec) -> dict[str, pd.DataFrame]:
    nominal = _read_csv_dataset(spec.csv_path, spec)

    if spec.preprocess_before_w is not None:
        nominal = spec.preprocess_before_w(nominal)

    stat = nominal.copy(deep=True)
    syst = nominal.copy(deep=True)
    for pressure in spec.pressures:
        err = error_label(pressure)
        stat[err] = stat[stat_error_label(pressure)]
        syst[err] = syst[syst_error_label(pressure)]

    nominal = apply_w_scaling(nominal, spec)
    stat = apply_w_scaling(stat, spec)
    syst = apply_w_scaling(syst, spec)

    if spec.preprocess is not None:
        nominal = spec.preprocess(nominal)
        stat = spec.preprocess(stat)
        syst = spec.preprocess(syst)

    columns = _selected_columns(nominal, spec)
    nominal = nominal.loc[:, columns].copy()
    stat = stat.loc[:, columns].copy()
    syst = syst.loc[:, columns].copy()

    return {"all": nominal, "stat": stat, "syst": syst}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def drop_concentration_value(conc_col: str, value: float, *, atol: float = 1e-12):
    """Return a preprocessor that removes rows with conc_col ~= value."""

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        x = out[conc_col].to_numpy(dtype=float)
        return out.loc[~np.isclose(x, float(value), atol=atol)].copy()

    return preprocess


def move_concentration_value(
    conc_col: str,
    old_value: float,
    new_value: float,
    *,
    atol: float = 1e-12,
):
    """Return a preprocessor that moves one concentration value to another."""

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        x = out[conc_col].to_numpy(dtype=float)
        mask = np.isclose(x, float(old_value), atol=atol)
        out.loc[mask, conc_col] = float(new_value)
        return out

    return preprocess
