from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .fit_io import error_label, pressure_label
from .fit_types import DatasetSpec, SystematicSource, ToySpec


def _copy_nominal_dict(nominal: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {k: v.copy(deep=True) for k, v in nominal.items()}


def make_stat_toy(
    nominal: dict[str, pd.DataFrame],
    stat_errors: dict[str, pd.DataFrame],
    specs: list[DatasetSpec],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    toy = _copy_nominal_dict(nominal)

    for spec in specs:
        df = toy[spec.key]
        err_df = stat_errors[spec.key]
        for pressure in spec.pressures:
            col = pressure_label(pressure)
            err_col = error_label(pressure)
            if col not in df.columns or err_col not in err_df.columns:
                continue
            sigma = err_df[err_col].to_numpy(dtype=float)
            shift = rng.normal(0.0, 1.0, size=len(df)) * sigma
            df[col] = df[col].to_numpy(dtype=float) + shift

    return toy


def _source_applies(source: SystematicSource, spec: DatasetSpec, pressure: float) -> bool:
    if source.datasets is not None and spec.key not in set(source.datasets):
        return False
    if source.pressures is not None and float(pressure) not in {float(p) for p in source.pressures}:
        return False
    return True


def make_syst_toy(
    nominal: dict[str, pd.DataFrame],
    syst_errors: dict[str, pd.DataFrame],
    specs: list[DatasetSpec],
    rng: np.random.Generator,
    sources: list[SystematicSource],
) -> dict[str, pd.DataFrame]:
    toy = _copy_nominal_dict(nominal)

    if not sources:
        sources = [SystematicSource(name="default_by_dataset", mode="by_dataset")]

    global_z: dict[str, float] = {}
    dataset_z: dict[tuple[str, str], float] = {}
    pressure_z: dict[tuple[str, str, float], float] = {}

    for source in sources:
        if source.mode == "global":
            global_z[source.name] = rng.normal()

    for spec in specs:
        df = toy[spec.key]
        err_df = syst_errors[spec.key]

        for pressure in spec.pressures:
            col = pressure_label(pressure)
            err_col = error_label(pressure)
            if col not in df.columns:
                continue

            total_shift = np.zeros(len(df), dtype=float)

            for source in sources:
                if not _source_applies(source, spec, pressure):
                    continue

                if source.mode == "global":
                    z = global_z[source.name]
                elif source.mode == "by_pressure":
                    key = (source.name, spec.key, float(pressure))
                    pressure_z.setdefault(key, rng.normal())
                    z = pressure_z[key]
                else:
                    key = (source.name, spec.key)
                    dataset_z.setdefault(key, rng.normal())
                    z = dataset_z[key]

                if source.absolute is not None:
                    sigma = np.full(len(df), float(source.absolute), dtype=float)
                elif source.relative is not None:
                    sigma = np.abs(df[col].to_numpy(dtype=float)) * float(source.relative)
                elif err_col in err_df.columns:
                    sigma = err_df[err_col].to_numpy(dtype=float)
                else:
                    sigma = np.zeros(len(df), dtype=float)

                total_shift += z * sigma

            df[col] = df[col].to_numpy(dtype=float) + total_shift

    return toy


def summarize_toys(central: np.ndarray, toys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if toys.size == 0:
        z = np.zeros_like(central, dtype=float)
        return z, z

    q16 = np.nanpercentile(toys, 16, axis=0)
    q84 = np.nanpercentile(toys, 84, axis=0)
    minus = np.clip(central - q16, 0.0, None)
    plus = np.clip(q84 - central, 0.0, None)
    return minus, plus
