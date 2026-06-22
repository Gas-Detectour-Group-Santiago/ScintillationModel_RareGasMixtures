from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .prediction_types import NormalizationConfig, SecondarySelection


class PrimaryModelAdapter:
    def __init__(
        self,
        *,
        fit_name: str,
        degrad_csv: Path,
        components: Mapping[str, Callable],
    ):
        self.fit_name = fit_name
        self.degrad_csv = Path(degrad_csv)
        self.components = dict(components)
        self._degrad: pd.DataFrame | None = None

    @property
    def degrad(self) -> pd.DataFrame:
        if self._degrad is None:
            self._degrad = pd.read_csv(self.degrad_csv)
        return self._degrad

    def evaluate_raw(self, params: np.ndarray, component: str, concentration, pressure: float):
        if component not in self.components:
            raise KeyError(f"Componente {component!r} no definida para {self.fit_name}.")
        return self.components[component](params, self.degrad, concentration, pressure)


class SecondaryModelAdapter:
    """Adapter skeleton shared by future secondary predictions.

    It keeps the nuisance/casuistry part outside the plotting scripts: gap,
    pressure tolerances, electric-field cuts and npe selection are encoded in a
    SecondarySelection object.
    """

    def __init__(self, *, fit_name: str, garfield_csv: Path, components: Mapping[str, Callable]):
        self.fit_name = fit_name
        self.garfield_csv = Path(garfield_csv)
        self.components = dict(components)
        self._garfield: pd.DataFrame | None = None

    @property
    def garfield(self) -> pd.DataFrame:
        if self._garfield is None:
            self._garfield = pd.read_csv(self.garfield_csv)
        return self._garfield

    def select(self, selection: SecondarySelection) -> pd.DataFrame:
        df = self.garfield.copy()
        mask = np.ones(len(df), dtype=bool)

        if selection.pressure is not None and "pressure" in df.columns:
            mask &= np.isclose(df["pressure"].to_numpy(dtype=float), selection.pressure, atol=selection.pressure_atol)
        if selection.gap_mm is not None:
            gap_col = "gap_mm" if "gap_mm" in df.columns else "gap"
            if gap_col in df.columns:
                mask &= np.isclose(df[gap_col].to_numpy(dtype=float), selection.gap_mm, atol=selection.gap_atol)
        if selection.electric_field is not None and "electric_field" in df.columns:
            mask &= np.isclose(df["electric_field"].to_numpy(dtype=float), selection.electric_field)
        if selection.electric_field_min is not None and "electric_field" in df.columns:
            mask &= df["electric_field"].to_numpy(dtype=float) > selection.electric_field_min
        if selection.electric_field_max is not None and "electric_field" in df.columns:
            mask &= df["electric_field"].to_numpy(dtype=float) <= selection.electric_field_max
        if selection.npe is not None and selection.npe_column in df.columns:
            mask &= np.isclose(df[selection.npe_column].to_numpy(dtype=float), selection.npe)
        if selection.ne_min is not None and "ne" in df.columns:
            mask &= df["ne"].to_numpy(dtype=float) >= float(selection.ne_min)
        if selection.ne_max is not None and "ne" in df.columns:
            mask &= df["ne"].to_numpy(dtype=float) <= float(selection.ne_max)
        if selection.gain_min is not None or selection.gain_max is not None:
            if selection.gain_column in df.columns:
                gain = df[selection.gain_column].to_numpy(dtype=float)
            elif "ne" in df.columns and selection.npe:
                gain = df["ne"].to_numpy(dtype=float) / float(selection.npe)
            else:
                gain = None
            if gain is not None:
                if selection.gain_min is not None:
                    mask &= gain >= float(selection.gain_min)
                if selection.gain_max is not None:
                    mask &= gain <= float(selection.gain_max)

        for col, rule in selection.extra_masks.items():
            if col not in df.columns:
                continue
            values = df[col].to_numpy()
            if isinstance(rule, tuple) and len(rule) >= 2:
                op, target = rule[0], rule[1]
                if op == ">":
                    mask &= values.astype(float) > float(target)
                elif op == ">=":
                    mask &= values.astype(float) >= float(target)
                elif op == "<":
                    mask &= values.astype(float) < float(target)
                elif op == "<=":
                    mask &= values.astype(float) <= float(target)
                elif op == "isclose":
                    atol = float(rule[2]) if len(rule) > 2 else 1e-8
                    mask &= np.isclose(values.astype(float), float(target), atol=atol)
                elif op == "==":
                    mask &= values == target
            else:
                mask &= values == rule

        return df.loc[mask].reset_index(drop=True)


def prepare_parameters(params: np.ndarray, normalization: NormalizationConfig) -> np.ndarray:
    params = np.asarray(params, dtype=float).copy()
    if normalization.mode == "set_norm_one" and len(params):
        params[0] = 1.0
    return params


def apply_normalization(raw, params: np.ndarray, normalization: NormalizationConfig, reference_norm: float | None = None):
    y = np.asarray(raw, dtype=float)
    params = np.asarray(params, dtype=float)

    if normalization.mode == "as_fit" or normalization.mode == "set_norm_one":
        denom = 1.0
    elif normalization.mode == "own_norm":
        denom = float(params[0])
    elif normalization.mode == "reference_norm":
        if reference_norm is None:
            raise ValueError("reference_norm es obligatorio con normalization.mode='reference_norm'.")
        denom = float(reference_norm)
    elif normalization.mode == "fixed_norm":
        if normalization.fixed_norm is None:
            raise ValueError("fixed_norm es obligatorio con normalization.mode='fixed_norm'.")
        denom = float(normalization.fixed_norm)
    else:
        raise ValueError(f"Modo de normalización desconocido: {normalization.mode!r}")

    if denom == 0:
        return np.full_like(y, np.nan, dtype=float)
    return y * float(normalization.output_scale) / denom
