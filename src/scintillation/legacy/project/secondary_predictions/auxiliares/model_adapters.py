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

    def evaluate_raw(self, params: np.ndarray, component: str, concentration, pressure: float | None):
        if component not in self.components:
            raise KeyError(f"Componente {component!r} no definida para {self.fit_name}.")
        return self.components[component](params, self.degrad, concentration, pressure)


_NUMERIC_ALIASES = {
    "E": "electric_field",
    "e": "electric_field",
    "field": "electric_field",
    "electric_field_kvcm": "electric_field",
    "P": "pressure",
    "p": "pressure",
    "pressure_bar": "pressure",
    "gap": "gap_mm",
    "gap_cm": "gap_mm",
    "concentration_percent": "concentration",
    "cf4": "concentration",
    "fCF4": "concentration",
    "f_cf4": "concentration",
    "NPE": "npe",
    "NE": "ne",
    "NI": "ni",
}


_METADATA_NUMERIC_COLUMNS = {
    "concentration",
    "concentration_percent",
    "concentration_gas_1",
    "concentration_gas_2",
    "electric_field",
    "gap_mm",
    "pressure",
    "npe",
    "ne",
    "ne_std",
    "ni",
    "ni_std",
    "n_entries",
}


def _pre_normalize_column(normalize_by: str | None) -> str | None:
    """Return the row-wise pre-interpolation normalization column.

    ``normalize_by='pre_ne'`` means: divide all population columns by ``ne``
    before giving the selected Garfield table to the fitted model.  The final
    absolute scaling then divides only by NPE and the model normalization.
    This reproduces the old paper scripts without changing the Garfield
    analysis/export script.
    """
    if normalize_by is None:
        return None
    mode = str(normalize_by).strip().lower()
    for prefix in ("pre_", "pre:", "before_interp_", "before_interp:"):
        if mode.startswith(prefix):
            column = mode[len(prefix):].strip()
            return column or None
    return None


def _normalization_is_pre(normalize_by: str | None) -> bool:
    return _pre_normalize_column(normalize_by) is not None


def _population_columns_for_pre_normalization(df: pd.DataFrame, norm_column: str) -> list[str]:
    out: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in _METADATA_NUMERIC_COLUMNS or col == norm_column:
            continue
        # Everything numeric that is not metadata is a population/count or an
        # uncertainty on a population/count.  Those must be normalized by the
        # same avalanche denominator before interpolation.
        out.append(col)
    return out


class SecondaryModelAdapter:
    """Adapter for Garfield population summaries.

    The adapter reads ``data/Secondary_GarfieldData/*/populations/*_secondary.csv``.
    The user-facing masks are applied to the raw Garfield metadata, while the
    selected population table is converted internally to the same concentration
    convention used by the fitted models (fraction, not percent).
    """

    def __init__(self, *, fit_name: str, garfield_csv: Path, components: Mapping[str, Callable]):
        self.fit_name = fit_name
        self.garfield_csv = Path(garfield_csv)
        self.components = dict(components)
        self._garfield: pd.DataFrame | None = None
        self._garfield_cache: dict[Path, pd.DataFrame] = {}
        self._model_cache: dict[str, pd.DataFrame] = {}

    def _population_csv_for_selection(self, selection: SecondarySelection | None = None) -> Path:
        """Return the Garfield population CSV that should be used here.

        By default the adapter uses ``self.garfield_csv``.  For the paper
        secondary plots each selection can point to a self-contained reference
        folder, for example ``data/Secondary_GarfieldData/ArCF4_paper/gem_1bar``.
        In that case only ``<reference_dir>/populations/<population_filename>``
        is read.  A direct ``population_csv`` can also be supplied.
        """
        if selection is not None:
            if selection.population_csv is not None:
                return Path(selection.population_csv)
            if selection.reference_dir is not None:
                ref = Path(selection.reference_dir)
                candidates = (
                    ref / "populations" / selection.population_filename,
                    ref / selection.population_filename,
                )
                for candidate in candidates:
                    if candidate.is_file():
                        return candidate
                # Return the canonical path for the eventual FileNotFoundError.
                return candidates[0]
        return self.garfield_csv

    def _read_population_csv(self, csv_path: Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if csv_path in self._garfield_cache:
            return self._garfield_cache[csv_path]
        if not csv_path.is_file():
            raise FileNotFoundError(f"No encuentro Garfield CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        self._garfield_cache[csv_path] = df
        return df

    @property
    def garfield(self) -> pd.DataFrame:
        return self._read_population_csv(self.garfield_csv)

    def _garfield_for_selection(self, selection: SecondarySelection | None = None) -> pd.DataFrame:
        return self._read_population_csv(self._population_csv_for_selection(selection))

    @staticmethod
    def _as_float(values) -> np.ndarray:
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def _is_numeric_array(values: np.ndarray) -> bool:
        try:
            values.astype(float)
            return True
        except Exception:
            return False

    def _resolve_column(self, df: pd.DataFrame, column: str, selection: SecondarySelection) -> str | None:
        if column in df.columns:
            return column
        alias = _NUMERIC_ALIASES.get(column, column)
        if alias in df.columns:
            return alias
        if column == "gain":
            if selection.gain_column in df.columns:
                return selection.gain_column
            if "gain" in df.columns:
                return "gain"
            # The current summary CSV does not contain a separate gain column.
            # For this project we allow selecting gain on the chosen column; in
            # the default test config gain_column='ne', so gain means NE.
            if "ne" in df.columns:
                return "ne"
        return None

    @staticmethod
    def _mask_from_rule(values, rule) -> np.ndarray:
        arr = np.asarray(values)
        numeric = SecondaryModelAdapter._is_numeric_array(arr)
        arr_float = SecondaryModelAdapter._as_float(arr) if numeric else None

        if isinstance(rule, Mapping):
            mask = np.ones(len(arr), dtype=bool)
            if "between" in rule:
                lo, hi = rule["between"]
                mask &= arr_float >= float(lo)
                mask &= arr_float <= float(hi)
            if "min" in rule:
                mask &= arr_float >= float(rule["min"])
            if "max" in rule:
                mask &= arr_float <= float(rule["max"])
            if "eq" in rule or "value" in rule:
                target = rule.get("eq", rule.get("value"))
                if numeric:
                    atol = float(rule.get("atol", 1e-8))
                    mask &= np.isclose(arr_float, float(target), atol=atol)
                else:
                    mask &= arr == target
            if "in" in rule:
                allowed = set(rule["in"])
                mask &= np.isin(arr, list(allowed))
            return mask

        if isinstance(rule, tuple):
            if len(rule) == 2 and all(isinstance(v, (int, float, np.number)) for v in rule):
                lo, hi = rule
                return (arr_float >= float(lo)) & (arr_float <= float(hi))

            if len(rule) >= 2:
                op, target = rule[0], rule[1]
                if op == ">":
                    return arr_float > float(target)
                if op == ">=":
                    return arr_float >= float(target)
                if op == "<":
                    return arr_float < float(target)
                if op == "<=":
                    return arr_float <= float(target)
                if op == "isclose":
                    atol = float(rule[2]) if len(rule) > 2 else 1e-8
                    return np.isclose(arr_float, float(target), atol=atol)
                if op == "==":
                    if numeric:
                        return np.isclose(arr_float, float(target), atol=1e-8)
                    return arr == target

        if numeric and isinstance(rule, (int, float, np.number)):
            return np.isclose(arr_float, float(rule), atol=1e-8)
        return arr == rule

    def _apply_rule(self, df: pd.DataFrame, mask: np.ndarray, column: str, rule, selection: SecondarySelection) -> np.ndarray:
        resolved = self._resolve_column(df, column, selection)
        if resolved is None:
            raise KeyError(
                f"No existe la columna/alias {column!r} en {self._population_csv_for_selection(selection).name}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        return mask & self._mask_from_rule(df[resolved].to_numpy(), rule)

    def select(self, selection: SecondarySelection) -> pd.DataFrame:
        source_csv = self._population_csv_for_selection(selection)
        df = self._garfield_for_selection(selection).copy()
        mask = np.ones(len(df), dtype=bool)

        if selection.gas and "gas_mixture" in df.columns:
            mask &= df["gas_mixture"].astype(str).str.lower().eq(selection.gas.lower())

        if selection.pressure is not None:
            mask = self._apply_rule(df, mask, "pressure", {"eq": selection.pressure, "atol": selection.pressure_atol}, selection)
        if selection.pressure_min is not None:
            mask = self._apply_rule(df, mask, "pressure", {"min": selection.pressure_min}, selection)
        if selection.pressure_max is not None:
            mask = self._apply_rule(df, mask, "pressure", {"max": selection.pressure_max}, selection)
        if selection.gap_mm is not None:
            mask = self._apply_rule(df, mask, "gap_mm", {"eq": selection.gap_mm, "atol": selection.gap_atol}, selection)
        if selection.gap_min is not None:
            mask = self._apply_rule(df, mask, "gap_mm", {"min": selection.gap_min}, selection)
        if selection.gap_max is not None:
            mask = self._apply_rule(df, mask, "gap_mm", {"max": selection.gap_max}, selection)
        electric_field_exact = selection.electric_field if selection.electric_field is not None else selection.E
        electric_field_atol = selection.electric_field_atol if selection.electric_field is not None else selection.E_atol
        electric_field_min = selection.electric_field_min if selection.electric_field_min is not None else selection.Emin
        electric_field_max = selection.electric_field_max if selection.electric_field_max is not None else selection.Emax
        if electric_field_exact is not None:
            mask = self._apply_rule(df, mask, "electric_field", {"eq": electric_field_exact, "atol": electric_field_atol}, selection)
        if electric_field_min is not None:
            mask = self._apply_rule(df, mask, "electric_field", {"min": electric_field_min}, selection)
        if electric_field_max is not None:
            mask = self._apply_rule(df, mask, "electric_field", {"max": electric_field_max}, selection)
        if selection.concentration is not None:
            mask = self._apply_rule(
                df, mask, "concentration", {"eq": selection.concentration, "atol": selection.concentration_atol}, selection
            )
        if selection.concentration_min is not None:
            mask = self._apply_rule(df, mask, "concentration", {"min": selection.concentration_min}, selection)
        if selection.concentration_max is not None:
            mask = self._apply_rule(df, mask, "concentration", {"max": selection.concentration_max}, selection)
        if selection.npe is not None:
            mask = self._apply_rule(df, mask, selection.npe_column, {"eq": selection.npe, "atol": selection.npe_atol}, selection)
        if selection.npe_min is not None:
            mask = self._apply_rule(df, mask, selection.npe_column, {"min": selection.npe_min}, selection)
        if selection.npe_max is not None:
            mask = self._apply_rule(df, mask, selection.npe_column, {"max": selection.npe_max}, selection)
        if selection.ne is not None:
            mask = self._apply_rule(df, mask, "ne", {"eq": selection.ne, "atol": selection.ne_atol}, selection)
        if selection.ne_min is not None:
            mask = self._apply_rule(df, mask, "ne", {"min": selection.ne_min}, selection)
        if selection.ne_max is not None:
            mask = self._apply_rule(df, mask, "ne", {"max": selection.ne_max}, selection)
        if selection.ni is not None:
            mask = self._apply_rule(df, mask, "ni", {"eq": selection.ni, "atol": selection.ni_atol}, selection)
        if selection.ni_min is not None:
            mask = self._apply_rule(df, mask, "ni", {"min": selection.ni_min}, selection)
        if selection.ni_max is not None:
            mask = self._apply_rule(df, mask, "ni", {"max": selection.ni_max}, selection)
        if selection.gain is not None:
            mask = self._apply_rule(df, mask, "gain", {"eq": selection.gain, "atol": selection.gain_atol}, selection)
        if selection.gain_min is not None:
            mask = self._apply_rule(df, mask, "gain", {"min": selection.gain_min}, selection)
        if selection.gain_max is not None:
            mask = self._apply_rule(df, mask, "gain", {"max": selection.gain_max}, selection)

        for col, rule in {**selection.extra_masks, **selection.masks}.items():
            mask = self._apply_rule(df, mask, col, rule, selection)

        out = df.loc[mask].copy().reset_index(drop=True)
        return out.sort_values([c for c in ("concentration", "pressure", "electric_field", "gap_mm", "npe") if c in out.columns]).reset_index(drop=True)

    def _model_axis_column(self, df: pd.DataFrame, x_axis: str, selection: SecondarySelection) -> str:
        axis = (x_axis or "concentration").strip()
        if axis in {"concentration_percent", "cf4_percent", "fCF4_percent"} and "concentration_percent" in df.columns:
            return "concentration_percent"
        resolved = self._resolve_column(df, axis, selection)
        if resolved is not None:
            return resolved
        alias = _NUMERIC_ALIASES.get(axis, axis)
        if alias in df.columns:
            return alias
        raise KeyError(
            f"No existe la columna/alias de eje x {x_axis!r} en {self.garfield_csv.name}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    def selected_model_data(self, selection: SecondarySelection, x_axis: str = "concentration") -> pd.DataFrame:
        axis = (x_axis or "concentration").strip()
        source_csv = self._population_csv_for_selection(selection)
        cache_key = f"{source_csv.resolve()}::{selection.id}::{axis}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key].copy()

        selected = self.select(selection)
        if selected.empty:
            raise ValueError(f"La selección secundaria {selection.id!r} no deja ninguna fila en {source_csv}.")
        if "concentration" not in selected.columns:
            raise KeyError(f"{source_csv} no contiene columna 'concentration'.")

        source_df = self._garfield_for_selection(selection)
        work = selected.copy()
        conc_raw = pd.to_numeric(work["concentration"], errors="coerce").to_numpy(dtype=float)
        all_conc_raw = pd.to_numeric(source_df["concentration"], errors="coerce").to_numpy(dtype=float)
        # The Garfield summaries store gas fractions in percent.  A fixed 1%
        # scan can have concentration.max()==1, so that column alone is
        # ambiguous.  Use the explicit gas-fraction metadata as the stronger
        # signal (e.g. 99 + 1), falling back to the concentration range.
        fraction_meta = []
        for meta_col in ("concentration_gas_1", "concentration_gas_2"):
            if meta_col in source_df.columns:
                fraction_meta.append(
                    pd.to_numeric(source_df[meta_col], errors="coerce").to_numpy(dtype=float)
                )
        max_fraction_meta = max(
            (float(np.nanmax(values)) for values in fraction_meta if np.any(np.isfinite(values))),
            default=float("nan"),
        )
        percent_like = np.nanmax(all_conc_raw) > 1.5 or (
            np.isfinite(max_fraction_meta) and max_fraction_meta > 1.5
        )
        work["concentration_percent"] = conc_raw if percent_like else conc_raw * 100.0
        work["concentration"] = conc_raw / 100.0 if percent_like else conc_raw

        pre_norm_col = _pre_normalize_column(selection.normalize_by)
        if pre_norm_col is not None:
            resolved_norm_col = self._resolve_column(work, pre_norm_col, selection)
            if resolved_norm_col is None:
                raise KeyError(
                    f"No existe la columna de pre-normalización {pre_norm_col!r} "
                    f"en {source_csv}. Columnas disponibles: {list(work.columns)}"
                )
            denom = pd.to_numeric(work[resolved_norm_col], errors="coerce").to_numpy(dtype=float)
            pop_cols = _population_columns_for_pre_normalization(work, resolved_norm_col)
            with np.errstate(divide="ignore", invalid="ignore"):
                for col in pop_cols:
                    values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
                    work[col] = values / denom
            work.attrs["pre_normalized_by"] = resolved_norm_col
            work.attrs["pre_normalized_columns"] = tuple(pop_cols)

        group_col = self._model_axis_column(work, axis, selection)

        if group_col == "concentration":
            # Match the old paper scripts: for concentration scans, pass the
            # selected Garfield rows directly to the fitted model.  The model
            # itself performs the PCHIP in concentration and handles duplicated
            # concentrations by keeping the first sorted row.  Averaging rows
            # here changes the effective populations and can create artificial
            # shoulders in the concentration scan.
            grouped = work.sort_values([c for c in ("concentration", "electric_field", "gap_mm", "pressure", "npe") if c in work.columns]).reset_index(drop=True)
            grouped["n_selected_rows"] = len(selected)
        else:
            numeric_cols = [
                c
                for c in work.select_dtypes(include=[np.number]).columns.tolist()
                if c != group_col
            ]
            grouped = work.groupby(group_col, as_index=False)[numeric_cols].mean(numeric_only=True)
            grouped["n_selected_rows"] = len(selected)
            grouped = grouped.sort_values(group_col).reset_index(drop=True)

        if "concentration_percent" not in grouped.columns and "concentration" in grouped.columns:
            grouped["concentration_percent"] = grouped["concentration"] * 100.0
        if "concentration" not in grouped.columns and "concentration_percent" in grouped.columns:
            grouped["concentration"] = grouped["concentration_percent"] / 100.0

        grouped.attrs["x_axis"] = axis
        grouped.attrs["x_column"] = group_col
        self._model_cache[cache_key] = grouped.copy()
        return grouped

    @staticmethod
    def _interp_column(model_data: pd.DataFrame, x_grid: np.ndarray, column: str, x_column: str = "concentration") -> np.ndarray:
        if column not in model_data.columns:
            return np.full(len(x_grid), np.nan, dtype=float)
        if x_column not in model_data.columns:
            raise KeyError(f"El eje de interpolación {x_column!r} no está en model_data.")
        x = pd.to_numeric(model_data[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(model_data[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) == 0:
            return np.full(len(x_grid), np.nan, dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]
        if len(xu) == 1:
            return np.full(len(x_grid), yu[0], dtype=float)
        return np.interp(x_grid, xu, yu, left=yu[0], right=yu[-1])

    def _interpolated_model_data(self, model_data: pd.DataFrame, x_grid: np.ndarray, x_column: str) -> pd.DataFrame:
        x = np.asarray(x_grid, dtype=float)
        out = pd.DataFrame({x_column: x})
        for col in model_data.select_dtypes(include=[np.number]).columns:
            if col == x_column:
                continue
            out[col] = self._interp_column(model_data, x, col, x_column=x_column)
        if "concentration" not in out.columns and "concentration_percent" in out.columns:
            out["concentration"] = out["concentration_percent"] / 100.0
        if "concentration_percent" not in out.columns and "concentration" in out.columns:
            out["concentration_percent"] = out["concentration"] * 100.0
        return out

    def secondary_metadata(self, selection: SecondarySelection, x_grid, x_axis: str = "concentration") -> dict[str, np.ndarray | int | str]:
        axis = (x_axis or "concentration").strip()
        model_data = self.selected_model_data(selection, axis)
        x = np.asarray(x_grid, dtype=float)
        x_column = self._model_axis_column(model_data, axis, selection)
        meta: dict[str, np.ndarray | int | str] = {
            "selection_id": selection.id,
            "n_selected_rows": int(model_data["n_selected_rows"].iloc[0]) if "n_selected_rows" in model_data.columns else len(model_data),
            "x_axis": axis,
            "x_column": x_column,
            "x": x,
        }
        for col in (
            "concentration",
            "concentration_percent",
            "pressure",
            "gap_mm",
            "electric_field",
            "npe",
            "ne",
            "ni",
            "ne_std",
            "ni_std",
            "n_entries",
        ):
            meta[col] = self._interp_column(model_data, x, col, x_column=x_column)
        gain_col = selection.gain_column if selection.gain_column in model_data.columns else "ne"
        meta["gain"] = self._interp_column(model_data, x, gain_col, x_column=x_column)
        return meta

    def evaluate_secondary_raw(
        self,
        params: np.ndarray,
        component: str,
        selection: SecondarySelection,
        x_grid,
        pressure: float | None = None,
        x_axis: str = "concentration",
    ):
        if component not in self.components:
            raise KeyError(f"Componente {component!r} no definida para {self.fit_name}.")
        axis = (x_axis or "concentration").strip()
        model_data = self.selected_model_data(selection, axis)
        x = np.asarray(x_grid, dtype=float)
        x_column = self._model_axis_column(model_data, axis, selection)

        if pressure is None:
            if selection.pressure is not None:
                pressure_eval = float(selection.pressure)
            elif x_column == "pressure":
                pressure_eval = x
            elif "pressure" in model_data.columns:
                pressure_eval = float(np.nanmean(model_data["pressure"].to_numpy(dtype=float)))
            else:
                pressure_eval = 1.0
        else:
            pressure_eval = float(pressure)

        if x_column == "concentration":
            raw = self.components[component](params, model_data, x, pressure_eval)
            return raw, self.secondary_metadata(selection, x, axis)

        # For scans in electric field, pressure, gap, gain, etc., the model's
        # concentration argument is not the x-axis. Evaluate row-by-row using
        # the Garfield populations interpolated along the chosen x-axis while
        # keeping the physical concentration from the selected rows.
        interpolated = self._interpolated_model_data(model_data, x, x_column)
        concentration_eval = interpolated["concentration"].to_numpy(dtype=float)
        if np.ndim(pressure_eval) == 0:
            pressure_values = np.full(len(x), float(pressure_eval), dtype=float)
        else:
            pressure_values = np.asarray(pressure_eval, dtype=float)

        values = []
        for i in range(len(x)):
            row = interpolated.iloc[[i]].copy()
            value = self.components[component](params, row, float(concentration_eval[i]), float(pressure_values[i]))
            values.append(float(np.ravel(value)[0]))
        raw = np.asarray(values, dtype=float)
        return raw, self.secondary_metadata(selection, x, axis)


def prepare_parameters(params: np.ndarray, normalization: NormalizationConfig) -> np.ndarray:
    params = np.asarray(params, dtype=float).copy()
    if normalization.mode == "set_norm_one" and len(params):
        params[0] = 1.0
    return params


def _normalization_denominator(params: np.ndarray, normalization: NormalizationConfig, reference_norm: float | None = None) -> float:
    params = np.asarray(params, dtype=float)

    if normalization.mode in {"as_fit", "set_norm_one"}:
        return 1.0
    if normalization.mode == "own_norm":
        return float(params[0])
    if normalization.mode in {"reference_norm", "secondary"} and normalization.reference_fit_name:
        if reference_norm is None:
            raise ValueError("reference_norm es obligatorio con reference_fit_name.")
        return float(reference_norm)
    if normalization.mode == "secondary":
        if len(params) == 0:
            raise ValueError("No puedo normalizar en modo secondary sin parámetros.")
        return float(params[0])
    if normalization.mode == "reference_norm":
        if reference_norm is None:
            raise ValueError("reference_norm es obligatorio con normalization.mode='reference_norm'.")
        return float(reference_norm)
    if normalization.mode == "fixed_norm":
        if normalization.fixed_norm is None:
            raise ValueError("fixed_norm es obligatorio con normalization.mode='fixed_norm'.")
        return float(normalization.fixed_norm)
    raise ValueError(f"Modo de normalización desconocido: {normalization.mode!r}")


def apply_normalization(raw, params: np.ndarray, normalization: NormalizationConfig, reference_norm: float | None = None):
    y = np.asarray(raw, dtype=float)
    denom = _normalization_denominator(params, normalization, reference_norm=reference_norm)
    if denom == 0:
        return np.full_like(y, np.nan, dtype=float)
    return y * float(normalization.output_scale) / denom


def apply_secondary_normalization(
    raw,
    params: np.ndarray,
    normalization: NormalizationConfig,
    metadata: Mapping[str, object],
    *,
    normalize_by: str = "ne",
    npe_column: str = "npe",
    reference_norm: float | None = None,
):
    y = np.asarray(raw, dtype=float)
    norm = _normalization_denominator(params, normalization, reference_norm=reference_norm)
    npe = np.asarray(metadata.get(npe_column), dtype=float)
    if (
        normalize_by is None
        or str(normalize_by).strip().lower() in {"", "none", "unity", "one", "1"}
        or _normalization_is_pre(normalize_by)
    ):
        # For pre_* modes the population columns were already divided by the
        # requested avalanche column before interpolation/model evaluation.
        # Do not divide by the same quantity again here.
        avalanche = np.ones_like(npe, dtype=float)
    else:
        avalanche = np.asarray(metadata.get(normalize_by), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = float(normalization.output_scale) / (norm * npe * avalanche)
        out = y * scale
    out[~np.isfinite(out)] = np.nan
    return out
