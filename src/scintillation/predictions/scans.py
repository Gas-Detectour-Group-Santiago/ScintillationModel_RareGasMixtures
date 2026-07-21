from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from scintillation.plotting.recipe_config import as_bool


@dataclass(frozen=True)
class ScanSpec:
    scan_id: str
    mixture_id: str
    x: str
    y: str
    series: str | None
    facet: str | None
    xlabel: str
    ylabel: str
    xscale: str
    yscale: str
    filters: Mapping[str, object]
    output: str
    status: str

    @property
    def active(self) -> bool:
        return self.status.lower() == "active"


def _optional(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_secondary_scans(project_root: str | Path) -> tuple[ScanSpec, ...]:
    """Load generic transport/avalanche scans from the canonical secondary CSV."""
    path = Path(project_root) / "config" / "plots" / "secondary.csv"
    if not path.exists():
        return tuple()
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["plot_type"] == "scan"]
    scans: list[ScanSpec] = []
    for _, row in frame.iterrows():
        filters_text = str(row.get("filters", "")).strip()
        filters = json.loads(filters_text) if filters_text else {}
        scans.append(
            ScanSpec(
                scan_id=str(row["plot_id"]),
                mixture_id=str(row.get("model_id", "")),
                x=str(row.get("x", "")),
                y=str(row.get("y", "")),
                series=_optional(row.get("series")),
                facet=_optional(row.get("facet")),
                xlabel=str(row.get("xlabel", "")),
                ylabel=str(row.get("ylabel", "")),
                xscale=str(row.get("xscale", "linear")),
                yscale=str(row.get("yscale", "linear")),
                filters=filters,
                output=str(row.get("output", f"secondary/scans/{row['plot_id']}")),
                status="active" if as_bool(row.get("enabled"), True) else "disabled",
            )
        )
    return tuple(scans)


def apply_filters(frame: pd.DataFrame, filters: Mapping[str, object]) -> pd.DataFrame:
    selected = frame.copy()
    for column, condition in filters.items():
        if column not in selected:
            raise KeyError(f"Scan filter column {column!r} is missing")
        if isinstance(condition, dict):
            if "min" in condition:
                selected = selected.loc[selected[column] >= condition["min"]]
            if "max" in condition:
                selected = selected.loc[selected[column] <= condition["max"]]
            if "in" in condition:
                selected = selected.loc[selected[column].isin(condition["in"])]
        elif isinstance(condition, list):
            selected = selected.loc[selected[column].isin(condition)]
        else:
            selected = selected.loc[selected[column] == condition]
    return selected


def select_scan(frame: pd.DataFrame, spec: ScanSpec) -> pd.DataFrame:
    required = {spec.x, spec.y, "gas_mixture"}
    if spec.series:
        required.add(spec.series)
    if spec.facet:
        required.add(spec.facet)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Scan {spec.scan_id} misses catalog columns: {sorted(missing)}")
    selected = frame.loc[frame["gas_mixture"].astype(str) == spec.mixture_id].copy()
    selected = apply_filters(selected, spec.filters)
    selected = selected.dropna(subset=[spec.x, spec.y])
    return selected.sort_values([column for column in (spec.facet, spec.series, spec.x) if column])
