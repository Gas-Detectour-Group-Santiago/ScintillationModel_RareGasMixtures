from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.paths import find_project_root


@dataclass(frozen=True)
class PopulationRule:
    mixture_id: str
    display_name: str
    name_tokens: tuple[str, ...]
    gas: str
    energy_low_eV: float
    energy_high_eV: float
    output_name: str
    enabled: bool
    status: str
    description: str


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_population_rules(project_root: str | Path, mixture_id: str, *, include_disabled: bool = False) -> tuple[PopulationRule, ...]:
    path = find_project_root(project_root) / "config" / "population_groups.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["mixture_id"].astype(str) == str(mixture_id)]
    if not include_disabled:
        frame = frame.loc[frame["enabled"].map(_as_bool)]
    rules = []
    for row in frame.itertuples(index=False):
        tokens = tuple(token.strip() for token in str(row.name_tokens).split("|") if token.strip())
        rules.append(PopulationRule(
            mixture_id=str(row.mixture_id), display_name=str(row.display_name), name_tokens=tokens,
            gas=str(row.gas), energy_low_eV=float(row.energy_low_eV), energy_high_eV=float(row.energy_high_eV),
            output_name=str(row.output_name), enabled=_as_bool(row.enabled), status=str(row.status),
            description=str(row.description),
        ))
    return tuple(rules)


def rules_as_legacy_dataframe(project_root: str | Path, mixture_id: str) -> pd.DataFrame:
    """Return the historical transposed DataFrame expected by the ROOT reader."""
    rules = load_population_rules(project_root, mixture_id)
    if not rules:
        raise ValueError(f"No enabled hLevels population rules for {mixture_id!r}")
    values = {
        rule.display_name: [list(rule.name_tokens), rule.gas, rule.energy_low_eV, rule.energy_high_eV, rule.output_name]
        for rule in rules
    }
    return pd.DataFrame(values, index=["name principal", "gas", "energy low", "energy up", "name output"])
