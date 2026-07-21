from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .populations import load_population_rules

@dataclass(frozen=True)
class PopulationMappingReport:
    mixture_id: str
    mapped_levels: int
    unmapped_levels: int
    output_names: tuple[str,...]


def aggregate_hlevels(levels: pd.DataFrame, project_root: str | Path, mixture_id: str) -> tuple[dict[str,float], PopulationMappingReport]:
    rules=load_population_rules(project_root,mixture_id)
    work=levels.copy()
    work["gas_norm"]=work.get("gas","").astype(str).str.lower()
    energy=pd.to_numeric(work.get("energy_eV"),errors="coerce")
    state=work.get("state_name","").fillna("").astype(str)
    values=pd.to_numeric(work.get("n_events",0),errors="coerce").fillna(0.0)
    out={}
    mapped=pd.Series(False,index=work.index)
    for rule in rules:
        mask=work["gas_norm"].eq(rule.gas.lower()) & energy.ge(rule.energy_low_eV) & energy.lt(rule.energy_high_eV)
        for token in rule.name_tokens:
            mask &= state.str.contains(token,case=False,regex=False)
        out[rule.output_name]=float(values.loc[mask].sum())
        mapped |= mask
    return out, PopulationMappingReport(mixture_id,int(mapped.sum()),int((~mapped).sum()),tuple(out))
