from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class ModelChannel:
    channel_id: str
    model_id: str
    parameter_family: str
    primary_enabled: bool
    secondary_enabled: bool
    status: str


def load_model_channels(project_root: str | Path) -> tuple[ModelChannel,...]:
    frame=pd.read_csv(Path(project_root)/"config/channels.csv")
    def b(v): return str(v).lower() in {"1","true","yes","on"}
    return tuple(ModelChannel(str(r.channel_id),str(r.model_id),str(r.parameter_family),b(r.primary_enabled),b(r.secondary_enabled),str(r.status)) for r in frame.itertuples(index=False))
