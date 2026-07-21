from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PlotSemantics:
    color: str = "channel_or_series"
    marker: str = "geometry"
    linestyle: str = "incident_particle_or_model"
    fillstyle: str = "data_source"
    extrapolation: str = "dotted"

DEFAULT_SEMANTICS = PlotSemantics()
