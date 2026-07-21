from __future__ import annotations
from pathlib import Path
from ..core.runtime import LegacyRuntime


def run_primary(project_root: str | Path) -> None:
    runtime=LegacyRuntime.from_root(project_root)
    for script in ("primary_predictions/run_primary_predictions.py","primary_predictions/run_primary_ir_low_pressure_predictions.py","primary_predictions/run_joint_ir_predictions.py"):
        runtime.run(script)
    runtime.collect("primary")


def run_secondary(project_root: str | Path) -> None:
    runtime=LegacyRuntime.from_root(project_root)
    runtime.run("secondary_predictions/run_secondary_predictions.py")
    runtime.collect("secondary")
