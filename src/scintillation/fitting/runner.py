from __future__ import annotations
from pathlib import Path
from ..core.runtime import LegacyRuntime
from ..physics.parameters import build_project_parameter_registry


def run_all_fits(project_root: str | Path, *, toys: int=100) -> None:
    runtime=LegacyRuntime.from_root(project_root)
    value=str(int(toys))
    env={key:value for key in (
        "PRIMARY_FIT_N_TOYS","PRIMARY_FIT_N_STAT_TOYS","PRIMARY_FIT_N_SYST_TOYS",
        "JOINT_IR_N_TOYS","JOINT_IR_N_STAT_TOYS","JOINT_IR_N_SYST_TOYS")}
    runtime.run("primary_fits/run_primary_fits.py",extra_env=env)
    runtime.collect("fits")
    build_project_parameter_registry(project_root)
