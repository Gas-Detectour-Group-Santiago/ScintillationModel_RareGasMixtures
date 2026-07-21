#!/usr/bin/env python3
from __future__ import annotations
import os
from _bootstrap import ROOT
from scintillation.core.runtime import LegacyRuntime


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def main() -> None:
    runtime=LegacyRuntime.from_root(ROOT); runtime.prepare()
    runtime.run("primary_predictions/run_primary_predictions.py")
    if _enabled("RUN_LOW_PRESSURE_PRIMARY"):
        runtime.run("primary_predictions/run_primary_ir_low_pressure_predictions.py")
    if _enabled("RUN_JOINT_IR_PRIMARY"):
        runtime.run("primary_predictions/run_joint_ir_predictions.py")
    runtime.collect("primary")
    print("[workflow] primary predictions complete")

if __name__=="__main__": main()
