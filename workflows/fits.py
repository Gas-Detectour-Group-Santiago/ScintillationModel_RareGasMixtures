#!/usr/bin/env python3
from __future__ import annotations
import argparse
from _bootstrap import ROOT
from scintillation.core.paths import ProjectPaths
from scintillation.core.runtime import LegacyRuntime
from scintillation.fitting.toy_cache import compact_fit_toys, prune_fit_cache
from scintillation.physics.parameters import build_project_parameter_registry


def main() -> None:
    parser=argparse.ArgumentParser(description="Run every independent and joint primary fit")
    parser.add_argument("--toys",type=int,default=100)
    parser.add_argument("--refresh-runtime",action="store_true")
    parser.add_argument("--keep-toy-csv",action="store_true",help="Keep large toy CSV files in addition to compressed NPZ")
    args=parser.parse_args()
    env={key:str(args.toys) for key in ("PRIMARY_FIT_N_TOYS","PRIMARY_FIT_N_STAT_TOYS",
        "PRIMARY_FIT_N_SYST_TOYS","JOINT_IR_N_TOYS","JOINT_IR_N_STAT_TOYS","JOINT_IR_N_SYST_TOYS")}
    runtime=LegacyRuntime.from_root(ROOT); runtime.prepare(refresh=args.refresh_runtime)
    runtime.run("primary_fits/run_primary_fits.py",extra_env=env)
    paths=ProjectPaths.from_root(ROOT)
    compact_fit_toys(paths.fit_cache/"products",remove_csv=not args.keep_toy_csv)
    runtime.collect("fits")
    prune_fit_cache(paths.fit_cache/"products")
    build_project_parameter_registry(ROOT)
    print(f"[workflow] fits complete: {args.toys} stat + {args.toys} syst toys; NPZ cache ready")

if __name__=="__main__": main()
