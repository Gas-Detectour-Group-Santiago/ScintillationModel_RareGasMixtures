#!/usr/bin/env python3
from __future__ import annotations
import argparse
from _bootstrap import ROOT
from scintillation.core.runtime import LegacyRuntime
from scintillation.predictions.secondary_catalog import build_secondary_catalog


def main()->None:
    parser=argparse.ArgumentParser(description="Build processed experimental, Degrad, Garfield and spectral inputs")
    parser.add_argument("--refresh-runtime",action="store_true")
    parser.add_argument("--skip-secondary-catalog",action="store_true")
    args=parser.parse_args()
    runtime=LegacyRuntime.from_root(ROOT); runtime.prepare(refresh=args.refresh_runtime)
    runtime.run("data/run_analysis.py")
    if not args.skip_secondary_catalog: build_secondary_catalog(ROOT)
    print("[workflow] analysis complete; reusable tables are in data/processed and data/cache")

if __name__=="__main__": main()
