#!/usr/bin/env python3
from __future__ import annotations
import argparse
from _bootstrap import ROOT
from scintillation.core.runtime import LegacyRuntime


def main()->None:
    parser=argparse.ArgumentParser(description="Optional diagnostic products")
    parser.add_argument("--integrals",action="store_true")
    parser.add_argument("--cross-sections",action="store_true")
    parser.add_argument("--populations",action="store_true")
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    if not any((args.integrals,args.cross_sections,args.populations,args.all)): args.all=True
    runtime=LegacyRuntime.from_root(ROOT); runtime.prepare()
    scripts=[]
    if args.all or args.integrals: scripts.append("integral_comparations/run_integral_comparisons.py")
    if args.all or args.cross_sections: scripts.append("cross_sections/plot_cross_section.py")
    if args.all or args.populations: scripts.append("populations_histograms/run_population_histograms.py")
    for script in scripts: runtime.run(script)
    runtime.collect("diagnostics")
    print("[workflow] requested diagnostics complete")

if __name__=="__main__": main()
