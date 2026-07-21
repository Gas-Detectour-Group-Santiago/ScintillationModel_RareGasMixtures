#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from _bootstrap import ROOT
from scintillation.core.outputs import OutputManager
from scintillation.core.paths import ProjectPaths
from scintillation.core.runtime import LegacyRuntime
from scintillation.predictions.secondary_catalog import build_secondary_catalog
from scintillation.predictions.scans import load_secondary_scans,select_scan
from scintillation.plotting.style import FIGSIZE_WIDE,LINEWIDTH_MAIN,MARKERSIZE,boxed_legend_kwargs,marker_for_geometry,palette,setup_style


def _label(value: object,column: str|None)->str:
    return f"{float(value):g}%" if column=="concentration_percent" else str(value)

def _render_scans(catalog: pd.DataFrame)->None:
    setup_style(grid=False,use_latex=False,context="single")
    outputs=OutputManager(ROOT); paths=ProjectPaths.from_root(ROOT)
    export_data=os.environ.get("EXPORT_SCAN_DATA","0").lower() in {"1","true","yes","on"}
    for spec in load_secondary_scans(ROOT):
        if not spec.active: continue
        selected=select_scan(catalog,spec)
        facets=["all"] if spec.facet is None else list(pd.unique(selected[spec.facet].dropna()))
        for facet in facets:
            subset=selected if spec.facet is None else selected.loc[selected[spec.facet]==facet]
            if subset.empty: continue
            fig,ax=plt.subplots(figsize=FIGSIZE_WIDE)
            values=[None] if spec.series is None else sorted(pd.unique(subset[spec.series]),key=lambda x:float(x))
            colors=palette(max(len(values),2),start=0.08,stop=0.92)
            marker=marker_for_geometry(str(facet) if spec.facet=="geometry" else "UNSPECIFIED")
            for idx,value in enumerate(values):
                group=subset if value is None else subset.loc[subset[spec.series]==value]
                group=group.sort_values(spec.x)
                ax.plot(group[spec.x],group[spec.y],marker=marker,linestyle="-" if len(group)>1 else "none",
                        color=colors[idx],linewidth=LINEWIDTH_MAIN,markersize=MARKERSIZE,
                        label=None if value is None else _label(value,spec.series))
            ax.set(xlabel=spec.xlabel,ylabel=spec.ylabel,xscale=spec.xscale,yscale=spec.yscale)
            ax.set_title(spec.scan_id.replace("_"," ")+(f" — {facet}" if spec.facet else ""))
            if spec.series:
                ax.legend(
                    title=spec.series.replace("_", " "),
                    **boxed_legend_kwargs(
                        loc="best",
                        ncol=2 if len(values) > 7 else 1,
                        fontsize=8.5,
                        title_fontsize=8.5,
                    ),
                )
            relative = Path(spec.output)
            out = paths.figures / relative
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out/f"{facet}.pdf"); plt.close(fig)
            if export_data:
                cache=paths.secondary_cache/"scans"/spec.scan_id; cache.mkdir(parents=True,exist_ok=True)
                subset.to_csv(cache/f"{facet}.csv.gz",index=False,compression="gzip")

def main()->None:
    runtime=LegacyRuntime.from_root(ROOT); runtime.prepare()
    runtime.run("secondary_predictions/run_secondary_predictions.py")
    runtime.collect("secondary")
    catalog=build_secondary_catalog(ROOT)
    _render_scans(catalog)
    print("[workflow] secondary predictions and configured scans complete")

if __name__=="__main__": main()
