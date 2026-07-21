#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plot_style import LEGEND, FIGSIZE_SINGLE, setup_style


def plot_cross_sections(name_txt, output_pdf, main_gas, elastic_index, ion_index, att_index, inel_index):
    data=[]
    with open(name_txt, "r", encoding="latin-1") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                data.append([float(x) for x in line.split()])
    arr=np.asarray(data,float); energy=arr[:,0]
    elastic=arr[:,elastic_index[0]:elastic_index[1]].sum(axis=1)
    ion=arr[:,ion_index[0]:ion_index[1]].sum(axis=1)
    attachment=arr[:,att_index[0]:att_index[1]].sum(axis=1)
    inelastic=arr[:,inel_index[0]:inel_index[1]].sum(axis=1)
    def positive(values):
        values=np.asarray(values,float).copy(); values[values<=0]=np.nan; return values
    setup_style(grid=False,use_latex=False,context="single")
    fig,ax=plt.subplots(figsize=FIGSIZE_SINGLE)
    for values,label in ((elastic,"Elastic"),(ion,"Ionisation"),(attachment,"Attachment"),(inelastic,"Inelastic")):
        ax.plot(energy,positive(values),label=label)
    ax.set(xscale="log",yscale="log",xlabel="Electron energy [eV]",ylabel=r"Cross section [cm$^2$]",title=main_gas)
    ax.legend(**LEGEND.as_kwargs())
    fig.savefig(output_pdf); plt.close(fig)


def main():
    Path("pdf").mkdir(exist_ok=True)
    plot_cross_sections("data/Ar.txt","pdf/Ar_cs.pdf","Argon",[1,2],[2,9],[9,10],[10,-1])
    plot_cross_sections("data/CF4.txt","pdf/CF4_cs.pdf",r"CF$_4$",[1,2],[2,14],[14,15],[15,-1])
    plot_cross_sections("data/N2.txt","pdf/N2_cs.pdf",r"N$_2$",[1,2],[2,14],[14,15],[15,-1])
    plot_cross_sections("data/He.txt","pdf/He_cs.pdf","He",[1,2],[2,4],[4,5],[5,-1])

if __name__ == "__main__": main()
