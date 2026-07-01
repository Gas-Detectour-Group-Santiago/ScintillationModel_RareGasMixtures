#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re
import scienceplots

def plot_cross_sections(name_txt,output_pdf,main_gas,elastic_index,ion_index,att_index,inel_index):

    fname = name_txt

    data = []
    with open(fname, "r", encoding="latin-1") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            data.append([float(x) for x in line.split()])

    arr = np.array(data)

    E = arr[:, 0]
    sigma_el = arr[:, elastic_index[0]:elastic_index[1]]

    # Según tu PrintGas():
    # Level 1..12 → ionisation → columnas 2..13
    sigma_ion = arr[:, ion_index[0]:ion_index[1]].sum(axis=1)

    # Level 13 → attachment → columna 14
    sigma_att = arr[:, att_index[0]:att_index[1]]

    # Level 14 en adelante → inelastic + superelastic + dissociation
    sigma_inel = arr[:, inel_index[0]:inel_index[1]].sum(axis=1)

    def nz(x):
        y = x.copy()
        y[y <= 0] = np.nan
        return y

    plt.figure(figsize=(6, 4))
    plt.style.use('grid')
    plt.grid(True, which='major', alpha=0.3)
    plt.grid(True, which='minor', alpha=0.08)


    plt.rcParams.update({
        "font.family": "serif",   # specify font family here
        "font.serif": ["Times"],  # specify font here
    })                            # specify font size here

    plt.xscale("log")
    plt.yscale("log")

    plt.plot(E, nz(sigma_el), label="Elastic")
    plt.plot(E, nz(sigma_ion), label="Ionisation")
    plt.plot(E, nz(sigma_att), label="Attachment")
    plt.plot(E, nz(sigma_inel), label="Inelastic")

    plt.xlabel(r"electron energy [eV]")
    plt.ylabel(r"cross section [cm$^2$]")
    plt.legend()
    plt.title(f"{main_gas}")
    plt.tight_layout()

    plt.savefig(output_pdf)

plot_cross_sections("data/Ar.txt","pdf/Ar_cs.pdf","Argon Cross Section",
                    elastic_index=[1,2],
                    ion_index=[2,9],
                    att_index=[9,10],
                    inel_index=[10,-1])

plot_cross_sections("data/CF4.txt","pdf/CF4_cs.pdf","CF$_4$ Cross Section",
                    elastic_index=[1,2],
                    ion_index=[2,14],
                    att_index=[14,15],
                    inel_index=[15,-1])


plot_cross_sections("data/N2.txt","pdf/N2_cs.pdf","N$_2$ Cross Section",
                    elastic_index=[1,2],
                    ion_index=[2,14],
                    att_index=[14,15],
                    inel_index=[15,-1])


plot_cross_sections("data/He.txt","pdf/He_cs.pdf","He Cross Section",
                    elastic_index=[1,2],
                    ion_index=[2,4],
                    att_index=[4,5],
                    inel_index=[5,-1])
