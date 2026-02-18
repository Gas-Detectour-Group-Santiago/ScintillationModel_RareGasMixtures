import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path

import mplhep as hep
import matplotlib as mpl

# Usar estilo CMS
hep.style.use("LHCb2")
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"   # Computer Modern
mpl.rcParams["axes.unicode_minus"] = False


names = ["ar_degrad_output_80Ar_20CF4.csv","cf4_degrad_output_80Ar_20CF4.csv"]
pdfs = ["ar_degrad_output_80Ar_20CF4.pdf","cf4_degrad_output_80Ar_20CF4.pdf"]

OUTDIR = Path("../histograms_levels")
OUTDIR.mkdir(exist_ok=True)

for i in range(len(names)):
    df = pd.read_csv(names[i])
    df = df[~df["Proceso"].str.contains("ELASTIC", case=False, na=False)]


    indexes=df.index.to_numpy()
    process=df["Proceso"].to_numpy()
    events=df["Eventos"].to_numpy()
    
    plt.figure(figsize=(12,9))
    plt.bar(indexes,events)
    plt.xlabel("Indice de cada evento")
    plt.ylabel("Eventos")
    #plt.yscale("log")
    plt.tight_layout()
    plt.savefig(OUTDIR / pdfs[i])

