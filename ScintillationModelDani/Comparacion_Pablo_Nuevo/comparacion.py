import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from ArCF4_Completed import * 
from ArCF4_Pablo import *

visible = pd.read_csv("yield_vis_cal.csv")
ultravioleta = pd.read_csv("yield_uv_cal.csv")

par_completed = pd.read_csv("NewModel.csv")
par_pablo = pd.read_csv("PabloModel.csv")

f = pd.read_csv("poblations_Ar_3rd.csv")["fCF4"].to_numpy()
pAr3rd = pd.read_csv("poblations_Ar_3rd.csv")["Ar3rd all"].to_numpy()
pArdbleStar = pd.read_csv("poblations_Ar_dbleStar.csv")["Ar** all"].to_numpy()
pCF3 = pd.read_csv("poblations_CF3.csv")["CF3 >11.5"].to_numpy()
pCF4 = pd.read_csv("poblations_CF4.csv")["CF4 all"].to_numpy()


# Cogemos los parámetros

x_completed = par_completed[par_completed["type"].str.contains("value")].drop(columns=["type"]).to_numpy()
x_pablo= par_pablo[par_pablo["type"].str.contains("value")].drop(columns=["type"]).to_numpy()



# Mezclas

fCF4 = np.logspace(-5,0,100)

# Mejoramos los valores:
pob_Ar3rd = np.zeros_like(fCF4)
pob_ArdbleStar = np.zeros_like(fCF4)
pob_CF3 = np.zeros_like(fCF4)
pob_CF4 = np.zeros_like(fCF4)

for i in range(len(fCF4)): 
    pob_Ar3rd[i] = np.interp(fCF4[i],f,pAr3rd)
    pob_ArdbleStar[i] = np.interp(fCF4[i],f,pArdbleStar) 
    pob_CF3[i] = np.interp(fCF4[i],f,pCF3)
    pob_CF4[i] = np.interp(fCF4[i],f,pCF4)
    

plt.figure()
plt.plot(fCF4,pob_Ar3rd,color="blue",label="$N_{Ar_{3rd}}$")
plt.plot(fCF4,pob_ArdbleStar,color="red",label="$N_{Ar^{**}}$")
plt.plot(fCF4,pob_CF3,color="green",label="$N_{CF_3}$")
plt.plot(fCF4,pob_CF4,color="orange",label="$N_{CF_4}$")
plt.legend()
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("x")
plt.ylabel("Eventos/Polbaciones Degrad")
plt.savefig("Poblaciones.pdf",bbox_inches="tight")

n = 1
n1_completed = theory_yield_uv(x_completed[0,:],fCF4,n,pob_CF3,pob_ArdbleStar,pob_CF4,pob_Ar3rd)
n1_pablo = theory_yield_uv_pablo(x_pablo[0,:],fCF4,n,pob_CF3,pob_ArdbleStar,pob_CF4,pob_Ar3rd)

n = 5
n5_completed = theory_yield_uv(x_completed[0,:],fCF4,n,pob_CF3,pob_ArdbleStar,pob_CF4,pob_Ar3rd)
n5_pablo = theory_yield_uv_pablo(x_pablo[0,:],fCF4,n,pob_CF3,pob_ArdbleStar,pob_CF4,pob_Ar3rd)


fig,ax = plt.subplots(ncols=2,figsize=(8.5,3.5))
ax[0].plot(fCF4*100,n1_completed,color="blue",label="Nuevo modelo")
ax[0].plot(fCF4*100,n1_pablo,color="red",linestyle="--",label="Modelo Anterior")
ax[0].errorbar(ultravioleta["fCF4 real"],
             ultravioleta["1.0bar"],
             yerr=ultravioleta["Err 1.0bar"],
             marker="o",
             linestyle="none",
             label="Datos Experimentales")


ax[1].plot(fCF4*100,n5_completed,color="blue",label="Nuevo modelo")
ax[1].plot(fCF4*100,n5_pablo,color="red",linestyle="--",label="Modelo Anterior")
ax[1].errorbar(ultravioleta["fCF4 real"],
             ultravioleta["4.0bar"],
             yerr=ultravioleta["Err 4.0bar"],
             marker="o",
             linestyle="none",
             label="Datos Experimentales")


ax[0].legend()
ax[0].set_xscale("log")
ax[0].set_xlabel("$f_{CF_4}$ (%)")
ax[0].set_yscale("log")
ax[0].set_title("1 bar")
ax[1].set_title("4 bar")
ax[1].legend()
ax[1].set_xscale("log")
ax[1].set_xlabel("$f_{CF_4}$ (%)")
ax[1].set_yscale("log")
ax[0].set_ylabel("Yields")

ax[0].set_ylim(3e-2,1)
ax[1].set_ylim(3e-2,1)
fig.tight_layout()
fig.savefig("n1bar_comparacion.pdf",bbox_inches="tight")