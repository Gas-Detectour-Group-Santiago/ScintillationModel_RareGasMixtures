#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fname = "cf4.txt"

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
sigma_el = arr[:, 1]

# Según tu PrintGas():
# Level 1..12 → ionisation → columnas 2..13
sigma_ion = arr[:, 2:14].sum(axis=1)

# Level 13 → attachment → columna 14
sigma_att = arr[:, 14]

# Level 14 en adelante → inelastic + superelastic + dissociation
sigma_inel = arr[:, 15:].sum(axis=1)

def nz(x):
    y = x.copy()
    y[y <= 0] = np.nan
    return y

plt.figure(figsize=(6.2, 4.4))
plt.xscale("log")
plt.yscale("log")

plt.plot(E, nz(sigma_el), label="Elastic")
plt.plot(E, nz(sigma_ion), label="Ionisation (sum of 12)")
plt.plot(E, nz(sigma_att), label="Attachment")
plt.plot(E, nz(sigma_inel), label="Inelastic (sum of 40)")

plt.xlabel("electron energy [eV]")
plt.ylabel(r"cross section [cm$^2$]")
plt.legend()
plt.tight_layout()
plt.savefig("cs_cf4_pretty.pdf")

print("OK: cs_cf4_pretty.pdf / cs_cf4_pretty.png")
