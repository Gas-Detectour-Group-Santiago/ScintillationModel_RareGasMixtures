#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re

FNAME = "ar.txt"

# ---------- leer header (labels) + data ----------
labels = []
rows = []
in_types = False

with open(FNAME, "r", encoding="latin-1", errors="replace") as f:
    for line in f:
        s = line.rstrip("\n")

        if s.startswith("#"):
            if "cross-section types" in s:
                in_types = True
                continue
            if in_types:
                txt = s[1:].strip()
                if txt:
                    labels.append(re.sub(r"\s+", " ", txt))
            continue

        if s.strip():
            rows.append([float(x) for x in s.split()])

arr = np.array(rows)
E = arr[:, 0]
S = arr[:, 1:]

assert len(labels) == S.shape[1], "Header/column mismatch"

def nz(x):
    y = x.copy()
    y[y <= 0] = np.nan
    return y

# ---------- clasificar columnas ----------
elastic_idx = next(i for i, l in enumerate(labels)
                   if l.lower().startswith("elastic"))

ion_idx = [i for i, l in enumerate(labels)
           if "ION" in l.upper()]

exc_idx = [i for i, l in enumerate(labels)
           if "EXC" in l.upper()]

sigma_el  = S[:, elastic_idx]
sigma_ion = S[:, ion_idx].sum(axis=1)
sigma_exc = S[:, exc_idx].sum(axis=1)

# ---------- plot ----------
plt.figure(figsize=(6.6, 4.6))
plt.xscale("log")
plt.yscale("log")

plt.plot(E, nz(sigma_el),  label="Elastic", linewidth=2.2)
plt.plot(E, nz(sigma_exc), label=f"Excitation (sum of {len(exc_idx)})", linewidth=2.2)
plt.plot(E, nz(sigma_ion), label=f"Ionisation", linewidth=2.2)

plt.xlabel("electron energy [eV]")
plt.ylabel(r"cross section [cm$^2$]")
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig("cs_Ar.pdf")
print("OK: ar_el_exc_ion.pdf / ar_el_exc_ion.png")
