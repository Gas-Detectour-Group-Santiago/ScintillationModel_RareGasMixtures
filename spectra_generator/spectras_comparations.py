import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import scienceplots
import scipy.special as sc

plt.style.use(["science", "grid"])

# =========================================================
# CARGA ROBUSTA DE PICKLES
# =========================================================
class CompatUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == "scipy.special._special_ufuncs" and hasattr(sc, name):
            return getattr(sc, name)
        return super().find_class(module, name)

def safe_dill_load(path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()

# =========================================================
# RUTAS: funciona tanto en script como en notebook
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

MODELS_DIR = (BASE_DIR / "../models").resolve()
DATA_DIR = (BASE_DIR / "../data").resolve()

sys.path.append(str(MODELS_DIR))
sys.path.append(str(DATA_DIR))

from ArN2 import *
from ArN2_infrarred import *
from ArCF4 import *
from ArCF4_infrarred import *

# =========================================================
# CONFIGURACIÓN
# =========================================================
DATA_DIR_DEGRAD = BASE_DIR / "../data/Primary_DegradData"
DATA_DIR_PAR = BASE_DIR / "../data/Parameters"
DATA_DIR_EXP = BASE_DIR / "../data/Experimental"

pressure = [1]
concentrations = [0.1, 1, 5, 100]
wavelength = np.linspace(200, 800, 2000)

# referencia de normalización: 95/5 -> 5% aditivo, 1 bar
REF_CON = 5
REF_PRES = 1

# ventana para buscar el pico visible experimental de CF4
VISIBLE_MIN = 500
VISIBLE_MAX = 750

# =========================================================
# DATOS
# =========================================================
degrad_data_cf4 = pd.read_csv(DATA_DIR_DEGRAD / "ArCF4.csv")
degrad_data_cf4_IR = pd.read_csv(DATA_DIR_DEGRAD / "ArCF4_IR.csv")
degrad_data_n2 = pd.read_csv(DATA_DIR_DEGRAD / "ArN2.csv")
degrad_data_n2_IR = pd.read_csv(DATA_DIR_DEGRAD / "ArN2_IR.csv")

parameter_data_cf4 = pd.read_csv(DATA_DIR_PAR / "ArCF4_primary.csv")["parameter"].to_numpy()
parameter_data_cf4_IR = pd.read_csv(DATA_DIR_PAR / "ArCF4_IR_primary.csv")["parameter"].to_numpy()
parameter_data_n2 = pd.read_csv(DATA_DIR_PAR / "ArN2_primary.csv")["parameter"].to_numpy()
parameter_data_n2_IR = pd.read_csv(DATA_DIR_PAR / "ArN2_IR_primary.csv")["parameter"].to_numpy()

norm_cf4 = parameter_data_cf4[0].copy()
norm_n2 = parameter_data_n2[0].copy()

print("norm =", norm_cf4, norm_n2)

norm = 1

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def gaussiana(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normalize_all_spectra(all_spectra, ref_value):
    out = []
    for con, spectra_con in all_spectra:
        new_spectra = []
        for pres, y in spectra_con:
            new_spectra.append((pres, y / ref_value))
        out.append((con, new_spectra))
    return out

def get_cf4_exp_visible_reference(df_cf4, con_ref=REF_CON, pres_ref=REF_PRES,
                                  visible_min=VISIBLE_MIN, visible_max=VISIBLE_MAX):
    mask = np.isclose(df_cf4["concentracion"], con_ref) & np.isclose(df_cf4["presion"], pres_ref)
    if not np.any(mask):
        raise ValueError(f"No encuentro el espectro experimental CF4 para con={con_ref}, P={pres_ref}")

    dic = df_cf4.loc[mask].iloc[0]["data(norm)"]
    wavelen = np.asarray(dic["wavelength"], dtype=float)
    intensity = np.asarray(dic["intensity"], dtype=float)

    vis_mask = (wavelen >= visible_min) & (wavelen <= visible_max)
    if not np.any(vis_mask):
        raise ValueError("La ventana visible está vacía en el espectro experimental de referencia.")

    return np.max(intensity[vis_mask])

# =========================================================
# CARGA DE PICKLES
# =========================================================
df_N2 = safe_dill_load(DATA_DIR_EXP / "ArN2/N2_primary_data_final.pkl")
df_CF4 = safe_dill_load(DATA_DIR_EXP / "ArCF4/CF4_primary_data_final.pkl")

# =========================================================
# ECUACIONES IR
# =========================================================
equations_n2 = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}

# Ajusta estos nombres si en tu módulo se llaman distinto
equations_cf4 = {
    "696": theory_yield_ArCF4_Ir_696,
    "727": theory_yield_ArCF4_Ir_727,
    "750": theory_yield_ArCF4_Ir_750,
    "763": theory_yield_ArCF4_Ir_763,
    "772": theory_yield_ArCF4_Ir_772,
    "794": theory_yield_ArCF4_Ir_794,
}

# =========================================================
# PRIMERA PASADA: ESPECTROS TEÓRICOS SIN NORMALIZAR
# =========================================================
all_spectra_n2_raw = []
all_spectra_cf4_raw = []

theory_visible_ref = None  # máximo del visible teórico CF4 a 95/5

# ---------- N2 ----------
for con in concentrations:
    spectra_con = []
    factor = 1 # (1 / 0.012) * W_ArN2(con / 100) / norm

    for pres in pressure:
        yield_N2 = theory_yield_N2_uv(
            parameter_data_n2, degrad_data_n2, np.array([con / 100]), pres
        )

        #yield_total = 0.13 * factor * yield_N2[0] * gaussiana(wavelength, 310, 3)
        yield_total = 0.47 * factor * yield_N2[0] * gaussiana(wavelength, 335, 2.5*1.5)
        yield_total += 0.32 * factor * yield_N2[0] * gaussiana(wavelength, 355, 2.5*1.5)
        yield_total += 0.13 * factor * yield_N2[0] * gaussiana(wavelength, 378, 2.5*1.5)
        yield_total += 0.08 * factor * yield_N2[0] * gaussiana(wavelength, 403, 2.5*1.5)


        print("integral  n2 = ",np.trapezoid(yield_total,wavelength))

        for name, yield_IR in equations_n2.items():
            yield_ir = yield_IR(
                parameter_data_n2_IR, degrad_data_n2_IR, np.array([con / 100]), pres
            )
            yield_total += factor * yield_ir[0] * gaussiana(wavelength, float(name), 2.8)

        spectra_con.append((pres, yield_total))

    all_spectra_n2_raw.append((con, spectra_con))

# ---------- CF4 ----------
for con in concentrations:
    spectra_con = []

    for pres in pressure:
        factor = 1 # (1 / 0.015) * ion_potential(con / 100) / norm

        yield_vis = theory_yield_vis(
            parameter_data_cf4, degrad_data_cf4, np.array([con / 100]), pres
        ) * factor

        yield_uv, yield_cf4, yield_ArDbleStar, yield_cf3_uv = theory_yield_uv(
            parameter_data_cf4,
            degrad_data_cf4,
            np.array([con / 100]),
            pres,
            activate_components=True
        )

        yield_uv *= factor
        yield_cf4 *= factor
        yield_ArDbleStar *= factor
        yield_cf3_uv *= factor

        yield_vis_spec = yield_vis[0] * gaussiana(wavelength, 630, 40)

        yield_cf4_230 = (0.8 / 1.85) * yield_cf4[0] * gaussiana(wavelength, 230, 20)
        yield_cf4_290 = (0.8 / 1.85) * yield_cf4[0] * gaussiana(wavelength, 290, 20)
        yield_cf4_364 = (0.25 / 1.85) * yield_cf4[0] * gaussiana(wavelength, 364, 40)
        yield_cf4_spec = yield_cf4_230 + yield_cf4_290 + yield_cf4_364

        yield_arDbleStar_spec = yield_ArDbleStar[0] * gaussiana(wavelength, 220, 60)
        yield_CF3_spec = yield_cf3_uv[0] * gaussiana(wavelength, 245, 60)

        yield_total = (
            yield_vis_spec
            + yield_cf4_spec
            + yield_arDbleStar_spec
            + yield_CF3_spec
        )


        print("integral  cf4 = ",np.trapezoid(yield_total,wavelength))
        
        for name, yield_IR in equations_cf4.items():
            yield_ir = yield_IR(
                parameter_data_cf4_IR, degrad_data_cf4_IR, np.array([con / 100]), pres
            )
            yield_total += factor * yield_ir[0] * gaussiana(wavelength, float(name), 2.7)

        # referencia teórica: pico del visible CF4 a 95/5
        if np.isclose(con, REF_CON) and np.isclose(pres, REF_PRES):
            theory_visible_ref = np.max(yield_vis_spec)

        spectra_con.append((pres, yield_total))

    all_spectra_cf4_raw.append((con, spectra_con))

if theory_visible_ref is None or theory_visible_ref <= 0:
    raise ValueError("No pude determinar la referencia teórica del pico visible para CF4 al 95/5.")

# referencia experimental: pico visible experimental CF4 a 95/5
exp_visible_ref = 1 #  get_cf4_exp_visible_reference(df_CF4)

print(f"Referencia teórica visible CF4 (95/5): {theory_visible_ref}")
print(f"Referencia experimental visible CF4 (95/5): {exp_visible_ref}")

# =========================================================
# NORMALIZACIÓN GLOBAL
# =========================================================
all_spectra_n2 = normalize_all_spectra(all_spectra_n2_raw, theory_visible_ref)
all_spectra_cf4 = normalize_all_spectra(all_spectra_cf4_raw, theory_visible_ref)

# =========================================================
# CÁLCULO DE ymax GLOBAL NORMALIZADO
# =========================================================
global_ymax = 0.0

for _, spectra_con in all_spectra_n2:
    for _, y in spectra_con:
        global_ymax = max(global_ymax, np.max(y))

for _, spectra_con in all_spectra_cf4:
    for _, y in spectra_con:
        global_ymax = max(global_ymax, np.max(y))

# también incluimos los espectros experimentales ya normalizados
for con in concentrations:
    for pres in pressure:
        mask_n2 = np.isclose(df_N2["N2 concentration (%)"], con) & np.isclose(df_N2["P (bar)"], pres)
        if np.any(mask_n2):
            dic = df_N2.loc[mask_n2].iloc[0]["spectrum_new_cal"]
            intensity =   np.asarray(dic["intensity"], dtype=float) / exp_visible_ref
            global_ymax = max(global_ymax, np.max(intensity))

        mask_cf4 = np.isclose(df_CF4["concentracion"], con) & np.isclose(df_CF4["presion"], pres)
        if np.any(mask_cf4):
            dic = df_CF4.loc[mask_cf4].iloc[0]["data(norm)"]
            intensity = np.asarray(dic["intensity"], dtype=float) / exp_visible_ref
            global_ymax = max(global_ymax, np.max(intensity))

# =========================================================
# FIGURA FINAL
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
axs = axs.ravel()

for ax, (con_cf4, spectra_cf4_con), (con_n2, spectra_n2_con) in zip(
    axs, all_spectra_cf4, all_spectra_n2
):
    # -------------------------
    # N2 teórico + experimental
    # -------------------------
    for pres, yield_total_n2 in spectra_n2_con:
        ax.plot(
            wavelength,
            yield_total_n2,
            color="red",
            lw=2,
            label=f"N$_2$ Teo. {pres:.1f} bar"
        )

        mask_n2 = (df_N2["N2 concentration (%)"] == con_n2) & (df_N2["P (bar)"] == pres)

        # print("con N2", con_n2)
        # print(mask_n2)

        if np.any(mask_n2):
            dic = df_N2.loc[mask_n2].iloc[0]["mean_spectrum"]
            wavelen = np.asarray(dic["wavelength"], dtype=float)
            intensity =  np.asarray(dic["intensity"], dtype=float) / exp_visible_ref  

            ax.plot(
                wavelen,
                intensity,
                color="green",
                lw=1.8,
                label=f"N$_2$ Exp. {pres:.1f} bar"
            )
            nn1 = int(len(wavelen)/6)
            nn2 = int(len(wavelen)/2.8)
            
            integral = np.trapezoid(intensity[nn1:nn2],wavelen[nn1:nn2])

            print("Integral n2 exp",integral)

    # -------------------------
    # CF4 teórico + experimental
    # -------------------------
    for pres, yield_total_cf4 in spectra_cf4_con:
        ax.plot(
            wavelength,
            yield_total_cf4,
            color="blue",
            lw=2,
            label=f"CF$_4$ Teo. {pres:.1f} bar"
        )

        mask_cf4 = np.isclose(df_CF4["concentracion"], con_cf4) & np.isclose(df_CF4["presion"], pres)
        if np.any(mask_cf4):
            dic = df_CF4.loc[mask_cf4].iloc[0]["data(norm)"]
            wavelen = np.asarray(dic["wavelength"], dtype=float)
            intensity = np.asarray(dic["intensity"], dtype=float) / exp_visible_ref

            ax.plot(
                wavelen,
                intensity,
                color="orange",
                lw=1.8,
                label=f"CF$_4$ Exp. {pres:.1f} bar"
            )

            integral2 = np.trapezoid(intensity[:],wavelen[:])
            print("integral experimental cf4",integral2)


    ax.set_title(f"{con_cf4:.1f} $\%$ Aditivo")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel("Intensidad normalizada")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.08)
    ax.set_xlim(200, 800)
    ax.set_ylim(0, 1.15 * global_ymax)
    ax.legend(ncol=2, loc="upper right", fontsize=8)

fig.suptitle(
    r"Primary Ar-N$_2$ \& Ar-CF$_4$ Spectra Prediction (normalizado al pico visible de 95/5 = 1)",
    fontsize=13
)
fig.tight_layout()
fig.savefig("Comparation_normalized.pdf", dpi=300, bbox_inches="tight")

print("razones integral",integral/integral2)
plt.show()

