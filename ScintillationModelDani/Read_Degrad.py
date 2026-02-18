import re
import pandas as pd
from pathlib import Path
import numpy as np 


"""
Script que nos permite leer los datos de las poblaciones de Degrad CF3/Ar**/CF4/Ar3rd, sacándolos en formato pickle y csv.
"""

#############################################################################################################
########################## FUNCION QUE LE UN ARCHIVO DE DEGRAD ##############################################
#############################################################################################################

def read_input(archivo_entrada, archivo_salida_Ar="collisions_Ar.csv", archivo_salida_CF4="collisions_CF4.csv"):
    # hecho por chatgpt, coregido por dvl
    def split_by_gas(block_text: str):
        # Encabezados tipo "ARGON ANISOTROPIC ...\n--------"
        headers = list(re.finditer(
            r"^\s*(?P<gas>[A-Z0-9]+)\s+ANISOTROPIC[^\n]*\n[-]{8,}\s*",
            block_text, flags=re.M
        ))
        parts = []
        for i, h in enumerate(headers):
            gas = h.group("gas").strip()
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(block_text)
            parts.append((gas, block_text[start:end]))
        return parts

    def parse_lines(gas: str, gas_block: str) -> pd.DataFrame:
        # Proceso [ELOSS/ELEVEL= ...] valor +- error %
        pattern = re.compile(
            r"^\s*(?P<proc>.+?)"
            r"(?:\s+(?:E(?:LEVEL|LOSS)=\s*(?P<energy>-?\d*\.?\d+(?:D[+-]?\d+)?)))?"
            r"\s+(?P<value>-?\d*\.?\d+)\s*\+\-\s*(?P<err>-?\d*\.?\d+)\s*%",
            flags=re.M
        )
        rows = []
        for m in pattern.finditer(gas_block):
            proc = re.sub(r"\s{2,}", " ", m.group("proc").strip())
            energy = m.group("energy")
            energy = float(energy.replace("D", "E")) if energy else None
            rows.append({
                "Gas": gas,
                "Proceso": proc,
                "Energia": energy,
                "Eventos": float(m.group("value")),
                "Error%": float(m.group("err")),
            })
        return pd.DataFrame(rows)

    def parse_file(path: str) -> pd.DataFrame:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        title = "NUMBER OF COLLISIONS PER EVENT FOR EACH GAS"
        idx = text.find(title)
        if idx == -1:
            return pd.DataFrame()
        # Toma desde el título hasta el final (evita cortar en el primer “----”)
        sub = text[idx:]
        frames = [parse_lines(g, b) for g, b in split_by_gas(sub)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # --- EJECUCIÓN ---
    df_all = parse_file(archivo_entrada)

    if df_all.empty:
        print("No se encontró el bloque de colisiones en el archivo.")
    else:
        df_all[df_all["Gas"] == "ARGON"].to_csv(archivo_salida_Ar, index=False)
        df_all[df_all["Gas"] == "CF4"].to_csv(archivo_salida_CF4, index=False)
        print(f"Guardado: {archivo_salida_Ar} y {archivo_salida_CF4}")

    return df_all

#############################################################################################################
########################## PROGRAMA PRINCIPAL ##############################################
#############################################################################################################

########################## Inputs de Degrad ##############################################

archives=np.array(["input/output_99.9Ar_0.1CF4.txt",
                   "input/output_99.8Ar_0.2CF4.txt",
                   "input/output_99.5Ar_0.5CF4.txt",
                   "input/output_99Ar_1CF4.txt",
                   "input/output_98Ar_2CF4.txt",
                   "input/output_95Ar_5CF4.txt",
                   "input/output_90Ar_10CF4.txt",
                   "input/output_80Ar_20CF4.txt",
                   "input/output_50Ar_50CF4.txt",
                   "input/output_PureCF4.txt"])

archives_ar=np.array(["input/ar_degrad_output_99.9Ar_0.1CF4.csv",
                   "input/ar_degrad_output_99.8Ar_0.2CF4.csv",
                   "input/ar_degrad_output_99.5Ar_0.5CF4.csv",
                   "input/ar_degrad_output_99Ar_1CF4.csv",
                   "input/ar_degrad_output_98Ar_2CF4.csv",
                   "input/ar_degrad_output_95Ar_5CF4.csv",
                   "input/ar_degrad_output_90Ar_10CF4.csv",
                   "input/ar_degrad_output_80Ar_20CF4.csv",
                   "input/ar_degrad_output_50Ar_50CF4.csv",
                   "input/ar_degrad_output_PureCF4.csv"])

archives_cf4=np.array(["input/cf4_degrad_output_99.9Ar_0.1CF4.csv",
                   "input/cf4_degrad_output_99.8Ar_0.2CF4.csv",
                   "input/cf4_degrad_output_99.5Ar_0.5CF4.csv",
                   "input/cf4_degrad_output_99Ar_1CF4.csv",
                   "input/cf4_degrad_output_98Ar_2CF4.csv",
                   "input/cf4_degrad_output_95Ar_5CF4.csv",
                   "input/cf4_degrad_output_90Ar_10CF4.csv",
                   "input/cf4_degrad_output_80Ar_20CF4.csv",
                   "input/cf4_degrad_output_50Ar_50CF4.csv",
                   "input/cf4_degrad_output_PureCF4.csv"])


######################## Inicialización de dataframe ################################################

fCF4 = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1])   # concentraciones

poblations_CF4 = pd.DataFrame({"fCF4":fCF4})
poblations_CF3 = pd.DataFrame({"fCF4":fCF4})
poblations_Ar_dbleStar  = pd.DataFrame({"fCF4":fCF4})
poblations_Ar_3rd  = pd.DataFrame({"fCF4":fCF4})


name_CF4         =  np.array(["CF4 all"])
name_CF3         =  np.array(["CF3 >11.5","CF3 >12.5","CF3 >14.5"])
name_Ar_dbleStar =  np.array(["Ar** all", "Ar** 2exc"])
name_Ar_3rd      =  np.array(["Ar3rd all"])

"""
for col in name_CF4:
    poblations_CF4[col] = np.nan

for col in name_CF3:
    poblations_CF3[col] = np.nan
    
for col in name_Ar_dbleStar:
    poblations_Ar_dbleStar[col] = np.nan
    
for col in name_Ar_3rd:
    poblations_Ar_3rd[col] = np.nan
""" 
################ Lectura y guardado en dataframe ##############################################
for i in range(len(archives)):
    df = read_input(
        archives[i],
        archivo_salida_Ar=archives_ar[i],
        archivo_salida_CF4=archives_cf4[i]
    )

    ar  = df.loc[df['Gas'] == "ARGON", :]
    cf4 = df.loc[df['Gas'] == "CF4", :]

    # --- CF4 ---
    mask_cf4_all = cf4['Proceso'].str.contains("ION CF3 +")
    poblations_CF4.loc[i, "CF4 all"] = cf4.loc[mask_cf4_all, 'Eventos'].sum()
    poblations_CF4.loc[i, "Err CF4 all"] =  np.sqrt((cf4.loc[mask_cf4_all, 'Eventos']**2 * cf4.loc[mask_cf4_all, 'Error%']**2).sum())/100

    # --- Ar 3rd ---
    mask_ar3rd = ar['Proceso'].str.contains("CHARGE STATE =2")
    poblations_Ar_3rd.loc[i, "Ar3rd all"] = ar.loc[mask_ar3rd, 'Eventos'].sum()
    mask_ar3rd = ar['Proceso'].str.contains("CHARGE STATE =3")
    poblations_Ar_3rd.loc[i, "Ar3rd all"] += ar.loc[mask_ar3rd, 'Eventos'].sum()
    poblations_Ar_3rd.loc[i, "Err Ar3rd all"] =  np.sqrt((ar.loc[mask_ar3rd, 'Eventos']**2 * ar.loc[mask_ar3rd, 'Error%']**2).sum())/100+0.01

    # --- CF3 ---
    for E in [11.4, 12.5, 14.5, 15.62]:
        mask_cf3 = cf4['Proceso'].str.contains("NEUTRAL DISS") & (cf4['Energia'] > E)
        colname = f"CF3 >{E}"
        poblations_CF3.loc[i, colname] = cf4.loc[mask_cf3, 'Eventos'].sum()
        poblations_CF3.loc[i, f"Err {colname}"] =  np.sqrt((cf4.loc[mask_cf3, 'Eventos']**2 * cf4.loc[mask_cf3, 'Error%']**2).sum())/100

    # --- Ar** ---
    mask_ardbl = ar['Proceso'].str.contains("EXC")
    poblations_Ar_dbleStar.loc[i, "Ar** all"] = ar.loc[mask_ardbl, 'Eventos'].iloc[2:].sum()
    poblations_Ar_dbleStar.loc[i, "Err Ar** all"] = np.sqrt((ar.loc[mask_ardbl, 'Eventos'].iloc[2:]**2 * ar.loc[mask_ardbl, 'Error%'].iloc[2:]**2).sum())/100+0.01

    mask_ardbl2 = ar['Proceso'].str.contains("EXC") & (ar['Energia'] > 11.827)
    poblations_Ar_dbleStar.loc[i, "Ar** 2exc"] = ar.loc[mask_ardbl2, 'Eventos'].iloc[2:].sum()
    poblations_Ar_dbleStar.loc[i, "Err Ar** 2exc"] =  np.sqrt((ar.loc[mask_ardbl2, 'Eventos'].iloc[2:]**2 * ar.loc[mask_ardbl2, 'Error%'].iloc[2:]**2).sum())/100+0.01


########################## Guardado en Pickle/CSV ##############################################

# Diccionario con todas las variables que quieres guarda
to_save = {
    "poblations_CF4": poblations_CF4,
    "poblations_CF3": poblations_CF3,
    "poblations_Ar_dbleStar": poblations_Ar_dbleStar,
    "poblations_Ar_3rd": poblations_Ar_3rd,
}
# Recorremos y guardamos cada uno como .pkl
for name, df in to_save.items():
    df.to_pickle(f"pickle_data/{name}.pkl")
    print(f"✅ Guardado: {name}.pkl")
    
# Recorremos y guardamos cada uno como .csv
for name, df in to_save.items():
    df.to_csv(f"csv_data/{name}.csv", index=False)
    print(f"✅ Guardado: {name}.csv")