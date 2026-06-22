import re
import pandas as pd
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use(['science'])



def read_input(archivo_entrada, archivo_salida_1="collisions_Ar.csv", archivo_salida_2="collisions_CF4.csv", gas1 = "ARGON", gas2 = "CF4"):
    # hecho por chatgpt, coregido por dvl
    def split_by_gas(block_text: str):
        # Encabezados tipo "ARGON ANISOTROPIC ...\n--------"
        headers = list(re.finditer(
            r"^\s*(?P<gas>[A-Z0-9]+)(?:\s+\d{4})?\s+ANISOTROPIC[^\n]*\n[-]{8,}\s*",
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
        df_all[df_all["Gas"] == gas1].to_csv(archivo_salida_1, index=False)
        df_all[df_all["Gas"] == gas2].to_csv(archivo_salida_2, index=False)
        #print(f"Guardado: {archivo_salida_1} y {archivo_salida_2}")

    return df_all

def read_degrad(archivo_entrada, archivo_salida_1, archivo_salida_2, gas1, gas2, concentration, dataframe, output_dir, output_general_name):


    population_gen  = pd.DataFrame({"concentration": concentration})

    ################ Lectura y guardado en dataframe ##############################################
    
    for nombre_col in dataframe.columns:

        population  = pd.DataFrame({"concentration": concentration})
        for i in range(len(archivo_entrada)):

            df = read_input(
                archivo_entrada[i],
                archivo_salida_1=archivo_salida_1[i],
                archivo_salida_2=archivo_salida_2[i],
                gas1 = gas1,
                gas2 = gas2
            )

        

            # Nombre principal con el que seleccionamos la columna 
            name_of_state = dataframe.loc["name principal", nombre_col]

            # Nombre del gas 
            gas = dataframe.loc["gas", nombre_col]
            
            # Limites ingerior y superior energéticos
            energy_upper_limit = dataframe.loc["energy up", nombre_col]
            energy_lower_limit = dataframe.loc["energy low", nombre_col]

            # Nombre del archivo de salida
            name_of_output= dataframe.loc["name output", nombre_col]

            
            df_main_gas = df.loc[df['Gas'] == gas, :]

            mask_aux = pd.Series(True, index=df_main_gas.index)

            for _ in name_of_state:
                mask_aux = df_main_gas['Proceso'].str.contains(_) & mask_aux

            mask = mask_aux & (df_main_gas['Energia'] >= energy_lower_limit) & (df_main_gas['Energia'] < energy_upper_limit)
         
        
            population.loc[i, name_of_output] = df_main_gas.loc[mask, 'Eventos'].sum()
            population.loc[i, "Err"+name_of_output] =  np.sqrt((df_main_gas.loc[mask, 'Eventos']**2 * df_main_gas.loc[mask, 'Error%']**2).sum())/100

            population_gen.loc[i, name_of_output] = df_main_gas.loc[mask, 'Eventos'].sum()
            population_gen.loc[i, "Err"+name_of_output] =  np.sqrt((df_main_gas.loc[mask, 'Eventos']**2 * df_main_gas.loc[mask, 'Error%']**2).sum())/100


            ########################## Guardado en Pickle/CSV ##############################################

        population_gen.fillna(0)
        population.to_csv(f"{output_dir}{name_of_output}.csv", index=False)
        print(f"✅ Guardado: {name_of_output}.csv")

    population_gen.fillna(0)
    population_gen.to_csv(f"{output_general_name}.csv", index=False)
    print(f"✅ Guardado: {name_of_output}.csv")
            #population.to_pickle(f"pickle_data/{name_of_output}.pkl")
            #print(f"✅ Guardado: {name}.pkl")

###############3

