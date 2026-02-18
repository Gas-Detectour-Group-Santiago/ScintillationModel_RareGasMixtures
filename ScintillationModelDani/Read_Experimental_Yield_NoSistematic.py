
import numpy as np 
import pandas as pd 
import dill
import scipy.special
import importlib

"""
Script que nos permite leer los datos de los yields de visible/ultravioleta, sacándolos en formato pickle y csv.
"""


#############################################################################################################
########################## FUNCION PARA LEER LOS PICKLES ##############################################
#############################################################################################################

# no se que demonios pasaba aqui, chatgpt es el puto amo
# Cargar el módulo compilado de bajo nivel
_special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")

# Lista de funciones que pueden faltar
funcs = ["erf", "erfc", "erfi", "gamma", "lgamma","wofz"]

# Inyectarlas si no existen
for name in funcs:
    if not hasattr(_special_ufuncs, name) and hasattr(scipy.special, name):
        setattr(_special_ufuncs, name, getattr(scipy.special, name))
        #print(f"🔧 Añadida función faltante: {name}")

with open("input/CF4_primary_data_final.pkl", "rb") as f:
    df = dill.load(f)

#############################################################################################################
########################## PROGRAMA PRINCIPAL ##############################################
#############################################################################################################

######################## Inicialización de dataframe ################################################


yield_uv       = pd.DataFrame()
yield_vis      = pd.DataFrame()

presion = np.array([1,2,2.5,3,4,5])
concetraciones_reales = np.array([0,0.096 , 0.28 , 0.434 , 0.929 , 1.64 , 5.55 , 10.25 , 100])

name_uv = ["fCF4","fCF4 real", "Err fCF4 real"]
name_vis = ["fCF4","fCF4 real", "Err fCF4 real"]

for i in presion:
    name_uv += ["%.1fbar"%i]
    name_uv += ["Err %.1fbar"%i]
    name_vis += ["%.1fbar"%i]
    name_vis += ["Err %.1fbar"%i]
    
for col in name_uv:
    yield_uv[col] = [np.nan]*9
    
for col in name_vis:
    yield_vis[col] = [np.nan]*9

    
################ Lectura y guardado en dataframe ##############################################

# concentraciones objetivo (de la primera presión, por ejemplo)
df_pressure0 = df[df["presion"] == presion[0]].copy()
concentraciones_og = df_pressure0["concentracion"].to_numpy()

print(concentraciones_og)
concentraciones_og2 = concentraciones_og.copy()
concentraciones_og2[0] = 1e-3

# Inicializa las tablas destino alineadas por fCF4
yield_uv  = pd.DataFrame({"fCF4": concentraciones_og,
                          "fCF4 real": concentraciones_og2,
                          "Err fCF4 real": concentraciones_og2*0.1})

yield_vis = pd.DataFrame({"fCF4": concentraciones_og,
                          "fCF4 real": concentraciones_og2,
                          "Err fCF4 real": concentraciones_og2*0.1})



for i in range(0,len(presion)):
    # OJO: i y presion alineados 1:1
    df_pressure = df[df["presion"] == presion[i]].copy()
    
    
    #'uyields_cal', 'uyields_estadistico'
    # Extrae UV y vis como Series indexadas por concentracion
    s_uv = df_pressure.set_index("concentracion")["yields_zonas"].apply(lambda d: d["UV"])
    s_vis = df_pressure.set_index("concentracion")["yields_zonas"].apply(lambda d: d["vis"])
    
    err_s_uv  = df_pressure.set_index("concentracion")["uyields_estadistico"].apply(lambda d: d["UV"])
    err_s_vis = df_pressure.set_index("concentracion")["uyields_estadistico"].apply(lambda d: d["vis"])


    # Si las concentraciones son floats con posibles decimales “sucios”, puedes redondear:
    # s_uv.index  = np.round(s_uv.index.astype(float), 8)
    # s_vis.index = np.round(s_vis.index.astype(float), 8)
    # conc_idx    = np.round(concentraciones_og.astype(float), 8)
    # Luego reindexar con conc_idx

    # Reindexa para alinear por concentración objetivo
    col_uv  = name_uv[i*2+3]
    col_vis = name_vis[i*2+3]
    err_col_uv  = name_uv[i*2+1+3]
    err_col_vis = name_vis[i*2+1+3]

    yield_uv[col_uv]  = pd.Series(s_uv).reindex(concentraciones_og).to_numpy()
    yield_vis[col_vis]= pd.Series(s_vis).reindex(concentraciones_og).to_numpy()
    yield_uv[err_col_uv]  = pd.Series(err_s_uv).reindex(concentraciones_og).to_numpy()
    yield_vis[err_col_vis]= pd.Series(err_s_vis).reindex(concentraciones_og).to_numpy()





# Rellena faltantes con -1 si quieres ese marcador
yield_uv  = yield_uv.fillna(0)
yield_vis = yield_vis.fillna(0)
    
########################## Guardado en Pickle/CSV ##############################################
# Diccionario con todas las variables que quieres guardar
yield_uv = yield_uv#.drop(index=0)
yield_vis = yield_vis.drop(index=0)

to_save = {
    "yield_uv": yield_uv,
    "yield_vis": yield_vis,
}
            
# Recorremos y guardamos cada uno como .pkl
for name, df in to_save.items():
    df.to_pickle(f"pickle_data/{name}_cal.pkl")
    print(f"✅ Guardado: {name}_cal.pkl")
    
# Recorremos y guardamos cada uno como .csv
for name, df in to_save.items():
    df.to_csv(f"csv_data/{name}_cal.csv", index=False)
    print(f"✅ Guardado: {name}_cal.csv")