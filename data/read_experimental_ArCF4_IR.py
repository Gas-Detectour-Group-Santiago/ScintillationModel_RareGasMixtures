import os
import re
import numpy as np
import pandas as pd
import dill
import scipy.special
import importlib
import matplotlib.pyplot as plt
try:
    plt.style.use(['science'])
except Exception:
    pass

"""
Script que nos permite leer los datos de los yields infrarrojos (IR)
almacenados en un pickle con formato plano y sacarlos en csv con el mismo
formato de salida que `read_experimental.py`.

Formato esperado del pickle IR:
    - conc_CF4
    - Pressure
    - line_nm
    - yield_IR_norm

La función pública mantiene la misma firma que en el script original:

    read_experimental(
        archivo_entrada,
        yields,
        presiones,
        output_dir,
        concentraciones_reales=None,
        no_sistematic=True,
        output_concentration_name=None,
    )

Notas:
    * `no_sistematic` se acepta por compatibilidad, pero en este pickle no hay
      columnas de incertidumbre. Por tanto, las columnas "Err ..." se rellenan
      con ceros.
    * `yields` se interpreta como las líneas IR a extraer. Se aceptan enteros,
      floats que representen enteros, o strings tipo "696", "696nm",
      "line_696", etc.
"""

#############################################################################################################
########################## FUNCION PARA LEER LOS PICKLES ####################################################
#############################################################################################################

# Cargar el módulo compilado de bajo nivel
_special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")

# Lista de funciones que pueden faltar
funcs = ["erf", "erfc", "erfi", "gamma", "lgamma", "wofz"]

# Inyectarlas si no existen
for name in funcs:
    if not hasattr(_special_ufuncs, name) and hasattr(scipy.special, name):
        setattr(_special_ufuncs, name, getattr(scipy.special, name))


def _find_first_column(df, candidates, what="columna"):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"No encontré {what}. Probé {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )



def _is_nan_like(x):
    try:
        return pd.isna(x)
    except Exception:
        return False



def _extract_item(obj, key):
    """
    Extrae obj[key] si obj es dict/list/array/Series.
    Si no existe, devuelve np.nan.

    Se conserva por compatibilidad con el script original.
    """
    if obj is None or _is_nan_like(obj):
        return np.nan

    # dict
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        # búsqueda case-insensitive si key es str
        if isinstance(key, str):
            lower_map = {str(k).lower(): v for k, v in obj.items()}
            return lower_map.get(key.lower(), np.nan)

        return np.nan

    # pandas Series u objetos con .get
    if hasattr(obj, "get") and not isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return obj.get(key, np.nan)
        except Exception:
            pass

    # lista/tupla/array
    try:
        return obj[key]
    except Exception:
        pass

    # si key es str pero representa entero
    if isinstance(key, str) and key.isdigit():
        try:
            return obj[int(key)]
        except Exception:
            pass

    return np.nan



def _first_non_null(values):
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return None



def _parse_ir_yield_name(yield_name):
    """
    Convierte un identificador de línea IR en el entero asociado a line_nm.

    Acepta, por ejemplo:
        696
        696.0
        "696"
        "696nm"
        "line_696"
        "Ar_727nm"

    Si no puede interpretarlo, lanza ValueError con información útil.
    """
    if isinstance(yield_name, (int, np.integer)):
        return int(yield_name)

    if isinstance(yield_name, (float, np.floating)):
        if float(yield_name).is_integer():
            return int(yield_name)
        raise ValueError(
            f"El yield '{yield_name}' no representa una longitud de onda entera."
        )

    s = str(yield_name).strip()

    # Caso exacto: "696"
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)

    # Buscar el primer entero dentro del string: "696nm", "line_696", etc.
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))

    raise ValueError(
        f"No pude interpretar el yield '{yield_name}' como una línea IR. "
        f"Usa, por ejemplo, 696, '696', '696nm' o 'line_696'."
    )



def _get_series_for_yield(df_pressure, conc_col, yield_name, no_sistematic=True):
    """
    Devuelve:
        s      -> serie de valores
        err_s  -> serie de errores
    indexadas por concentración.

    En el pickle IR no hay columnas de incertidumbre, así que err_s se rellena
    con ceros manteniendo la misma interfaz del script original.
    """
    line_col = _find_first_column(
        df_pressure,
        ["line_nm", "line", "wavelength_nm", "lambda_nm"],
        "columna de línea IR"
    )

    value_col = _find_first_column(
        df_pressure,
        ["yield_IR_norm", "yield_ir_norm", "yield_IR", "yield"],
        "columna de yield IR"
    )

    target_line = _parse_ir_yield_name(yield_name)

    df_line = df_pressure[df_pressure[line_col] == target_line].copy()

    if df_line.empty:
        available_lines = sorted(pd.unique(df_pressure[line_col]).tolist())
        raise KeyError(
            f"La línea IR '{yield_name}' (interpretada como {target_line} nm) "
            f"no aparece en el pickle. Líneas disponibles: {available_lines}"
        )

    s = df_line.set_index(conc_col)[value_col].sort_index()

    # Compatibilidad de salida: este pickle no contiene errores
    err_s = pd.Series(0.0, index=s.index, dtype=float)

    return s, err_s



def read_experimental(
    archivo_entrada,
    yields,
    presiones,
    output_dir,
    concentraciones_reales=None,
    no_sistematic=True,
    output_concentration_name=None,
):
    with open(archivo_entrada, "rb") as f:
        df = dill.load(f)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"El archivo '{archivo_entrada}' no contiene un pandas.DataFrame. "
            f"Tipo detectado: {type(df)}"
        )

    print("Columnas detectadas:")
    print(df.columns)

    # Detectar columnas del formato IR
    pressure_col = _find_first_column(
        df,
        ["Pressure", "pressure", "presion", "presiones", "P (bar)"],
        "columna de presión"
    )

    conc_col = _find_first_column(
        df,
        ["conc_CF4", "concentracion", "concentraciones", "CF4 concentration (%)"],
        "columna de concentración"
    )

    # Nombre de la columna de salida para la concentración
    if output_concentration_name is None:
        if conc_col in {"conc_CF4", "concentracion", "concentraciones"}:
            output_concentration_name = "fCF4"
        else:
            output_concentration_name = conc_col

    # Concentraciones base tomadas de la primera presión pedida
    df_pressure0 = df[df[pressure_col] == presiones[0]].copy()
    if df_pressure0.empty:
        available_pressures = sorted(pd.unique(df[pressure_col]).tolist())
        raise ValueError(
            f"La presión inicial {presiones[0]} no existe en el archivo. "
            f"Presiones disponibles: {available_pressures}"
        )

    concentraciones = np.sort(pd.unique(df_pressure0[conc_col].to_numpy()))

    if concentraciones_reales is not None:
        concentraciones = np.asarray(concentraciones_reales)

    # Índice redondeado para evitar problemas típicos de floats
    try:
        conc_idx = np.round(np.asarray(concentraciones, dtype=float), 8)
        use_numeric_reindex = True
    except Exception:
        conc_idx = np.asarray(concentraciones)
        use_numeric_reindex = False

    os.makedirs(output_dir, exist_ok=True)

    for y in yields:
        yield_out = pd.DataFrame({output_concentration_name: concentraciones})

        for p in presiones:
            df_pressure = df[df[pressure_col] == p].copy()

            if df_pressure.empty:
                yield_out[f"{p:.1f}bar"] = 0.0
                yield_out[f"Err {p:.1f}bar"] = 0.0
                continue

            s, err_s = _get_series_for_yield(
                df_pressure=df_pressure,
                conc_col=conc_col,
                yield_name=y,
                no_sistematic=no_sistematic,
            )

            if use_numeric_reindex:
                try:
                    s.index = np.round(s.index.astype(float), 8)
                    err_s.index = np.round(err_s.index.astype(float), 8)
                    target_idx = conc_idx
                except Exception:
                    target_idx = concentraciones
            else:
                target_idx = concentraciones

            yield_out[f"{p:.1f}bar"] = pd.Series(s).reindex(target_idx).to_numpy()
            yield_out[f"Err {p:.1f}bar"] = pd.Series(err_s).reindex(target_idx).to_numpy()

        yield_out = yield_out.fillna(0)

        # Por si el nombre del yield lleva barras o cosas raras
        safe_y = str(y).replace("/", "_")
        out_path = os.path.join(output_dir, f"{safe_y}.csv")
        yield_out.to_csv(out_path, index=False)
        print(f"✅ Guardado: {out_path}")


if __name__ == "__main__":
    # Ejemplo mínimo de uso:
    archivo_entrada = "IR_yields.pkl"
    yields = [696, 727, 750, 763, 772, 794]
    presiones = [1, 2, 3, 4, 5]
    output_dir = "../data/Experimental/ArCF4/"

    read_experimental(
        archivo_entrada,
        yields,
        presiones,
        output_dir,
        concentraciones_reales=None,
        no_sistematic=True,
    )
