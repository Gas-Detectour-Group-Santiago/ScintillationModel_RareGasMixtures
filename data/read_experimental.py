import os
import numpy as np
import pandas as pd
try:
    import dill
except ModuleNotFoundError:  # fallback for environments without dill
    import pickle as dill
import scipy.special
import importlib
import matplotlib.pyplot as plt

try:
    import scienceplots  # registra el estilo 'science'
    plt.style.use(['science'])
except Exception:
    pass

"""
Script que nos permite leer los datos de los yields de visible/ultravioleta/infrarrojo,
sacándolos en formato pickle y csv.
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
        out = pd.isna(x)
        if isinstance(out, (bool, np.bool_)):
            return bool(out)
        return False
    except Exception:
        return False


def _is_missing_value(x):
    """True solo para None/NaN escalares, no para dict/list/array."""
    if x is None:
        return True
    return _is_nan_like(x)


def _extract_item(obj, key):
    """
    Extrae obj[key] si obj es dict/list/array/Series.
    Si no existe, devuelve np.nan.
    """
    if obj is None or _is_nan_like(obj):
        return np.nan

    # dict
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        # Si key es str y representa un entero, prueba también con int(key).
        # Esto es necesario para líneas IR guardadas como claves enteras: 696, 727, ...
        if isinstance(key, str) and key.strip().isdigit():
            ikey = int(key.strip())
            if ikey in obj:
                return obj[ikey]

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
    if isinstance(key, str) and key.strip().isdigit():
        try:
            return obj[int(key.strip())]
        except Exception:
            pass

    return np.nan


def _extract_from_nested_dict(obj, outer_key, inner_key):
    """Extrae obj[outer_key][inner_key] tolerando claves str/int."""
    outer = _extract_item(obj, outer_key)
    if _is_missing_value(outer):
        return np.nan
    return _extract_item(outer, inner_key)


def _extract_yield_from_zonas(obj, yield_name, yield_mode="auto"):
    """
    Extrae un yield desde columnas tipo yields_zonas/u_yields_zonas.

    yield_mode:
        - "auto": primero intenta la clave directa; si no existe, busca dentro de "ir".
        - "direct": busca solo la clave directa: "UV", "vis", etc.
        - "ir": busca dentro de obj["ir"][yield_name], por ejemplo obj["ir"][696].
    """
    yield_mode = _normalize_yield_mode(yield_mode)

    if yield_mode == "ir":
        return _extract_from_nested_dict(obj, "ir", yield_name)

    if yield_mode == "direct":
        return _extract_item(obj, yield_name)

    # auto: compatible con UV/vis antiguos y con líneas IR dentro de yields_zonas["ir"]
    direct_value = _extract_item(obj, yield_name)
    if not _is_missing_value(direct_value):
        return direct_value

    ir_value = _extract_from_nested_dict(obj, "ir", yield_name)
    if not _is_missing_value(ir_value):
        return ir_value

    return np.nan


def _first_non_null(values):
    for v in values:
        if v is None:
            continue
        try:
            out = pd.isna(v)
            if isinstance(out, (bool, np.bool_)) and out:
                continue
        except Exception:
            pass
        return v
    return None


def _normalize_uncertainty_mode(uncertainty_mode):
    if not isinstance(uncertainty_mode, str):
        raise ValueError("introduzca una válida")

    mode = uncertainty_mode.strip().lower()

    aliases = {
        "stadistic": "stadistic",
        "statistic": "stadistic",
        "statistical": "stadistic",
        "estadistico": "stadistic",
        "estadística": "stadistic",
        "estadistica": "stadistic",
        "sistematic": "sistematic",
        "systematic": "sistematic",
        "sistematico": "sistematic",
        "sistemática": "sistematic",
        "sistematica": "sistematic",
        "all": "all",
        "combined": "all",
        "total": "all",
        "todos": "all",
    }

    mode = aliases.get(mode, mode)
    valid_modes = {"stadistic", "sistematic", "all"}

    if mode not in valid_modes:
        raise ValueError("introduzca una válida")

    return mode


def _normalize_yield_mode(yield_mode):
    if not isinstance(yield_mode, str):
        raise ValueError("yield_mode debe ser 'auto', 'direct' o 'ir'")

    mode = yield_mode.strip().lower()

    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "normal": "auto",
        "zona": "auto",
        "zonas": "auto",
        "direct": "direct",
        "directo": "direct",
        "uvvis": "direct",
        "uv/vis": "direct",
        "visible": "direct",
        "vis": "direct",
        "uv": "direct",
        "ir": "ir",
        "infrared": "ir",
        "infrarred": "ir",       # por compatibilidad con tu nombre de módulo
        "infrarrojo": "ir",
        "infrarrojos": "ir",
        "infra-rojo": "ir",
    }

    mode = aliases.get(mode, mode)
    valid_modes = {"auto", "direct", "ir"}

    if mode not in valid_modes:
        raise ValueError("yield_mode debe ser 'auto', 'direct' o 'ir'")

    return mode


def _get_error_candidates(mode, schema="new", is_n2=False):
    if schema == "old":
        mapping = {
            "stadistic": ["uyields_estadistico", "u_yields_estadistico"],
            "sistematic": [
                "uyields_sistematico", "u_yields_sistematico",
                "uyields_systematic", "u_yields_systematic",
                "uyields_cal",
            ],
            "all": ["u_yields_zonas"],
        }
        return mapping[mode]

    if is_n2:
        mapping = {
            "stadistic": ["u_yield_n2_estadistico"],
            "sistematic": [
                "u_yield_n2_sistematico", "u_yield_n2_systematic",
                "u_yield_n2_cal",
            ],
            "all": ["u_yield_n2_combined"],
        }
        return mapping[mode]

    mapping = {
        "stadistic": ["u_yields_estadistico"],
        "sistematic": [
            "u_yields_sistematico", "u_yields_systematic",
            "u_yields_cal", "u_yields_picos_cal",
        ],
        "all": ["u_yields_picos"],
    }
    return mapping[mode]


def _get_series_for_yield(
    df_pressure,
    conc_col,
    yield_name,
    uncertainty_mode="stadistic",
    yield_mode="auto",
):
    """
    Devuelve:
        s      -> serie de valores
        err_s  -> serie de errores
    indexadas por concentración.
    """
    uncertainty_mode = _normalize_uncertainty_mode(uncertainty_mode)
    yield_mode = _normalize_yield_mode(yield_mode)

    # =========================
    # ESQUEMA VIEJO
    # =========================
    if "yields_zonas" in df_pressure.columns:
        value_col = "yields_zonas"
        err_col = _find_first_column(
            df_pressure,
            _get_error_candidates(uncertainty_mode, schema="old"),
            f"columna de error '{uncertainty_mode}' del esquema viejo"
        )

        s = df_pressure.set_index(conc_col)[value_col].apply(
            lambda d: _extract_yield_from_zonas(d, yield_name, yield_mode=yield_mode)
        )
        err_s = df_pressure.set_index(conc_col)[err_col].apply(
            lambda d: _extract_yield_from_zonas(d, yield_name, yield_mode=yield_mode)
        )

        # Si no encontró nada, da info útil
        if s.isna().all():
            example = _first_non_null(df_pressure[value_col].tolist())
            if isinstance(example, dict):
                available = list(example.keys())
                ir_available = []
                if isinstance(example.get("ir"), dict):
                    ir_available = list(example["ir"].keys())
                raise KeyError(
                    f"El yield '{yield_name}' no aparece en '{value_col}' usando yield_mode='{yield_mode}'. "
                    f"Claves superiores disponibles: {available}. "
                    f"Líneas IR disponibles: {ir_available}"
                )

        return s, err_s

    # =========================
    # ESQUEMA NUEVO
    # =========================
    yn = str(yield_name).strip().lower()

    # Caso especial: yield total de N2
    if yn in {"yield_n2", "n2", "total_n2", "yield n2"}:
        value_col = _find_first_column(
            df_pressure,
            ["yield_N2"],
            "columna yield_N2"
        )

        err_col = _find_first_column(
            df_pressure,
            _get_error_candidates(uncertainty_mode, schema="new", is_n2=True),
            f"columna de error '{uncertainty_mode}' para yield_N2"
        )

        s = df_pressure.set_index(conc_col)[value_col]
        err_s = df_pressure.set_index(conc_col)[err_col]
        return s, err_s

    # Resto: asumimos yields_picos
    value_col = _find_first_column(
        df_pressure,
        ["yields_picos"],
        "columna yields_picos"
    )

    err_col = _find_first_column(
        df_pressure,
        _get_error_candidates(uncertainty_mode, schema="new", is_n2=False),
        f"columna de error '{uncertainty_mode}' para yields_picos"
    )

    s = df_pressure.set_index(conc_col)[value_col].apply(lambda d: _extract_item(d, yield_name))
    err_s = df_pressure.set_index(conc_col)[err_col].apply(lambda d: _extract_item(d, yield_name))

    # Si no encontró nada, da info útil
    if s.isna().all():
        example = _first_non_null(df_pressure[value_col].tolist())
        if isinstance(example, dict):
            raise KeyError(
                f"El yield '{yield_name}' no aparece en '{value_col}'. "
                f"Claves de ejemplo disponibles: {list(example.keys())}"
            )

    return s, err_s


def read_experimental(
    archivo_entrada,
    yields,
    presiones,
    output_dir,
    concentraciones_reales=None,
    uncertainty_mode="stadistic",
    output_concentration_name=None,
    yield_mode="auto",
):
    """
    Lee un pickle experimental y genera un CSV por yield.

    Parámetros nuevos:
        yield_mode="auto"  -> detecta UV/vis directos y también líneas dentro de yields_zonas["ir"].
        yield_mode="ir"    -> fuerza la lectura de infrarrojos: yields_zonas["ir"][696], etc.
        yield_mode="direct"-> fuerza la lectura directa: yields_zonas["UV"], yields_zonas["vis"], etc.
    """
    uncertainty_mode = _normalize_uncertainty_mode(uncertainty_mode)
    yield_mode = _normalize_yield_mode(yield_mode)

    with open(archivo_entrada, "rb") as f:
        df = dill.load(f)

    print("Columnas detectadas:")
    print(df.columns)

    # Detectar columnas compatibles con ambos formatos
    pressure_col = _find_first_column(
        df,
        ["presion", "presiones", "P (bar)"],
        "columna de presión"
    )

    conc_col = _find_first_column(
        df,
        ["concentracion", "concentraciones", "N2 concentration (%)", "CF4 concentration (%)"],
        "columna de concentración"
    )

    # Nombre de la columna de salida para la concentración
    if output_concentration_name is None:
        if conc_col in {"concentracion", "concentraciones"}:
            output_concentration_name = "fCF4"
        else:
            output_concentration_name = conc_col

    # Concentraciones base
    df_pressure0 = df[df[pressure_col] == presiones[0]].copy()
    concentraciones = df_pressure0[conc_col].to_numpy()

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
                uncertainty_mode=uncertainty_mode,
                yield_mode=yield_mode,
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

        safe_y = str(y).replace("/", "_")
        out_path = os.path.join(output_dir, f"{safe_y}.csv")
        yield_out.to_csv(out_path, index=False)
        print(f"✅ Guardado: {out_path}")
