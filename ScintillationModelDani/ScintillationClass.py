import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import scipy.optimize as opt
import inspect
import re



def normalize_tokens(s):
    """Convierte 'Ar dble Star' → ['ar','dble','star']."""
    s = s.lower().replace("_", " ").strip()
    return [tok for tok in re.split(r"[ \-]+", s) if tok]


def match_param_to_species(param, species):
    """
    Empareja P_CF3 <-> CF3
              P_Ar_dbleStar <-> Ar dble Star
              P_Ar3rd <-> Ar 3rd

    Regla:
        TODOS los tokens del parámetro deben existir en la especie.
    """

    # quitar prefijo P_
    if param.lower().startswith("p_"):
        param_clean = param[2:]
    else:
        param_clean = param

    param_tokens   = normalize_tokens(param_clean)
    species_tokens = normalize_tokens(species)

    # Condición correcta: todos los tokens de param están en species
    return all(tok in species_tokens for tok in param_tokens)

def darken(color, factor=0.9):
    """
    Mezcla el color con negro.
    factor=0 → negro
    factor=1 → color original
    """
    r, g, b, a = color
    return (factor*r, factor*g, factor*b, a)


###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
############# CORRECCIÓN DE LAS POBLACIONES DE DEGRAD ##############################

class Scintillation:
    """
    Clase para almacenar distintos diccionarios o DataFrames relacionados con:
    - Poblaciones de yields (vis, uv)
    - Poblaciones de degradación
    - Funciones de centelleo
    Además hace:
    - Comparaión entre modelos de centelleo con poblaciones de Degrad y datos experimentales
    - Ajuste a nuevos modelos de poblaciones de Degrad
    - Expansión de poblaciones de Degrad a diferentes concentraciones a través de interpolación.
    - Grafica de modelos de Centelleo a parámetros dados o ajustados, a presión elegida, con datos experimentales. 
    - Diferentes opciones a la hora de graficar.
    """

    def __init__(
        self,
        yields: Optional[Dict[str, List[Any]]] = None,
        poblation_degrad: Optional[Dict[str, List[Any]]] = None,
        scintillation_models: Optional[Dict[str, List[Any]]] = None,
    ):
        # Entrada, con lo que se define: 
        self.yields = yields
        self.poblation_degrad = poblation_degrad
        self.scintillation_models = scintillation_models
        
        # Importantes
        self.fCF4 = self.yields["fCF4"]/100
        self.min_fCF4_10log = np.log10(np.min(self.fCF4))
        self.max_fCF4_10log = np.log10(np.max(self.fCF4))
        self.fCF4_orig = self.poblation_degrad["fCF4"]
        
        # Corrección de Degrad con interpolacion
        self.poblation_degrad_corr=self._compute_poblation_degrad_corr() 
        
        # Inicializamos donde se guardan los resultados de los fits
        self.fit_results        = {}
        self.fit_results_global = {}
        
        # Inicializamos donde se guardan los resultados de los fits
        self.contributions        = {}
        self.contributions_global = {}

        # Para las gráficas de los fits
        self.plot_settings = {
            "normalization": {},   # band → config
            "show_exp": {},        # band → list of pressures
            "show_teo": {},        # band → list of pressures
            "global_norm": {}      # NUEVO
        }

    ###################################################################################
    ############# CORRECCIÓN DE LAS POBLACIONES DE DEGRAD ##############################
    
    def _compute_poblation_degrad_corr(self) -> Dict[str, Any]:
        """
        Interpola cada DataFrame de poblation_degrad usando la malla nueva self.fCF4.
        El eje original es self.poblation_degrad["fCF4"].
        """
        result = {}

        x_orig = self.fCF4_orig   # eje original
        x_new  = self.fCF4        # eje nuevo

        for name, obj in self.poblation_degrad.items():

            # saltar la clave del eje
            if name == "fCF4":
                continue

            # solo DataFrames
            if isinstance(obj, pd.DataFrame):

                corr_df = pd.DataFrame(index=x_new)

                for col in obj.columns:

                    if "Err" in col:
                        continue   # saltamos errores

                    y_old = obj[col].to_numpy(dtype=float)
                    y_new = np.interp(x_new, x_orig, y_old)

                    corr_df[col] = y_new

                result[name] = corr_df

        return result

    ###################################################################################
    #############  GRAFICO POBLACIONES DE DEGRAD INTERPOLADAS ##############################
    
    def plotPoblationInterpolation(self, name,savefig=""):

        original = self.poblation_degrad[name]

        x_orig = self.fCF4_orig   # ESTE es el eje original real

        plt.figure(figsize=(7,5))

        for col in original.columns:

            if "Err" in col:
                continue

            y_orig = original[col].to_numpy()

            # puntos rojos: datos originales
            plt.scatter(x_orig, y_orig, color="red", s=40)

            # línea azul: recta trozo a trozo entre datos originales
            plt.plot(x_orig, y_orig, color="blue", linewidth=2)

        plt.xlabel("fCF4")
        plt.ylabel("Población")
        plt.title(f"Interpolación lineal trozo a trozo de {name}")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        if not(savefig==""):
            plt.savefig(savefig,dpi=300)
        else:
            plt.show()

    #############################################################################
    ############## FUNCIONES QUE AÑADEN INFORMACION ##############################

    def add_yields(self, key: str, values: List[Any]):
        if self.yields is None:
            self.yields = pd.DataFrame({key: pd.Series(values)})
        else:
            self.yields[key] = pd.Series(values)

    def add_poblation_degrad(self, key: str, values: List[Any]):
        if self.poblation_degrad is None:
            self.poblation_degrad = pd.DataFrame({key: pd.Series(values)})
        else:
            self.poblation_degrad[key] = pd.Series(values)

    def add_scintillation_models(self, key: str, values: List[Any]):
        if self.scintillation_models is None:
            self.scintillation_models = pd.DataFrame({key: pd.Series(values)})
        else:
            self.scintillation_models[key] = pd.Series(values)

    ###########################################################################################
    ############# FUNCIONES QUE CREAN LAS COMBINACIONES TEORICAS ##############################
    ###########################################################################################
        
       
    def buildTheoryFunction(self, scintillation_definition):
        """
        Construye funciones teóricas combinando modelos físicos y pesos.
        Guarda las funciones resultantes en self.theory_functions.
        Sirven para obtener los parámetros del ajuste.
        """
        self.theory_functions = {}
        contribs = {}   
        
        for band_name, components in scintillation_definition.items():

            comp = list(components.items())  # para cerrar bien valores

            def make_theory_func(comp):

                def theory_func(x, fCF4, n, **kwargs):
                    total = 0.0
                    idx = 0

                    for model_name, properties in comp:

                        model_func = self.scintillation_models[model_name]
                        modes = properties   # puede haber varias (["Relajacion","Centelleo"])
                        weight = 1.0         # empezamos con factor 1

                        for mode in modes:

                            if mode == "Probabilidad":
                                weight *= x[idx]
                                idx += 1

                            elif mode == "Relajacion":
                                k = x[idx]
                                weight *= n * fCF4 / (k + n * fCF4)
                                idx += 1

                            elif mode == "Centelleo":
                                k = x[idx]
                                weight *= k / (k + n * fCF4)
                                idx += 1

                            else:
                                weight *= 1.0


                        # ===== 2. Parámetros del modelo físico =====
                        sig = inspect.signature(model_func)
                        
                        params = {}

                        for par in sig.parameters.values():
                            pname = par.name
                            

                            if pname == "f_cf4":
                                params[pname] = fCF4
                                continue

                            if pname == "n":
                                params[pname] = n
                                continue

                            # --- BÚSQUEDA GENERAL DE LA POBLACIÓN ---
                            found = False
                            
                            for species, df in self.poblation_degrad.items():

                                # ⛔ Saltar la malla fCF4 porque NO es un DataFrame
                                if species == "fCF4":
                                    continue
                                
                                flag = match_param_to_species(pname,species)
                                #flag=True
                                if flag: 
                                    valid_cols = [c for c in df.columns
                                                if "err" not in c.lower()]

                                    if not valid_cols:
                                        continue
                                    
                                    # primera columna válida (general para cualquier gas)
                                    col = valid_cols[0]
                                    y_old = df[col].to_numpy()
                                    y_new = np.interp(fCF4, self.fCF4_orig, y_old)

                                    
                                    params[pname] = y_new
                                    
                                    found = True
                                    
                            
                            #print(params)
                            if not found:
                                raise ValueError(
                                    f"No se encontró población para parámetro '{pname}' "
                                    f"en modelo '{model_name}'."
                                )

                        # ===== 3. Suma ponderada =====
                        
                        total += weight * model_func(**params)
                        contribs[model_name] = weight * model_func(**params)

                    #return total
                    if kwargsfitParamtersWithNormalization.get("return_components", False):
                        return total, contribs
                    return total

                return theory_func

            self.theory_functions[band_name] = make_theory_func(comp)
            
    ##########################################################################
    ######################### FUNCIONES GLOBALES ###########################
    ##########################################################################

    # Dentro de class Scintillation:  ---------------------------------
    def buildYieldFunctionsFromRaw(self, scintillation_raw):
        """
        Recibe un diccionario de funciones 'crudas' que dependen explícitamente
        de las poblaciones de Degrad, por ejemplo:

            def theory_yield_vis(x, fCF4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd): ...

        y devuelve un diccionario de funciones efectivas:

            f_vis(x, fCF4, n), f_uv(x, fCF4, n)

        donde las poblaciones P_* se obtienen automáticamente de
        self.poblation_degrad_corr en función de fCF4.
        """

        result = {}

        for band, func in scintillation_raw.items():
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            # Esperamos algo tipo: x, fCF4, n, P_CF3, P_Ar_dbleStar, ...
            if len(params) < 3:
                raise ValueError(
                    f"La función '{func.__name__}' de banda '{band}' debe "
                    "tener al menos argumentos (x, fCF4, n, ...)."
                )

            # Nombres de parámetros de poblaciones: desde el 4.º en adelante
            pop_param_names = [p.name for p in params[3:]]

            def make_wrapper(func, pop_param_names):
                def wrapped(x, fCF4, n):
                    fCF4_arr = np.asarray(fCF4, dtype=float)

                    pop_kwargs = {}

                    for pname in pop_param_names:
                        # Buscamos la especie correspondiente usando el helper
                        found = False

                        for species, df in self.poblation_degrad_corr.items():

                            if species == "fCF4":
                                continue

                            if match_param_to_species(pname, species):
                                # Elegimos primera columna sin "err"
                                valid_cols = [
                                    c for c in df.columns
                                    if "err" not in c.lower()
                                ]
                                if not valid_cols:
                                    continue

                                col = valid_cols[0]
                                y_old = df[col].to_numpy(dtype=float)

                                # self.fCF4 es la malla nueva sobre la que está y_old
                                # Interpolamos a los fCF4 que nos pasen
                                y_new = np.interp(
                                    fCF4_arr,
                                    self.fCF4,   # eje ya corregido
                                    y_old
                                )

                                pop_kwargs[pname] = y_new
                                found = True
                                break

                        if not found:
                            raise ValueError(
                                f"No se encontró población adecuada para el "
                                f"parámetro '{pname}' en la banda '{band}'."
                            )

                    # Llamamos a la función original metiéndole las poblaciones ya calculadas
                    return func(x, fCF4_arr, n, **pop_kwargs)

                return wrapped

            result[band] = make_wrapper(func, pop_param_names)

        # opcional: guardarlo en self.theory_functions para el flujo estándar
        self.theory_functions = result

        return result

    ##########################################################################
    ######################### AJUSTE DE PARÁMETROS ###########################
    ##########################################################################

    def fitParamtersWithNormalization(
        self,
        band: str,
        x0: np.ndarray,
        n0: float = 1.0,
        idx_ref: int = -1,
        method: str = "BFGS",
    ):
        """
        Parametros: 
            - **band**: elijes a cual ajusta de los datos experimentales ("vis","uv"...)
            - **x0**: valores iniciales de los parámetros
        Ajusta los parámetros x para el canal `band` ('vis', 'uv', ...)
        usando una normalización a un punto de referencia:
            - presión de referencia n0 (ej. 1.0 bar)
            - índice idx_ref en la malla fCF4 (ej. -1 → último punto)

        Chi² se calcula sobre cantidades normalizadas:
            y_exp_norm  = y_exp / y_exp_ref
            y_th_norm   = y_th  / y_th_ref
        """

        if not hasattr(self, "theory_functions") or band not in self.theory_functions:
            raise ValueError(
                f"No existe función teórica para el canal '{band}'. "
                "Llama antes a build_theory_functions()."
            )

        if band not in self.yields:
            raise ValueError(
                f"No hay datos de yield para el canal '{band}' en self.yields."
            )

        theory_func = self.theory_functions[band]
        df_yield = self.yields[band]
        fCF4 = self.fCF4

        # Columnas físicas (sin errores)
        cols_phys = [c for c in df_yield.columns if "err" not in c.lower()]

        # Buscamos la columna cuya presión coincida con n0 (por ejemplo "1.0bar")
        col_ref = None
        for col in cols_phys:
            try:
                n_col = float(col.replace("bar", ""))
            except Exception:
                continue
            if abs(n_col - n0) < 1e-6:
                col_ref = col
                break

        if col_ref is None:
            raise ValueError(
                f"No se encontró ninguna columna de presión ~={n0} bar para el canal '{band}'."
            )

        # Esta función construye el chi² con normalización a n0, idx_ref
        def chi2(x):
            chi2_val = 0.0

            # Primero obtenemos la referencia experimental y teórica en n0
            y_ref_exp = df_yield[col_ref].to_numpy()
            y0_exp = y_ref_exp[idx_ref]

            # si el valor de referencia experimental es 0, evitar problemas
            if y0_exp == 0:
                return 1e30  # penalizamos esta elección de x

            # error asociado a la referencia (si existe)
            err_ref = None
            for ec in [f"Err {col_ref}", f"Err_{col_ref}", f"{col_ref} Err", f"{col_ref}_Err"]:
                if ec in df_yield.columns:
                    err_ref = df_yield[ec].to_numpy()
                    break

            # Recorremos todas las columnas físicas -> Datos Experimentales 
            for col in cols_phys:
                y_exp = df_yield[col].to_numpy()

                # errores
                sigma = None
                for ec in [f"Err {col}", f"Err_{col}", f"{col} Err", f"{col}_Err"]:
                    if ec in df_yield.columns:
                        sigma = df_yield[ec].to_numpy()
                        break
                if sigma is None:
                    sigma = np.ones_like(y_exp)

                # presión de esta columna
                try:
                    n_val = float(col.replace("bar", ""))
                except Exception:
                    n_val = 1.0

                # modelo teórico
                y_th = theory_func(x=x, fCF4=fCF4, n=n_val)

                # referencia teórica en n0 (misma idx_ref)
                y_ref_th = theory_func(x=x, fCF4=fCF4, n=n0)
                y0_th = y_ref_th[idx_ref]

                if y0_th == 0:
                    return 1e30

                # normalizamos datos y modelo
                y_exp_norm = y_exp / y0_exp
                y_th_norm = y_th / y0_th

                # normalizamos errores (simplificado)
                sigma_norm = sigma / abs(y0_exp)
                mask = sigma_norm > 0
                chi2_val += np.sum(((y_th_norm[mask] - y_exp_norm[mask]) / sigma_norm[mask]) ** 2)

            return chi2_val

        result = opt.minimize(chi2, x0, method=method)

        if not hasattr(self, "fit_results"):
            self.fit_results = {}
    
        popt = result.x
        self.fit_results[band] = popt
        return popt
    

    ###################################################################################
    ######################### AJUSTE DE PARÁMETROS GLOBALES ###########################
    ###################################################################################
    
    def fitParametersGlobalRaw_residuals(self, bands, x0, bounds):
        """
        Ajuste global de parámetros usando least_squares y un vector de residuos:

            r = (Y_exp - Y_th) / sigma

        El objetivo es minimizar ||r||^2, equivalente a minimizar chi².

        Parameters
        ----------
        bands : list[str]
            Lista de bandas a ajustar, por ejemplo ["vis", "uv"]
        x0 : ndarray
            Vector inicial de parámetros (N_global en x[0], etc)
        bounds : (lower, upper)
            Límites inferiores y superiores para cada parámetro

        Guarda:
        -------
        self.fit_results["global"]  → parámetros ajustados
        self.global_fit_info        → objeto result de least_squares
        """

        

        #------------------------------------------
        # Comprobaciones mínimas
        #------------------------------------------
        for band in bands:
            if band not in self.theory_functions:
                raise ValueError(f"No existe teoría para banda '{band}'.")
            if band not in self.yields:
                raise ValueError(f"No hay datos experimentales para banda '{band}'.")

        fCF4_data = self.fCF4   # eje de yield experimental

        #------------------------------------------
        # Construcción del vector de residuos
        #------------------------------------------
        def residuals(x):
            res_list = []

            for band in bands:
                dfY = self.yields[band]

                # columnas físicas (p.ej., "1bar", "4bar", ...)
                cols_phys = [
                    c for c in dfY.columns
                    if ("err" not in c.lower() and "fcf4" not in c.lower())
                ]

                for col in cols_phys:
                    # experimental
                    y_exp = dfY[col].to_numpy(dtype=float)

                    # errores experimentales
                    err_col_candidates = [
                        f"Err {col}", f"Err_{col}", f"{col} Err", f"{col}_Err"
                    ]
                    s_exp = None
                    for ec in err_col_candidates:
                        if ec in dfY.columns:
                            s_exp = dfY[ec].to_numpy(dtype=float)
                            break

                    if s_exp is None:
                        s_exp = np.ones_like(y_exp)

                    # Evitar σ = 0
                    s_exp_eff = s_exp.copy()
                    mask0 = (s_exp_eff == 0)
                    if np.any(mask0):
                        s_exp_eff[mask0] = 1e+12

                    # presión
                    try:
                        n_val = float(col.replace("bar", ""))
                    except:
                        n_val = 1.0

                    # teoría
                    y_th = self.theory_functions[band](x, fCF4_data, n_val)

                    if len(y_th) > len(y_exp):
                        n = len(y_th)  -len(y_exp)
                        y_th = y_th[n:]
                            
                    # residuo
                    res = (y_exp - y_th) / s_exp_eff
                    res_list.append(res)

            return np.concatenate(res_list)

        #------------------------------------------
        # Ajuste least_squares
        #------------------------------------------
        result = opt.least_squares(
            residuals,
            x0,
            bounds=bounds,
            method="trf",  # método estable para problemas con bounds
            verbose=2      # si quieres texto de diagnóstico
        )

        # Guardar resultado
        self.fit_results["global"] = result.x
        self.global_fit_info = result

        return result
    

    ###################################################################################
    ######################### Exprotamos a csv ###########################
    ###################################################################################
    
    def exportParamsToCSV(self, archive="params.csv", names=None, band="global"):
        """
        Exporta parámetros ajustados y sus incertidumbres a un CSV usando pandas.
        Además, para el ajuste global exporta un segundo CSV con la matriz de correlaciones.

        CSV 1 (archive):
            FILA 1 → valores
            FILA 2 → incertidumbres
            Columnas → nombres de parámetros

        CSV 2 (archive + '_corrMatrix.csv'):
            Matriz de correlación entre parámetros
        """

        import pandas as pd
        import numpy as np

        cov = None  # matriz de covarianza
        corr = None # matriz de correlación

        # ============================================================
        # 1. Global o banda
        # ============================================================
        if band == "global":
            if "global" not in self.fit_results:
                raise ValueError("No se ha ejecutado un ajuste global todavía.")

            x = np.asarray(self.fit_results["global"], dtype=float)
            info = self.global_fit_info

            J = info.jac
            P = len(x)
            N = len(info.fun)

            # χ² reducido
            dof = max(1, N - P)
            cost_factor = 2 * info.cost / dof

            # Covarianza
            try:
                JTJ_inv = np.linalg.inv(J.T @ J)
                cov = JTJ_inv * cost_factor
                sigma = np.sqrt(np.diag(cov))

                # ------------------------------------------------------------
                # Construir matriz de correlación
                # ------------------------------------------------------------
                denom = np.outer(sigma, sigma)
                corr = cov / denom

            except Exception:
                cov = None
                corr = None
                sigma = np.full_like(x, np.nan)

        else:
            # _______________________________________________
            # Ajuste no global (no hay covarianzas)
            # _______________________________________________
            if band not in self.fit_results:
                raise ValueError(f"No se encontró ajuste para banda '{band}'.")

            x = np.asarray(self.fit_results[band], dtype=float)
            sigma = np.full_like(x, np.nan)
            corr = None   # no disponible

        # ============================================================
        # 2. Nombres de parámetros
        # ============================================================
        if names is None:
            names = [f"param_{i}" for i in range(len(x))]
        if len(names) != len(x):
            raise ValueError("La longitud de 'names' no coincide con el nº de parámetros.")

        # ============================================================
        # 3. Exportar parámetros + incertidumbres
        # ============================================================
        df = pd.DataFrame([x, sigma], index=["value", "uncertainty"], columns=names)
        df.to_csv(archive)

        print(f"[OK] Parámetros exportados a '{archive}'.")

        # ============================================================
        # 4. Exportar matriz de correlaciones (solo global)
        # ============================================================
        if band == "global":
            if corr is None:
                print("[WARN] No se pudo calcular la matriz de correlaciones.")
                return

            df_corr = pd.DataFrame(corr, columns=names, index=names)
            archive_corr = archive.replace(".csv", "") + "_corrMatrix.csv"
            df_corr.to_csv(archive_corr)

            print(f"[OK] Matriz de correlaciones exportada a '{archive_corr}'.")
 

        import numpy as np

        # ------------------------------------------------------------
        # Helper: formateo científico en LaTeX: a.bcd × 10^{n}
        # ------------------------------------------------------------
    def exportParamsToTeX(
        self,
        archive: str = "params.tex",
        names=None,
        band: str = "global",
        caption: str = "Parámetros del ajuste",
        label: str = "tab:fit_params",
        precision: int = 3,
    ):
        """
        Exporta parámetros ajustados y sus incertidumbres a una tabla LaTeX.
        - Usa siunitx (\num{}) para los valores numéricos.
        - Columnas numéricas con S (siunitx).
        - Tres columnas numéricas: Valor, Error (%), Error (abs.).
        - NO modifica los 'names' (pueden llevar $...$ y _ tal cual).
        """

        import numpy as np

        # -----------------------------
        # Helper: formateo para \num{}
        # -----------------------------
        def _format_num(x, prec=precision):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return r"\text{--}"
            if x == 0:
                return r"\num{0}"
            # usamos notación científica estilo 1.234e-03, que siunitx entiende bien
            s = f"{x:.{prec}e}"
            return fr"\num{{{s}}}"

        # -----------------------------
        # obtener parámetros y sigmas
        # -----------------------------
        if band == "global":
            x = np.asarray(self.fit_results["global"], dtype=float)
            info = self.global_fit_info
            J = info.jac

            P = len(x)
            N = len(info.fun)
            dof = max(1, N - P)
            cost_factor = 2 * info.cost / dof

            try:
                JTJ_inv = np.linalg.inv(J.T @ J)
                cov = JTJ_inv * cost_factor
                sigma = np.sqrt(np.diag(cov))
            except Exception:
                sigma = np.full_like(x, np.nan)
        else:
            x = np.asarray(self.fit_results[band], dtype=float)
            sigma = np.full_like(x, np.nan)

        if names is None:
            names = [f"$x_{i}$" for i in range(len(x))]

        # -----------------------------
        # error %
        # -----------------------------
        error_percent = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            if x[i] == 0:
                error_percent[i] = np.nan
            else:
                error_percent[i] = abs(sigma[i] / abs(x[i])) * 100.0

        # -----------------------------
        # construir tabla LaTeX
        # -----------------------------
        lines = []
        lines.append(r"\begin{table}[h!]")
        lines.append(r"    \centering")
        # l para el nombre del parámetro, SSS para las tres columnas numéricas
        lines.append(r"    \begin{tabular}{lSSS}")
        lines.append(r"        \hline")
        lines.append(r"        Parámetro & {Valor} & {Error (\%)} & {Error (abs.)} \\")
        lines.append(r"        \hline")

        for name, val, sig, errp in zip(names, x, sigma, error_percent):
            val_tex  = _format_num(val)
            sig_tex  = _format_num(sig)
            errp_tex = _format_num(errp)

            # name se escribe TAL CUAL (puede llevar $...$ y _)
            lines.append(
                f"        {name} & {val_tex} & {errp_tex} & {sig_tex} \\\\"
            )

        lines.append(r"        \hline")
        lines.append(r"    \end{tabular}")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")
        lines.append(r"\end{table}")
        lines.append("")

        # -----------------------------
        # guardar archivo
        # -----------------------------
        with open(archive, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[OK] Tabla LaTeX exportada a '{archive}'. Usa siunitx para los números.")

    ###################################b",#######################################
    ######################### GRAFICA ###########################
    ##########################################################################
    
    def _ensure_band_in_settings(self, band):
        if band not in self.plot_settings["normalization"]:
            self.plot_settings["normalization"][band] = ("none", None)

        if band not in self.plot_settings["show_exp"]:
            self.plot_settings["show_exp"][band] = []

        if band not in self.plot_settings["show_teo"]:
            self.plot_settings["show_teo"][band] = []
            
    def choosePlotNormalization(self, band, mode="index", value=1.0, idx_ref=-1, global_bands=None):
        """
        Configura la normalización de un canal específico.

        Existen los siguientes modos de normalización:

        - **mode =

        # -----------------------------
        #  MODO 2: normalización GLOBAL
        # -----------------------------
        "none"**
            → No se aplica ningún tipo de normalización.

        - **mode = "N0"**
            → Se normaliza usando el valor de la curva teórica evaluada 
              a la presión indicada en `value`.
              (Por ejemplo: value = 1.0 → normaliza a 1 bar)

        - **mode = "index"**
            → Se normaliza usando el valor del array en la posición `idx_ref`.
              (Por defecto idx_ref = -1, que normalmente corresponde a la 
               concentración fCF4 = 100%)

        - **mode = "global"**
            → Normalización conjunta entre varias bandas. Debes pasar 
              `global_bands=[...]` con las bandas involucradas. 
              El valor de referencia se calcula sumando los valores de estas bandas 
              en el índice `idx_ref`.

        Parámetros
        ----------
        band : str
            Nombre del canal (banda) a normalizar.

        mode : {"none", "N0", "index", "global"}
            Tipo de normalización deseada.

        value : float
            Presión (en bar) usada únicamente cuando mode="N0".

        idx_ref : int
            Índice de referencia usado en mode="index" o mode="global".

        global_bands : list[str], opcional
            Lista de bandas para normalización conjunta cuando mode="global".
        """

        # --- Validación general correcta ----
        if band not in self.theory_functions:
            raise ValueError(f"No existe banda '{band}' en theory_functions")

        # --- Asegurar la estructura interna ---
        if not hasattr(self, "plot_settings"):
            self.plot_settings = {
                "normalization": {},
                "show_exp": {},
                "show_teo": {},
                "global_norm": {},     # NUEVO
            }

        # Guardar siempre algo
        if band not in self.plot_settings["normalization"]:
            self.plot_settings["normalization"][band] = ("none", None)

        # -----------------------------
        #  MODO 1: normalización normal
        # -----------------------------
        if mode == "none":
            self.plot_settings["normalization"][band] = ("none", None)

        elif mode == "N0":
            self.plot_settings["normalization"][band] = ("N0", value)

        elif mode == "index":
            self.plot_settings["normalization"][band] = ("index", idx_ref)
        
        elif mode == "handle_global":
            # No normaliza exp, divide teoría entre N_global
            self.plot_settings["normalization"][band] = ("handle_global", None)


        # -----------------------------
        #  MODO 2: normalización GLOBAL
        # -----------------------------
        elif mode == "global":
            if global_bands is None:
                raise ValueError("Debes proporcionar global_bands=[...] en modo global")

            # Guardamos SOLO UNA vez — solo se usa cuando se grafique
            self.plot_settings["global_norm"][band] = {
                "bands": global_bands,
                "mode": "index",   # siempre por índice (fácil y lógico)
                "idx": idx_ref
            }

            # Y marcamos al canal como normalización global
            self.plot_settings["normalization"][band] = ("global", None)

        else:
            raise ValueError("mode debe ser: 'none', 'N0', 'index', 'global'")  

        
    def enableExperimentalData(self, band, n):
        self._ensure_band_in_settings(band)
        if n not in self.plot_settings["show_exp"][band]:
            self.plot_settings["show_exp"][band].append(n)
            
    def enableTeoCurve(self, band, n):
        self._ensure_band_in_settings(band)
        if n not in self.plot_settings["show_teo"][band]:
            self.plot_settings["show_teo"][band].append(n)
            
    def plotTeoCurve(self, band, n=1.0, figsize=(7,5),cmap=plt.get_cmap("viridis"), savefig=None):
        # Asegurar que el canal existe
        if band not in self.theory_functions:
            raise ValueError(f"No existe función teórica para el canal '{band}'.")

        self._ensure_band_in_settings(band)

        f_th = self.theory_functions[band]
        fCF4 = self.fCF4

        # Normalización seleccionada
        norm_mode, norm_val = self.plot_settings["normalization"][band]

        # ---- FUNCIÓN PARA NORMALIZACIÓN GLOBAL  ----
        def compute_global_reference(idx, mode="teo"):
            """
            mode = 'exp' → usa datos experimentales
            mode = 'teo' → usa curvas teóricas
            """

            info = self.plot_settings["global_norm"][band]
            bands = info["bands"]
            idx_ref = info["idx"]

            ref_sum = 0.0

            for b in bands:

                if mode == "exp":   # NORMALIZA CONTRA EXPERIMENTAL
                    dfY = self.yields[b]
                    col = f"{n}bar"

                    if col not in dfY:
                        raise ValueError(
                            f"No hay datos experimentales para banda '{b}' a {n} bar"
                        )

                    ref_sum += dfY[col].to_numpy()[idx_ref]

                else:               # NORMALIZA CONTRA TEÓRICA
                    f_b = self.theory_functions[b]
                    x_b = self.fit_results[b]
                    y_b = f_b(x=x_b, fCF4=fCF4, n=n)
                    ref_sum += y_b[idx_ref]

            return ref_sum

        # ---- NORMALIZACIÓN TEÓRICA ----
        def normalize(arr,mode="teo"):
            if norm_mode == "none":
                return arr

            elif norm_mode == "index":
                return arr / arr[norm_val]

            elif norm_mode == "global":
                ref_sum = compute_global_reference(norm_val,mode=mode)
                return arr / ref_sum
            
            elif norm_mode == "handle_global":
                x = self.fit_results["global"]
                N_global = x[0]
                return arr


            return arr

        # ---- NORMALIZACIÓN EXP + ERRORES ----
        def normalize_pair(y, sy,mode="exp"):
            if norm_mode == "none":
                return y, sy

            elif norm_mode == "index":
                ref = y[norm_val]
                sref = sy[norm_val]
                sy_new = np.sqrt((sy/ref)**2 + (sref*y/ref**2)**2)
                return y/ref, sy_new

            elif norm_mode == "global":
                ref_sum = compute_global_reference(norm_val,mode=mode)
                return y / ref_sum, sy / ref_sum
            
            elif norm_mode == "handle_global":
                return y, sy   # sin normalización para exp

            return y, sy

        # ---------- PLOT ----------
        plt.figure(figsize=figsize)

        # Elegimos colores
        n_theo = len(self.plot_settings["show_teo"][band])
        n_exp  = len(self.plot_settings["show_exp"][band])
        maxima = max(n_theo, n_exp)
        colors = cmap(np.linspace(0.2, 0.8, maxima))

        # --- TEORÍA ---
        k = 0
        for n_plot in self.plot_settings["show_teo"][band]:
            fCF4_array = np.logspace(self.min_fCF4_10log, self.max_fCF4_10log, num=100)
            # Selección de parámetros de ajuste
            
            if norm_mode == "handle_global":
                if "global" not in self.fit_results:
                    raise ValueError("No se ha encontrado el ajuste global en self.fit_results['global']")
                x_fit = self.fit_results["global"]   # usa el vector global
            else:
                x_fit = self.fit_results[band]       # modo clásico (por banda)

            # Evaluación teórica
            y = f_th(x=x_fit, fCF4=fCF4_array, n=n_plot)

            
            y = normalize(y)
            plt.plot(
                fCF4_array*100,
                y,
                label=f"Theory {band}, {n_plot} bar",
                lw=2,
                color=darken(colors[k], factor=0.6)
            )
            
            k += 1

        # --- EXPERIMENTAL ---
        dfY = self.yields[band]
        k = 0
        for n_plot in self.plot_settings["show_exp"][band]:
            col = f"{n_plot}bar"
            scol = f"Err {n_plot}bar"

            if col not in dfY:
                print(f"[WARN] No hay datos experimentales para {col}")
                continue

            y = dfY[col].to_numpy()
            sy = dfY[scol].to_numpy()

            y, sy = normalize_pair(y, sy)

            fCF4_aux=fCF4[:]
            if len(fCF4)>len(y):
                n=len(fCF4)-len(y)
                fCF4_aux=fCF4[n:]
            
            
            plt.errorbar(
                fCF4_aux*100, y, yerr=sy,
                marker="o",
                linestyle="none",
                color=colors[k],
                label=f"Exp {band}, {n_plot} bar"
            )
            k += 1

        # --- Opciones comunes ---
        
        plt.xlim(fCF4_aux[0]*0.9*100,fCF4_aux[-1]*1.1*100)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("fCF4 %")
        plt.ylabel("yield normalizado")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if savefig:
            plt.savefig(savefig, dpi=300)
        else:
            plt.show()
