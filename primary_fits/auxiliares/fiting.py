import numpy as np
import scipy.optimize as opt
try:
    import scienceplots  # optional style package
except ModuleNotFoundError:
    scienceplots = None

def fitParameters(
    equations,
    experimental_data,
    degrad_data,
    x0,
    bounds,
    is_infrared=False,
    fit_on_experimental_concentrations=False,
    concentration_columns=None,
    first_point_anchor_weight=1.0,
    fixed_idx=None,
    fixed_values=None,
    fixed_error=np.nan,   # usa 0.0 si prefieres que los fijados salgan con error 0
    verbose=0,
):
    """
    Ajusta parámetros con posibilidad de fijar algunos índices.

    Parámetros
    ----------
    fit_on_experimental_concentrations : bool
        Si es True, evalúa cada modelo en la columna x de su dataset
        experimental. Evita asociar puntos por posición o recortar arrays.
    concentration_columns : dict[str, str] o None
        Mapa dataset -> columna de concentración, usado cuando
        fit_on_experimental_concentrations=True.
    fixed_idx : list[int] o None
        Índices de los parámetros que se quieren fijar.
    fixed_values : list[float] o None
        Valores a fijar en esos índices. Si es None, se usan los de x0.
    fixed_error : float
        Error asignado a los parámetros fijados en result.perr.
        Recomendado: np.nan. Si quieres que salgan como 0, usa 0.0.

    Devuelve
    --------
    result : OptimizeResult enriquecido con:
        - result.x         -> parámetros completos
        - result.x_free    -> parámetros libres
        - result.perr      -> errores completos
        - result.perr_free -> errores solo de libres
        - result.pcov      -> covarianza completa
        - result.pcov_free -> covarianza solo de libres
        - result.free_idx
        - result.fixed_idx
        - result.n_full
    """

    degrad_concentration = np.asarray(degrad_data["concentration"], dtype=float)
    concentration_columns = {} if concentration_columns is None else dict(concentration_columns)

    first_point_anchor_weight = float(first_point_anchor_weight)
    if not np.isfinite(first_point_anchor_weight) or first_point_anchor_weight < 1.0:
        raise ValueError("first_point_anchor_weight debe ser un número finito >= 1.0. Usa 1.0 para desactivarlo.")
    first_point_anchor_extra_scale = np.sqrt(first_point_anchor_weight - 1.0)

    x0 = np.asarray(x0, dtype=float).copy()
    lb = np.asarray(bounds[0], dtype=float).copy()
    ub = np.asarray(bounds[1], dtype=float).copy()

    if x0.ndim != 1:
        raise ValueError("x0 debe ser un array 1D.")
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("bounds debe tener la misma longitud que x0.")

    n_full = len(x0)

    # -----------------------------
    # Gestión de parámetros fijados
    # -----------------------------
    if fixed_idx is None:
        fixed_idx = []

    fixed_idx = np.array(sorted(set(fixed_idx)), dtype=int)

    if fixed_idx.size > 0:
        if np.any(fixed_idx < 0) or np.any(fixed_idx >= n_full):
            raise ValueError("Hay índices en fixed_idx fuera del rango de x0.")

    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if len(fixed_values) != len(fixed_idx):
            raise ValueError("fixed_values debe tener la misma longitud que fixed_idx.")
        x0[fixed_idx] = fixed_values

    free_mask = np.ones(n_full, dtype=bool)
    free_mask[fixed_idx] = False
    free_idx = np.where(free_mask)[0]

    # Comprobar que x0 está dentro de bounds
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("Algún valor de x0 está fuera de los bounds.")

    # Subespacio libre
    x0_free = x0[free_idx]
    lb_free = lb[free_idx]
    ub_free = ub[free_idx]

    def build_full_x(x_free):
        x_full = x0.copy()
        x_full[free_idx] = x_free
        return x_full

    def residuals_from_full_x(x, is_infrared=is_infrared):
        res_list = []

        for key, theory_yield in equations.items():
            exp_data = experimental_data[key]

            if fit_on_experimental_concentrations:
                x_col = concentration_columns.get(key)
                if x_col is None:
                    raise KeyError(
                        f"No se definió la columna de concentración para el dataset {key!r}."
                    )
                if x_col not in exp_data.columns:
                    raise KeyError(
                        f"El dataset {key!r} no contiene su columna de concentración {x_col!r}."
                    )
                # Experimental CSVs store additive concentration in percent,
                # while the kinetic models use fractions in [0, 1].
                concentration_eval = (
                    np.asarray(exp_data[x_col], dtype=float) * 0.01
                )
            else:
                concentration_eval = degrad_concentration

            cols_phys = [
                c for c in exp_data.columns
                if not str(c).startswith("Err")
            ]

            for col in cols_phys:
                y_exp = exp_data[col].to_numpy(dtype=float)

                err_col_candidates = [
                    f"Err {col}", f"Err_{col}", f"{col} Err", f"{col}_Err"
                ]

                s_exp = None
                for ec in err_col_candidates:
                    if ec in exp_data.columns:
                        s_exp = exp_data[ec].to_numpy(dtype=float)
                        break

                if s_exp is None:
                    s_exp = np.ones_like(y_exp)

                s_exp_eff = np.abs(np.asarray(s_exp, dtype=float).copy())
                mask0 = (~np.isfinite(s_exp_eff)) | (s_exp_eff <= 0)
                if np.any(mask0):
                    s_exp_eff[mask0] = 1e12

                try:
                    n_val = float(str(col).replace("bar", ""))
                except Exception:
                    continue

                y_th = theory_yield(x, degrad_data, concentration_eval, n_val)

                if fit_on_experimental_concentrations:
                    if np.shape(y_th) != np.shape(y_exp):
                        raise RuntimeError(
                            f"Dimensiones incompatibles en {key!r}, {col!r}: "
                            f"teoría={np.shape(y_th)}, datos={np.shape(y_exp)}. "
                            "No se recortan arrays cuando se ajusta en las "
                            "concentraciones experimentales."
                        )
                else:
                    # Legacy behaviour retained for fits that have not opted
                    # into exact experimental-concentration evaluation.
                    if (len(y_th) > len(y_exp)) and (not is_infrared):
                        n = len(y_th) - len(y_exp)
                        y_th = y_th[n:]
                    elif (len(y_th) > len(y_exp)) and is_infrared:
                        n = len(y_th) - len(y_exp)
                        y_th = y_th[:-n]
                    elif len(y_exp) > len(y_th):
                        n = len(y_exp) - len(y_th)
                        y_exp = y_exp[n:]
                        s_exp_eff = s_exp_eff[n:]

                y_th = np.asarray(y_th, dtype=float)
                y_exp = np.asarray(y_exp, dtype=float)
                s_exp_eff = np.asarray(s_exp_eff, dtype=float)

                valid = (
                    np.isfinite(y_exp)
                    & np.isfinite(y_th)
                    & np.isfinite(s_exp_eff)
                    & (s_exp_eff > 0)
                )
                if not np.any(valid):
                    continue

                y_exp = y_exp[valid]
                y_th = y_th[valid]
                s_exp_eff = s_exp_eff[valid]

                res = (y_exp - y_th) / s_exp_eff
                res_list.append(res)

                if is_infrared and first_point_anchor_extra_scale > 0.0 and res.size > 0:
                    # Restricción blanda: el primer punto válido de cada línea y
                    # presión IR cuenta más, sin modificar errores ni CSVs.
                    # Como el residuo normal ya está incluido una vez, añadimos
                    # solo el peso extra para que el peso total sea
                    # ``first_point_anchor_weight``.
                    res_list.append(np.asarray([first_point_anchor_extra_scale * res[0]], dtype=float))

        if not res_list:
            return np.array([], dtype=float)

        return np.concatenate(res_list)

    def residuals(x_free, is_infrared=is_infrared):
        x_full = build_full_x(x_free)
        return residuals_from_full_x(x_full, is_infrared=is_infrared)

    # ---------------------------------------------------------
    # Caso extremo: todos los parámetros fijados, no se ajusta
    # ---------------------------------------------------------
    if len(free_idx) == 0:
        fun = residuals_from_full_x(x0, is_infrared=is_infrared)
        cost = 0.5 * np.dot(fun, fun)

        result = opt.OptimizeResult()
        result.x = x0.copy()
        result.x_free = np.array([], dtype=float)
        result.fun = fun
        result.cost = cost
        result.success = True
        result.status = 0
        result.message = "No hay parámetros libres: todos los parámetros están fijados."
        result.nfev = 1
        result.njev = 0
        result.jac = np.empty((len(fun), 0), dtype=float)
        result.jac_free = result.jac.copy()
        result.jac_full = np.zeros((len(fun), n_full), dtype=float)

        result.pcov_free = np.empty((0, 0), dtype=float)
        result.perr_free = np.array([], dtype=float)

        result.pcov = np.full((n_full, n_full), np.nan, dtype=float)
        result.perr = np.full(n_full, fixed_error, dtype=float)

        result.free_idx = free_idx
        result.fixed_idx = fixed_idx
        result.n_full = n_full
        result.first_point_anchor_weight = first_point_anchor_weight

        return result

    # -----------------
    # Ajuste de verdad
    # -----------------
    
    result = opt.least_squares(
        residuals,
        x0_free,
        bounds=(lb_free, ub_free),
        method="trf",
        verbose=verbose
    )

    # =========================================================
    # Reconstrucción obligatoria al espacio completo
    # =========================================================
    
    x_free_opt = np.asarray(result.x, dtype=float).copy()
    x_full_opt = build_full_x(x_free_opt)

    J_free = np.asarray(result.jac, dtype=float)
    m, n_free = J_free.shape

    # =========================================================
    # Covarianza en el subespacio libre
    # =========================================================
    try:
        U, s, VT = np.linalg.svd(J_free, full_matrices=False)

        if s.size == 0:
            pcov_free = np.full((n_free, n_free), np.nan, dtype=float)
        else:
            threshold = np.finfo(float).eps * max(J_free.shape) * s[0]
            mask = s > threshold
            s = s[mask]
            VT = VT[mask]

            if s.size == 0:
                pcov_free = np.full((n_free, n_free), np.nan, dtype=float)
            else:
                JTJ_inv = (VT.T / (s ** 2)) @ VT

                dof_fit = m - n_free
                if dof_fit > 0:
                    s2 = 2.0 * result.cost / dof_fit
                    pcov_free = s2 * JTJ_inv
                else:
                    pcov_free = np.full((n_free, n_free), np.nan, dtype=float)

    except np.linalg.LinAlgError:
        pcov_free = np.full((n_free, n_free), np.nan, dtype=float)

    perr_free = np.sqrt(np.clip(np.diag(pcov_free), 0.0, None))

    # =========================================================
    # Expandir al espacio completo
    # =========================================================
    pcov_full = np.full((n_full, n_full), np.nan, dtype=float)
    pcov_full[np.ix_(free_idx, free_idx)] = pcov_free

    perr_full = np.full(n_full, fixed_error, dtype=float)
    perr_full[free_idx] = perr_free

    jac_full = np.zeros((m, n_full), dtype=float)
    jac_full[:, free_idx] = J_free

    # =========================================================
    # Estadísticos globales
    # =========================================================
    chi2 = 2.0 * result.cost
    N_res = result.fun.size
    N_free = n_free
    N_total = n_full
    dof = N_res - N_free
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # =========================================================
    # Guardar todo en result
    # =========================================================
    result.x_free = x_free_opt
    result.x = x_full_opt

    result.jac_free = J_free
    result.jac_full = jac_full

    result.pcov_free = pcov_free
    result.perr_free = perr_free

    result.pcov = pcov_full
    result.perr = perr_full

    result.free_idx = free_idx
    result.fixed_idx = fixed_idx
    result.n_free = N_free
    result.n_total = N_total

    result.chi2 = chi2
    result.dof = dof
    result.chi2_red = chi2_red
    result.first_point_anchor_weight = first_point_anchor_weight

    # =========================================================
    # Comprobaciones duras: que falle AQUÍ y no luego en export
    # =========================================================
    if len(result.x) != n_full:
        raise RuntimeError(
            f"fitParameters: result.x tiene longitud {len(result.x)} "
            f"pero debería tener {n_full}."
        )

    if result.pcov.shape != (n_full, n_full):
        raise RuntimeError(
            f"fitParameters: result.pcov tiene forma {result.pcov.shape} "
            f"pero debería ser ({n_full}, {n_full})."
        )

    if len(result.perr) != n_full:
        raise RuntimeError(
            f"fitParameters: result.perr tiene longitud {len(result.perr)} "
            f"pero debería tener {n_full}."
        )

    return result


import numpy as np
try:
    import lmfit
except ModuleNotFoundError:
    lmfit = None
from types import SimpleNamespace


def fitParameters_lmfit(
    equations,
    experimental_data,
    degrad_data,
    x0,
    bounds,
    is_infrared=False,
    fixed_idx=None,
    fixed_values=None,
    fixed_error=np.nan,   # usa 0.0 si prefieres que los fijados salgan con error 0
    method="least_squares",
    scale_covar=True,
    verbose=2,
    **fit_kws,
):
    """
    Versión en lmfit de fitParameters.

    Parámetros
    ----------
    equations : dict
        Diccionario de funciones teóricas por banda.
    experimental_data : dict
        Diccionario de DataFrames con datos experimentales.
    degrad_data : dict
        Diccionario con datos de degradación; debe contener "concentration".
    x0 : array-like
        Vector inicial completo de parámetros.
    bounds : tuple(lower, upper)
        Límites inferiores y superiores para cada parámetro.
    is_infrared : bool
        Si True, cuando y_th es más largo que y_exp, recorta por el final.
    fixed_idx : list[int] o None
        Índices de parámetros fijados.
    fixed_values : list[float] o None
        Valores de los parámetros fijados. Si es None, usa los de x0.
    fixed_error : float
        Error asignado a los parámetros fijados en result.perr.
    method : str
        Método de lmfit. Para emular tu scipy.least_squares usa "least_squares".
    scale_covar : bool
        Escalado automático de la covarianza de lmfit.
    verbose : int
        Se pasa a scipy least_squares cuando method="least_squares".
    **fit_kws :
        Argumentos extra para lmfit/scipy.
    """

    if lmfit is None:
        raise ModuleNotFoundError("lmfit is required for fitParameters_lmfit(). Use fitParameters() or install lmfit.")

    concentration = degrad_data["concentration"]

    x0 = np.asarray(x0, dtype=float).copy()
    lb = np.asarray(bounds[0], dtype=float).copy()
    ub = np.asarray(bounds[1], dtype=float).copy()

    if x0.ndim != 1:
        raise ValueError("x0 debe ser un array 1D.")
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("bounds debe tener la misma longitud que x0.")

    n_full = len(x0)

    # -----------------------------
    # Gestión de parámetros fijados
    # -----------------------------
    if fixed_idx is None:
        fixed_idx = []

    fixed_idx = np.array(sorted(set(fixed_idx)), dtype=int)

    if fixed_idx.size > 0:
        if np.any(fixed_idx < 0) or np.any(fixed_idx >= n_full):
            raise ValueError("Hay índices en fixed_idx fuera del rango de x0.")

    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if len(fixed_values) != len(fixed_idx):
            raise ValueError("fixed_values debe tener la misma longitud que fixed_idx.")
        x0[fixed_idx] = fixed_values

    free_mask = np.ones(n_full, dtype=bool)
    free_mask[fixed_idx] = False
    free_idx = np.where(free_mask)[0]

    # Comprobar que x0 está dentro de bounds
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("Algún valor de x0 está fuera de los bounds.")

    # -----------------------------
    # Construcción de Parameters
    # -----------------------------
    param_names = [f"p{i}" for i in range(n_full)]
    fixed_set = set(fixed_idx.tolist())

    params = lmfit.Parameters()
    for i, name in enumerate(param_names):
        params.add(
            name=name,
            value=float(x0[i]),
            min=float(lb[i]),
            max=float(ub[i]),
            vary=(i not in fixed_set),
        )

    name_to_fullidx = {name: i for i, name in enumerate(param_names)}

    def params_to_full_x(params_obj):
        return np.array([params_obj[name].value for name in param_names], dtype=float)

    def residuals_from_params(
        params_obj,
        equations,
        experimental_data,
        degrad_data,
        concentration,
        is_infrared=False,
    ):
        x = params_to_full_x(params_obj)
        res_list = []

        for key, theory_yield in equations.items():
            exp_data = experimental_data[key]

            cols_phys = [
                c for c in exp_data.columns
                if not str(c).startswith("Err")
            ]

            for col in cols_phys:
                y_exp = exp_data[col].to_numpy(dtype=float)

                err_col_candidates = [
                    f"Err {col}", f"Err_{col}", f"{col} Err", f"{col}_Err"
                ]

                s_exp = None
                for ec in err_col_candidates:
                    if ec in exp_data.columns:
                        s_exp = exp_data[ec].to_numpy(dtype=float)
                        break

                if s_exp is None:
                    s_exp = np.ones_like(y_exp)

                s_exp_eff = np.abs(np.asarray(s_exp, dtype=float).copy())
                mask0 = (~np.isfinite(s_exp_eff)) | (s_exp_eff <= 0)
                if np.any(mask0):
                    s_exp_eff[mask0] = 1e12

                try:
                    n_val = float(str(col).replace("bar", ""))
                except Exception:
                    continue

                y_th = np.asarray(
                    theory_yield(x, degrad_data, concentration, n_val),
                    dtype=float
                )

                if (len(y_th) > len(y_exp)) and (not is_infrared):
                    n = len(y_th) - len(y_exp)
                    y_th = y_th[n:]
                elif (len(y_th) > len(y_exp)) and is_infrared:
                    n = len(y_th) - len(y_exp)
                    y_th = y_th[:-n]
                elif len(y_exp) > len(y_th):
                    n = len(y_exp) - len(y_th)
                    y_exp = y_exp[n:]
                    s_exp_eff = s_exp_eff[n:]

                y_th = np.asarray(y_th, dtype=float)
                y_exp = np.asarray(y_exp, dtype=float)
                s_exp_eff = np.asarray(s_exp_eff, dtype=float)

                valid = (
                    np.isfinite(y_exp)
                    & np.isfinite(y_th)
                    & np.isfinite(s_exp_eff)
                    & (s_exp_eff > 0)
                )
                if not np.any(valid):
                    continue

                y_exp = y_exp[valid]
                y_th = y_th[valid]
                s_exp_eff = s_exp_eff[valid]

                res = (y_exp - y_th) / s_exp_eff
                res_list.append(res)

        if not res_list:
            return np.array([], dtype=float)

        return np.concatenate(res_list)

    # ---------------------------------------------------------
    # Caso extremo: todos los parámetros fijados, no se ajusta
    # ---------------------------------------------------------
    if len(free_idx) == 0:
        fun = residuals_from_params(
            params, equations, experimental_data, degrad_data,
            concentration, is_infrared=is_infrared
        )
        m = fun.size

        result = SimpleNamespace()
        result.params = params
        result.var_names = []
        result.covar = np.empty((0, 0), dtype=float)

        result.residual = fun
        result.fun = fun
        result.chisqr = float(np.dot(fun, fun))
        result.redchi = np.nan
        result.ndata = m
        result.nvarys = 0
        result.nfree = m

        result.cost = 0.5 * result.chisqr
        result.success = True
        result.status = 0
        result.message = "No hay parámetros libres: todos los parámetros están fijados."
        result.method = method
        result.nfev = 1

        result.x = params_to_full_x(params)
        result.x_free = np.array([], dtype=float)

        result.jac = np.empty((m, 0), dtype=float)
        result.jac_free = result.jac.copy()
        result.jac_full = np.zeros((m, n_full), dtype=float)

        result.pcov_free = np.empty((0, 0), dtype=float)
        result.perr_free = np.array([], dtype=float)

        result.pcov = np.full((n_full, n_full), np.nan, dtype=float)
        result.perr = np.full(n_full, fixed_error, dtype=float)

        result.free_idx = free_idx
        result.fixed_idx = fixed_idx
        result.n_full = n_full
        result.n_total = n_full

        result.chi2 = result.chisqr
        result.dof = result.nfree
        result.chi2_red = result.redchi

        return result

    # -----------------
    # Ajuste de verdad
    # -----------------
    fit_kws = dict(fit_kws)
    if method == "least_squares":
        fit_kws.setdefault("verbose", verbose)

    minner = lmfit.Minimizer(
        userfcn=residuals_from_params,
        params=params,
        fcn_args=(equations, experimental_data, degrad_data, concentration),
        fcn_kws={"is_infrared": is_infrared},
        scale_covar=scale_covar,
        nan_policy="raise",
    )

    result = minner.minimize(method=method, **fit_kws)

    # =========================================================
    # Reconstrucción al espacio completo
    # =========================================================
    x_full_opt = params_to_full_x(result.params)
    x_free_opt = x_full_opt[free_idx]

    # =========================================================
    # Covarianza en el subespacio libre
    # =========================================================
    n_free = len(result.var_names)

    if result.covar is None:
        pcov_free = np.full((n_free, n_free), np.nan, dtype=float)
    else:
        pcov_free = np.asarray(result.covar, dtype=float)

    perr_free = np.full(n_free, np.nan, dtype=float)
    for j, name in enumerate(result.var_names):
        stderr = result.params[name].stderr
        if stderr is not None:
            perr_free[j] = float(stderr)
        elif result.covar is not None and np.isfinite(result.covar[j, j]):
            perr_free[j] = np.sqrt(max(result.covar[j, j], 0.0))

    # =========================================================
    # Expandir al espacio completo
    # =========================================================
    pcov_full = np.full((n_full, n_full), np.nan, dtype=float)
    if result.covar is not None:
        for irow, name_i in enumerate(result.var_names):
            ii = name_to_fullidx[name_i]
            for jcol, name_j in enumerate(result.var_names):
                jj = name_to_fullidx[name_j]
                pcov_full[ii, jj] = result.covar[irow, jcol]

    perr_full = np.full(n_full, fixed_error, dtype=float)
    for j, name in enumerate(result.var_names):
        perr_full[name_to_fullidx[name]] = perr_free[j]

    jac_free = getattr(result, "jac", None)
    if jac_free is not None:
        jac_free = np.asarray(jac_free, dtype=float)
        jac_full = np.zeros((jac_free.shape[0], n_full), dtype=float)
        for j, name in enumerate(result.var_names):
            jac_full[:, name_to_fullidx[name]] = jac_free[:, j]
    else:
        jac_full = None

    # =========================================================
    # Añadir aliases para no romper tu flujo actual
    # =========================================================
    result.fun = np.asarray(result.residual, dtype=float)
    result.cost = 0.5 * float(result.chisqr)

    result.x = x_full_opt
    result.x_free = x_free_opt

    result.jac_free = jac_free
    result.jac_full = jac_full

    result.pcov_free = pcov_free
    result.perr_free = perr_free

    result.pcov = pcov_full
    result.perr = perr_full

    result.free_idx = free_idx
    result.fixed_idx = fixed_idx
    result.n_free = n_free
    result.n_total = n_full
    result.n_full = n_full

    result.chi2 = float(result.chisqr)
    result.dof = int(result.nfree)
    result.chi2_red = float(result.redchi)

    # =========================================================
    # Comprobaciones duras
    # =========================================================
    if len(result.x) != n_full:
        raise RuntimeError(
            f"fitParameters_lmfit: result.x tiene longitud {len(result.x)} "
            f"pero debería tener {n_full}."
        )

    if result.pcov.shape != (n_full, n_full):
        raise RuntimeError(
            f"fitParameters_lmfit: result.pcov tiene forma {result.pcov.shape} "
            f"pero debería ser ({n_full}, {n_full})."
        )

    if len(result.perr) != n_full:
        raise RuntimeError(
            f"fitParameters_lmfit: result.perr tiene longitud {len(result.perr)} "
            f"pero debería tener {n_full}."
        )

    return result


import numpy as np
import scipy.optimize as opt
try:
    import scienceplots  # optional style package
except ModuleNotFoundError:
    scienceplots = None


def fitParameters_minimize(
    equations,
    experimental_data,
    degrad_data,
    x0,
    bounds,
    is_infrared=False,
    fixed_idx=None,
    fixed_values=None,
    fixed_error=np.nan,   # usa 0.0 si prefieres que los fijados salgan con error 0
    method="L-BFGS-B",
    options=None,
):
    """
    Ajusta parámetros con scipy.optimize.minimize, permitiendo fijar índices.

    Devuelve un OptimizeResult enriquecido con:
        - result.x         -> parámetros completos
        - result.x_free    -> parámetros libres
        - result.fun       -> vector de residuos (compatibilidad con tu flujo)
        - result.fun_scalar-> valor escalar minimizado = 0.5*sum(res^2)
        - result.cost      -> 0.5*sum(res^2)
        - result.chi2      -> sum(res^2)
        - result.chi2_red  -> chi2 reducido
        - result.pcov      -> covarianza completa (aprox)
        - result.perr      -> errores completos (aprox)
        - result.pcov_free -> covarianza subespacio libre
        - result.perr_free -> errores subespacio libre
        - result.free_idx
        - result.fixed_idx
        - result.n_full
    """

    concentration = degrad_data["concentration"]

    x0 = np.asarray(x0, dtype=float).copy()
    lb = np.asarray(bounds[0], dtype=float).copy()
    ub = np.asarray(bounds[1], dtype=float).copy()

    if x0.ndim != 1:
        raise ValueError("x0 debe ser un array 1D.")
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("bounds debe tener la misma longitud que x0.")

    n_full = len(x0)

    # -----------------------------
    # Gestión de parámetros fijados
    # -----------------------------
    if fixed_idx is None:
        fixed_idx = []

    fixed_idx = np.array(sorted(set(fixed_idx)), dtype=int)

    if fixed_idx.size > 0:
        if np.any(fixed_idx < 0) or np.any(fixed_idx >= n_full):
            raise ValueError("Hay índices en fixed_idx fuera del rango de x0.")

    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if len(fixed_values) != len(fixed_idx):
            raise ValueError("fixed_values debe tener la misma longitud que fixed_idx.")
        x0[fixed_idx] = fixed_values

    free_mask = np.ones(n_full, dtype=bool)
    free_mask[fixed_idx] = False
    free_idx = np.where(free_mask)[0]

    # Comprobar que x0 está dentro de bounds
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("Algún valor de x0 está fuera de los bounds.")

    # Subespacio libre
    x0_free = x0[free_idx]
    lb_free = lb[free_idx]
    ub_free = ub[free_idx]

    # bounds para minimize
    bounds_free = list(zip(lb_free, ub_free))

    def build_full_x(x_free):
        x_full = x0.copy()
        x_full[free_idx] = x_free
        return x_full

    def residuals_from_full_x(x, is_infrared=is_infrared):
        res_list = []

        for key, theory_yield in equations.items():
            exp_data = experimental_data[key]

            cols_phys = [
                c for c in exp_data.columns
                if not str(c).startswith("Err")
            ]

            for col in cols_phys:
                y_exp = exp_data[col].to_numpy(dtype=float)

                err_col_candidates = [
                    f"Err {col}", f"Err_{col}", f"{col} Err", f"{col}_Err"
                ]

                s_exp = None
                for ec in err_col_candidates:
                    if ec in exp_data.columns:
                        s_exp = exp_data[ec].to_numpy(dtype=float)
                        break

                if s_exp is None:
                    s_exp = np.ones_like(y_exp)

                s_exp_eff = np.abs(np.asarray(s_exp, dtype=float).copy())
                mask0 = (~np.isfinite(s_exp_eff)) | (s_exp_eff <= 0)
                if np.any(mask0):
                    s_exp_eff[mask0] = 1e12

                try:
                    n_val = float(str(col).replace("bar", ""))
                except Exception:
                    continue

                y_th = np.asarray(
                    theory_yield(x, degrad_data, concentration, n_val),
                    dtype=float
                )

                if (len(y_th) > len(y_exp)) and (not is_infrared):
                    n = len(y_th) - len(y_exp)
                    y_th = y_th[n:]
                elif (len(y_th) > len(y_exp)) and is_infrared:
                    n = len(y_th) - len(y_exp)
                    y_th = y_th[:-n]
                elif len(y_exp) > len(y_th):
                    n = len(y_exp) - len(y_th)
                    y_exp = y_exp[n:]
                    s_exp_eff = s_exp_eff[n:]

                res = (y_exp - y_th) / s_exp_eff
                res_list.append(res)

        if not res_list:
            return np.array([], dtype=float)

        return np.concatenate(res_list)

    def residuals(x_free, is_infrared=is_infrared):
        x_full = build_full_x(x_free)
        return residuals_from_full_x(x_full, is_infrared=is_infrared)

    def objective(x_free, is_infrared=is_infrared):
        r = residuals(x_free, is_infrared=is_infrared)
        return 0.5 * np.dot(r, r)

    # ---------------------------------------------------------
    # Caso extremo: todos los parámetros fijados, no se ajusta
    # ---------------------------------------------------------
    if len(free_idx) == 0:
        fun = residuals_from_full_x(x0, is_infrared=is_infrared)
        cost = 0.5 * np.dot(fun, fun)

        result = opt.OptimizeResult()
        result.x = x0.copy()
        result.x_free = np.array([], dtype=float)

        result.fun_scalar = cost
        result.fun = fun
        result.cost = cost

        result.success = True
        result.status = 0
        result.message = "No hay parámetros libres: todos los parámetros están fijados."
        result.nfev = 1
        result.njev = 0

        result.jac = np.empty((len(fun), 0), dtype=float)
        result.jac_free = result.jac.copy()
        result.jac_full = np.zeros((len(fun), n_full), dtype=float)

        result.pcov_free = np.empty((0, 0), dtype=float)
        result.perr_free = np.array([], dtype=float)

        result.pcov = np.full((n_full, n_full), np.nan, dtype=float)
        result.perr = np.full(n_full, fixed_error, dtype=float)

        result.free_idx = free_idx
        result.fixed_idx = fixed_idx
        result.n_full = n_full
        result.n_total = n_full
        result.n_free = 0

        result.chi2 = 2.0 * cost
        result.dof = len(fun)
        result.chi2_red = result.chi2 / result.dof if result.dof > 0 else np.nan

        return result

    # -----------------
    # Ajuste de verdad
    # -----------------
    if options is None:
        options = {"disp": True, "maxiter": 1000}

    result = opt.minimize(
        objective,
        x0_free,
        args=(is_infrared,),
        method=method,
        bounds=bounds_free,
        options=options,
    )

    # =========================================================
    # Reconstrucción al espacio completo
    # =========================================================
    x_free_opt = np.asarray(result.x, dtype=float).copy()
    x_full_opt = build_full_x(x_free_opt)

    res_opt = residuals(x_free_opt, is_infrared=is_infrared)
    m = res_opt.size
    n_free = len(x_free_opt)

    # =========================================================
    # Covarianza aproximada en el subespacio libre
    # =========================================================
    pcov_free = np.full((n_free, n_free), np.nan, dtype=float)

    # Para L-BFGS-B, hess_inv suele venir como operador lineal / objeto especial.
    if hasattr(result, "hess_inv") and result.hess_inv is not None:
        try:
            if hasattr(result.hess_inv, "todense"):
                hess_inv_dense = np.asarray(result.hess_inv.todense(), dtype=float)
            else:
                hess_inv_dense = np.asarray(result.hess_inv, dtype=float)

            if hess_inv_dense.shape == (n_free, n_free):
                dof_fit = m - n_free
                if dof_fit > 0:
                    # Aproximación análoga a tu escalado en least_squares
                    s2 = 2.0 * result.fun / dof_fit
                    pcov_free = hess_inv_dense * s2
                else:
                    pcov_free = hess_inv_dense.copy()
        except Exception:
            pcov_free = np.full((n_free, n_free), np.nan, dtype=float)

    perr_free = np.sqrt(np.clip(np.diag(pcov_free), 0.0, None))

    # =========================================================
    # Expandir al espacio completo
    # =========================================================
    pcov_full = np.full((n_full, n_full), np.nan, dtype=float)
    pcov_full[np.ix_(free_idx, free_idx)] = pcov_free

    perr_full = np.full(n_full, fixed_error, dtype=float)
    perr_full[free_idx] = perr_free

    # Con minimize no tenemos Jacobiano de residuos como en least_squares
    jac_full = None
    jac_free = None

    # =========================================================
    # Estadísticos globales
    # =========================================================
    chi2 = 2.0 * result.fun
    N_res = m
    N_free = n_free
    N_total = n_full
    dof = N_res - N_free
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # =========================================================
    # Guardar todo en result
    # =========================================================
    result.fun_scalar = float(result.fun)   # guardamos el escalar original
    result.fun = res_opt                    # compatibilidad con tu flujo anterior
    result.cost = 0.5 * chi2

    result.x_free = x_free_opt
    result.x = x_full_opt

    result.jac_free = jac_free
    result.jac_full = jac_full

    result.pcov_free = pcov_free
    result.perr_free = perr_free

    result.pcov = pcov_full
    result.perr = perr_full

    result.free_idx = free_idx
    result.fixed_idx = fixed_idx
    result.n_free = N_free
    result.n_total = N_total
    result.n_full = n_full

    result.chi2 = chi2
    result.dof = dof
    result.chi2_red = chi2_red

    # =========================================================
    # Comprobaciones duras
    # =========================================================
    if len(result.x) != n_full:
        raise RuntimeError(
            f"fitParameters_minimize: result.x tiene longitud {len(result.x)} "
            f"pero debería tener {n_full}."
        )

    if result.pcov.shape != (n_full, n_full):
        raise RuntimeError(
            f"fitParameters_minimize: result.pcov tiene forma {result.pcov.shape} "
            f"pero debería ser ({n_full}, {n_full})."
        )

    if len(result.perr) != n_full:
        raise RuntimeError(
            f"fitParameters_minimize: result.perr tiene longitud {len(result.perr)} "
            f"pero debería tener {n_full}."
        )

    return result
