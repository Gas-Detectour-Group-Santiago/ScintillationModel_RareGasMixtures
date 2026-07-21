import numpy as np
import pandas as pd
from pathlib import Path


def _coerce_result_like(result, sx0=None):
    """
    Convierte distintas entradas a un formato homogéneo compatible
    con el exportador.

    Acepta:
      - OptimizeResult / objetos con atributos .x, .perr, .pcov, .jac
      - dict con claves 'x', 'perr', 'pcov', ...
      - np.ndarray / list / tuple interpretados como un x0

    Para x0/list/ndarray:
      - x    = vector de parámetros
      - perr = sx0 si se proporciona; si no, NaN
      - pcov = diag(perr^2) si sx0 se proporciona; si no, NaN

    Si result es un ajuste real (OptimizeResult o dict con info propia),
    sx0 se ignora.
    """
    # Caso 1: vector puro -> lo interpretamos como x0
    if isinstance(result, (list, tuple, np.ndarray)):
        x = np.asarray(result, dtype=float)
        n = len(x)

        if sx0 is None:
            perr = np.full(n, np.nan, dtype=float)
            pcov = np.full((n, n), np.nan, dtype=float)
        else:
            if np.isscalar(sx0):
                perr = np.full(n, float(sx0), dtype=float)
            else:
                perr = np.asarray(sx0, dtype=float)
                if perr.shape != (n,):
                    raise ValueError(
                        f"sx0 tiene forma {perr.shape}, pero debería ser ({n},)"
                    )
            perr = np.abs(perr)
            pcov = np.diag(perr**2)

        return {
            "x": x,
            "perr": perr,
            "pcov": pcov,
        }

    # Caso 2: dict con clave x
    if isinstance(result, dict):
        if "x" not in result:
            raise ValueError("Si 'result' es un dict, debe contener la clave 'x'.")

        x = np.asarray(result["x"], dtype=float)
        n = len(x)

        out = dict(result)

        if "perr" in out and out["perr"] is not None:
            out["perr"] = np.asarray(out["perr"], dtype=float)
        else:
            out["perr"] = np.full(n, np.nan, dtype=float)

        if "pcov" in out and out["pcov"] is not None:
            out["pcov"] = np.asarray(out["pcov"], dtype=float)
        else:
            out["pcov"] = np.full((n, n), np.nan, dtype=float)

        out["x"] = x
        return out

    # Caso 3: objeto tipo OptimizeResult -> lo dejamos tal cual
    return result


def _extract_fit_info(result, sx0=None):
    """
    Extrae de forma robusta:
      - popt : parámetros completos
      - perr : errores completos
      - pcov : covarianza completa

    Acepta:
      - OptimizeResult / objetos con atributos .x, .perr, .pcov, .jac
      - dict con claves equivalentes
      - list / tuple / np.ndarray interpretados como x0

    Si result es un x0 y sx0 no es None, usa sx0 como incertidumbre.
    """
    result = _coerce_result_like(result, sx0=sx0)

    def _has(obj, key):
        return hasattr(obj, key) or (isinstance(obj, dict) and key in obj)

    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    if not _has(result, "x"):
        raise ValueError("El objeto 'result' no tiene atributo/claves 'x'.")

    popt = np.asarray(_get(result, "x"), dtype=float)
    n_total = len(popt)

    has_full_pcov = _has(result, "pcov")
    has_full_perr = _has(result, "perr")

    if has_full_pcov or has_full_perr:
        pcov = None
        perr = None

        if has_full_pcov:
            pcov = np.asarray(_get(result, "pcov"), dtype=float)
            if pcov.shape != (n_total, n_total):
                raise ValueError(
                    f"result.pcov tiene forma {pcov.shape}, pero debería ser "
                    f"({n_total}, {n_total})."
                )

        if has_full_perr:
            perr = np.asarray(_get(result, "perr"), dtype=float)
            if perr.shape != (n_total,):
                raise ValueError(
                    f"result.perr tiene longitud {len(perr)}, pero debería ser {n_total}."
                )

        if (perr is None) and (pcov is not None):
            perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

        if (pcov is None) and (perr is not None):
            pcov = np.full((n_total, n_total), np.nan, dtype=float)
            np.fill_diagonal(pcov, perr**2)

        return popt, perr, pcov

    if not _has(result, "jac"):
        perr = np.full(n_total, np.nan, dtype=float)
        pcov = np.full((n_total, n_total), np.nan, dtype=float)
        return popt, perr, pcov

    J = np.asarray(_get(result, "jac"), dtype=float)
    m, n = J.shape

    _, svals, VT = np.linalg.svd(J, full_matrices=False)

    if svals.size == 0:
        pcov = np.full((n, n), np.nan, dtype=float)
    else:
        threshold = np.finfo(float).eps * max(J.shape) * svals[0]
        mask = svals > threshold

        if not np.any(mask):
            pcov = np.full((n, n), np.nan, dtype=float)
        else:
            VT = VT[mask]
            svals = svals[mask]
            pcov = (VT.T / (svals**2)) @ VT

            dof = m - n
            if dof > 0 and _has(result, "cost"):
                s_sq = 2.0 * _get(result, "cost") / dof
                pcov = pcov * s_sq
            else:
                pcov[:] = np.nan

    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

    if len(perr) != n_total:
        raise ValueError(
            f"Incompatibilidad interna: len(result.x)={n_total} pero "
            f"la covarianza inferida desde jac da {len(perr)} parámetros."
        )

    return popt, perr, pcov


# =========================================================
# Utilidades numéricas
# =========================================================

def _round_sig(x, sig=2):
    if not np.isfinite(x) or x == 0:
        return x
    return round(float(x), sig - 1 - int(np.floor(np.log10(abs(x)))))


def _needs_scientific(x, sci_low=-2, sci_high=3):
    """
    Usa notación científica si |x| < 1e-2 o |x| >= 1e3.
    """
    if not np.isfinite(x) or x == 0:
        return False
    exp = int(np.floor(np.log10(abs(x))))
    return (exp <= sci_low) or (exp >= sci_high)


def _format_number_string(x, sigfigs=2):
    """
    Devuelve un string parseable por siunitx.
    """
    if x is None or not np.isfinite(x):
        return None

    x = float(x)

    if x == 0:
        return "0"

    xr = _round_sig(x, sigfigs)

    if _needs_scientific(xr):
        exp = int(np.floor(np.log10(abs(xr))))
        mant = xr / (10 ** exp)
        mant = _round_sig(mant, sigfigs)

        if mant == 0:
            decimals = 0
        else:
            exp_m = int(np.floor(np.log10(abs(mant))))
            decimals = max(sigfigs - 1 - exp_m, 0)

        return f"{mant:.{decimals}f}e{exp}"

    exp_x = int(np.floor(np.log10(abs(xr))))
    decimals = max(sigfigs - 1 - exp_x, 0)
    return f"{xr:.{decimals}f}"


def _format_uncertainty_siunitx_inner(value, uncertainty, err_sigfigs=2):
    """
    Devuelve el contenido interno para \\num{...} o \\SI{...}{...}
    usando formato de incertidumbre de siunitx.

    Ejemplo:
      340000 ± 110000 -> 3.4(11)e5
    """
    if value is None or not np.isfinite(value):
        return None

    value = float(value)

    if uncertainty is None or not np.isfinite(uncertainty):
        return _format_number_string(value, sigfigs=err_sigfigs)

    uncertainty = abs(float(uncertainty))

    if uncertainty == 0:
        return _format_number_string(value, sigfigs=err_sigfigs)

    ref = max(abs(value), uncertainty)
    use_scientific = _needs_scientific(ref)

    exp = int(np.floor(np.log10(ref))) if ref != 0 else 0
    scale = 10 ** exp if use_scientific else 1.0

    v_scaled = value / scale
    u_scaled = uncertainty / scale

    u_rounded = _round_sig(u_scaled, err_sigfigs)

    if not np.isfinite(u_rounded) or u_rounded == 0:
        return _format_number_string(value, sigfigs=err_sigfigs)

    exp_u = int(np.floor(np.log10(abs(u_rounded))))
    decimals = max(err_sigfigs - 1 - exp_u, 0)

    v_rounded = round(v_scaled, decimals)
    u_rounded = round(u_rounded, decimals)

    if v_rounded == 0:
        v_rounded = 0.0

    v_str = f"{v_rounded:.{decimals}f}"

    unc_digits = int(round(u_rounded * (10 ** decimals)))
    unc_str = str(abs(unc_digits))

    if use_scientific:
        return f"{v_str}({unc_str})e{exp}"
    return f"{v_str}({unc_str})"


def _format_num_latex(x, sigfigs=2):
    s = _format_number_string(x, sigfigs=sigfigs)
    if s is None:
        return r"--"
    return rf"\num{{{s}}}"


def _format_value_uncertainty_latex_siunitx(value, uncertainty, unit=None, err_sigfigs=2):
    """
    Devuelve:
      - \\num{valor(incertidumbre)}
      - o \\SI{valor(incertidumbre)}{unidad}
    """
    s = _format_uncertainty_siunitx_inner(
        value=value,
        uncertainty=uncertainty,
        err_sigfigs=err_sigfigs
    )

    if s is None:
        return r"--"

    if unit is None or str(unit).strip() == "":
        return rf"\num{{{s}}}"

    return rf"\SI{{{s}}}{{{unit}}}"


def _format_scalar_text(x, sigfigs=2):
    s = _format_number_string(x, sigfigs=sigfigs)
    return "--" if s is None else s


def _format_value_uncertainty_text(value, uncertainty, err_sigfigs=2):
    if value is None or not np.isfinite(value):
        return "--"

    if uncertainty is None or not np.isfinite(uncertainty):
        return _format_scalar_text(value, sigfigs=err_sigfigs)

    value = float(value)
    uncertainty = abs(float(uncertainty))

    if uncertainty == 0:
        return _format_scalar_text(value, sigfigs=err_sigfigs)

    ref = max(abs(value), uncertainty)
    use_scientific = _needs_scientific(ref)

    exp = int(np.floor(np.log10(ref))) if ref != 0 else 0
    scale = 10 ** exp if use_scientific else 1.0

    v_scaled = value / scale
    u_scaled = uncertainty / scale

    u_rounded = _round_sig(u_scaled, err_sigfigs)
    exp_u = int(np.floor(np.log10(abs(u_rounded)))) if u_rounded != 0 else 0
    decimals = max(err_sigfigs - 1 - exp_u, 0)

    v_rounded = round(v_scaled, decimals)
    u_rounded = round(u_scaled, decimals)

    if v_rounded == 0:
        v_rounded = 0.0

    v_str = f"{v_rounded:.{decimals}f}"
    u_str = f"{u_rounded:.{decimals}f}"

    if use_scientific:
        return f"({v_str} ± {u_str})e{exp}"
    return f"{v_str} ± {u_str}"


# =========================================================
# Normalización de entrada múltiple
# =========================================================

def _normalize_results_input(results):
    """
    Acepta:
      - un único result
      - una lista/tupla de results
      - un np.ndarray/list/tuple como x0

    Devuelve siempre una lista de entradas.
    """
    if isinstance(results, np.ndarray):
        if results.ndim == 1:
            return [results]
        raise ValueError("'results' no puede ser un ndarray de dimensión > 1.")

    if isinstance(results, (list, tuple)):
        if len(results) == 0:
            raise ValueError("'results' no puede estar vacío.")

        is_scalar_vector = all(np.isscalar(x) for x in results)
        if is_scalar_vector:
            return [np.asarray(results, dtype=float)]

        return list(results)

    return [results]


def _normalize_relative_incertainty(relative_incertainty, n_results):
    """
    Normaliza el parámetro relative_incertainty.

    Acepta:
      - None -> no modificar incertidumbres
      - float >= 0 -> mismo error relativo para todos los resultados
      - lista/tupla/ndarray de longitud n_results -> uno por resultado

    Convención:
      sigma_i = relative_incertainty * abs(x_i)
    """
    if relative_incertainty is None:
        return [None] * n_results

    if np.isscalar(relative_incertainty):
        rel = float(relative_incertainty)
        if rel < 0:
            raise ValueError("'relative_incertainty' debe ser >= 0.")
        return [rel] * n_results

    rels = list(relative_incertainty)
    if len(rels) != n_results:
        raise ValueError(
            f"len(relative_incertainty) = {len(rels)} pero hay {n_results} resultados."
        )

    rels = [float(r) if r is not None else None for r in rels]

    for r in rels:
        if r is not None and r < 0:
            raise ValueError("Todos los valores de 'relative_incertainty' deben ser >= 0.")

    return rels


def _normalize_sx0(sx0, n_results):
    """
    Normaliza sx0 para que haya una entrada por resultado.

    Acepta:
      - None -> [None, ..., None]
      - lista/tupla de longitud n_results
      - para n_results == 1, también acepta un vector/scalar directo

    Cada entrada sx0[i] puede ser:
      - None
      - escalar
      - vector de longitud compatible con x0
    """
    if sx0 is None:
        return [None] * n_results

    if n_results == 1:
        if not isinstance(sx0, (list, tuple)):
            return [sx0]

        if len(sx0) == 1:
            return [sx0[0]]

        is_numeric_vector = all(np.isscalar(v) for v in sx0)
        if is_numeric_vector:
            return [np.asarray(sx0, dtype=float)]

    if not isinstance(sx0, (list, tuple)):
        raise ValueError(
            "Cuando hay varios resultados, 'sx0' debe ser una lista/tupla "
            "con una entrada por resultado."
        )

    if len(sx0) != n_results:
        raise ValueError(
            f"len(sx0) = {len(sx0)} pero hay {n_results} resultados."
        )

    return list(sx0)


def _build_results_payload(results, names, relative_incertainty=None, sx0=None):
    """
    Devuelve una lista de diccionarios con:
      popt, perr, pcov, rel_err_percent

    Si relative_incertainty no es None, sobreescribe temporalmente
    las incertidumbres como:
      perr = relative_incertainty * abs(popt)

    y reconstruye pcov como matriz diagonal:
      pcov = diag(perr^2)

    Si result es un x0 y sx0 no es None, usa sx0 como incertidumbre.
    Si result es un ajuste real, sx0 se ignora.
    """
    n_results = len(results)
    rel_incert_list = _normalize_relative_incertainty(relative_incertainty, n_results)
    sx0_list = _normalize_sx0(sx0, n_results)

    payload = []

    for result, rel_inc, sx0_i in zip(results, rel_incert_list, sx0_list):
        popt, perr, pcov = _extract_fit_info(result, sx0=sx0_i)

        if len(names) != len(popt):
            raise ValueError(
                f"len(names) = {len(names)} pero len(result.x) = {len(popt)}"
            )

        popt = np.asarray(popt, dtype=float)
        perr = np.asarray(perr, dtype=float)
        pcov = np.asarray(pcov, dtype=float)

        if rel_inc is not None:
            perr = rel_inc * np.abs(popt)
            pcov = np.diag(perr**2)

        rel_err_percent = np.full_like(perr, np.nan, dtype=float)
        nonzero = (popt != 0) & np.isfinite(perr)
        rel_err_percent[nonzero] = 100.0 * np.abs(perr[nonzero] / popt[nonzero])

        zero_mask = (popt == 0) & np.isfinite(perr)
        rel_err_percent[zero_mask] = 0.0

        payload.append(
            {
                "popt": popt,
                "perr": perr,
                "pcov": pcov,
                "rel_err_percent": rel_err_percent,
                "relative_incertainty": rel_inc,
                "sx0": sx0_i,
            }
        )

    return payload


# =========================================================
# Export LaTeX
# =========================================================

def export_fit_table_latex(
    results,
    names,
    filename,
    caption,
    label,
    column_names=None,
    units=None,
    err_sigfigs=2,
    rel_sigfigs=2,
    show_relative_error=False,
    relative_incertainty=None,
    sx0=None,
):
    """
    Exporta una tabla LaTeX con uno o varios resultados.

    Parámetros
    ----------
    results : OptimizeResult o list[OptimizeResult]
        Puede ser un único fit o varios.
    names : list[str]
        Nombres de parámetros.
    filename : str
        Fichero .tex de salida.
    caption : str
    label : str
    column_names : list[str] | None
        Nombres de las columnas-modelo.
    units : list[str|None] | None
        Unidades por parámetro. Si se da, se usa \\SI.
    err_sigfigs : int
        Cifras significativas de la incertidumbre.
    rel_sigfigs : int
        Cifras significativas del error relativo.
    show_relative_error : bool
        Si True, añade una subcolumna (%) por cada modelo.
    relative_incertainty : None, float o lista de float
        Si no es None, sobreescribe temporalmente todas las incertidumbres como
        sigma_i = relative_incertainty * abs(x_i)
    sx0 : None o lista
        Si un elemento de results es un x0, permite asignarle incertidumbres.
        Si el elemento correspondiente es un ajuste real, sx0 se ignora.

    Requiere en el preámbulo:
      \\usepackage{siunitx}
      \\usepackage{booktabs}
      \\sisetup{separate-uncertainty = true}
    """
    results = _normalize_results_input(results)
    n_results = len(results)

    if column_names is None:
        column_names = [f"Model {i+1}" for i in range(n_results)]

    if len(column_names) != n_results:
        raise ValueError(
            f"len(column_names) = {len(column_names)} pero hay {n_results} resultados."
        )

    if units is not None and len(units) != len(names):
        raise ValueError(
            f"len(units) = {len(units)} pero len(names) = {len(names)}"
        )

    payload = _build_results_payload(
        results,
        names,
        relative_incertainty=relative_incertainty,
        sx0=sx0,
    )

    if show_relative_error:
        colspec = "l" + "cc" * n_results
    else:
        colspec = "l" + "c" * n_results

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    if show_relative_error:
        header_top = [r"Parámetro"]
        for cname in column_names:
            header_top.append(rf"\multicolumn{{2}}{{c}}{{{cname}}}")
        lines.append(" & ".join(header_top) + r" \\")
        lines.append(r"\cmidrule(lr){2-" + str(1 + 2 * n_results) + r"}")

        header_bottom = [""]
        for _ in column_names:
            header_bottom.extend([r"Valor", r"Incertidumbre (\%)"])
        lines.append(" & ".join(header_bottom) + r" \\")
    else:
        header = [r"Parámetro"] + list(column_names)
        lines.append(" & ".join(header) + r" \\")

    lines.append(r"\midrule")

    for ip, pname in enumerate(names):
        row = [pname]

        for ir in range(n_results):
            val = payload[ir]["popt"][ip]
            err = payload[ir]["perr"][ip]
            rel = payload[ir]["rel_err_percent"][ip]
            unit = None if units is None else units[ip]

            val_fmt = _format_value_uncertainty_latex_siunitx(
                value=val,
                uncertainty=err,
                unit=unit,
                err_sigfigs=err_sigfigs
            )

            if show_relative_error:
                rel_fmt = _format_num_latex(rel, sigfigs=rel_sigfigs)
                row.extend([val_fmt, rel_fmt])
            else:
                row.append(val_fmt)

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_table, encoding="utf-8")

    return latex_table, payload


# =========================================================
# Export CSV
# =========================================================

def export_to_csv(
    output,
    results,
    names,
    column_names=None,
    err_sigfigs=2,
    rel_sigfigs=2,
    relative_incertainty=None,
    sx0=None,
):
    """
    Exporta CSV con uno o varios resultados.

    Para cada modelo crea columnas:
      - <Model>_value
      - <Model>_uncertainty
      - <Model>_value_pm_uncertainty
      - <Model>_siunitx_uncertainty
      - <Model>_rel_err_percent
      - <Model>_rel_err_percent_fmt

    Si relative_incertainty no es None, sobreescribe temporalmente
    todas las incertidumbres como:
      sigma_i = relative_incertainty * abs(x_i)

    Si result es un x0 y sx0 no es None, usa sx0 como incertidumbre.
    """
    results = _normalize_results_input(results)
    n_results = len(results)

    if column_names is None:
        column_names = [f"Model_{i+1}" for i in range(n_results)]

    if len(column_names) != n_results:
        raise ValueError(
            f"len(column_names) = {len(column_names)} pero hay {n_results} resultados."
        )

    payload = _build_results_payload(
        results,
        names,
        relative_incertainty=relative_incertainty,
        sx0=sx0,
    )

    data = {}

    for cname, item in zip(column_names, payload):
        safe_name = str(cname).replace(" ", "_")

        popt = item["popt"]
        perr = item["perr"]
        rel_err_percent = item["rel_err_percent"]

        data[f"parameter"] = popt
        data[f"uncertainty"] = perr
        # data[f"{safe_name}_value_pm_uncertainty"] = [
        #     _format_value_uncertainty_text(v, e, err_sigfigs=err_sigfigs)
        #     for v, e in zip(popt, perr)
        # ]
        # data[f"{safe_name}_siunitx_uncertainty"] = [
        #     _format_uncertainty_siunitx_inner(v, e, err_sigfigs=err_sigfigs)
        #     for v, e in zip(popt, perr)
        # ]
        # data[f"{safe_name}_rel_err_percent"] = rel_err_percent
        # data[f"{safe_name}_rel_err_percent_fmt"] = [
        #     _format_scalar_text(r, sigfigs=rel_sigfigs)
        #     for r in rel_err_percent
        # ]

    df = pd.DataFrame(data, index=names)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)

    return df


# =========================================================
# Export Typst
# =========================================================

def export_fit_table_typst(
    results,
    names,
    filename,
    column_names=None,
    caption=None,
    label=None,
    err_sigfigs=2,
    rel_sigfigs=2,
    show_relative_error=False,
    relative_incertainty=None,
    sx0=None,
):
    """
    Exporta una tabla Typst con uno o varios resultados.

    Si relative_incertainty no es None, sobreescribe temporalmente
    todas las incertidumbres como:
      sigma_i = relative_incertainty * abs(x_i)

    Si result es un x0 y sx0 no es None, usa sx0 como incertidumbre.
    """
    results = _normalize_results_input(results)
    n_results = len(results)

    if column_names is None:
        column_names = [f"Model {i+1}" for i in range(n_results)]

    if len(column_names) != n_results:
        raise ValueError(
            f"len(column_names) = {len(column_names)} pero hay {n_results} resultados."
        )

    payload = _build_results_payload(
        results,
        names,
        relative_incertainty=relative_incertainty,
        sx0=sx0,
    )

    def fmt_name_typst(s):
        s = str(s)
        if s.startswith("$") and s.endswith("$"):
            return s
        return f'"{s}"'

    rows = []
    for ip, pname in enumerate(names):
        row = [f"[{fmt_name_typst(pname)}]"]

        for ir in range(n_results):
            val = payload[ir]["popt"][ip]
            err = payload[ir]["perr"][ip]
            rel = payload[ir]["rel_err_percent"][ip]

            val_pm = _format_value_uncertainty_text(val, err, err_sigfigs=err_sigfigs)
            row.append(f'["{val_pm}"]')

            if show_relative_error:
                rel_fmt = _format_scalar_text(rel, sigfigs=rel_sigfigs)
                row.append(f'["{rel_fmt}"]')

        rows.append("    " + ", ".join(row) + ",")

    caption_text = caption if caption is not None else "Fit parameters"
    label_suffix = f" <{label}>" if label else ""

    ncols = 1 + n_results * (2 if show_relative_error else 1)

    lines = []
    lines.append(f"#figure{label_suffix}(")
    lines.append("  table(")
    lines.append(f"    columns: {ncols},")

    align = ["left"] + ["center"] * (ncols - 1)
    lines.append(f"    align: ({', '.join(align)}),")
    lines.append("    stroke: none,")
    lines.append("    table.header(")

    header_cells = ['      [*Parámetro*],']
    if show_relative_error:
        for cname in column_names:
            header_cells.append(f'      [*{cname}*],')
            header_cells.append('      [*Incertidumbre (%)*],')
    else:
        for cname in column_names:
            header_cells.append(f'      [*{cname}*],')

    lines.extend(header_cells)
    lines.append("    ),")
    lines.extend(rows)
    lines.append("  ),")
    lines.append(f"  caption: [{caption_text}],")
    lines.append(")")

    typst_table = "\n".join(lines)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(typst_table, encoding="utf-8")

    return typst_table, payload