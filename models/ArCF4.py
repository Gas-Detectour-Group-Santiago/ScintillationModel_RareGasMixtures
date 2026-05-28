import numpy as np
from scipy.interpolate import PchipInterpolator

# Santorilli and Diego et al doi 10.1140/epjc/s10052-021-09375-3
tau_3rd = 5.02
tercer_continuo = 0.4866

# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

energy_X_ray_CF4 = 15

def ion_potential(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W


def _prepare_f_cf4(fCF4):
    scalar_input = np.isscalar(fCF4) or np.asarray(fCF4).ndim == 0
    f_cf4 = np.atleast_1d(np.asarray(fCF4, dtype=float))
    return f_cf4, scalar_input


def _interpolate_yields(degrad_data, f_cf4):
    concentration = np.asarray(degrad_data["concentration"], dtype=float)

    cols = ["CF3", "Ar_dbleStar", "CF4", "Ar_3rd"]
    Y = np.asarray(degrad_data[cols].to_numpy(), dtype=float)

    idx = np.argsort(concentration)
    conc_sorted = concentration[idx]
    y_sorted = Y[idx]

    # quitar duplicados en concentración si los hubiera
    conc_unique, unique_idx = np.unique(conc_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    # si solo hay un punto en degrad_data, repetirlo para cualquier entrada
    if len(conc_unique) == 1:
        Y_interp = np.repeat(y_unique, len(f_cf4), axis=0)
        return Y_interp

    # si coinciden exactamente número de puntos y valores, reutiliza directo
    if len(f_cf4) == len(conc_unique) and np.allclose(f_cf4, conc_unique):
        return y_unique

    interp = PchipInterpolator(conc_unique, y_unique, axis=0, extrapolate=True)
    Y_interp = interp(f_cf4)

    # por si scipy devuelve shape (4,) cuando len(f_cf4)==1
    Y_interp = np.atleast_2d(Y_interp)
    return Y_interp


def theory_yield_vis(x, degrad_data, fCF4, n, activate_components=False):
    f_cf4, scalar_input = _prepare_f_cf4(fCF4)
    Y_interp = _interpolate_yields(degrad_data, f_cf4)

    P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd = Y_interp.T

    N = x[0]
    p_CF3 = x[1]
    p_DbleStar = x[2]
    K = x[3]
    K2 = x[4]

    denom = n * f_cf4 * K2 + n * (1.0 - f_cf4) * K + 1 / 30
    frac = np.where(denom == 0, 0.0, K2 * n * f_cf4 / denom)

    total =  N * (
        p_CF3 * P_CF3 + frac * p_DbleStar * P_Ar_dbleStar
    )
    total =  N * (
        p_CF3 * P_CF3 + frac * p_DbleStar * P_Ar_dbleStar
    )
    

    if activate_components:
        if scalar_input:
            return total.item()
        return total/energy_X_ray_CF4

    if scalar_input:
        return total.item()/energy_X_ray_CF4
    
    return total/energy_X_ray_CF4

def theory_yield_uv(x, degrad_data, fCF4, n, activate_components=False):
    f_cf4, scalar_input = _prepare_f_cf4(fCF4)
    Y_interp = _interpolate_yields(degrad_data, f_cf4)

    P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd = Y_interp.T

    N = x[0]
    p_CF3 = x[1]
    p_DbleStar = x[2]
    K = x[3]
    K2 = x[4]
    K1 = x[5]
    K3 = x[6]
    p_CF3 = x[7]
    K4 = x[8]
    PAr_3rd = x[9]
    p_CF3_uv = x[10]


    denom = n * f_cf4 * K2 + n * (1.0 - f_cf4) * K + 1 / 30
    frac = np.where(denom == 0, 0.0, K2 * n * f_cf4 / denom)

    numer = f_cf4 * n
    denom = f_cf4 * n + K1
    frac1 = np.where(denom == 0, 0.0, numer / denom)

    numer = 1.0
    denom = 1.0 + K3 * n * f_cf4
    frac2 = np.where(denom == 0, 0.0, numer / denom)

    denom = (1.0 / tau_3rd) + f_cf4 * n * K4
    numer = f_cf4 * n * K4
    frac3 = np.where(denom == 0, 0.0, numer / denom)

    denom = (1.0 / tau_3rd) + f_cf4 * n * K4
    numer = 1.0 / tau_3rd
    frac4 = np.where(denom == 0, 0.0, numer / denom)

    total = p_CF3_uv *  N * (
        p_CF3 * P_CF3 + frac * p_DbleStar * P_Ar_dbleStar
    )

    total += N * (
        + (frac1 * frac2) * (p_CF3 * P_CF4 + frac3 * P_Ar_3rd * PAr_3rd)
        + tercer_continuo * frac4 * P_Ar_3rd
    )
    

    if activate_components:
        comp_cf4= N * (frac1 * frac2) * (p_CF3 * P_CF4 + frac3 * P_Ar_3rd * PAr_3rd) / energy_X_ray_CF4
        comp_arDbleStar = N * (tercer_continuo * frac4 * P_Ar_3rd) / energy_X_ray_CF4
        comp_cf3 = p_CF3_uv*(theory_yield_vis(x, degrad_data, fCF4, n, activate_components=False)) / energy_X_ray_CF4
        if scalar_input:
            return total.item(), comp_cf3.item()
        return total, comp_cf4, comp_arDbleStar, comp_cf3

    if scalar_input:
        return total.item()/energy_X_ray_CF4
    
    return total/energy_X_ray_CF4