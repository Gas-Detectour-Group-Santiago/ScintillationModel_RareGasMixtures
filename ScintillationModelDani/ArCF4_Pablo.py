

import numpy as np


# Santorilli and Diego et al doi 10.1140/epjc/s10052-021-09375-3 (en principio igual que el de Pablo Amoedo)
tau_3rd                      = 5.02               
tercer_continuo              = 0.4866


def theory_yield_vis(x, fCF4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    f_cf4 = np.asarray(fCF4, dtype=float)

    N           = x[0]
    p_CF3       = x[1]
    p_DbleStar  = x[2]
    K           = x[3]

    denom = f_cf4 + (1.0 - f_cf4) * K
    frac  = np.where(denom == 0, 0.0, f_cf4 / denom)
    
    

    # OJO: aquí faltaba un "*" en tu ejemplo: p_CF3(P_CF3 + ...) → p_CF3 * (...)
    return N *(p_CF3 * P_CF3 + frac * p_DbleStar * P_Ar_dbleStar)


def theory_yield_uv(x, fCF4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    f_cf4 = np.asarray(fCF4, dtype=float)

    

    N      = x[0]
    p_CF3  = x[4]
    K3     = x[5]
    K4     = x[6]

    

    # frac3
    denom = (1.0 / tau_3rd) + f_cf4 * n * (K3)
    numer = f_cf4 * n * K3 
    frac3 = np.where(denom == 0, 0.0, numer / denom)

    # frac4
    denom = (1.0 / tau_3rd) + f_cf4 * n * (K3)
    numer = 1.0 / tau_3rd
    frac4 = np.where(denom == 0, 0.0, numer / denom)

    return N * ((p_CF3 * P_CF4 + frac3 * P_Ar_3rd * K4)
        + tercer_continuo * frac4 * P_Ar_3rd )

############### Eliminamos el parámetro K4 #################################

def theory_yield_uv_noP(x, fCF4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    f_cf4 = np.asarray(fCF4, dtype=float)

    

    N      = x[0]
    p_CF3  = x[4]
    K3     = x[5]
    

    # frac3
    denom = (1.0 / tau_3rd) + f_cf4 * n * (K3)
    numer = f_cf4 * n * K3 
    frac3 = np.where(denom == 0, 0.0, numer / denom)

    # frac4
    denom = (1.0 / tau_3rd) + f_cf4 * n * (K3)
    numer = 1.0 / tau_3rd
    frac4 = np.where(denom == 0, 0.0, numer / denom)

    return N * ((p_CF3 * P_CF4 + frac3 * P_Ar_3rd)
        + tercer_continuo * frac4 * P_Ar_3rd )
