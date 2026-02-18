

import numpy as np

""" 
Modelo de pablo para el ajuste del cociente
"""


# ------------------------------------------------------------------------------
# Datos del paper: 
# Observation of strong wavelength-shifting in the argon-tetraﬂuoromethane system 
# DOI 10.3389/fdest.2023.1282854

K_Ar3rd_to_CF4_plus_star    = 49    # ns-1
K_Ar3rd_to_Ar               = 4.1   # ns-1
n                           = 1     # ns # (??)

# Santorilli and Diego et al doi 10.1140/epjc/s10052-021-09375-3 (en principio igual que el de Pablo Amoedo)
tau_3rd                      = 5.02               

# Realmente  es el cociente K_Ar_dblstar_to_CF3_star/K_Ar_dblstar_to_Ar_star = 36.5, pero a efectos prácticos es lo mismo asumir esto
K_Ar_dblstar_to_CF3_star    = 36.5  # ns-1
K_Ar_dblstar_to_Ar_star     = 1     # ns-1

# ------------------------------------------------------------------------------

def Pgamma_CF3dir(f_cf4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    
    f_cf4       = np.asarray(f_cf4, dtype=float)
    
    return  P_CF3   


def Pgamma_CF3ArDbleStar(f_cf4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    
    f_cf4       = np.asarray(f_cf4, dtype=float)
    
    denom       = f_cf4*K_Ar_dblstar_to_CF3_star + ((1 - f_cf4)) * K_Ar_dblstar_to_Ar_star
    frac        = f_cf4*K_Ar_dblstar_to_CF3_star / denom
   
    return  frac*P_Ar_dbleStar  

def Pgamma_CF4dir(f_cf4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    
    f_cf4       = np.asarray(f_cf4, dtype=float)
    
    return  P_CF4
 

def Pgamma_CF4Ar3rd(f_cf4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    
    f_cf4       = np.asarray(f_cf4, dtype=float)

    denom       = (1 / tau_3rd) + f_cf4 * n * (K_Ar3rd_to_CF4_plus_star + K_Ar3rd_to_Ar)
    numer       = f_cf4 * n * K_Ar3rd_to_CF4_plus_star
    frac        = np.where(denom == 0, 0, numer / denom)  # evitar divisiones por cero
    
    return  P_Ar_3rd * frac

def Pgamma_Ar3rd(f_cf4, n, P_CF3, P_Ar_dbleStar, P_CF4, P_Ar_3rd):
    
    f_cf4       = np.asarray(f_cf4, dtype=float)
    
    denom       = (1 / tau_3rd) + f_cf4 * n * (K_Ar3rd_to_CF4_plus_star + K_Ar3rd_to_Ar)
    numer       = 1 / tau_3rd
    frac        = np.where(denom == 0, 0, numer / denom)
    
    return  P_Ar_3rd * frac * 0.4866

