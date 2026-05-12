
import numpy as np
from scipy.interpolate import interp1d


from scipy.interpolate import PchipInterpolator

energy_X_ray_N2 = 12 # eV

def W_ArN2(xN2, WAr=26.4, WN2=34.8):
    return 1.0 / ((1.0-xN2)/WAr + xN2/WN2)


def theory_yield_N2_uv(x, degrad_data, fN2, n, activate_components=False):
    fN2 = np.atleast_1d(np.asarray(fN2, dtype=float))
    W = W_ArN2(fN2)

    concentration = degrad_data["concentration"].to_numpy(dtype=float)

    cols = ["N2_star", "Ar_meta", "Ar_res", "Ar_dbleStar"]
    Y = degrad_data[cols].to_numpy(dtype=float)

    idx = np.argsort(concentration)
    conc_sorted = concentration[idx]
    y_sorted = Y[idx]

    interp = PchipInterpolator(conc_sorted, y_sorted, axis=0)
    Y_interp = interp(fN2)

    Pob_N2, Pob_Ar_meta, Pob_Ar_res, Pob_Ar_dbleStar = Y_interp.T

    Nnorm          = x[0]
    P_N2           = x[1]
    tau_N2         = x[2]
    K_N2_Q_N2      = x[3]
    K_N2_Q_Ar      = x[4]
    K_ArMeta_Q_N2c = x[5]
    K_ArMeta_Q_N2b = x[6]
    K_ArMeta_Q_2Ar = x[7]
    K_ArRes_Q_N2c  = x[8]
    K_ArRes_Q_N2b  = x[9]
    K_ArRes_Q_2Ar  = x[10]
    P_Ar_dbleStar  = x[11]
    frac_Ar_dbleStar  = x[12]


    frac_1 = (1/tau_N2) / (1/tau_N2 + n * fN2 * K_N2_Q_N2 + n * (1 - fN2) * K_N2_Q_Ar)
    factor_N2 = frac_1

    frac_2 = (K_ArMeta_Q_N2c * fN2 * n) / (
        (K_ArMeta_Q_N2b + K_ArMeta_Q_N2c) * fN2 * n +
        (K_ArMeta_Q_2Ar * (1-fN2)**2 * n**2)
    )
    factor_Ar_meta = frac_2

    frac_5 = (K_ArRes_Q_N2c * fN2 * n) / (
        (K_ArRes_Q_N2b + K_ArRes_Q_N2c) * fN2 * n +
        K_ArRes_Q_2Ar * (1-fN2)**2 * n**2
    )
    factor_Ar_res = frac_5

    total = (W) * Nnorm/30 * factor_N2 * (
        Pob_N2 * P_N2 + Pob_Ar_meta * factor_Ar_meta + Pob_Ar_res * factor_Ar_res 
        + Pob_Ar_dbleStar * P_Ar_dbleStar * (frac_Ar_dbleStar * factor_Ar_meta + (1-frac_Ar_dbleStar) * factor_Ar_res)
    )

    if activate_components:
        return (
            total/energy_X_ray_N2,
            Nnorm * factor_N2 * (Pob_N2 * P_N2)/energy_X_ray_N2,
            Nnorm * factor_N2 * (Pob_Ar_meta * factor_Ar_meta)/energy_X_ray_N2,
            Nnorm * factor_N2 * (Pob_Ar_res * factor_Ar_res)/energy_X_ray_N2,
        )
    return total/energy_X_ray_N2

############################
## VERSION ANTIGUA

def _theory_yield_N2_uv(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)

    W = W_ArN2(fN2)# (1/ion_potential(fN2))


    concentration = degrad_data["concentration"]
    P_N2 = degrad_data["N2_star"].to_numpy()
    P_Ar_Star = degrad_data["Ar_Star"].to_numpy()
    P_Ar_dbleStar = degrad_data["Ar_dbleStar"].to_numpy()

    N           = x[0]
    p_Star      = x[1]
    K1          = x[2]
    K2          = x[3]
    K3          = x[4]
    tau_emision = x[5]
    p_dbleStar  = x[6]
    K4          = x[7]
    K5          = x[8]

    if len(fN2)>len(P_N2):
        P_N2 = np.interp(fN2,concentration,P_N2)
        P_Ar_Star =  np.interp(fN2,concentration,P_Ar_Star)
        P_Ar_dbleStar =  np.interp(fN2,concentration,P_Ar_dbleStar)

    denom = n * fN2 * K1 + (n**2) * (1.0 - fN2) * K2 
    frac1  = np.where(denom == 0, 0.0, K1 * n * fN2 / denom)
     

    denom = 1 + fN2 * n * tau_emision + (1.0 - fN2) * n * K3
    frac2  = np.where(denom == 0, 0.0, 1 / denom)

    denom = (fN2) * n * K4 + 1/30 + (1-fN2) * n * K5
    frac3  = np.where(denom == 0, 0.0, (1/30)/ denom)
    
    if activate_components:
        return ( N * (P_N2 + (P_Ar_Star + P_Ar_dbleStar * frac3) * frac1) * frac2,
                 N * P_N2 * frac2)
    else:
        return  N * (P_N2 + (P_Ar_Star + P_Ar_dbleStar * p_dbleStar * frac3) * p_Star * frac1) * frac2
