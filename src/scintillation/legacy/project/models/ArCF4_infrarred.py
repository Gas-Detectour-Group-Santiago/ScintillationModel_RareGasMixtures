
import numpy as np
from scipy.interpolate import interp1d


from scipy.interpolate import PchipInterpolator


# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

energy_X_ray_CF4 = 15

def W_ArCF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W

def interpolation(yvals, fN2, conc):
    conc_sel = np.asarray(conc, dtype=float)
    y_sel = np.asarray(yvals, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)

    if y_sel.ndim == 1:
        mask = np.isfinite(conc_sel) & np.isfinite(y_sel)
    else:
        mask = np.isfinite(conc_sel) & np.all(np.isfinite(y_sel), axis=1)

    conc_sel = conc_sel[mask]
    y_sel = y_sel[mask]

    if len(conc_sel) == 0:
        raise ValueError("No hay ningún punto válido para interpolar.")

    order = np.argsort(conc_sel)
    conc_sorted = conc_sel[order]
    y_sorted = y_sel[order]

    conc_unique, unique_idx = np.unique(conc_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    # ------------------------------------------------------------
    # Caso especial: solo hay una concentración disponible
    # ------------------------------------------------------------
    if len(conc_unique) == 1:
        y0 = y_unique[0]

        # Si fN2 era escalar, devolvemos un escalar o un vector, según y0
        if fN2.ndim == 0:
            return y0

        # Si fN2 es un array, devolvemos y0 repetido para cada punto de fN2
        return np.broadcast_to(
            y0,
            fN2.shape + np.shape(y0)
        ).copy()

    # ------------------------------------------------------------
    # Caso normal: interpolación PCHIP
    # ------------------------------------------------------------
    interp = PchipInterpolator(conc_unique, y_unique, axis=0)
    return interp(fN2)

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_696(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)

    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()


    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T

    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]

  
    frac1 = PAr_star_696 * (1/tau_N2_696) / ( 1/tau_N2_696 + n * fN2 * K_Ar_Q_N2_696 + n * (1 - fN2) * K_Ar_Q_Ar_696)

    if activate_components:
        return (frac1 * Pob_Ar_696, frac1 * Pob_Ar_696) / energy_X_ray_CF4
    else:
        return  frac1 * Pob_Ar_696 / energy_X_ray_CF4
    
    

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_727(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)

    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()

    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T

    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]

  
    frac1 = PAr_star_727 * (1/tau_N2_727) / ( 1/tau_N2_727 + n * fN2 * K_Ar_Q_N2_727+ n * (1 - fN2) * K_Ar_Q_Ar_727) 


    if activate_components:
        return (frac1 * Pob_Ar_727, frac1 * Pob_Ar_727) / energy_X_ray_CF4
    else:
        return  frac1 * Pob_Ar_727 / energy_X_ray_CF4
    

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_750(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)

    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()


    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T

    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]

    frac1 = PAr_star_750 * (1/tau_N2_750) / ( 1/tau_N2_750 + n * fN2 * K_Ar_Q_N2_750 + n * (1 - fN2) * K_Ar_Q_Ar_750)


    if activate_components:
        return (frac1 * Pob_Ar_750, frac1 * Pob_Ar_750) / energy_X_ray_CF4
    else:
        return frac1 * Pob_Ar_750 / energy_X_ray_CF4
    

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_763(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)

    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()

    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T

    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]
  
    frac1 = PAr_star_764 * (1/tau_N2_764) / ( 1/tau_N2_764 + n * fN2 * K_Ar_Q_N2_764+ n * (1 - fN2) * K_Ar_Q_Ar_764)


    if activate_components:
        return (frac1 * Pob_Ar_764, frac1 * Pob_Ar_764) / energy_X_ray_CF4
    else:
        return  frac1 * Pob_Ar_764 / energy_X_ray_CF4
    

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_772(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)

    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()

    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T

    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]

    frac1 = PAr_star_772 * (1/tau_N2_772) / ( 1/tau_N2_772 + n * fN2 * K_Ar_Q_N2_772 + n * (1 - fN2) * K_Ar_Q_Ar_772)


    if activate_components:
        return (frac1 * Pob_Ar_772, frac1 * Pob_Ar_772) / energy_X_ray_CF4
    else:
        return  frac1 * Pob_Ar_772 / energy_X_ray_CF4
    

###################################
###################################
###################################
###################################
###################################
    

def theory_yield_ArCF4_Ir_794(x, degrad_data, fN2, n, activate_components = False):
    fN2 = np.asarray(fN2, dtype=float)
    W = 1/W_ArCF4(fN2)
    
    concentration = degrad_data["concentration"]
    Pob_Ar_696 = degrad_data["Ar_696"].to_numpy()
    Pob_Ar_727 = degrad_data["Ar_727"].to_numpy()
    Pob_Ar_750 = degrad_data["Ar_750"].to_numpy()
    Pob_Ar_764 = degrad_data["Ar_763"].to_numpy()
    Pob_Ar_772 = degrad_data["Ar_772"].to_numpy()
    Pob_Ar_794 = degrad_data["Ar_794"].to_numpy()


    cols = ["Ar_696", "Ar_727", "Ar_750", "Ar_763", "Ar_772", "Ar_794"]
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)
    
    Y = np.asarray(Y, dtype=float)
    fN2 = np.asarray(fN2, dtype=float)



    Pob_Ar_696, Pob_Ar_727, Pob_Ar_750, Pob_Ar_764, Pob_Ar_772, Pob_Ar_794= interpolation(Y,fN2,concentration).T


    PAr_star_696    = x[0]
    tau_N2_696      = x[1]
    K_Ar_Q_Ar_696   = x[2]
    K_Ar_Q_N2_696   = x[3]

    PAr_star_727    = x[4]
    tau_N2_727      = x[5]
    K_Ar_Q_Ar_727   = x[6]
    K_Ar_Q_N2_727   = x[7]

    PAr_star_750    = x[8]
    tau_N2_750      = x[9]
    K_Ar_Q_Ar_750   = x[10]
    K_Ar_Q_N2_750   = x[11]

    PAr_star_764    = x[12]
    tau_N2_764      = x[13]
    K_Ar_Q_Ar_764   = x[14]
    K_Ar_Q_N2_764   = x[15]

    PAr_star_772    = x[16]
    tau_N2_772      = x[17]
    K_Ar_Q_Ar_772   = x[18]
    K_Ar_Q_N2_772   = x[19]



    PAr_star_794    = x[20]
    tau_N2_794      = x[21]
    K_Ar_Q_Ar_794   = x[22]
    K_Ar_Q_N2_794   = x[23]
  
    frac1 = PAr_star_794 * (1/tau_N2_794) / ( 1/tau_N2_794 + n * fN2 * K_Ar_Q_N2_794 + n * (1 - fN2) * K_Ar_Q_Ar_794)


    if activate_components:
       return (frac1 * Pob_Ar_794, frac1 * Pob_Ar_794) / energy_X_ray_CF4
    else:
       return  frac1 * Pob_Ar_794 / energy_X_ray_CF4
    