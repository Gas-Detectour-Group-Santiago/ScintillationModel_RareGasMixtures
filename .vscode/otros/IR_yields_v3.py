import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill
import lmfit as lm
import sys
sys.path.append(r"C:\Users\rauls\Desktop\Lab\Primario\Programas")
from read_DEGRAD_output import read_input
sys.path.append(r"C:\Users\rauls\Desktop\Lab\Primario\Programas")
from estimacion_errores_yields import yield_total_CF4_puro
from scipy.interpolate import interp1d


path_data_primary = r'C:\Users\rauls\Desktop\Lab\Primario\CF4\Datos\CF4_primary_data_final.pkl'
archivo_entrada_01 = r'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_99.9Ar_0.1CF4.txt'
df_01 = read_input(archivo_entrada_01)
archivo_entrada_02 = r'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_99.8Ar_0.2CF4.txt'
df_02 = read_input(archivo_entrada_02)
archivo_entrada_05 = r'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_99.5Ar_0.5CF4.txt'
df_05 = read_input(archivo_entrada_05)
archivo_entrada_1 = r'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_99Ar_1CF4.txt'
df_1 = read_input(archivo_entrada_1)
archivo_entrada_2 = r'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_98Ar_2CF4.txt'
df_2 = read_input(archivo_entrada_2)


with open(path_data_primary, 'rb') as f:
    data = dill.load(f)



# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100])/100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

def ion_potential(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W


def gaussiana(x,A,sigma,mu):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def data_selection(data, conc_b, P):
    mask_conc_b = data['concentracion'] == conc_b
    mask_P = data['presion'] == P

    line = data[mask_conc_b & mask_P]

    return line

def plot_spectrum(data, conc_b, P):
    line = data_selection(data, conc_b, P)

    wavelength = line['data(norm)'].iloc[0]['wavelength']
    intensity = line['data(norm)'].iloc[0]['intensity']

    mask = wavelength>= 500

    #plt.axvline(667, linestyle='--')
    plt.axvline(696, linestyle='--')
    #plt.axvline(714, linestyle='--')
    plt.axvline(727, linestyle='--')
    plt.axvline(750, linestyle='--')
    plt.axvline(763, linestyle='--')
    plt.axvline(772, linestyle='--')
    plt.axvline(794, linestyle='--')

    plt.plot(wavelength[mask], intensity[mask])

    mask_fit = (wavelength >= 540) & (wavelength <= 680)
    x_fit = wavelength[mask_fit]
    y_fit = intensity[mask_fit]
    

    skew_model = lm.models.SkewedGaussianModel()

    params = skew_model.make_params(
        amplitude=np.max(y_fit),
        center=630,
        sigma=30,
        gamma=0.5   # controla la asimetría (positivo = derecha)
    )

    params['gamma'].set(max=1.2)

    # Ajuste
    result = skew_model.fit(y_fit, params, x=x_fit)
    x_plot = np.linspace(500, 800, 2000)
    plt.plot(x_plot, result.eval(x=x_plot), 'r')
    print(result.fit_report())



def yields_IR(data, conc_b, P):
    yields = {}
    line = data_selection(data, conc_b, P)
    
    #params = line['parametros'].iloc[0]
    #print(params.keys())

    wavelength = line['data(norm)'].iloc[0]['wavelength']
    intensity = line['data(norm)'].iloc[0]['intensity']

    

    mask = wavelength >= 500
    
    plt.plot(wavelength[mask], intensity[mask])


    mask_fit = (wavelength >= 540) & (wavelength <= 680)
    x_fit = wavelength[mask_fit]
    y_fit = intensity[mask_fit]
    

    skew_model = lm.models.SkewedGaussianModel()

    params = skew_model.make_params(
        amplitude=np.max(y_fit),
        center=630,
        sigma=30,
        gamma=0.5   # controla la asimetría (positivo = derecha)
    )

    params['gamma'].set(max=1.2)

    # Ajuste
    result = skew_model.fit(y_fit, params, x=x_fit)
    x_plot = np.linspace(500, 800, 2000)
    plt.plot(x_plot, result.eval(x=x_plot), 'r')
    
    positions = [696, 727, 750, 763, 772, 794]#, 798]
    cuts = [(5,6), (4.5,3), (5,6.5), (6.5,3.5), (5.5,4), (2,2)]#, (2,2)]

    for pos, cut in zip(positions, cuts):
        low, high = cut
        plt.axvline(pos, linestyle='--', alpha=0.1, linewidth=1)
        plt.axvline(pos - low, linestyle='--', color='r', alpha=0.1, linewidth=1)
        plt.axvline(pos + high, linestyle='--', color='r', alpha=0.1, linewidth=1)

        mask = (wavelength >= pos-low) & (wavelength <= pos + high)
        yield_ar_plus_cf4 = np.trapezoid(intensity[mask], wavelength[mask])
        yield_cf4 = np.trapezoid(result.eval(x=wavelength[mask]), wavelength[mask])
        yields[pos] = yield_ar_plus_cf4 - yield_cf4
        if yields[pos] < 0:
            yields[pos] = 0
    
    return yields





lifetimes = {696: 28.3, 727: 28.3, 750: 21.7, 763: 29.4, 772: 28.3, 794: 29.3}
levels_start = {696: 21, 727: 21, 750: 22, 763: 17, 772: 21, 794: 19}
peak_positions = [696, 727, 750, 763, 772, 794]





yields_ar_ir_dic = {}

pressures = [1, 2, 3, 4, 5]
concs = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

plt.figure()
for P in pressures:
    yields = yields_IR(data, 0, P)
    yields_norm = {k: v / yield_total_CF4_puro for k, v in yields.items()} #misma normalización que para UV y VIS
    yields_ar_ir_dic[(0, P)] = yields_norm


plt.figure()
for conc in concs:
    yields = yields_IR(data, conc, 1)
    yields_norm = {k: v / yield_total_CF4_puro for k, v in yields.items()} #misma normalización que para UV y VIS
    yields_ar_ir_dic[(conc, 1)] = yields_norm


plt.figure()
plt.xlim(680, 805)
for P in pressures:
    for conc in concs:
        yields = yields_IR(data, conc, P)
        yields_norm = {k: v / yield_total_CF4_puro for k, v in yields.items()} #misma normalización que para UV y VIS
        yields_ar_ir_dic[(conc, P)] = yields_norm



df = pd.DataFrame(
    [
        (conc, P, pos, y_val)
        for (conc, P), y_dict in yields_ar_ir_dic.items()
        for pos, y_val in y_dict.items()
    ],
    columns=['conc_CF4', 'Pressure', 'line_nm', 'yield_IR_norm']
)





def curva1(P, N, K):
    return N / (1+ K * P)

list_colors = ['r', 'g', 'b', 'tab:orange', 'y', 'k']

fig_p, ax_p = plt.subplots(dpi=150)

df_p = df[df['conc_CF4'] == 0]

for line, c in zip(sorted(df_p['line_nm'].unique()), list_colors):
    subset = df_p[df_p['line_nm'] == line]
    ax_p.plot(subset['Pressure'], subset['yield_IR_norm'], marker='o', color=c, label=f'{line} nm')

ax_p.set_xlabel('Pressure')
ax_p.set_ylabel('Yield IR')
ax_p.set_title('Yield vs Pressure (conc CF4 = 0)')
ax_p.legend()
ax_p.grid()







fig_c, ax_c = plt.subplots(dpi=150)

df_c = df[df['Pressure'] == 1]

for line, c in zip(sorted(df_c['line_nm'].unique()), list_colors):
    subset = df_c[df_c['line_nm'] == line]
    ax_c.plot(subset['conc_CF4'], subset['yield_IR_norm'], marker='o', color=c, label=f'{line} nm')

ax_c.set_xlabel('CF4 concentration')
ax_c.set_ylabel('Yield IR')
ax_c.set_title('Yield vs CF4 concentration (Pressure = 1)')
ax_c.legend()
ax_c.grid()






def quenching_model(N_norm, P_ar, K_ar, K_cf4, tau, conc_cf4, pressure, N_ar):
    f = conc_cf4 * 1e-2

    return N_norm * tau / ion_potential(f) * P_ar * N_ar / (1 + f * pressure * tau * K_cf4 + (1 - f) * pressure * tau * K_ar)

def quenching_model_simplified(N_norm, K_ar_P_ar, K_cf4_P_ar, tau, conc_cf4, pressure, N_ar):
    f = conc_cf4 * 1e-2

    return N_norm / ion_potential(f) * N_ar / ( f * pressure * tau * K_cf4_P_ar + (1 - f) * pressure * tau * K_ar_P_ar)




def quenching_model_ar(pressure, N_norm, P_ar, K_ar, tau, N_ar ):

    return N_norm / ion_potential(0.)  * P_ar * N_ar / (1 + pressure * tau * K_ar)


def quenching_model_ar_simplified(pressure, N_norm, K_ar_P_ar, tau, N_ar ):
    return N_norm / ion_potential(0.)   * N_ar / ( pressure * tau * K_ar_P_ar)






def get_N_ar(conc_cf4, pos):
    n_start = levels_start[pos]

    df_conc_path = fr'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_{100-conc_cf4:g}Ar_{conc_cf4:g}CF4.txt'
    if conc_cf4 == 0:
        df_conc_path = fr'C:\Users\rauls\Desktop\Lab\Simulacion_primario_CF4\Ar_CF4\output_99.9Ar_0.1CF4.txt'

    df_conc = read_input(df_conc_path)

    N_ar = sum(df_conc['Eventos'][n_start:53]) 

    return N_ar



inter_for_ar_states = {}
for pos in peak_positions:
    N_ar_list = []
    for conc in concs:
            N_ar_iter =  get_N_ar(conc, pos) 
            N_ar_list.append(N_ar_iter)
    inter_for_ar_states[pos] = interp1d(concs, N_ar_list, fill_value='extrapolate')



def N_ar_interpolator(conc_cf4, pos):
    
    if conc_cf4 in concs:
        N_ar = get_N_ar(conc_cf4, pos)
    
    else:
        interpolador = inter_for_ar_states[pos]
        N_ar = interpolador(conc_cf4)
    
    return N_ar
        

pressure = df['Pressure'].values
conc_cf4 = df['conc_CF4'].values
lines    = df['line_nm'].values
y_data   = df['yield_IR_norm'].values

N_ar = np.array([get_N_ar(c, l) for c, l in zip(conc_cf4, lines)])



def residual_peak(params, pressure, conc_cf4, N_ar, y, 
                  N_norm_peak, tau_peak):
    
    P_ar  = params['P_ar']
    K_ar  = params['K_ar']
    K_cf4 = params['K_cf4']
    
    model_y = quenching_model(
        N_norm=N_norm_peak,
        P_ar=P_ar,
        K_ar=K_ar,
        K_cf4=K_cf4,
        tau=tau_peak,
        conc_cf4=conc_cf4,
        pressure=pressure,
        N_ar=N_ar
    )
    
    return y - model_y


conc_limits = {
    696: {1: 5, 2: 5, 3: 5},
    727: {1: 1, 2: 0.5, 3: 0.5},
    750: {1: 10, 2: 10, 3: 5},
    763: {1: 5, 2: 5, 3: 2},
    772: {1: 5, 2: 5, 3: 5}
}
results_by_peak = {}
concs_plot = np.logspace(-3, 2, 2000)
#df_red = df[df['Pressure'] <= 3]
for line in df['line_nm'].unique():
    
    df_line = df[(df['line_nm'] == line) & (df['Pressure'] <= 3)]
    
    df_filtered_list = []
    
    for P in df_line['Pressure'].unique():
        
        df_P = df_line[df_line['Pressure'] == P]
        
        if line in conc_limits and P in conc_limits[line]:
            max_conc = conc_limits[line][P]
            df_P = df_P[df_P['conc_CF4'] <= max_conc]
        
        df_filtered_list.append(df_P)
    
    # reconstruimos dataframe filtrado
    df_line = pd.concat(df_filtered_list)


    pressure = df_line['Pressure'].values
    conc_cf4 = df_line['conc_CF4'].values
    y_data   = df_line['yield_IR_norm'].values
    
    # construir N_ar para este pico
    N_ar = np.array([get_N_ar(c, line) for c in conc_cf4])
    
    # datos conocidos por pico
    tau_peak = lifetimes[line]
    N_norm   = 0.2189
    
    # parámetros a ajustar
    params = lm.Parameters()
    params.add('P_ar',  value=1, vary=False)
    params.add('K_ar',  value=1e-3, min=0, max=200)
    params.add('K_cf4', value=1e-3, min=0)
    
    minner = lm.Minimizer(
        residual_peak,
        params,
        fcn_args=(pressure, conc_cf4, N_ar, y_data,
                  N_norm, tau_peak)
    )
    
    result = minner.minimize(method='least_squares')
    
    results_by_peak[line] = result
    
    # =============================
    #        GRAFICA
    # =============================
    
    fig, ax = plt.subplots(dpi=150)
    
    P_ar  = result.params['P_ar'].value
    K_ar  = result.params['K_ar'].value
    K_cf4 = result.params['K_cf4'].value
    
    pressures_unique = np.sort(df_line['Pressure'].unique())
    
    for P, c in zip(pressures_unique, list_colors):
        
        mask = (df_line['Pressure'] == P)
        df_P = df_line[mask].sort_values('conc_CF4')
        
        concs = df_P['conc_CF4'].values
        yields_data = df_P['yield_IR_norm'].values
        
        N_ar_P = np.array([N_ar_interpolator(c, line) for c in concs_plot])
        
        yields_model = quenching_model(
            N_norm=N_norm,
            P_ar=P_ar,
            K_ar=K_ar,
            K_cf4=K_cf4,
            tau=tau_peak,
            conc_cf4=concs_plot,
            pressure=P,
            N_ar=N_ar_P
        )
        
        ax.plot(concs, yields_data, marker='o', linestyle='', color=c, label=f'Data {P} bar')
        ax.plot(concs_plot, yields_model, '-', color=c, label=f'Fit {P} bar')
    
    ax.minorticks_on()
    ax.set_xlabel("CF4 Concentration(%)")
    ax.set_ylabel("IR yield")
    Y = degrad_data[cols].to_numpy()   # shape: (n_puntos, 4)

    if len(fN2) > len(concentration):
        # Por si acaso, ordena x e y
        idx = np.argsort(concentration)
        xn = concentration[idx]
        y = Y[idx]

        interp = PchipInterpolator(xn, y, axis=0)
        Y_new = interp(fN2)
    else:
        Y_new = Y
    ax.set_title(f'{line} nm')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlim(0)
    
    


rows = []

for line, result in results_by_peak.items():
    print(f"\n===== Line {line} nm =====")
    print("Chi2 reducido:", result.redchi)
    #result.params.pretty_print()
    lm.report_fit(result)

    row = {'line_nm': line}
    for name, par in result.params.items():
        row[name] = par.value
    
    rows.append(row)

df_results = pd.DataFrame(rows)


path_save = r'C:\Users\rauls\Desktop\Lab\Primario\CF4\Datos\IR_fit_results_2.csv'

# df_results.to_csv(path_save)

# solo pa pasarlle a Diego
# path_save = r'C:\Users\rauls\Desktop\Lab\Primario\CF4\Datos\IR_fit_results.xlsx'
# df_results.to_excel(path_save)




