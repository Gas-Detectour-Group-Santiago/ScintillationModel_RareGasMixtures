import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots
plt.style.use('grid')

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

sys.path.append(models_dir)
sys.path.append(data_dir)

from ArN2 import *
from read_Degrad import read_degrad
from read_experimental import read_experimental
from fiting import fitParameters
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure

#########################################################
####### CREAMOS LOS ARCHIVOS + LOS CARGAMOS 


archivo_entrada=np.array(["/output_Argon_0.1_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_0.5_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_1.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_5.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_10.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_20.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_Argon_50.0_N2_E_0.0_Vcmbar_P_1_bar_12_keV.txt",
                          "/output_100.0N2_E_0.0Vcmbar_P_1bar_12keV.txt"
                          ])

archivo_salida_1=np.array(["/ar_degrad_output_99.9Ar_0.1N2.csv",
                   "/ar_degrad_output_99.5Ar_0.5N2.csv",
                   "/ar_degrad_output_99Ar_1N2.csv",
                   "/ar_degrad_o1.5utput_95Ar_5N2.csv",
                   "/ar_degrad_output_90Ar_1N2.csv",
                   "/ar_degrad_output_80Ar_20N2.csv",
                   "/ar_degrad_output_50Ar_50N2.csv",
                   "/ar_degrad_output_PureN2.csv"
                   ])

archivo_salida_2=np.array(["/n2_degrad_output_99.9Ar_0.1N2.csv",
                   "/n2_degrad_output_99.5Ar_0.5N2.csv",
                   "/n2_degrad_output_99Ar_1N2.csv",
                   "/n2_degrad_output_95Ar_5N2.csv",
                   "/n2_degrad_output_90Ar_10N2.csv",
                   "/n2_degrad_output_80Ar_20N2.csv",
                   "/n2_degrad_output_50Ar_50N2.csv",
                   "/n2_degrad_output_PureN2.csv"
                   ])


prefijo = "../data/Primary_DegradData/ArN2/txt"
archivo_entrada = np.char.add(prefijo, archivo_entrada)

prefijo = "../data/Primary_DegradData/ArN2/csv"
archivo_salida_1 = np.char.add(prefijo, archivo_salida_1)
archivo_salida_2 = np.char.add(prefijo, archivo_salida_2)

gas1 = "ARGON"
gas2 = "NITROGEN"
concentration = np.array([0.001, 0.005, 0.01, 0.05,
                           0.1, 0.2, 0.5, 1
                          ])   

dataframe = pd.DataFrame(
    {    
        "Ar Meta":   [["EXC"],     "ARGON",     0, 11.6, "Ar_meta"],
        "Ar Res":   [["EXC"],     "ARGON",      11.6, 11.7, "Ar_res"],
        "Ar**":   [["EXC"],     "ARGON",     11.7, 100, "Ar_dbleStar"],
        "N2*":    [["C 3PI"], "NITROGEN",  11, 15.5, "N2_star"] #C 3PI
    }, 
    index=["name principal", "gas", "energy low", "energy up", "name output"]
)



output_dir = "../data/Primary_DegradData/ArN2/"
output_general_name =  "../data/Primary_DegradData/ArN2"

read_degrad(archivo_entrada, archivo_salida_1, archivo_salida_2, gas1, gas2, concentration, dataframe, output_dir, output_general_name)


archivo_entrada = "../data/Experimental/ArN2/N2_primary_data_final.pkl"
yields = ["yield_N2"]
presiones = [1,2,2.5,3,4,5]
concentraciones_reales= None


output_dir = "../data/Experimental/ArN2/"

read_experimental(archivo_entrada, yields, presiones, output_dir, concentraciones_reales=concentraciones_reales, uncertainty_mode="all")

#####################################################
###### Traemos los datos anteriormente generados 

DATA_DIR = "../data/Experimental/ArN2/"
yield_N2_uv  = pd.read_csv(os.path.join(DATA_DIR, "yield_N2.csv"))

mask = yield_N2_uv["N2 concentration (%)"] != 120
mask2 = yield_N2_uv["N2 concentration (%)"] != 150
yield_N2_uv = yield_N2_uv[mask & mask2]


"""
columns = yield_N2_uv.columns
concentrations = yield_N2_uv["N2 concentration (%)"].to_numpy()
for i, column in enumerate(columns):
    if "Err" in column:
        yield_N2_uv[column] = yield_N2_uv[columns[i-1]]/10 * np.log10(concentrations*1000)
"""

DATA_DIR = "../data/Primary_DegradData"

degrad_data        = pd.read_csv(os.path.join(DATA_DIR, "ArN2.csv"))


#########################################################3
####### AJUSTE

to_m3  = 2.69 * 10**(25) * 10**(-9) * 273.15 / 300
to_cm3 = 2.69 * 10**(19) * 10**(-9) * 273.15 / 300


tau_N2          = 1e2/np.mean(np.array([2.6,2.07,3.3,2.5,2.74,2.66])) 
K_N2_Q_N2       = to_m3*1e-17*np.mean(np.array([0.71,1.12,1,1.4]))
K_N2_Q_Ar       = to_m3*1e-19*np.mean(np.array([5.6,8.6]))

# tau_N2    = 38.53757225433526
# K_N2_Q_N2 = 0.26444464914836885

K_ArMeta_Q_N2c  = to_m3*1e-17*np.mean(np.array([3.2,3.0,1.1]))
K_ArMeta_Q_N2b  = to_m3*1e-17*np.mean(np.array([0.16]))
K_ArMeta_Q_2Ar  = 1e-9*np.mean(np.array([7.93e6]))

K_ArRes_Q_N2c   = to_m3*1e-17*np.mean(np.array([1.5,3.6]))
K_ArRes_Q_N2b   = to_m3*1e-17*np.mean(np.array([1.5,0]))
K_ArRes_Q_2Ar   = 1e-9*np.mean(np.array([9.24e5]))  


x0_semifixed = np.array([
               0.0, 
               0.0,
               tau_N2, K_N2_Q_N2, K_N2_Q_Ar, 
               K_ArMeta_Q_N2c, K_ArMeta_Q_N2b, K_ArMeta_Q_2Ar,
               K_ArRes_Q_N2c, K_ArRes_Q_N2b, K_ArRes_Q_2Ar,
               0.0, 0.0
               ])

lower_semifixed = x0_semifixed/2e0
upper_semifixed = x0_semifixed*2e0

lower_og       = np.array([
               0.0, 
               0.0,
               0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0
               ]) # + lower_semifixed

x0_og         = np.array([
               0.0044564, 
               0.0,
               0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0
               ]) # + x0_semifixed

upper_og          = np.array([
               1.0, 
               1.0,
               0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0,
               1.0, 1.0
               ]) #+ upper_semifixed

bounds=(list(lower_og+lower_semifixed), list(upper_og+upper_semifixed))

equations = {
    "vis": theory_yield_N2_uv,
}

def W_N2(xN2, WAr=26.4, WN2=34.8):
    return 1.0 / ((1.0-xN2)/WAr + xN2/WN2)


w_n2 =  W_N2(yield_N2_uv["N2 concentration (%)"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar",  "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_n2)[:, None]

yield_N2_uv[y_cols]  = yield_N2_uv[y_cols].to_numpy() * factor

experimental_data = {
    "vis": yield_N2_uv,
}

popt = fitParameters(equations, experimental_data, degrad_data, x0=x0_og+x0_semifixed, bounds=bounds,  is_infrared = True, fixed_idx = [0,2], fixed_error= 0.376)


N_res = popt.fun.size
N_par = popt.x.size
dof   = N_res - N_par
chi2 = 2 * popt.cost
chi2_red = chi2 / dof


print("="*60)
print("Parámetros globales: \n", popt.x)
print(f"Grados de libertad: {dof}")
print(f"Chi2 (real): {chi2}")
print(f"Chi2 reducido: {chi2_red}")
print("="*60)

names_csv = [
    "Nnorm",               

    "P_N2"    ,            

    "tau_N2",              
    "K_N2_Q_N2" ,          
    "K_N2_Q_Ar" ,          

    "K_ArMeta_Q_N2c"  ,    
    "K_ArMeta_Q_N2b"   ,   
    "K_ArMeta_Q_2Ar"    ,  

    "K_ArRes_Q_N2c"  ,    
    "K_ArRes_Q_N2b"   ,   
    "K_ArRes_Q_2Ar"    ,  

    "P_Ar_dbleStar"    ,   
    "frac_Ar_dbleStar"    ,   
    
]
export_to_csv("../data/Parameters/ArN2_primary.csv", popt, names_csv)

# J = popt.jac
# m, p = J.shape
# s2 = 2 * popt.cost / (m - p)
# cov_theta =  s2 * np.linalg.inv(J.T @ J)
# chi2 = 2 * popt.cost
# N_res = popt.fun.size
# N_par = popt.x.size
# dof   = N_res - N_par
# chi2_red = chi2 / dof

#######################################################################
# =================== PLOT ========================
#######################################################################

pressure = [1,2,3,4,5]

concentrations = np.logspace(-4, 0, 1000)

fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
    df_exp=yield_N2_uv,
    theory_func=theory_yield_N2_uv,
    fit_params=popt.x,
    degrad_data=degrad_data,
    concentration_grid=concentrations,
    pressures = pressure,
    x_col="N2 concentration (%)",
    x_plot_factor=100,
    min_positive_x=1e-3,
    title="Primary ArN$_2$ fit ",
    xlabel="Concentration of N$_2$ [$\\%$]",
    ylabel="Normalized Yield",
    xlim=(0.1 * 0.9, 100 * 1.1),
    #ylim=(0.001, 0.1),
    xscale="log",
    yscale="log",
    cmap="viridis",
    darken_factor=-0.15,
    legend=True,
    legend_kwargs={"ncol": 2, "fontsize": 9},
    output="plots/ArN2_global.pdf",
    show=False,
    show_secondary_yaxis = False
)


pressure = [1]

fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
    df_exp=yield_N2_uv,
    theory_func=theory_yield_N2_uv,
    fit_params=popt.x,
    degrad_data=degrad_data,
    concentration_grid=concentrations,
    pressures = pressure,
    x_col="N2 concentration (%)",
    x_plot_factor=100,
    min_positive_x=1e-3,
    title="Primary ArN$_2$ fit 1 bar components",
    xlabel="Concentration of N$_2$ [$\\%$]",
    ylabel="Normalized Yield",
    xlim=(0.1 * 0.9, 100 * 1.1),
    # ylim=(0.001, 0.1),
    xscale="log",
    yscale="log",
    cmap="viridis",
    darken_factor=-0.15,
    legend=True,
    legend_kwargs={"ncol": 2, "fontsize": 9},
    line_label_fmt=["{p:g} bar completed",
                    "{p:g} bar N2 dir",
                    "{p:g} bar Ar* Meta ",
                    "{p:g} bar Ar* Res bar",
                    "{p:g} bar Ar** bar"],
    output="plots/ArN2_global_components_1bar.pdf",
    show=False,
    activate_components = True
)

pressure = [5]

fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
    df_exp=yield_N2_uv,
    theory_func=theory_yield_N2_uv,
    fit_params=popt.x,
    degrad_data=degrad_data,
    concentration_grid=concentrations,
    pressures = pressure,
    x_col="N2 concentration (%)",
    x_plot_factor=100,
    min_positive_x=1e-993,
    title="Primary ArN$_2$ fit 5 bar components",
    xlabel="Concentration of N$_2$ [$\\%$]",
    ylabel="Normalized Yield",
    xlim=(0.1 * 0.9, 100 * 1.1),
    # ylim=(0.001, 0.1),
    xscale="log",
    yscale="log",
    cmap="viridis",
    darken_factor=-0.15,
    legend=True,
    legend_kwargs={"ncol": 2, "fontsize": 9},
    line_label_fmt=["{p:g} bar completed",
                    "{p:g} bar N2 dir",
                    "{p:g} bar Ar* Meta ",
                    "{p:g} bar Ar* Res bar",
                    "{p:g} bar Ar** bar"],
    output="plots/ArN2_global_components_5bar.pdf",
    show=False,
    activate_components = True
)


#######################################################################
# =================== LATEX, TYPST, CSV EXPORT ====================== #
#######################################################################

names_tex = [
    "$N_{\\text{norm}}$",               
    
    "$P_{\\text{N}_2}$"    ,            

    "$\\tau_{\\text{N}_2}$ [ns]",              
    "$K_{\\text{N}_2 Q (\\text{N}_2)}$ [ns$^{-1}$]" ,      
    "$K_{\\text{N}_2 Q (\\text{Ar})}$ [ns$^{-1}$]" ,      

    "$K_{\\text{Ar}_{1s5} Q (\\text{N}_2(\\text{C}))}$ [ns$^{-1}$]" ,    
    "$K_{\\text{Ar}_{1s5} Q (\\text{N}_2(\\text{B}))}$ [ns$^{-1}$]" ,  
    "$K_{\\text{Ar}_{1s5} Q (\\text{2Ar})}$ [ns$^{-1}$]" ,  

    "$K_{\\text{Ar}_{1s4} Q (\\text{N}_2(\\text{C}))}$ [ns$^{-1}$]" ,    
    "$K_{\\text{Ar}_{1s4} Q (\\text{N}_2(\\text{B}))}$ [ns$^{-1}$]" ,  
    "$K_{\\text{Ar}_{1s4} Q (\\text{2Ar})}$ [ns$^{-1}$]" ,  

    "$P_{\\text{Ar}^{**}}$"    ,            
    "$f_{\\text{Ar}^{**}}$"    ,              

 ]



latex_table, _ = export_fit_table_latex(
    results=popt,
    names=names_tex,
    filename="tex_param/ArN2_param.tex",
    caption="Parámetros obtenidos del ajuste global Ar-N$_2$.",
    label="tab:ArN2_fit_params",
    err_sigfigs=2,
)

#######################################################################
# =================== CORRELATION MATRIX ========================
#######################################################################

# # Construimos matriz de correlación a partir de covarianzas
# diag = np.sqrt(np.diag(cov_theta))
# outer = np.outer(diag, diag)
# corr = cov_theta / outer

# # Seguridad numérica
# corr = np.clip(corr, -1, 1)

# # DataFrame para seaborn
# corr_df = pd.DataFrame(corr, columns=names_csv, index=names_csv)

# # --- Plot estilo seaborn ---
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     corr_df,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     fmt=".2f",
#     linewidths=0.5,
#     square=True,
#     cbar_kws={"label": "Correlación"}
# )
# plt.title("Matriz de Correlación de Parámetros Ajustados", fontsize=14)
# plt.tight_layout()

# plt.savefig("plots/ArN2_CorrelationMatrix_GlobalFit.pdf", dpi=300)

#######################################################################
# =================== Parameters ========================
#######################################################################


x0 = x0_og + x0_semifixed

# lower_semifixed = x0_semifixed*0.8
# upper_semifixed = x0_semifixed*1.2
# bounds=(list(lower_og+lower_semifixed), list(upper_og+upper_semifixed))
# popt1 = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds, fixed_idx = [2])
# x1 = popt1.x


# lower_semifixed = x0_semifixed*0.66
# upper_semifixed = x0_semifixed*1.5
# bounds=(list(lower_og+lower_semifixed), list(upper_og+upper_semifixed))
# popt2 = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds, fixed_idx = [2])
# x2 = popt2.x

# lower_semifixed = x0_semifixed*0.5
# upper_semifixed = x0_semifixed*2
# bounds=(list(lower_og+lower_semifixed), list(upper_og+upper_semifixed))
# popt3 = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds, fixed_idx = [2])
# x3 = popt3.x

# lower_semifixed = x0_semifixed*0.33
# upper_semifixed = x0_semifixed*3
# bounds=(list(lower_og+lower_semifixed), list(upper_og+upper_semifixed))
# popt4 = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds, fixed_idx = [2])
# x4 = popt4.x


# popt_secondary = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds)


latex_table, payload = export_fit_table_latex(
    results=[x0, popt],
    names=names_tex,
    filename="tex_param/ArN2_free_factor.tex",
    caption="Parámetros conocidos ($x_0$) y el resultado del ajuste",
    label="tab:ArN2_free_factor",
    column_names=["x0","Ajuste Global"],
    units=None,
    err_sigfigs=2,
    show_relative_error=False,
    relative_incertainty=[0.0, 0.2],

)