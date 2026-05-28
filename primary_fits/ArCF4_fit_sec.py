import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import scienceplots  

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))


sys.path.append(models_dir)
from ArCF4 import *

sys.path.append(data_dir)
from read_Degrad import read_degrad
from read_experimental import read_experimental
from fiting import fitParameters,fitParameters_lmfit,fitParameters_minimize
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure

#########################################################
####### CREAMOS LOS ARCHIVOS + LOS CARGAMOS 


archivo_entrada=np.array(["/output_99.9Ar_0.1CF4.txt",
                   "/output_99.8Ar_0.2CF4.txt",
                   "/output_99.5Ar_0.5CF4.txt",
                   "/output_99Ar_1CF4.txt",
                   "/output_98Ar_2CF4.txt",
                   "/output_95Ar_5CF4.txt",
                   "/output_90Ar_10CF4.txt",
                   "/output_80Ar_20CF4.txt",
                   "/output_50Ar_50CF4.txt",
                   "/output_PureCF4.txt"])

archivo_salida_1=np.array(["/ar_degrad_output_99.9Ar_0.1CF4.csv",
                   "/ar_degrad_output_99.8Ar_0.2CF4.csv",
                   "/ar_degrad_output_99.5Ar_0.5CF4.csv",
                   "/ar_degrad_output_99Ar_1CF4.csv",
                   "/ar_degrad_output_98Ar_2CF4.csv",
                   "/ar_degrad_output_95Ar_5CF4.csv",
                   "/ar_degrad_output_90Ar_10CF4.csv",
                   "/ar_degrad_output_80Ar_20CF4.csv",
                   "/ar_degrad_output_50Ar_50CF4.csv",
                   "/ar_degrad_output_PureCF4.csv"])

archivo_salida_2=np.array(["/cf4_degrad_output_99.9Ar_0.1CF4.csv",
                   "/cf4_degrad_output_99.8Ar_0.2CF4.csv",
                   "/cf4_degrad_output_99.5Ar_0.5CF4.csv",
                   "/cf4_degrad_output_99Ar_1CF4.csv",
                   "/cf4_degrad_output_98Ar_2CF4.csv",
                   "/cf4_degrad_output_95Ar_5CF4.csv",
                   "/cf4_degrad_output_90Ar_10CF4.csv",
                   "/cf4_degrad_output_80Ar_20CF4.csv",
                   "/cf4_degrad_output_50Ar_50CF4.csv",
                   "/cf4_degrad_output_PureCF4.csv"])


prefijo = "../data/Primary_DegradData/ArCF4/txt"
archivo_entrada = np.char.add(prefijo, archivo_entrada)

prefijo = "../data/Primary_DegradData/ArCF4/csv"
archivo_salida_1 = np.char.add(prefijo, archivo_salida_1)
archivo_salida_2 = np.char.add(prefijo, archivo_salida_2)

gas1 = "ARGON"
gas2 = "CF4"
concentration = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1])   

E_th_Ar  = 12.9
E_th_CF3 = 12.9

dataframe = pd.DataFrame(
    {
        "CF4":    [["ION CF3 +"],                            "CF4",       15, 100, "CF4"],

        "Ar**":   [["EXC"],                                  "ARGON",   E_th_Ar, 100, "Ar_dbleStar"],

        "CF3":    [["NEUTRAL DISS"],                         "CF4",  E_th_CF3, 100, "CF3"],

        "Ar3rd":  [["CHARGE STATE ="],      "ARGON",    40, 100, "Ar_3rd"]
        
    }, 
    index=["name principal", "gas", "energy low", "energy up", "name output"]
)

output_dir = "../data/Primary_DegradData/ArCF4/"
output_general_name =  "../data/Primary_DegradData/ArCF4"

read_degrad(archivo_entrada, archivo_salida_1, archivo_salida_2, gas1, gas2, concentration, dataframe, output_dir, output_general_name)


archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"
yields = ["vis","UV"]
presiones = [1,2,2.5,3,4,5]
#concentraciones_reales= np.array([0.00001,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1])*100


output_dir = "../data/Experimental/ArCF4/"

read_experimental(archivo_entrada, yields, presiones, output_dir, uncertainty_mode="all")

dic = {"E_th_Ar**": [E_th_Ar], "E_th_CF3": [E_th_CF3]}

df = pd.DataFrame(data=dic)

df.to_csv("../data/Thresholds/ArCF4.csv", index=False)

#####################################################
###### Traemos los datos anteriormente generados 

DATA_DIR = "../data/Experimental/ArCF4/"
yield_uv  = pd.read_csv(os.path.join(DATA_DIR, "UV.csv"))
yield_vis = pd.read_csv(os.path.join(DATA_DIR, "vis.csv"))



# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

energy_X_ray_CF4 = 15

def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W

w_cf4 =  W_CF4(yield_vis["fCF4"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar", "2.5bar", "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_cf4)[:, None]

yield_vis[y_cols]  = yield_vis[y_cols].to_numpy() * factor
yield_uv[y_cols]  = yield_uv[y_cols].to_numpy() * factor

#####################################################

DATA_DIR = "../data/Primary_DegradData"
degrad_data        = pd.read_csv(os.path.join(DATA_DIR, "ArCF4.csv"))


names_csv = [
    "Nnorm",
    "PCF3dir vis$",
    "PAr**",
    "KAr**QAr",
    "KAr**QCF4",
    "1/tauDiscKrelax",
    "tauUvKCF4QCF4",
    "PCF4dir",
    "KAr++QCF4",
    "PAr++",
    "PCF3dir uv$",
]

names_tex = [
    "$N_{\\text{norm}}$",
    "$P_{\\mathrm{CF_3}}|_{\\mathrm{vis,dir}}$",
    "$P_{\\mathrm{Ar}^{**}} $",
    "${K_{\\mathrm{Ar^{**},Q(Ar)}}}$ [ns]",
    "${K_{\\mathrm{Ar^{**},Q(CF_4)}}}$ [ns]",
    "$1 / {\\tau_{\\mathrm{dis}} K_{\\mathrm{relax}}}$",
    "$\\tau_{\mathrm{uv}} K_{\mathrm{CF_4^{+,*}Q(CF_4)}}$",
    "$P_{\\mathrm{CF_4^{+,*}}}|_{\\mathrm{dir}}$",
    "$K_{\\mathrm{Ar^{++},Q(CF_4)}}$ [ns]",
    "$P_{\\mathrm{Ar}^{++}}$",
    "$P_{\\mathrm{CF_3}}|_{\\mathrm{uv,dir}}$",
]



#######################################################################
# =================== PRIMARIO ========================
#######################################################################

x0 = np.array([0.14,
               0.10, 0.99, 3, 0.037*3,
               1.0 ,0.065, 0.48, 50.10, 0.37,
               0.00001])


x0 = np.array([4.45646783e-03, 6.97523694e-02, 1.50511150e-01, 3.89959971e-01,
 1.14938357e+01 ,5.67737517e-03 ,6.50000000e-02, 2.36307283e-01,
 5.00500000e+01, 5.18971635e-01 ,1.00000000e-04])

lower = [0.0,
         0.0, 0.0, 0.0, 0.0,
         0.00, 0.065, 0.0, 50, 0.0,
         0.0]

upper = [0.99, 
         1.0, 1.0, 10000.0, 10000.0,
         10000.0, 0.066, 1.0, 50.2, 1.0, 
         0.0001]

bounds=(lower, upper)

equations = {
    "vis": theory_yield_vis,
    "uv": theory_yield_uv
}

yield_uv.loc[0,"fCF4"]=0.001

yield_vis.loc[0, :].drop(['Err 1.0bar','Err 2.0bar','Err 2.5bar','Err 3.0bar','Err 4.0bar','Err 5.0bar'])
yield_vis.fillna(0)

experimental_data = {
    "vis": yield_vis,
    "uv": yield_uv
}


popt_primary = fitParameters(equations, experimental_data, 
                     degrad_data, x0=x0, bounds=bounds,
                     #fixed_error=[0.01],
                     #is_infrared=True
                     )

par_primary = popt_primary.x.copy()
print("Parámetros globales primario:", popt_primary.x)

par_primary[1] *= 0.9    # 0.9
par_primary[2] *= 1.3    # 1.3
par_primary[10] = 0.25   # 1.3


#######################################################################
# =================== SECUNDARIO ========================
#######################################################################

# Para poder describir primario y secundario a la vez (fenomenologicamente) con la prescricpión de Pscint común y um threshold inferior de energía diferente tenemos que poner las siguientes exigencias. Estas son, aprox:
# -- Nnorm ~ 0.14
# -- Par** ~ 0.27
# -- Pcf3  ~ 0.09
# Con esto tienes un chi2 ~ 0.9 

x0 = np.array([0.00,
               0.0, 0.99, 3, 0.037*3,
               1.0 ,0.1, 0.48, 50.1, 0.37,
               0.2])
lower = [0.0,
         0.0, 0.0, 0.0, 0.0,
         0.00, 0.0, 0.0, 0, 0.0,
         0.0]

upper = [1.0, 
         1.0, 1.0, 10000.0, 10000.0,
         10000.0, 10000.0, 1.0, 10000, 1.0, 
         1.0]
         
bounds=(lower, upper)

popt_secondary = fitParameters(equations,
                               experimental_data,
                               degrad_data, 
                               x0=x0,
                               bounds=bounds,
                               fixed_idx=[0,1,2,3,4,5,6,7,8,9,10],
                               fixed_values = par_primary[:])

export_to_csv("../data/Parameters/ArCF4_secondary_1_3.csv",popt_secondary,names_csv)


latex_table, payload = export_fit_table_latex(
    results=[popt_primary, popt_secondary],
    names=names_tex,
    filename="tex_param/ArCF4_param_primary_secondary.tex",
    caption="Parámetros obtenidos del ajuste global de Ar--CF$_4$.",
    label="tab:cf4_fit_params",
    column_names=["Modelo Primario", "Modelo Primario y Secundario"],
    units=None,
    err_sigfigs=2,
    show_relative_error=False,
)

chi2 = 2 * popt_secondary.cost
N_res = popt_secondary.fun.size
N_par = popt_secondary.x.size
dof   = N_res - N_par
chi2_red = chi2 / dof

print("="*60)
print("Parámetros globales secundario:", popt_secondary.x)
print(f"Chi2 (real): {chi2}")
print(f"Grados de libertad: {dof}")
print(f"Chi2 reducido: {chi2_red}")
print("="*60)

par = popt_secondary.x

#############################33
#### PLOT 

pressure = [1,3,4]


concentrations = np.logspace(-4, 0, 1000)

fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
    df_exp=yield_vis,
    theory_func=theory_yield_vis,
    fit_params= par,
    degrad_data=degrad_data,
    concentration_grid=concentrations,
    pressures = pressure,
    x_col="fCF4",
    x_plot_factor=100,
    min_positive_x=1e-3,
    title="Primary ArCF$_4$ Visible Yield fit ",
    xlabel="Concentration of CF$_4$ [$\%$]",
    ylabel="Normalized Yield",
    xlim=(0.1 * 0.9, 100 * 1.1),
    ylim=(0.0002, 0.02),
    xscale="log",
    yscale="log",
    cmap="viridis",
    darken_factor=-0.15,
    legend=True,
    legend_kwargs={"ncol": 2, "fontsize": 9, "loc":"upper left"},
    output="plots/ArCF4_sec/ArCF4_visible.pdf",
    show=False,
)



concentrations = np.logspace(-6, 0, 1000)

fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
    df_exp=yield_uv,
    theory_func=theory_yield_uv,
    fit_params= par,
    degrad_data=degrad_data,
    concentration_grid=concentrations,
    pressures = pressure,
    x_col="fCF4",
    x_plot_factor=100,
    min_positive_x=1e-5,
    title="Primary ArCF$_4$ UV Yield fit ",
    xlabel="Concentration of CF$_4$ [$\%$]",
    ylabel="Normalized Yield",
    xlim=(0.001 * 0.9, 100 * 1.1),
    ylim=(0.001, 0.1),
    xscale="log",
    yscale="log",
    cmap="viridis",
    darken_factor=-0.15,
    legend=True,
    legend_kwargs={"ncol": 2, "fontsize": 9, "loc":"upper left"},
    line_label_fmt=["{p:g} bar completed",
                    "{p:g} bar CF3*"],
    output="plots/ArCF4_sec/ArCF4_uv.pdf",
    show=True,
    activate_components=False
)

