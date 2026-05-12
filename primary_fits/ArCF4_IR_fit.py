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
sys.path.append(data_dir)

from ArCF4_infrarred import *
from read_Degrad import read_degrad
from read_experimental import read_experimental
from fiting import fitParameters
from parameter_export import export_fit_table_latex, export_to_csv
from ploting import plot_fit_vs_experiment_by_pressure


#########################################################


def apply_global_threshold(df, conc_col="fCF4",is_727=False):
    bar_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar"]
    err_cols = [f"Err {c}" for c in bar_cols]

    # 1) Región de referencia: 50% a 100%
    df_ref     = df[df[conc_col] >= 50].copy()
    df_ref_10  = df[df[conc_col] == 10].copy()
    df_ref_20  = df[df[conc_col] == 20].copy()
    df_ref_50  = df[df[conc_col] == 50].copy()
    df_ref_100 = df[df[conc_col] == 100].copy()

    # 2) Threshold global del dataframe = máximo entre todas las columnas de presión
    threshold_10 = df_ref_10[bar_cols].max().max()
    threshold_20 = df_ref_20[bar_cols].max().max()
    threshold_50 = df_ref_50[bar_cols].max().max()
    threshold_100 = df_ref_100[bar_cols].max().max()

    threshold = 0 # min(threshold_20,threshold_50,threshold_100) # 

    # 3) Nos quedamos con la región < 50% para ajustar
    df_low = df[df[conc_col] < 11].copy()
    if is_727:
        df_low = df[df[conc_col] < 6].copy()

    # 4) Máscara celda a celda: conservar solo yields >= threshold
    mask = df_low[bar_cols] >= threshold

    # 5) Aplicar máscara a yields
    df_low[bar_cols] = df_low[bar_cols].where(mask)

    # 6) Aplicar la misma máscara a los errores correspondientes
    for bar, err in zip(bar_cols, err_cols):
        df_low[err] = 0.0009
        df_low[err] = df_low[err].where(mask[bar])

    df_low["fCF4"] *= 1
    return df_low, threshold


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

dataframe = pd.DataFrame(
    {    
        "Ar* 696":   [["EXC"],     "ARGON",     13.32, 13.32 + 10, "Ar_696"],
        "Ar* 727":   [["EXC"],     "ARGON",     13.32, 13.32 + 10, "Ar_727"],
        "Ar* 750":   [["EXC"],     "ARGON",     13.47, 13.47 + 10, "Ar_750"],
        "Ar* 763":   [["EXC"],     "ARGON",     13.17, 13.17 + 10, "Ar_763"],
        "Ar* 772":   [["EXC"],     "ARGON",     13.32, 13.32 + 10, "Ar_772"],
        "Ar* 794":   [["EXC"],     "ARGON",     13.28, 13.28 + 10, "Ar_794"],
    }, 
    index=["name principal", "gas", "energy low", "energy up", "name output"]
)



output_dir = "../data/Primary_DegradData/ArCF4/"
output_general_name =  "../data/Primary_DegradData/ArCF4_IR"

read_degrad(archivo_entrada, archivo_salida_1, archivo_salida_2, gas1, gas2, concentration, dataframe, output_dir, output_general_name)


archivo_entrada = "../data/Experimental/ArCF4/CF4_primary_data_final.pkl"
yields = ["696", "727", "750", "763", "772", "794"]
presiones = [1, 2, 3, 4, 5]
concentraciones_reales = None

output_dir = "../data/Experimental/ArCF4/"

read_experimental(
    archivo_entrada,
    yields,
    presiones,
    output_dir,
    concentraciones_reales=concentraciones_reales,
    uncertainty_mode="stadistic",
    yield_mode="ir",
)

# archivo_entrada = "../data/Experimental/ArCF4/IR_yields.pkl"
# yields = ["696","727","750","763","772","794"]
# presiones = [1,2,3,4,5]
# concentraciones_reales= None
# no_sistematic = False

# output_dir = "../data/Experimental/ArCF4/"

# read_experimental(
#     archivo_entrada,
#     yields,
#     presiones,
#     output_dir,
#     concentraciones_reales=concentraciones_reales,
#     no_sistematic=no_sistematic,
# )


#####################################################
###### Traemos los datos anteriormente generados 

# % de CF4 en Ar
cf4_pct = np.array([0, 1.0, 2.0, 5.0, 10, 20, 30, 50, 75, 100]) / 100

# Potencial de ionización (según la columna Ar/CF4)
ion_pot = np.array([26.4, 26.7, 26.9, 27.4, 28.1, 29.4, 30.2, 31.7, 33.0, 34.3])

energy_X_ray_CF4 = 15

def W_CF4(f):
    f_cf4 = np.asarray(f, dtype=float)
    W = np.interp(f_cf4, cf4_pct, ion_pot)
    return W


DATA_DIR = "../data/Experimental/ArCF4/"
yield_696_ir  = pd.read_csv(os.path.join(DATA_DIR, "696.csv"))
yield_727_ir  = pd.read_csv(os.path.join(DATA_DIR, "727.csv"))
yield_750_ir  = pd.read_csv(os.path.join(DATA_DIR, "750.csv"))
yield_763_ir  = pd.read_csv(os.path.join(DATA_DIR, "763.csv"))
yield_772_ir  = pd.read_csv(os.path.join(DATA_DIR, "772.csv"))
yield_794_ir  = pd.read_csv(os.path.join(DATA_DIR, "794.csv"))


w_cf4 =  W_CF4(yield_696_ir["fCF4"].to_numpy()/100) 
y_cols = ["1.0bar", "2.0bar", "3.0bar", "4.0bar", "5.0bar", 'Err 1.0bar','Err 2.0bar','Err 3.0bar','Err 4.0bar','Err 5.0bar']
factor = (1 / w_cf4)[:, None]

yield_696_ir[y_cols]  = yield_696_ir[y_cols].to_numpy() * factor
yield_727_ir[y_cols]  = yield_727_ir[y_cols].to_numpy() * factor
yield_750_ir[y_cols]  = yield_750_ir[y_cols].to_numpy() * factor
yield_763_ir[y_cols]  = yield_763_ir[y_cols].to_numpy() * factor
yield_772_ir[y_cols]  = yield_772_ir[y_cols].to_numpy() * factor
yield_794_ir[y_cols]  = yield_794_ir[y_cols].to_numpy() * factor


yield_696_ir_n, thr_696 = apply_global_threshold(yield_696_ir)
yield_727_ir_n, thr_727 = apply_global_threshold(yield_727_ir)
yield_750_ir_n, thr_750 = apply_global_threshold(yield_750_ir)
yield_763_ir_n, thr_763 = apply_global_threshold(yield_763_ir)
yield_772_ir_n, thr_772 = apply_global_threshold(yield_772_ir)
yield_794_ir_n, thr_794 = apply_global_threshold(yield_794_ir)

print(thr_696, thr_727, thr_750, thr_763, thr_772, thr_794)


"""
columns = yield_N2_uv.columns
concentrations = yield_N2_uv["N2 concentration (%)"].to_numpy()
for i, column in enumerate(columns):
    if "Err" in column:
        yield_N2_uv[column] = yield_N2_uv[columns[i-1]]/10 * np.log10(concentrations*1000)
"""

DATA_DIR = "../data/Primary_DegradData"

degrad_data = pd.read_csv(os.path.join(DATA_DIR, "ArCF4_IR.csv"))


#########################################################3
####### AJUSTE

to_m3  = 2.69 * 10**(25) * 10**(-9)
to_cm3 = 2.69 * 10**(19) * 10**(-9)


x0_semifixed = np.array([
               0.0, 28.3, 0.0, 0.0, 
               0.0, 28.3, 0.0, 0.0, 
               0.0, 21.7, 0.0, 0.0, 
               0.0, 29.4, 0.0, 0.0, 
               0.0, 28.3, 0.0, 0.0, 
               0.0, 29.3, 0.0, 0.0, 
               ])

lower_semifixed = x0_semifixed*0.999999999999999
upper_semifixed = x0_semifixed*1.000000000000001

lower       = np.array([
               0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0, 
               0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               ]) + lower_semifixed

x0          = np.array([
               0.02, 0.0, 1.0, 1.0, 
               0.02, 0.0, 1.0, 1.0, 
               0.02, 0.0, 1.0, 1.0, 
               0.02, 0.0, 1.0, 1.0,  
               0.02, 0.0, 1.0, 1.0, 
               0.02, 0.0, 1.0, 1.0, 
               ]) + x0_semifixed

upper          = np.array([
               0.02, 0.0, 1000.0, 1000.0, 
               0.02, 0.0, 1000.0, 1000.0, 
               0.02, 0.0, 1000.0, 1000.0, 
               0.02, 0.0, 1000.0, 1000.0, 
               0.02, 0.0, 1000.0, 1000.0, 
               0.02, 0.0, 1000.0, 1000.0,  
               ]) + upper_semifixed

bounds=(list(lower), list(upper))

equations = {
    "696": theory_yield_ArCF4_Ir_696,
    "727": theory_yield_ArCF4_Ir_727,
    "750": theory_yield_ArCF4_Ir_750,
    "763": theory_yield_ArCF4_Ir_763,
    "772": theory_yield_ArCF4_Ir_772,
    "794": theory_yield_ArCF4_Ir_794,
}

experimental_data = {
    "696": yield_696_ir_n.fillna(0),
    "727": yield_727_ir_n.fillna(0),
    "750": yield_750_ir_n.fillna(0),
    "763": yield_763_ir_n.fillna(0),
    "772": yield_772_ir_n.fillna(0),
    "794": yield_794_ir_n.fillna(0),
}


popt = fitParameters(equations, experimental_data, degrad_data, x0=x0, bounds=bounds,  is_infrared = True,
                    fixed_idx = [1,5,9,13,17,21],
                    fixed_error= 0.1)


cov_theta = popt.pcov
chi2 = popt.chi2
dof = popt.dof
chi2_red = popt.chi2_red

print("="*60)
print("Parámetros globales: \n", popt.x)
print(f"Grados de libertad: {dof}")
print(f"Chi2 (real): {chi2}")
print(f"Chi2 reducido: {chi2_red}")
print("="*60)


#######################################################################
# =================== PLOT ========================
#######################################################################

yield_696_ir_n.loc[0,"fCF4"]=0.001
yield_727_ir_n.loc[0,"fCF4"]=0.001
yield_750_ir_n.loc[0,"fCF4"]=0.001
yield_763_ir_n.loc[0,"fCF4"]=0.001
yield_772_ir_n.loc[0,"fCF4"]=0.001
yield_794_ir_n.loc[0,"fCF4"]=0.001

experimental_data = {
    "696": yield_696_ir_n,
    "727": yield_727_ir_n,
    "750": yield_750_ir_n,
    "763": yield_763_ir_n,
    "772": yield_772_ir_n,
    "794": yield_794_ir_n,
}

pressure = [1,2,3,4,5]

for name in equations:

    concentrations = np.logspace(-6, 0, 1000)

    fig, ax, pressure_cols = plot_fit_vs_experiment_by_pressure(
        df_exp=experimental_data[name],
        theory_func=equations[name],
        fit_params=popt.x,
        degrad_data=degrad_data,
        concentration_grid=concentrations,
        pressures = pressure,
        x_col="fCF4",
        x_plot_factor=100,
        min_positive_x=1e-3,
        title=f"Primary ArCF$_4$ IR ({name} nm) Yield fit",
        xlabel="Concentration of CF$_4$ [$\%$]",
        ylabel="Normalized Yield",
        xlim=(0.001 * 0.9, 100 * 1.1),
        ylim=(0.000001, 0.005),
        xscale="log",
        yscale="log",
        cmap="viridis",
        darken_factor=-0.15,
        legend=True,
        legend_kwargs={"ncol": 2, "fontsize": 9},
        output=f"plots/ArCF4_IR/ArCF4_global_{name}.pdf",
        show=False,
        activate_components = False
    )



#######################################################################
# =================== LATEX, TYPST, CSV EXPORT ========================
#######################################################################

names_tex = [
    "$P_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 696 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 696 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 696 \\mathrm{nm}}$",


    "$P_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 727 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 727 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 727 \\mathrm{nm}}$",


    "$P_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 750 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 750 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 750 \\mathrm{nm}}$",


    "$P_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 764 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 764 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 764 \\mathrm{nm}}$",

    "$P_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 772 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 772 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 772 \\mathrm{nm}}$",


    "$P_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
    "$\\tau_{\\mathrm{Ar}^* \\ 794 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{Ar}) \\ 794 \\mathrm{nm}}$",
    "$K_{\\mathrm{Ar}^*, Q(\\mathrm{CF}_4) \\ 794 \\mathrm{nm}}$",
]


names_csv = [
    "PAr_star_696"    ,
    "tau_N2_696"     ,
    "K_Ar_Q_Ar_696"   ,
    "K_Ar_Q_N2_696"  ,

    "PAr_star_727"   ,
    "tau_N2_727"   ,
    "K_Ar_Q_Ar_727"   ,
    "K_Ar_Q_N2_727" ,

    "PAr_star_750"   ,
    "tau_N2_750"    ,
    "K_Ar_Q_Ar_750"   ,
    "K_Ar_Q_N2_750"  ,

    "PAr_star_764"   ,
    "tau_N2_764"     ,
    "K_Ar_Q_Ar_764"  ,
    "K_Ar_Q_N2_764" ,

    "PAr_star_772" ,
    "tau_N2_772",
    "K_Ar_Q_Ar_772",
    "K_Ar_Q_N2_772" ,

    "PAr_star_794" ,
    "tau_N2_794",
    "K_Ar_Q_Ar_794",
    "K_Ar_Q_N2_794" ,

]

export_to_csv("../data/Parameters/ArCF4_IR_primary.csv",popt,names_csv)


latex_table, _, perr, rel = export_fit_table_latex(
    result=popt,
    names=names_tex,
    filename="tex_param/ArCF4_IR_param.tex",
    caption="Parámetros obtenidos del ajuste del centelleo IR en Ar--CF$_4$.",
    label="tab:ArCF4_IR_fit",
    sigfigs=4
)

#######################################################################
# =================== CORRELATION MATRIX ========================
#######################################################################


# Construimos matriz de correlación a partir de covarianzas
diag = np.sqrt(np.diag(cov_theta))
outer = np.outer(diag, diag)
corr = cov_theta / outer

# Seguridad numérica
corr = np.clip(corr, -1, 1)

# DataFrame para seaborn
corr_df = pd.DataFrame(corr, columns=names_tex, index=names_tex)

# --- Plot estilo seaborn ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Correlación"}
)
plt.title("Matriz de Correlación de Parámetros Ajustados", fontsize=14)
plt.tight_layout()

plt.savefig("plots/ArCF4_IR/ArCF4_IR_CorrelationMatrix_GlobalFit.pdf", dpi=300)
