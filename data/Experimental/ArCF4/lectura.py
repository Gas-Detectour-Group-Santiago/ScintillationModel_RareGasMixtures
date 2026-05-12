import os
import numpy as np
import pandas as pd
import dill
import scipy.special
import importlib
import matplotlib.pyplot as plt 

"""
Script que nos permite leer los datos de los yields de visible/ultravioleta,
sacándolos en formato pickle y csv.
"""

#############################################################################################################
########################## FUNCION PARA LEER LOS PICKLES ####################################################
#############################################################################################################

# Cargar el módulo compilado de bajo nivel
_special_ufuncs = importlib.import_module("scipy.special._special_ufuncs")

# Lista de funciones que pueden faltar
funcs = ["erf", "erfc", "erfi", "gamma", "lgamma", "wofz"]

for name in funcs:
    if not hasattr(_special_ufuncs, name) and hasattr(scipy.special, name):
        setattr(_special_ufuncs, name, getattr(scipy.special, name))


with open("CF4_primary_data_final_with_IR.pkl", "rb") as f:
    df = dill.load(f)
