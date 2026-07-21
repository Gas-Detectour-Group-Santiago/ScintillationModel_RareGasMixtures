import dill 


with open("N2_data_pure_CF4_normalised.pkl", "rb") as f:
    df = dill.load(f)
