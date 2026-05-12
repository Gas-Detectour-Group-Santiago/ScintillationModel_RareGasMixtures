import dill 


with open("N2_primary_data_final.pkl", "rb") as f:
    df = dill.load(f)
