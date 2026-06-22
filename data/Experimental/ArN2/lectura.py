import dill 


with open("N2_data.pkl", "rb") as f:
    df = dill.load(f)
