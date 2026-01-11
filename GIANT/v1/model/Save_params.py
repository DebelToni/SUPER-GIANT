import pickle
def save_params(params, fname="model_params.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(params, f)

