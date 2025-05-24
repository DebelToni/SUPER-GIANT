import pickle
def save_params(params):
    with open("model_params.pkl", "wb") as f:
        pickle.dump(params, f)
