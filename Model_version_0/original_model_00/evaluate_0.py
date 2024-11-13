import numpy as np

def evaluate_model(model, X_heldout, Y_heldout):
    mean, variance = model.predict(X_heldout)
    mse = np.mean((Y_heldout - mean)**2)
    uncertainty = np.mean(np.sqrt(variance))
    return mse, uncertainty