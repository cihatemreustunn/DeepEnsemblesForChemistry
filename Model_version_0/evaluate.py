import numpy as np

def evaluate_model(model, X_test, Y_test):
    mean, variance = model.predict(X_test)
    mse = np.mean((Y_test - mean)**2)
    uncertainty = np.mean(np.sqrt(variance))
    return mse, uncertainty