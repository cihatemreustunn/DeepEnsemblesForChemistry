import numpy as np

def generate_data(n_samples=1000, noise_std=0.05):
    t = np.linspace(0, 10, n_samples)
    x = np.exp(-0.1 * t) * np.cos(t)
    v = -0.1 * np.exp(-0.1 * t) * np.cos(t) - np.exp(-0.1 * t) * np.sin(t)
    X = np.column_stack((x, v))
    X_next = np.column_stack((x[1:], v[1:]))
    X = X[:-1] + np.random.normal(0, noise_std, X[:-1].shape)
    return X, X_next




