import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

def data_import():
    
    input_data = pd.read_csv("TRAINING_50K_input.csv")
    output_data = pd.read_csv("TRAINING_50K_output.csv")
    feature_cols = ["T", "Y_h", "Y_h2", "Y_o", "Y_o2", "Y_oh", "Y_h2o", "Y_ho2", "Y_h2o2"]
    X = input_data[feature_cols].to_numpy()
    X_next = output_data[feature_cols].to_numpy()
    
    test = pd.read_csv("paper-test.csv").to_numpy()
    X_test = test[:-1]
    Y_test = test[1:]
    t_test = np.linspace(0, 2500, 1)
    
    X, X_next = shuffle(X, X_next)
    
    return X, X_next, X_test, Y_test, t_test



