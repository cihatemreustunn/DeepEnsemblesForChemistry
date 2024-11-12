import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def data_import(split_outputs=True):
    """
    Import data and compute deltas for both training and test data.
    """
    input_data = pd.read_csv("TRAINING_50K_input.csv")
    output_data = pd.read_csv("TRAINING_50K_output.csv")
    feature_cols = ["T", "Y_h", "Y_h2", "Y_o", "Y_o2", "Y_oh", "Y_h2o", "Y_ho2", "Y_h2o2"]
    
    # Import data
    X = input_data[feature_cols].to_numpy()
    X_next = output_data[feature_cols].to_numpy()
    
    # Compute training deltas
    deltas = X_next - X
    
    # Import test data
    test = pd.read_csv("paper-test.csv").to_numpy()
    X_test = test[:-1]
    Y_test = test[1:]
    
    # Compute test deltas
    test_deltas = Y_test - X_test
    
    # Shuffle training data
    X, deltas = shuffle(X, deltas)
    
    if split_outputs:
        # Split deltas into list of arrays, one for each dimension
        deltas_split = [deltas[:, i:i+1] for i in range(deltas.shape[1])]
        test_deltas_split = [test_deltas[:, i:i+1] for i in range(test_deltas.shape[1])]
        
        return X, deltas_split, X_test, test_deltas_split, None
    
    return X, deltas, X_test, test_deltas, None