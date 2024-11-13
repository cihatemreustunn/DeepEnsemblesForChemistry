import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

def data_import(validation_size=0.2, random_state=42):
    """
    Import data and split into train and validation sets.
    
    Args:
        validation_size (float): Proportion of data to use for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (X_train, X_val, Y_train, Y_val, X_heldout, Y_heldout)
    """
    input_data = pd.read_csv("/Users/cihatemreustun/Downloads/DeepEnsemblesForChemistry/TRAINING_50K_input.csv")
    output_data = pd.read_csv("/Users/cihatemreustun/Downloads/DeepEnsemblesForChemistry/TRAINING_50K_output.csv")
    feature_cols = ["T", "Y_h", "Y_h2", "Y_o", "Y_o2", "Y_oh", "Y_h2o", "Y_ho2", "Y_h2o2"]
    X = input_data[feature_cols].to_numpy()
    Y = output_data[feature_cols].to_numpy()
    
    heldout = pd.read_csv("/Users/cihatemreustun/Downloads/DeepEnsemblesForChemistry/paper-test.csv").to_numpy()
    X_heldout = heldout[:-1]
    Y_heldout = heldout[1:]
    
    # Shuffle the data
    X, Y = shuffle(X, Y, random_state=random_state)
    
    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, 
        test_size=validation_size,
        random_state=random_state
    )
    
    return X_train, X_val, Y_train, Y_val, X_heldout, Y_heldout



