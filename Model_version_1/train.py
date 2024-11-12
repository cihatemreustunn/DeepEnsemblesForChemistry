import tensorflow as tf
from data_preprocessing import data_import
from deep_ensemble import MultiOutputDeepEnsemble
    
def train_model():
    # Get data with split deltas
    X, deltas_split, X_test, test_deltas_split, _ = data_import(split_outputs=True)
    
    # Create and compile ensemble
    ensemble = MultiOutputDeepEnsemble(n_models=5)
    ensemble.compile()
    
    # Train ensemble with input states and target deltas
    ensemble.fit((X, deltas_split))
    
    return ensemble