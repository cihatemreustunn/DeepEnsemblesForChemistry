from data_preprocessing_0 import data_import
from deep_ensemble_0 import DeepEnsemble
import tensorflow as tf

def train_model(learning_rate=0.0001, n_models=5, train_model=True):
    # Get the preprocessed data
    X_train, X_val, Y_train, Y_val, X_heldout, Y_heldout = data_import()
    
    if not train_model:
        return None, (X_val, Y_val, X_heldout, Y_heldout), None
    
    # Create dataset from training data
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(32)
    
    # Create and train ensemble with specified learning rate
    ensemble = DeepEnsemble(n_models=n_models, learning_rate=learning_rate)
    ensemble.compile()
    histories = ensemble.fit(dataset)
    
    return ensemble, (X_val, Y_val, X_heldout, Y_heldout), histories