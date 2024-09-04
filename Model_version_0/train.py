import tensorflow as tf
from data_preprocessing import data_import
from deep_ensemble import DeepEnsemble
    
def train_model():
    X, X_next, X_test, Y_test, t_test = data_import()
    dataset = tf.data.Dataset.from_tensor_slices((X, X_next)).shuffle(47405).batch(32)
    ensemble = DeepEnsemble(n_models=10)
    ensemble.compile()
    ensemble.fit(dataset)

    return ensemble