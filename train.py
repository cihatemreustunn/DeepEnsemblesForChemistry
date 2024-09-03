import tensorflow as tf
from data_preprocessing import generate_data
from deep_ensemble import DeepEnsemble

def train_model():
    X, X_next = generate_data()
    dataset = tf.data.Dataset.from_tensor_slices((X, X_next)).shuffle(1000).batch(32)

    ensemble = DeepEnsemble(n_models=5)
    ensemble.compile()
    ensemble.fit(dataset)

    return ensemble
    