import tensorflow as tf
from data_preprocessing import generate_data
from deep_ensemble import DeepEnsemble
    
def train_model():
    dataset = tf.data.Dataset.from_tensor_slices((X, X_next)).shuffle(47405).batch(32)
    ensemble = DeepEnsemble(n_models=10)
    ensemble.compile()
    ensemble.fit(dataset)

    return ensemble