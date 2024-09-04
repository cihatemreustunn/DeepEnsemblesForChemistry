import numpy as np
from base_model import BaseModel
import tensorflow as tf

class DeepEnsemble:
    def __init__(self, n_models=10):
        self.n_models = n_models
        self.models = [BaseModel() for _ in range(n_models)]

    def compile(self, optimizer='adam', loss='mse'):
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss)

    def fit(self, dataset, epochs=500, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)]):
        for model in self.models:
            model.fit(dataset, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        return mean, variance