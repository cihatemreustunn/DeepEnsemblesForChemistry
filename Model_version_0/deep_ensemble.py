import numpy as np
from base_model import BaseModel
import tensorflow as tf
import os

class DeepEnsemble:
    def __init__(self, n_models=10, model_dir='saved_models'):
        self.n_models = n_models
        self.models = [BaseModel() for _ in range(n_models)]
        self.model_dir = model_dir
        self.state_dim = 9  # Define state dimension

    def compile(self, optimizer='adam', loss='mse'):
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss)

    def fit(self, dataset, epochs=500, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)]):
        for i, model in enumerate(self.models):
            print(f'Training model {i+1}/{self.n_models}')
            model.fit(dataset, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        return mean, variance
    
    def predict_iterative(self, initial_state, n_steps, include_variance=True):
        """
        Perform iterative predictions starting from an initial state.
        
        Args:
            initial_state: numpy array of shape (9,) containing the state variables
            n_steps: number of time steps to predict
            include_variance: whether to return variance estimates
            
        Returns:
            predictions: numpy array of shape (n_steps, 9) containing state predictions
            variances: numpy array of shape (n_steps, 9) containing variance estimates
                      (only if include_variance=True)
        """
        # Verify input dimension
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"Initial state must have shape ({self.state_dim},), got {initial_state.shape}")
        
        # Initialize arrays to store predictions and variances
        predictions = np.zeros((n_steps, self.state_dim))
        if include_variance:
            variances = np.zeros((n_steps, self.state_dim))
        
        # Set initial state
        current_state = initial_state.reshape(1, -1)  # Reshape to (1, 9) for model input
        
        # Perform iterative predictions
        for step in range(n_steps):
            # Get prediction and variance for current state
            mean, variance = self.predict(current_state)
            
            # Store results
            predictions[step] = mean[0]  # Remove batch dimension
            if include_variance:
                variances[step] = variance[0]
            
            # Update current state for next prediction
            current_state = mean
        
        if include_variance:
            return predictions, variances
        return predictions
    
    def save_models(self, custom_dir=None):
        """Save all models in the ensemble"""
        save_dir = custom_dir if custom_dir else self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'model_{i}')
            model.save(model_path)
        
        metadata = {
            'n_models': self.n_models,
            'model_dir': save_dir,
            'state_dim': self.state_dim
        }
        np.save(os.path.join(save_dir, 'metadata.npy'), metadata)
        
        print(f'Ensemble saved to {save_dir}')

    @classmethod
    def load_models(cls, model_dir):
        """Load a previously saved ensemble"""
        if not os.path.exists(model_dir):
            raise ValueError(f"Directory {model_dir} does not exist")
            
        metadata = np.load(os.path.join(model_dir, 'metadata.npy'), allow_pickle=True).item()
        ensemble = cls(n_models=metadata['n_models'], model_dir=model_dir)
        
        ensemble.models = []
        for i in range(metadata['n_models']):
            model_path = os.path.join(model_dir, f'model_{i}')
            model = tf.keras.models.load_model(model_path)
            ensemble.models.append(model)
            
        print(f'Loaded ensemble from {model_dir}')
        return ensemble