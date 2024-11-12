# deep_ensemble.py
import numpy as np
from base_model import BaseModel
import tensorflow as tf
import os

class MultiOutputDeepEnsemble:
    def __init__(self, n_models=5, state_dim=9, model_dir='saved_models'):
        self.n_models = n_models
        self.state_dim = state_dim
        self.model_dir = model_dir
        
        # Create ensembles
        self.ensembles = [
            [BaseModel(output_dim=1, seed=i*j) 
             for j in range(n_models)]
            for i in range(state_dim)
        ]

    def compile(self, optimizer='adam', loss='mse'):
        for ensemble in self.ensembles:
            for model in ensemble:
                model.compile(optimizer=optimizer, loss=loss)

    def fit(self, dataset, epochs=700, verbose=1):
        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=30,
            restore_best_weights=True
        )
        
        # Unpack dataset
        X, delta_split = dataset
        
        for dim in range(self.state_dim):
            print(f'\nTraining ensemble for dimension {dim+1}/{self.state_dim}')
            # Use the pre-split delta data for this dimension
            delta_dim = delta_split[dim]
            
            # Convert to tf.data.Dataset with prefetch
            dim_dataset = tf.data.Dataset.from_tensor_slices((X, delta_dim))\
                .shuffle(len(X))\
                .batch(128)\
                .prefetch(tf.data.AUTOTUNE)
            
            # Train each model in the ensemble for this dimension
            for i, model in enumerate(self.ensembles[dim]):
                print(f'Training model {i+1}/{self.n_models}')
                model.fit(
                    dim_dataset,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=[early_stopping]
                )

    def predict(self, X):
        # Ensure X is the right shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Initialize arrays for deltas and variances
        delta_predictions = np.zeros((self.state_dim, len(X)))
        variances = np.zeros((self.state_dim, len(X)))
        
        # Get predictions from each ensemble
        for dim in range(self.state_dim):
            dim_predictions = []
            for model in self.ensembles[dim]:
                pred = model(X).numpy()
                dim_predictions.append(pred.flatten())
            
            dim_predictions = np.array(dim_predictions)
            delta_predictions[dim] = np.mean(dim_predictions, axis=0)
            variances[dim] = np.var(dim_predictions, axis=0)
        
        # Reshape predictions and variances
        delta_predictions = delta_predictions.T
        variances = variances.T
        
        # Convert deltas to actual predictions
        predictions = X + delta_predictions
        
        return predictions, variances

    def predict_iterative(self, initial_state, n_steps, include_variance=True):
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"Initial state must have shape ({self.state_dim},), got {initial_state.shape}")
        
        # Initialize arrays
        predictions = np.zeros((n_steps, self.state_dim))
        if include_variance:
            variances = np.zeros((n_steps, self.state_dim))
        
        current_state = initial_state.reshape(1, -1)
        
        # Perform iterative prediction
        for step in range(n_steps):
            # Get prediction for current state
            next_state, variance = self.predict(current_state)
            
            # Store results
            predictions[step] = next_state[0]
            if include_variance:
                variances[step] = variance[0]
            
            # Update current state
            current_state = next_state
        
        if include_variance:
            return predictions, variances
        return predictions

    def save_models(self, custom_dir=None):
        save_dir = custom_dir if custom_dir else self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for dim in range(self.state_dim):
            dim_dir = os.path.join(save_dir, f'dimension_{dim}')
            os.makedirs(dim_dir, exist_ok=True)
            
            for i, model in enumerate(self.ensembles[dim]):
                model_path = os.path.join(dim_dir, f'model_{i}')
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
        if not os.path.exists(model_dir):
            raise ValueError(f"Directory {model_dir} does not exist")
            
        metadata = np.load(os.path.join(model_dir, 'metadata.npy'), allow_pickle=True).item()
        ensemble = cls(n_models=metadata['n_models'], state_dim=metadata['state_dim'], model_dir=model_dir)
        
        ensemble.ensembles = []
        for dim in range(metadata['state_dim']):
            dim_ensemble = []
            dim_dir = os.path.join(model_dir, f'dimension_{dim}')
            
            for i in range(metadata['n_models']):
                model_path = os.path.join(dim_dir, f'model_{i}')
                model = tf.keras.models.load_model(model_path)
                dim_ensemble.append(model)
            
            ensemble.ensembles.append(dim_ensemble)
            
        print(f'Loaded ensemble from {model_dir}')
        return ensemble