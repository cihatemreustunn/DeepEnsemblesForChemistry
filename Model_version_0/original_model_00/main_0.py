from train_0 import train_model
from evaluate_0 import evaluate_model
from visualize_0 import plot_results, plot_iterative_results, plot_training_history
from deep_ensemble_0 import DeepEnsemble
import numpy as np
import os

def main():
    # Training parameters
    learning_rate = 0.0001
    n_models = 5
    model_dir = 'saved_ensemble'
    
    # Check if saved model exists
    if os.path.exists(model_dir):
        print("Loading existing ensemble...")
        ensemble = DeepEnsemble.load_models(model_dir)
        # We still need to get the validation/heldout data for evaluation
        _, (X_val, Y_val, X_heldout, Y_heldout), _ = train_model(
            learning_rate=learning_rate,
            n_models=n_models,
            train_model=False  # Add this parameter to train_0.py
        )
        histories = None
    else:
        print("Training new ensemble...")
        # Train the model and get validation/heldout data
        ensemble, (X_val, Y_val, X_heldout, Y_heldout), histories = train_model(
            learning_rate=learning_rate,
            n_models=n_models
        )
        # Save the trained ensemble
        ensemble.save_models(model_dir)
        print(f"Ensemble saved to {model_dir}")

    # Plot training histories if available (only for newly trained models)
    if histories is not None:
        plot_training_history(histories)

    # Evaluate on validation set
    val_mse, val_uncertainty = evaluate_model(ensemble, X_val, Y_val)
    print(f'Validation Mean Squared Error: {val_mse}')
    print(f'Validation Average Uncertainty: {val_uncertainty}')

    # Evaluate on heldout set
    heldout_mse, heldout_uncertainty = evaluate_model(ensemble, X_heldout, Y_heldout)
    print(f'Heldout Mean Squared Error: {heldout_mse}')
    print(f'Heldout Average Uncertainty: {heldout_uncertainty}')

    # Make iterative predictions starting from first heldout state
    n_steps = 250
    predictions, variances = ensemble.predict_iterative(X_heldout[0], n_steps)
    
    # Visualize the results
    plot_iterative_results(predictions, variances, dt=1)

if __name__ == "__main__":
    main()