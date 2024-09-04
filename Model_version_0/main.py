from train import train_model
from visualize import plot_iterative_results, plot_state_correlation
from deep_ensemble import DeepEnsemble
from data_preprocessing import data_import
import os

def main():

    X, X_next, X_test, Y_test, t_test = data_import()
    # Define paths
    saved_model_path = 'my_saved_ensemble'

    # Check for existing model
    if not os.path.exists(saved_model_path):
        print("No existing model found. Training new ensemble...")
        try:
            # Train the model using train_model()
            ensemble = train_model()
            
            # Save the trained ensemble
            print("Training complete. Saving model...")
            ensemble.save_models(saved_model_path)
            print(f"Model saved to {saved_model_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return
    else:
        # Load the trained model
        print("Loading existing ensemble...")
        try:
            ensemble = DeepEnsemble.load_models(saved_model_path)
            print("Ensemble loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return

    # Define initial condition (9-dimensional state)
    initial_state = X[0, :]
    
    # Perform iterative predictions
    n_steps = 500
    print(f"Performing iterative predictions for {n_steps} steps...")
    try:
        predictions, variances = ensemble.predict_iterative(initial_state, n_steps=n_steps)
        
        # Visualize the results
        print("Plotting results...")
        plot_iterative_results(predictions, variances, dt=0.1)
        plot_state_correlation(predictions, (0, 1))
        
    except Exception as e:
        print(f"Error during prediction or plotting: {str(e)}")

if __name__ == "__main__":
    main()