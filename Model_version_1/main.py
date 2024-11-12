# main.py
from train import train_model
from visualize import plot_iterative_results, plot_dimension_performance
from deep_ensemble import MultiOutputDeepEnsemble
from data_preprocessing import data_import
from evaluate import evaluate_model
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Import data with split outputs for training
    X, deltas_split, X_test, test_deltas_split, _ = data_import(split_outputs=True)
    
    # Reconstruct full test data for plotting (X_test + test_deltas = Y_test)
    Y_test = X_test + np.hstack([d.reshape(-1, 1) for d in test_deltas_split])
    
    # Define paths
    saved_model_path = 'multi_dim_ensemble'

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
            ensemble = MultiOutputDeepEnsemble.load_models(saved_model_path)
            print("Ensemble loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return

    # Evaluate model performance for each dimension
    print("\nEvaluating model performance...")
    mse_per_dim, uncertainty_per_dim = evaluate_model(ensemble, X_test, test_deltas_split)
    
    # Print performance metrics for each dimension
    feature_names = ["T", "Y_h", "Y_h2", "Y_o", "Y_o2", "Y_oh", "Y_h2o", "Y_ho2", "Y_h2o2"]
    print("\nPer-dimension performance metrics:")
    print("Dimension    | MSE        | Uncertainty")
    print("-" * 45)
    for i, (mse, uncertainty) in enumerate(zip(mse_per_dim, uncertainty_per_dim)):
        print(f"{feature_names[i]:<11} | {mse:.2e} | {uncertainty:.2e}")

    # Define initial condition (9-dimensional state)
    initial_state = X_test[0, :]  # Use first test point as initial condition
    
    # Define number of steps
    n_steps = 500  # You can adjust this number as needed
    
    print(f"\nPerforming iterative predictions for {n_steps} steps...")
    try:
        predictions, variances = ensemble.predict_iterative(initial_state, n_steps=n_steps)
        
        # Ensure test data matches prediction length for plotting
        test_data_plot = Y_test[:n_steps] if len(Y_test) > n_steps else Y_test
        
        # Visualize the results
        print("Plotting results...")
        
        # Plot time evolution of all states with test data
        plot_iterative_results(predictions, variances, dt=0.1, 
                             feature_names=feature_names,
                             test_data=test_data_plot)
        
        # Plot dimension-wise performance comparison
        plot_dimension_performance(mse_per_dim, uncertainty_per_dim, feature_names)
        
    except Exception as e:
        print(f"Error during prediction or plotting: {str(e)}")

if __name__ == "__main__":
    main()