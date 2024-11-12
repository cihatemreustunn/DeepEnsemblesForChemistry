import numpy as np

def evaluate_model(model, X_test, test_deltas_split):
    """
    Evaluate model performance for each dimension.
    
    Args:
        model: MultiOutputDeepEnsemble instance
        X_test: Test input data
        test_deltas_split: List of arrays, each containing test deltas for one dimension
    
    Returns:
        mse_per_dim: List of MSE values for each dimension
        uncertainty_per_dim: List of uncertainty values for each dimension
    """
    # Get predicted deltas
    predictions, variance = model.predict(X_test)
    predicted_deltas = predictions - X_test  # Convert predictions back to deltas
    
    mse_per_dim = []
    uncertainty_per_dim = []
    
    for dim in range(model.state_dim):
        # Compare predicted deltas with actual deltas
        mse = np.mean((test_deltas_split[dim] - predicted_deltas[:, dim:dim+1])**2)
        uncertainty = np.mean(np.sqrt(variance[:, dim:dim+1]))
        
        mse_per_dim.append(mse)
        uncertainty_per_dim.append(uncertainty)
    
    return mse_per_dim, uncertainty_per_dim