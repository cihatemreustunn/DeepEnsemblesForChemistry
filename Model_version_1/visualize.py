
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font', family='serif', serif=['Computer Modern Roman'], size=12)
rc('text', usetex=True)

def plot_iterative_results(predictions, variances=None, dt=0.1, feature_names=None, test_data=None):
    """
    Plot the results of iterative predictions for 9D state with test data comparison.
    
    Args:
        predictions: numpy array of shape (n_steps, 9) containing state predictions
        variances: numpy array of shape (n_steps, 9) containing variance estimates
        dt: time step size for creating time array
        feature_names: list of strings with feature names
        test_data: numpy array of shape (n_steps, 9) containing test data
    """
    if feature_names is None:
        feature_names = [f"State {i+1}" for i in range(predictions.shape[1])]
    
    # Create time array
    t = np.arange(len(predictions)) * dt
    
    # Calculate number of rows needed for 9 subplots (3x3 grid)
    n_rows = 3
    n_cols = 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    fig.suptitle('Time Evolution of State Variables', fontsize=16)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each state variable
    for i in range(9):
        ax = axes_flat[i]
        
        # Plot predictions
        ax.plot(t, predictions[:, i], 'b-', label='Prediction')
        
        # Plot test data if available
        if test_data is not None:
            t_test = np.arange(len(test_data)) * dt
            ax.plot(t_test, test_data[:, i], 'r--', label='Test Data')
        
        # Add uncertainty bounds if available
        if variances is not None:
            ax.fill_between(t,
                          predictions[:, i] - 2*np.sqrt(variances[:, i]),
                          predictions[:, i] + 2*np.sqrt(variances[:, i]),
                          color='b', alpha=0.2, label='95% Confidence')
        
        # Customize subplot
        ax.set_xlabel('Time')
        ax.set_ylabel(feature_names[i])
        ax.set_title(f'{feature_names[i]} Evolution')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_dimension_performance(mse_per_dim, uncertainty_per_dim, feature_names):
    """Plot performance metrics for each dimension."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot MSE
    ax1.bar(feature_names, mse_per_dim)
    ax1.set_yscale('log')
    ax1.set_xlabel('State Variable')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE by Dimension')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot Uncertainty
    ax2.bar(feature_names, uncertainty_per_dim)
    ax2.set_yscale('log')
    ax2.set_xlabel('State Variable')
    ax2.set_ylabel('Average Uncertainty')
    ax2.set_title('Uncertainty by Dimension')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()