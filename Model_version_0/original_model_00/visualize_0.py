import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font', family='serif', serif=['Computer Modern Roman'], size=22)
rc('text', usetex=True)

def plot_results(t_test, x_test, mean, variance):
    plt.figure(figsize=(12, 6))
    plt.plot(t_test, x_test[:, 0], 'b-', label='True x')
    plt.plot(t_test, mean[:, 0], 'r--', label='Predicted x')
    plt.fill_between(t_test, mean[:, 0] - 2*np.sqrt(variance[:, 0]), 
                     mean[:, 0] + 2*np.sqrt(variance[:, 0]), color='r', alpha=0.2)
    plt.xlabel('Time') 
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.show()

def plot_iterative_results(predictions, variances=None, dt=1):
    """
    Plot the results of iterative predictions for 9D state.
    
    Args:
        predictions: numpy array of shape (n_steps, 9) containing state predictions
        variances: numpy array of shape (n_steps, 9) containing variance estimates
        dt: time step size for creating time array
    """
    # Create time array
    t = np.arange(len(predictions)) * dt
    
    # Calculate number of rows needed for 9 subplots (3x3 grid)
    n_rows = 3
    n_cols = 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each state variable
    for i in range(9):
        ax = axes_flat[i]
        
        # Plot predictions
        ax.plot(t, predictions[:, i], 'b-', label=f'State {i+1}')
        
        # Add uncertainty bounds if available
        if variances is not None:
            ax.fill_between(t,
                          predictions[:, i] - 2*np.sqrt(variances[:, i]),
                          predictions[:, i] + 2*np.sqrt(variances[:, i]),
                          color='b', alpha=0.2)
        
        # Customize subplot
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'State {i+1}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_state_correlation(predictions, state_indices, dt=0.1):
    """
    Plot correlation between two selected state variables.
    
    Args:
        predictions: numpy array of shape (n_steps, 9) containing state predictions
        state_indices: tuple of two integers specifying which states to plot
        dt: time step size for creating time array
    """
    i, j = state_indices
    
    plt.figure(figsize=(8, 8))
    plt.plot(predictions[:, i], predictions[:, j], 'b-')
    plt.xlabel(f'State {i+1}')
    plt.ylabel(f'State {j+1}')
    plt.title(f'State {i+1} vs State {j+1}')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_training_history(histories):
    """
    Plot training histories for all models in the ensemble.

    Args:
        histories: list of training histories from model.fit()
    """
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Model {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()