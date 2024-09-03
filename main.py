from train import train_model
from data_preprocessing import generate_data
from evaluate import evaluate_model
from visualize import plot_results
import numpy as np

def main():
    # Train the model
    ensemble = train_model()

    # Generate test data
    t_test = np.linspace(0, 10, 100)
    x_test = np.exp(-0.1 * t_test) * np.cos(t_test)
    v_test = -0.1 * np.exp(-0.1 * t_test) * np.cos(t_test) - np.exp(-0.1 * t_test) * np.sin(t_test)
    X_test = np.column_stack((x_test, v_test))

    # Evaluate the model
    mean, variance = ensemble.predict(X_test)
    mse, uncertainty = evaluate_model(ensemble, X_test, np.column_stack((x_test, v_test)))

    print(f'Mean Squared Error: {mse}')
    print(f'Average Uncertainty (std dev): {uncertainty}')

    # Visualize the results
    plot_results(t_test, x_test, mean, variance)

if __name__ == "__main__":
    main()

