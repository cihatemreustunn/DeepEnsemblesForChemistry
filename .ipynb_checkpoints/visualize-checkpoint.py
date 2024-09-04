import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font', family='serif', serif=['Computer Modern Roman'], size=22)
rc('text', usetex=True)

def plot_results(t_test, x_test, mean, variance):
    plt.figure(figsize=(12, 6))
    plt.plot(t_test, x_test, 'b-', label='True x')
    plt.plot(t_test, mean[:, 0], 'r--', label='Predicted x')
    plt.fill_between(t_test, mean[:, 0] - 2*np.sqrt(variance[:, 0]), 
                     mean[:, 0] + 2*np.sqrt(variance[:, 0]), color='r', alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Position (x)')
    plt.legend()
    plt.show()