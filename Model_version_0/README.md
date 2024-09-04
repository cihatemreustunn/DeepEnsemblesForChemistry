# Deep Ensemble for ODE Integration

This project implements a Deep Ensemble model for integrating Ordinary Differential Equations (ODEs).

## Project Structure

The project is organized as follows:


- `data_preprocessing.py`: Data preprocessing and generation
- `base_model.py`: Model definitions
- `deep_ensemble.py`: Model definitions
- `train.py`: Model training script
- `evaluate.py`: Model evaluation script
- `visualize.py`: Plotting and visualization tools
- `main.py`: The main script to run the entire pipeline

## Setup and Running

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```
   python main.py
   ```

This will train the Deep Ensemble, evaluate it on test data, and generate a plot of the results.

## Extending the Project

To adapt this project for your specific ODE system:

1. Modify the `generate_data` function in `data_preprocessing.py`.
2. Adjust the model architecture in `base_model.py` if needed.
3. Update the evaluation metrics in `evaluate.py` as appropriate for your system.

