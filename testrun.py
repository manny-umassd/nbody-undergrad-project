import logging
import numpy as np
import torch
from model import NBodyGNN

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log start of script execution
logging.info("Starting the N-Body Simulation script.")

try:
    # Load preprocessed data
    logging.info("Attempting to load X_sequences.npy...")
    X = np.load('/workspaces/nbody-undergrad-project/data/simulations/X_sequences.npy')
    logging.info("Successfully loaded X_sequences.npy")

    logging.info("Attempting to load y_targets.npy...")
    y = np.load('/workspaces/nbody-undergrad-project/data/simulations/y_targets.npy')
    logging.info("Successfully loaded y_targets.npy")

except Exception as e:
    logging.error(f"Error loading data files: {e}")
    raise

# Log model initialization
try:
    logging.info("Initializing NBodyGNN model...")
    model = NBodyGNN(input_dim=X.shape[-1], hidden_dim=128, output_dim=y.shape[-1], n_bodies=X.shape[1])
    logging.info("Model initialized successfully.")

except Exception as e:
    logging.error(f"Error initializing model: {e}")
    raise

# Example: Logging data shape
logging.debug(f"Shape of X: {X.shape}")
logging.debug(f"Shape of y: {y.shape}")

# Forward pass through the model
try:
    logging.info("Performing forward pass...")
    output = model(torch.tensor(X, dtype=torch.float32))
    logging.info("Forward pass completed successfully.")

    # Log the shape of the output
    logging.debug(f"Shape of output: {output.shape}")

except Exception as e:
    logging.error(f"Error during forward pass: {e}")
    raise

# Log script completion
logging.info("N-Body Simulation script completed.")
