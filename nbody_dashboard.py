# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:25:04 2024

@author: Manny Admin
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from model import NBodyGNN  # Assuming your model is in `model.py`

# Set up logging
import logging
import requests
logging.basicConfig(level=logging.DEBUG)

# Function to download files from GitHub
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"File downloaded from {url} to {save_path}")
    else:
        st.error(f"Error downloading file from {url}. Status code: {response.status_code}")
        logging.error(f"Error downloading file from {url}. Status code: {response.status_code}")

# Page configuration
st.set_page_config(page_title="N-Body Simulation Dashboard", layout="wide")

# Title and introduction
st.title("N-Body Simulation Dashboard")
st.markdown("""
    Welcome to the N-Body Simulation Dashboard! This dashboard allows you to visualize 
    the predicted and actual positions of celestial bodies based on our Graph Neural Network (GNN) model.
""")

# Load the trained model
model = NBodyGNN()
model.eval()

# Load normalized data
# Option 1: Uploading data manually
uploaded_positions = st.file_uploader("Upload positions .npy file", type="npy")
uploaded_velocities = st.file_uploader("Upload velocities .npy file", type="npy")

if uploaded_positions is not None and uploaded_velocities is not None:
    try:
        # Load the uploaded files
        positions = np.load(uploaded_positions)
        velocities = np.load(uploaded_velocities)
        # Normalize data
        positions = (positions - positions.mean()) / positions.std()
        velocities = (velocities - velocities.mean()) / velocities.std()
        logging.info("Files uploaded and loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the .npy files: {e}")
        logging.error(f"Error loading .npy files: {e}")
else:
    # Option 2: Download files from GitHub if not uploaded
    st.markdown("Alternatively, attempting to download default data from GitHub.")
    try:
        positions_url = "https://raw.githubusercontent.com/manny-umassd/nbody-undergrad-project/main/data/simulations/positions_over_time.npy"
        velocities_url = "https://raw.githubusercontent.com/manny-umassd/nbody-undergrad-project/main/data/simulations/velocities_over_time.npy"
        
        download_file(positions_url, 'positions_over_time.npy')
        download_file(velocities_url, 'velocities_over_time.npy')
        
        # Load the downloaded files
        positions = np.load('positions_over_time.npy')
        velocities = np.load('velocities_over_time.npy')
        
        # Normalize data
        positions = (positions - positions.mean()) / positions.std()
        velocities = (velocities - velocities.mean()) / velocities.std()
        logging.info("Downloaded files loaded successfully.")
    except Exception as e:
        st.error(f"An unexpected error occurred while downloading/loading files: {e}")
        logging.error(f"Error downloading/loading .npy files from GitHub: {e}")

# Dummy data for demonstration if no files are uploaded
if 'positions' not in locals():
    st.warning("Using dummy data for demonstration. Please upload .npy files for actual results.")
    positions = np.random.rand(100, 3)
    velocities = np.random.rand(100, 3)

# Define the number of bodies and input dimension
n_bodies = 3  # Number of celestial bodies
input_dim = 7  # [position (3), velocity (3), mass (1)]

# Create Data object for prediction (dummy for demo purposes)
data = Data(
    x=torch.tensor(positions, dtype=torch.float),
    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    edge_attr=torch.tensor(velocities, dtype=torch.float)
)

# Make predictions with the model
try:
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr)
        next_positions = predictions[:, :3].detach().numpy()

        # Plotting the predicted vs. actual positions
        fig, ax = plt.subplots()
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Actual Positions', alpha=0.6)
        ax.scatter(next_positions[:, 0], next_positions[:, 1], c='red', label='Predicted Positions', alpha=0.6)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.set_title("Actual vs. Predicted Positions of Bodies")

        # Display plot
        st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred while predicting with the model: {e}")
    logging.error(f"Prediction error: {e}")

# Future Trajectories Visualization (optional)
try:
    fig2, ax2 = plt.subplots()
    for i in range(n_bodies):
        ax2.plot(next_positions[:, 0], next_positions[:, 1], label=f"Predicted Trajectory Body {i+1}")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.legend()
    ax2.set_title("Predicted Future Trajectories of Bodies")

    # Display plot
    st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred while visualizing future trajectories: {e}")
    logging.error(f"Trajectory visualization error: {e}")

