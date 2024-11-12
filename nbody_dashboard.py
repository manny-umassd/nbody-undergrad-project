import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

# Load trained model
from model import NBodyGNN  # Make sure this is your GNN model

n_bodies = 3  # Define the number of bodies here
input_dim = 7  # [position (3), velocity (3), mass (1)]
hidden_dim = 128  # Same as during training
output_dim = n_bodies * 6  # [next position (3) + next velocity (3) for each body]

model = NBodyGNN(input_dim, hidden_dim, output_dim, n_bodies)
model.load_state_dict(torch.load('gnn_model_with_trajectories.pth'))
model.eval()

# Load normalized data
positions = np.load('data/positions_over_time.npy')
velocities = np.load('data/velocities_over_time.npy')
masses = np.load('data/masses.npy')

# Sidebar: File Inputs
st.sidebar.header('Simulation Data')
positions_file = st.sidebar.file_uploader("Upload Positions File", type=["npy"])
velocities_file = st.sidebar.file_uploader("Upload Velocities File", type=["npy"])
masses_file = st.sidebar.file_uploader("Upload Masses File", type=["npy"])

if positions_file is not None and velocities_file is not None and masses_file is not None:
    positions = np.load(positions_file)
    velocities = np.load(velocities_file)
    masses = np.load(masses_file)

# Title
st.title("N-Body Simulation: GNN Prediction Dashboard")

# Training and Validation Loss Plot
st.header("Training and Validation Loss Over Epochs")
fig, ax = plt.subplots()
losses = np.load('data/losses.npy')  # Replace with actual file path for losses
validation_losses = np.load('data/validation_losses.npy')  # Replace with actual file path for validation losses
ax.plot(range(1, len(losses) + 1), losses, label='Training Loss')
ax.plot(range(1, len(validation_losses) + 1), validation_losses, linestyle='--', label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training and Validation Loss Over Epochs')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Actual vs Predicted Positions
st.header("Actual vs. Predicted Positions of Bodies")
with torch.no_grad():
    test_positions = positions[-1]  # Use the last time step for testing
    test_velocities = velocities[-1]
    test_node_features = []
    for i in range(n_bodies):
        test_features = np.concatenate((test_positions[i], test_velocities[i], [masses[i]]))
        test_node_features.append(test_features)
    test_x = torch.tensor(np.array(test_node_features), dtype=torch.float32)
    edge_index = torch.tensor(
        [[i, j] for i in range(n_bodies) for j in range(n_bodies) if i != j],
        dtype=torch.long
    ).t().contiguous()
    test_data = Data(x=test_x, edge_index=edge_index)
    test_predicted = model(test_data).view(n_bodies, 6)
    predicted_positions = test_predicted[:, :3].detach().numpy()

fig, ax = plt.subplots()
for i in range(n_bodies):
    ax.scatter(test_positions[i][0], test_positions[i][1], color='blue', label=f'Actual Body {i+1}' if i == 0 else "")
    ax.scatter(predicted_positions[i][0], predicted_positions[i][1], color='red', marker='x', label=f'Predicted Body {i+1}' if i == 0 else "")
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Actual vs. Predicted Positions of Bodies')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Predicted Future Trajectories
st.header("Predicted Future Trajectories of Bodies")
future_steps = st.slider("Number of Future Steps to Predict", 1, 20, 10)

future_positions = [test_positions]
future_velocities = [test_velocities]

for step in range(future_steps):
    test_node_features = []
    for i in range(n_bodies):
        test_features = np.concatenate((future_positions[-1][i], future_velocities[-1][i], [masses[i]]))
        test_node_features.append(test_features)
    test_x = torch.tensor(np.array(test_node_features), dtype=torch.float32)
    test_data = Data(x=test_x, edge_index=edge_index)
    test_predicted = model(test_data).view(n_bodies, 6)

    next_positions = test_predicted[:, :3].detach().numpy()
    next_velocities = test_predicted[:, 3:].detach().numpy()

    future_positions.append(next_positions)
    future_velocities.append(next_velocities)

fig, ax = plt.subplots()
for i in range(n_bodies):
    actual_traj = np.array([pos[i] for pos in future_positions])
    ax.plot(actual_traj[:, 0], actual_traj[:, 1], label=f'Predicted Trajectory Body {i+1}')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Predicted Future Trajectories of Bodies')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Evaluation Metrics
st.header("Evaluation Metrics")
mae = np.mean(np.abs(predicted_positions - test_positions))
rmse = np.sqrt(np.mean((predicted_positions - test_positions) ** 2))
ss_res = np.sum((predicted_positions - test_positions) ** 2)
ss_tot = np.sum((test_positions - np.mean(test_positions)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"R-Squared (R²): {r2_score:.4f}")
