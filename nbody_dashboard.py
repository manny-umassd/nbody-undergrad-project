import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.data import Data, Batch
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

# Define a function to download and load numpy files from a URL
def download_and_load_npy(url):
    response = requests.get(url)
    response.raise_for_status()
    try:
        return np.load(io.BytesIO(response.content), allow_pickle=True)
    except UnicodeDecodeError as e:
        print("Error decoding file. Checking problematic byte...")
        with io.BytesIO(response.content) as file:
            file.seek(e.start)
            byte = file.read(1)
            print(f"The byte at position {e.start} is: {byte}")
        raise

# Define GNN model with additional layers, attention mechanism, and dropout for regularization
class NBodyGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_bodies):
        super(NBodyGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)  # Attention mechanism
        self.fc = torch.nn.Linear(hidden_dim * n_bodies, output_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # Dropout for regularization
        self.residual = torch.nn.Linear(input_dim, hidden_dim)  # Residual connection for input
        
        # Weight initialization
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_res = self.residual(x)  # Residual connection
        x = self.conv1(x, edge_index)
        x = F.relu(x + x_res)  # Add residual and apply activation
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = x.view(1, -1)  # Flatten all node features per graph (assuming a single graph for prediction)
        out = self.fc(x)
        return out

# Load the trained GNN model
n_bodies = 3  # Define the number of bodies here
input_dim = 7  # [position (3), velocity (3), mass (1)]
hidden_dim = 128  # Increased hidden dimension for better learning capacity
output_dim = n_bodies * 6  # [next position (3) + next velocity (3) for each body]

model = NBodyGNN(input_dim, hidden_dim, output_dim, n_bodies)

# Initialize model weights
model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

model.train()

# Load normalized data using download_and_load_npy function
positions_url = 'https://github.com/manny-umassd/nbody-undergrad-project/blob/main/data/simulations/positions_over_time.npy?raw=true'
velocities_url = 'https://github.com/manny-umassd/nbody-undergrad-project/blob/main/data/simulations/velocities_over_time.npy?raw=true'
masses_url = 'https://github.com/manny-umassd/nbody-undergrad-project/blob/main/data/simulations/masses.npy?raw=true'

positions = download_and_load_npy(positions_url)
positions = (positions - positions.mean()) / positions.std()
velocities = download_and_load_npy(velocities_url)
velocities = (velocities - velocities.mean()) / velocities.std()
masses = download_and_load_npy(masses_url)
masses = (masses - masses.mean()) / masses.std()

# Prepare the data for training
node_features = []
for i in range(n_bodies):
    features = np.concatenate((positions[-1, i], velocities[-1, i], [masses[i]]))
    node_features.append(features)
x = torch.tensor(np.array(node_features), dtype=torch.float32)  # Convert to numpy array first and use float32

# Fully connected graph edges
edge_index = torch.tensor(
    [[i, j] for i in range(n_bodies) for j in range(n_bodies) if i != j],
    dtype=torch.long
).t().contiguous()

# Create Data object for training
training_data = Data(x=x, edge_index=edge_index)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce LR by half every 20 epochs

# Training loop
n_epochs = 50
losses = []
validation_losses = []
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    predicted = model(training_data)
    predicted = predicted.view(n_bodies, 6)
    actual = torch.cat((torch.tensor(positions[-1], dtype=torch.float32), torch.tensor(velocities[-1], dtype=torch.float32)), dim=1)
    
    loss = F.mse_loss(predicted, actual)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    # Validation (using a different time step)
    model.eval()
    with torch.no_grad():
        val_positions = positions[-2]  # Use the second-to-last time step for validation
        val_velocities = velocities[-2]
        val_node_features = []
        for i in range(n_bodies):
            val_features = np.concatenate((val_positions[i], val_velocities[i], [masses[i]]))
            val_node_features.append(val_features)
        val_x = torch.tensor(np.array(val_node_features), dtype=torch.float32)
        val_data = Data(x=val_x, edge_index=edge_index)
        val_predicted = model(val_data).view(n_bodies, 6)
        val_actual = torch.cat((torch.tensor(val_positions, dtype=torch.float32), torch.tensor(val_velocities, dtype=torch.float32)), dim=1)
        val_loss = F.mse_loss(val_predicted, val_actual)
        validation_losses.append(val_loss.item())
        print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss.item():.4f}")

# Plot training and validation loss over epochs
plt.figure()
plt.plot(range(1, n_epochs + 1), losses, label='Training Loss')
plt.plot(range(1, n_epochs + 1), validation_losses, linestyle='--', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.grid()
plt.legend()
plt.show()

# Evaluation: Predict future positions and velocities
model.eval()
with torch.no_grad():
    test_positions = positions[-1]  # Use the last time step for testing
    test_velocities = velocities[-1]
    test_node_features = []
    for i in range(n_bodies):
        test_features = np.concatenate((test_positions[i], test_velocities[i], [masses[i]]))
        test_node_features.append(test_features)
    test_x = torch.tensor(np.array(test_node_features), dtype=torch.float32)
    test_data = Data(x=test_x, edge_index=edge_index)
    test_predicted = model(test_data).view(n_bodies, 6)

    # Actual values
    actual_positions = test_positions
    actual_velocities = test_velocities

    # Predicted values
    predicted_positions = test_predicted[:, :3].detach().numpy()
    predicted_velocities = test_predicted[:, 3:].detach().numpy()

    # Plot actual vs. predicted positions for body 1
    plt.figure()
    for i in range(n_bodies):
        plt.scatter(actual_positions[i][0], actual_positions[i][1], color='blue', label=f'Actual Body {i+1}')
        plt.scatter(predicted_positions[i][0], predicted_positions[i][1], color='red', marker='x', label=f'Predicted Body {i+1}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Actual vs. Predicted Positions of Bodies')
    plt.legend()
    plt.grid()
    plt.show()

# Calculate evaluation metrics (MAE, RMSE, and R�)
mae = np.mean(np.abs(predicted_positions - actual_positions))
rmse = np.sqrt(np.mean((predicted_positions - actual_positions) ** 2))
ss_res = np.sum((predicted_positions - actual_positions) ** 2)
ss_tot = np.sum((actual_positions - np.mean(actual_positions)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (R�): {r2_score:.4f}")

# Generate future trajectories
future_steps = 10
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

# Plot future trajectories
plt.figure()
for i in range(n_bodies):
    actual_traj = np.array([pos[i] for pos in future_positions])
    plt.plot(actual_traj[:, 0], actual_traj[:, 1], label=f'Predicted Trajectory Body {i+1}')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Predicted Future Trajectories of Bodies')
plt.legend()
plt.grid()
plt.show()

# Save the model for deployment
torch.save(model.state_dict(), 'gnn_model_with_trajectories.pth')
