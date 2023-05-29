import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

# Define the Graph Neural Network model
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

# Load and preprocess the CSV data
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    time_column = df['time']
    sensor_columns = df.drop('time', axis=1)

    x = torch.tensor(sensor_columns.values, dtype=torch.float)
    edge_index = torch.tensor([[i, j] for i in range(x.size(0)) for j in range(x.size(0)) if i != j], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)

    return data, time_column

# Perform multivariate analysis and anomaly detection
def perform_anomaly_detection(data, num_epochs, hidden_size):
    model = GNNModel(data.num_features, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    loader = DataLoader([data], batch_size=1, shuffle=False)
    for epoch in range(num_epochs):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.x)
            loss.backward()
            optimizer.step()

    predictions = model(data.x, data.edge_index)
    errors = torch.abs(predictions - data.x)
    anomaly_values = torch.mean(errors, dim=0)

    return anomaly_values

# Perform anomaly detection for each CSV file and identify anomalous sensors
def perform_anomaly_detection_for_files(file_paths, num_epochs, hidden_size):
    all_anomaly_values = []
    for file_path in file_paths:
        data, _ = load_csv_data(file_path)
        anomaly_values = perform_anomaly_detection(data, num_epochs, hidden_size)
        all_anomaly_values.append(anomaly_values)

    all_anomaly_values = torch.stack(all_anomaly_values)
    avg_anomaly_values = torch.mean(all_anomaly_values, dim=0)
    std_anomaly_values = torch.std(all_anomaly_values, dim=0)
    threshold = avg_anomaly_values + 2 * std_anomaly_values

    anomalous_sensors = torch.where(avg_anomaly_values > threshold)[0].tolist()

    return anomalous_sensors

# Define the file paths of the CSV files
file_paths = ['file1.csv', 'file2.csv', ...]  # Add the paths to all CSV files

# Set the hyperparameters
num_epochs = 100
hidden_size = 64

# Perform anomaly detection and get anomalous sensors
anomalous_sensors = perform_anomaly_detection_for_files(file_paths, num_epochs, hidden_size)

print("Anomalous Sensors:")
print(anomalous_sensors)
