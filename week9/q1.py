import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load dataset
df = pd.read_csv("/home/student/PycharmProjects/PythonProject2/week9/daily_natural_gas.csv")


# Preprocess the data - Drop NA values
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y) + 1, 1)

# Normalize the target variable
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)

# Define sequence length
Sequence_Length = 10
X, Y = [], []

# Prepare sequences
for i in range(len(y) - Sequence_Length):  # Fixed range
    X.append(y[i:i + Sequence_Length])
    Y.append(y[i + Sequence_Length])

# Convert to NumPy arrays
X = np.array(X).reshape(-1, Sequence_Length, 1)  # Reshape for LSTM
Y = np.array(Y)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42, shuffle=False)


# Define Dataset class
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# Create dataset and DataLoader
train_dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Take last time step output
        output = self.fc1(torch.relu(output))
        return output


# Initialize model, loss function, and optimizer
model = LSTMModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x).reshape(-1)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Evaluate on test set
test_dataset = NGTimeSeries(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Enable evaluation mode
model.eval()
test_pred = []

with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_pred = model(batch_x).view(-1).detach().numpy()
        test_pred.extend(batch_pred)

test_pred = np.array(test_pred)  # Convert list to NumPy array

# Undo normalization
y_test_actual = y_test * (y_max - y_min) + y_min
y_pred_denorm = test_pred * (y_max - y_min) + y_min

# Plot original vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Original')
plt.plot(y_pred_denorm, label='Predicted')
plt.legend()
plt.show()

# Final results on full dataset
plt.figure(figsize=(10, 5))
plt.plot(y * (y_max - y_min) + y_min)  # Original prices
plt.plot(range(len(y) - len(y_pred_denorm), len(y)), y_pred_denorm, color='red')
plt.show()
