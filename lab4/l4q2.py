import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# XOR Data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).to(device)

# Model
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# DataLoader
train_loader = torch.utils.data.DataLoader(MyDataset(X, Y), batch_size=1, shuffle=True)

# Initialize Model, Loss Function, and Optimizer
model = XORModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

# Training Loop
loss_list = []
for epoch in range(1000):
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss_list.append(total_loss / len(train_loader))
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{1000}, Loss: {total_loss / len(train_loader)}')

# Print parameters
for param in model.named_parameters():
    print(param)

# Prediction
model.eval()
input_data = torch.tensor([0, 1], dtype=torch.float32).to(device)
output = model(input_data)
print(f"The input is {input_data}")
print(f"Output y predicted is {output}")

# Plot loss
plt.plot(loss_list)
plt.show()
