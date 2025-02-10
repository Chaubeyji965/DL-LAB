import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. Define the Feed Forward Neural Network with two hidden layers
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # First hidden layer (input: 28x28, output: 512)
        self.fc2 = nn.Linear(512, 256)    # Second hidden layer (input: 512, output: 256)
        self.fc3 = nn.Linear(256, 10)     # Output layer (input: 256, output: 10 classes)
        self.dropout = nn.Dropout(0.2)    # Dropout layer to prevent overfitting

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image into a 1D vector of size 28*28
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = self.dropout(x)      # Apply dropout after the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)          # Output layer (raw logits)
        return x

# 3. Initialize the model, loss function, and optimizer
model = FFNN()
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {train_accuracy:.2f}%')

# 5. Evaluate the model on the test set
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# 6. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 7. Verify the number of learnable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total learnable parameters: {total_params}')
