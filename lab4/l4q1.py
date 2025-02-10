import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_list = []

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).to(device)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(2, 2, bias=True)
        self.activation1 = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_data_loader = torch.utils.data.DataLoader(MyDataset(X, Y), batch_size=1, shuffle=True)
model = XORModel().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

def train_one_epoch():
    total_loss = 0.0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data_loader)

for epoch in range(1000):
    model.train(True)
    avg_loss = train_one_epoch()
    loss_list.append(avg_loss)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{1000} , Loss: {avg_loss}')

for param in model.named_parameters():
    print(param)

input = torch.tensor([0, 1], dtype=torch.float32).to(device)
model.eval()
print(f"Input: {input}")
print(f"Output: {model(input)}")

plt.plot(loss_list)
plt.show()
