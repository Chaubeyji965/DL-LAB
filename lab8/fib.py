import torch
import torch.nn as nn
import numpy as np


sequence = [0, 1]
for i in range(2, 100):
    next_value = sequence[-1] + sequence[-2]
    sequence.append(next_value)

sequence = np.array(sequence)


sequence_length = 3
X, y = [], []

for i in range(len(sequence) - sequence_length):
    X.append(sequence[i:i+sequence_length])
    y.append(sequence[i+sequence_length])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(300):
    model.train()
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/300], Loss: {loss.item():.4f}')


user_input = input("Enter three integers separated by space (Fibonacci sequence): ").split()
user_input = [int(i) for i in user_input]


input_sequence = torch.tensor([user_input], dtype=torch.float32).unsqueeze(-1)
model.eval()
predicted = model(input_sequence).detach().numpy().flatten()
predicted = int(round(predicted[0]))

print(f"Next Fibonacci number after {user_input} is {predicted}")
