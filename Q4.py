
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torch import nn

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
train = datasets.MNIST('.', train= True, download= True, transform= transforms)
test = datasets.MNIST('.', train= False, download= True, transform=transforms)
train_loader = DataLoader(train, batch_size=64, shuffle= True)
test_loader = DataLoader(test, batch_size= 64, shuffle= True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride= 2),
                                 nn.Conv2d(32, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 32, kernel_size= 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride= 2))
        self.classify_head = nn.Sequential(nn.Linear(32, 20, bias= True),
                                           nn.Linear(20, 10, bias= True))

    def forward(self, x):
        return self.classify_head(self.net(x).reshape(-1, 32))

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print(f'Epoch - {epoch}, loss = {running_loss}')



model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for input, target in test_loader:
        output = model(input)
        val, index = torch.max(output, 1)
        all_preds.extend(index)
        all_labels.extend(target)
cm = confusion_matrix(all_labels, all_preds)
print(cm)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


from sklearn.metrics import accuracy_score
print(accuracy_score(all_labels, all_preds))




