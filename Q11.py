import torch

a = torch.rand(size=(1, 1, 1, 10))
a = a.squeeze()
print(a)
torch.manual_seed(7)
a = torch.rand(size=(1, 1, 1, 10))
a = a.squeeze()
print(a)