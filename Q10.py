import torch

a = torch.rand(size=(1, 5))
print(a, torch.argmax(a), torch.argmin(a))