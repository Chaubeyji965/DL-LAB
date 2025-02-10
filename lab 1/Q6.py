import torch

a = torch.rand(size=(1, 7))
b = torch.rand(size= (1, 7))
c = a @ b.T
print(c, c.shape)