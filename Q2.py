import torch

# permute helps permuting the shape of tensor
a = torch.rand(size=(3, 4))
print(a)
a = torch.permute(a, dims=(1, 0))
print(a)