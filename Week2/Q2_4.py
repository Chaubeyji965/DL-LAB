import torch


def f(x):
    return torch.exp(-x**2 - 2*x - torch.sin(x))


x = torch.tensor(1.0, requires_grad=True)
y = f(x)


y.backward()
pytorch_grad = x.grad


analytical_grad = torch.exp(-x**2 - 2*x - torch.sin(x)) * (-2*x - 2 - torch.cos(x))


print("Gradient (PyTorch):", pytorch_grad.item())
print("Gradient (Analytical):", analytical_grad.item())
