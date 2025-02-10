import torch


def y(x):
    return 8 * x**4 + 3 * x**3 + 7 * x**2 + 6 * x + 3


x = torch.tensor(2.0, requires_grad=True)
output = y(x)


output.backward()
pytorch_grad = x.grad


analytical_grad = 32 * x**3 + 9 * x**2 + 14 * x + 6


print("Gradient (PyTorch):", pytorch_grad.item())
print("Gradient (Analytical):", analytical_grad.item())
