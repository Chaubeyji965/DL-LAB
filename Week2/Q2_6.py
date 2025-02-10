import torch


def f(x, y, z):
    a = 2 * x
    b = torch.sin(y)
    c = a / b
    d = c * z
    e = torch.log(d + 1)
    return torch.tanh(e)


x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)


output = f(x, y, z)


output.backward()


grad_x = x.grad
grad_y = y.grad
grad_z = z.grad


print("Gradient w.r.t x (PyTorch):", grad_x.item())
print("Gradient w.r.t y (PyTorch):", grad_y.item())
print("Gradient w.r.t z (PyTorch):", grad_z.item())



a = 2 * x
b = torch.sin(y)
c = a / b
d = c * z
e = torch.log(d + 1)

grad_f_e = 1 - torch.tanh(e)**2
grad_e_d = 1 / (d + 1)
grad_d_c = z
grad_c_a = 1 / b
grad_a_x = 2

grad_f_x = grad_f_e * grad_e_d * grad_d_c * grad_c_a * grad_a_x


grad_c_b = -a / (b**2)
grad_b_y = torch.cos(y)

grad_f_y = grad_f_e * grad_e_d * grad_d_c * (grad_c_b * grad_b_y)


grad_d_z = c
grad_f_z = grad_f_e * grad_e_d * grad_d_z


print("Gradient w.r.t x (Analytical):", grad_f_x.item())
print("Gradient w.r.t y (Analytical):", grad_f_y.item())
print("Gradient w.r.t z (Analytical):", grad_f_z.item())
