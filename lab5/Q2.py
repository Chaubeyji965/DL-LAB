import torch
import torch.nn as nn
import torch.nn.functional as F


H, W, C_in, C_out = 6, 6, 1, 3
image = torch.rand(1, 1, H, W)


conv_layer = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1, bias=False)


output_nn = conv_layer(image)
print("Output using nn.Conv2d shape:", output_nn.shape)


kernel = conv_layer.weight
print("Kernel shape:", kernel.shape)


output_fn = F.conv2d(image, kernel, stride=1, padding=1)
print("Output using functional conv2d shape:", output_fn.shape)


print("Outputs match:", torch.allclose(output_nn, output_fn, atol=1e-6))
