import torch
import torch.nn.functional as F


H, W, C = 6, 6, 1
image = torch.rand(H, W)
print("Original image shape:", image.shape)


image = image.unsqueeze(0).unsqueeze(0)
print("Image shape after adding batch and channel dimensions:", image.shape)


K = 3
kernel = torch.rand(1, 1, K, K)
print("Kernel shape:", kernel.shape)


stride_values = [1, 2]
padding_values = [0, 1]

for stride in stride_values:
    for padding in padding_values:
        outimage = F.conv2d(image, kernel, stride=stride, padding=padding)
        print(f"Output image shape with stride={stride}, padding={padding}: {outimage.shape}")


        H_out = ((H + 2 * padding - K) // stride) + 1
        W_out = ((W + 2 * padding - K) // stride) + 1
        print(f"Expected Output Size: ({H_out}, {W_out})")
