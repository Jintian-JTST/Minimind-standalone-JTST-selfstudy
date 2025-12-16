import torch
import torch_directml

device = torch_directml.device()
print("DirectML device:", device)

# 做一个简单矩阵乘
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)

# 触发计算
c = a @ b
print("Result mean:", c.mean().item())
