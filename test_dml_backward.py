import torch
import torch_directml

dml = torch_directml.device()
x = torch.randn(1024, 1024, device=dml, requires_grad=True)
y = (x @ x).mean()
y.backward()
print("backward ok, grad mean =", x.grad.mean().item())
