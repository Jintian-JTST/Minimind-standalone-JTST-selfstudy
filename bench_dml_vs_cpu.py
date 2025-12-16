import time
import torch
import torch_directml

def bench(device, n=10, size=2048):
    # 预热
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(3):
        _ = a @ b

    t0 = time.time()
    for _ in range(n):
        c = a @ b
        _ = c.mean().item()  # 强制取值，避免“偷懒”
    return time.time() - t0

dml = torch_directml.device()
cpu = torch.device("cpu")

t_cpu = bench(cpu)
t_dml = bench(dml)

print("CPU time:", t_cpu)
print("DML time:", t_dml)
print("speedup:", t_cpu / t_dml if t_dml > 0 else None)
