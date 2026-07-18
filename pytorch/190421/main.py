import torch

@torch.compile
def f():
    x = torch.full((2,), 0.0)          # log(0) → -inf
    y = torch.log(x)
    z = torch.sum(y, dtype=torch.int32).float()
    return z

print(f())