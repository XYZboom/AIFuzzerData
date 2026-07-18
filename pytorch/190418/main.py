import torch
import torch.nn.functional as F

def fn():
    x = torch.zeros(1).argmax().float()
    y = x * x
    return F.log_softmax(y, dim=-1)

compiled = torch.compile(fn, backend="inductor")
print(compiled())