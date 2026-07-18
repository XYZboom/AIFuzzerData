import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        v0 = torch.zeros((2, 1), dtype=torch.float32, device="cpu")
        v1 = torch.log2(v0.float())
        v2 = F.layer_norm(v1.float(), (v1.shape[-1],))
        return v2

model = TestModule()

# Eager — OK
with torch.no_grad():
    ref = model()

# Compiled — crashes
compiled = torch.compile(model, mode="default")
with torch.no_grad():
    cmp = compiled()