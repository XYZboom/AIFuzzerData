import torch
import torch.nn as nn
import torch.nn.functional as F

class MinModule(nn.Module):
    def forward(self, x):
        s = torch.sign(x)
        m = torch.mean(s.float(), dim=-1, keepdim=False)
        c = torch.ceil(m)          # can produce -0.0
        e = F.elu(c, alpha=1.14)   # elu(-0.0) = -0.0 in eager, +0.0 in inductor
        return torch.rsqrt(e)      # rsqrt(-0.0) = -inf vs rsqrt(+0.0) = +inf

torch.manual_seed(42)
x = torch.randn(33, 57, 32, dtype=torch.float32, device="cpu")

with torch.no_grad():
    ref = MinModule()(x)
with torch.no_grad():
    out_aot = torch.compile(MinModule(), backend="aot_eager")(x)
with torch.no_grad():
    out_ind = torch.compile(MinModule(), mode="default")(x)

print("aot vs eager:", torch._dynamo.utils.same(ref, out_aot))  # True
print("ind vs eager:", torch._dynamo.utils.same(ref, out_ind))  # False

# Direct elu test
neg_zero = torch.full((1, 8), -0.0, dtype=torch.float32)
eager_elu = F.elu(neg_zero, alpha=1.14)
compiled_elu = torch.compile(lambda t: F.elu(t, alpha=1.14), mode="default")(neg_zero)
print("eager elu(-0.0):", eager_elu[0, 0].item())     # -0.0
print("compiled elu(-0.0):", compiled_elu[0, 0].item())  # 0.0
