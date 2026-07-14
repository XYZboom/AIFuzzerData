import torch
import torch.nn as nn

class BugModule(nn.Module):
    def forward(self, x):
        y = torch.add(x, torch.zeros_like(x))  # converts -0.0 to +0.0 per IEEE 754
        a = torch.sqrt(y)
        b = torch.rsqrt(a)
        return b

x = torch.zeros(4, 4, dtype=torch.float32, device="cpu")
x[0, 0] = -0.0
x[0, 1] = -0.0
x[1, 0] = 0.0   # positive zero (control)
x[1, 1] = 1.0   # normal value (control)
x[2, 0] = 0.5
x[2, 1] = -0.0

with torch.no_grad():
    ref = BugModule()(x.double()).float()
with torch.no_grad():
    out_aot = torch.compile(BugModule(), backend="aot_eager")(x)
with torch.no_grad():
    out_ind = torch.compile(BugModule(), mode="default")(x)

print("aot vs fp64:", torch._dynamo.utils.same(out_aot, ref, fp64_ref=ref.double()))
print("ind vs fp64:", torch._dynamo.utils.same(out_ind, ref, fp64_ref=ref.double()))
print("inf sign diff:", (torch.isinf(out_aot) & torch.isinf(out_ind) & (out_aot != out_ind)).sum().item())