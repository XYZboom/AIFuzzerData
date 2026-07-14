import torch
import torch.nn as nn
import torch.nn.functional as F

class BugModule(nn.Module):
    def __init__(self, alpha: float = 1.24):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        elu_val = F.elu(x, alpha=self.alpha)
        exp_elu = torch.exp(elu_val)
        v = torch.add(elu_val, exp_elu)
        return torch.reciprocal(v)

torch.manual_seed(42)
x = torch.randn(24, 57, 37, dtype=torch.float32, device="cpu")

model = BugModule(alpha=1.24)

with torch.no_grad():
    eager_out = model(x)

with torch.no_grad():
    aot_out = torch.compile(model, backend="aot_eager")(x)

with torch.no_grad():
    ind_out = torch.compile(model, mode="default")(x)

# fp64 reference
x_fp64 = x.to(torch.float64)
model_fp64 = BugModule(alpha=1.24)
with torch.no_grad():
    fp64_ref = model_fp64(x_fp64).to(torch.float32)

print(f"max |aot_eager - inductor|  : {(aot_out - ind_out).abs().max().item():.4f}")
print(f"max |eager     - aot_eager| : {(eager_out - aot_out).abs().max().item():.6f}")
print(f"max |fp64_ref  - inductor|  : {(fp64_ref - ind_out).abs().max().item():.4f}")
print(f"torch._dynamo.utils.same(aot_eager, inductor) : {torch._dynamo.utils.same(aot_out, ind_out)}")

# Intermediate analysis
v = torch.add(F.elu(x, alpha=1.24), torch.exp(F.elu(x, alpha=1.24)))
near_zero = (v.abs() < 0.01).sum().item()
print(f"v = elu + exp(elu): |v| < 0.01 count = {near_zero}")
print(f"v min: {v.min().item():.8f}, max: {v.max().item():.8f}")