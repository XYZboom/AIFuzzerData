import torch
import torch.nn.functional as F

class BugModule(torch.nn.Module):
    def forward(self, x):
        return torch.div(F.elu(x, alpha=0.52), torch.zeros_like(x))

x = torch.tensor([-1e-8, -1e-9, -1e-10, -1e-20, -1e-30, -1e-38],
                 dtype=torch.float32)

model = BugModule()

with torch.no_grad():
    ref = model(x)  # eager reference

# aot_eager: matches eager
compiled_aot = torch.compile(model, backend="aot_eager")
with torch.no_grad():
    cmp_aot = compiled_aot(x)

# inductor: buggy
compiled_ind = torch.compile(model, backend="inductor")
with torch.no_grad():
    cmp_ind = compiled_ind(x)

# fp64 reference for elu values
elu_fp64 = F.elu(x.double(), alpha=0.52)
elu_fp32 = F.elu(x, alpha=0.52)

print(f"aot_eager matches eager: {torch._dynamo.utils.same(ref, cmp_aot)}")
print(f"inductor -inf→nan mismatches: {(torch.isneginf(ref) & torch.isnan(cmp_ind)).sum().item()}/{x.numel()}")
print(f"eager:   {ref}")
print(f"compile: {cmp_ind}")
print(f"elu(fp32): {elu_fp32}")
print(f"elu(fp64): {elu_fp64}")