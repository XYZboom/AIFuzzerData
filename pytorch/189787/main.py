import torch
import torch.nn.functional as F
import torch._dynamo
import torch._inductor.config

# Fuzzer requirement: fallback_random + seed reset
torch._inductor.config.fallback_random = True

torch.manual_seed(42)
x = torch.randn(43, 59, dtype=torch.float32, device="cpu")
x_fp64 = x.to(torch.float64)

class SubMod(torch.nn.Module):
    def forward(self, x):
        s = F.softmax(x, dim=-1)
        total = torch.sum(s, dim=-1, keepdim=False)
        return torch.floor(total)

model = SubMod()

# Eager (with seed reset)
torch.manual_seed(42)
with torch.no_grad():
    eager = model(x)

# Compile (with seed reset)
torch._dynamo.reset()
torch.manual_seed(42)
compiled = torch.compile(model, mode="default")
with torch.no_grad():
    comp = compiled(x)

# Verify raw sums pass torch._dynamo.utils.same with fp64 reference
class SubModRaw(torch.nn.Module):
    def forward(self, x):
        s = F.softmax(x, dim=-1)
        total = torch.sum(s, dim=-1, keepdim=False)
        return total

model_raw = SubModRaw()
with torch.no_grad():
    fp64_ref = model_raw(x_fp64)
    fp32_eager_raw = model_raw(x)

torch._dynamo.reset()
compiled_raw = torch.compile(model_raw, mode="default")
with torch.no_grad():
    fp32_comp_raw = compiled_raw(x)

same_result = torch._dynamo.utils.same(fp32_eager_raw, fp32_comp_raw, fp64_ref=fp64_ref)
print(f"torch._dynamo.utils.same (raw sums, fp64 ref): {same_result}")

# Show floor mismatch
diff_idx = (eager != comp)
print(f"eager:   {eager[:10].tolist()}")
print(f"compile: {comp[:10].tolist()}")
print(f"floor() mismatch: {diff_idx.sum().item()} / {eager.numel()}")

# fp64 ground truth
with torch.no_grad():
    fp64_floor = torch.floor(model_raw(x_fp64))
comp_missed = ((fp64_floor == 0) & (comp == 1)).sum().item()
print(f"compile wrong per fp64 (returns 1, should be 0): {comp_missed}")