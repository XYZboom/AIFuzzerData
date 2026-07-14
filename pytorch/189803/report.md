---
name: "🐛 [Bug C] Inductor produces large numerical discrepancy in reciprocal(elu + exp(elu)) due to catastrophic cancellation near zero"
about: Create a report to help us reproduce and fix the bug
labels: ["oncall: pt2", "topic: fuzzer", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` with the `inductor` backend produces large numerical discrepancies (up to ~108 in absolute value) for the expression `torch.reciprocal(elu(x, alpha) + exp(elu(x, alpha)))` compared to both eager and aot_eager backends.

**Root cause**: For inputs `x ≈ -0.6113` with `alpha=1.24`:
- `F.elu(x, 1.24) ≈ -0.5671`
- `torch.exp(F.elu(x, 1.24)) ≈ +0.5671`
- The sum `v = elu + exp(elu)` can be as small as ~2e-5 in float32 due to catastrophic cancellation

The tiny difference in `v` (≈4e-6) between backends is amplified by `reciprocal` to a difference of ~100-800 in the output. This is a classic case of catastrophic cancellation amplified by reciprocal, but the key issue is that **inductor and aot_eager produce different values for `v`** in the first place — the fusion or instruction reordering in inductor changes the order of floating-point operations, and the result is numerically different at the level of ~4e-6, which then gets blown up.

This is NOT a simple precision issue — it's a correctness concern because the difference (up to 108) is far beyond what any reasonable tolerance can accept.

**Minimal reproducer** (`bug_003_minimal.py`):
```python
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
```

## Expected behavior

All backends (aot_eager, inductor) should produce numerically similar results. Since aot_eager matches eager exactly (`max diff = 0.0`), inductor should also match.

## Actual behavior

```
max |aot_eager - inductor|  : 108.3516
max |eager     - aot_eager| : 0.000000
max |fp64_ref  - inductor|  : 86.2344
torch._dynamo.utils.same(aot_eager, inductor) : False

v = elu + exp(elu): |v| < 0.01 count = 322
v min: -0.93436801, max: 72.31462860
```

The discrepancy reaches **108.35** in absolute value, which is extremely large for float32 computations.

## Error logs

```
E0714 08:16:54.381000 589089 site-packages/torch/_dynamo/utils.py:3722] Accuracy failed: allclose not within tol=0.0001
...
BUG CONFIRMED: aot_eager vs inductor max_abs_diff = 108.3516
```

## Specifications

- PyTorch Version: 2.13.0+cu130
- torch.compile backend: inductor (`mode="default"`)
- CUDA: Not available (CPU-only repro)
- Python: 3.11.15
- OS: Ubuntu 24.04.4 LTS (x86_64)
- CPU: AMD Ryzen 9 9950X 16-Core Processor

## Versions

```
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: 15.0.7
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA

CPU:
Model name:                           AMD Ryzen 9 9950X 16-Core Processor
CPU(s):                               32

Versions of relevant libraries:
[pip3] torch==2.13.0
[pip3] triton==3.7.1
[pip3] numpy==2.4.6
[conda] mkl                       2025.0.0
```