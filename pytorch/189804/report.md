---
name: "🐛 [Bug D] Inductor vectorized GELU kernel produces tiny negative value (~-1.65e-7) where eager produces -0.0, causing sign() mismatch"
about: Create a report to help us reproduce and fix the bug
labels: ["oncall: pt2", "topic: fuzzer", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` with the `inductor` backend produces a numerical discrepancy in `F.gelu(x)` for input values near `x ≈ -5.53`. In eager mode, `F.gelu(-5.53)` produces `-0.0` (negative zero, 0x80000000). In inductor, it produces a tiny negative value (`≈ -1.65e-7`). When followed by `torch.sign()`, this causes `sign(-0.0) = 0` in eager vs `sign(tiny_negative) = -1` in inductor — a difference of 1.0 in the output.

**Root cause**: The inductor Triton kernel for GELU uses a vectorized approximation that differs from the eager (scalar) implementation in how it handles the tail of the GELU function near `x ≈ -5.53`. The discrepancy only manifests with multi-element tensors (≥2 elements), suggesting inductor uses a vectorized GELU kernel that processes elements in batches and may use a different polynomial approximation.

In the original fuzzer-generated model, the chain `log2(x) → log2(result) → div → gelu → sign → sigmoid` produces `max_diff=0.23` at output index 13 (shape [23, 64, 62]).

**Minimal reproducer** (`bug_010_minimal.py`):
```python
import torch
import torch.nn.functional as F

# Values from the original computation chain where gelu produces -0.0
x = torch.tensor([-5.533906936645508, -5.531643390655518], dtype=torch.float32)

class BugModel(torch.nn.Module):
    def forward(self, x):
        return torch.sign(F.gelu(x))

model = BugModel()

with torch.no_grad():
    ref = model(x)  # eager

with torch.no_grad():
    aot_out = torch.compile(model, backend="aot_eager")(x)

with torch.no_grad():
    ind_out = torch.compile(model, mode="default")(x)

print(f"Input:  {x.tolist()}")
print(f"Eager:  {ref.tolist()}   (gelu produces -0.0, sign(-0.0)=0)")
print(f"aot:    {aot_out.tolist()}")
print(f"Induct: {ind_out.tolist()}   (gelu produces ~-1.65e-7, sign(tiny_neg)=-1)")
print(f"aot vs ref: same={torch._dynamo.utils.same(ref.double(), aot_out.double())}")
print(f"ind  vs ref: same={torch._dynamo.utils.same(ref.double(), ind_out.double())}")
print(f"max_diff (ind vs ref): {(ref.double() - ind_out.double()).abs().max().item():.6f}")

# Also the full chain from the original source
print("\n=== Full chain from original source ===")
torch.manual_seed(42)
v6 = torch.randn(23, 64, 62, dtype=torch.float32, device="cpu")

class FullChainModel(torch.nn.Module):
    def forward(self, x):
        a = torch.log2(x)
        b = torch.log2(a)
        c = torch.div(a, b)
        d = F.gelu(c)
        e = torch.sign(d)
        return torch.sigmoid(e)

full_model = FullChainModel()
with torch.no_grad():
    full_ref = full_model(v6)
full_ind = torch.compile(full_model, mode="default")
with torch.no_grad():
    full_ind_out = full_ind(v6)

print(f"Full chain ind vs ref: same={torch._dynamo.utils.same(full_ref.double(), full_ind_out.double())}")
print(f"max_diff: {(full_ref.double() - full_ind_out.double()).abs().max().item():.6f}")
```

## Expected behavior

```
Eager:  [0.0, 0.0]   (sign(-0.0) = 0)
aot:    [0.0, 0.0]   (same as eager)
Induct: [0.0, 0.0]   (should also be 0.0)
```

## Actual behavior

```
Input:  [-5.533906936645508, -5.531643390655518]
Eager:  [0.0, 0.0]   (gelu produces -0.0, sign(-0.0)=0)
aot:    [0.0, 0.0]
Induct: [-1.0, -1.0]   (gelu produces ~-1.65e-7, sign(tiny_neg)=-1)
```

Full chain with shape [23, 64, 62]:
```
Full chain inductor vs ref: same=False
max_diff: 0.231059
```

## Error logs

```
E0714 08:17:05.717000 ... site-packages/torch/_dynamo/utils.py:3706] RMSE (res-fp64): 1.00000, (ref-fp64): 0.00000
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