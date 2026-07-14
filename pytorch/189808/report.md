---
name: "🐛 [Bug E] Inductor sqrt produces negative NaN (0xffc00000) for negative inputs instead of positive NaN (0x7fc00000)"
about: Create a report to help us reproduce and fix the bug
labels: ["oncall: pt2", "topic: fuzzer", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` with the `inductor` backend produces a **different NaN bit pattern** for `torch.sqrt(negative_input)` compared to eager and aot_eager. Inductor produces a **negative NaN** (0xffc00000, quiet NaN with sign bit = 1), while eager and aot_eager produce a **positive NaN** (0x7fc00000, quiet NaN with sign bit = 0).

While both are technically NaN, the sign bit difference propagates through downstream computations (sub, gelu, tanh, log, elu, ceil, selu, etc.) and can change NaN/inf counts in final outputs. Additionally, inductor has slightly different precision for `sqrt` on positive inputs (max diff ≈ 1.19e-07 in float32), which combined with the NaN sign difference causes boundary crossings in `tanh(gelu(x))` near zero.

In the original fuzzer-generated model, this caused 4 output mismatches (indices 17, 19, 20, 22) with different nan/inf counts.

**Minimal reproducer** (`bug_012_minimal.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqrtNegativeNaN(nn.Module):
    def forward(self, x):
        return torch.sqrt(x)

def check_nan_sign_bit(tensor, label):
    int_view = tensor.view(torch.int32)
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        nan_bits = int_view[nan_mask]
        sign_bit = (nan_bits >> 31) & 1
        print(f"  {label}: {nan_mask.sum().item()} NaNs, "
              f"sign_bit={sign_bit.unique().tolist()}, "
              f"first_nan=0x{nan_bits[0].item():08x}")
    else:
        print(f"  {label}: no NaNs")

# Test 1: Single negative value
x = torch.tensor([-1.0], dtype=torch.float32, device="cpu")
model = SqrtNegativeNaN()

with torch.no_grad():
    eager = model(x)
with torch.no_grad():
    aot = torch.compile(model, backend="aot_eager")(x)
with torch.no_grad():
    ind = torch.compile(model, mode="default")(x)

check_nan_sign_bit(eager, "eager")
check_nan_sign_bit(aot, "aot_eager")
check_nan_sign_bit(ind, "inductor")

# Test 2: Mixed tensor
torch.manual_seed(42)
x2 = torch.randn(24, 23, 13, dtype=torch.float32, device="cpu")

with torch.no_grad():
    eager2 = model(x2)
with torch.no_grad():
    aot2 = torch.compile(model, backend="aot_eager")(x2)
with torch.no_grad():
    ind2 = torch.compile(model, mode="default")(x2)

print("\n--- Mixed tensor ---")
check_nan_sign_bit(eager2, "eager")
check_nan_sign_bit(aot2, "aot_eager")
check_nan_sign_bit(ind2, "inductor")

# Positive sqrt precision difference
pos_mask = x2 > 0
pos_diff = (eager2[pos_mask] - ind2[pos_mask]).abs()
print(f"Positive sqrt max diff: {pos_diff.max().item():.2e}")
```

## Expected behavior

All backends should produce the same NaN bit pattern for `sqrt(negative_input)`. The sign bit of NaN should be consistent (0, i.e., positive NaN, is the standard convention used by eager and aot_eager).

## Actual behavior

```
Test 1: sqrt(-1.0) NaN sign bit
============================================================
  eager: 1 NaNs, sign_bit=[0], first_nan=0x7fc00000
  aot_eager: 1 NaNs, sign_bit=[0], first_nan=0x7fc00000
  inductor: 1 NaNs, sign_bit=[1], first_nan=0x-0400000

  eager vs aot_eager: same=False
  eager vs inductor:  same=False
  aot_eager vs inductor: same=False

Test 2: sqrt of mixed tensor
============================================================
  eager: 3563 NaNs, sign_bit=[0], first_nan=0x7fc00000
  aot_eager: 3563 NaNs, sign_bit=[0], first_nan=0x7fc00000
  inductor: 3563 NaNs, sign_bit=[1], first_nan=0x-0400000

  Positive sqrt max diff: 1.19e-07
```

Downstream propagation (outputs 17, 19, 20, 22 from original model):
```
  v28 (elu, idx 17): FAIL  aot(nan=5767,inf=0) ind(nan=5771,inf=0)
  v30 (ceil, idx 19): FAIL  aot(nan=5767,inf=1409) ind(nan=5771,inf=1405)
  v31 (transpose, idx 20): FAIL  aot(nan=5767,inf=1409) ind(nan=5771,inf=1405)
  v36 (selu, idx 22): FAIL  aot(nan=5767,inf=0) ind(nan=5771,inf=0)
```

The NaN sign bit difference propagates through the entire computation graph. Note that inductor's output has **more NaNs** (5771 vs 5767) and **fewer infs** (1405 vs 1409), showing that the NaN sign difference can change boundary conditions in downstream functions like `tanh(gelu(x))`.

## Error logs

```
E0714 08:17:13.041000 ... Accuracy failed: allclose not within tol=0.0001
(15 occurrences across 4 output tensors)
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