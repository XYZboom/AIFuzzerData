---
name: "🐛 [Bug B] inductor rounds slightly negative F.elu outputs to exactly 0.0, causing div-by-zero to produce nan instead of -inf"
about: Create a report to help us reproduce and fix the bug
labels: ["oncall: pt2", "topic: fuzzer", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` with the `inductor` backend rounds slightly negative `F.elu(x, alpha=0.52)` outputs to exactly `0.0` when the ELU output is a subnormal float32 value (magnitude < 1.18e-38 for `alpha=0.52`). This causes `torch.div(0.0, 0.0)` to produce `nan` instead of `-inf` (which is the correct result from `negative / 0 = -inf`).

> **Duplicate check**: Related issue [#185480](https://github.com/pytorch/pytorch/issues/185480) reports a similar bug for `F.celu` on CUDA bf16 subnormals. PR [#185878](https://github.com/pytorch/pytorch/pull/185878) fixed the root cause in `expm1` for CUDA. However, our bug affects **CPU float32** `F.elu` with **normal float32 values** (e.g., `-5.2e-9` is a normal float32), and the trigger is not through `expm1` but through the ELU Triton kernel itself. This is a **different trigger path** — not covered by #185878.

**Root cause**: For input `x ≈ -1e-8`, `F.elu(x, alpha=0.52) ≈ -5.2e-9` in fp64 (a normal float32 value). The eager implementation correctly computes this. The inductor Triton kernel for ELU appears to truncate or round values to zero when they become very small, possibly due to a denormal-flush-to-zero (DAZ) mode or a kernel implementation that doesn't handle tiny values accurately.

**Minimal reproducer** (`bug_001_minimal.py`):
```python
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
```

## Expected behavior

```
Eager output:   tensor([-inf, -inf, -inf, -inf, -inf, -inf])
Inductor output: should also be [-inf, -inf, -inf, -inf, -inf, -inf]
```

## Actual behavior

```
Eager:   tensor([-inf, -inf, -inf, -inf, -inf, -inf])
Compile: tensor([nan, nan, nan, nan, nan, nan])
```

The ELU values in fp32 are `[-5.2e-09, -5.2e-10, -5.2e-11, -5.2e-21, -5.2e-31, -5.2e-39]`. The last value `-5.2e-39` is a subnormal float32. Inductor appears to round all of these to exactly `0.0`, causing `0.0 / 0.0 = nan`.

## Error logs

```
[1] aot_eager matches eager: True
[2] inductor -inf→nan mismatches: 6/6
[3] FP64 vs FP32 elu same: True

    eager:   tensor([-inf, -inf, -inf, -inf, -inf, -inf])
    compile: tensor([nan, nan, nan, nan, nan, nan])
    elu(fp32): tensor([-5.2000e-09, -5.2000e-10, -5.2000e-11, -5.2000e-21, -5.2000e-31,
        -5.2000e-39])
    elu(fp64): tensor([-5.2000e-09, -5.2000e-10, -5.2000e-11, -5.2000e-21, -5.2000e-31,
        -5.2000e-39], dtype=torch.float64)

BUG CONFIRMED: True
  Inductor rounds slightly negative elu outputs to 0.0,
  causing 0/0=nan instead of neg/0=-inf.
```

## Specifications

- PyTorch Version: 2.13.0+cu130
- torch.compile backend: inductor (`backend="inductor"`)
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