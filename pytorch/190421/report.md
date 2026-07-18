---
name: "🐛 torch.compile Bug Report"
about: "Create a report to help us reproduce and fix the bug"
title: "[Inductor] OverflowError in index_propagation.TypedExpr when converting inf to int32 in sum"
labels: ["oncall: pt2", "topic: fuzzer"]
---

## 🐛 Describe the bug

`torch.compile` crashes with `OverflowError: cannot convert float infinity to integer` inside `index_propagation.py` when `torch.sum(..., dtype=torch.int32)` receives a tensor containing `inf` or `-inf` values (e.g., from `log(0)`).

### Root cause

`torch/_inductor/index_propagation.py:69` in `TypedExpr.__post_init__` calls `dtype_to_type(self.dtype)(expr)` where `expr` is a sympy expression evaluating to infinity and `dtype` is `torch.int32`. Python's `int(float('inf'))` and `int(float('-inf'))` raise `OverflowError`.

The Inductor's index propagation pass propagates sympy expressions through `to_dtype` operations. When the value being cast is an infinity (`1/0`, `log(0)`, etc.) and the target dtype is an integer type, the conversion fails because Python's `int()` cannot handle infinity.

### Minimal reproducer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# log(0.0) = -inf → sum(dim=-1, dtype=torch.int32) → int(-inf) → OverflowError
class TestModule(nn.Module):
    def forward(self):
        v = torch.full((1, 4, 3, 1), 0.0, dtype=torch.float32, device="cpu")
        x = torch.log(v.float())      # → -inf
        y = torch.sum(x, dim=-1, keepdim=False, dtype=torch.int32).float()
        return y

model = TestModule()
ref = model()
print(f"Eager: {ref}")

compiled = torch.compile(model, mode="default")
cmp = compiled()  # CRASH here
```

### Ablation

- ❌ `torch.compile(model, mode="default")` — crashes
- ❌ `torch.compile(model, mode="reduce-overhead")` — crashes
- ❌ `torch.compile(model, mode="max-autotune")` — crashes
- ✅ Eager execution — works fine (inf clamped to INT32_MIN = -2.1475e+09)
- ✅ Removing `dtype=torch.int32` (default float) — works fine (inf propagates through float)
- ✅ Using positive float values instead of zeros (no inf) — works fine
- ✅ Using `torch.int64` — also crashes on `int(float('-inf'))`

### Workaround

Avoid `torch.sum(..., dtype=torch.intXX)` on tensors that may contain non-finite values. Use float accumulation and convert afterwards, or clamp before accumulation:

```python
y = torch.sum(x, dim=-1)  # float accumulation
y = y.to(torch.int32)     # convert after, avoids index_propagation
```

---

## Error logs

```
Traceback (most recent call last):
  File "/home/xyzboom/Code/kotlin/aiFuzzer/daemon/pytorch_daemon.py", line 91, in run_source
    exec(source, {
  File "<string>", line 108, in <module>
  File "/.../torch/_dynamo/eval_frame.py", line 511, in __call__
    return super().__call__(*args, **kwargs)
  [ ... torch.compile compilation frames ... ]
  File "/.../torch/_inductor/compile_fx.py", line 1079, in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
  File "/.../torch/_inductor/index_propagation.py", line 69, in __post_init__
    expr = dtype_to_type(self.dtype)(expr)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._inductor.exc.InductorError: OverflowError: cannot convert float infinity to integer
```

Full call chain: `compile_fx` → `Scheduler.__init__` → `SchedulerNode._compute_attrs` → `LoopBody` → `store_output` → `to_dtype` → `index_propagation.TypedExpr.__post_init__` → `int(float('-inf'))` → `OverflowError`.

---

## Versions

```
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: 13.3.0
Clang version: 15.0.7
CMake version: 3.28.3
Python version: 3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0]
Python platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
Is CUDA available: False

CPU:
  AMD Ryzen 9 9950X 16-Core Processor (32 threads)

Versions of relevant libraries:
[pip3] torch==2.13.0
[pip3] triton==3.7.1
```

---

## Additional context

- Discovered by [aiFuzzer](https://github.com/XYZboom/AIFuzzer) — an AI compiler fuzzing framework
- The bug was reduced from a 70-node program to 3 nodes (95.7% reduction) via DDMin + wire-around reconstruction
- The crash happens at compile-time (in Inductor's code generation), not at runtime
- Eager mode correctly handles this case by clamping `inf`/`-inf` to `INT32_MIN`/`INT32_MAX`
- `torch._inductor.index_propagation.TypedExpr.__post_init__` at line 69: `expr = dtype_to_type(self.dtype)(expr)` — this assumes the sympy expression is finite, but doesn't guard against infinity
- Potential fix: in `index_propagation.py`, catch `OverflowError` during `to_dtype` propagation and fall back to non-propagated codegen
