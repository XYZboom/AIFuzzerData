---
name: "🐛 torch.compile Bug Report"
about: "Create a report to help us reproduce and fix the bug"
title: "[Inductor] IndexError in _other_is_broadcasted_in_dim when log_softmax receives 0D scalar from mul"
labels: ["oncall: pt2", "topic: fuzzer"]
---

## 🐛 Describe the bug

`torch.compile` crashes with `IndexError: list index out of range` inside `_other_is_broadcasted_in_dim` in `joint_graph.py` when a `log_softmax(dim=-1)` receives a 0D (scalar) tensor produced by `mul` of two 0D scalars.

### Root cause

The function `_other_is_broadcasted_in_dim` at `torch/_inductor/fx_passes/joint_graph.py:1018` guards against negative dimensions like this:

```python
if any(d >= len(other_shape) for d in dim):
    return False
```

However, `-1 >= 0` evaluates to `False` in Python, so a negative dimension slips through the guard and causes `other_shape[-1]` at line 1021 to raise `IndexError` on an empty shape.

The fix should be:

```python
if any(d >= len(other_shape) or d < -len(other_shape) for d in dim):
    return False
```

### Minimal reproducer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModule(nn.Module):
    def forward(self):
        v = torch.zeros((1), dtype=torch.float32, device="cpu")
        x = torch.argmax(v, dim=-1).float()     # → 0D scalar
        y = torch.mul(x.float(), x.float())      # → 0D scalar
        z = F.log_softmax(y.float(), dim=-1)     # → IndexError
        return z

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
- ✅ Eager execution — works fine
- ✅ Removing `dim=-1` or changing to `dim=0` — works fine
- ✅ Using a non-0D input (e.g. shape `[1]` instead of `()`) — works fine

### Workaround

Avoid `log_softmax(dim=-1)` on 0D tensors with `torch.compile`. Explicitly reshape to at least 1D, or use `dim=0` for scalar inputs.

---

## Error logs

```
Traceback (most recent call last):
  File "/home/xyzboom/Code/kotlin/aiFuzzer/daemon/pytorch_daemon.py", line 91, in run_source
    exec(source, {
  File "<string>", line 111, in <module>
  File "/home/xyzboom/Programs/miniconda3/envs/aifuzzer/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 511, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xyzboom/Programs/miniconda3/envs/aifuzzer/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [ ... frame repeated through dynamo tracing ... ]
  File "/home/xyzboom/Programs/miniconda3/envs/aifuzzer/lib/python3.11/site-packages/torch/_inductor/fx_passes/joint_graph.py", line 1021, in _other_is_broadcasted_in_dim
    return all(statically_known_true(other_shape[d] == 1) for d in dim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xyzboom/Programs/miniconda3/envs/aifuzzer/lib/python3.11/site-packages/torch/_inductor/fx_passes/joint_graph.py", line 1021, in <genexpr>
    return all(statically_known_true(other_shape[d] == 1) for d in dim)
                                     ~~~~~~~~~~~^^^
IndexError: list index out of range
```

---

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

CPU:
  AMD Ryzen 9 9950X 16-Core Processor (32 threads)

Versions of relevant libraries:
[pip3] torch==2.13.0
[pip3] triton==3.7.1
[pip3] numpy==2.4.6
[conda] torch==2.13.0
[conda] triton==3.7.1
```

---

## Additional context

- Discovered by [aiFuzzer](https://github.com/XYZboom/AIFuzzer) — an AI compiler fuzzing framework
- The bug was reduced from a 68-node program to 4 nodes (94.1% reduction) via DDMin + wire-around reconstruction
- Triggers on `mul.Tensor + log_softmax` pattern when `other` (the scaling factor) is a 0D tensor: the `_other_is_broadcasted_in_dim` guard at line 1018 uses `d >= len(other_shape)` which does not catch negative `dim` on empty shapes because `-1 >= 0` is `False`
- Related: `_partial_softmax_pattern` at line 1050 registers with `extra_check=_other_is_broadcasted_in_dim` — every pattern using this checker is susceptible
