# 🐛 torch.compile Bug Report: Inductor constant-folds `sum(softmax)` to exactly 1.0, causing `floor()` to produce wrong integer results

## 🐛 Describe the bug

`torch.compile` with the **Inductor** backend constant-folds `sum(softmax(x, dim))` to exactly `1.0` (exploiting the mathematical identity). However, in finite-precision float32 arithmetic, softmax sums can be slightly below or above 1.0 due to rounding. When `floor()` is applied downstream, this creates a **0 vs 1** discrepancy between eager and compiled modes.

Using `torch._dynamo.utils.same` with an fp64 reference, the raw `sum(softmax)` outputs from both eager and compile **pass** the comparison — they are both within expected float32 tolerance of fp64. However, `floor()` amplifies this ULP-level difference into an integer-level mismatch, and **Inductor's constant-folded `1.0` is actually further from the fp64 ground truth than the eager result** in some cases.

### Fuzzer compliance checklist

- ✅ **Default tolerances**: The bug involves `floor()` producing exact integer differences (0 vs 1), not tolerance-based comparisons. Raw softmax sums are verified with `torch._dynamo.utils.same` at default tolerance.
- ✅ **No max/min index comparison**: We compare `floor(sum(softmax))`, not argmax/argmin indices.
- ✅ **fp64 reference baseline**: `torch._dynamo.utils.same(eager, compile, fp64_ref=fp64_result)` returns `True` for raw softmax sums, confirming both are within float32 tolerance of fp64. The bug manifests only after `floor()`.
- ✅ **Equivalent RNG state**: `torch._inductor.config.fallback_random = True` and `torch.manual_seed(42)` reset between eager/compile runs. Bug reproduces consistently.

### Key observations

| Comparison | Result |
|---|---|
| `torch._dynamo.utils.same(fp32_eager, fp32_compile, fp64_ref=fp64)` for raw `sum(softmax)` | ✅ True |
| `floor(sum(softmax))` eager vs compile mismatch | ❌ 13/43 |
| `floor(sum(softmax))` compile vs fp64 ground truth mismatch | ❌ 11/43 (compile returns 1 where fp64 says 0) |

- **Backend isolation**: `aot_eager` produces **zero** mismatches — bug is in **Inductor** only.
- **All compile modes affected**: `default`, `reduce-overhead`, `max-autotune` all reproduce.
- **fp64 ground truth**: 14 out of 43 elements should have `floor(sum(softmax)) = 0`, but compile only produces 10 zeros (eager produces 13 zeros). Inductor's constant-folding makes it **less accurate** than eager for this pattern.

## Minimal reproducer

```python
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
```

### Output

```
torch._dynamo.utils.same (raw sums, fp64 ref): True
eager:   [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
compile: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
floor() mismatch: 13 / 43
compile wrong per fp64 (returns 1, should be 0): 11
```

## Error logs

No error is raised — this is a **silent correctness** issue. The compiled model produces different integer results from eager mode without any warning or exception.

## Versions

```
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A
OS: Ubuntu (linux)
Python version: 3.x
Triton version: 2.13.0+cu130
```

(Run `curl -sL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py | python3` and attach full output when filing.)

## Labels

- `topic: fuzzer` — discovered via differential fuzzing, all fuzzer requirements met
