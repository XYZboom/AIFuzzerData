### Expected behavior

`relax.build(mod, target="cuda")` should compile successfully for any valid Relax IR module containing `relax.op.nn.conv2d` with valid input shapes. The compiled module should execute and produce correct convolution results.

### Actual behavior

`relax.build` crashes with a `ScheduleError` during the dlight GPU scheduling pass:

```
tvm.s_tir.schedule.schedule.ScheduleError: ScheduleError: An error occurred in the schedule primitive 'bind'.
```

The crash occurs in `tvm/s_tir/dlight/gpu/reduction.py:285` in `_sch_inner_spatial`. The dlight scheduler's `sch.bind(s, "threadIdx.x")` call fails because the SRef tree's child block is neither a local complete block nor a local reduction block, violating the compact dataflow requirement.

The crash happens at build time (before execution) and is deterministic — the same input shape always triggers it.

### Environment

- **OS**: Linux (x86_64, conda environment)
- **GPU**: NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
- **CUDA driver**: 580.76.05
- **TVM version**: 0.25.0.post1 (installed via pip)
- **Target**: `cuda` (GPU compilation)
- **Python**: 3.12

### Steps to reproduce

```python
import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(shape=relax.ShapeExpr([2, 1, 4, 20]), dtype="float32"))
w = relax.Var("w", relax.TensorStructInfo(shape=relax.ShapeExpr([1, 1, 3, 1]), dtype="float32"))
with bb.function("f", [v, w]):
    c = bb.emit(relax.op.nn.conv2d(v, w, strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(c)
mod = bb.get()

# Crashes here during dlight GPU scheduling
ex = relax.build(mod, target="cuda")
```

**Trigger condition**: `relax.op.nn.conv2d` with input shape `[N, C, H, W]` and weight shape `[1, C, KH, 1]` where **all** of the following hold:

| Parameter | Condition | Minimal trigger |
|-----------|-----------|-----------------|
| Batch (N) | ≥ 2 | 2 |
| Input channels (C) | ≥ 1 | 1 |
| Input height (H) | ≥ 4 | 4 |
| Input width (W) | ≥ 18 | 20 |
| Kernel height (KH) | ≥ 2 | 3 |

Boundary testing results:

| Input shape | Weight shape | Result |
|-------------|-------------|--------|
| `[2, 1, 4, 20]` | `[1, 1, 3, 1]` | ❌ BUG |
| `[2, 1, 4, 18]` | `[1, 1, 3, 1]` | ❌ BUG |
| `[2, 1, 4, 16]` | `[1, 1, 3, 1]` | ✅ OK |
| `[1, 1, 4, 20]` | `[1, 1, 3, 1]` | ✅ OK (N=1) |
| `[2, 1, 3, 20]` | `[1, 1, 3, 1]` | ✅ OK (H=3) |
| `[2, 1, 2, 20]` | `[1, 1, 3, 1]` | ✅ OK (H=2) |
| `[2, 1, 4, 20]` | `[1, 1, 1, 1]` | ✅ OK (KH=1) |
| `[2, 1, 4, 20]` | `[1, 1, 2, 1]` | ❌ BUG |

The bug is specific to the CUDA target (`target="cuda"`). With `target="llvm"` the same program compiles successfully.

### Triage

* bug
* CUDA
* dlight
* reduction
* conv2d
* codegen
* needs-triage