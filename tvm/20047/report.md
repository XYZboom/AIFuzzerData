### Expected behavior

`relax.build(mod, target="cuda")` should compile successfully for any valid Relax IR module containing `relax.op.nn.conv2d`. The compiled module should execute and produce correct convolution results.

### Actual behavior

`relax.build` crashes with an `AttributeError` during the dlight GPU scheduling pass:

```
AttributeError: 'FloorDiv' object has no attribute 'value'
```

The crash occurs in `tvm/s_tir/dlight/gpu/gemv.py:228` where `assert sch.get(ts_o).extent.value == 1` assumes the loop extent is a constant integer, but the extent is a `FloorDiv` symbolic expression. The crash happens at build time (before execution) and is deterministic — the same input shape always triggers it.

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
v = relax.Var("v", relax.TensorStructInfo(shape=relax.ShapeExpr([1, 1, 3, 10]), dtype="float32"))
w = relax.Var("w", relax.TensorStructInfo(shape=relax.ShapeExpr([1, 1, 1, 2]), dtype="float32"))
with bb.function("f", [v, w]):
    c = bb.emit(relax.op.nn.conv2d(v, w, strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(c)
mod = bb.get()

# Crashes here during dlight GPU scheduling
ex = relax.build(mod, target="cuda")
```

**Trigger condition**: `relax.op.nn.conv2d` with input shape `[1, 1, H, W]` and weight shape `[1, 1, 1, kW]` where H ≥ 2 and kW = 2, and the resulting output spatial dimension `W - kW + 1` is not divisible by the kernel size `kW` in a way that produces a `FloorDiv` extent in the dlight schedule.

Boundary testing results:

| Input shape | Weight shape | Result |
|-------------|-------------|--------|
| `[1, 1, 3, 10]` | `[1, 1, 1, 2]` | ❌ crashes |
| `[1, 1, 2, 10]` | `[1, 1, 1, 2]` | ❌ crashes |
| `[1, 1, 3, 32]` | `[1, 1, 1, 2]` | ❌ crashes |
| `[1, 1, 3, 10]` | `[1, 1, 1, 1]` | ✅ passes |
| `[1, 1, 3, 10]` | `[1, 1, 1, 3]` | ✅ passes |
| `[1, 1, 1, 10]` | `[1, 1, 1, 2]` | ✅ passes (H=1) |
| `[1, 1, 3, 4]` | `[1, 1, 1, 2]` | ✅ passes |
| `[1, 1, 3, 9]` | `[1, 1, 1, 2]` | ✅ passes |
| `[1, 1, 3, 10]` | `[1, 1, 2, 2]` | ✅ passes |
| `[1, 1, 3, 10]` | `[1, 1, 3, 2]` | ✅ passes |

The bug is specific to the CUDA target (`target="cuda"`). The LLVM target (`target="llvm"`) may not trigger the same code path.

### Triage

* bug
* CUDA
* dlight
* gemv
* conv2d
* needs-triage