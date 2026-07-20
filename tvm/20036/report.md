### Expected behavior

`relax.build(mod, target="cuda")` should compile successfully for any valid Relax IR module, including `relax.op.nn.batch_norm` with any valid input shape. The compiled module should execute and produce correct batch normalization results.

### Actual behavior

`relax.build` crashes with an `InternalError` during CUDA codegen:

```
tvm.error.InternalError: Check failed: (it != info_map_.end()) is false: Load/Store of buffer v_red (0x9c5db70) occurred before its declaration.
```

The generated CUDA code contains a reduction buffer (`*_red`) that is used **before** its declaration, violating SSA dominance. The crash happens at build time (before execution) and is deterministic — the same input shape always triggers it.

### Environment

- **OS**: Linux (x86_64, remote GPU server)
- **GPU**: NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
- **CUDA driver**: 580.76.05
- **TVM version**: 0.25.0.post1 (installed via pip)
- **Target**: `cuda` (GPU compilation)
- **Python**: 3.12

### Steps to reproduce

```python
import tvm
from tvm import relax
import numpy as np

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 2, 1]), dtype="float32"))
with bb.function("f", [v]):
    bn = bb.emit(relax.op.nn.batch_norm(v,
        gamma=relax.op.ones(relax.ShapeExpr([2]), dtype="float32"),
        beta=relax.op.zeros(relax.ShapeExpr([2]), dtype="float32"),
        moving_mean=relax.op.zeros(relax.ShapeExpr([2]), dtype="float32"),
        moving_var=relax.op.ones(relax.ShapeExpr([2]), dtype="float32"),
        axis=1))
    out = bb.emit(relax.TupleGetItem(bn, 0))
    bb.emit_func_output(out)

mod = bb.get()
ex = relax.build(mod, target="cuda")  # crashes with buffer_red error
```

**Trigger condition**: `relax.op.nn.batch_norm` on a 3D (or 2D/4D) tensor where **batch dimension = 1 AND the last dimension = 1**, with normalization along `axis=1` (or equivalently, the last axis when last dim = 1).

Boundary testing results:

| Shape | axis | Result |
|-------|------|--------|
| `[1, N, 1]` (any N) | 1 | ❌ crashes |
| `[2, N, 1]` (any N) | 1 | ✅ passes |
| `[4, N, 1]` (any N) | 1 | ✅ passes |
| `[1, N, 2]` (any N) | 1 | ✅ passes |
| `[1, 1, 2]` | 2 | ❌ crashes |
| `[1, N, 1]` | 0 | ✅ passes |
| `[1, N, 1]` | 2 | ✅ passes |

The bug is specific to the CUDA target (`target="cuda"`). The LLVM target (`target="llvm"`) compiles successfully with the same input.

### Triage

* bug
* CUDA
* codegen
* tir
* needs-triage