### Expected behavior

`relax.build(mod, target="cuda")` should compile successfully for any valid Relax IR module, including `relax.op.argmin` with any valid input shape. The compiled module should execute and produce correct argmin results.

### Actual behavior

`relax.build` crashes with an `InternalError` during CUDA codegen:

```
tvm.error.InternalError: Check failed: scope != "global" (global vs. global) : Cannot allocate global memory when targeting CUDA. You must pass all global arrays as input instead
```

The crash occurs in `codegen_cuda.cc:759` (`PrintStorageScope`). When `argmin` (or any reduce-to-scalar op) produces a 0D (scalar) output, the CUDA code generator assigns a global storage scope to the reduction buffer, but CUDA kernels cannot allocate global memory — all global arrays must be passed as inputs. The `PrintStorageScope` check catches this inconsistency and asserts.

The crash is deterministic — same input shape always triggers it, and it happens at build time (before execution).

### Environment

- **OS**: Linux (x86_64, conda environment)
- **GPU**: NVIDIA GeForce RTX 3080 Ti (12GB VRAM, CUDA 580.76.05)
- **TVM version**: 0.25.0.post1
- **Target**: `cuda` (GPU compilation)
- **Python**: 3.12

### Steps to reproduce

```python
import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[1], dtype="float32"))
with bb.function("f", [v]):
    out = bb.emit(relax.op.astype(relax.op.argmin(v, axis=-1), dtype="float32"))
    bb.emit_func_output(out)
mod = bb.get()

# Crashes here during CUDA codegen
ex = relax.build(mod, target="cuda")
```

**Trigger condition**: Any reduce-to-scalar op (`argmin`, `argmax`, `reduce_mean`, `reduce_sum`, `reduce_max`, `reduce_min`) that produces a **0D (scalar) output** on the CUDA target.

Boundary testing results:

| Op | Input shape | Target | Result |
|----|-------------|--------|--------|
| `argmin` | `[1]` | `cuda` | ❌ crashes |
| `argmin` | `[2]` | `cuda` | ❌ crashes |
| `argmin` | `[N]` (any N) | `cuda` | ❌ crashes |
| `argmin` | `[1]` | `llvm` | ✅ passes |
| `argmax` | `[1]` | `cuda` | ❌ crashes |
| `reduce_mean` | `[1]` | `cuda` | ❌ crashes |
| `reduce_sum` | `[1]` | `cuda` | ❌ crashes |
| `reduce_max` | `[1]` | `cuda` | ❌ crashes |
| `reduce_min` | `[1]` | `cuda` | ❌ crashes |

**All reduce-to-scalar ops crash on CUDA.** The LLVM target (`target="llvm"`) compiles successfully with the same input.

The crash is NOT specific to 1D input — any reduction along the last axis that produces a scalar (e.g., `reduce_mean` on `[1, 1]` with `axis=[-1]`) will trigger the same bug.

### Root cause

The CUDA code generator (`codegen_cuda.cc`) uses `PrintStorageScope` to determine the CUDA storage class (`__shared__` or `__global__`) for each buffer. When a reduction produces a scalar output, the reduction buffer is assigned `"global"` scope, but:

1. The buffer is **not** a function parameter — it's an internal temporary
2. CUDA kernels cannot allocate global memory internally
3. The codegen asserts `scope != "global"` for non-parameter buffers

The fix should either:
- Change the storage scope of scalar reduction buffers to `"local"` or `"register"`
- Or allocate the buffer properly as a global parameter

### Variants

See `variant/` directory for additional instances:
- 3 additional instances found via fuzzing (different shapes, same root cause)

### Triage

* bug
* backend:cuda
* codegen
* reduce-to-scalar
* needs-triage