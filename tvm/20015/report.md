### Expected behavior

`relax.build(mod, target="llvm")` should compile successfully for any valid Relax IR module, including `avg_pool2d` with any valid input shape. The compiled module should execute and produce correct average pooling results.

### Actual behavior

`relax.build` crashes with an `InternalError` during LLVM codegen verification:

```
tvm.error.InternalError: LLVM module verification failed with the following errors: 
Instruction does not dominate all uses!
  %314 = shufflevector <4 x float> %287, <4 x float> %311, <2 x i32> <i32 2, i32 6>
  %90 = shufflevector <2 x float> %314, <2 x float> %89, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
```

The generated LLVM IR contains a `shufflevector` instruction whose operand is defined in a block that does not dominate the use site, violating SSA dominance. The crash is deterministic — same input shape always triggers it, and it happens at build time (before execution).

### Environment

- **OS**: Linux (x86_64, conda environment)
- **TVM version**: 0.25.0.post1
- **Target**: `llvm` (CPU compilation)
- **Python**: 3.11
- **LLVM triple**: `x86_64-conda-linux-gnu`

### Steps to reproduce

```python
import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[1, 4, 6, 3], dtype="float32"))
with bb.function("f", [v]):
    p = bb.emit(relax.op.nn.avg_pool2d(
        v, pool_size=[2, 2], strides=[1, 1], padding=[0, 0]
    ))
    bb.emit_func_output(p)
mod = bb.get()

# Crashes here during LLVM codegen
ex = relax.build(mod, target="llvm")
```

**Trigger condition**: The bug occurs when **all** of the following hold for the input tensor shape `[N, C, H, W]`:

| Parameter | Triggering Value |
|-----------|-----------------|
| Channels (`C`) | exactly **4** |
| Width (`W`) | exactly **3** |
| Height (`H`) | **≥ 6 and even** |

Verified with the following shape matrix:

| Shape | Result |
|-------|--------|
| `[1, 4, 6, 3]` | **CRASH** |
| `[1, 4, 8, 3]` | **CRASH** |
| `[1, 4, 10, 3]` | **CRASH** |
| `[1, 4, 5, 3]` | OK (H=5, odd) |
| `[1, 4, 7, 3]` | OK (H=7, odd) |
| `[1, 3, 6, 3]` | OK (C=3) |
| `[1, 5, 6, 3]` | OK (C=5) |
| `[1, 4, 6, 2]` | OK (W=2) |
| `[1, 4, 6, 4]` | OK (W=4) |
| `[1, 6, 6, 3]` | OK (C=6) |

Additional notes:

- `max_pool2d` with the **same shape** does **not** crash — the bug is specific to `avg_pool2d`.
- `adaptive_avg_pool2d` with equivalent output shape also does not crash.
- The Relax-level IR is correct; the generated TIR is also correct. The crash is in the TIR → LLVM codegen phase (`tvm::codegen::CodeGenLLVM::Verify()`).
- The issue appears related to vectorization: `C=4` maps to 4-wide float vectors, and `W=3` does not align with the vector width, causing a `shufflevector` to be emitted in a non-dominating block.

### Triage

* needs-triage
* bug
* backend:llvm
* frontend:relax