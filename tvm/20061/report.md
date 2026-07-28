### Expected behavior

`relax.build(mod, target="cuda")` should compile successfully for any valid Relax IR module. The compiled module should execute and produce correct results.

### Actual behavior

`relax.build` crashes with an `InternalError` during the CUDA dLight scheduling pass:

```
tvm.error.InternalError: Check failed: block_realize == old_block_realize_.get() (0x9a04a00 vs. 0x99fa0f0) :
```

The crash occurs in `s_tir/schedule/primitive/reduction.cc:1166` (`BlockReplacer::VisitStmt_`) during the `rfactor` transformation invoked by `dlight/gpu/reduction.py:197` (`_sch_inner_reduction`).

The error is in the TIR schedule's `SBlock` mutation: when `rfactor` is applied to a reduction loop, the `BlockReplacer` mutator replaces the `SBlock` node, but the cached `block_realize` pointer in the `SBlock` is not updated to match the new block. The assert `block_realize == old_block_realize_.get()` fails because the old `SBlock`'s `BlockRealize` was already replaced by a previous mutator pass.

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
v = relax.Var("x", relax.TensorStructInfo(shape=[6, 6], dtype="float32"))
with bb.function("f", [v]):
    r1 = bb.emit(relax.op.rsqrt(relax.op.astype(v, dtype="float32")))
    s = bb.emit(relax.op.sigmoid(r1))
    m1 = bb.emit(relax.op.mean(s, axis=[-1], keepdims=False))
    m2 = bb.emit(relax.op.mean(m1, axis=[-1], keepdims=False))
    mish = bb.emit(relax.op.multiply(m1, relax.op.tanh(
        relax.op.log(relax.op.add(relax.const(1.0, dtype="float32"), relax.op.exp(m1))))))
    d = bb.emit(relax.op.divide(relax.op.astype(mish, dtype="float32"),
                relax.op.astype(m2, dtype="float32")))
    out = bb.emit(relax.op.expand_dims(d, axis=0))
    bb.emit_func_output(out)
mod = bb.get()

# Crashes here during dLight CUDA scheduling
ex = relax.build(mod, target="cuda")
```

**Trigger condition**: The bug occurs when **all** of the following hold:

1. There is a **chain of two consecutive reductions** (`reduce_mean` → `reduce_mean`), where the first reduces 2D→1D and the second reduces 1D→0D
2. The 1D intermediate result is used by **both** the second reduction AND a separate element-wise computation (e.g., `mish = f(m1)`)
3. The element-wise computation has a **different shape path** than the reduction chain, creating a divergent TIR structure
4. Target is `cuda`

The crash is specific to the CUDA target. The LLVM target (`target="llvm"`) compiles the same module successfully.

### Root cause

The dLight GPU reduction scheduler (`reduction.py`) applies `rfactor` to split reduction loops for GPU parallelism. The `BlockReplacer` in `reduction.cc` replaces `SBlock` nodes during the transformation. However, when a block is referenced by both the original `SBlock` pointer and a cached copy, the replacement creates a stale pointer:

1. The first `reduce_mean` lowers to TIR with `SBlock` A
2. The second `reduce_mean` lowers to TIR with `SBlock` B  
3. dLight processes both reductions, calling `rfactor` on B
4. `rfactor`'s `BlockReplacer` replaces `SBlock` B → B'
5. But `SBlock` A still holds a reference to the old `BlockRealize` of B
6. When the assert `block_realize == old_block_realize_.get()` runs, it compares the old (pre-replacement) pointer against the new one → mismatch → crash

### Variants

See `variant/` directory for additional instances:
- 4 additional instances found via fuzzing (different shapes and op chains, same root cause)

### Triage

* bug
* backend:cuda
* dlight
* rfactor
* s-tir
* needs-triage