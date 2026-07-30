---
name: ONNX-DIFF-INF-SIGN-FLIP
category: ONNX/ORT/DIFF/SIGN
severity: MEDIUM — Potential correctness bug
status: CUDA_UNOPTIMIZED_NEGATIVE_ZERO_BUG
---

## Bug Report: ORT Abs CUDA Kernel Returns -0.0 for +0.0 Input

**Filed against:** ONNX Runtime (github.com/microsoft/onnxruntime)
**Version:** 1.27.0
**Not an ONNX spec bug** — the `Abs` operator semantics are clear: `Abs(x) >= 0`
for all `x`. A CUDA kernel that produces `-0.0` from `+0.0` violates this invariant.

---

### Describe the issue

The `Abs` CUDA kernel, when run without graph optimizations (i.e. the actual
runtime kernel, not a constant-folded path), returns `-0.0` for `+0.0` input.

**Reduction summary:**
- Original: 78 nodes (graph_1 from multi-graph pipeline)
- Reduced: **2 nodes, scalar shape** (Constant → Abs, shape=[], 108 bytes)
- Reduction ratio: **97.4%**
- Verification: ✅ `signbit(ref)` is True

| Kernel variant              | Input  | Output   | signbit(.0) | Mathematically correct? |
|------------------------------|--------|----------|-------------|-------------------------|
| Optimized (default session)  | `+0.0` | `+0.0`   | False       | ✅ Yes                  |
| Unoptimized (OPT_DISABLE_ALL)| `+0.0` | **-0.0** | **True**    | ❌ **No**               |

The `-0.0` from `Abs` may go undetected by tolerance-based checks
(`np.allclose(-0.0, +0.0, atol=0.5, rtol=0.1)` is True), but it is
mathematically incorrect. Any downstream op sensitive to the sign of zero
amplifies the bug:

```
Abs(+0.0) = -0.0  →  Reciprocal(-0.0) = -inf   vs   Reciprocal(+0.0) = +inf
Abs(+0.0) = -0.0  →  Div(1, -0.0)    = -inf   vs   Div(1, +0.0)    = +inf
Abs(+0.0) = -0.0  →  Sqrt(-0.0)      = -0.0   vs   Sqrt(+0.0)      = +0.0
```

**Important:**
- Only reproduces with `CUDAExecutionProvider`. CPU is not affected.
- `Abs(-0.0)` correctly returns `+0.0` in both paths — the sign bit is being
  **generated**, not propagated from the input.

---

### To reproduce

```python
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Tensor: scalar zero
z_t = helper.make_tensor("z0_v", TensorProto.FLOAT, [], [0.0])

nodes = [
    helper.make_node("Constant", inputs=[], outputs=["v_c"], value=z_t),
    helper.make_node("Abs", inputs=["v_c"], outputs=["v_out"]),
]

v_vi = helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [])
graph = helper.make_graph(nodes, "minimal_abs_bug", [], [v_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

# With CUDA optimizer (default)
sess_opt = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# With CUDA optimizer disabled (reference)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    sess_options=so_ref,
)

opt = np.asarray(sess_opt.run(None, {})[0])
ref = np.asarray(sess_ref.run(None, {})[0])

print(f"opt (optimizer on):  signbit={np.signbit(opt).any()}, values={opt.flatten()[:8]}")
print(f"ref (optimizer off): signbit={np.signbit(ref).any()}, values={ref.flatten()[:8]}")

# np.allclose(-0.0, +0.0) is True, so this won't catch it:
print(f"allclose: {np.allclose(opt, ref)}")

# But signbit reveals the bug:
assert np.signbit(ref).any(), "BUG DID NOT REPRODUCE — no negative zero in unoptimized output"
```

---

### Why tolerance-based diff detection misses this

The aiFuzzer pipeline compares `opt` vs `ref` with `np.allclose(a, b, atol=0.5, rtol=0.1)`.
Since `np.allclose(-0.0, +0.0) == True`, the `-0.0` from `Abs` alone is invisible.
It only surfaces when a downstream op like `Reciprocal` converts the sign:

```
Abs(+0.0) = -0.0  →  Reciprocal(-0.0) = -inf   vs   Reciprocal(+0.0) = +inf
```

This means the bug affects all pipelines where `Abs(zeros)` feeds into:
- `Reciprocal` / `Div` (1/0 → ±inf sign flip)
- `Sqrt` (√-0.0 preserves sign, affecting downstream ops)
- `Log` (log(-0.0) = -inf, log(+0.0) = -inf — same result, so Log is safe)
- Any op that distinguishes `±0.0` (e.g. IEEE 754 `signbit`, `copysign`)

---

### Urgency

Medium. The `Abs` kernel is widely used in normalization layers (LayerNorm,
BatchNorm), activation functions, and loss computations. Any model that applies
`Abs` to a tensor that may contain zeros and then uses the result in
sign-sensitive operations (reciprocal, division) can silently produce wrong
outputs when graph optimizations are disabled (e.g. in production inference
with `ORT_DISABLE_ALL` or when the optimizer cannot fold the subgraph).

---

### System information

| Field                         | Value                                |
|-------------------------------|--------------------------------------|
| **Platform**                  | Linux                                |
| **OS Version**                | 5.4.0-162-generic                    |
| **ONNX Runtime Installation** | Released Package (pip)               |
| **ONNX Runtime Version**      | 1.27.0                               |
| **ONNX Runtime API**          | Python                               |
| **Architecture**              | X64                                  |
| **Execution Provider**        | CUDA                                 |
| **EP Library Version**        | CUDA 13.0, Driver 580.76.05, GPU: NVIDIA GeForce RTX 3080 Ti |

---

### Root cause analysis

The optimized path likely constant-folds `Abs(0.0)` at graph-optimization time
(computing `0.0 → 0.0` on the host), which correctly preserves the positive
sign. The unoptimized path executes the actual CUDA kernel at runtime, and that
kernel produces `-0.0` for `+0.0`.

The CUDA kernel for `Abs` is typically implemented as a bitwise AND with the
sign-bit mask: `__float_as_int(x) & 0x7FFFFFFF`. This operation should clear
the sign bit unconditionally, so producing `-0.0` from `+0.0` suggests one of:

1. **The kernel incorrectly uses signed subtraction instead of bitwise mask**
   (e.g. `x >= 0 ? x : -x` where `-0.0` on CUDA can produce sign-bit issues).
2. **A fused-multiply-add (FMA) or other arithmetic operation is being used**
   internally that introduces a spurious sign bit.
3. **An optimization pass within the CUDA compiler** (nvcc) is rearranging
   the abs computation in a way that introduces negative zero.

Regardless of the mechanism, the result violates the mathematical invariant
`Abs(x) >= 0` and is a correctness bug in the CUDA kernel.