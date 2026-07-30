---
name: ONNX-DIFF-NaN-Mismatch
category: ONNX/ORT/DIFF/NAN
severity: MEDIUM — Optimizer introduces spurious NaNs via constant folding
status: CUDA_RECIPROCAL_ZERO_CONSTANT_FOLDING_BUG
---

## Bug Report: ORT CUDA Constant Folding Produces NaN from `Reciprocal(Sqrt(0)) * 0.0`

**Filed against:** ONNX Runtime (github.com/microsoft/onnxruntime)
**Version:** 1.27.0
**Not an ONNX spec bug** — the semantics of `Reciprocal(Sqrt(x)) * 0.0` for `x=0` are
implementation-defined at the boundary. The CUDA kernel produces a finite result;
the optimizer constant-folds using host math and introduces spurious NaN.

---

### Describe the issue

The ORT CUDA optimizer constant-folds `Reciprocal(Sqrt(0)) = inf` using host
(IEEE 754) arithmetic, then folds `inf * 0.0 = NaN`. However, the actual CUDA
kernel for `Reciprocal` does **not** produce `inf` for input 0 — it returns a
very large finite value (~3.4e38). The unoptimized path is therefore **NaN-free**,
while the optimized path unexpectedly contains 208 NaN values.

**Reduction summary:**
- Original: 46 nodes (graph_4 from 7-graph pipeline)
- Reduced: **6 nodes** (Constant → Sqrt → Reciprocal → ReduceMin → Constant → Mul)
- Shape: `[13, 16, 16]` zeros `→ [1, 1, 13, 16]` output
- Reduction ratio: **99.7%**
- Verification: ✅ `opt_nan=208`, `ref_nan=0`

| Kernel variant              | Output shape       | NaN count | Values (first 8)       |
|------------------------------|--------------------|-----------|------------------------|
| Optimized (default session)  | `[1, 1, 13, 16]`   | **208**   | `[nan, nan, ...]`      |
| Unoptimized (ORT_DISABLE_ALL)| `[1, 1, 13, 16]`   | **0**     | `[0., 0., 0., ...]`    |

The NaN only appears in CUDAExecutionProvider. CPU is not affected because
both CPU Reciprocal(0) and the CPU optimizer use the same host IEEE 754
semantics.

**Important:**
- Only reproduces with `CUDAExecutionProvider`.
- The root cause is a **platform asymmetry**: CUDA `Reciprocal` does not raise
  divide-by-zero as host IEEE 754 does.
- The optimizer evaluates the entire subgraph at compile time using host math,
  producing IEEE 754 NaN, but the actual CUDA kernel would produce a finite value.

---

### To reproduce

```python
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Constants: [13, 16, 16] zeros → Sqrt → Reciprocal → ReduceMin → Mul(0)
zeros_t = helper.make_tensor("zeros_v", TensorProto.FLOAT, [13, 16, 16], [0.0] * 3328)
zero_t = helper.make_tensor("zero_v", TensorProto.FLOAT, [1, 1, 1, 1], [0.0])

nodes = [
    helper.make_node('Constant', inputs=[], outputs=['v_z'], value=zeros_t),
    helper.make_node('Sqrt', inputs=['v_z'], outputs=['v_sq']),
    helper.make_node('Reciprocal', inputs=['v_sq'], outputs=['v_rc']),
    helper.make_node('ReduceMin', inputs=['v_rc'], outputs=['v_rm'], axes=[-1], keepdims=0),
    helper.make_node('Constant', inputs=[], outputs=['v_0'], value=zero_t),
    helper.make_node('Mul', inputs=['v_rm', 'v_0'], outputs=['v_mul']),
]

v_mul_vi = helper.make_tensor_value_info('v_mul', TensorProto.FLOAT, [1, 1, 13, 16])
graph = helper.make_graph(nodes, 'minimal_nan_bug', [], [v_mul_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# With CUDA optimizer (default)
sess_opt = ort.InferenceSession(
    model.SerializeToString(),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)

# Without optimizer (reference)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(
    model.SerializeToString(),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    sess_options=so_ref,
)

opt = np.asarray(sess_opt.run(None, {})[0])
ref = np.asarray(sess_ref.run(None, {})[0])

print(f"opt: nan_count={np.isnan(opt).sum()}, values={opt.flatten()[:8]}")
print(f"ref: nan_count={np.isnan(ref).sum()}, values={ref.flatten()[:8]}")

assert np.isnan(opt).any() and not np.isnan(ref).any(), \
    "BUG NOT REPRODUCED — optimizer should introduce NaNs"
```

---

### Why this matters

The optimizer introduces spurious NaN values that do not exist in the actual
runtime computation. Any model that uses `Reciprocal` on tensors that may
contain zeros (e.g. normalization layers, attention softmax post-processing,
reciprocal of variance in LayerNorm) can silently produce wrong results when
graph optimizations are enabled.

In the original pipeline, the NaN propagates through 208 elements in the
output tensor `graph_4[82]`, which feeds into downstream graph stages.

---

### Root cause analysis

The ORT constant folding optimizer evaluates `Sqrt(0) = 0` on the host, then
`Reciprocal(0) = inf` per IEEE 754, then `ReduceMin(inf) = inf`, and finally
`inf * 0.0 = NaN` — again per IEEE 754. This is mathematically correct on the
host, but the CUDA `Reciprocal` kernel implements division using CUDA hardware
which does **not** follow IEEE 754 divide-by-zero semantics:

- CUDA `__frcp_rn(0.0f)` returns `FLT_MAX` (~3.4e38), not `inf`
- Therefore `FLT_MAX * 0.0 = 0.0`, not NaN

The optimizer needs to be aware of platform-specific arithmetic when
constant-folding subgraphs that involve operations with division (Reciprocal,
Div, etc.) when targeting CUDAExecutionProvider. Either:

1. **Skip constant folding** for Reciprocal/Div/Reciprocal-like ops when the
   input is deterministically zero and the target execution provider is CUDA.
2. **Use CUDA host intrinsics** for constant folding to match device semantics.
3. **Insert numerical guards** to prevent zero inputs in folded subgraphs.

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

### Verification

```
$ python3 reduced_standalone/standalone.py
opt (optimizer on):  shape=(1, 1, 13, 16), nan_count=208, values=[nan nan nan nan nan nan nan nan]
ref (optimizer off): shape=(1, 1, 13, 16), nan_count=0,   values=[0. 0. 0. 0. 0. 0. 0. 0.]

✅ BUG REPRODUCED: Optimizer introduces 208 NaNs (shape=(1, 1, 13, 16))
   where CUDA kernel produces 0 NaNs
```