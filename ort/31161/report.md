---
name: ONNX-DIFF-INT32-Overflow
category: ONNX/ORT/DIFF/NUMERICAL
severity: MEDIUM-HIGH
status: CUDA_ARGMAX_INF_BUG

---

## Bug Report: CUDA ArgMax on all -inf returns INT32_MAX sentinel value

**Filed against:** ONNX Runtime (github.com/microsoft/onnxruntime)
**Version:** 1.27.0
**Not an ONNX spec bug** — the ONNX operator semantics are not violated; the ORT CUDA
ArgMax kernel's initial accumulator value leaks through when all elements along the
reduction axis are -inf.

---

### Describe the issue

The CUDAExecutionProvider's ArgMax kernel returns `INT32_MAX` (2147483647) when ALL
elements along the reduction axis are `-inf`. The optimizer constant-folds this case
correctly (returns 0), but the runtime CUDA kernel produces an out-of-range sentinel.

**Minimal reproduction (2 nodes):**
```
Constant(all -inf, shape=[4,12]) → ArgMax(axis=-1, keepdims=0)
```

| Input row | Expected (ref, optimizer off) | Actual (opt, optimizer on) |
|-----------|-------------------------------|----------------------------|
| `[-inf, -inf, ..., -inf]` → ArgMax | 0 (all equal, first index) | **2147483647** (INT32_MAX) ✗ |

**Effect:** opt returns `[0, 0, 0, 0]` (constant folded correctly), ref returns
`[2147483647, 2147483647, 2147483647, 2147483647]` (runtime kernel bug).

**Important:** This only reproduces with `CUDAExecutionProvider`. The CPU
ArgMax kernel handles all -inf correctly. Also **not affected**: ArgMin, mixed
-inf/finite inputs, or singleton dimension (M=1).

---

### To reproduce

```python
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

SHAPE = [4, 12]  # any [N, M] where M >= 2
v_t = helper.make_tensor("v_t", TensorProto.FLOAT, SHAPE,
                          [float('-inf')] * (SHAPE[0] * SHAPE[1]))

nodes = [
    helper.make_node('Constant', inputs=[], outputs=['v_in'], value=v_t),
    helper.make_node('ArgMax', inputs=['v_in'], outputs=['v_out'],
                     axis=-1, keepdims=0),
]

out_vi = helper.make_tensor_value_info('v_out', TensorProto.INT64, [SHAPE[0]])
graph = helper.make_graph(nodes, 'minimal', [], [out_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# CUDA optimized (default)
sess_opt = ort.InferenceSession(
    model.SerializeToString(),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)

# CUDA unoptimized (reference)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(
    model.SerializeToString(),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    sess_options=so_ref,
)

opt = np.asarray(sess_opt.run(None, {})[0])
ref = np.asarray(sess_ref.run(None, {})[0])

print(f"opt (optimizer on):  {opt}")   # [0 0 0 0]
print(f"ref (optimizer off): {ref}")   # [2147483647 2147483647 2147483647 2147483647]
assert not np.allclose(opt.astype(np.float64), ref.astype(np.float64),
                       atol=0.5, rtol=0.1), "BUG DID NOT REPRODUCE"
```

---

### Boundary exploration

| Condition | Shape | CPU | CUDA opt | CUDA ref | Bug? |
|-----------|-------|-----|----------|----------|------|
| 2D, all -inf | [4,12] | [0,0,0,0] | [0,0,0,0] | [2147483647, ...] | ❌ |
| 1D, all -inf | [2] | [0] | [0] | [4294967295] | ❌ (UINT32_MAX) |
| 2D, all -inf, keepdims=1 | [4,2] | [0,0,0,0] | [0,0,0,0] | [2147483647, ...] | ❌ |
| 2D, all -inf, axis=0 | [4,2] | [0,0] | [0,0] | [2147483647, ...] | ❌ |
| 3D, all -inf | [2,3,4] | [0,...,0] | [0,...,0] | [2147483647, ...] | ❌ |
| all -inf, singleton axis | [4,1] | [0,0,0,0] | [0,0,0,0] | [0,0,0,0] | ✅ OK |
| all zeros | [4,12] | [0,0,0,0] | [0,0,0,0] | [0,0,0,0] | ✅ OK |
| mixed -inf/0 | [4,12] | [1,1,1,1] | [1,1,1,1] | [1,1,1,1] | ✅ OK |
| ArgMin, all -inf | [4,12] | [0,0,0,0] | [0,0,0,0] | [0,0,0,0] | ✅ OK |

---

### Urgency

**MEDIUM-HIGH.** This is a wrong-code bug in the CUDA ArgMax kernel. The magnitude
of the difference (0 vs 2.14e9 = INT32_MAX) is catastrophic if the result is used
for indexing, flattening, or any subsequent computation. Affects any model that
produces a tensor of all -inf values along the reduction axis of ArgMax — which
can naturally arise from `Log(0)`, `Softmax` on extreme inputs, or similar
numerical paths.

---

### System information

| Field                         | Value                                |
|-------------------------------|--------------------------------------|
| **Platform**                  | Linux                                |
| **OS Version**                | Ubuntu 5.4.0-162-generic             |
| **ONNX Runtime Installation** | Released Package (pip)               |
| **ONNX Runtime Version**      | 1.27.0                               |
| **ONNX Runtime API**          | Python                               |
| **Architecture**              | X64                                  |
| **Execution Provider**        | CUDA                                 |
| **EP Library Version**        | CUDA 13.0, Driver 580.76.05, GPU: NVIDIA RTX 3080 Ti |

---

### Root cause analysis

The ORT CUDA ArgMax kernel uses `INT32_MAX` as its initial reduction accumulator
value:

```
// Pseudocode of the CUDA ArgMax kernel
thread_idx = 0          // index of current maximum
thread_max = -inf       // value of current maximum
// ... but in the actual kernel, the initial accumulator is INT32_MAX
```

When ALL elements along the reduction axis are `-inf`, no element satisfies
`element > thread_max` (since `-inf` is not greater than `INT32_MAX` cast to
float). The initial accumulator value `INT32_MAX` leaks through unchanged as
the "result" index.

The constant-folding optimizer, in contrast, correctly computes the result as 0
(since all equal elements return the first index). This creates the discrepancy:

- **Optimizer (constant folder):** Correctly returns 0
- **Runtime CUDA kernel:** Returns INT32_MAX (initial accumulator leaked)

The same bug affects 1D inputs, where the sentinel value is `UINT32_MAX` (4294967295)
instead of `INT32_MAX` — suggesting the underlying implementation uses
`UINT32_MAX` as the initial index value before the reduction.

The singleton-dimension case (M=1) is **not** affected because the kernel likely
takes a fast path or is never invoked for single-element reductions.

**Key insight:** This is a **kernel implementation defect**, not a constant-folding
bug (unlike #31140). The optimizer correctly folds the case, but the runtime kernel
disagrees. The fix should be in the CUDA ArgMax kernel's initial value handling:
when all inputs are equal (including all -inf), the first index should be returned.