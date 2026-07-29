---
name: ONNX-DIFF-INDEX-VALUE-Mismatch
category: ONNX/ORT/DIFF/NUMERICAL
severity: MEDIUM
status: CUDA_OPTIMIZER_NAN_BUG

## Bug Report: CUDA Optimizer Constant-Folding Bug — ArgMin on NaN

**Filed against:** ONNX Runtime (github.com/microsoft/onnxruntime)
**Version:** 1.27.0
**Not an ONNX spec bug** — the ONNX operator semantics are not violated; the ORT CUDA
optimizer's constant-folding pass produces a result that disagrees with the runtime kernel.

---

### Describe the issue

The CUDAExecutionProvider's graph optimizer constant-folds `ArgMin` on inputs
containing NaN values. The constant-folded result disagrees with the actual
runtime CUDA kernel, producing wrong indices.

**Computation chain:**
```
Constant(zeros[2,2]) → LogSoftmax → Log → Where(mask, NaN, 0) → ArgMin
```

Since `Log(LogSoftmax(zeros))` = `Log(-log(2))` = NaN, the `Where` node produces
a tensor with a mixture of NaN and 0.0 values. The CUDA optimizer constant-folds
this entire subgraph but computes `ArgMin` using different NaN semantics than the
runtime kernel.

| Input row                     | Expected (ref, optimizer off) | Actual (opt, optimizer on) |
|-------------------------------|-------------------------------|----------------------------|
| `[NaN, NaN]` → ArgMin         | 0                             | 0 (agrees)                 |
| `[0, NaN]` → ArgMin           | 1 (index of 0.0, non-NaN min)| **0** (index of NaN) ✗     |

**Effect:** The optimizer returns `[0, 0]` instead of `[0, 1]`.

**Important:** This only reproduces with `CUDAExecutionProvider`. The CPU
optimizer handles NaN correctly.

---

### To reproduce

```python
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

input_t = helper.make_tensor("v29", TensorProto.FLOAT, [2, 2],
                             [0.0, 0.0, 0.0, 0.0])
mask_t = helper.make_tensor("mk", TensorProto.BOOL, [2, 2],
                            [1, 1, 0, 1])
z_t = helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])

nodes = [
    helper.make_node("Constant", inputs=[], outputs=["v29"], value=input_t),
    helper.make_node("LogSoftmax", inputs=["v29"], outputs=["v34"], axis=-1),
    helper.make_node("Log", inputs=["v34"], outputs=["v39"]),
    helper.make_node("Constant", inputs=[], outputs=["mk"], value=mask_t),
    helper.make_node("Constant", inputs=[], outputs=["z"], value=z_t),
    helper.make_node("Where", inputs=["mk", "v39", "z"], outputs=["v44"]),
    helper.make_node("ArgMin", inputs=["v44"], outputs=["result"], axis=-1, keepdims=0),
]

out_vi = helper.make_tensor_value_info("result", TensorProto.INT64, [2])
graph = helper.make_graph(nodes, "minimal", [], [out_vi])
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

print(f"opt (optimizer on):  {opt}")   # [0 0]
print(f"ref (optimizer off): {ref}")   # [0 1]
assert not np.allclose(opt, ref), "BUG DID NOT REPRODUCE"
```

---

### Urgency

Medium. This affects any model where the CUDA optimizer constant-folds `ArgMin`
on NaN-containing tensors. The bug can silently produce wrong index values in
differentiable indexing pipelines (e.g., attention masking, argmax-based
selection, nearest-neighbor search).

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

The ORT CUDA optimizer's constant-folding pass evaluates `ArgMin` during graph
optimization, but applies different NaN semantics than the runtime CUDA kernel:

- **Runtime CUDA ArgMin kernel:** NaN values are treated as unordered — they do
  not participate in the minimum comparison. `ArgMin([0, NaN])` → index `1` (the
  0.0 wins). This matches IEEE 754 semantics where NaN comparisons are false.
- **Optimizer constant folder:** NaN is treated as if it were the minimum value,
  returning its index. `ArgMin([0, NaN])` → index `0` (the NaN "wins").

This is a constant-folding correctness bug: the optimizer's pre-computed
result contradicts the actual kernel execution. The same issue likely affects
`ArgMax` as well.