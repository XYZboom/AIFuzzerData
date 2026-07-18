# Bug Report

### Is the issue related to model conversion?

No. This bug is triggered by ONNX Runtime's level-2 optimizer (`SimplifiedLayerNormFusion`), not by the ONNX model itself. The ONNX checker validates the model successfully; the error only occurs when ORT applies its graph optimizations during `InferenceSession` creation.

### Describe the bug

ONNX Runtime's **SimplifiedLayerNormFusion** optimizer incorrectly fuses a downstream `Mul` node into a LayerNorm pattern when the `Mul`'s scale operand has a shape that doesn't match the input's last dimension.

The fusion pattern detects:

```
input → [ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div] → layer_norm_out → Mul(scale, layer_norm_out) → output
```

and replaces it with a single `SimplifiedLayerNormalization(scale=scale)` node. However, during the fusion the **original LayerNorm's gamma** (which is a scalar `[]`, broadcastable to any shape) is dropped, and the **downstream Mul's scale** becomes the new `SimplifiedLayerNormalization` scale parameter — even when its shape doesn't match `X.shape[axis:]`. This produces an invalid graph where `SimplifiedLayerNormalization` fails at runtime with:

```
Scale and (optional) bias must match X.shape[axis:] or be NumPy-broadcastable to it.
X.shape={1,1} scale.shape={2} bias.shape={} axis=1
```

### System information

- **ONNX Runtime version**: 1.20.1
- **ONNX version**: 1.17.0
- **Python version**: 3.11.14
- **OS Platform**: Linux Ubuntu 24.04
- **Optimization level**: All (default) — the bug triggers at level 2 (`ORT_ENABLE_EXTENDED`)

### Reproduction instructions

```python
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Constants
eps_t = helper.make_tensor("eps_v", TensorProto.FLOAT, [], [1.0e-5])
two_t = helper.make_tensor("two_v", TensorProto.FLOAT, [], [2.0])
scale_t = helper.make_tensor("scale_v", TensorProto.FLOAT, [2], [0.0, 0.0])

x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
out_vi = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])

# LayerNorm pattern with downstream Mul where scale.shape != X.shape[-1]
nodes = [
    helper.make_node("ReduceMean", inputs=["x"], outputs=["mn"], axes=[-1], keepdims=1),
    helper.make_node("Sub", inputs=["x", "mn"], outputs=["ct"]),
    helper.make_node("Pow", inputs=["ct", "two_v"], outputs=["sq"]),
    helper.make_node("ReduceMean", inputs=["sq"], outputs=["vr"], axes=[-1], keepdims=1),
    helper.make_node("Add", inputs=["vr", "eps_v"], outputs=["ve"]),
    helper.make_node("Sqrt", inputs=["ve"], outputs=["sd"]),
    helper.make_node("Div", inputs=["ct", "sd"], outputs=["nm"]),
    helper.make_node("Mul", inputs=["scale_v", "nm"], outputs=["out"]),
]

graph = helper.make_graph(
    nodes, "g", [x_vi], [out_vi], initializer=[eps_t, two_t, scale_t]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

sess = ort.InferenceSession(model.SerializeToString())
result = sess.run(None, {"x": np.random.randn(1, 1).astype(np.float32)})
```

**Expected result**: Program runs successfully. The LayerNorm produces a `[1, 1]` output, and `Mul([2], [1, 1])` broadcasts to `[1, 2]`.

**Actual result**: ONNX Runtime crashes with:

```
2026-07-18 17:25:35.07519 [E:onnxruntime:, sequential_executor.cc:620 ExecuteKernel]
Non-zero status code returned while running SimplifiedLayerNormalization node.
Name:'/SimplifiedLayerNormFusion/'
Status Message: Scale and (optional) bias must match X.shape[axis:] or be
NumPy-broadcastable to it. X.shape={1,1} scale.shape={2} bias.shape={} and axis=1
```

### Expected behavior

The `SimplifiedLayerNormFusion` optimizer should correctly stop at the LayerNorm's gamma `Mul` when gamma is a scalar (or equivalently, not fuse a downstream `Mul` whose scale shape is incompatible with `X.shape[axis:]`). The fusion should verify that the proposed scale/bias shapes are broadcast-compatible with the input tensor at the normalization axis before committing the fusion.

### Notes

**Trigger conditions** (found via systematic boundary testing of the fusion):

| Condition | Required | Details |
|-----------|----------|---------|
| Input rank | ≥ 2 | e.g., `[1, 1]`, `[N, 1]`, `[N, M, 1]` |
| Input last dim | ≠ scale dim | Typically last dim = 1 |
| LayerNorm pattern | Full sequence required | `ReduceMean[-1,kd1] → Sub → Pow(2) → ReduceMean[-1,kd1] → Add(eps) → Sqrt → Div` |
| Downstream Mul | Any position | Can be graph output or followed by other ops |
| Scale shape | Anything ≠ last dim | `[2]`, `[5]`, `[10]`, `[236]`, etc. |
| Scale source | Initializer or graph input | Both trigger the fusion |

**Minimal graph**: 8 ONNX ops (`[1,1]` input, `[2]` scale).

**Workaround**: Disable the `SimplifiedLayerNormFusion` optimizer:
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# Or more selectively:
sess_options.add_session_config_entry("session.disable_optimizers", "SimplifiedLayerNormFusion")
sess = ort.InferenceSession(model.SerializeToString(), sess_options)
```
