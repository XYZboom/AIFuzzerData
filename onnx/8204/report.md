# Bug Report

### Is the issue related to model conversion?
No. The issue occurs when constructing a model directly with the ONNX Python API and loading it with ONNX Runtime. It does not involve any external converter.

### Describe the bug
When loading a simple ONNX model that computes a normalized output using a sequence of operations (`ReduceMean`, `Sub`, `Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div`, `Mul`), ONNX Runtime fails with an `InvalidArgument` error:

```
[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'nm' is not a graph input, initializer, or output of a previous node.
```

The error only appears under a specific combination of conditions:
- Opset version = 11
- Both `ReduceMean` nodes have `keepdims=0` (the default is 1)
- The exponent in `Pow` is a scalar constant `2.0` (type `FLOAT`)
- The `eps` constant added to the variance is a scalar (shape `[]`)
- The `Sqrt` output is directly fed into a `Div` node whose result (`nm`) is later used by a `Mul` node.

Changing any of the following individually makes the error disappear:
- Set `keepdims=1` on both `ReduceMean` nodes
- Change the exponent to a different value (e.g., `3.0`)
- Give `eps` a shape `[1]` instead of `[]`
- Use opset version 13 (or 12) instead of 11

This suggests a subtle validation bug in ONNX Runtime's shape inference or constant folding when these specific conditions coincide.

### System information
<!-- Please fill in the details of your environment -->
- OS Platform and Distribution: *e.g., Ubuntu 20.04 / Windows 10*
- ONNX version: *e.g., 1.15.0*
- ONNX Runtime version: *e.g., 1.16.0*
- Python version: *e.g., 3.10*
- GCC/Compiler version: *if applicable*
- Protobuf version: *e.g., 3.20.3*
- Visual Studio version: *if applicable*

### Reproduction instructions
Run the following Python script (requires `onnx` and `onnxruntime` packages):

```python
from onnx import helper, TensorProto
import onnxruntime as ort

eps_t = helper.make_tensor("eps_v", TensorProto.FLOAT, [], [1.0e-5])
two_t = helper.make_tensor("two_v", TensorProto.FLOAT, [], [2.0])

nodes = [
    helper.make_node('Constant', inputs=[], outputs=['eps'], value=eps_t),
    helper.make_node('Constant', inputs=[], outputs=['two'], value=two_t),
    helper.make_node('ReduceMean', inputs=['x'], outputs=['mn'], axes=[-1], keepdims=0),
    helper.make_node('Sub', inputs=['x', 'mn'], outputs=['ct']),
    helper.make_node('Pow', inputs=['ct', 'two'], outputs=['sq']),
    helper.make_node('ReduceMean', inputs=['sq'], outputs=['vr'], axes=[-1], keepdims=0),
    helper.make_node('Add', inputs=['vr', 'eps'], outputs=['ve']),
    helper.make_node('Sqrt', inputs=['ve'], outputs=['sd']),
    helper.make_node('Div', inputs=['ct', 'sd'], outputs=['nm']),
    helper.make_node('Mul', inputs=['nm', 'z'], outputs=['out']),
]

graph = helper.make_graph(
    nodes,
    'graph_0',
    [
        helper.make_tensor_value_info('x', TensorProto.FLOAT, [2]),
        helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 1]),
    ],
    [helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 2])]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# This raises the InvalidArgument error
sess = ort.InferenceSession(model.SerializeToString())
```

**Expected behavior**  
The model should load successfully without errors. A valid ONNX model with these operations should be accepted by ONNX Runtime.

### Notes
- The error is **not** reproducible if any of the following changes are made (each independently fixes the issue):
    - Set `keepdims=1` on the two `ReduceMean` nodes.
    - Change the `Pow` exponent to a constant like `3.0` (while keeping opset 11 and `keepdims=0`).
    - Use a tensor of shape `[1]` for `eps` (e.g., `helper.make_tensor("eps_v", TensorProto.FLOAT, [1], [1e-5])`).
    - Use opset version 12 or 13.

- The error message points to `'nm'` (the output of `Div`) not being recognized as a valid node output, even though it is clearly defined in the graph. This may indicate a bug in the shape inference pass when dealing with reductions with `keepdims=0` combined with `Sqrt` and `Div`.

- The minimal reproducing script is self-contained and does not require any external model file.
```
