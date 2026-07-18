# Minimal reproduction of ONNX Runtime SimplifiedLayerNormFusion bug
# Reduced from 55 nodes (original) → 8 nodes
# Bug: ORT optimizer incorrectly fuses downstream Mul into LayerNorm,
# creating SimplifiedLayerNormalization with scale.shape=[2] that doesn't
# match X.shape[axis:] = [1]

from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Constants
eps_t = helper.make_tensor("eps_v", TensorProto.FLOAT, [], [1.0e-5])
two_t = helper.make_tensor("two_v", TensorProto.FLOAT, [], [2.0])
scale_t = helper.make_tensor("scale_v", TensorProto.FLOAT, [2], [0.0, 0.0])

x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
out_vi = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])

# LayerNorm: ReduceMean→Sub→Pow→ReduceMean→Add→Sqrt→Div
# Then downstream Mul(scale[2]) triggers the fusion bug
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
result = sess.run(
    None, {"x": np.random.randn(1, 1).astype(np.float32)}
)
print("Execution: OK")
