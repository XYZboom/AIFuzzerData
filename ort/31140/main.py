from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# ========== Constants ==========
input_t = helper.make_tensor("v29", TensorProto.FLOAT, [2, 2],
                             [0.0, 0.0, 0.0, 0.0])
mask_t = helper.make_tensor("mk", TensorProto.BOOL, [2, 2],
                            [1, 1, 0, 1])
z_t = helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])

# ========== Graph: 6 nodes ==========
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
graph = helper.make_graph(nodes, "minimal_2x2", [], [out_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

# ========== Test with CUDA ==========
sess_opt = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    sess_options=so_ref,
)

opt = np.asarray(sess_opt.run(None, {})[0])
ref = np.asarray(sess_ref.run(None, {})[0])

print(f"opt (CUDA optimizer enabled):  {opt.flatten()}")
print(f"ref (CUDA optimizer disabled): {ref.flatten()}")
