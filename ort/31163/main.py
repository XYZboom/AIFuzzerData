from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

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

print("=== Model structure ===")
for i, n in enumerate(graph.node):
    print(f"  n[{i}] {n.op_type}: {list(n.input)} -> {list(n.output)}")

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

print(f"opt (optimizer on):  shape={opt.shape}, nan_count={np.isnan(opt).sum()}, values={opt.flatten()[:8]}")
print(f"ref (optimizer off): shape={ref.shape}, nan_count={np.isnan(ref).sum()}, values={ref.flatten()[:8]}")
