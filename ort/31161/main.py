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