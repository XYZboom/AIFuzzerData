"""
ONNX-DIFF-INF-SIGN-FLIP — Minimal Reproducer (2 nodes)
==============================================
Root Cause: ORT unoptimized Abs CUDA kernel returns -0.0 for +0.0 input.

Chain: Constant(0.0) -> Abs -> output = -0.0 (unopt) vs +0.0 (opt)

Even 2 nodes are enough: Abs alone produces -0.0 on unoptimized CUDA.
Reciprocal downstream just amplifies -0.0 into visible -inf vs +inf.

CPU is NOT affected.
"""

from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Make tensor: scalar zero
z_t = helper.make_tensor("z0_v", TensorProto.FLOAT, [], [0.0])

nodes = [
    helper.make_node('Constant', inputs=[], outputs=['v_c'], value=z_t),
    helper.make_node('Abs', inputs=['v_c'], outputs=['v_out']),
]

v_vi = helper.make_tensor_value_info('v_out', TensorProto.FLOAT, [])
graph = helper.make_graph(nodes, 'minimal_abs_bug', [], [v_vi])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# Save model
with open('minimal.onnx', 'wb') as f:
    f.write(model.SerializeToString())

# Run optimized
sess_opt = ort.InferenceSession(model.SerializeToString(),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Run unoptimized
so_ref = ort.SessionOptions()
so_ref.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_ref = ort.InferenceSession(model.SerializeToString(),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=so_ref)

opt = np.asarray(sess_opt.run(None, {})[0])
ref = np.asarray(sess_ref.run(None, {})[0])

print(f"opt={opt}")
print(f"ref={ref}")