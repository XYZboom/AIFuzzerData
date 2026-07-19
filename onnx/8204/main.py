from onnx import helper, TensorProto
import onnxruntime as ort

eps_t   = helper.make_tensor("eps_v",   TensorProto.FLOAT, [], [1.0e-5])
two_t   = helper.make_tensor("two_v",   TensorProto.FLOAT, [], [2.0])

nodes = [
    helper.make_node('Constant', inputs=[], outputs=['eps'],   value=eps_t),
    helper.make_node('Constant', inputs=[], outputs=['two'],   value=two_t),
    helper.make_node('ReduceMean', inputs=['x'], outputs=['mn'], axes=[-1], keepdims=0),  # keepdims=0
    helper.make_node('Sub',  inputs=['x', 'mn'],  outputs=['ct']),
    helper.make_node('Pow',  inputs=['ct', 'two'], outputs=['sq']),
    helper.make_node('ReduceMean', inputs=['sq'], outputs=['vr'], axes=[-1], keepdims=0), # keepdims=0
    helper.make_node('Add',  inputs=['vr', 'eps'], outputs=['ve']),
    helper.make_node('Sqrt', inputs=['ve'],       outputs=['sd_raw']),
    helper.make_node('Identity', inputs=['sd_raw'], outputs=['sd']),
    helper.make_node('Div', inputs=['ct', 'sd'], outputs=['nm']),
    helper.make_node('Mul', inputs=['nm', 'z'], outputs=['out']),
]

graph = helper.make_graph(nodes, 'graph_0', [
    helper.make_tensor_value_info('x', TensorProto.FLOAT, [2]),
    helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 1]),
], [helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 2])])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
sess = ort.InferenceSession(model.SerializeToString())