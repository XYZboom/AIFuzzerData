import tvm
from tvm import relax
import numpy as np

bb = relax.BlockBuilder()

# Minimal trigger: single conv2d, N=1, C_in=1, C_out=1, output spatial dims > 1
v_input = relax.Var("input", relax.TensorStructInfo(shape=relax.ShapeExpr([1, 1, 6, 8]), dtype="float32"))
v_weight = relax.Var("weight", relax.TensorStructInfo(shape=relax.ShapeExpr([1, 1, 1, 3]), dtype="float32"))

with bb.function("main", [v_input, v_weight]):
    v_out = bb.emit(relax.op.nn.conv2d(v_input, v_weight, strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(v_out)

mod = bb.get()

ex = relax.build(mod, target="cuda")
vm = relax.VirtualMachine(ex, tvm.cuda())

np_input = np.random.uniform(0.0, 1.0, size=(1, 1, 6, 8)).astype(np.float32)
np_weight = np.random.uniform(-0.1, 0.1, size=(1, 1, 1, 3)).astype(np.float32)

result = vm["main"](tvm.runtime.tensor(np_input, device=tvm.cuda()), tvm.runtime.tensor(np_weight, device=tvm.cuda()))
print("Execution: OK")