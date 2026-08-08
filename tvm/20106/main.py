# Reduced reproducer for bug 014 (original: cumprod on [24,4,14,300] axis=1)
# Root cause: TVM CUDA cumprod_kernel launches with grid=(1, rows, 1) where
#   rows = product of all dims except the cumprod axis. CUDA gridDim.y max = 65535.
#   rows=65536 -> CUDA_ERROR_INVALID_VALUE.
# Minimal shape: [65536, 1] -> rows=65536, only 65536 elements total.

import tvm
from tvm import relax
import numpy as np

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(
    shape=relax.ShapeExpr([65536, 1]), dtype="float32"))
with bb.function("main", [v]):
    out = bb.emit(relax.op.cumprod(v, axis=1))
    bb.emit_func_output(out)

mod = bb.get()
ex = relax.build(mod, target="cuda")
vm = relax.VirtualMachine(ex, tvm.cuda())

np_in = np.random.uniform(0.0, 1.0, size=(65536, 1)).astype(np.float32)
t_in = tvm.runtime.tensor(np_in, device=tvm.cuda())
result = vm["main"](t_in)
print("Done:", result.numpy().shape)