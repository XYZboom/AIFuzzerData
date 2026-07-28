import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[1], dtype="float32"))
with bb.function("f", [v]):
    out = bb.emit(relax.op.astype(relax.op.argmin(v, axis=-1), dtype="float32"))
    bb.emit_func_output(out)
mod = bb.get()

# Crashes here during CUDA codegen
ex = relax.build(mod, target="cuda")