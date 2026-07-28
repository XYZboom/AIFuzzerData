import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[6, 6], dtype="float32"))
with bb.function("f", [v]):
    r1 = bb.emit(relax.op.rsqrt(relax.op.astype(v, dtype="float32")))
    s = bb.emit(relax.op.sigmoid(r1))
    m1 = bb.emit(relax.op.mean(s, axis=[-1], keepdims=False))
    m2 = bb.emit(relax.op.mean(m1, axis=[-1], keepdims=False))
    mish = bb.emit(relax.op.multiply(m1, relax.op.tanh(
        relax.op.log(relax.op.add(relax.const(1.0, dtype="float32"), relax.op.exp(m1))))))
    d = bb.emit(relax.op.divide(relax.op.astype(mish, dtype="float32"),
                                relax.op.astype(m2, dtype="float32")))
    out = bb.emit(relax.op.expand_dims(d, axis=0))
    bb.emit_func_output(out)
mod = bb.get()

# Crashes here during dLight CUDA scheduling
ex = relax.build(mod, target="cuda")