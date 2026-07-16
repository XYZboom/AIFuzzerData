from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[1, 4, 6, 3], dtype="float32"))
with bb.function("f", [v]):
    p = bb.emit(relax.op.nn.avg_pool2d(
        v, pool_size=[2, 2], strides=[1, 1], padding=[0, 0]
    ))
    bb.emit_func_output(p)
mod = bb.get()

# Crashes here — never reaches execution
ex = relax.build(mod, target="llvm")