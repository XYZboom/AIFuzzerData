#!/usr/bin/env python3
"""Minimal reproduction: silu LLVM shufflevector crash"""
import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("x", relax.TensorStructInfo(shape=[4, 5, 2, 1], dtype="float32"))
with bb.function("f", [v]):
    out = bb.emit(relax.op.nn.silu(v))
    bb.emit_func_output(out)
mod = bb.get()

# Crashes here
ex = relax.build(mod, target="llvm")
print("Build OK")