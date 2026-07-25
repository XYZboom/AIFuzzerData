"""TVM 0.25.0 CUDA dlight reduction conv2d bind Bug
=====================================================
**根因**: TVM 的 dlight GPU 调度器 reduction.py:285 在
         _sch_inner_spatial 中尝试 sch.bind(s, "threadIdx.x") 时，
         卷积的 TIR SRef 树子块不满足 compact dataflow 条件。

**最小触发**: conv2d([2,1,4,20], [1,1,3,1]) — 单算子即可触发。
             需要 N>=2, C>=1, H>=4, W>=20, KH>=2。

**触发规律**:
  - [2, 1, 4, 20] x [1, 1, 3, 1]  → BUG!
  - [2, 1, 4, 18] x [1, 1, 3, 1]  → BUG! (W=18)
  - [2, 1, 4, 16] x [1, 1, 3, 1]  → OK (W=16)
  - [1, 1, 4, 20] x [1, 1, 3, 1]  → OK (N=1)
  - [2, 1, 4, 20] x [1, 1, 1, 1]  → OK (KH=1)
"""

import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(
    shape=relax.ShapeExpr([2, 1, 4, 20]), dtype="float32"))
w = relax.Var("w", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 1, 3, 1]), dtype="float32"))
with bb.function("f", [v, w]):
    c = bb.emit(relax.op.nn.conv2d(v, w,
        strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(c)
mod = bb.get()
relax.build(mod, target="cuda")
