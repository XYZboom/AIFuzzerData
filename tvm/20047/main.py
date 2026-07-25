"""TVM 0.25.0 CUDA dlight gemv FloorDiv Bug — 最小触发程序
==============================================
**根因**: TVM 的 dlight GPU 调度器 gemv.py:228 假设 loop extent
         是常量整数，但 conv2d 的某些形状导致 extent 为 FloorDiv
         符号表达式，.value 属性不存在。

**最小触发条件**: conv2d([1, 1, H, 10], [1, 1, 1, 2]) 其中 H >= 2
                 且输出宽度 W - kW + 1 = 9 导致 FloorDiv extent。

**验证**: LLVM 目标不触发，只在 CUDA 目标下触发。
"""

import tvm
from tvm import relax

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 1, 3, 10]), dtype="float32"))
w = relax.Var("w", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 1, 1, 2]), dtype="float32"))
with bb.function("f", [v, w]):
    c = bb.emit(relax.op.nn.conv2d(v, w,
        strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(c)
mod = bb.get()
ex = relax.build(mod, target="cuda")