"""
TVM 0.25.0 CUDA buffer_red Bug — 最小触发程序
=============================================
**根因**: TVM 的 TIR→CUDA 代码生成中，BATCH_NORM 在特定形状下
         reduction buffer (*_red) 声明在首次使用之后，违反 SSA 支配。
         storage_rewrite.cc:1328 VectorTypeAccessChecker::OnArrayAccess

**最小触发条件**: BATCH_NORM 对 axis=1，输入形状 [1, N, 1]（batch=1, last dim=1）
                 任意 N 都能触发。单 op 即够。

**触发形状规律**:
  - [1, N, 1] axis=1 → ❌ buffer_red (任意 N)
  - [2, N, 1] axis=1 → ✅ 通过 (batch > 1)
  - [1, N, 2] axis=1 → ✅ 通过 (last dim > 1)
  - [1, 1, 2] axis=2 → ❌ buffer_red
  - [1, N, 1] axis=0 → ✅ 通过
  - [1, N, 1] axis=2 → ✅ 通过

**验证**: 本地 LLVM 目标不触发，只在 CUDA 目标下触发。
"""

import tvm
from tvm import relax
import numpy as np

bb = relax.BlockBuilder()
v = relax.Var("v", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 2, 1]), dtype="float32"))
with bb.function("f", [v]):
    bn = bb.emit(relax.op.nn.batch_norm(v,
        gamma=relax.op.ones(relax.ShapeExpr([2]), dtype="float32"),
        beta=relax.op.zeros(relax.ShapeExpr([2]), dtype="float32"),
        moving_mean=relax.op.zeros(relax.ShapeExpr([2]), dtype="float32"),
        moving_var=relax.op.ones(relax.ShapeExpr([2]), dtype="float32"),
        axis=1))
    out = bb.emit(relax.TupleGetItem(bn, 0))
    bb.emit_func_output(out)

mod = bb.get()
try:
    ex = relax.build(mod, target="cuda")
    vm = relax.VirtualMachine(ex, tvm.cuda())
    np_v = np.random.uniform(0.0, 1.0, size=(1, 2, 1)).astype(np.float32)
    tvm_v = tvm.runtime.tensor(np_v, device=tvm.cuda())
    result = vm["f"](tvm_v)
    print("✅ Build + VM execution succeeded")
    print(f"   Output shape: {list(result.numpy().shape)}")
except Exception as e:
    err = str(e)
    if "buffer" in err and "_red" in err:
        print(f"❌ buffer_red! {err}")
    else:
        print(f"❌ {err}")