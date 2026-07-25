### Expected behavior

TVM Relax should successfully compile and execute a conv2d operator with output spatial dimensions greater than 1 on CUDA target. The dlight GPU scheduler's gemv path should handle non-unit output spatial dimensions gracefully.

### Actual behavior

`relax.build` crashes with `AssertionError` at `gemv.py:228`:

```
File "/root/miniconda3/lib/python3.12/site-packages/tvm/s_tir/dlight/gpu/gemv.py", line 228, in apply
    assert sch.get(ts_o).extent.value == 1
AssertionError
```

```
Target cuda missing 'max_shared_memory_per_block'; using 49152 bytes.
Traceback (most recent call last):
  File "/root/Code/kotlin/aifuzzer/daemon/tvm_daemon.py", line 72, in run_source
    exec(source, {"tvm": tvm, "relax": relax, "op": op})
  File "<string>", line 203, in <module>
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/relax/vm_build.py", line 270, in build
    mod = relax_pipeline(mod)
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/ir/transform.py", line 171, in __call__
    return _ffi_transform_api.RunPass(self, mod)
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/s_tir/dlight/base/transform.py", line 88, in _apply_rules
    space = rule.apply(func, target, tunable)
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/s_tir/dlight/gpu/gemv.py", line 79, in apply
    return self.sch_inner_reduction(sch, target, block, vector_input_buffers, epilogue)
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/s_tir/dlight/gpu/gemv.py", line 412, in sch_inner_reduction
    return apply(
  File "/root/miniconda3/lib/python3.12/site-packages/tvm/s_tir/dlight/gpu/gemv.py", line 228, in apply
    assert sch.get(ts_o).extent.value == 1
AssertionError
```

### Environment

- **OS**: Linux (x86_64, conda environment)
- **GPU**: NVIDIA RTX 3080 Ti (12GB)
- **CUDA driver**: (use `nvidia-smi`)
- **Compiler version**: TVM 0.25.0.post1
- **Target**: `cuda`
- **Python**: 3.12

### Steps to reproduce

```python
import tvm
from tvm import relax
import numpy as np

bb = relax.BlockBuilder()

v_input = relax.Var("input", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 1, 6, 8]), dtype="float32"))
v_weight = relax.Var("weight", relax.TensorStructInfo(
    shape=relax.ShapeExpr([1, 1, 1, 3]), dtype="float32"))

with bb.function("main", [v_input, v_weight]):
    v_out = bb.emit(relax.op.nn.conv2d(
        v_input, v_weight,
        strides=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))
    bb.emit_func_output(v_out)

mod = bb.get()
ex = relax.build(mod, target="cuda")  # crashes here
```

**Minimal trigger condition**: `conv2d([1, 1, H, W], [1, 1, kH, kW])` where:
- N=1, C_in=1, C_out=1
- Output spatial H ≥ 6 (with stride=1, padding=0)
- Output spatial W ≥ 2

### Boundary testing results

**Three distinct error regions discovered at `gemv.py:228`:**

| Output H | Behavior | Error Type |
|----------|----------|------------|
| 1–2 | ✅ passes | — |
| 3–5 | ❌ `FloorDiv` | `AttributeError: 'FloorDiv' object has no attribute 'value'` |
| 6+ | ❌ `AssertionError` | `assert sch.get(ts_o).extent.value == 1` failed |

**This bug report covers the AssertionError variant (H ≥ 6).**

Full boundary matrix (conv2d with `[1,1,1,3]` kernel, stride=1, padding=0):

| Condition | Parameter | Result |
|-----------|-----------|--------|
| Output H | 1–2 | ✅ passes |
| Output H | 3–5 | ❌ FloorDiv (known bug) |
| Output H | 6+ | ❌ **AssertionError (this bug)** |
| Batch dim N | 1 | ❌ |
| Batch dim N | 2+ | ✅ passes |
| C_out | 1 | ❌ |
| C_out | 2+ | ✅ passes |
| C_in | 1 | ❌ |
| C_in | 2+ | ✅ passes |
| Kernel W | 1 | ✅ passes |
| Kernel W | 2+ | ❌ AssertionError |
| Stride | 1 | ❌ AssertionError |
| Stride | 2+ | ✅ passes (falls into FloorDiv or OK region) |
| Kernel H | 1 | ❌ AssertionError |
| Kernel H | 2+ | ✅ passes (output H ≤ 5, falls into FloorDiv region) |

### Relationship to known FloorDiv bug

The same assertion `gemv.py:228` produces **two different errors** depending on the output spatial dimension:

| Bug | Output H | `extent` type | Error |
|-----|----------|---------------|-------|
| FloorDiv bug (known) | 3–5 | `FloorDiv` (symbolic) | `AttributeError` — no `.value` attr |
| This bug (new variant) | 6+ | `IntImm` (constant) | `AssertionError` — `.value` ≠ 1 |

Both stem from the same root cause: `gemv.py:228`'s `assert sch.get(ts_o).extent.value == 1` assumes the output spatial loop extent is always 1, which is incorrect for conv2d with non-unit output spatial dimensions. However, they manifest differently because TVM's lowering generates different TIR for small vs. large spatial dimensions.

### Triage

* bug
* cuda
* dlight-gpu-gemv
* needs-triage