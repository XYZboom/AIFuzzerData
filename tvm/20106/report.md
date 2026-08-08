# TVM CUDA `cumprod` launch grid overflow (CUDA_ERROR_INVALID_VALUE)

- **Bug ID**: 014
- **Backend**: TVM Relax (daemon)
- **Seed**: 20261649
- **Original dir**: `bug_014_TVM_Relax_(daemon)_seed20261649_1_20260808_095843/`
- **Reduced script**: `reduced.py`

## Expected behavior

`relax.op.cumprod` is a valid Relax operator. `relax.build(mod, target="cuda")` should compile successfully, and the compiled module should execute and return correct cumulative-product results for any valid input shape.

## Actual behavior

The CUDA `cumprod` kernel launch throws `CUDALaunch Error: CUDA_ERROR_INVALID_VALUE`:

```
tvm.error.InternalError: CUDALaunch Error: CUDA_ERROR_INVALID_VALUE
 grid=(1,100800,1),  block=(1024,1,1)
// func_name=cumprod_kernel
```

The crash happens at **execution time** (kernel launch), not at build time. The kernel launch config `grid=(1, rows, 1)` puts `gridDim.y = rows`, which exceeds the CUDA hardware limit of 65535, so the launch fails. The error is deterministic — the same input shape always triggers it.

## Environment

- **OS**: Linux (x86_64, conda environment)
- **GPU**: NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
- **CUDA driver**: 580.76.05
- **TVM version**: 0.25.0.post1 (pip install)
- **Target**: `cuda`
- **Python**: 3.12.13

## Steps to reproduce

```python
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
result = vm["main"](t_in)   # crashes here
```

## Trigger condition

The trigger condition is unique: **`rows = Π(all dims except the cumprod axis) >= 65536`**. The kernel places `rows` into `gridDim.y`, whose CUDA hard limit is 65535.

Boundary test results (dims along the cumprod axis are excluded from the `rows` product):

| Shape | cumprod axis | rows (non-axis product) | Result |
|-------|-------------|-------------------------|--------|
| `[65535, 1]` | 1 | 65535 | ✅ passes |
| `[65536, 1]` | 1 | 65536 | ❌ crashes |
| `[1, 65536]` | 0 | 65536 | ❌ crashes |
| `[65536, 1, 1]` | 1 | 65536 | ❌ crashes |
| `[1, 65536, 1]` | 1 | 1 | ✅ passes |
| `[256, 256, 1]` | 1 | 256 | ✅ passes |
| `[256, 256, 1]` | 2 | 65536 | ❌ crashes |
| `[24, 4, 14, 300]` (original) | 1 | 100800 | ❌ crashes |

**Conclusion**: as long as the product of all dims except the cumprod axis exceeds 65535, the crash occurs regardless of axis position, rank, or shape layout. The trigger depends only on the numeric value of `rows`.

## Root cause

TVM's CUDA `cumprod` kernel generator (`cumprod_kernel`) maps each independent cumprod row to `blockIdx.y`:

```c
extern "C" __global__ void __launch_bounds__(1024) cumprod_kernel(float* __restrict__ output_buf, float* __restrict__ v) {
  if (((int)threadIdx.x) < 4) {
    output_buf[((((int)blockIdx.y) * 4) + ((int)threadIdx.x))] = v[...];
  }
}
```

At launch, the kernel puts the **row count** (product of all dims except the cumprod axis) into `gridDim.y`. When the row count exceeds CUDA's 65535 limit for `gridDim.y`, `cudaLaunchKernel` returns `CUDA_ERROR_INVALID_VALUE`, which TVM wraps into an `InternalError` at `cuda_module.cc:305`.

**Fix directions**:
- Split the `gridDim.y` row index across `gridDim.z` (limit 65535) + `gridDim.y`, or use `gridDim.x` (limit 2^31-1) to carry the row count;
- Or launch in batches (loop) when the row count exceeds a single-launch limit;
- Or process multiple rows per block (increase block dim, reduce grid dim).

## Reduction walkthrough

1. Original program: 310 lines, 3 chained graphs, 403,200 total elements; crash point is `cumprod` in `graph_1` (input `[24, 4, 14, 300]`, `rows = 24*14*300 = 100800`).
2. Manually stripped to a standalone `cumprod` reproducer: `[24, 4, 14, 300]` → `[1, 4, 1, 65536]` → `[65536, 4]` → `[65536, 2]` → `[65536, 1]`.
3. Minimal reproducer `[65536, 1]` (only 65,536 elements, `rows = 65536`), ~15x reduction, error signature identical to the original (`cumprod_kernel` / `CUDA_ERROR_INVALID_VALUE`).

The reduced script is saved as `reduced.py` in the same directory.

## Triage

* bug
* backend:cuda
* cumprod
* launch-bounds
* grid-dim
* needs-triage