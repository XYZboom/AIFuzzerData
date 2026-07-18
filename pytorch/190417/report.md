---
name: "🐛 torch.compile: layer_norm on log2(zeros) triggers sympy NaN in value_ranges (bounds.py:sub)"
labels: ["oncall: pt2", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` (inductor backend) crashes with `AssertionError: sympy expression is NaN` when the model computes `layer_norm(log2(zeros(...)))`. 

**Root cause**: `zeros(2,1)` → `log2(0) = -inf` → `layer_norm(-inf)` triggers inductor's bounds analysis to compute `sub(-oo, -oo) = nan` → `ValueRanges.__init__` → `simple_sympify` raises because sympy `nan` is not allowed.

Eager mode succeeds; only the `torch.compile` path crashes.

### Minimal reproducer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        v0 = torch.zeros((2, 1), dtype=torch.float32, device="cpu")
        v1 = torch.log2(v0.float())
        v2 = F.layer_norm(v1.float(), (v1.shape[-1],))
        return v2

model = TestModule()

# Eager — OK
with torch.no_grad():
    ref = model()

# Compiled — crashes
compiled = torch.compile(model, mode="default")
with torch.no_grad():
    cmp = compiled()
```

### Crash site

```
torch/utils/_sympy/value_ranges.py:73, in simple_sympify
    raise AssertionError("sympy expression is NaN")
```

### Call chain (simplified)

1. `layer_norm`'s inductor lowering creates a reduction body
2. Bounds analysis (`bounds.py`) calls `sub(a, b)` on value ranges from `log2(zeros(...))`
3. `sub` → `add(a, neg(b))` → `coordinatewise_increasing_map(fn)` → `fn(x.lower, y.lower)`
4. The sympy `add` produces `nan` (e.g. `oo + (-oo)`)
5. `ValueRanges.__init__` → `simple_sympify(nan)` → raises

### Ablation

- `torch.compile(backend="eager")` — no crash (bypasses inductor)
- `torch.compile(backend="inductor")` — crashes
- Replacing `log2` with `relu` or any non-inf-producing op — no crash
- Replacing `layer_norm` with a pointwise op (e.g. `torch.add`) — no crash (no reduction loop → no bounds sub)

### Related issues

- [#188225](https://github.com/pytorch/pytorch/issues/188225) — *"index_propagation ValueError: The argument 'nan' is not comparable"*, a sibling bug with the same root cause (sympy NaN through different op path: `log2 → atan2 → arccosh → softmax` → crash in `index_propagation`)
- [#130585](https://github.com/pytorch/pytorch/pull/130585) (merged) — *"[Inductor] Avoid nan in value_ranges"*, fixed a similar NaN issue in `safe_mul` but the `sub` path remains unprotected

---

## Error logs

```
Traceback (most recent call last):
  File "pytorch_daemon.py", line 91, in run_source
    exec(source, {
  File "<string>", line 17, in <module>
  File ".../torch/_dynamo/eval_frame.py", line 511, in __call__
    return super().__call__(*args, **kwargs)
  File ".../torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File ".../torch/_dynamo/eval_frame.py", line 1183, in compile_wrapper
    raise e.remove_dynamo_frames() from None
  File ".../torch/_inductor/compile_fx.py", line 1079, in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
  File ".../torch/_inductor/compile_fx.py", line 1059, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
  File ".../torch/_inductor/compile_fx.py", line 1847, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
  File ".../torch/_inductor/compile_fx.py", line 1608, in codegen_and_compile
    compiled_module = graph.compile_to_module()
  File ".../torch/_inductor/graph.py", line 2669, in compile_to_module
    return self._compile_to_module()
  File ".../torch/_inductor/graph.py", line 2675, in _compile_to_module
    self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
  File ".../torch/_inductor/graph.py", line 2611, in codegen
    self.scheduler.codegen()
  File ".../torch/_inductor/scheduler.py", line 9206, in codegen
    self._codegen_partitions()
  File ".../torch/_inductor/scheduler.py", line 9354, in _codegen_partitions
    self._codegen(partition)
  File ".../torch/_inductor/scheduler.py", line 9508, in _codegen
    self.get_backend(device).codegen_node(node)
  File ".../torch/_inductor/codegen/cpp.py", line 5551, in codegen_node
    cpp_kernel_proxy.codegen_nodes(nodes)
  File ".../torch/_inductor/codegen/cpp.py", line 4714, in codegen_nodes
    self.codegen_functions(fn_list, var_sizes_list)
  File ".../torch/_inductor/codegen/cpp.py", line 4503, in codegen_functions
    scalar_kernel = codegen_kernel(self.kernel_cls)
  File ".../torch/_inductor/codegen/cpp.py", line 4480, in codegen_kernel
    run(kernel)
  File ".../torch/_inductor/codegen/cpp.py", line 4492, in run
    fn(vars, reduction_vars)
  File ".../torch/_inductor/codegen/cpp.py", line 4695, in fn
    return node.codegen(index_vars)
  File ".../torch/_inductor/scheduler.py", line 2495, in codegen
    with (
  File ".../contextlib.py", line 137, in __enter__
    return next(self.gen)
  File ".../torch/_inductor/codegen/common.py", line 2206, in set_current_node
    self.node_to_bounds = node._body.bounds().get_bounds()
  File "<string>", line 6, in get_bounds_cache_on_self
  File ".../torch/_inductor/bounds.py", line 82, in get_bounds
    interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
  File ".../torch/_inductor/loop_body.py", line 64, in run
    return super().run(*args, **kwargs)
  File ".../torch/fx/interpreter.py", line 197, in run
    self.env[node] = self.run_node(node)
  File ".../torch/_inductor/loop_body.py", line 60, in run_node
    return super().run_node(n)
  File ".../torch/fx/interpreter.py", line 294, in run_node
    return getattr(self, n.op)(n.target, args, kwargs)
  File ".../torch/fx/interpreter.py", line 402, in call_method
    return getattr(self_obj, target)(*args_tail, **kwargs)
  File ".../torch/_inductor/bounds.py", line 267, in sub
    return cls.add(a, cls.neg(b))
  File ".../torch/utils/_sympy/value_ranges.py", line 653, in add
    return ValueRanges.coordinatewise_increasing_map(
  File ".../torch/utils/_sympy/value_ranges.py", line 413, in coordinatewise_increasing_map
    return ValueRanges(
  File ".../torch/utils/_sympy/value_ranges.py", line 158, in __init__
    lower = simple_sympify(lower)
  File ".../torch/utils/_sympy/value_ranges.py", line 73, in simple_sympify
    raise AssertionError("sympy expression is NaN")
torch._inductor.exc.InductorError: AssertionError: sympy expression is NaN
```

---

## Versions

```
PyTorch version: 2.13.0+cu130
Python version:  3.11.15 (main, Jun 11 2026) [GCC 14.3.0]
OS:              Linux-6.17.0-35-generic-x86_64-with-glibc2.39
CPU:             x86_64 (AVX512)
CUDA:            not available (CPU-only)
```

---

## Additional context

Discovered via [aiFuzzer](https://github.com/XYZboom/aiFuzzer) — an AI compiler fuzzing framework.

The full reduction chain (original 72 ops → 3 ops minimal reproducer) is available alongside this report. Key sympy code in `value_ranges.py` line 70–75 already acknowledges this class of bug:

```python
# NaNs can occur when doing things like 0 * sympy.oo, but it is better
# if the operator notices this and takes care of it, because sometimes
# the NaN is inappropriate (for example, for ints, the [-oo, oo] range
# should go to zero when multiplied with [0, 0])
if e == sympy.nan:
    raise AssertionError("sympy expression is NaN")
```

The comment says NaN should be handled upstream, but `bounds.py:sub` (and potentially other operators) does not guard against it. Either `ValueRanges.add`/`sub` should clamp NaN to the valid range, or the `layer_norm` lowering should avoid bounds analysis on `inf`-containing ranges.
