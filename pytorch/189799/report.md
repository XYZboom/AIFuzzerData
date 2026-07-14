---
name: "🐛 [Bug A] Inductor systematically mishandles negative zero (-0.0) sign across multiple operations (add, elu, sqrt+rsqrt fusion)"
about: Create a report to help us reproduce and fix the bug
labels: ["oncall: pt2", "topic: fuzzer", "module: inductor"]
---

## 🐛 Describe the bug

`torch.compile` with the `inductor` backend systematically mishandles the sign of negative zero (-0.0) across multiple operations. This is a **systemic issue** — three distinct code paths exhibit the same root cause: inductor fails to preserve IEEE 754 semantics for -0.0, resulting in downstream sign flips of infinity.

> **Duplicate check**: Related issues #185610 (minimum/maximum signed zeros, fixed by #185765) and #187353 (Triton neg lowering, fixed) and #185878 (expm1 subnormal, fixed) addressed some signed-zero bugs in Inductor, but the three manifestations below are **not covered** by those fixes.

### Manifestation 1: `add(-0.0, +0.0)` preserves -0.0 (violates IEEE 754)

**Root cause**: `torch.add(-0.0, torch.zeros_like(-0.0))` should produce `+0.0` per IEEE 754-2008 (Section 6.3: "When neither input is NaN, the sign of the sum is determined by the rules of addition... The sum of two operands with different signs (or the difference of two operands with like signs) is exactly zero with sign +0.0"). Eager and aot_eager correctly implement this. Inductor incorrectly preserves `-0.0`.

**Minimal reproducer** (`bug_008_minimal.py`):
```python
import torch
import torch.nn as nn

class BugModule(nn.Module):
    def forward(self, x):
        v = torch.ceil(x)          # ceil(-0.5) = -0.0
        v = torch.add(v, torch.zeros_like(v))  # should be +0.0, inductor gives -0.0
        v = torch.reciprocal(v)    # reciprocal(+0.0) = +inf, reciprocal(-0.0) = -inf
        return v

x = torch.tensor([-0.5, -0.1, -0.9], dtype=torch.float32)

with torch.no_grad():
    eager_out = BugModule()(x)
with torch.no_grad():
    aot_out = torch.compile(BugModule(), backend="aot_eager")(x)
with torch.no_grad():
    ind_out = torch.compile(BugModule(), mode="default")(x)

print("Eager:  ", eager_out.tolist())   # [inf, inf, inf]
print("aot:    ", aot_out.tolist())     # [inf, inf, inf]
print("Induct: ", ind_out.tolist())     # [-inf, -inf, -inf]
```

### Manifestation 2: `sqrt+rsqrt` fusion loses the sign correction of `add(-0.0, +0.0)`

**Root cause**: The chain `add(-0.0, +0.0) → sqrt → rsqrt` is fused into `x^(-0.25)` by inductor, which bypasses the `add` operation's sign correction (`-0.0 + 0.0 = +0.0`). The fusion directly computes `rsqrt(sqrt(-0.0))` = `rsqrt(-0.0)` = `-inf`, whereas the correct result is `+inf`.

**Minimal reproducer** (`bug_009_minimal.py`):
```python
import torch
import torch.nn as nn

class BugModule(nn.Module):
    def forward(self, x):
        y = torch.add(x, torch.zeros_like(x))  # converts -0.0 to +0.0 per IEEE 754
        a = torch.sqrt(y)
        b = torch.rsqrt(a)
        return b

x = torch.zeros(4, 4, dtype=torch.float32, device="cpu")
x[0, 0] = -0.0
x[0, 1] = -0.0
x[1, 0] = 0.0   # positive zero (control)
x[1, 1] = 1.0   # normal value (control)
x[2, 0] = 0.5
x[2, 1] = -0.0

with torch.no_grad():
    ref = BugModule()(x.double()).float()
with torch.no_grad():
    out_aot = torch.compile(BugModule(), backend="aot_eager")(x)
with torch.no_grad():
    out_ind = torch.compile(BugModule(), mode="default")(x)

print("aot vs fp64:", torch._dynamo.utils.same(out_aot, ref, fp64_ref=ref.double()))
print("ind vs fp64:", torch._dynamo.utils.same(out_ind, ref, fp64_ref=ref.double()))
print("inf sign diff:", (torch.isinf(out_aot) & torch.isinf(out_ind) & (out_aot != out_ind)).sum().item())
# Output: aot=+inf (correct), inductor=-inf (wrong) at positions with -0.0 input
```

### Manifestation 3: `F.elu(-0.0)` loses negative sign

**Root cause**: `F.elu(-0.0, alpha=1.14)` in eager mode outputs `-0.0` (preserving sign), but the inductor Triton kernel for ELU outputs `+0.0`. This causes `rsqrt(0.0) = +inf` instead of `rsqrt(-0.0) = -inf`.

**Minimal reproducer** (`bug_014_minimal.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinModule(nn.Module):
    def forward(self, x):
        s = torch.sign(x)
        m = torch.mean(s.float(), dim=-1, keepdim=False)
        c = torch.ceil(m)          # can produce -0.0
        e = F.elu(c, alpha=1.14)   # elu(-0.0) = -0.0 in eager, +0.0 in inductor
        return torch.rsqrt(e)      # rsqrt(-0.0) = -inf vs rsqrt(+0.0) = +inf

torch.manual_seed(42)
x = torch.randn(33, 57, 32, dtype=torch.float32, device="cpu")

with torch.no_grad():
    ref = MinModule()(x)
with torch.no_grad():
    out_aot = torch.compile(MinModule(), backend="aot_eager")(x)
with torch.no_grad():
    out_ind = torch.compile(MinModule(), mode="default")(x)

print("aot vs eager:", torch._dynamo.utils.same(ref, out_aot))  # True
print("ind vs eager:", torch._dynamo.utils.same(ref, out_ind))  # False

# Direct elu test
neg_zero = torch.full((1, 8), -0.0, dtype=torch.float32)
eager_elu = F.elu(neg_zero, alpha=1.14)
compiled_elu = torch.compile(lambda t: F.elu(t, alpha=1.14), mode="default")(neg_zero)
print("eager elu(-0.0):", eager_elu[0, 0].item())     # -0.0
print("compiled elu(-0.0):", compiled_elu[0, 0].item())  # 0.0
```

## Expected behavior

All three manifestations should produce the same results as eager mode:
1. `add(-0.0, +0.0)` → `+0.0` (IEEE 754-2008 Section 6.3)
2. `sqrt(rsqrt(add(-0.0, +0.0)))` → `+inf` (sign correction should not be bypassed by fusion)
3. `F.elu(-0.0)` → `-0.0` (sign should be preserved)

## Error logs

```
=== Manifestation 1 (bug_008) ===
Input: [-0.5, -0.1, -0.9]
Eager:   [inf, inf, inf]
aot_eager: [inf, inf, inf]
Inductor: [-inf, -inf, -inf]
Inf sign mismatches: 3/3

=== Manifestation 2 (bug_009) ===
inf sign differences (aot=+inf, ind=-inf): 3
BUG: inductor produces -inf instead of +inf
  [(0, 0)] ref=inf, aot=inf, ind=-inf
  [(0, 1)] ref=inf, aot=inf, ind=-inf
  [(2, 1)] ref=inf, aot=inf, ind=-inf

=== Manifestation 3 (bug_014) ===
aot_eager  vs eager: same=True
inductor   vs eager: same=False
F.elu(-0.0, alpha=1.14):
  eager:    -0.0 (sign preserved: True)
  compiled: 0.0 (sign preserved: True)
BUG: F.elu(-0.0) loses negative sign in inductor
```

## Specifications

- PyTorch Version: 2.13.0+cu130
- torch.compile backend: inductor (`mode="default"`)
- CUDA: Not available (CPU-only repro)
- Python: 3.11.15
- OS: Ubuntu 24.04.4 LTS (x86_64)
- GCC: 13.3.0
- CPU: AMD Ryzen 9 9950X 16-Core Processor
- Triton: 3.7.1

## Versions

```
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: 15.0.7
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               32
On-line CPU(s) list:                  0-31
Vendor ID:                            AuthenticAMD
Model name:                           AMD Ryzen 9 9950X 16-Core Processor
CPU family:                           26
Model:                                68
Thread(s) per core:                   2
Core(s) per socket:                   16
Socket(s):                            1
Stepping:                             0
CPU(s) scaling MHz:                   72%
CPU max MHz:                          5756.4521
CPU min MHz:                          624.1940
BogoMIPS:                             8599.99
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d amd_lbr_pmc_freeze
Virtualization:                       AMD-V
L1d cache:                            768 KiB (16 instances)
L1i cache:                            512 KiB (16 instances)
L2 cache:                             16 MiB (16 instances)
L3 cache:                             64 MiB (2 instances)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-31
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected

Versions of relevant libraries:
[pip3] torch==2.13.0
[pip3] triton==3.7.1
[pip3] numpy==2.4.6
[conda] mkl                       2025.0.0
```