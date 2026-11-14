# Slice: the Metal source set and the Metal planning target

**Status:** implemented. **Unit:** U3, slice 1, of
tensor4all/tenferro-rs#2402, following
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§7 ("the Metal target") and §10 (the U3 row).

## Why this slice, and what it is not

Metal kernels are MSL. MSL is not CUDA-source compatible — `__global__` becomes
`kernel void`, pointers carry an address space, parameters are bound by index, and
there is no `extern "C"` symbol per entry because a pipeline is looked up by name.
So the Metal entries are a *second source set* beside `cuda/`, compiled at run time
from the text this crate ships, exactly as the CUDA text is handed to NVRTC.

U3 in the design is "Metal kernel sources and target". A Metal mirror of all 58
catalogue entries would not be reviewable, so this slice carries the
infrastructure and **one family**, the same way the CUDA catalogue was built:
sources, the planning target, the entry table, the ABI document section, and the
flat elementwise family in f32 and Complex32. Later slices are listed under
"Deferred" below.

Not in this slice, by design and recorded rather than left as a gap:

- **bf16 and f16.** The design lists them as non-goals for the first version (§2),
  and the dtype set is f32 + Complex32 (§8). (The dispatch text for this task says
  "f32/bf16/c32"; the design takes precedence, per the maintainer decision of
  2026-11-14 on #2402.)
- **MPS calls.** GEMM and contraction through `MPSMatrixMultiplication` are U5, and
  FFT is a later unit (§8, §10). Both belong to the engine: a kernel catalogue that
  encoded MPS matrices itself would be a second path to the same operation. This
  slice therefore ships only what MPS cannot serve, and says so in the ABI
  document.
- **The MSL fusion emitter** (U6) and **the macOS CI lane** (U4's row in §9).
- **The shared planning `Target`** with a vendor field, and the `T4A_SYM` prefix
  macro: those are `rocm-backend.md` M1a, which also renames the crate. Metal
  planning is a sibling module until then; the MSL text is unaffected by that move.

## The contract this slice implements

1. **No build-time framework linking.** The crate ships text. Nothing here opens,
   refers to or links Metal, MetalPerformanceShaders or Accelerate; the device lane
   is the only thing that touches a device, and it does so through
   `t4a-metal-runtime`, which loads the frameworks with `dlopen`.
2. **f32 and Complex32 only, with a typed refusal.** `metal::DTYPES` is the set;
   anything else is `UnsupportedDType` before a launch, and the ABI document says
   where f64 goes (the paired `CpuEngine`).
3. **One vocabulary.** `Entry`, `DType` (and its tags), `UnaryOp` (and its tags and
   `accepts` matrix), `Error` are reused. The one table this slice adds is
   `metal::SYMBOLS`, which maps a served `Entry` to its MSL name. There is no
   `metal::Entry` and no second tag space.
4. **Checked planning.** `metal::plan_*` validates the dtype, the op/pair, the
   length sign, the byte size (checked `u64`) and the threadgroup count, and returns
   the dispatch plus the operand's byte size. `len == 0` is an empty dispatch the
   caller skips, never a zero-thread launch.
5. **Explicit coverage.** The flat form is `ceil(len/256)` threadgroups of 256
   threads (eight SIMD groups of width 32, which is the planning input), and every
   kernel guards its tail — Metal does not bound-check an index.
6. **Declining is explicit.** An elementwise `Entry` without a Metal kernel
   (`FillOne`, `Binary`, `Compare`, `Select`, `Clamp`) is `UnsupportedOp`, not a
   plan that no kernel can serve.

## MSL versus CUDA: the differences that shape the text

| CUDA | MSL | Consequence |
| --- | --- | --- |
| `double`, `complex<float>`/`num_complex` layouts | neither exists | f32 + a two-scalar Complex32 struct; `static_assert` size 8 / alignment 4 |
| `float2` for a complex pair | `float2` is align 8 | the struct holds two `float`s, as `t4a_cuda_c32` already does (`cuda/include/t4a/abi.cuh:34-40`, asserts at `:277-278`) |
| `expm1`, `log1p`, `erf`, `hypotf` | absent from the standard library | `Z` tags refused with `UnsupportedOp`; `abs` is a scaled magnitude |
| `#include <t4a/abi.cuh>` with an include path | `newLibraryWithSource:` has no include path | the module keeps an include line for a file-based compile and `compose` removes it while prepending the header |
| `extern "C" __global__ void t4a_cuda_x` | `kernel void t4a_metal_x(...[[buffer(n)]])` | the entry name is the pipeline name; arguments bind by index |
| `__syncthreads`, atomics, `float4` | different names and semantics | not used by this slice |
| `nvcc --use_fast_math` off by default | `MTLCompileOptions.fastMathEnabled` defaults to **YES** | every transcendental and every contract-bearing division calls `metal::precise::*`; **subnormals flush to zero**, which the CUDA entries do not do, and so do comparisons — `sign` of a subnormal f32 is `+0` |
| `x != x` and `isnan(x)` are reliable | fast math may fold both under a no-NaN assumption | NaN and infinity are classified by bit pattern |

The fast-math rows are measured facts about the runtime this slice targets
(`t4a-metal-runtime` compiles with `nil` options, so Metal's default applies), not
inferences. The subnormal flush is a measured difference from the CUDA
twin on otherwise identical input; the device lane asserts it so it cannot change silently, the
ABI document states it, and removing it needs `MTLCompileOptions.fastMathEnabled =
NO`, which is a compile-options seam in the runtime (U4's area) and therefore not a
second code path here.

## Acceptance

- Host lane (`cargo test`, no device): registry ↔ MSL text ↔ `Entry` names, header
  and module composition, tag synchronization against `DType::tag()` and
  `UnaryOp::tag()`, the Complex32 layout, every rejected dtype and op, geometry over
  `{0, 1, 255, 256, 257, 100_003}`, `NegativeCount`, and `LaunchLimit` at both ends —
  a length of `i64::MAX` and one element past `u32::MAX`, the 32-bit position bound.
- Device lane (`--features gpu-tests-metal`, Apple GPU): every served entry × dtype
  × `{1, 3, 255, 256, 257, 4096, 100_003}`, unary included, against a CPU reference,
  with a sentinel-filled destination tail, so a kernel that stopped after the first
  threadgroup and a tail overrun both fail; the f32 arithmetic and sign entries at
  0 ulp over normal-range operands and the transcendentals within 8 ulp; `div`
  bit-identical to the correctly rounded quotient; `stats().launches`
  advances by exactly one per entry call; `abs` of a Complex32 at 1e-30 and 1e30
  stays finite and its NaN/infinity precedence matches `hypotf`; NaN/inf propagation,
  signed zero, and the measured subnormal flush.
- The link check: `otool -L` on the device test binary shows no Metal,
  MetalPerformanceShaders or Accelerate entry.
- The A100 CUDA suites and every CUDA plan are untouched: the only change in
  `src/lib.rs` is `pub mod metal;`.

## Deferred

Slice 2: the op-tag elementwise entry (`Binary`/`Compare`/`Select`/`Clamp`) with
`scalar_side`, and `fill_one`. Slice 3: structural movement (strided materialize,
contiguous→view, strided→strided, transpose, broadcast, concatenate). Slice 4:
reductions, including the split and combine policies the CUDA catalogue derives.
Later: gather/scatter, linalg, then U5 (MPS GEMM) and U6 (the MSL fusion emitter)
in their own units. Complex32 division and `expm1`/`log1p`/`erf` stay out until
there is a robust form or a library function to call. The subnormal flush is the
open numerical difference from the CUDA entries, and it closes in the runtime, with
a compile-options seam (`fastMathEnabled = NO`), not here.
