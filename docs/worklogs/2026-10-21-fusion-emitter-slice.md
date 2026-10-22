# Fused-expression emitter — local evidence

Date: 2026-10-21. Plan:
[slice-fusion-emitter.md](../design/slice-fusion-emitter.md).

## What landed

The crate now emits fused element-wise kernels instead of the baseline's run-time
CubeCL IR:

- `fusion::FusionPlan` — dtype, input count, a **topologically ordered** instruction
  list (operands name an input or an earlier value) and the live-out values. An
  instruction list, not a tree, so eight successive squarings are eight instructions
  where a tree would need 511 nodes.
- `fusion::emit_fusion` — validation (eligibility matrix, arity, operand order, the
  256-instruction bound) and a self-contained CUDA C++ source that embeds
  `cuda/include/t4a/abi.cuh` verbatim, so the helper semantics and their notices come
  from the single canonical definitions, plus a `FusionManifest`.
- `fusion::ArtifactKey` — the canonical key material (ABI revision, entry, dtype,
  input/output counts, source, and the caller's compiler options, target and compiler
  identity), usable directly as a cache key. It is the material, not a digest: nothing
  can be dropped silently, and no hash dependency was added.
- `fusion::plan_fused` — the flat launch for each runtime `len`; a prepared artifact
  reuses code and bindings, never a stale grid.
- Supporting refactor: the element-wise helpers the emitter calls (`exp`, `log`, `sin`,
  `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, `log1p`, `erf`, `pow`, `fmod`, the ordered
  clamp, the complex magnitude and a value-form scale-robust complex division) moved
  into `cuda/include/t4a/abi.cuh`, and `cuda/src/elementwise.cu` now calls them instead
  of keeping its own copies.

Classification: dtype, op set, arity and output count are compile-time; the element
count, every buffer address and the launch geometry are runtime and never enter the
source or the key. The emitted entry takes exactly the output pointers, the input
pointers and `len`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| Toolkit | `nvcc` 12.0.140 and NVRTC from `/usr/local/cuda-12.0`, harness arch `sm_80` |
| Rust | `cargo` 1.97 toolchain, `-j 16` |

## Commands and results

| Command | Result |
| --- | --- |
| `cargo fmt --check` | clean |
| `cargo clippy -j 16 --all-targets -- -D warnings` | clean |
| `cargo test -j 16` | 72 planning + 7 fusion host tests pass |
| `cargo test -j 16 --features gpu-tests` | 112 pass, 0 failed, 0 ignored (9 new) |

## Host-side coverage (`tests/fusion.rs`)

- the eligibility matrix for all 20 ops against all 7 dtypes, including every
  rejected pair;
- arity, missing input, forward reference, empty-plan and instruction-bound
  rejections;
- emitted-source structure: self-contained (no `#include`), the canonical header
  inside, the entry name, the argument order (outputs, then inputs, then `len`), the
  tail guard, one temporary per region value, the complex component forms with the
  robust division, and the instruction-value numbering (a regression for a numbering
  bug found while building this slice: instruction values are `v{inputs + index}`);
- the key: stable for the same plan, distinct for a different op set, dtype, option
  string, target or compiler identity, and unchanged by the runtime length;
- `plan_fused` geometry: empty, one block, a ragged tail, and the launch-limit
  rejection;
- shared values computed once (eight squarings → eight multiplies, two outputs
  referencing different values).

## GPU coverage (`tests/gpu_fusion.rs`)

A reference consumer: NVRTC compiles the emitted source, the driver loads the PTX, and
the harness launches it. Every launch checks a sentinel guard past `len *
element_size`.

- F32 and F64 two-live-out real plans (`y0 = (a*b + c) / d`,
  `y1 = tanh(exp(a*b + c))` over a shared `a*b`) at `len` 1, 513 and 65536 against a CPU
  reference; a single plan exercising all 19 eligible real operations is asserted
  present and launched.
- C32 and C64 chains built only from eligible operations (`add`, `mul`, `div`, `neg`,
  `conj`) with two live-outs at the same lengths, plus the revision-6 extreme-division
  fixtures (`2/(1e38+1e38i)`, subnormal over subnormal, an infinite divisor with a
  finite numerator keeping its signed zeros, NaN components).
- A fused `max`/`min`/`clamp` chain with NaN operands and signed zeros against the
  revision-6 ordered rules.
- The contraction fixture (`a*b + c` at `FLT_MAX`, `2`, `-FLT_MAX`) recorded as the
  documented tolerance-level difference from two separate launches.
- **Observed NVRTC and module counts** at the consumer's own boundary (the
  `compile_ptx_with_opts` and `cuModuleLoadData` wrappers): one artifact key across
  four extents → exactly one compile and one module load; four warm launches → zero
  further NVRTC calls, exactly one physical launch each; a different op set → a new key
  and one more compile; a changed `fmad` option and a changed architecture → two more
  keys and two more compiles.
- Two independent consumers on the one GPU — both on the device's primary context, since
  that is what the harness retains; true multi-context isolation is not exercised — each
  compile and load their own module, hold distinct function handles, and dropping one
  leaves the other's binding usable with no recompilation.
- `emitted_source_compiles_under_nvrtc_verbatim`: the pristine emitted source — no
  include path, no text repair — compiles for both a real and a complex plan.

## Findings

- The NVRTC consumer found **two real defects in the submitted state**, both fixed in
  the canonical header rather than worked around in the test:
  1. `abi.cuh` guarded its struct offsets with `__builtin_offsetof`, which NVRTC does
     not provide, so the header's own claim that the same text compiles under NVRTC did
     not hold and no emitted source compiled there. The offset assertions are now
     skipped under `__CUDACC_RTC__`; the size and alignment assertions, and the Rust
     mirrors, still run.
  2. `t4a_cuda_div_complex<C, F>` could not deduce `F` from its `(C, C)` arguments, and
     the emitter calls it without template arguments, so every emitted complex division
     was rejected. It is now a pair of overloads.
  The first version of this slice's test file passed by repairing the source text and
  ignoring the verbatim test; that repair is gone and the verbatim test runs.
- Building the slice also found a numbering bug in the emitter itself: instruction
  values were emitted as `v{index}` instead of `v{inputs + index}`, so outputs
  referenced input names. `tests/fusion.rs` now pins the numbering.
- The per-operation element-wise helpers moved into `abi.cuh` and `elementwise.cu`
  consumes them, so the fused source and the per-op entries cannot drift; the
  element-wise, linalg and RRQR GPU suites were re-run after that refactor.
- The diff review of that refactor found a real regression it had introduced: real
  `abs` was routed to the sign-preserving magnitude used by the complex division, so
  `abs(-0)` returned `-0` where revision 6 documents `fabs`. `t4a_cuda_fabs` is now a
  canonical helper and both the per-op entry and the emitter use it; the sign-preserving
  helper stays inside the complex division only.
- The same review round closed four more holes: an instruction-free plan could emit
  `float*` accesses for a Bool dtype (the dtype is now validated first), the emitted
  bundle's fields were independently mutable so a cloned-and-edited source could keep
  its old key (they are private behind accessors), the real dtype accepted `Conj` (now
  complex-only, with the planning test using an independently written matrix instead of
  the implementation's own predicate), and the operand count was not bounded by CUDA's
  4096-byte kernel-parameter limit (now `MAX_FUSION_PARAMETER_BYTES`).
