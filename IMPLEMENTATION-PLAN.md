# Implementation plan — U3 slice 1: the Metal source set and the Metal planning target

Issue: tensor4all/tenferro-rs#2402, unit **U3** of
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md) §7 and §10.
Repository: `tensor4all/t4a-cuda-kernels` (the design's `t4a-gpu-kernels`; the rename is
`rocm-backend.md` M1a and has not happened). Branch `feat/metal-elementwise-slice`, worktree
`/private/tmp/wt-metal-kernels`, base `b32744d`.
Model: `opencode-go/deepseek-v4.1-flash`. Date 2026-11-14.

## 0. Why this slice, and what "U3" is

The design's U3 is *"Metal kernel sources and target in t4a-gpu-kernels (after its rename,
rocm-backend.md M1a)"* (§10). M1a — the GitHub/crate rename to `t4a-gpu-kernels`, the `T4A_SYM`
prefix macro, the shared `Target { vendor, arch, wave_size }`, the `t4a_hip_*` header from one
template — is a separate unit and is untouched here. U1 (`83fe982d`, tenferro-rs) and U2
(`b8dee5e9`, t4a-gpu-backend#16) are merged and are not redone.

A full Metal mirror of all 58 catalogue entries would not be reviewable. This is **U3 slice 1**,
the same slice-by-family shape the CUDA catalogue itself used (`arithmetic`, `movement`,
`elementwise`, `reduce`, … each its own PR): the source-set and target *infrastructure* plus the
first family, flat elementwise, in f32 and Complex32. Later slices are named in §8.

Where the dispatch text and the design disagree, the design wins (maintainer decision of
2026-11-14 on #2402). Two consequences are recorded deliberately:

- **bf16 is not in this slice.** The dispatch says "f32/bf16/c32"; the design lists bf16 as a
  non-goal for the first version (§2) and its dtype set is f32 + Complex32 (§8). Follow the design.
- **No MPS call in this slice.** GEMM/contraction through `MPSMatrixMultiplication` is U5 in
  `tenferro-gpu` (§8, §10), FFT is "a later unit" returning typed `Unsupported` (§8). A kernel
  catalogue that launched `MPSMatrixMultiplication` itself would duplicate the engine's job.
  This slice ships only what MPS cannot serve, and states the routing in the ABI document.

## 1. Files, functions and current behaviour

| File | Now | After |
| --- | --- | --- |
| `src/lib.rs` (4310 lines) | CUDA-only: `Entry`, `DType`, `Error`, `Launch`, `UnaryOp`, `BinaryOp`, `ScalarSide`, the `source` module | **one line**: `pub mod metal;` beside `pub mod fusion;`. `ABI_REVISION` stays 14; no CUDA entry, planner, tag or source text changes |
| `src/metal.rs` | absent | new module: `SYMBOLS`, `SIMD_WIDTH`/`THREADGROUP`, `Dispatch`/`Plan`, the four `plan_*` planners, MSL source registry and `compose` |
| `metal/include/t4a/abi.msl` | absent | new MSL header: revision, `T4A_METAL_UNARY_*` tags, `t4a_metal_c32`, arithmetic/complex/sign helpers, the guarded-index helper |
| `metal/src/elementwise.msl` | absent | new module: `t4a_metal_fill_zero`, `t4a_metal_copy_exact`, `t4a_metal_add`, `t4a_metal_sub`, `t4a_metal_mul`, `t4a_metal_div`, `t4a_metal_unary` |
| `tests/metal_planning.rs` | absent | new host lane (no device, runs in the existing `ubuntu-latest` job) |
| `tests/metal_kernels.rs` | absent | new device lane, `required-features = ["gpu-tests-metal"]` |
| `Cargo.toml` | 58 entries, `gpu-tests` only | `+"/metal/**"` in `include`, `[features] gpu-tests-metal`, optional `t4a-metal-runtime` dep, `[[test]]` rows |
| `Cargo.lock` | CUDA-only graph | the new git source entry for `t4a-metal-runtime` at the pinned rev, so `--locked` keeps working |
| `.github/workflows/ci.yml` | lock check runs `--offline` before anything is fetched | `cargo fetch --locked` step before it; `cargo test --features gpu-tests-metal --no-run` compile-only step |
| `docs/abi/c-kernel-abi.md` | CUDA C ABI, revision 14 | new "Metal" section (contents in §1.3) |
| `docs/design/slice-metal-elementwise.md` | absent | new slice design: scope, MSL-vs-CUDA differences, deferred list, validation |
| `docs/worklogs/2026-11-14-metal-elementwise-slice.md` | absent | new: decisions, M5 evidence, deferred items, review record |
| `README.md`, `AGENTS.md` | CUDA commands only | Metal source set + the local `gpu-tests-metal` gate next to them |

Behaviour that does not change: every CUDA symbol, planner, tag and ABI revision; `fusion.rs`;
the NVRTC `source` module; `Entry`/`DType`/`Error`/`UnaryOp`/`BinaryOp`/`ScalarSide` semantics.

**No second vocabulary.** Everything existing is reused: `DType` (and `DType::tag()`, `size()`),
`Error`, `Entry` (including `Entry::supports`, whose CUDA dtype table already says `Div` is
f32/f64 and `Add/Sub/Mul` include C32 — so a Metal dtype rejection is the same rejection the CUDA
planner makes), `UnaryOp` (and its `tag()` and `accepts(dst, src)` pair matrix), `ScalarSide`
(unused in this slice; not referenced). This slice adds no `metal::Entry`, no `metal::UnaryOp`,
and no Metal tag space of its own.

### 1.1 `src/metal.rs` — the Metal planning target

```
pub const SIMD_WIDTH: u32 = 32;                  // planning input: Metal SIMD-group width
pub const THREADGROUP: u32 = 8 * SIMD_WIDTH;     // 256 threads per threadgroup, flat form
pub const ABI_REVISION: u32 = 1;                 // Metal's own revision; CUDA's stays 14

/// `Entry` → MSL kernel name, for the entries this source set serves.
pub const SYMBOLS: &[(Entry, &str)] = &[
    (Entry::FillZero,  "t4a_metal_fill_zero"),
    (Entry::CopyExact, "t4a_metal_copy_exact"),
    (Entry::Add,       "t4a_metal_add"),
    (Entry::Sub,       "t4a_metal_sub"),
    (Entry::Mul,       "t4a_metal_mul"),
    (Entry::Div,       "t4a_metal_div"),
    (Entry::Unary,     "t4a_metal_unary"),
];
pub fn symbol(entry: Entry) -> Option<&'static str>;

pub struct Dispatch { pub threadgroups: u32, pub threads_per_threadgroup: u32 }
pub struct Plan { pub dispatch: Dispatch, pub bytes: u64 }

pub fn plan_elementwise(entry: Entry, dtype: DType, len: i64) -> Result<Plan, Error>
pub fn plan_fill_zero(dtype: DType, len: i64) -> Result<Plan, Error>
pub fn plan_copy_exact(dtype: DType, len: i64) -> Result<Plan, Error>
pub fn plan_unary(op: UnaryOp, dst: DType, src: DType, len: i64) -> Result<Plan, Error>
```

Names and argument order follow the existing CUDA planners (`plan_elementwise`,
`plan_fill_zero`, `plan_copy_exact`, `plan_unary` at `src/lib.rs:1591`, `1600`, `1623`, `1985`),
so a reader who knows one knows the other; the return type differs because Metal's launch form
does. Validation order and error mapping: unsupported dtype/pair → `UnsupportedDType`; an op the
pair admits but MSL cannot serve → `UnsupportedOp`; negative length → `NegativeCount`;
`len * dtype.size()` overflow → `Overflow`; a threadgroup count above `u32::MAX` → `LaunchLimit`.
`len == 0` plans an empty `Dispatch` the caller skips. Geometry:
`threadgroups = ceil(len / THREADGROUP)`, `threads_per_threadgroup = THREADGROUP`; **every Metal
kernel guards its own tail**, which `Dispatch`'s doc states, because `LaunchConfig::tiled` in the
runtime does not bound-check an index. An entry outside `SYMBOLS` — `FillOne`, `Binary`,
`Compare`, `Select`, `Clamp` are all `Entry::is_elementwise()` — is **`UnsupportedOp`**, not a
plan: those entries exist in the CUDA catalogue and have no Metal kernel yet, so a plan without a
kernel would be a lie. Documented in the `plan_*` doc comments and the ABI document, with a host
test per entry.

Dtype and op admission (all reached through the existing tables, then narrowed to Metal):

| Entry | f32 | Complex32 | rejected |
| --- | --- | --- | --- |
| `FillZero`, `CopyExact`, `FillOne` | ✔ | ✔ | `F64`, `C64`, `I32`, `I64`, `Bool` → `UnsupportedDType` (and `FillOne` → `UnsupportedOp`, below) |
| `Add`, `Sub`, `Mul` | ✔ | ✔ | `I32`, `I64`, `F64`, `C64`, `Bool` → `UnsupportedDType` |
| `Div` | ✔ | ✘ | Complex32 → `UnsupportedDType` (matches `Entry::Div.supported_dtypes()`), `F64` likewise |
| `Unary` | `Neg, Exp, Log, Sin, Cos, Tanh, Sqrt, Rsqrt, Abs, Sign` | `Neg, Conj, Sign`, and `Abs` with `dst = F32` | `Expm1`, `Log1p`, `Erf` → `UnsupportedOp` (**MSL has no such function**); `Conj` on anything but C32 → `UnsupportedOp` (same as `UnaryOp::accepts`); every dtype outside f32/c32 → `UnsupportedDType` |

Deferred on purpose and recorded in the design doc and worklog: Complex32 division (CUDA keeps the
scaled Baudin-Smith form at `cuda/include/t4a/abi.cuh:969`, and `cuda/src/elementwise.cu` excludes
it this revision too), `Expm1`/`Log1p`/`Erf`, and the whole op-tag `Binary` entry (it arrives with
slice 2's `compare`/`select`/`clamp`, so this slice adds no `BinaryOp` tag rows at all).

`metal::source::{ABI_HEADER, Module, MODULES, compose}` mirrors the existing NVRTC `source`
module (`src/lib.rs:37-130`): `compose` prepends the header, because `newLibraryWithSource:` has
no include path, so a module keeps one include line for a file-based compile and `compose` removes that line while prepending the header (the test asserts both).

### 1.2 `metal/include/t4a/abi.msl` and `metal/src/elementwise.msl`

- Header: `T4A_METAL_ABI_REVISION`, `T4A_METAL_F32 0` / `T4A_METAL_C32 5` (the values of
  `DType::tag()`), `T4A_METAL_UNARY_*` with **exactly the values of `UnaryOp::tag()`**
  (`src/lib.rs:1795-1808`), and
  `struct t4a_metal_c32 { float re; float im; }` with MSL `static_assert`s for size 8 and
  alignment 4 — **not** a `float2` member, which is align 8 in MSL; the CUDA header makes the same
  choice for the same reason (`cuda/include/t4a/abi.cuh:34-40`, asserts at `:277-278`), because the
  host layout is `num_complex::Complex32` at alignment 4 and a binding offset of, say, 4 would
  misalign an 8-aligned element type. Plus helpers: zero, conj, add/sub/mul/div for f32 and c32,
  `t4a_metal_abs_c32` (a scaled hypot: MSL has none, and `sqrt(x²+y²)` is not the robust form),
  `t4a_metal_sign_f32/c32` (CUDA parity: `+0` for either signed zero, NaN propagated, complex
  normalized by magnitude), `t4a_metal_guarded_index`.
- Elementwise: seven `kernel void t4a_metal_*` entries over `device uchar*` buffers with
  `constant int&`/`constant long&` scalars, mirroring `cuda/src/elementwise.cu` and
  `cuda/src/arithmetic.cu` semantics op by op.
- Transcendentals call `metal::precise::*` explicitly. The runtime (U2, merged) compiles with
  `nil` `MTLCompileOptions`, i.e. `fastMathEnabled = YES`, and has no seam for compile options;
  explicit `precise::` selects the higher-accuracy form of each such call (the device lane's bar
  is 8 ulp against the host's libm; it is not a rounding guarantee, and it does not change the mode
  of the surrounding expressions).
- Both files open with the provenance notice the review checklist §6 requires: derived from
  `cuda/src/elementwise.cu` and `cuda/src/arithmetic.cu`, hence from tenferro-rs
  `crates/tenferro-gpu/src/kernels/…` at the pinned commit those files name, and
  `MIT OR Apache-2.0`. `THIRD_PARTY_NOTICES.md` is checked for a needed addition.

### 1.3 The ABI document section

`docs/abi/c-kernel-abi.md` gains a Metal part that states, per entry: the kernel signature, the
buffer indices in order, the by-value parameters, the dtype matrix, the tag table (and that the
tags are the values of `UnaryOp::tag()`), the Complex32 layout, and — per checklist §3 — the
**injective/alias rules**: `fill_zero` writes only its destination; `copy_exact` copies element
*bit patterns* and must not alias (`in` and `out` overlapping is undefined); `add/sub/mul/div`
and `unary` write every destination element exactly once from read-only operands, and a
destination that aliases an operand is undefined; every entry fully overwrites `[0, len)` and
touches nothing at or beyond `len`. It also names what is **not** here and who owns it: GEMM and
contraction → `MPSMatrixMultiplication` in U5, FFT and linalg → typed `Unsupported` until their
units land, f64/Complex64 → `Unsupported` naming the paired `CpuEngine`.

## 2. Tests and measurements, and the acceptance bars

| Command | Bar |
| --- | --- |
| `cargo fmt --check` | no diff |
| `cargo clippy -j 16 --locked --all-targets -- -D warnings` | clean |
| `cargo clippy -j 16 --locked --all-targets --features gpu-tests-metal -- -D warnings` | clean |
| `cargo test -j 16 --locked` | all host tests pass; CUDA tests unchanged; no device needed |
| `cargo test -j 16 --locked --features gpu-tests-metal` on the M5 Max | every served entry × dtype × the size table below passes; **no test skipped**; fails on a host with no device |
| `otool -L` on the device test binary | `grep -Ei 'metal|metalperformance|accelerate'` prints nothing |
| `cargo metadata --locked --offline --format-version 1 > /dev/null` | exit 0 after the new fetch step (it is the CI lock check) |

**Host lane** (`tests/metal_planning.rs`, no device): `SYMBOLS` ↔ `Entry::name()` drift (each
symbol equals its entry's name with `t4a_cuda_` → `t4a_metal_`); every `SYMBOLS` name appears
exactly once across `MODULES` and no other `kernel void` name exists in the Metal text;
`module_of`/`symbol` round-trip; a composed module starts with the header and neither it nor the
module text contains `#include`; no `double` and no `float64` in the Metal text (the provenance
comment is phrased without the word); the MSL
`T4A_METAL_F32/C32` and `T4A_METAL_UNARY_*` values equal `DType::tag()`/`UnaryOp::tag()` (parsed
from the header text, the same doc-sync shape as `tests/planning.rs`); `t4a_metal_c32` mirrors
`#[repr(C)] struct { re: f32, im: f32 }` — size 8, align 4, offsets 0 and 4; every rejected dtype
and rejected op of §1.1; geometry for `len ∈ {0, 1, 255, 256, 257, 100_003}` (`100_003` → 391
threadgroups); `NegativeCount`, `Overflow` at `i64::MAX`, `LaunchLimit` at a length whose
threadgroup count exceeds `u32::MAX`.

**Device lane** (`tests/metal_kernels.rs`, `gpu-tests-metal`), against a CPU reference:

- Every served entry × `{F32, C32}` (and `Div` f32 only) × `len ∈ {1, 3, 255, 256, 257, 4096,
  100_003}` — the last two are far larger than one threadgroup, which is the only size range that
  can show a single-threaded kernel.
- Coverage, not geometry: the value comparison over the whole `[0, len)` result plus a
  **sentinel-filled destination tail** proves the grid reaches every element and writes nothing
  past `len`; the planned geometry is asserted in the host lane, where it is exact.
  `Runtime::stats().launches` must advance by exactly 1 per entry call.
- `fill_zero`/`copy_exact`: bit-exact, including the sentinel tail and `-0.0`/NaN payload bits in
  the copied range.
- `unary`: per-op comparison (exact for neg/abs/conj/`abs`-complex; ≤4 ulp relative for
  transcendentals); `abs` c32→f32 over extreme magnitudes (1e-30, 1e30, inf, NaN) against a
  scaled-hypot reference; `sign` on `±0`, `NaN`, `±inf`.
- `add/sub/mul`: NaN/inf propagation and signed zero; `div` f32 including `x/0` and `0/0`.

Evidence recorded in the worklog (hardware, OS, GPU compiler version, exact commands, observed
results): Apple M5 Max, macOS 26.7.2 (25G332), MSL via
`GPUCompiler.framework/Versions/32023`, `Runtime::open` capability probe output, and the header
path used for the "no expm1/log1p/erf/hypot" check.

## 3. Edge cases, compatibility, and what is deliberately not changed

- **Empty and single-element tensors**: `len == 0` plans an empty dispatch the caller skips (no
  zero-thread launch); `len == 1` is one threadgroup of 256 threads with a guard.
- **Tails**: every kernel guards `index >= len`; the sentinel-tail cases are the proof.
- **Size arithmetic**: bytes are checked in `i64`/`u64`, never `as usize`; this crate allocates
  nothing.
- **Compatibility**: additive only. `ABI_REVISION` stays 14; Metal's revision is its own. Nothing
  in `src/lib.rs` changes except `pub mod metal;`, so all 58 CUDA entries, planners and source
  compositions are byte-identical, and the CUDA A100 evidence remains valid. No tenferro-rs
  change and no consumer in this PR (that is U4).
- **Non-goals in this slice** (written into the design doc rather than left as a gap): reductions,
  strided movement, transpose/broadcast, gather/scatter, linalg, FFT, GEMM (MPS, U5), the op-tag
  `Binary`/`Compare`/`Select`/`Clamp` entries, complex division, `expm1`/`log1p`/`erf`, bf16/f16,
  MPSGraph, the MSL fusion emitter (U6), the macOS CI lane (U4's row in §9).
- The existing `gpu-tests` feature, `cudarc` dep and CI lanes are untouched apart from the added
  fetch step, which is additive for every lane.

## 4. Risks, and the alternatives rejected

1. **The MSL text is not compiled by CI in this slice.** `xcrun metal` is not installed and the
   macOS CI lane is U4's row (§9/§10). Mitigation: the host lane checks registry/structure/tags;
   the device lane is the local gate (`cargo test --features gpu-tests-metal`) and its M5 run is
   recorded in the worklog and PR body; U4 adds the macOS lane that compiles it in CI. This is
   the slice's main residual risk and is written into the design doc.
2. **A cross-repo optional dependency.** `t4a-metal-runtime` is unpublished (0.1.0), so the device
   lane needs a git dependency pinned to the merged `b8dee5e9`, and `cargo metadata --locked
   --offline` then needs a checkout: measured here — with a cold `CARGO_HOME` it exits 101
   ("can't checkout … you are in the offline mode"). Hence the `cargo fetch --locked` step before
   the lock check. It mirrors the existing `gpu-tests`/`cudarc` pattern rather than inventing one.
3. **Fast math.** `MTLCompileOptions.fastMathEnabled` defaults to YES and U2's runtime passes
   `nil` options, so it is fast by default. Handled in-source with `metal::precise::*` and ulp
   tolerances; a compile-options seam in the runtime is a follow-up for U4, recorded as a
   limitation rather than worked around with a second code path.
4. **Metal planning is not yet folded into a shared `Target`.** `SIMD_WIDTH`/`THREADGROUP` and the
   four `plan_*` functions live in `metal` beside the CUDA planners; M1a (rocm-backend.md §6.1
   step 2) introduces `Target` and `T4A_SYM`. Folding Metal in now would mean rewriting the whole
   CUDA planning surface in this PR. Recorded as an overlap to consolidate in M1a; the MSL text
   is unaffected by that move.
5. **Placement before the rename.** The sources land in `t4a-cuda-kernels` because that is the
   repository the design names and M1a has not renamed it; the later move is `git mv metal/` plus
   a crate rename, mechanical, and the design's §7 paths (`kernels/metal/`) remain the target.
   The sequence deviation is to be noted on #2402.
6. **~1 ulp transcendental agreement**, so the ABI document says which operations are bit-exact
   (copies, zero, neg, conj, add/sub/mul/div in f32, sign) and which are not.

Alternatives rejected:

- **A new `t4a-metal-kernels` crate in `t4a-gpu-backend`** (in-workspace device tests, no git
  dependency). Rejected: §7 puts kernel sources in the kernel catalogue; a second kernel home in
  the runtime repository would be a second source of truth, and the runtime must not depend on
  kernels.
- **Do M1a first** (repo/crate rename, `T4A_SYM`, shared `Target`). Rejected: a separate unit with
  a much larger blast radius (org rename, published-crate rename, tenferro-rs pins) that would
  delay the first Metal kernels for no correctness gain.
- **Port the whole 58-entry catalogue now.** Rejected: unreviewable; the CUDA catalogue itself was
  built family by family.
- **Route reductions through MPS here** because the dispatch text says so. Rejected: the design
  gives the reduction family to U4 and MPS has no arbitrary-axis reduction class; a
  matmul-with-ones encoding is a performance guess that belongs to the engine, not the catalogue.
- **A `metal::Entry`/`metal::UnaryOp`/Metal tag space** as drafted first. Rejected after review:
  `Entry`, `UnaryOp` (with `tag()` and `accepts`) and `DType` already exist, so the Metal module
  reuses them and adds one 7-row symbol table only.

## 5. Review record

- Review 1 (pre-implementation gate): model **`ollama-cloud/glm-5.3-flash`** (GLM-5.3-flash, agent
  `glm-brainstormer`, read-only), verdict **CHANGES**, no blockers; 4 IMPORTANT and 4 MINOR
  findings. Disposition:
  - IMPORTANT §2 acceptance bar "a 4096-element run reports 16 threadgroups" was unprovable
    (`Runtime::stats()` counts dispatches only, `runtime.rs:92-114`) → geometry is now asserted in
    the host lane; the device lane proves coverage by value comparison and sentinel tails.
  - IMPORTANT §1.1 unary matrix contradicted itself and diverged from `UnaryOp::accepts`
    (`src/lib.rs:1811-1845`: `Conj` is complex-only) → matrix rewritten; `accepts` is now reused
    and `Expm1/Log1p/Erf/Conj`-on-f32 rejections are stated once.
  - IMPORTANT provenance/derivation notices missing (checklist §6) → added to §1.2.
  - IMPORTANT ABI document lacked per-entry injectivity/alias rules (checklist §3) → added §1.3.
  - MINOR `t4a_metal_c32` layout assertions → added to the host lane (§2).
  - MINOR the "no `#include <t4a/abi.cuh>`" wording was wrong for MSL → now "no `#include` at all".
  - MINOR `SIMD_WIDTH` unused → now used: `THREADGROUP = 8 * SIMD_WIDTH`.
  - MINOR plan-stage review model vs the repository's `gpt-6.1-sol` rule → the dispatch of this
    task mandates GLM-5.3-flash for the plan gate; the finished diff gets the repository's own
    independent review, and both are recorded in the worklog and the PR body.
  - Its number claims (`fastMathEnabled` default YES, `LaunchConfig::tiled` not bounds-checking)
    were re-verified against `crates/t4a-metal-runtime/src/compile.rs:222-231` and
    `kernel.rs:66-91`; the "no expm1/log1p/erf/hypot in MSL" claim was verified directly against
    `…/GPUCompiler.framework/Versions/32023/…/metal/metal_math`.
- Review 2 (revised plan, same reviewer model `ollama-cloud/glm-5.3-flash`): verdict **CHANGES**,
  no blockers; dispositions applied to this document:
  - IMPORTANT the planned `struct t4a_metal_c32 { float2 v; }` is align 8 in MSL while the host
    layout is align 4 → two scalar members with MSL `static_assert`s, matching
    `cuda/include/t4a/abi.cuh:34-40`; §1.2 and the host lane updated.
  - IMPORTANT `plan_elementwise` had no defined answer for an elementwise entry outside
    `SYMBOLS` and would return a plan with no kernel → `UnsupportedOp`, one host test per such
    entry; `Bool` added to the `Add/Sub/Mul` rejected column.
  - MINOR the `otool` grep in §2 had markdown-escaped pipes → plain pattern.
  - MINOR a "no `t4a_cuda_`" text assertion would fire on the provenance notice → assertion
    narrowed to `double`/`float64`, and the notice is phrased without them.
  - MINOR `Cargo.lock` must gain the git source entry for `--locked` → added to §1.
- Review 3 (post-implementation, repository rule): independent review of the finished diff before
  the PR (`reviewer-gpt`, GPT-6.1 Sol, read-only), recorded in the PR body with the model and
  verdict together with the repository's own adversarial-review gate.

## 6. Completion status

Every row of §1 is in this branch, and every gate of §2 has been run on this host:

| Row | State |
| --- | --- |
| `src/lib.rs`, `src/metal.rs`, `metal/include/t4a/abi.msl`, `metal/src/elementwise.msl` | done; 7 entries, `SYMBOLS` closed by a host test in both directions |
| `tests/metal_planning.rs` (15 tests), `tests/metal_kernels.rs` (12 device tests) | done; all pass, host lane on the plain toolchain and device lane on the M5 Max |
| `Cargo.toml`, `Cargo.lock` | done; the lock holds the pinned `t4a-metal-runtime` git source |
| `.github/workflows/ci.yml` | done; the `cargo fetch --locked` step was measured against a cold `CARGO_HOME` (see the worklog), and the Metal compile-only step is in the same job |
| `docs/abi/c-kernel-abi.md` | done; the Metal section carries the signatures, buffer indices, dtype and tag tables, numerics, alias rules, geometry and what the set does not contain |
| `docs/design/slice-metal-elementwise.md`, `docs/worklogs/2026-11-14-metal-elementwise-slice.md` | done, with `docs/worklogs/evidence/2026-11-14-metal-elementwise/` holding the OS/GPU, MSL-header, gate, test and link-check transcripts |
| `README.md`, `AGENTS.md` | done |

Gates, in §2's order: `cargo fmt --check` clean; `cargo clippy --all-targets -D warnings` clean
with and without `--features gpu-tests-metal`; `cargo test` 7 + 15 + 91 pass with no failure and no
skip; `cargo test --features gpu-tests-metal` adds 12 device passes on the M5 Max; `otool -L` on the
device test binary shows `libiconv` and `libSystem` only, so no Metal, MPS or Accelerate linkage.

One acceptance bar is weaker than §2 first asked, and the reason is recorded in §5: the device lane
cannot read back a planned threadgroup count, because `Runtime::stats()` counts dispatches. Coverage
is therefore proven by value comparison over `{1, 3, 255, 256, 257, 4096, 100_003}` and by a
sentinel-filled destination tail, and the geometry itself is asserted in the host lane.

Review 3 found four IMPORTANT issues in the finished diff, all fixed here and recorded in §5 and the
worklog: the planner admitted lengths no 32-bit kernel position can reach (and the Complex32 copy
scaled its word index in 32-bit arithmetic), `abs` returned 0 for `(NaN, 0)`, the unary device tests
covered one threadgroup and no tail, and the fast-math claim was broader than the mitigation.
Measuring that last one also established the numerical difference this slice has to state: with
Metal's default fast math, **subnormals flush to zero**, which the CUDA entries do not do. The
device lane asserts the flush, the ABI document and the slice design state it, and removing it needs
`MTLCompileOptions.fastMathEnabled = NO`, a compile-options seam in `t4a-metal-runtime` that this
slice does not own.
