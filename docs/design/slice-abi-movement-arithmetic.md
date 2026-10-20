# Slice: C ABI, movement and elementary arithmetic

Status: slice plan, 2026-10-20. First implementation slice of the migration
design's [implementation sequence](cubecl-kernel-port.md#6-implementation-sequence)
steps 1–2, limited to the movement and arithmetic families. Reductions,
conversions, indexing/scatter, diagonal, linalg/RRQR, the fused-expression
emitter, the NVRTC prepare path and artifact caches remain for later slices.

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
(`crates/tenferro-gpu/src/kernels/{structural,elementwise,helpers}.rs`); the
local checkout `7e3b1eba6` has no changes to those files.

## 1. Scope

| Entry | Ported from | Supported dtypes |
| --- | --- | --- |
| `t4a_cuda_fill_zero` | `fill_zero_kernel` | all |
| `t4a_cuda_fill_zero_view` | `fill_zero_view_kernel` | all |
| `t4a_cuda_copy_exact` | `copy_bool_kernel` generalised | all |
| `t4a_cuda_materialize_strided` | `materialize_strided_kernel` | all |
| `t4a_cuda_contiguous_to_view` | `contiguous_to_view_kernel` | all |
| `t4a_cuda_strided_to_strided` | `strided_to_strided_kernel` | all |
| `t4a_cuda_tiled_transpose` | `tiled_transpose_kernel` | all |
| `t4a_cuda_add` / `_sub` / `_mul` | `add_*`, `sub_*`, `mul_*` | F32/F64/I32/I64/C32/C64 |
| `t4a_cuda_div` | `div_float` | F32/F64 |

"all" is F32/F64/I32/I64/Bool/C32/C64. Arithmetic excludes Bool: the baseline
dispatch has no Bool arithmetic entry (`cubecl/mod.rs:4812-4840`) and `uint8_t`
`1 + 1` would produce an invalid Bool byte. Complex division is deferred: it
must port the scaled Baudin–Smith form, not `ComplexCore`'s `/`. The
`*_comptime_layout_kernel` twins are the documented WebGPU/Metal-only exception
and are excluded. A separate one-fill entry is not needed: the baseline serves
it with the same fill body.

## 2. Repository layout

```
Cargo.toml                     crate t4a-cuda-kernels, no runtime dependencies
src/lib.rs                     facade: manifest, validation, launch plans
cuda/include/t4a/abi.cuh       POD types, static asserts, device helpers
cuda/src/movement.cu           fill/copy/materialize/strided/transpose entries
cuda/src/arithmetic.cu         add/sub/mul/div entries
docs/abi/c-kernel-abi.md       documented C ABI (deliverable)
tests/planning.rs              CPU-only validation/planning tests
tests/gpu_kernels.rs, tests/gpu/mod.rs   nvcc + raw driver API harness
LICENSE-MIT, LICENSE-APACHE, NOTICE
```

`cargo build`/`cargo test` need no CUDA installation. GPU tests are behind
`--features gpu-tests` and fail, never silently pass, when no device is usable.

## 3. Kernel ABI

The canonical ABI — struct layouts with sizes, offsets and alignment, the dtype
tag table, per-entry signatures and semantics, overlap rules and launch
geometry — is [docs/abi/c-kernel-abi.md](../abi/c-kernel-abi.md). This slice
implements revision 1 exactly as documented there; the summary below is not a
second copy of it.

Decisions carried by this slice that the ABI document then fixes:

- One `extern "C" __global__` entry per operation with a runtime dtype tag
  dispatching to a templated body, so there is one symbol per operation for both
  Rust and C++ consumers and one place holding the supported-dtype gate.
- Layout metadata travels by value, so one compiled module serves every layout up
  to rank 8. Rank, extents, strides and offsets are runtime arguments; none of
  them is emitted as a constant.
- `t4a_cuda_tiled_transpose` takes its tile configuration as a runtime argument
  selecting one templated body over dynamic shared memory, which keeps the whole
  baseline configuration family (`16x8-p1-v1`, `16x8-p1-v2`, `32x8-p1-v4`, ...)
  available without an instantiation explosion.
- Movement destinations, `fill_zero_view` included, must be injective, with no
  broadcast axis; a destination that repeats an address is rejected at planning.

## 4. Rust facade

No dependencies, no CUDA initialization, no `unsafe`:

- `DType` (tag, element size), `Layout` (`new`, `compact`, `element_count`,
  `min_reachable_offset`, `max_reachable_offset`, `rejects_broadcast_axis`),
  `TransposeConfig`, `Launch`, `Entry` manifest, `Error`, and the `CLayout` /
  `CTransposeConfig` C mirrors.
- One plan function per entry, taking that entry's operation metadata:
  `plan_fill_zero`, `plan_fill_zero_view`, `plan_copy_exact`,
  `plan_materialize_strided`, `plan_contiguous_to_view`,
  `plan_strided_to_strided`, `plan_tiled_transpose`, `plan_elementwise`.
  Each rejects unsupported dtypes, bad extents, a reachable span outside the
  supplied allocation (in elements and in representable byte offsets),
  broadcast destination axes and over-limit launch geometry. Geometry-only
  helpers stay private.

## 5. Tests

CPU (`cargo test`, no CUDA): layout validation (rank bounds, rank 0, negative
extents, checked products, negative-stride reachability, empty extents,
allocation spans and byte-offset representability), broadcast destinations,
grid/byte arithmetic at the `grid.x` and `grid.y/z` limits, transpose
shared-memory and grid limits, per-entry dtype matrix, manifest/source/ABI-doc
consistency.

GPU (`cargo test --features gpu-tests`, local A100, `nvcc -cubin -arch=sm_80`
then the raw driver API through cudarc, no `t4a-cuda-runtime`):

- Numerical comparison with a CPU reference for every dtype route: fill, exact
  copy, materialize with negative and zero strides, contiguous→view,
  strided→strided, tiled transpose in both orientations with odd/ragged sizes
  and batches, add/sub/mul/div.
- Bit-pattern equality (not tolerance) on **every** movement route, with a
  sentinel-filled destination allocation compared in full, so holes, tails and
  nonfinite payloads are covered, not just the written values.
- Edge cases: empty, length 1, 511/512/513, offsets, broadcast stride zero,
  misaligned (offset 1) views, integer wrapping at `i32::MAX`/`i64::MIN`,
  inf/NaN/signed-zero through arithmetic, `div` by ±0 and inf.
- Physical launch counts: one launch per entry, zero for a validated empty call.
- Arithmetic tolerance: `-fmad` contraction is the nvcc default, so sums and
  products are compared with a relative tolerance; copies are bit-exact.

Compiler options are fixed and recorded: `-std=c++17`, no `--use_fast_math`, no
`-ftz=true`.

## 6. Provenance

New files are dual licensed `MIT OR Apache-2.0`. The upstream copyright line
`Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors`
(`tenferro-rs/LICENSE-MIT:3`) is preserved in `NOTICE` and in each ported file's
derivation notice, alongside the new attribution rather than replaced by it.
Each `.cu` file names the tenferro-rs file and function range it translates, at
the pinned commit. `structural.rs` and the ported arithmetic carry no
third-party notice at that commit; reduction and complex-division notices
belong to later slices and must be audited before those families are ported.

## 7. Acceptance for this slice

- `cargo fmt --check`, `cargo clippy -j 16 --all-targets -- -D warnings`,
  `cargo test -j 16` pass with no CUDA installation present.
- `cargo test -j 16 --features gpu-tests` passes on the local A100 with the
  recorded hardware/driver/toolkit/command.
- The ABI document, the manifest and the `.cu` entry list agree, checked by a
  test.
- [Review checklist](../review-checklist.md) sections 1–7 satisfied with
  evidence.
