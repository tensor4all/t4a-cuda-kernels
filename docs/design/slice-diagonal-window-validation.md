# Slice B: diagonal, windowing and validation-status kernels

Status: implemented and GPU-validated, 2026-10-20 (ABI revision 2). Second implementation slice of
[the migration design](cubecl-kernel-port.md), covering the parts of
implementation-sequence step 2 that reuse the revision-1 layout ABI.

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
(`crates/tenferro-gpu/src/kernels/{diagonal,indexing,structural}.rs` and the
caller contracts in `crates/tenferro-gpu/src/cubecl/mod.rs`).

## 1. Scope

| Entry | Ported from | Dtypes |
| --- | --- | --- |
| `t4a_cuda_extract_diagonal` | `extract_diagonal_kernel` (diagonal.rs L5) | all |
| `t4a_cuda_embed_diagonal_copy` | `embed_diagonal_copy_kernel` (L33) | all |
| `t4a_cuda_tril` / `t4a_cuda_triu` | `tril_kernel` (L61), `triu_kernel` (L75) | all |
| `t4a_cuda_slice` | `slice_kernel` (indexing.rs L174) | all |
| `t4a_cuda_dynamic_slice` | `dynamic_slice_kernel` (L193) | all |
| `t4a_cuda_pad` | `pad_kernel` (L215) | all |
| `t4a_cuda_init_validation` | `init_float_index_validation_flag` (L12) | F32/F64 |
| `t4a_cuda_validate_float_indices` | `validate_float_indices_kernel` (L22) | F32/F64 |
| `t4a_cuda_extract_invalid_float_index` | `extract_invalid_float_index_kernel` (L47) | F32/F64 |
| `t4a_cuda_validate_real_cast` | `validate_real_cast` (structural.rs L419) | F32/F64 |
| `t4a_cuda_extract_invalid_real_cast` | `extract_invalid_real_cast` (L447) | F32/F64 |

`scatter_copy_kernel` needs no new entry: it is a bit-preserving flat copy and
[`t4a_cuda_copy_exact`](slice-abi-movement-arithmetic.md) already covers it; the
manifest records the mapping.

Deferred with reasons: **gather/scatter** need a dimension-map descriptor
(window/offset/start-index/scatter/update-window maps, index-vector dim, three
ranks) plus the additive-scatter atomic contract — a new ABI concept, reviewed
separately. **Conversions** are ~20 entries whose supported-pair matrix must be
audited against the baseline dtype gates. **Reductions** need the strategy/split
planner and partial workspace.

## 2. ABI additions (revision 2)

Revision 1 entries and structs are unchanged.

```c
typedef struct {          /* 72 bytes, align 8, by value */
  int32_t rank;           /* 0..=8, equals the operation rank */
  int32_t reserved;       /* 0 */
  int64_t values[8];
} t4a_cuda_axis_values;

void t4a_cuda_extract_diagonal(void* out, const void* in, int32_t dtype,
                               t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                               int32_t axis_a, int32_t axis_b, int64_t len);
void t4a_cuda_embed_diagonal_copy(void* out, const void* in, int32_t dtype,
                                  t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                                  int32_t axis_a, int32_t axis_b, int64_t len);
void t4a_cuda_tril(void* out, const void* in, int32_t dtype, int64_t k,
                   int64_t rows, int64_t cols, int64_t len);
void t4a_cuda_triu(void* out, const void* in, int32_t dtype, int64_t k,
                   int64_t rows, int64_t cols, int64_t len);
void t4a_cuda_slice(void* out, const void* in, int32_t dtype,
                    t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                    t4a_cuda_axis_values starts, t4a_cuda_axis_values steps, int64_t len);
void t4a_cuda_dynamic_slice(void* out, const void* in, const void* starts, int32_t dtype,
                            int32_t index_dtype, t4a_cuda_layout out_layout,
                            t4a_cuda_layout in_layout, int64_t len);
void t4a_cuda_pad(void* out, const void* in, int32_t dtype,
                  t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                  t4a_cuda_axis_values edge_low, t4a_cuda_axis_values edge_high,
                  t4a_cuda_axis_values interior, int64_t len);
void t4a_cuda_init_validation(uint32_t* flag, void* value, int32_t dtype);
void t4a_cuda_validate_float_indices(const void* indices, uint32_t* flag, int32_t dtype,
                                     double max_exact_integer, int64_t len);
void t4a_cuda_extract_invalid_float_index(const void* indices, const uint32_t* flag,
                                          void* value, int32_t dtype);
void t4a_cuda_validate_real_cast(const void* input, uint32_t* flag, int32_t dtype,
                                 double min, double max, int64_t stride, int32_t max_inclusive,
                                 int64_t len);
void t4a_cuda_extract_invalid_real_cast(const void* input, const uint32_t* flag, void* value,
                                        int32_t dtype, int64_t stride);
```

### Contract notes and deliberate corrections

- **Compactness.** The baseline decodes coordinates from a layout but stores at
  the flat position (`out[flat]`), so it requires a compact destination.
  Revision 2 states that requirement instead of porting the ambiguity: `out` is
  compact for extract-diagonal, tril/triu, slice, dynamic-slice and pad; `in` is
  compact for embed-diagonal and tril/triu. The strided operand (input for
  extract/slice/dynamic-slice/pad, output for embed) keeps signed strides.
- **Diagonal extraction corrects a baseline bug.** `diagonal.rs:20-26` consumes
  the diagonal output coordinate where a remaining non-diagonal coordinate is
  required: it never skips the diagonal output axis while walking the other
  output coordinates, so any rank ≥ 3 layout with two or more remaining output
  axes reads a wrong input coordinate (input `[2,3,4]`, axes `(0,2)`, output
  coordinate `(0,1)` reads input `(0,0,0)` instead of `(0,1,0)`). The caller
  derives `diag_output_axis = axis_a - (axis_b < axis_a)` (`mod.rs:5366`), so
  the defect is reachable. Revision 2 implements the intended map, derives that
  output position in the planner instead of accepting it as an argument, and
  tests against an independent CPU oracle built by enumerating input
  coordinates rather than by walking output cursors.
- **Embedding axis rules differ from extraction.** `axis_a` indexes the input
  (`< in.rank`); `axis_b` is an output insertion position in `[0, in.rank]`, so
  `axis_a == axis_b` is valid (`[2,3]` with `(1,1)` gives `[2,3,3]`). The kernel
  writes only diagonal positions, so the output must be zero-filled before it:
  the ABI documents that ordering (the baseline caller does exactly this,
  `mod.rs:2606-2637`) and the tests compose `fill_zero` + embed.
- **tril/triu keep every rank ≥ 2 and batches.** The baseline accepts rank ≥ 2
  and masks axes 0 and 1 of the compact layout across the whole flat domain, so
  a `[a, b, c]` operand is a batch of `a×b` matrices along `c`. The ABI passes
  `rows`, `cols` and the total `len`; the kernel derives `row = flat % rows`,
  `col = (flat / rows) % cols`. The full output is written (value or zero).
  Comparisons are written overflow-free as `col - row <= k` (tril keeps) and
  `col - row >= k` (triu keeps) instead of the baseline's `col - k`, whose
  signed overflow is UB for extreme `k`; `k = i64::MIN/MAX` is tested.
- **slice** takes runtime `starts` and `steps`. The facade checks
  `0 <= starts[a] <= in.extents[a]`, `steps[a] >= 1`, that `out.extents[a]` does
  not exceed the reachable count (a shorter output is the caller's slice limit,
  matching the baseline's explicit `limit`) and that every reached input
  coordinate is inside the input span.
- **dynamic_slice** derives its window sizes from `out.extents` (the caller
  requires output extents equal to the slice sizes, `mod.rs:3373-3402`) and
  requires `in.extents[a] >= out.extents[a]`. Starts are device-resident
  `index_dtype` (F32/F64/I32/I64) values, never read on the host, and must be
  the *validated* starts: the baseline runs `I::validate` first, so a fractional
  or nonfinite float start is an error, not something to truncate. The clamping
  is `clamp_window_start`: `max_start = dim - window`, `start <= 0 → 0`, else
  `min(start, max_start)`.
- **pad** keeps the unsigned-magnitude treatment of negative edge padding
  (`0 - low` for `low < 0`, so `i64::MIN` behaves) and requires
  `interior >= 0` (the baseline rejects negative interior padding,
  `mod.rs:5514`; a negative interior would also divide by zero). The facade
  checks `out.extents[a] == edge_low[a] + edge_high[a] + (in.extents[a] == 0 ?
  0 : (in.extents[a]-1)*(interior[a]+1) + 1)`, all in checked `i64`. Positions
  outside the input write zero, so the whole output is written.
- **Validation flags.** The flag is one `uint32_t` word holding the lowest
  invalid position, initialized to `UINT32_MAX`, updated with `atomicMin`.
  Therefore the domain must satisfy `samples <= UINT32_MAX`: the element count
  for `validate_float_indices` (the baseline's `mod.rs:4296` cap) and the sample
  count `ceil(len/stride)` for `validate_real_cast` (the baseline caps the
  complex-real sample count, `mod.rs:4105`). Both planners reject larger
  domains, and the boundary is CPU-tested without allocating it. The flag and
  value buffers must be distinct allocations and must not alias the validated
  input; `init_validation` -> validate -> extract are ordered on one stream by
  the caller.
  `init_validation` writes `UINT32_MAX` and zero to the value slot. Extraction
  is one thread in one block and is the only reader; the caller decides when to
  read it back. Predicates and the extracted *value* semantics are preserved,
  including `max_inclusive` and the `stride` sampling of `validate_real_cast`.

## 3. Facade additions

New `Entry` variants, `Error` variants for axis/shape/compactness/direction
violations, one `plan_*` per entry, `CAxisValues` mirror, and
`ABI_REVISION = 2`. `diag_output_axis` is derived, not accepted.
`plan_pad` derives the expected output extents. All existing revision-1 checks
(span in elements and byte offsets, broadcast destinations, grid limits) are
reused.

## 4. Tests

CPU (no CUDA): axis and shape rejection (extract axes equal/out of range, embed
insertion position up to `in.rank`, derivation of the diagonal output axis and
of the expected output layout), compactness requirements, tril/triu extent and
length checks, slice start/step bounds and reachable-span checks, dynamic-slice
window/rank checks, pad's non-negative interior and derived output extents,
validation domain cap at `UINT32_MAX`, dtype gates, launch geometry (`grid` 1x1x1
for the one-thread entries), manifest/source/ABI consistency, `CAxisValues`
layout.

GPU (A100, `nvcc -cubin -arch=sm_80`, raw driver API):

- Extract diagonal against an independent CPU oracle (not only round trips) for
  rank 3 and rank 4 inputs, both axis orders, diagonal output position 0/middle/
  last, non-square and zero-length diagonals, non-compact input strides, and
  nonfinite payload bits. Embed as `fill_zero` + copy, verifying untouched
  off-diagonal sentinels.
- tril/triu against a CPU mask for rank 2 and rank 3 (batched) operands, `k`
  both signs, `|k|` beyond the extents, and `k = i64::MIN/MAX`.
- slice with steps > 1, starts at the extent boundary, non-compact input, odd
  tail; full-allocation sentinel comparison.
- dynamic_slice with device starts including negative and oversize values
  (clamping), float and integer index dtypes, window equal to the extent, and a
  fully clamped (single-window) case.
- pad with mixed positive/negative edge padding, interior padding > 0, negative
  edge low of large magnitude, and a fully padded-out axis; zero fill verified.
- Validation: initialization, earliest-position selection over a 2^16-element
  multi-block domain with NaN/Inf/non-integral/out-of-range values and many
  contending positions, no-write when everything is valid, the exact-integer
  cutoff neighbours, and `validate_real_cast` with `stride = 2`, nonfinite values
  in the skipped slots only, and both `max_inclusive` settings.
- Large (2^16 element) domains for every new entry that maps a tensor domain,
  with one physical launch per call.

## 5. Provenance

Same terms as the previous slice: dual MIT/Apache-2.0, the tenferro-rs copyright
line preserved, and a file-level derivation notice naming the pinned commit and
the translated functions, in `NOTICE` and in each `.cu` file.

## 6. Acceptance

- `cargo fmt --check`, `cargo clippy -j 16 --all-targets --features gpu-tests --
  -D warnings`, `cargo test -j 16` pass without CUDA.
- `cargo test -j 16 --features gpu-tests` passes on the local A100 with the
  hardware, driver, toolkit and command recorded.
- ABI document, manifest and `.cu` symbols agree (tested).
- [Review checklist](../review-checklist.md) sections 1–7 satisfied with
  evidence.
