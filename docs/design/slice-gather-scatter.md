# Slice D: gather and additive scatter

Status: implemented and GPU-validated, 2026-10-20 (ABI revision 4). Fourth implementation slice of
[the migration design](cubecl-kernel-port.md), completing the indexing family.

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-gpu/src/kernels/indexing.rs` (`gather_kernel` L263,
`scatter_copy_kernel` L343, `scatter_float_kernel` L350,
`scatter_complex_kernel` L455, `clamp_window_start` L63, `index_component` L82,
`flat_to_index_batch_index` L108, `flat_to_update_window_index` L139,
`update_window_len` L160) and the caller contracts in
`crates/tenferro-gpu/src/cubecl/mod.rs` (`gather_launch_meta` L5622,
`scatter_launch_meta`, `scatter_typed` L3820-4000, `GatherConfig`).

## 1. Scope

| Entry | Ported from | Dtypes |
| --- | --- | --- |
| `t4a_cuda_gather` | `gather_kernel` | all data dtypes; index F32/F64/I32/I64 |
| `t4a_cuda_scatter_add` | `scatter_float_kernel`, `scatter_complex_kernel` | F32/F64/C32/C64 |

`scatter_copy_kernel` needs no new entry: it is a compact flat copy that
`t4a_cuda_copy_exact` already provides; the manifest records the mapping. The
additive scatter is the two-launch composition the baseline uses: copy the
operand into the destination (caller-owned, `copy_exact` on the caller's
stream), then add the updates at the scattered positions.

Deferred: conversions and linalg/RRQR helpers (own slices), the fused-expression
emitter, NVRTC prepare/cache, and any non-XLA indexing form.

## 2. ABI additions (revision 4)

The indexing dimension maps are operation attributes, not layout values, so they
travel in a by-value POD per operation: the gather plan is 280 bytes and the
scatter plan is 216 bytes, both align 8.

```c
typedef struct {          /* gather plan, 280 bytes, align 8 */
  int32_t operand_rank;   /* 1..=8 */
  int32_t out_rank;       /* 0..=8; a scalar gather has rank 0 */
  int32_t index_vector_dim;   /* 0..=start_indices_rank */
  int32_t window_rank;        /* number of window dims */
  int32_t start_map_len;      /* == index vector size */
  int32_t reserved;           /* 0 */
  int64_t window_dims[8];     /* operand axis per window position */
  int64_t offset_dims[8];     /* output axis per window position */
  int64_t start_index_map[8]; /* operand axis per start component */
  int64_t slice_sizes[8];     /* window extent per operand axis */
} t4a_cuda_gather_plan;

typedef struct {          /* scatter plan, 216 bytes, align 8 */
  int32_t operand_rank;
  int32_t updates_rank;
  int32_t index_vector_dim;
  int32_t window_rank;
  int32_t scatter_map_len;    /* == index vector size */
  int32_t reserved;
  int64_t window_dims[8];        /* operand axis per window position */
  int64_t update_window_dims[8]; /* updates axis per window position */
  int64_t scatter_dims[8];       /* operand axis per start component */
} t4a_cuda_scatter_plan;

void t4a_cuda_gather(void* out, const void* operand, const void* indices,
                     int32_t dtype, int32_t index_dtype, t4a_cuda_layout out_layout,
                     t4a_cuda_layout operand_layout, t4a_cuda_layout indices_layout,
                     t4a_cuda_gather_plan plan, int64_t output_len);
void t4a_cuda_scatter_add(void* out, const void* indices, const void* updates,
                          int32_t dtype, int32_t index_dtype, t4a_cuda_layout out_layout,
                          t4a_cuda_layout indices_layout, t4a_cuda_layout updates_layout,
                          t4a_cuda_scatter_plan plan, int64_t update_len);
```

The additive entry takes no operand pointer: the destination already holds the
operand copy, and `out_layout` supplies every extent the fit check needs. The
scatter window shape is the updates extents at `update_window_dims` (one on every
other operand axis), exactly as the baseline derives it; scatter therefore
carries no `slice_sizes`.

Semantics preserved from the baseline:

- **Gather.** Output coordinates are decoded in order; the window offsets fill the
  `window_dims` positions, the remaining output coordinates fill the index batch
  shape in order. Each start component `c` addresses operand axis
  `start_index_map[c]`, read from the index tensor at
  `index_component` (either the flat batch position, or the component inserted at
  `index_vector_dim`), and the window start is `clamp_window_start`
  (`start <= 0 -> 0`, else `min(trunc(start), extent - window)`). The element is
  read at the strided operand address. This is the XLA `Gather` semantics the
  baseline implements; the operand is read directly, never materialized. A rank-0
  output (scalar gather, one element) is valid.
- **Scatter add.** One thread per update element. The update window index is
  decoded over `update_window_dims` (in ascending updates-axis order); the batch
  index comes from the remaining updates coordinates, which must equal the index
  batch shape; the operand base is the clamped window start per `scatter_dims`;
  a window that does not fit is skipped (the baseline checks
  `operand_base + window_shape <= operand.extent`). Scatter therefore does not
  reject a window larger than the operand — the fit check runs per element.
  Otherwise the destination is `operand_base + window_offset` and the update is
  added atomically. A complex update adds its two components as separate
  32/64-bit atomics.
- **Index tensors.** `start_indices` / `scatter_indices` are device-resident
  **compact** tensors of F32/F64/I32/I64. The index layout is an entry argument,
  so strides are technically transported, but the baseline's batch/component
  addressing assumes a compact index tensor; a reversed or offset index view is
  therefore rejected at planning. Their
  values must be the validated integral values the baseline accepts: integers,
  or floats that are integral and within `|v| <= 2^24` (F32) / `2^53` (F64).
  That bound is a raw-entry precondition; the kernels do not re-validate, and the
  ABI never reads the index tensor on the host.
- **Additive scatter is not deterministic under collisions** and needs device
  atomic support for the element type. The caller initializes the destination
  with an operand copy on the same stream before the additive launch. Which
  movement entry that copy uses depends on the layouts: `t4a_cuda_copy_exact` for
  two compact buffers, `t4a_cuda_contiguous_to_view` when the destination is a
  strided view, and `t4a_cuda_strided_to_strided` when both are strided.
  `copy_exact` ignores offsets and strides, so it must not be used for a view.
- **Empty semantics.** An empty destination or zero updates plan no additive
  launch. A nonempty destination with zero updates still needs the
  initialization copy, exactly as the baseline does; the caller decides by
  checking the update length it computed.
- **Aliasing.** The gather output must be disjoint from the operand and the index
  tensor; additive-scatter updates and indices must be disjoint from the
  destination. Destinations must be injective for the copy stage (the gather
  destination and the scatter destination), which the planners check with the
  revision-1 rule; full injectivity remains the caller's proof.

## 3. Facade additions

- `GatherPlan` / `ScatterPlan` mirrors (`CGatherPlan`, `CScatterPlan`),
  `ABI_REVISION = 4`, entries `Gather`, `ScatterAdd`, and the recorded
  `scatter_copy -> CopyExact` consolidation.
- `plan_gather(dtype, index_dtype, out_layout, out_len, operand_layout,
  operand_len, indices_layout, indices_len, plan)` validating: ranks and index
  vector size match `start_index_map`, unique/valid axes, `window_dims` and
  `offset_dims` consistency, the derived output extents equal `out_layout`
  (a rank-0 output included), the slice bound on every operand axis
  (`<= extent` on window axes, `== 1` on collapsed axes), compact `indices`,
  spans and grid limits.
- `plan_scatter_add(dtype, index_dtype, out_layout, out_len, indices_layout,
  updates_layout, updates_len, plan)` validating the scatter maps, compact
  indices, non-window updates extents equal to the index batch shape,
  `update_len == batch_product * window_product`, `out_layout` extents equal to
  the operand extents implied by the maps and updates, and the destination
  checks. Scatter does not require `slice_sizes <= operand`.
- A `window_dims(operand_rank, collapsed)` helper so callers do not reimplement
  the window-axis derivation; the planners derive and check the output extents
  themselves.

## 4. Tests

CPU: index-vector-size and map-length mismatches, duplicate/out-of-range axes,
`slice_sizes` exceeding the operand, `collapsed` dims with a non-unit slice,
batch/window extent derivation, spans (including a reversed/offset index
tensor), grid limits, dtype gates (Bool data with float indices, complex scatter
accepted, int scatter rejected), plan-mirror layouts.

GPU (A100), all against CPU references:

- Gather with: no index vector (scalar starts), an index vector in the middle
  and at the end, offset dims not in sorted order, collapsed dims, clamping at
  both ends, a strided operand layout with an offset, F64/I64 index dtypes, a
  rank-0 output, and a large (2^16) case. Reversed or offset index layouts are
  rejected at planning.
- Scatter add with: no index vector, an index vector, unsorted
  `update_window_dims`, a window larger than one, collisions (compared against
  the CPU sum), real and complex dtypes, a window that does not fit (skipped), a
  strided destination and strided updates with sentinel holes, the ordered
  two-launch composition (copy then add, enqueued with `launch_async`, one
  synchronization, two launches, nonzero operand), the zero-update case (one
  copy launch), and a 2^16-update collision case.
- Launch counts: one for gather, one for scatter add, none for an empty plan,
  two for the ordered scatter composition.
- The untouched-storage rule: destination sentinel positions outside the written
  region stay untouched for gather.

## 5. Provenance

Same terms as the earlier slices: tenferro-rs translation notice in
`cuda/src/gather_scatter.cu` and the host derivation region of `src/lib.rs`, with
the inherited notices recorded in `THIRD_PARTY_NOTICES.md` (this family carries
no additional third-party notice at the pinned commit).

## 6. Acceptance

- Local gates (fmt, clippy, CPU tests) pass without CUDA; GPU tests pass on the
  A100 with hardware/driver/toolkit/command recorded.
- ABI document, manifest and `.cu` symbols agree.
- [Review checklist](../review-checklist.md) sections 1–7 satisfied.
