# Slice C: single-axis reductions with unit, plane and split strategies

Status: implemented and GPU-validated, 2026-10-20 (ABI revision 3). Third implementation slice of
[the migration design](cubecl-kernel-port.md), covering implementation-sequence
step 2's reduction family.

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-gpu/src/kernels/reduce/{kernels,launch,routines}.rs` and
`kernels/helpers.rs`, plus the [performance preservation contract §3
"Reductions"](cubecl-performance-parity.md#reductions).

## 1. Scope

Two entries cover five operations and three algorithm strategies:

| Entry | Ported from | Strategies |
| --- | --- | --- |
| `t4a_cuda_reduce` | `reduce_{sum,prod,sum_squares}_{float,complex,int}`, `reduce_{max,min}_{float,int}`, `_plane` twins | unit, plane |
| `t4a_cuda_reduce_partials` | `reduce_*_chunked` | split pass 1 |

Operations (runtime tag): `SUM`, `PROD`, `SUM_SQUARES`, `MAX`, `MIN`.

| Op | Dtypes |
| --- | --- |
| `SUM`, `PROD` | F32/F64/I32/I64/C32/C64 |
| `SUM_SQUARES` | F32/F64 |
| `MAX`, `MIN` | F32/F64/I32/I64 |

Deferred with reasons: multi-axis reduction and non-keepdims output belong to the
consumer's semantics layer (the baseline reduces one axis into a keepdims shape);
bool reductions and the dynamic-update/indexing gaps are not in the baseline
reduction set either.

## 2. ABI additions (revision 3)

Revision 1 and 2 entries are unchanged.

The canonical signatures, tag values and struct layouts are in
[docs/abi/c-kernel-abi.md](../abi/c-kernel-abi.md) (revision 3); `plane_width` is
launch geometry taken from `blockDim.x`, not an entry argument.

Semantics that must be preserved:

- **Keepdims shape.** `out` has the input rank with `out.extents[axis] == 1`;
  `output_len == out.element_count()`. The caller reduces one axis only.
  `in.extents[axis] >= 1` (the baseline rejects a zero-length reduce axis, and
  the kernels seed the accumulator with the first element rather than an
  identity).
- **Unit strategy.** One thread per output, the thread's global index decodes to
  an output coordinate, and it walks the axis serially. The planner only allows
  it when `reduce_len <= plane_width`, which is the contract that keeps
  tensor-sized work off a single thread.
- **Plane strategy.** One block per output; lane `l` starts at axis index `l`
  and strides by `plane_width`; the plane reduction combines the lanes; lane 0
  writes. The planner only allows it when `reduce_len >= plane_width`. Plane
  reduction uses `__shfl_xor_sync` with ascending offsets `1, 2, 4, ... <
  plane_width`, matching the baseline's combination order, so cancellation
  results are not reordered. `plane_width` must be a power of two in
  `{8, 16, 32}`, the block is exactly `plane_width`, and every lane participates
  (no lane returns before the shuffles; lane 0 is the only writer).
- **Split pass 1.** One block per `(output, split)` with
  `cube = output_index * splits + split`; the block reduces
  `[split*chunk_len, min(axis_len, split*chunk_len + chunk_len))`, and the last
  split absorbs `reduce_len % splits` extra elements. It writes one partial at
  the reduce axis coordinate `split` of the partials layout.
  The partials layout keeps the output's **non-axis extents** and sets the axis
  extent to `splits`; its strides are the caller's own and are validated
  independently, because inheriting the output strides can make two distinct
  partials address the same element (compact output `[2,1,3]` has strides
  `[1,2,2]`; the `[2,2,3]` partials would overlap).
  A chunk shorter than a plane cannot be reduced safely, because every lane
  seeds its accumulator from its own chunk element: the planner requires
  `chunk_len == reduce_len / splits` and `chunk_len >= plane_width`, which is
  exactly the baseline's floor split with `min_chunk = plane_width * 8`.
- **Pass 2** is the same `t4a_cuda_reduce` over the partials, with `SUM` for a
  `SUM_SQUARES` pass 1 (the partials already hold sums of squares) and the same
  op otherwise. Passes are never recursively split; the caller issues exactly two
  launches on one stream with no host synchronization between them.
- **Arithmetic.** `SUM`/`PROD` seed from the first element; integer `SUM`/`PROD`
  wrap; `SUM_SQUARES` squares each element through `fma(v, v, 0)` so the square
  rounds before the addition and cannot be contracted into it; complex `SUM`/
  `PROD` use the element-wise complex forms from revision 1.
- **Single-element axes.** The accumulator is seeded with the first element (or
  its square for `SUM_SQUARES`), so a length-1 axis returns that element for sum,
  prod, max and min: `SUM([-0.0])` keeps its sign and `PROD([(Inf, 1)])` stays
  `(Inf, 1)` instead of acquiring a NaN imaginary part from a complex identity.
  `SUM_SQUARES([3])` is `9`. Tests cover this bit-sensitively.
- **NaN semantics.** Float `MAX`/`MIN` accumulate with the NaN-propagating
  pairwise form and, across the plane, report NaN when any lane holds one
  (summing the NaN values, matching the baseline). A NaN payload is therefore
  not preserved bit-for-bit; only "is NaN" is contractual. `-0.0`/`+0.0`
  ordering follows `fmaxf`/`fminf` semantics used by the baseline.
- **Aliasing.** Each pass's source and destination must be disjoint and the
  destination must be injective (no broadcast axis, per the revision-1 rule).
  Partials are pass 2's source: they must be disjoint from the original input and
  from the final output. Span checks alone do not
  establish this, and full injectivity remains the caller's proof.
- **Determinism.** The split topology is fixed by the planner
  (`splits = clamp(reduce_len / (plane_width*8), 1, 4096)`, floor `chunk_len`),
  so a given plan is reproducible; results are not bitwise equal to a CPU sum in
  a different order.
- **Geometry.** Unit: `grid = ceil(output_len / plane_width)`, `block =
  plane_width`. Plane: `grid = output_len`, `block = plane_width`. Pass 1:
  `grid = output_len * splits`, `block = plane_width`. All products, grid limits
  and spans are checked by the facade.

## 3. Facade additions

- `ReduceOp`, `ReduceStrategy`, `Split { splits, chunk_len }` and
  `ABI_REVISION = 3`; `Entry::{Reduce, ReducePartials}`.
- `plan_reduce(dtype, op, strategy, out_layout, out_len, in_layout, in_len, axis,
  plane_width)`: dtype gate, keepdims shape and output-length cross-check,
  `reduce_len >= 1`, strategy bounds (`unit: reduce_len <= plane_width`,
  `plane: reduce_len >= plane_width`), `plane_width in {8,16,32}`, span checks,
  grid limits. `out_len` is the destination **allocation capacity**, as in the
  revision-1/2 planners; the kernel's `output_len` argument is
  `out_layout.element_count()`.
- `plan_reduce_partials(...)`: same checks plus `splits >= 2`,
  `chunk_len >= plane_width`, `chunk_len == reduce_len / splits`,
  `partials.extents[axis] == splits`, equal non-axis extents,
  `splits * chunk_len <= reduce_len`, injective partials destination, and
  `grid = output_len * splits` within the limit.
- `split_policy(reduce_len, plane_width) -> Option<Split>`: the baseline's
  documented policy, host-side and unit-testable without a device.
- `combine_op(op) -> ReduceOp`: `SUM_SQUARES -> SUM`, otherwise identity.
- `plan_reduction(...) -> ReductionPlan`: the Auto decision, ported as a pure
  host function so a consumer cannot accidentally leave a long axis on one warp.
  It resolves the direct strategy (`reduce_len <= plane_width ? Unit : Plane`),
  applies `split_policy`, and for a split returns both launches plus the combine
  strategy resolved for the `splits`-long axis **without** applying the split
  policy again. `ReductionPlan::Direct { strategy, launch }` or
  `ReductionPlan::Split { split, pass1, combine_strategy, pass2, combine_op }`.
  A caller that wants a specific direct pass can still plan it explicitly with
  `plan_reduce`, which is also how pass 2 is expressed.

## 4. Tests

CPU: strategy boundary exactly at `plane_width` (both sides), unit strategy
rejected beyond it, plane strategy rejected below it, plane-width validation,
keepdims/axis/rank and output-length rejection, span checks, split policy values
across 255/256/511/512/513/1e6/1e7 and the 4096 cap, remainder handling in the
last split, partials layout cross-checks, combine-op mapping, dtype gates
(`MAX` on C64 rejected, `SUM_SQUARES` on I32 rejected), grid limits, manifest and
ABI-document consistency.

GPU (A100, `nvcc -cubin -arch=sm_80`, raw driver API), all against CPU
references. The split tests enqueue both passes on one stream before any
observation, so the "no host synchronization between passes" contract is actually
exercised; partials are inspected only after the second pass has been submitted
and the stream synchronized once.

- Every op and dtype route, both strategies, on a rank-3 input reducing a
  **middle** axis (so skipping the axis cannot pass), with a strided input
  (including a reversed axis) and an output with a non-compact stride.
- The strategy boundary: `reduce_len = 31/32/33` for plane-width 32.
- Split reductions with remainders: `reduce_len = 511/512/513/1000`, verifying
  pass 1 partials element by element (sentinel-filled partials allocation, so an
  omitted store or a hole is visible), then pass 2, and that `SUM_SQUARES` uses a
  plain `SUM` in pass 2. A NaN confined to one split must survive pass 2.
- A plane combine: `reduce_len = 8448` gives 33 partials, so pass 2 is a plane
  reduction; and a capped case above 4096*256 elements splits into 4096 chunks.
- Length-1 axes: `SUM([-0.0])` keeps its sign, `PROD` of a single complex value
  with an infinite component keeps its bits, and an FMA-sensitive
  `SUM_SQUARES` case is checked under the declared compiler options.
- NaN/Inf/signed-zero for float `MAX`/`MIN` (all-NaN slice, NaN in one lane
  only, `-0.0` vs `+0.0`), integer wrap for `SUM`/`PROD` at the `i32`/`i64`
  boundaries, complex `SUM`/`PROD` with nonfinite components.
- Plane widths 32, 16 and 8 (each requiring `reduce_len` on both sides of that
  width), so subwarp participation is exercised.
- Launch counts: one physical launch for a policy-chosen direct reduction,
  exactly **two** for a split reduction (including the 2^16-element long-axis
  case, which the policy splits into 256 chunks), and none for a validated empty
  output.
- 2^16-element domains in both shapes: few outputs over a long axis (split, two
  launches) and many outputs over a short axis (one launch).

## 5. Provenance

The reduction sources record derivation from cubek-reduce
(`9cf90b797107d46829e1c9d9355ce801c3dd4a7d`), "Copyright (c) 2022 Nathaniel
Simard & CubeCL Framework Contributors", MIT OR Apache-2.0. This is the first
slice with a third-party notice inherited beyond tenferro-rs itself, so:

- `THIRD_PARTY_NOTICES.md` is added with the cubek-reduce section from
  tenferro-rs (upstream URLs, commit, original source paths, the original
  copyright line and the full MIT text; the Apache-2.0 text is the repository's
  `LICENSE-APACHE`), plus the tenferro-rs attribution.
- `cuda/src/reduce.cu` carries the same copyright line, the upstream source
  paths, the pin and the nature of the translation in a file-level notice, and
  so does the reduction-planning region of `src/lib.rs`.

## 6. Acceptance

- `cargo fmt --check`, `cargo clippy -j 16 --all-targets --features gpu-tests --
  -D warnings`, `cargo test -j 16` pass without CUDA.
- `cargo test -j 16 --features gpu-tests` passes on the local A100 with hardware,
  driver, toolkit and command recorded.
- ABI document, manifest and `.cu` symbols agree; the performance contract's
  reduction requirements (bounded unit work, plane strategy, split planner
  parameters, last-chunk remainder, plain-sum pass 2, no host sync between
  passes, no recursive split) are each covered by a check.
- [Review checklist](../review-checklist.md) sections 1–7 satisfied.
