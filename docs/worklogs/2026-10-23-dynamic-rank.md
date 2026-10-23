# Dynamic rank (ABI revision 9) — local evidence

Date: 2026-10-23. Issue:
[t4a-cuda-kernels#13](https://github.com/tensor4all/t4a-cuda-kernels/issues/13).
Design: [slice-dynamic-rank.md](../design/slice-dynamic-rank.md) (revision 2, after
the pre-implementation review; §8 answers every finding of revision 1).

## What landed

- `cuda/include/t4a/abi.cuh`: `T4A_CUDA_ABI_REVISION 9`, the four pointer-geometry
  structs (`t4a_cuda_layout_any`, `t4a_cuda_axis_values_any`,
  `t4a_cuda_gather_plan_any`, `t4a_cuda_scatter_plan_any`) and the additive
  general-path helpers (`t4a_cuda_decode_any`, `t4a_cuda_axis_value_any`,
  `t4a_cuda_axis_add_any`, `t4a_cuda_logical_count_any`,
  `t4a_cuda_compact_coordinate_any`). **The revision 1-8 helpers and bodies are
  unchanged**, so their compiled shapes and the paired-performance baseline stay
  valid.
- Thirty-one `_any` twins, one per rank-walking entry (movement ×4, window ×3,
  diagonal ×2, `combined_strided`, gather, scatter, both reductions, linalg-aux
  ×10, RRQR ×7). The general bodies keep no rank-sized state: the diagonal pair
  and `scatter_add` recompute each coordinate and per-axis base on demand instead
  of using the fast path's `[T4A_CUDA_MAX_RANK]` arrays, and no general body uses
  the masked eight-slot select or an unrolled `T4A_CUDA_MAX_RANK` bound.
- Facade: public `LayoutRef<'a>` (any rank, borrowed) and `WideLayout` (owned),
  `RankVariant` + `Launch::variant` / `Launch::with_variant`, `Entry::general()` +
  `GeneralEntry` (one variant per new symbol, `Entry::ALL` still enumerates the
  fast exports), the packing helpers (`LayoutRef::pack`, `pack_gather_plan`,
  `pack_scatter_plan`), the `CLayoutAny`/`CAxisValuesAny`/`CGatherPlanAny`/
  `CScatterPlanAny` mirrors with size/offset assertions, and `WideLayout`
  returns from `partials_layout_for`/`reduction_partials_layout`.
- All 45 layout-taking planners take `impl Into<LayoutRef<'a>>`, so every existing
  `&Layout` call site compiles unchanged while a rank-9/16 layout of either type
  is accepted.
- `docs/abi/c-kernel-abi.md` revision 9: the new types, the geometry contract, the
  rank rule, the fast/general pair table, "nothing in revisions 1-8 changed", and
  a rank-policy row in the specialization record.

## Local gates

Host: one physical **NVIDIA A100 80GB PCIe**, driver **580.126.09**, CUDA toolkit
**12.6** (`/usr/local/cuda-12.6`; `nvcc`/`ptxas`/`cuobjdump` 12.6 for the resource
table). Rust 1.97.

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | clean |
| `cargo clippy -j 16 --all-targets -- -D warnings` (default features) | clean |
| `cargo test -j 16` | 78 `planning` + 7 `fusion` passed |
| `cargo test -j 16 --features gpu-tests` | convert 7, **dynamic_rank 21**, elementwise 11, fusion 9, gather_scatter 16, kernels 14, linalg_aux 12, reduce 16, rrqr 17, slice_b 11 — all passed |
| `scripts/sass-diff.sh 36a5511` | 53/53 revision 1-8 symbols bit-identical; 31 added (re-checked after the header assertion change) |
| `scripts/resource-usage.sh` | every revision 1-8 row unchanged (0 stack / 0 `LDL`+`STL`; `t4a_cuda_unary` 72 / 20). Every `_any` row is also 0 / 0, e.g. `slice_any` 37 registers, `pad_any` 32, `reduce_any` 43, `extract_diagonal_any` 34 |

`cargo clippy --all-targets --features gpu-tests` fails on `main` too (78 errors in
`tests/gpu_fusion.rs`, `assign_op_pattern` and similar); that target set is not
this repository's gate and this change does not add to it.

## Test evidence

`tests/gpu_dynamic_rank.rs` (new target):

- `general_and_fast_agree_on_every_rank_up_to_eight` — ranks 2-8 with reversed
  compact strides and a non-trivial offset: the `_any` symbol and its revision 8
  counterpart produce the same destination, and both match the CPU reference.
- `rank_nine_and_sixteen_match_the_cpu_reference` — ranks 9 and 16, domain larger
  than one block, reversed strides, full-allocation comparison.
- `rank_nine_slice_matches_the_cpu_reference` — rank 9 with `CAxisValuesAny`
  starts/steps.
- `rank_nine_reduce_matches_the_cpu_reference` — rank 9, axis 4, unit strategy.
- `rank_nine_gather_matches_the_cpu_reference` — rank-9 operand with a rank-1
  gathered window (the mixed-rank case).
- `rank_nine_scatter_add_matches_the_cpu_reference` — rank 9 with a full-rank batch
  index tensor; the allocation-free base recomputation matches the CPU reference.

`tests/planning.rs` (CPU): rank 9/16 acceptance and `RankVariant::General` for
every in-scope family; the fast policy at rank ≤ 8; a rank-8 `WideLayout` still
selecting `Fast` (the variant follows ranks, not the host storage type); the
variant selection across several layouts and map lengths; packing order and
lengths; `Overflow` for an unrepresentable geometry length and an extent product
that leaves `i64`; `GeneralEntry` mirroring its fast entry's dtype matrix; and the
extended export test (`Entry::ALL` + `GeneralEntry::ALL` == the exported symbols,
no `_any` twin for a flat entry).

## Not done here

- The paired A100 performance matrix is **not** rerun, and the justification is now
  compiled-code evidence rather than source identity: `scripts/sass-diff.sh
  36a5511` compares the SASS of every kernel symbol present in both trees
  (`nvcc -std=c++17 -cubin -arch=sm_80`, CUDA 12.6) with the line-address comment
  and cuobjdump's column padding normalised out. Result: **53 symbols present in
  both trees, 53 bit-identical instruction streams, 31 added by this branch**. So
  every revision 1-8 entry is unchanged at the compiled level and none needs a
  paired rerun; the accepted performance residuals therefore stay accepted.
- The general path is covered at rank 9/16 for movement, slice, reduce, gather and
  scatter; the diagonal, pad, `combined_strided`, linalg-aux and RRQR general bodies
  are covered by the rank ≤ 8 cross-variant equality only (movement is the only
  family with both). Their rank-9/16 numerical cases are the next addition.
- The tenferro consumer port (tenferro-rs#2208 U1-U7) is untouched by this slice.
