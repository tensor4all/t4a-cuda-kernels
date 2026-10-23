# The remaining kernel families and the inventory proof (ABI revision 10)

Date: 2026-10-23. Issue:
[t4a-cuda-kernels#14](https://github.com/tensor4all/t4a-cuda-kernels/issues/14).
Design: [slice-remaining-families.md](../design/slice-remaining-families.md)
(revision 2, after the pre-implementation review; §8 answers every finding).

## What landed

- **Two new entries** with their revision 9 general-path twins, all in
  `cuda/src/movement.cu`:
  - `t4a_cuda_broadcast_in_dim` / `_any` — rank-mapped broadcast. The map reuses
    `t4a_cuda_axis_values` (by value) and `t4a_cuda_axis_values_any` (device
    memory); `dims.values[src_axis]` is the destination axis, `rank` is the input
    rank and may be 0 (a scalar input), and an input extent of one broadcasts.
  - `t4a_cuda_concatenate_copy` / `_any` — one launch per input, destination shifted
    by `axis_offset` along `axis`.
- **Two families folded instead of duplicated**: `reverse` is a strided copy with
  negated strides and a rebased offset (`offset' += (extent - 1) * stride`, then
  `stride = -stride`) through `materialize_strided` (reading) and
  `contiguous_to_view` (writing); `scale_in_place_float`/`_complex` is
  `t4a_cuda_binary` with `op = MUL`, a scalar `rhs` and `out` aliasing `lhs`, which
  the ABI's aliasing rule already permits. Both mappings are proved by GPU tests.
- **Facade**: `Entry::{BroadcastInDim, ConcatenateCopy}` plus their
  `GeneralEntry` twins, `plan_broadcast_in_dim`, `plan_concatenate_copy`,
  `ABI_REVISION = 10`, and the ABI document's revision 10 section (types reused, the
  two entry rows, the fold decisions, the map row in the specialization record).
- **The inventory proof**: `tests/data/tenferro-cuda-inventory.tsv` pins the 173
  `entry`/`macro_entry` rows of `docs/kernel-inventory.md` with the ABI target each
  one resolves to, and `every_pinned_tenferro_cuda_kernel_resolves_to_an_entry`
  asserts a pinned row count, valid kinds and fields, that every target is an
  exported entry or a named vendor route (`cuBLAS`/`cuBLASLt`/`cuTENSOR`/`cuSOLVER`/
  `cuFFT`), that the two pinned lists name the same symbol set, and that every fast
  entry is either mapped or listed as ABI-introduced (`div`, `fill_one`,
  `lu_parity`, `init_validation`, the fused entry).

## Local gates

Host: one physical **NVIDIA A100 80GB PCIe**, driver **580.126.09**, CUDA toolkit
**12.6**.

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | clean |
| `cargo clippy -j 16 --all-targets -- -D warnings` | clean |
| `cargo test -j 16` | 82 `planning` (incl. the inventory proof and the two new planners) + 7 `fusion` passed |
| `cargo test -j 16 --features gpu-tests --test gpu_remaining_families` | 6 passed |
| `scripts/resource-usage.sh` | `t4a_cuda_broadcast_in_dim` 0 stack / 0 `LDL`+`STL`, `t4a_cuda_concatenate_copy` 0 / 0; their `_any` twins 0 / 0; every revision 1-9 row unchanged (`t4a_cuda_unary` still 72 / 20) |
| `scripts/sass-diff.sh 6237c15` | 84/84 revision 1-9 symbols bit-identical, 4 added — the paired matrix is not rerun |

## Test evidence

`tests/gpu_remaining_families.rs`:

- `broadcast_in_dim_matches_the_cpu_reference_for_every_dtype` — all seven dtypes
  bitwise (Bool, complex and integer included), an identity map, an extent-1
  broadcast axis, a non-self-inverse permutation `dims = [2, 0, 1]` with the
  destination extents permuted to match, a scalar input (`rank = 0`), and rank 9 and
  16 with non-unit extents on the high axes and more than one block. Fast and
  general variants are both checked against the CPU image and their guards survive.
- `concatenate_copy_matches_the_cpu_reference_for_every_dtype` — all seven dtypes
  bitwise with two successive offsets and then a duplicated (overlapping) offset in
  stream order, a strided destination on a non-leading axis, and rank 16 with the
  shifted axis at 9.
- `new_entries_agree_between_variants_up_to_rank_eight` — fast versus `_any`
  byte equality for both entries at ranks 1-8.
- `reverse_is_a_strided_copy_in_both_directions` — three masked axes out of four
  with odd extents, reading through `materialize_strided` and writing through
  `contiguous_to_view`, compared against the permutation the reversed layout
  predicts.
- `copy_entries_preserve_nonfinite_payloads_bitwise` — NaN payloads, signed zeros and
  infinities through both copy entries, and a genuinely non-compact broadcast source
  (`extents [2, 3]`, `strides [1, 5]`, capacity 12).
- `scale_in_place_is_the_scalar_mul_of_binary` — F32/F64/C32/C64 in place through
  `t4a_cuda_binary` with the destination aliasing `lhs`; every element is compared
  against its product with the factor, the slack keeps its sentinel and the status
  word stays clear.

## Not done here

- Integer scaling is **not** claimed: tenferro has float and complex in-place
  kernels only.
- The post-implementation review corrected one mapping: tenferro's
  `scatter_copy_kernel` is a flat `out[i] = operand[i]` initialization copy, not an
  indexed scatter, so it maps to `t4a_cuda_copy_exact`. Only `scatter_float_kernel`
  and `scatter_complex_kernel` are additive scatters and map to
  `t4a_cuda_scatter_add`.
- The inventory test now cross-checks kind, source file and line against
  `docs/kernel-inventory.md` for every row, not only the names, and rejects an
  unpinned line number.
- `docs/kernel-inventory.md` is pinned to `tenferro-rs@14bc1ab9`; when tenferro
  changes its CUDA kernel set this list and the test's pinned count must move with
  it. The test fails loudly when they do not.
