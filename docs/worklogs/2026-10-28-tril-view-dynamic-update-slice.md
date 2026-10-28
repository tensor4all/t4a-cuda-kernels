# Triangle views and the window write (ABI revision 11)

Date: 2026-10-28. Issue:
[t4a-cuda-kernels#18](https://github.com/tensor4all/t4a-cuda-kernels/issues/18).
Design: [slice-layout-tril-dus.md](../design/slice-layout-tril-dus.md)
(revision 2, answering the pre-implementation findings), consumer tenferro-rs#2223.

Baseline `main` at ABI revision 10 (merge `5019c71`); branch
`feat/rev11-tril-dus`.

## What landed

- **Six new symbols, four new CUDA bodies, no new struct**: revision 11 is
  additive, like revision 9.
  - `t4a_cuda_tril_view` / `t4a_cuda_triu_view` and their `_any` twins in
    `cuda/src/diagonal.cu` (`triangular_view_body`, `triangular_view_body_any`):
    the flat mask formula applied to a compact output over the input's extents,
    with the input read through its own offset and signed strides (transposed,
    negative-stride and zero-stride inputs need no staging copy). The mask
    predicate is `col - row` against `k`, so `i64::MIN`/`i64::MAX` are defined;
    a masked element stores zero and touches no input address.
  - `t4a_cuda_dynamic_update_slice` and its `_any` twin in
    `cuda/src/window.cu` (`dynamic_update_slice_body`,
    `dynamic_update_slice_body_any`): one thread per **update** element, the
    existing `clamp_window_start` for the window origin, no new clamp helper and
    no atomics. The compact destination address is decoded from the output's own
    extents, so the stores are injective by construction; an empty update plans
    zero launches and a rank-0 update writes one element without reading a start.
- **Facade**: `Entry::{TrilView, TriuView, DynamicUpdateSlice}` (so
  `Entry::ALL` = 58), the three `GeneralEntry` twins (`GeneralEntry::ALL` = 36),
  `plan_tril_view`, `plan_triu_view`, `plan_dynamic_update_slice`,
  `ABI_REVISION = 11`. `plan_triangular_view` reuses `require_matrix` for the
  rank >= 2 rule and `check_span` for the input; the output needs no layout (it
  is the compact form of the input's extents at offset zero), so `out_len` is
  only the caller's allocation length. `plan_dynamic_update_slice` checks the
  data dtype, the index dtype ({F32, F64, I32, I64}), `require_compact(out)`,
  equal ranks including `starts_rank`, `update.extents[a] <= out.extents[a]`,
  and both spans.
- **ABI document**: `docs/abi/c-kernel-abi.md` gets a "Revision 11" section (the
  six entry rows, the per-entry contracts, the overlap rule), the revision
  paragraph, the `window.cu`/`diagonal.cu` fast/general pair rows and a launch
  geometry row. `cuda/include/t4a/abi.cuh` only bumps
  `T4A_CUDA_ABI_REVISION` to 11: like every earlier revision, it declares types
  and helpers, not entry prototypes.
- **Inventory**: the pinned 173-row upstream inventory is historical and
  unchanged. The three new fast symbols are registered as ABI-introduced in the
  reverse inventory gate (`tests/planning.rs`, next to `t4a_cuda_div`).

## Local gates

Host: one physical **NVIDIA A100 80GB PCIe**, driver **580.178.04**, CUDA
toolkit **12.6** (`Cuda compilation tools, release 12.6, V12.6.85`), `sm_80`,
4 cores (`-j 4`, GPU tests `--test-threads 1`). The GPU had no compute processes
before the runs (`nvidia-smi --query-compute-apps=pid --format=csv,noheader`
empty).

| Command | Result |
| --- | --- |
| `cargo fmt --check` | clean |
| `cargo clippy -j 4 --all-targets -- -D warnings` | clean |
| `cargo clippy -j 4 --all-targets --features gpu-tests -- -D warnings` | **fails on pre-existing test files**, see below; the new `tests/gpu_tril_dus.rs` contributes 0 warnings |
| `cargo test -j 4` | 7 `fusion` + 86 `planning` passed |
| `cargo test -j 4 --features gpu-tests -- --test-threads 1` | 14 targets, **253 tests passed, 0 failed** (incl. the new 9 in `gpu_tril_dus`) |
| `scripts/sass-diff.sh 5019c71` | 88 symbols present in both trees, **88 identical, 0 differ, 6 added** |
| `scripts/resource-usage.sh` | 88 rows identical to `5019c71`, 6 added, all 0 stack / 0 `LDL`+`STL` |

### The `gpu-tests` clippy gate is pre-existing broken on this toolchain

`cargo clippy --all-targets --features gpu-tests -- -D warnings` fails on files
this branch does not touch: `tests/gpu_fusion.rs`, `tests/gpu_linalg_aux.rs`,
`tests/gpu_rrqr.rs`, `tests/gpu_dynamic_rank.rs`, `tests/gpu_elementwise.rs`,
`tests/gpu_remaining_families.rs` (lints such as `assign_op_pattern`,
`too_many_arguments`, `needless_borrows_for_generic_args`, `useless_vec`,
`unnecessary_cast`). The same command on the untouched baseline worktree
(`git worktree add 5019c71`) fails the same way:

- `cargo clippy -j 4 --all-targets --features gpu-tests` on `5019c71` with
  `rustc 1.99.1` → `error: could not compile t4a-cuda-kernels (test
  "gpu_linalg_aux") due to 3 previous errors; 12 warnings emitted`.
- the same on `5019c71` with `1.89.0` → 28 errors (all in `gpu_elementwise` and
  other untouched test files); `1.96.0` → 4 errors in `gpu_rrqr`.

The failing files are outside this slice, so they were left alone rather than
reformatted here. Filtering the head run's diagnostics to
`tests/gpu_tril_dus.rs` yields no output, i.e. the new file is warning-free under
the default toolchain.

## SASS identity of the unchanged entries

`scripts/sass-diff.sh 5019c71` compiles every translation unit in both trees
with `nvcc -std=c++17 -cubin -arch=sm_80`, normalizes the `cuobjdump -sass` text
per symbol and compares the instruction words:

```
symbols present in both trees: 88
identical: 88
added by this branch: 6
```

A targeted per-symbol digest (sha256 of the normalized SASS body, first 16 hex
digits) for the entries named by the task plus their neighbours:

| source | symbol | 5019c71 | head | |
| --- | --- | --- | --- | --- |
| `diagonal.cu` | `t4a_cuda_tril` | `f61cd30e718f89b1` | `f61cd30e718f89b1` | identical |
| `diagonal.cu` | `t4a_cuda_triu` | `a264f56fe8631ff9` | `a264f56fe8631ff9` | identical |
| `diagonal.cu` | `t4a_cuda_extract_diagonal` | `41dfc3d7dd203cd4` | `41dfc3d7dd203cd4` | identical |
| `diagonal.cu` | `t4a_cuda_embed_diagonal_copy` | `c0d2006b724b9233` | `c0d2006b724b9233` | identical |
| `window.cu` | `t4a_cuda_dynamic_slice` | `5f1d9dfc7106f7d8` | `5f1d9dfc7106f7d8` | identical |
| `window.cu` | `t4a_cuda_slice` | `cff7934b13a10e7c` | `cff7934b13a10e7c` | identical |
| `window.cu` | `t4a_cuda_pad` | `3d145a763f054ded` | `3d145a763f054ded` | identical |
| `diagonal.cu` | `t4a_cuda_tril_view` | — | `49f09fdca6066a19` | new |
| `diagonal.cu` | `t4a_cuda_triu_view` | — | `50261e7fab70366a` | new |
| `diagonal.cu` | `t4a_cuda_tril_view_any` | — | `4b6d54c5bc9b892b` | new |
| `diagonal.cu` | `t4a_cuda_triu_view_any` | — | `d57ff6244bfa35a3` | new |
| `window.cu` | `t4a_cuda_dynamic_update_slice` | — | `f010973230342d07` | new |
| `window.cu` | `t4a_cuda_dynamic_update_slice_any` | — | `fc12d28001458f93` | new |

So the compact fast path did not change: no entry of revisions 1-10 needs a
paired re-run, and the new view entries are additional symbols rather than a
changed signature.

## Resource usage

`scripts/resource-usage.sh` (columns: `registers shared stack local
spill_stores spill_loads local_memory_ops`). The rows of the new entries:

| source | entry | registers | shared | stack | local | spill stores | spill loads | local-memory ops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `diagonal.cu` | `t4a_cuda_tril_view` | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diagonal.cu` | `t4a_cuda_triu_view` | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diagonal.cu` | `t4a_cuda_tril_view_any` | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diagonal.cu` | `t4a_cuda_triu_view_any` | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| `window.cu` | `t4a_cuda_dynamic_update_slice` | 26 | 0 | 0 | 0 | 0 | 0 | 0 |
| `window.cu` | `t4a_cuda_dynamic_update_slice_any` | 41 | 0 | 0 | 0 | 0 | 0 | 0 |

Both by-value view entries and the by-value window write have **zero local
memory**; the `_any` twins do too, which revision 9 permits but does not
require. The complete tables of the two trees were compared row by row: 88 base
rows, 94 head rows, **0 changed rows**, 6 added rows.

## Test evidence

`tests/gpu_tril_dus.rs` (9 cases). Every comparison is over the **whole
destination allocation bytes**: the `logical count + 4` element allocation is
sentinel-filled, the plan is given the padded `out_len`, the launch gets the
logical `len`, and the whole allocation is compared with the CPU model, so NaN
payloads, signed zeros, subnormals, masked zeros, untouched operand bytes and the
tail guard are all asserted. The CPU reference never reads a device result.

- `triangular_views_match_a_cpu_reference_for_every_dtype_and_k` — all seven
  dtypes × eight geometries (compact, transposed, offset, negative stride, zero
  stride, batched rank 3, non-compact rank 4, and non-compact ranks 5 and 8) ×
  `k ∈ {i64::MIN, -1, 0, 1, i64::MAX}` × tril/triu × **both symbols** (by-value
  and `_any`), with the physical launch counted (exactly one per launch) and the
  grid checked against `ceil(count/256)`.
- `triangular_views_reach_rank_nine_through_the_general_symbol` — rank 9 with a
  **non-unit extent on axis 8** (`[2,3,2,2,2,2,2,2,3]`, permuted strides,
  `k = i64::MAX` for tril and `k = -1` for triu), where the planner selects the
  general symbol and the kernel must address the ninth axis.
- `triangular_views_cover_one_and_a_million_elements` — tril at 1 and 1 000 000
  elements; **triu at 37×41 = 1517 (grid 6) and at 1 000 000 (grid 3907)**, so
  both mask bodies are proved to run beyond one block; one physical launch in
  every case, whole-allocation comparison, and an empty input plans no launch.
- `copies_preserve_special_values_bitwise` — explicit fixtures for kept and
  copied positions at every dtype: both zeros, both infinities, the smallest and
  largest subnormal and a NaN with a non-default payload
  (`0x7fc12345` for f32, `0x7ff80123456789ab` for f64; both components for C32 and
  C64), plus integer extremes and Bool bytes. A `k = i64::MAX` tril and a
  `k = i64::MIN` triu copy every element through both symbols and must reproduce
  the image bit for bit; a `k = -1` tril must write exactly zero bytes where it
  masks a `-0.0` or a NaN; a full-size window write copies the same image. Every
  case is also asserted to differ from the operand, so "copied nothing" cannot
  pass.
- `dynamic_update_slice_matches_a_cpu_reference_for_every_dtype_and_index_dtype`
  — all seven dtypes × six update geometries (interior window, negative-stride
  update, window filling an axis, odd extents with a unit axis, length 1, rank 1)
  × all four index dtypes × three start vectors (ordinary, both sides clamped,
  and the route's extremes `i32::MIN/MAX`, `i64::MIN/MAX`, and the largest
  validated floats `-2^24/2^24` (F32) and `-2^53/2^53` (F64)) × **both symbols**.
  The operand and update payloads differ, the destination holds the whole
  operand copy plus the guard, and the comparison is whole-allocation.
- `empty_and_scalar_updates_plan_the_documented_domains` — an empty update plans
  nothing, performs **zero** launches and leaves the operand copy byte-identical;
  a rank-0 update plans `[1,1,1]` and writes exactly one element.
- `dynamic_update_slice_agrees_between_variants_up_to_rank_eight` — ranks 2-8
  with a non-compact, offset update: both symbols against the reference on the
  guarded allocation.
- `dynamic_update_slice_reaches_rank_nine_and_multi_block_windows` — rank 9 with
  a non-unit extent on axis 8: a two-element strided window at offset 5 with the
  origin clamped, and **two multi-block windows (384 updates, grid 2; 19 683
  updates, grid 77)** through `_any`, each compared whole and with
  `plan.grid[0] > 1` asserted.
- `dynamic_update_slice_launches_once_for_one_and_a_million_elements` — a
  1 000 000-element window and a 1-element window on a 1000×1000 operand: one
  physical launch each, grids `[3907,1,1]` and `[1,1,1]`, operand and update
  payloads distinct, whole-allocation comparison, and for the full-size window an
  explicit assertion that the expected image **differs from the operand in the
  last block** and that the last block holds the expected bytes — a kernel that
  wrote nothing, or a prefix, cannot pass.

**Negative controls** (each run and reverted, to show the tests can fail):

| mutation | failing test |
| --- | --- |
| view mask `offset > k + 1` instead of `offset > k` | `triangular_views_match_a_cpu_reference_for_every_dtype_and_k` |
| window clamped against `out.extents[a]` instead of `update.extents[a]` | `dynamic_update_slice_matches_a_cpu_reference_for_every_dtype_and_index_dtype` |
| view body returns for `flat >= 256` (first block only) | `triangular_views_cover_one_and_a_million_elements` |
| window body writes nothing (`return` before the loop) | `dynamic_update_slice_launches_once_for_one_and_a_million_elements` |
| general window body returns for `flat >= 256` (first block only) | `dynamic_update_slice_reaches_rank_nine_and_multi_block_windows` |

CPU tests added to `tests/planning.rs`:

- `revision_eleven_entries_are_in_the_manifest` — the three names, `Entry::ALL`
  membership, the `_any` twins and their pairing, and the `DType::ALL` matrix.
- `triangular_views_read_through_the_input_layout` — compact, transposed, offset,
  negative-stride and zero-stride acceptance, the shorter-output and short-span
  refusals, rank < 2 → `RankOutOfRange`, the empty plan, every dtype, the grid
  for 1 and 1e6 elements, and `General` at rank 9 versus `Fast` at rank 8.
- `dynamic_update_slice_plans_the_update_domain` — every index dtype and every
  data dtype, the Bool index refusal, `NotCompact` for a strided destination, a
  strided/negative-stride update, rank and window mismatches, both span checks,
  the empty and rank-0 plans, rank 9 general selection with the axis-8 extent
  check, and the grid for 1 versus 1e6 elements.

### Findings from the post-implementation review

The independent review found no source-level defect in the kernels, planners, ABI
document or inventory; it found the following test-strength gaps, all closed
here:

| Finding | Change |
| --- | --- |
| The million-element window used identical operand/update bytes, so a kernel writing nothing passed | the update gets its own seed, and the test asserts the expected image differs from the operand in the last block and that the last block holds the expected bytes |
| No multi-block evidence for `triu_view` (max 48 elements) or `dynamic_update_slice_any` (max 16 updates) | triu at 1517 and 1 000 000 elements; `_any` windows of 384 and 19 683 updates, with `grid[0] > 1` asserted |
| The general path saw a single data dtype for each family and one DUS index route | every reference case now runs through both symbols, for all seven data dtypes and all four DUS index dtypes |
| No explicit special-value fixtures | `copies_preserve_special_values_bitwise` with ±0, ±inf, subnormal extremes and non-default NaN payloads per component width, compared bit-exactly |
| Variant, rank-9 and million-element destinations lacked the sentinel slab | every destination is now `logical count + 4` with the logical count as `len`, and every comparison covers the whole allocation |
| Redundant fast-versus-general byte comparisons | removed; both symbols are compared against the same reference bytes instead |

## Notes and deliberate exclusions

- No in-place form: `tril_view`/`triu_view` require `out` disjoint from `in`, and
  `dynamic_update_slice` requires `update`/`starts` disjoint from `out` (they may
  alias each other). Both are documented in the ABI document rather than assumed.
- `starts` stays a compact device vector; a strided index operand is the
  tenferro copy, per the maintainer decision recorded in the design (§5).
- A destination injectivity solver is not added: `out` is compact for the window
  write, which makes the stores injective, and the triangle views write one
  compact output with one thread per element.
- `README.md`'s status line already said "revision 8" before revision 10 and was
  not updated here; the ABI document is the revisioned source of truth.
