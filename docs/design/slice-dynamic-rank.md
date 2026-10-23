# Slice I: dynamic rank (ABI revision 9)

Status: design revision 2 for review, 2026-10-23. Issue:
[t4a-cuda-kernels#13](https://github.com/tensor4all/t4a-cuda-kernels/issues/13) —
the kernels must not narrow tenferro's dynamic rank when they replace its CUDA
backend (tenferro-rs#2208). Maintainer decision of 2026-10-23: extend the
kernels, do not impose a rank ceiling.

Baseline: `main` at ABI revision 8 (`36a5511`). The
[resource provenance](../worklogs/2026-10-23-perf-remainder.md) §2-§3 records the
zero-local-memory property of the current bodies, which this work keeps as a gate.

Revision 2 incorporates the [pre-implementation review of revision 1](https://github.com/tensor4all/t4a-cuda-kernels/issues/13)
in full; §8 lists each finding and where it is answered.

## 1. Scope: every entry with a rank-bearing parameter

Thirty-one symbols gain a general-path twin. The list is derived from the actual
signatures, not from a family name.

| Source | Entries and their rank-bearing parameters |
| --- | --- |
| `movement.cu` | `fill_zero_view`, `materialize_strided`, `contiguous_to_view` (1 layout each), `strided_to_strided` (2) |
| `elementwise.cu` | `combined_strided` (3) |
| `window.cu` | `slice` (2 layouts + `axis_values` starts + `axis_values` steps), `dynamic_slice` (2 layouts), `pad` (2 layouts + `axis_values` edge_low + `axis_values` interior + `axis_values` interior spacing) |
| `diagonal.cu` | `extract_diagonal`, `embed_diagonal_copy` (2 layouts each) |
| `gather_scatter.cu` | `gather` (3 layouts + `gather_plan`), `scatter_add` (3 layouts + `scatter_plan`) |
| `reduce.cu` | `reduce` (2 layouts), `reduce_partials` (2 layouts) |
| `linalg_aux.cu` | `fill_one` (1), `matrix_adjoint` (2), `householder_explicit_v` (2), `identity_shifted` (1), `qr_phase` (2), `qr_apply_phase` (3), `upper_trapezoidal_violation` (2), `householder_assemble` (3), `lu_extract_outputs` (4), `lu_apply_pivots` (2) |
| `rrqr.cu` | `column_norms` (2), `select_pivot` (2), `swap_columns` (2), `reflector` (2), `apply_reflector` (3), `extract_r` (2), `prefix_rank` (1) |

**Deviation from issue #13's scope line, from source evidence.** The issue
excludes "the linalg and RRQR entries (they take explicit matrix/batch extents,
not a runtime rank)". That is not what the sources do: every one of the 17
entries above takes one or more `t4a_cuda_layout` values and walks their axes
above the matrix axes — `matrix_adjoint_body` walks `out_layout.rank`
(`cuda/src/linalg_aux.cu:157`), `shorter_batch_offset` / `matrix_batch_offset`
walk `layout.rank` from axis 1 and 2 (`cuda/src/rrqr.cu:93`, `:108`). A rank-9
batched operand therefore cannot reach them at all. Since revision 9's purpose is
that nothing narrows rank, excluding them would leave exactly the hole the
revision is meant to close, so they are in scope.

Genuinely unaffected, with the reason:

- **Flat entries**: `fill_zero`, `copy_exact`, `add`, `sub`, `mul`, `div`,
  `unary`, `binary`, `compare`, `select`, `clamp`, `convert`, `tril`, `triu`,
  `init_validation`, `validate_float_indices`, `extract_invalid_float_index`,
  `validate_real_cast`, `extract_invalid_real_cast`. Their parameters are `len`,
  element pointers, dtypes, tags and `rows`/`cols`; no rank and no layout. Issue
  #13 lists several of them as rank-walking entries — they are not.
- `tiled_transpose`: rank 2 by contract (`t4a_cuda_transpose_config`).

## 2. ABI revision 9

`T4A_CUDA_ABI_REVISION` becomes 9. **Nothing in revisions 1-8 changes**: the 53
existing symbols, the five existing struct layouts (144/16/72/280/216 bytes) and
their rank rules are untouched. In particular the revision 1-8 symbols keep the
unrolled literal-subscript bodies the resource table measured.

Revision 9 adds four geometry structs whose arrays live in caller-owned device
memory. Field names and the `int` rank/`int` map lengths match their by-value
counterparts, so one body source can name both.

```c
typedef struct {              /* rank 0, reserved 4, offset 8, extents 16, strides 24 */
  int rank;                   /* 0..=INT32_MAX */
  int reserved;               /* must be 0 */
  long long offset;
  const long long* extents;   /* rank values, device memory */
  const long long* strides;   /* rank values, device memory */
} t4a_cuda_layout_any;        /* 32 bytes, align 8 */

typedef struct {              /* rank 0, reserved 4, values 8 */
  int rank;
  int reserved;
  const long long* values;    /* rank values, device memory */
} t4a_cuda_axis_values_any;   /* 16 bytes, align 8 */

typedef struct {              /* ints 0-20, pointers 24, 32, 40, 48 */
  int operand_rank;
  int out_rank;
  int index_vector_dim;
  int window_rank;
  int start_map_len;
  int reserved;
  const long long* window_dims;      /* window_rank */
  const long long* offset_dims;      /* window_rank */
  const long long* start_index_map;  /* start_map_len */
  const long long* slice_sizes;      /* operand_rank */
} t4a_cuda_gather_plan_any;   /* 56 bytes, align 8 */

typedef struct {              /* ints 0-20, pointers 24, 32, 40 */
  int operand_rank;
  int updates_rank;
  int index_vector_dim;
  int window_rank;
  int scatter_map_len;
  int reserved;
  const long long* window_dims;        /* window_rank */
  const long long* update_window_dims; /* window_rank */
  const long long* scatter_dims;       /* scatter_map_len */
} t4a_cuda_scatter_plan_any;  /* 48 bytes, align 8 */
```

and one `_any` twin per in-scope entry (`t4a_cuda_slice_any`,
`t4a_cuda_gather_any`, `t4a_cuda_matrix_adjoint_any`, …). A twin takes exactly the
arguments of its revision 8 counterpart, with each rank-bearing parameter replaced
by the `_any` form; every other argument keeps its revision 8 type.

### Geometry contract

| Array | Length |
| --- | --- |
| `layout_any.extents`, `.strides` | `rank` |
| `axis_values_any.values` | `rank` |
| `gather_any.window_dims`, `.offset_dims` | `window_rank` |
| `gather_any.start_index_map` | `start_map_len` |
| `gather_any.slice_sizes` | `operand_rank` |
| `scatter_any.window_dims`, `.update_window_dims` | `window_rank` |
| `scatter_any.scatter_dims` | `scatter_map_len` |

- Each array is a caller-owned device allocation, `int64` elements, tightly
  packed (no row stride, no padding between elements).
- Every **used** array must be 8-byte aligned, readable for its whole length and
  unchanged until the launch completes. A zero-length array may pass `NULL` or any
  other value; it is never dereferenced and no pointer arithmetic is performed on
  it.
- The consumer uploads the geometry itself. The upload must be ordered before the
  launch **on the launch's stream, or by an event the launch stream waits on**; an
  upload queued on another stream alone is not ordering.
- The buffers must not overlap the operands this launch reads or writes.
- The kernels allocate no scratch, take no stream, synchronize nothing, and read
  the geometry through ordinary global loads. No caching is promised, so one
  geometry buffer may serve several entries with the same shape.
- The host must not rely on the kernel to bounds-check the geometry: it is
  produced by the facade (or the equivalent host arithmetic), and a wrong length
  is caller UB, like a wrong `rank` in revision 8.

### Rank rule

The fast path keeps `0..=T4A_CUDA_MAX_RANK` (8). The general path accepts
`0..=INT32_MAX`; `_any` rank, map lengths and the existing `int axis` arguments are
`int`, so this is the representable domain and the document says so rather than
claiming a `u32` range the fields cannot carry. The facade additionally refuses a
plan whose extent product or address arithmetic leaves `i64`, and one whose `len`
exceeds the launch limits — both already exist. `T4A_CUDA_MAX_RANK` is **not**
raised: any finite ceiling reintroduces the contract gap, and raising it would
change the 144-byte layout and every revision 1-8 signature.

## 3. Device structure

Two rules decide how each general body is written.

1. **The revision 1-8 bodies are not edited.** Their symbols keep calling the
   existing body functions, whose source text and compiled instantiation are
   unchanged, so the resource rows and the paired-performance baseline stay valid.
2. **General bodies never inherit a fast-path trick.** The fast path's tricks are
   the unrolled `axis < T4A_CUDA_MAX_RANK` walk with literal subscripts, the
   masked select `t4a_cuda_axis_value` / `t4a_cuda_axis_add` (which reads and
   writes **all** eight slots), and the `[T4A_CUDA_MAX_RANK]` thread-local
   coordinate arrays (`extract_diagonal`/`embed_diagonal_copy`
   `cuda/src/diagonal.cu:52`, `:92`; `scatter_add`'s `base[]` and `t4a_cuda_axis_add`,
   `cuda/src/gather_scatter.cu:146`, `:200`). A general body keeps no per-axis
   array at all: it recomputes a coordinate per axis and accumulates addresses in
   scalars, so its state is O(1) whatever the rank, and `LDL`/`STL` traffic comes
   only from the compiler's own spilling.

The general helpers are new, additive functions in `t4a/abi.cuh`:
`t4a_cuda_decode_any` (loop to a runtime `rank`), `t4a_cuda_axis_value_any`
(direct `values[axis]`, never a masked eight-slot read), `t4a_cuda_axis_add_any`,
`t4a_cuda_logical_count_any`, `t4a_cuda_compact_coordinate_any` (the coordinate of
one axis, computed on demand). The existing helpers are untouched.

Where a body is already a plain dynamic loop over `rank` with dynamic subscripts
(`matrix_adjoint`, `combined_strided`, `slice`, `dynamic_slice`, `pad`, `reduce`,
`reduce_partials`, the RRQR bodies), the general body is that loop with the `_any`
helpers and the `_any` structs. Where the body is unrolled with literal
subscripts and rank-sized thread-local arrays (`movement`, `diagonal`, `gather`,
`scatter_add`), the general body is a new allocation-free implementation of the
same formula; §5's cross-variant equality test is what proves it computes the
same thing.

Loop bounds in general bodies come from their own operand: an outer walk uses
`layout.rank`, a window-position search uses `window_rank`, a component loop uses
`start_map_len`/`scatter_map_len`. No general loop is bounded by
`T4A_CUDA_MAX_RANK`.

## 4. Facade

- `Layout` stays the `Copy`, 8-array fast-path type; `Layout::new` keeps rejecting
  rank > `MAX_RANK` with `Error::RankOutOfRange`, because it *is* the fast-path
  parameter type. The general path takes caller-owned slices: a new public
  `LayoutRef<'a>` (extents `&'a [i64]`, strides `&'a [i64]`, offset `i64`) with
  `From<&'a Layout>`, `From<(&'a [i64], &'a [i64], i64)>` and `From<&'a WideLayout>`.
  `WideLayout` is a thin owned helper (`Vec<i64>` extents and strides, any rank)
  for callers that do not already hold slices; it is not required to use the
  general path.
- The 31 in-scope `plan_*` functions take `dst: impl Into<LayoutRef<'a>>` with the
  lifetime named in the signature, so every existing `&Layout` call site keeps
  compiling unchanged and no private type appears in a public bound. Helpers
  (`element_count`, `is_compact`, `check_span`, `reachable_offset`, extents
  comparison) move onto `LayoutRef`.
- **Variant selection uses every participating rank and map length, never the
  host storage type**: `RankVariant::select(layouts: &[&LayoutRef], lengths: &[usize])`
  returns `Fast` only when every value is `<= MAX_RANK`. So a rank-8 `WideLayout`
  selects the fast symbol and a gather whose output rank is 8 but whose operand
  rank is 9 selects the general one.
- `Entry` gains one variant per new symbol, so `Entry::ALL` continues to enumerate
  every export (the existing test in `tests/planning.rs:335` enforces that), the
  dtype matrix and the ABI-document cross-check cover them, and a new
  `Entry::general_variant()` / `Entry::fast_variant()` pairs the two symbols.
  `Launch::rank_variant()` reports the selection; because the symbol name is part
  of the consumer's artifact / prepared-plan key, the two variants cannot share a
  cache entry, and the document says the symbol identity is the key component.
- Packing helpers operate on `LayoutRef` and produce the arrays the structs point
  at: `pack_layout -> Vec<i64>` (extents then strides), `pack_axis_values`,
  `pack_gather_plan`, `pack_scatter_plan`, plus `CLayoutAny::new(&LayoutRef,
  extents_ptr, strides_ptr)` and the analogous constructors. This crate still
  never allocates device memory and never uploads. `pack_axis_values` exists only
  where the caller does not already hold the slice (the windowing planners derive
  starts/steps/padding themselves).
- `partials_layout_for` and the public `reduction_partials_layout` (the split
  reduction's second operand) stop returning the rank-limited `Layout` and return
  a `WideLayout`, so a rank-16 split reduction can be planned at all.
- No new error variant: a rank or geometry length that cannot be represented is
  `Error::Overflow` (checked conversions), and `Error::RankOutOfRange` keeps its
  revision 8 meaning for `Layout`.

## 5. Tests

- **Cross-variant equality (GPU, every in-scope entry, every supported dtype).**
  For ranks 1-8 with non-unit extents and non-zero strides, the `_any` symbol and
  its revision 8 counterpart run on identical inputs and must agree bitwise,
  including the sentinel-guarded untouched remainder of every allocation. This is
  the acceptance test for "numerically identical" and it is what guards the newly
  written general bodies against formula drift.
- **Rank 9 and rank 16 (GPU).** Per family (movement, combined strided, slice,
  dynamic slice, pad, diagonal, gather, scatter, both reductions, each linalg-aux
  and RRQR entry), compared against a CPU reference on the complete destination.
  Axes 8 and 15 carry non-unit extents and non-zero strides, the flat domain is
  larger than one block (>256 elements), and the cases include a zero-length map,
  a mixed case (rank-9 operand, rank-8 output) and a rank-16 scalar output. A
  unit-padded low-rank case would not prove anything; these do not use one.
- **Planning (CPU).** Every in-scope planner accepts rank 9 and 16, selects
  `General`, packs geometry in the documented order and length, and keeps
  selecting `Fast` with revision 8's rejections at rank <= 8. The overflow
  boundary (extent product beyond `i64`, rank beyond `INT32_MAX`, `len` beyond the
  launch limits) is refused with the documented error. `Entry::ALL` export/dtype
  tests and the ABI-document cross-check are extended to the `_any` symbols, and a
  test asserts that a variant is chosen from ranks, not from the storage type.
- **Resource usage.** `scripts/resource-usage.sh` rows for every revision 1-8
  entry must match the recorded table (0 stack / 0 `LDL`+`STL` for the entries the
  performance work fixed, 72/20 for `t4a_cuda_unary`). The `_any` rows are
  recorded and labelled as the general path (local memory allowed).
- **Performance.** The revision 8 bodies are not edited, so the claim "the fast
  path did not change" is checked rather than asserted: `cuobjdump -sass` for
  every affected fast entry is compared between `36a5511` and this branch, and
  any entry whose SASS differs is rerun in the paired matrix instead of being
  marked "not rerun". Entries with identical SASS are marked "not rerun, SASS
  identical", and the worklog records which was which.

## 6. Acceptance mapping

| Issue requirement | Where |
| --- | --- |
| fast path keeps the unrolled literal-subscript body and its zero-local-memory property | §3 rule 1 (bodies unedited) + the resource and SASS evidence in §5 |
| general path, any rank, numerically identical, no allocation | §2 rank rule + §3 rule 2 + cross-variant equality and rank-9/16 tests |
| planner accepts arbitrary rank, rejects nothing revision 8 accepted, reports the variant | §4 `LayoutRef`/`select`/`Launch::rank_variant` + planning tests |
| no ceiling on `T4A_CUDA_MAX_RANK` | §2 rank rule |
| ABI revision 9 document, "nothing earlier changed", variant in the specialization record | §2 + `docs/abi/c-kernel-abi.md` |
| paired matrix rerun only where the fast path changed | §5 performance rule (SASS evidence) |

## 7. Rejected alternatives

- **Raise `T4A_CUDA_MAX_RANK`** — rejected by the issue; also changes the 144-byte
  layout and every revision 1-8 signature.
- **Host pointers to extents/strides** — invalid once the launch is queued.
- **A flexible array inside the by-value struct** — same as raising the ceiling.
- **Rank-specialized fast paths above 8** — a second ladder for a case the general
  path serves; the issue asks for one general path.
- **Templating the revision 1-8 bodies over the geometry type** — it would
  recompile the fast path, so "the fast path did not change" could no longer be
  proven without a full paired rerun; separate general bodies keep the evidence
  cheap and exact.
- **A separate `.cu` file for the general path** — the general bodies sit beside
  their fast counterparts in the same file, so a formula change is visible in one
  place.

## 8. Findings from the revision 1 review and where they are answered

| Finding | Answer |
| --- | --- |
| Linalg/RRQR exclusion is false (they walk runtime batch ranks) | §1 includes all 17, with the source evidence and the deviation recorded |
| Specialization record missing; ABI doc says variants are selected inside one symbol | §2/§4: host-selected variant, symbol identity is the key component; the ABI document adds a rank-policy row |
| A key from `Entry` does not distinguish variants; `Entry::ALL` must enumerate exports | §4: one `Entry` variant per symbol plus `fast_variant`/`general_variant` pairing |
| `int` rank cannot carry a `u32` domain | §2 rank rule: `0..=INT32_MAX`, stated as the representable domain instead of claiming `u32` |
| Policy templating does not keep zero local memory | §3/§7: the revision 1-8 bodies are not templated at all |
| Rank-sized state (`[MAX_RANK]` arrays, `axis_add`) cannot be made correct by a picker | §3 rule 2: allocation-free O(1) addressing, no per-axis arrays |
| Gather/scatter MAX_RANK searches, component loops, masked writes | §3: every general loop bounded by its own rank/map length; masked eight-slot helpers prohibited |
| Geometry contract lengths, alignment, ordering | §2 geometry contract table and rules |
| `impl Into<LayoutView<'_>>` does not compile / private bounds | §4: public `LayoutRef<'a>` with named lifetimes and `From` impls |
| Split-reduction partials layout stays rank-limited | §4: `partials_layout_for` / `reduction_partials_layout` return `WideLayout` |
| Packing tied to `WideLayout`; mixed 8/9 selection | §4: packing on `LayoutRef`, selection from ranks and map lengths |
| "No fast-body change / no rerun" unsupported | §5: SASS comparison decides, and changed entries are rerun |
| Rank labels with unit-padded axes prove nothing | §5: non-unit extents/strides on axes 8/15, >256 elements, mixed and degenerate cases |
| Inventory counts inconsistent, wrong window signatures | §1: exact per-symbol inventory |
| Delete list (separate `AnyEntry`, `WideLayout` requirement, `pack_axis_values`, `GeometryTooLarge`, "no duplicate body", older-NVRTC rationale) | §4/§7: one `Entry` identity, borrowed `LayoutRef` as the general-path input, `pack_axis_values` kept only where the caller has no slice, `Overflow` reused, general bodies separate, C++17 assumed |
