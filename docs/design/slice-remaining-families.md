# Slice J: the four remaining kernel families and the inventory proof (ABI revision 10)

Status: design revision 2 for review, 2026-10-23. The pre-implementation review's findings are
answered in §7. Issue:
[t4a-cuda-kernels#14](https://github.com/tensor4all/t4a-cuda-kernels/issues/14) —
tenferro's CUDA library has four families with no ABI entry, and they must be
served before that library is deleted (tenferro-rs#2208 unit U6).

Baseline: `main` at ABI revision 9 (`6237c15`). The revision 9 rule applies here:
every rank-walking entry added now also gets an `_any` twin with pointer geometry.

## 1. The four families, decided from their contracts

| tenferro kernel | decision | reason |
| --- | --- | --- |
| `broadcast_in_dim_kernel` | **new entry** `t4a_cuda_broadcast_in_dim` + `_any` twin | rank-mapped broadcast; `dims` is `comptime` today but tenferro passes it per call, and the parity contract forbids compiling per shape, so the map becomes a bounded by-value array (reusing `t4a_cuda_axis_values`) or a device array in the general form |
| `reverse_kernel` | **no new entry**: a reversed layout through `t4a_cuda_materialize_strided` (reading) or `t4a_cuda_contiguous_to_view` (writing) | a per-axis reversal is `offset' += (extent - 1) * stride` then `stride = -stride` on the masked axes, which the strided-movement entries already implement; the issue allows the fold "if that keeps one entry instead of two" |
| `concatenate_copy_kernel` | **new entry** `t4a_cuda_concatenate_copy` + `_any` twin | one launch per input at a runtime `axis`/`axis_offset`, as tenferro does |
| `scale_in_place_float_kernel`, `scale_in_place_complex_kernel` | **no new entry**: `t4a_cuda_binary` with `op = MUL`, a scalar `rhs` and `out` aliasing `lhs` | `docs/abi/c-kernel-abi.md` already states that "`MUL` may scale in place through `a`"; `combined_strided` cannot express it because it writes a distinct destination. Only F32/F64/C32/C64 are claimed, matching the upstream kernels |

Revision 10 adds **four symbols** — two entries and their revision 9 general-path
twins — and **no new struct**:

| fast symbol | arguments |
| --- | --- |
| `t4a_cuda_broadcast_in_dim` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, t4a_cuda_axis_values dims, long long out_len)` |
| `t4a_cuda_broadcast_in_dim_any` | the same with `t4a_cuda_layout_any` ×2 and `t4a_cuda_axis_values_any` |
| `t4a_cuda_concatenate_copy` | `(void* out, const void* in, int dtype, t4a_cuda_layout out, t4a_cuda_layout in, int axis, long long axis_offset, long long in_len)` |
| `t4a_cuda_concatenate_copy_any` | the same with the `_any` layouts |

The dimension map reuses `t4a_cuda_axis_values` / `t4a_cuda_axis_values_any`: int64
values, revision 9's alignment, lifetime and packing rules, `values[src_axis]` as the
destination axis, `rank = in.rank` with `0` allowed for a scalar input.

`dims[src_axis]` names the output axis that carries the input's `src_axis`, as in
tenferro; an input axis whose extent is 1 is broadcast (its coordinate is 0). The
map is bounded by `T4A_CUDA_MAX_RANK` exactly like the layout PODs, which issue #14
names as the expected answer; the *general* path removes the bound through the
pointer form.

## 2. Semantics (must match tenferro's kernels)

- **broadcast_in_dim** — one thread per destination element, destination read flat
  through its (compact) layout; the destination coordinates are decoded from the
  flat position, the input coordinate on `src_axis` is the destination coordinate
  on `map.dims[src_axis]` (zero when the input extent is 1), and the input address
  is the layout decode of that coordinate tuple. The planner requires: `map.rank ==
  in.rank`, every `dims` entry `< out.rank` and pairwise distinct, each input
  extent either equal to the destination extent on the mapped axis or exactly 1,
  and `out` a destination (no broadcast axis).
- **concatenate_copy** — one thread per source element, source read flat; the
  destination coordinate is the source coordinate with `axis_offset` added on
  `axis`. The planner requires `axis < in.rank == out.rank`, `out.extents()[axis]
  >= axis_offset + in.extents()[axis]` (so overlapping inputs are caller-planned
  and legal, which is what "duplicated offsets" in the acceptance exercises), every
  other axis equal, and a destination without broadcast axes. Several sources are
  several launches on the caller's stream, as tenferro does.
- **scale_in_place** — one thread per logical destination element, `out[i] *=
  factor[0]` for float and complex dtypes and the wrapping product for integers
  (the existing `t4a_cuda_mul`), `out` a destination, `factor_len >= 1`. Bool is
  not eligible. The factor is read once per thread, so a broadcast factor needs no
  special case.

## 3. Revision 10 document

`T4A_CUDA_ABI_REVISION` becomes 10. **Nothing in revisions 1-9 changes**: the 84
existing symbols (53 revision 1-8 + 31 revision 9 twins), their signatures and the
nine existing struct layouts are untouched. The document gains the new type, the
three entry rows with their semantics and planner preconditions, the pair table
rows for the three `_any` twins, the specialization-record row for the by-value map
and the "nothing in earlier revisions changed" statement.

## 4. Inventory proof (acceptance 1)

The issue's clause: a test that enumerates the kernels tenferro still launches and
asserts each one resolves to an ABI entry or a named vendor route, failing on any
unmapped operation.

`docs/kernel-inventory.md` already enumerates the pinned source (237 rows with
`entry` / `macro_entry` / `helper` kinds, generated from `tenferro-rs@14bc1ab9`).
The test:

1. parses that document and collects every `entry` and `macro_entry` row (the
   helpers are not launches);
2. looks each name up in a checked-in mapping table in the test file — every row
   names either an ABI `Entry` (with the operation tag / dtype restriction it maps
   to), or `Vendor(route)` for a library call (`cuBLAS` GEMM, `cuTENSOR`
   contraction, `cuSOLVER` factorization, `cuFFT`), or `Folded(target, reason)` for
   a kernel whose body is one arm of a consolidated entry (for example a scalar
   helper that `t4a_cuda_binary` performs);
3. fails on any inventory row with no mapping, and on any mapping that names a
   symbol this crate does not export;
4. fails when a mapping is `Vendor` but the route is not in the named-route list;
5. asserts the reverse direction: every exported `_any`/fast entry appears in the
   table or in a documented "no baseline kernel — introduced for the ABI" list
   (the validation, linalg-aux, RRQR, fusion and revision 9 twins are such entries).

The table is data, not prose: a reviewer can see one line per tenferro kernel.

## 5. Tests

- Numerical, ranks 1-8 fast path and rank 9/16 general path, sentinel-guarded full
  allocation comparisons, for all three new entries: `broadcast_in_dim` with an
  extent-1 broadcast axis and an identity map, `concatenate_copy` with two
  successive offsets and with duplicated (overlapping) offsets, `scale_in_place` for
  F32/F64/I32/I64/C32/C64 and a strided destination.
- Degenerate shapes: a destination with an empty axis (zero work, `Launch::EMPTY`
  and the variant still reported), extent 1 on every axis, `axis_offset` at the
  destination edge, and a `broadcast_in_dim` whose map is the identity (a pure
  gather) and one that drops an axis by broadcasting it.
- Planning: the by-value map bounds (`map.rank == in.rank`, `dims` in range,
  duplicate `dims` rejected, `rank > T4A_CUDA_MAX_RANK` rejected with
  `RankOutOfRange` for the by-value form and accepted by the `_any` form), the
  overflow boundary of the extent products, and the reversal mapping: a reversed
  layout is accepted by `plan_materialize_strided` and produces the reversed view
  `view_indices` predicts.
- `scripts/resource-usage.sh`: the three new flat entries report 0 stack / 0
  LDL+STL, and their `_any` twins are recorded and labelled.
- The existing gates (`entry_names_are_unique_and_exported_by_the_sources`,
  `entry_dtype_matrix_matches_the_abi_document`, `abi_revision_matches_the_c_header`)
  cover the new symbols automatically once `Entry` and the document are updated.

## 6. Rejected alternatives

- **A `reverse` entry** — it would duplicate the strided copy for one sign flip; the
  issue explicitly allows folding it into the movement entries. The mapping is
  proved by a test instead of a new symbol.
- **Compile-time `dims`/`axis` specializations** — the parity contract forbids
  compiling per shape, and tenferro passes both per call.
- **`scale_in_place` as a `combined_strided` mode** — that entry's contract writes a
  distinct destination; an in-place mode would change its signature and its
  aliasing rules.
- **An unbounded `dims` array** — a by-value parameter cannot be unbounded; the
  pointer form is the general path, exactly as in revision 9.

## 7. Findings from the pre-implementation review and where they are answered

| Finding | Answer |
| --- | --- |
| Integer scaling is a capability expansion | `scale_in_place` is folded into `binary MUL` and only F32/F64/C32/C64 are claimed |
| `rank >= 1` drops scalar broadcasting | `rank = 0` is allowed in both forms and tested (scalar input, zero-length geometry) |
| Reversal formula order is wrong; writing direction unproved | §7 states `offset' += (extent-1)*stride` **then** negate, checked, with empty/singleton handling, and both directions get GPU tests |
| The pinned inventory is a source list, not a launch list | §7's TSV of `#[cube(launch_unchecked)]` launch sites at the pinned commit, with its own pinned counts |
| Row kinds fail open | pinned row count and symbol set, malformed rows rejected, deletion/re-kind mutation tests |
| `int* dims` breaks revision 9's int64 packing | the dim map reuses `t4a_cuda_axis_values`/`_any`; no new POD |
| Identity maps cannot detect an ignored/reversed map | unequal-rank, `[2,0,1]`, extent-1 broadcast and scalar cases |
| Rank labels alone allow unit-padded passes; no >1 block; no fast/any equality | §7's test list: >256 odd domains, non-unit axes 8/15, fast/`_any` equality at ranks 1-8 |
| Movement dtype coverage unspecified | both entries compare every supported dtype bitwise, Bool and nonfinite payloads included |
| No compiled-code evidence for "nothing earlier changed" | `scripts/sass-diff.sh 6237c15` in the gate list |
| The agreement gates are not automatic | §7 extends both registries and adds focused signature/dtype/size assertions |
