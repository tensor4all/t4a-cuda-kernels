# Slice J: layout-taking Tril/Triu and dynamic_update_slice (ABI revision 11)

Status: design revision 2 (pre-review findings incorporated), 2026-10-28. Issue:
[t4a-cuda-kernels#18](https://github.com/tensor4all/t4a-cuda-kernels/issues/18).
Consumer: tenferro-rs#2223 (the CUDA read surface). Maintainer decision of
2026-10-28: add the entries here and bump the tenferro pin, rather than keep a
compact staging copy (tril/triu) and an unsupported error (dynamic_update_slice)
in tenferro.

Baseline: `main` at ABI revision 10 (`5019c71`).

## 1. Scope

| Entry | Change |
| --- | --- |
| `t4a_cuda_tril`, `t4a_cuda_triu` | **unchanged** (revision 2): the compact fast path keeps its symbol, signature and SASS |
| `t4a_cuda_tril_view`, `t4a_cuda_triu_view` | **new**: read the input through its own layout, write a compact output |
| `t4a_cuda_tril_view_any`, `t4a_cuda_triu_view_any` | **new** general-path twins (revision-9 pattern) |
| `t4a_cuda_dynamic_update_slice` | **new** windowed write at device-resident starts |
| `t4a_cuda_dynamic_update_slice_any` | **new** general-path twin |

The flat and view masks are two symbols for one operation in the same way
`copy_exact` and `materialize_strided` are: the flat body is a measured fast path
that a layout decode would slow down, and `with_variant` selects only between
the by-value and the general form by rank (`src/lib.rs` ~802-860), not by
compactness. The caller picks the flat entry for a compact input and the view
entry otherwise. Every revision 1-10 symbol and struct is unchanged, so this is
an additive revision like revision 9.

## 2. ABI revision 11

`T4A_CUDA_ABI_REVISION` becomes 11.

```c
void t4a_cuda_tril_view(void* out, const void* in, int dtype,
                        t4a_cuda_layout in_layout, long long k, long long len);
void t4a_cuda_triu_view(/* same */);
void t4a_cuda_tril_view_any(void* out, const void* in, int dtype,
                            t4a_cuda_layout_any in_layout, long long k, long long len);
void t4a_cuda_triu_view_any(/* same */);

void t4a_cuda_dynamic_update_slice(void* out, const void* update, const void* starts,
                                   int dtype, int index_dtype,
                                   t4a_cuda_layout out_layout, t4a_cuda_layout update_layout,
                                   long long len);
void t4a_cuda_dynamic_update_slice_any(void* out, const void* update, const void* starts,
                                       int dtype, int index_dtype,
                                       t4a_cuda_layout_any out_layout,
                                       t4a_cuda_layout_any update_layout,
                                       long long len);
```

### Tril/Triu view

- Rank >= 2; axes 0 and 1 carry the triangle (`rows = in.extents[0]`,
  `cols = in.extents[1]`), higher axes batch it. The output is compact with the
  input's extents, so it needs no layout argument (its offset is zero and its
  strides follow from the extents). `in` may have any offset and signed strides,
  including zero strides.
- One thread per output element: thread `t` decodes `t` over the extents (column
  major), keeps or zeroes by `col - row` against `k` (overflow-free, as in the
  flat body), and reads `in` through its layout only when it keeps the element.
  `len` = element count; the whole output is written.
- **Overlap:** the byte spans of `in` and `out` must be disjoint (caller
  contract, documented); there is no in-place form.
- Planner `plan_tril_view(dtype, out_len, in, in_len)` / `plan_triu_view`:
  dtype, rank >= 2, `in.check_span(in_len)`, `out_len >= element_count`; `k` is a
  launch scalar.

### dynamic_update_slice

- Semantics (StableHLO / XLA, and tenferro's CPU path): for every axis
  `s[a] = clamp(trunc(starts[a]), 0, out.extents[a] - update.extents[a])` (the
  same clamp `dynamic_slice` applies), then `out[s + i] = update[i]` for every
  update coordinate `i`. Elements of `out` outside the window are untouched:
  **the caller copies the operand into `out` first**, ordered before this launch
  on the same stream.
- `out` is **compact** (offset zero, column-major strides): with distinct
  window coordinates this makes the stores injective by construction, so no
  injectivity proof is asked of the caller. `out_layout` is still passed because
  the kernel needs its extents (and the general twin carries them in device
  memory). `update` may have any offset and signed strides.
- One thread per *update* element (`len = update element count`): the domain is
  the update, never the operand. An empty update launches nothing (the operand
  copy is still the caller's and is still required); a rank-0 update writes one
  element and dereferences no start.
- `starts` is a device vector of `out.rank` values of `index_dtype` in
  {F32, F64, I32, I64}; float starts must already be validated (integral,
  representable) with the existing validation entries, as for `dynamic_slice`.
- **Overlap:** `update` and `starts` must not overlap `out`; they may overlap
  each other (both are read-only).
- Planner `plan_dynamic_update_slice(dtype, index_dtype, out, out_len, update,
  update_len, starts_rank)`: dtype checks as `dynamic_slice`; equal ranks;
  `update.extents[a] <= out.extents[a]`; `starts_rank == out.rank`;
  `require_compact(out)`; both spans checked. Every dtype including Bool.

### General twins

The `_any` twins follow revision 9 exactly: geometry arrays in caller-owned
device memory, the same body source instantiated with the general layout type,
`plan_*` accepting `LayoutRef` for either form and choosing the general symbol
through `GeneralEntry` when rank > `MAX_RANK`. The general path may use local
memory, as revision 9 allows; the by-value view entries must not.

### Inventory

The pinned upstream inventory (173 rows) is historical and stays unchanged. The
reverse inventory gate (`tests/planning.rs` ~3075-3095) lists exports with no
baseline mapping; the new symbols are added there as ABI-introduced, like
`t4a_cuda_div`.

## 3. Performance contract

- The flat `t4a_cuda_tril`/`triu` symbols do not change; their SASS is compared
  against `5019c71` (`cuobjdump -sass`) to show it.
- The by-value view entries and dynamic_update_slice report registers and zero
  local memory in the resource table.
- dynamic_update_slice does work proportional to the update only.

## 4. Tests

- CPU: planner acceptance and refusals (rank, extents, spans, dtypes, compact
  output, starts rank), the inventory gate, header/Rust layout assertions.
- GPU (`--features gpu-tests`, local A100), every comparison over the whole
  output allocation bytes (bit-exact, including NaN payloads and signed zero):
  - tril/triu view on transposed, offset, negative-stride and zero-stride
    inputs; batched rank 3 and 4; `k` in {`i64::MIN`, -1, 0, 1, `i64::MAX`};
    every dtype; equality of the by-value and `_any` symbols at small rank; rank 9
    with a non-unit extent on axis 8; a domain of ~1e6 elements with the physical
    launch count checked against the plan for two very different sizes.
  - dynamic_update_slice: starts of every index dtype (F32, F64, I32, I64),
    including negative, too-large (clamped), integer extremes and the largest
    validated float; strided and negative-stride updates; untouched elements
    around the window; empty update (no launch, operand intact), rank 0, length
    1 and odd sizes; every dtype; by-value vs `_any` equality and rank 9 with a
    non-unit axis 8; a ~1e6-element update with the launch count checked.

## 5. Non-goals

- A strided index (starts) operand: starts stay compact (tenferro decision of
  2026-10-28: non-compact index views keep one copy).
- In-place dynamic_update_slice that skips the operand copy (would need an
  aliasing contract between operand and output; not requested).
