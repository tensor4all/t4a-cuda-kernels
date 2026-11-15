# Implementation plan — U3 slices 2–4: the Metal structural, elementwise-op and reduction families

Issue: tensor4all/tenferro-rs#2402 (umbrella). Unit **U3 slices 2–4** of
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§7 ("t4a-gpu-kernels: the Metal target") and §10 (the U3 row).
Repository: `tensor4all/t4a-cuda-kernels` (the design's `t4a-gpu-kernels`; the rename is
`rocm-backend.md` M1a and has not happened).
Worktree: `/private/tmp/wt-metal-slices-2-4`, branch `feat/metal-structural-slices`, base
`e870458` (U3 slice 1, #35, merged).
Model: `opencode-go/deepseek-v4.1-flash`. Date 2026-11-16.

## 0. What the slice is, and why this set of kernels

The design's U3 is "Metal kernel sources and target in t4a-gpu-kernels". Slice 1
(#35, merged) built the source set, the planner and the *flat elementwise* family. The
slice design `docs/design/slice-metal-elementwise.md` §"Deferred" names the rest of U3:

- **slice 2** — the op-tag elementwise entries (`Binary`, `Compare`, `Select`, `Clamp`)
  with `scalar_side`, and `fill_one`;
- **slice 3** — structural movement (strided materialize, contiguous→view,
  strided→strided, transpose/permutation, broadcast);
- **slice 4** — reductions.

That list is *this repository's* decomposition. What fixes the exact entry set is the
consumer: U4 slice 2/3 in `tenferro-gpu` (`tensor4all/tenferro-rs#2623`, draft). Its
`MetalEngine` (`crates/tenferro-gpu/src/metal/engine.rs`) refuses, with typed errors that
name the unit, exactly these families:

| Refused call in #2623 | Kernel it needs | This slice |
| --- | --- | --- |
| `maximum_read` / `minimum_read` (engine.rs:534, :541) | `Entry::Binary` (`Max`, `Min`) | slice 2 |
| `compare_read` (engine.rs:553) | `Entry::Compare` | slice 2 |
| `select_read` (engine.rs:565) | `Entry::Select` | slice 2 |
| `clamp_read` (engine.rs:576) | `Entry::Clamp` | slice 2 |
| `div_read` of C32 (engine.rs:493) | `Entry::Binary` (`DivComplex`, the scaled form) | slice 2 |
| `cast` f32 ↔ c32 (engine.rs:585) | `Entry::Convert` | slice 3 |
| `to_contiguous_read` of a strided/borrowed view (context.rs:575, engine.rs:83) | `Entry::MaterializeStrided`, `StridedToStrided` | slice 3 |
| a write into a borrowed view (engine.rs:442) | `Entry::ContiguousToView` | slice 3 |
| `broadcast_in_dim_read` (engine.rs:622) | `Entry::BroadcastInDim` | slice 3 |
| `extract_diagonal` / `embed_diagonal` (engine.rs:634, :646) | `Entry::ExtractDiagonal`, `EmbedDiagonalCopy` | slice 3 |
| `tril` / `triu` (engine.rs:656, :660) | `Entry::Tril`, `Triu`, `TrilView`, `TriuView` | slice 3 |
| `reduce_*_read` (engine.rs:664–694) | `Entry::Reduce` (the MSL path where MPS cannot serve) | slice 4 |

Two consequences are recorded deliberately:

- **The design wins over the dispatch text where they disagree.** The task says
  "f32/bf16/c32 only"; the design lists bf16/f16 as non-goals for the first version (§2)
  and its dtype set is f32 + Complex32 (§8), exactly as slice 1 recorded. This slice
  therefore adds **f32 and Complex32 only, no f64, no bf16**. There is no integer or Bool
  family: the design's §8 dtype table gives those to a different unit, and no MSL text
  here has an integer path.
- **Transpose is a stride relabelling, not a kernel.** `transpose_read`/`reshape_read`
  produce views; materialising one is `MaterializeStrided` over the view's own layout.
  A separate `TiledTranspose` kernel would be a second path to the same address map, and
  the CUDA tiled form is a *performance* variant whose MSL structure the design leaves to
  the builder (§11). §5 records the rejected alternative.

## 1. Files, functions and current behaviour

Current state (base `e870458`): `metal.rs` has 7 `SYMBOLS` (fill_zero, copy_exact,
add, sub, mul, div, unary), `source::MODULES` has one module (`elementwise`), the MSL
header is revision 1 with the 14 unary tags, and `tests/metal_planning.rs` +
`tests/metal_kernels.rs` cover that one family.

| File | Now | After |
| --- | --- | --- |
| `src/lib.rs` (4311 lines) | CUDA-only catalogue + shared vocabulary | **unchanged** — not one line. The Metal module already lives beside it and uses `CLayout`, `CAxisValues`, `MAX_RANK`, `Error`, `Entry`, `DType`, `BinaryOp`, `CompareOp`, `ScalarSide`, `ReduceOp`, `ReduceStrategy`, `Split` and the crate-private layout validators |
| `src/metal.rs` (247 lines) | `SYMBOLS` (7), flat planners, `Plan`/`Dispatch`, module registry | `SYMBOLS` grows to **24**; `plan_fill_zero`/`plan_copy_exact`/`plan_unary` unchanged; `plan_elementwise` narrowed to the six uniform flat entries (see §3.4); new planners for every added entry; `layout_mirror`, `layout_bytes`, `axis_values_bytes`; `FLAT_ENTRIES` |
| `metal/include/t4a/abi.msl` (223 lines) | revision 1, dtype + unary tags, c32 helpers, index guard | revision **2**; `t4a_metal_layout`, `t4a_metal_axis_values`, the binary/compare/scalar-side/reduce tags, `t4a_metal_decode`, `t4a_metal_coords`, NaN-safe extrema, the scaled complex division, f32/c32 fold helpers |
| `metal/src/elementwise.msl` (197 lines) | 7 kernels | + `t4a_metal_binary`, `t4a_metal_compare`, `t4a_metal_select`, `t4a_metal_clamp` |
| `metal/src/movement.msl` | absent | new: `t4a_metal_fill_zero_view`, `materialize_strided`, `contiguous_to_view`, `strided_to_strided`, `broadcast_in_dim` |
| `metal/src/diagonal.msl` | absent | new: `t4a_metal_extract_diagonal`, `embed_diagonal_copy`, `tril`, `triu`, `tril_view`, `triu_view` |
| `metal/src/convert.msl` | absent | new: `t4a_metal_convert` (f32 ↔ c32, and only that pair matrix) |
| `metal/src/reduce.msl` | absent | new: `t4a_metal_reduce` |
| `tests/metal_planning.rs` (316 lines) | slice-1 host lane | extended: entry/registry/tag/POD/geometry/rejection cases for every new entry; the two slice-1 expectations that the op-tag entries are "without a kernel" move to their new behaviour |
| `tests/metal_kernels.rs` (969 lines) | slice-1 device lane | extended: every new entry × dtype × layout × length, against CPU references, sentinel tails |
| `docs/abi/c-kernel-abi.md` (1227 lines) | Metal section, revision 1 | Metal section extended to revision 2: POD, parameter blocks, semantics, dtypes, absence list |
| `docs/design/slice-metal-structural.md` | absent | new slice design for slices 2–4 |
| `docs/worklogs/2026-11-16-metal-structural-slices.md` | absent | new: decisions, deferred items, M5 evidence, review record |
| `README.md`, `AGENTS.md`, `Cargo.toml`, `Cargo.lock`, `.github/workflows/ci.yml` | — | **unchanged**: same two commands, no dependency, no lock entry, no feature, no lane. CI already compiles `--features gpu-tests-metal --no-run` |

Behaviour that does not change: every CUDA symbol, planner, tag and `ABI_REVISION == 14`;
`fusion.rs`; the NVRTC `source` module; the slice-1 Metal symbols, their parameter
blocks, their kernels and their plans; `Entry`/`DType`/`Error`/`UnaryOp`/`BinaryOp`/
`CompareOp`/`ReduceOp` semantics; the device lane's local-only policy (R-ci-pr-60).

### 1.1 The entry table after this slice (24 kernels)

`SYMBOLS` is kept in `Entry::ALL` order. Names stay `t4a_metal_<suffix of Entry::name()>`,
which the existing host test enforces mechanically.

| Entry | MSL kernel | Module | Dtypes |
| --- | --- | --- | --- |
| FillZero, CopyExact, Add, Sub, Mul, Div, Unary | as in #35 | elementwise | as in #35 |
| Binary | `t4a_metal_binary` | elementwise | see §2.2 |
| Compare | `t4a_metal_compare` | elementwise | f32 |
| Select | `t4a_metal_select` | elementwise | f32 |
| Clamp | `t4a_metal_clamp` | elementwise | f32 |
| FillZeroView | `t4a_metal_fill_zero_view` | movement | f32, c32 |
| MaterializeStrided | `t4a_metal_materialize_strided` | movement | f32, c32 |
| ContiguousToView | `t4a_metal_contiguous_to_view` | movement | f32, c32 |
| StridedToStrided | `t4a_metal_strided_to_strided` | movement | f32, c32 |
| BroadcastInDim | `t4a_metal_broadcast_in_dim` | movement | f32, c32 |
| ExtractDiagonal, EmbedDiagonalCopy | `t4a_metal_extract_diagonal`, `t4a_metal_embed_diagonal_copy` | diagonal | f32, c32 |
| Tril, Triu, TrilView, TriuView | `t4a_metal_tril`, `triu`, `tril_view`, `triu_view` | diagonal | f32, c32 |
| Convert | `t4a_metal_convert` | convert | (f32, c32) and (c32, f32) only |
| Reduce | `t4a_metal_reduce` | reduce | see §2.4 |

Not added, and recorded where a reader looks (design doc + ABI doc): `FillOne`,
`ReducePartials` (and with it the split/combine policy), `MixedAdd/Sub/Mul/Div`,
`TiledTranspose`, `CombinedStrided`, `ConcatenateCopy`, `Slice`, `DynamicSlice`,
`DynamicUpdateSlice`, `Pad`, `Gather`, `ScatterAdd`, the validation entries,
`MatrixAdjoint` and every linalg entry, the fusion entry, and the dynamic-rank `_any`
twins. Each is deferred for the reason §0/§6 gives, not forgotten.

## 2. MSL ABI revision 2 — the contract

The Metal ABI is its own, not a CUDA revision: `T4A_METAL_ABI_REVISION` moves 1 → 2
because the entry set and the header grow. Every new kernel keeps the slice-1 shape:

- one flat launch over a logical domain, `threadgroups = ceil(len / 256)`,
  `threads_per_threadgroup = 256`;
- the position is a 32-bit `uint` (`[[thread_position_in_grid]]` or
  `[[threadgroup_position_in_grid]]`), the length is a 64-bit `long`, and the kernel
  guards its own tail (`t4a_metal_reaches`);
- parameters bind by index: device buffers first, then `constant T&` scalars/structs;
- no allocation, no synchronization, no barrier except the reduction fold's
  `threadgroup_barrier` (which is threadgroup-local, not a device sync).

### 2.1 New PODs, reusing the crate's existing C mirrors

`crate::CLayout` (144 B: `rank`, `reserved`, `offset`, `extents[8]`, `strides[8]`) and
`crate::CAxisValues` (72 B: `rank`, `reserved`, `values[8]`) are already the CUDA
by-value mirrors, and their sizes/offsets are asserted in `src/lib.rs:4139-4158`. The
MSL header declares the same two structs with `long` for `long long` (MSL `long` is
64-bit on macOS) and `static_assert`s size and alignment. **No new Rust POD type is
introduced**: `metal::layout_mirror(layout) -> Result<CLayout, Error>` zero-fills the
slots beyond `rank` and refuses rank > `MAX_RANK` with `Error::RankOutOfRange`;
`CAxisValues::new` already does the same for the dims vector. `metal::layout_bytes` and
`metal::axis_values_bytes` expose the `&[u8]` a runtime needs for `setBytes`, because the
engine cannot implement the runtime's foreign `Pod` trait for these types.

This keeps one vocabulary and one layout contract: the same struct the CUDA entries take,
now also taken by the Metal entries.

### 2.2 `Entry::Binary` (op-tag family)

`op` and `scalar_side` are runtime tags, values equal to `BinaryOp::tag()` and
`ScalarSide::tag()` (a host test keeps them in sync). Served, by dtype, are exactly the
tags a Metal caller can reach today (checked against the CUDA engine's call sites in
`tenferro-gpu/src/cuda/structural.rs:3834,4457,6492-6496`):

| dtype | tags | notes |
| --- | --- | --- |
| f32 | `Max`, `Min` (`Both`), `Mul` (scalar side), `Pow`, `Rem`, `Div` (scalar side) | `Max`/`Min` propagate NaN by bit test, as the CUDA fold does; `Mul`/`Div` with `Both` stay with `Entry::Mul`/`Div` |
| c32 | `Mul` with a scalar side, `DivComplex` with `Both` | `DivComplex` is the scaled Baudin–Smith form ported from `t4a_cuda_div_complex_parts_at` (`cuda/include/t4a/abi.cuh:969`), with every NaN test rewritten as a bit test |

`MixedAdd`/`MixedSub`/`MixedMul`/`MixedDiv` are **not served**: no caller reaches them
(the CUDA engine itself never emits one; `ElementwiseReadOp` has no mixed scalar variant),
and they would add a second buffer-interpretation rule to one kernel. Everything else
(`Pow`/`Rem` on c32, `Max`/`Min`/`DivComplex` on the wrong dtype, the checked integer
tags, `Both` forms of `Mul`/`Div`) is `UnsupportedOp` before a launch.
`Compare`, `Select` and `Clamp` are f32-only here: the shared `DType` tables
(`Entry::Compare`'s ordered set, `Entry::Clamp`'s float set, `Entry::Select`'s
ordered set — which has no Complex32) intersected with `metal::DTYPES` leave nothing
else, so the planners refuse a Complex32 pair before a launch.

`DivComplex` reads and writes two interleaved components per element (`part = 2 * index`
in 64-bit). Its planner therefore refuses `len > u32::MAX / 2` with `LaunchLimit`, so
both the position and the component index fit the 32-bit address arithmetic the rest of
the set uses; the host lane pins that cliff. That is the only component-length tag in
this set.

The MSL `clamp` is the documented statement order (value < low → low, else value > high
→ high), but each comparison is guarded by a bit-pattern NaN test so fast math cannot
fold the NaN cases away: a NaN input propagates, a NaN bound disables its comparison,
inverted bounds take the lower bound for a value below it and the upper one otherwise.
The CUDA twin relies on IEEE comparisons; the MSL text makes the same outcomes explicit.

### 2.3 Structural entries

Parameter blocks (buffers 0.. then scalars; `dtype`, `op`, `axis`, … are
`constant int&`; lengths, offsets, strides are `long`):

| kernel | buffers | scalars |
| --- | --- | --- |
| `fill_zero_view` | 0 `out` | 1 `dtype`, 2 `dst_layout`, 3 `len` |
| `materialize_strided` | 0 `dst`, 1 `src` | 2 `dtype`, 3 `src_layout`, 4 `len` |
| `contiguous_to_view` | 0 `dst`, 1 `src` | 2 `dtype`, 3 `dst_layout`, 4 `len` |
| `strided_to_strided` | 0 `dst`, 1 `src` | 2 `dtype`, 3 `src_layout`, 4 `dst_layout`, 5 `len` |
| `broadcast_in_dim` | 0 `out`, 1 `in` | 2 `dtype`, 3 `out_layout`, 4 `in_layout`, 5 `dims`, 6 `out_len` |
| `extract_diagonal`, `embed_diagonal_copy` | 0 `out`, 1 `in` | 2 `dtype`, 3 `out_layout`, 4 `in_layout`, 5 `axis_a`, 6 `axis_b`, 7 `len` |
| `tril`, `triu` | 0 `out`, 1 `in` | 2 `dtype`, 3 `k`, 4 `rows`, 5 `cols`, 6 `len` |
| `tril_view`, `triu_view` | 0 `out`, 1 `in` | 2 `dtype`, 3 `in_layout`, 4 `k`, 5 `len` |
| `convert` | 0 `out`, 1 `in` | 2 `dst_dtype`, 3 `src_dtype`, 4 `len` |

(A `row` is buffer indices 0.. in order, then the by-value parameters in the order
listed. The index in the table is the table's scalar order, not a buffer number.)

Semantics are the CUDA entries' (`docs/abi/c-kernel-abi.md` "Entries" onward), with the
MSL differences recorded:

- **Exact copies stay bit-exact.** Every movement entry moves the element's bytes; the
  Complex32 element is copied as two 32-bit words (`copy_exact`'s invariant from #35) or
  as an 8-byte `t4a_metal_c32` at 4-byte alignment, never through arithmetic.
- **The extract-diagonal map is the corrected one** (`cuda/src/diagonal.cu:13-18`): the
  diagonal output axis is `axis_a - (axis_b < axis_a)` and it is skipped while walking
  the remaining coordinates.
- **`tril`/`triu` compare `col - row` against `k`**, so `k = i64::MIN/MAX` is
  well-defined; `tril_view`/`triu_view` read the input through its signed layout and
  write the compact output over the input's extents, writing zero for masked elements
  and touching no input address for them.
- **`convert` writes the imaginary part as exactly +0.0** for f32→c32 and takes the real
  part for c32→f32 (`cuda/src/convert.cu` `convert_to_complex_body` /
  `convert_from_complex_body`).
- **`broadcast_in_dim` reads a source axis of extent 1 at coordinate 0** and maps the
  others through `dims`; the destination must be compact, as the CUDA planner requires.

### 2.4 `Entry::Reduce`

The tags equal `ReduceOp::tag()` and `ReduceStrategy::tag()`. Served dtype/op pairs are
`ReduceOp::supports(dtype)` ∩ `metal::DTYPES`: Sum/Prod on f32 and c32, SumSquares on
f32, Max/Min on f32. U4 slice 3's policy is "MPS first, MSL where MPS cannot serve", so
this MSL entry is the fallback (at least `prod`), not the engine's default path.

Parameter block: buffers 0 `out`, 1 `in`; scalars 2 `dtype`, 3 `op`, 4 `strategy`,
5 `out_layout`, 6 `in_layout`, 7 `axis`, 8 `output_len`.

Two strategies, both with `threads_per_threadgroup = 256` (`T4A_METAL_THREADGROUP`, which
a host test keeps equal to `metal::THREADGROUP`, because the threadgroup fold's static
array and stride are compile-time):

- **Unit** — one thread per output; `threadgroups = ceil(output_len / 256)`; the thread
  walks the axis serially from axis position 0. Same combination order as the CUDA unit
  body, so an f32 unit reduction is bit-identical to CUDA.
- **Plane** — one threadgroup per output; `threadgroups = output_len`; lane `l` seeds
  from its own first element (axis position `l`) and strides by 256, then a threadgroup
  tree fold (halving strides through `threadgroup` memory with a barrier per level)
  combines the lanes and lane 0 writes. Only Unit seeds from axis position 0.

`ReducePartials` and the split/combine policy are **not** in this slice: no caller exists
until the engine adopts a split policy (its U4 slice 3 goes to MPS first). The gap is
recorded in the design and ABI docs; adding pass 1 later is a second function over the
same fold helpers, not a change to this entry.

Vendor-specific geometry, recorded: CUDA's plane width is 8/16/32 and its split chunks
must be at least one plane; MSL's plane width is 256 (the flat threadgroup), so
`Unit` is legal only for `reduce_len <= 256` and `Plane` only for `reduce_len >= 256`.
The fold order inside a plane differs from CUDA's xor tree (different width, different
tree), so plane f32 results are compared by tolerance, never bit-for-bit; max/min are
exact except for NaN payload choice, and the device lane checks classification, not
payload, for those.

## 3. The Rust planning target

### 3.1 Validation is shared, geometry is not

Every planner reuses the crate's own validators rather than restating the contract:
`check_dtype`/`Entry::supports`, `LayoutRef::element_count`, `check_span`,
`require_compact`, `require_destination`, `require_axis`, `extract_diagonal_extents`,
`embed_diagonal_extents`, `in_layout.extents() != dst.extents()` (`CountMismatch`),
`ReduceOp::supports`, and the overflow checks the CUDA planners use.
The *geometry* is written per vendor: the CUDA `flat_launch` uses its own grid limits and
has no 32-bit position bound; the Metal `plan_len` refuses `len > u32::MAX` with
`Error::LaunchLimit` and returns `Dispatch { threadgroups: ceil(len/256), threads: 256 }`.

### 3.2 `Plan.bytes`

`Plan.bytes` is **the byte size of the destination allocation the plan writes**, which is
what the U4 engine uses to allocate the result (`engine.rs:199,269,346`). It is
`destination allocation * destination element size` in every planner — the allocation the
caller already validated with `check_span`, not the logical element count, because a
strided destination with an offset reaches past its own element count. The rows that are
not the obvious `positions * dtype.size()` are pinned by the host lane:

| planner | `bytes` |
| --- | --- |
| `plan_elementwise`, `plan_unary`, `plan_convert`, `plan_materialize_strided`, `plan_tril`, `plan_tril_view` | `len * dtype.size()` (the destination is compact with this dtype) |
| `plan_fill_zero_view`, `plan_contiguous_to_view`, `plan_strided_to_strided` | `dst_len * dtype.size()` — the caller's destination allocation, which a strided view with an offset can exceed in element count (`plan_into`) |
| `plan_broadcast_in_dim` | `out.element_count() * dtype.size()` (the destination is compact) |
| `plan_extract_diagonal` | `out.element_count() * dtype.size()` (the diagonal output is compact) |
| `plan_embed_diagonal_copy` | `out_len * dtype.size()` — the destination allocation, which the inserted axis makes larger than the launch's `in`-sized position domain, so this is never `positions * size` |
| `plan_binary` | f32/c32 flat tags: `len * dtype.size()`; `DivComplex`: `len * 8` |
| `plan_compare` | `len` bytes (the destination is Bool, one byte per position) |
| `plan_reduce` | `out_len * dtype.size()` (the keepdims output's allocation) |

### 3.3 32-bit position invariants

- Every flat planner refuses a negative count (`NegativeCount`) and a length above
  `u32::MAX` (`LaunchLimit`), and `bytes` is computed in `u64`.
- `plan_binary` refuses `len > u32::MAX / 2` for `DivComplex`, whose component index is
  `2 * position`.
- `plan_reduce` refuses `output_len > u32::MAX` (threadgroups) and an axis length above
  `u32::MAX` (the layout's element count).
- Layout spans are checked against the caller's allocation length in *elements* with the
  crate's `check_span`, so a negative stride/offset is inside the allocation before any
  kernel sees it.
- `layout_mirror` refuses rank > 8 (`RankOutOfRange`): the MSL kernels carry the by-value
  layout, and the device-resident `_any` geometry is not ported (a caller-filled device
  buffer would be a second path, and no U4 caller exists).

### 3.4 `plan_elementwise` narrowing (public API tightening)

Slice 1's `plan_elementwise(entry, dtype, len)` accepted any served `is_elementwise`
entry. That is now unsafe to keep: several of those entries need scalars (`op`,
`scalar_side`, `src_dtype`) or have a destination whose element size differs from
`dtype` (compare, convert, complex division), so the returned `bytes` would mis-size the
allocation. It becomes:

```rust
pub const FLAT_ENTRIES: [Entry; 6] = [FillZero, CopyExact, Add, Sub, Mul, Div];
pub fn plan_elementwise(entry, dtype, len) -> Result<Plan, Error> // only FLAT_ENTRIES
```

Anything else is `Error::UnsupportedOp` (the entry has no uniform flat parameter block;
use its own planner), and `!entry.is_elementwise()` stays `Error::NotElementwise`.
This is source-compatible with #2623: `MetalEngine::binary` passes exactly Add/Sub/Mul/
Div and the copy/fill paths pass FillZero/CopyExact (`engine.rs:188` is the only call
site). `plan_unary` is unchanged.

## 4. Tests, commands and bars

### 4.1 Host lane (no device) — `tests/metal_planning.rs`

Added/extended cases:

1. **Registry.** Every `SYMBOLS` entry is declared by exactly one module and every
   declared `kernel void` is a served symbol (the existing test; it extends by itself to
   24), `module_of` resolves, `compose` still prepends the header once, and the MSL text
   still contains no `double`/`float64`.
2. **Tags and PODs.** `T4A_METAL_ABI_REVISION == 2`; `T4A_METAL_THREADGROUP ==
   metal::THREADGROUP == 256`; binary/compare/scalar-side/reduce defines equal the Rust
   `tag()` values for every served variant; `T4A_METAL_MAX_RANK == 8`; the MSL
   `t4a_metal_layout`/`t4a_metal_axis_values` `static_assert`s match
   `size_of::<CLayout>() == 144`, `size_of::<CAxisValues>() == 72`, offsets 0/4/8/16/80
   and 0/4/8.
3. **`layout_mirror`.** Zero-fills slots past `rank`, copies extents/strides/offset,
   refuses rank 9 with `RankOutOfRange`, `layout_bytes(&mirror).len() == 144`, and reads
   the same bytes for two equal mirrors. `axis_values_bytes` likewise (72).
4. **Geometry.** For every flat planner over `{0, 1, 255, 256, 257, 4096, 100_003}`:
   `threadgroups == ceil(len/256)`, `threads == 256`, `is_empty() == (len == 0)`,
   `threadgroups * 256 >= len`, and the §3.2 `bytes`. For reduce: unit over an axis
   ≤ 256 (including the 256 boundary), plane over an axis ≥ 256 (including the
   boundary); `output_len == 0` is `Dispatch::EMPTY`; `plan_embed_diagonal_copy`'s
   `bytes` is the *out* layout's size, not the position count (the BLOCKER the plan
   review found).
5. **32-bit cliffs.** Three representative families — flat (`plan_materialize_strided`),
   structural-with-a-layout (`plan_broadcast_in_dim`) and reduce (`plan_reduce`) —
   take `len = u32::MAX` and refuse `u32::MAX + 1` / `i64::MAX` with `LaunchLimit`;
   `DivComplex` takes `len = u32::MAX / 2` and refuses `u32::MAX / 2 + 1`; negative
   lengths/offsets are `NegativeCount`.
6. **Rejections, per entry** (the shared tables intersected with the Metal set): wrong
   dtype (f64, c64, i32, i64, bool for every new entry; c32 for compare/clamp/
   sum_squares/max/min; f64 pairs for convert); same-dtype convert (`ShapeMismatch`);
   `Pow`/`Rem` on c32, `Max`/`Min`/`DivComplex` on the wrong dtype, every `Mixed*` tag,
   the checked integer tags and `Both` `Mul`/`Div` (`UnsupportedOp`); non-compact where
   the entry requires compact (broadcast destination, embed input, tril/triu operands);
   a stride-0 destination with an extent > 1 (`BroadcastDestination` — this is why
   `fill_zero_view` is never tested with a broadcast destination); axis out of range,
   equal diagonal axes, wrong diagonal output extents, mismatched strided-to-strided
   extents, rank > 8, `in` outside its allocation, `k` accepted at both i64 extremes,
   tril length not a multiple of `rows*cols`; reduce strategy mismatches (unit above 256,
   plane below 256), non-keepdims output, output extent ≠ 1 on the axis.
7. **The narrowed `plan_elementwise`.** The six `FLAT_ENTRIES` plan; `Unary`, `Binary`,
   `Compare`, `Select`, `Clamp`, `Convert`, `FillZeroView` return `UnsupportedOp` there
   and plan through their own planners; `Entry::Reduce` stays `NotElementwise`.
8. **Text guards.** The composed `reduce` module contains the threadgroup fold and the
   `long` length guard and uses `T4A_METAL_THREADGROUP`; the movement module decodes
   through `t4a_metal_layout`; no module contains `atomic`, `simd_shuffle` or an MPS
   class name (the fold is a plain tree; MPS belongs to the engine).

### 4.2 Device lane (Apple M5 Max) — `tests/metal_kernels.rs`

The harness (from #35) compiles every module, binds every symbol, dispatches once per
entry call, waits, and asserts the launch counter. Added cases:

- **Movement** (f32 and c32): `fill_zero_view` on offset and negative-stride
  destinations; `materialize_strided` from offset/negative-stride/transposed/stride-0
  sources compares bit-exactly with a CPU gather; `contiguous_to_view` writes only the
  layout's addresses (every other byte of the sentinel-filled destination is untouched);
  `strided_to_strided` between two different layouts; `broadcast_in_dim` with a dims map,
  a scalar input (rank 0), and extent-1 axes.
- **Diagonal** (both dtypes): extract/embed on rank 2 and rank 3 layouts, including the
  rank-3 case the baseline defect needed; `embed` over a pre-zeroed destination writes
  only diagonal positions; `tril`/`triu`/`tril_view`/`triu_view` at `k = -1, 0, 1` and
  `k = i64::MIN/MAX`, compact and transposed/strided inputs, batched higher axes.
- **Convert**: f32→c32 (real part exact, imaginary `+0.0` bits), c32→f32 (real part
  exact), 100_003 elements, sentinel tails.
- **Binary/compare/select/clamp**: max/min including NaN and ±0 outcomes by bit pattern;
  `Pow`/`Rem`/`Div` against host `powf`/`fmodf`/correctly-rounded division at 8 ulp
  (div bit-exact); c32 scalar `Mul`; `DivComplex` against a host reference of the same
  scaled form over extreme magnitudes (1e-30, 1e30, zero, inf, NaN); compare
  EQ/LT/LE/GT/GE with NaN → byte 0; select (f32 only); clamp NaN/inverted-bound cases.
- **Reduce**: Sum/Prod/SumSquares/Max/Min on f32 and Sum/Prod on c32, where the op
  supports the dtype; **unit at axis lengths {1, 2, 255, 256} and plane only at
  {256, 257, 1024, 100_003}**; output_len 1 and 4, non-contiguous output layout; unit
  results bit-identical to a CPU serial reference; plane results within 4 ulp (relative
  for large sums) of a CPU reference; max/min exact with NaN classification; `prod`
  containing a zero and an infinity; each lane seeds its own first element (the plane
  walk) and Unit seeds element 0 (a length-1 axis returns the element, `-0.0` keeps its
  sign for Sum).
- **Launch accounting**: one dispatch per entry call (the existing assertion), and a
  `100_003`-element case for every flat family so more than one threadgroup must run.

Bars: every added case must pass; the device lane is *fail-loud* (no device = failure).

### 4.3 Exact commands

```bash
cd /private/tmp/wt-metal-slices-2-4
cargo fmt --check
cargo clippy -j 4 --locked --all-targets -- -D warnings
cargo test -j 4 --locked                                  # host lane, includes metal_planning
cargo test -j 4 --locked --features gpu-tests-metal --no-run
cargo test -j 4 --locked --features gpu-tests-metal --test metal_kernels
```

Acceptance: all green; the host lane covers the tables above; the device lane reports
every one of the 24 symbols bound and passing; the evidence records the GPU model,
macOS/driver versions, the exact commands and the observed counts. No benchmark is added
and no performance bar is claimed (correctness/infrastructure slice; the reducer's
threadgroup tree is documented as untuned, and U4 slice 3 uses MPS for sum/max/min).
`metal::precise::fmod`, `precise::pow`, `precise::fma`, `precise::fmax/min` and
`precise::divide` were checked against the installed runtime's
`metal_math` header (`.../GPUCompiler.framework/.../include/metal/metal_math:1547-3071`).

## 5. Edge cases, compatibility and deliberate non-changes

- **Empty domains.** `len == 0` returns `Dispatch::EMPTY` in every planner (including
  reduce with `output_len == 0`); the caller skips the launch. Rank 0 (scalar) layouts
  have `element_count == 1` and work through the same path.
- **Rank > 8.** `RankOutOfRange` from `layout_mirror` and from the planners that take a
  layout. Not a silent truncation.
- **Signed strides and offsets.** Accepted anywhere the CUDA entry accepts them; the
  span check proves the minimum/maximum reachable address is inside the allocation.
  Stride 0 is allowed for reading operands and rejected for destinations
  (`BroadcastDestination`), exactly as the shared validator decides.
- **Empty/odd shapes.** `rows == 0 || cols == 0` with `len == 0` plans empty; with
  `len != 0` it is `ShapeMismatch` (CUDA's `plan_triangular`).
- **NaN, inf, signed zero, subnormals.** Copies/zero/one/convert are bit-exact;
  max/min/fold use bit-pattern NaN classification; clamp's documented outcomes are made
  explicit; the slice-1 statement that Metal flushes subnormals under its default compile
  options is unchanged and still asserted.
- **One vocabulary.** No `metal::Entry`, no Metal tag space of its own, no second layout
  type: `Entry`, `DType`, `BinaryOp`, `CompareOp`, `ScalarSide`, `ReduceOp`,
  `ReduceStrategy`, `CLayout`, `CAxisValues`, `Error` are the shared ones.
- **Deviations from the slice-1 "Deferred" note** (recorded in the new design doc, per
  the plan review): `ConcatenateCopy` is not served (its address map is
  `ContiguousToView` with a shifted layout, and no trait method needs it); the split and
  combine policy stays engine-side (no caller; MPS-first); `FillOne` and the `Mixed*`
  tags are deferred until an op reaches them. Everything else in that note is here.
- **Deliberately not changed**: `src/lib.rs`, `cuda/**`, `fusion.rs`, the CUDA ABI
  revision, the crate version, the dependency graph, `Cargo.lock`, the CI lanes, the
  slice-1 entries and parameter blocks, the local-GPU-only policy.
- **Deliberately absent** (each recorded in the design/ABI docs): the entries above plus
  tiled transpose, combined-strided, the indexing/window/gather/scatter family, the
  validation entries, linalg, fusion, MPS GEMM/FFT, the `_any` dynamic-rank twin, bf16
  and the integer/Bool families.

## 6. Risks and the rejected alternative

| Risk | Mitigation |
| --- | --- |
| MSL is text: a compile error in `metal/src/*.msl` is *not* caught by hosted CI (no macOS/Metal lane in this repo) | The device lane compiles every module and binds every symbol before any test runs; a compile failure fails the lane loudly. Evidence records the local M5 Max run. |
| Fast math folds comparisons and NaN tests | Every NaN/infinity test is a bit pattern; clamp guards its comparisons; max/min/fold use the bit-tested form; the device lane pins the outcomes. |
| 32-bit position/size invariants | Planners refuse > u32::MAX at the planning boundary; kernels read `uint` positions and compare against `long` lengths; `DivComplex` additionally caps at `u32::MAX / 2`; the host lane exercises the cliff for three families. |
| Plane fold order differs from CUDA | Documented; device bars are tolerance-based for f32 folds and exact for max/min classification and for the unit strategy. |
| A 24-kernel catalogue is a large diff to review | Four modules by CUDA family, one planner group per family, one test group per family; the PR body lists the slices and their entry tables; the ABI document is the single reference for parameter blocks. |
| `ReducePartials` has no caller | Dropped (plan review DELETE); the gap and its two-function shape are recorded in the design and ABI docs so the later split policy does not re-derive the fold. |
| `plan_elementwise` narrowing could break another consumer | The only consumer (#2623) passes exactly the six `FLAT_ENTRIES` (verified in `engine.rs:188` call sites); the change is documented in the ABI doc and the crate is pre-publication. |

**Rejected alternative**: port the CUDA performance machinery too — the shared-memory
tiled transpose with its runtime `TransposeConfig` switch, `CombinedStrided`'s three-layout
decode, the split-by-default `plan_reduction` Auto policy, and the level-14 multi-warp
pass-1 blocks. Rejected because no Metal caller needs those variants (transpose is a view
plus `MaterializeStrided`; the elementwise family stages non-compact operands), the
design leaves MSL kernel structure and threadgroup sizing to the builder (§11), and the
diff would double without a correctness gain. It is recorded as a performance follow-up,
not a gap.

Also rejected: a `metal::Layout` of its own (the crate already has `CLayout`, and a second
POD is a second contract); a device-resident `_any` general path (needs a caller-filled
device buffer, a second path for a rank no U4 caller has); bf16/f16 (design non-goal);
MPS calls in this crate (the design puts GEMM/FFT in the engine).

## 7. Review record

**Pre-implementation plan review — `ollama-cloud/glm-5.3-flash` (GLM-5.3-flash, a
different model family), read-only `pi --print` process, `scripts/pi_review.sh`, request
`/tmp/plan-review-u3-slices-2-4.md`, report `/tmp/plan-review-u3/plan-glm.out`. Verdict:
`VERDICT: CHANGES`.** Findings and disposition (all addressed in this revision):

| Severity | Finding | Disposition |
| --- | --- | --- |
| BLOCKER | `embed_diagonal_copy`'s `Plan.bytes` was the position domain (`in` count), under-allocating the larger output (the inserted axis) | §3.2 now defines `bytes` as the destination allocation and names embed as the non-obvious row; §4.1.4 pins it |
| BLOCKER | `DivComplex` addresses `2 * len` components while only `len <= u32::MAX` was checked | §2.2/§3.3: `plan_binary` refuses `len > u32::MAX / 2` for `DivComplex`; §4.1.5 pins the cliff |
| IMPORTANT | The reduce parameter block was missing from the table | §2.4 lists buffers/scalars 0–8 |
| IMPORTANT | "component-wise" mis-described the mixed ops; CUDA keeps zero cross terms | `Mixed*` is dropped entirely (no caller), which removes the ambiguity |
| IMPORTANT | `fill_zero_view` cannot have a stride-0 destination (`require_destination`) | §4.1.6/§4.2 now test stride-0 as a *source* and negative-stride/offset destinations |
| MINOR | The new design doc must name each deviation from the slice-1 Deferred note | §5 does; the design doc will |
| MINOR | "every fold seeds from the first element" is false for the plane walk | §2.4: Unit seeds element 0, each plane lane seeds its own first element |
| MINOR | Plane test lengths below 256 contradict the strategy rule; MSL availability unverified | §4.2 restricts plane to ≥ 256; the `metal_math` `precise` names were verified on this host and recorded in §4.3 |
| DELETE | `ReducePartials`, `FillOne`, `Mixed*`, two of the four cliff families | All four applied (24 kernels instead of 26) |
| NOT CHECKED | MSL text does not exist yet; publish chain missing from the plan | The device lane is the MSL check; §8 records the chain |

**Post-implementation diff review — `openai/gpt-6.1-sol` (repository gate, independent
read-only `pi --print` process), verdict `VERDICT: CHANGES`, one BLOCKER + eight
IMPORTANT + seven MINOR.** Every finding was checked against the source before
anything changed; all are addressed, none was dismissed:

| Severity | Finding | Disposition |
| --- | --- | --- |
| BLOCKER | A view destination planned `bytes` from `element_count()`, so `[2]` with stride 2 and offset 1 planned 8 bytes and wrote element 3 (`ContiguousToView`, `StridedToStrided`, `EmbedDiagonalCopy`, `Reduce`) | `plan_into` plans `bytes(dtype, dst_len)`; §3.2 rewritten; host lane pins three view planners, embed and reduce |
| IMPORTANT | `plan_reduce` capped only the output length, so a `Plane` axis of `i64::MAX` overflows the MSL step | axis capped at `u32::MAX` (`LaunchLimit`); host lane pins both sides |
| IMPORTANT | The planners admitted rank 9, which `layout_mirror` refuses later | `require_mirror_rank` first in every layout-taking planner, embed's enlarged output included; host lane pins ten refusals |
| IMPORTANT | The Complex32 reduce reference read the layout row-major and compared only the real part; the fixture's zero let a zero product pass | reference follows `row + 2 * column`, both components compared, product fixture is unit-circle (result exactly `i`) |
| IMPORTANT | The f32 `Plane` product fixture contained a zero | product case uses alternating ±1 (exact ±1, never zero) |
| IMPORTANT | No structural entry crossed a threadgroup | 600-position case over all five movement entries, broadcast, extraction, both triangle pairs, `compare`/`select`/`clamp`, both dtypes where admitted |
| IMPORTANT | Sentinel slack was allocated but not compared for several entries | every destination comparison covers the whole allocation |
| IMPORTANT | Evidence lacked GPU/OS/toolchain identity | `machine.log` added; the design's claim narrowed to what the lane asserts |
| IMPORTANT | The reduction module dropped its CUDA twin's CubeK notice | notice restored in `reduce.msl`, `THIRD_PARTY_NOTICES.md` scope widened |
| MINOR ×3 | "IEEE remainder"; "every division uses `precise::*`" (complex division used bare `/`); "`Unit` bit-identical" without the flush qualifier | wording corrected, the four complex-division sites now use `precise::divide`, the qualifier added |
| MINOR ×4 | Design doc: a helper signature that does not exist, `FillZeroView` counted as `is_elementwise`, `i32` where the tests use `i64`, "four select patterns" | all four corrected against the code |
| MINOR | The plan's `Select` row claimed c32 | f32 only (the shared table has no Complex32 select) |
| MINOR | README still advertised Metal revision 1 | revision 2, 24 entries, slice-design link |
| DELETE ×2 | Duplicate-axis `seen` vector; the two identity/forwarding Complex32 reduce wrappers | both deleted |

**Verification pass on those fixes — same reviewer and model, verdict `VERDICT: CHANGES`,
no BLOCKER, 2 IMPORTANT + 5 MINOR.** Both importants were gaps in the fix's own test
coverage, now closed: the large-domain case covers Complex32 triangles, Complex32
`materialize_strided` and a 300x300 embedded diagonal, and every reduction case plus the
short `select`/`clamp` cases and the Complex32 → f32 `convert` assert their sentinel tail.
The minors were stale text (this plan's `bytes` table still called views compact and still
said `Select` serves both dtypes; the slice design still said "IEEE remainder" and
`i32::MIN/MAX`) plus three unused fixture bindings — all fixed.

## 8. The cross-repo chain before tenferro-rs#2623 can leave draft

`#2623` pins `t4a-cuda-kernels = "=0.2.0"` (git rev) and `t4a-metal-runtime` (git rev),
and its pin gate fails with "no crates.io version of t4a-cuda-kernels satisfies
'=0.2.0'" because nothing beyond 0.1.2 is published. The order is bottom-up:

1. `t4a-metal-sys` 0.1.0 — publish (unpublished; the runtime's dependency).
2. `t4a-metal-runtime` 0.1.1 — publish (0.1.0 + `Send`, needed by the engine's
   `Mutex<Runtime>`; t4a-gpu-backend#17).
3. **`t4a-cuda-kernels`** — release this tree as **0.3.0** (0.2.0 is unpublished; this
   slice adds public API and moves the Metal ABI revision 1 → 2, so a minor bump, not a
   patch), tag it, and publish it.
4. tenferro-rs: switch both pins from git revisions to registry versions
   (`t4a-metal-runtime 0.1.1`, `t4a-cuda-kernels 0.3.0`), drop the two
   `git-pin-content-exceptions.toml` entries, and rerun the pin/publish gates.
5. Only then does `#2623` leave draft; its Metal engine keeps working across 0.2.0 → 0.3.0
   because this slice does not change any slice-1 entry, its parameter block, or
   `plan_fill_zero`/`plan_copy_exact`/`plan_unary`.

`t4a-metal-sys`/`t4a-metal-runtime` are outside this repository; this PR does not publish
anything (R-ci-pr-60/59: publishing is a separate, approved step).
