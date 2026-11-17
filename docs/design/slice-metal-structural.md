# Slice: the Metal structural entries

**Status:** implemented. **Unit:** U3, slices 2–4, of
tensor4all/tenferro-rs#2402, following
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§7 ("the Metal target") and §10 (the U3 row). Slice 1
([`slice-metal-elementwise.md`](slice-metal-elementwise.md)) built the MSL header,
the source registry, the planning target and the seven flat elementwise entries; this
slice adds the rest of the catalogue the Metal entries can serve without a second
address-map vocabulary: the op-tag elementwise family, movement, diagonals,
triangles, conversion and reductions.

## Why these families, in one slice

The dispatch for U3 slices 2, 3 and 4 lists movement, diagonal, convert, the
element-wise op family and reductions. They share one thing — a *by-value layout
mirror* (`t4a_metal_layout`) and a *dimension map* (`t4a_metal_axis_values`) — and that
mirror is the only genuinely new ABI surface here. Splitting the three families into
three pull requests would have meant three revisions of the same header and three
reviews of the same mirror, so they are one change with one revision bump
(`T4A_METAL_ABI_REVISION 1 → 2`).

Not in this slice, by design and recorded rather than left as a gap:

- **`FillOne`, `ReducePartials`, `MixedAdd/Sub/Mul/Div`.** The draft plan carried all
  six. The plan review deleted them and the deletion is kept: no Metal caller reaches
  `FillOne` (an auxiliary of the linalg family), a split reduction has no caller until
  the engine adopts a split policy (its U4 slice 3 uses MPS where it can), and the
  CUDA engine never emits a mixed real/complex pair (`src/engine.rs`,
  `ElementwiseReadOp` is Add/Subtract/Multiply/Negate/Conj/Divide), so
  `t4a_metal_binary` would have carried a second buffer-interpretation rule for no
  caller.
- **`TiledTranspose`, `CombinedStrided`, `ConcatenateCopy`.** Transpose is a stride
  relabelling into `materialize_strided`; the elementwise family stages a non-compact
  operand with `MaterializeStrided`/`ContiguousToView` first; `ConcatenateCopy`'s
  address map is `ContiguousToView` with a shifted layout. The CUDA tiled forms are
  performance variants whose MSL structure and threadgroup sizing design §11 leaves to
  the builder.
- **The indexing family, validation entries, linalg, fusion, `_any` twins.** Later
  units, or entries whose parameter block needs geometry this revision does not carry.
- **Complex32 division as `Entry::Div`.** It is `BinaryOp::DivComplex`, the scaled
  Baudin-Smith form the CUDA arithmetic module uses, so one division implementation
  exists per backend rather than two (`Div` stays f32-only on both).

The f32 + Complex32 dtype set, the MPS boundary for GEMM/FFT, and the absence of
bf16 are slice 1's decisions and are unchanged (design §2, §8; maintainer decision of
2026-11-14 on #2402).

## The contract this slice implements

### The layout mirror

`t4a_metal_layout` is 144 bytes: `rank` (`int`), `offset` (`long`), and eight
extents and eight strides (`MAX_RANK`, `src/lib.rs:137`). Field order is the Rust
`CLayout`'s, little-endian, and both `LAYOUT_BYTES = 144` and `AXIS_VALUES_BYTES = 72`
are pinned by host tests against the struct `size_of` — the crate is
`#![forbid(unsafe_code)]`, so encoding is a manual little-endian byte walk rather than
a `transmute`, and the pin is what keeps that walk honest.

The MSL side mirrors the same two structs with `static_assert` on both sizes and reads
them through two helpers: `t4a_metal_decode` turns a flat position into the address
`offset + sum(coordinate * stride)` in 64 bits, axis 0 fastest and each coordinate
taken modulo its extent, and `t4a_metal_reaches(index, len)` is the tail guard. A signed
stride and a stride-zero source axis are therefore one code path, not two. An offset is
carried as given: the planners refuse a layout whose reachable span leaves the caller's
allocation (`check_span`), and a negative offset is refused there rather than by the
kernel.

`rank > 8` is refused by the planner with `RankOutOfRange` before encoding, so the
mirror never silently truncates: `layout_mirror` returns `Result`.

### The entries

Twenty-four entries, `metal::SYMBOLS`: the seven slice-1 flat entries, the four
op-tag elementwise entries (`Binary`, `Compare`, `Select`, `Clamp`), five movement
entries (`FillZeroView`, `MaterializeStrided`, `ContiguousToView`,
`StridedToStrided`, `BroadcastInDim`), six diagonal and triangular entries
(`ExtractDiagonal`, `EmbedDiagonalCopy`, `Tril`, `Triu`, `TrilView`, `TriuView`),
`Convert` and `Reduce`. The ABI document's Metal section is the parameter table; this
document records only the choices that are not visible in it.

**One vocabulary.** `Entry`, `DType`, `BinaryOp`, `CompareOp`, `ScalarSide`,
`ReduceOp`, `ReduceStrategy`, `CLayout`, `CAxisValues` and `Error` are the crate's
existing types. The MSL tags are `#define`s whose values a host test compares against
`tag()`, so a drift is a test failure rather than a silent mis-dispatch. No
Metal-only tag space, and no second layout type.

**`Plan::bytes` is the destination's allocation.** The engine allocates from it
(`src/engine.rs:199,269,346`), so an entry whose position count differs from its
output length must plan the output: `embed_diagonal_copy` positions over the compact
input and writes a destination that gained an axis, and a masked `tril`/`triu` writes
the full compact output. The plan review caught this as a real under-allocation, and the
post-implementation review found the second half: an entry that writes *through a view*
must plan the caller's allocation (`dst_len`/`out_len`), not `element_count()`, because a
strided destination with an offset reaches past its logical count. Every planner that
takes a destination view now plans `bytes(dtype, dst_len)`, and the host lane pins it.

**32-bit positions, 64-bit lengths.** Every kernel reads
`[[thread_position_in_grid]]` as a `uint`. The planners refuse a domain above
`u32::MAX` with `LaunchLimit`; `BinaryOp::DivComplex` addresses two interleaved
components per element, so its planner refuses above `u32::MAX / 2`. Lengths and
addresses stay `long` everywhere else, and any index arithmetic that scales by two
widens before it multiplies.

**`plan_elementwise` is narrowed to the six flat entries.** `Entry::is_elementwise`
is a genre, not a promise: it also covers `FillOne`, `Binary`, `Compare`, `Select` and
`Clamp`, each of which needs a larger parameter block, and `FillZeroView` is a view entry
whose MSL takes a layout (`plan_elementwise` answers `NotElementwise` for it, as the host
lane asserts). The CUDA
catalogue answers this with a per-entry matcher and so does this one
(`FLAT_ENTRIES`); anything else elementwise returns `UnsupportedOp` rather than a plan
whose dispatch the MSL does not implement, and a non-elementwise entry keeps
`NotElementwise`. The consumer draft (`tenferro-rs` branch `tenferro-rs-metal-u4`,
`engine.rs:188`) only reaches the six flat entries, so this stays source compatible.

**`EmbedDiagonalCopy` writes the diagonal only.** The destination must be zero
before the launch; the ABI document states it, and the device test zero-fills. Writing
a whole destination would need a second kernel and a second launch for one entry.

**The diagonal map is the corrected one.** `cuda/src/diagonal.cu` gained a rank ≥ 3
fix: the diagonal output axis is `axis_a - (axis_b < axis_a)` and is skipped while the
remaining output coordinates are walked. The MSL extract inherits it, and the device
test pins a rank-3 case that the baseline's old map would fail. The post-implementation
review showed that the *planners* admitted a rank-9 layout which `layout_mirror` refuses
later, so every layout-taking planner now checks the mirror's rank first, including an
embed's enlarged output.

**Triangles compare `col - row` with `k`, not `col` with `col - k`.** The baseline's
form misbehaves at an extreme `k` (a 32-bit overflow in the baseline's arithmetic);
comparing on axes 0 and 1 of a compact pair is well defined for every `k`, and the device
test includes `i64::MIN` and `i64::MAX`. A masked element stores `+0` of its dtype
(`0.0f` or `(0, 0)`); no input address is read for it in the `*_view` variants.

**`Reduce` has two strategies and one geometry rule.** `UNIT` is one thread per
output, folding the axis serially from its first element — bit-identical to the CUDA
unit body, since both seed from element 0 (`cuda/src/reduce.cu:~560`) and fold in the
same order, and the two agree bit for bit wherever Metal's subnormal flush cannot
reach. `PLANE` is one threadgroup per output, 256 lanes (the slice-1
`THREADGROUP`), lane `l` seeding from axis position `l` and striding by the
threadgroup width, then a `threadgroup` tree with a barrier per level and lane 0
storing. The tree order is this vendor's, so the device lane compares `PLANE` against
the exact sum at `1e-5` relative and `UNIT` bit-exactly. `UNIT` is legal only when the
axis is at most one threadgroup wide, `PLANE` only when it is at least that wide: a
planner that admitted the other side would hand a kernel a geometry it cannot cover.
`Max`/`Min` propagate NaN through the same bit-test rules as the flat entries, and
`SumSquares` seeds with the square so the first element is not folded twice.

**`Convert` serves f32 ↔ Complex32 only**, and refuses an identity pair: the CUDA
entry's identity form is a `copy_exact`, and a second way to copy would be a second
code path for the same operation. The imaginary part of a widened value is `+0`, the
real part of a narrowed one is kept, and the device lane checks both components
because a `float2`-shaped read would get the ordering wrong.

### Numerical bars

Unchanged from slice 1 where the operation is unchanged: bit-exact for copies,
`fill_zero`, the seven f32 arithmetic and sign entries, `ADD`/`SUB` of Complex32 and
now `compare` (one byte per position) and `convert`'s kept component; 8 ulp for the
transcendentals, `BINARY_POW` and `BINARY_DIV_COMPLEX`; 4 ulp for Complex32 `MUL`; a
`PLANE` reduction at `1e-5` relative and a `UNIT` reduction bit-exact.

The subnormal flush is measured, not denied: the runtime compiles with Metal's
default options, so a subnormal operand and a subnormal result are flushed, and the
CUDA entries do not flush. Slice 1's device test pins it; this slice's `BINARY_MUL`
test re-asserts it for a subnormal product rather than assuming the flag never
reaches a new entry. Closing the gap needs a compile-options seam in
`t4a-metal-runtime` (U4), not a change here.

`MAX`/`MIN` follow the CUDA `t4a_cuda_nan_max/min`; the sign of a zero result is not
part of the contract, because C's `fmin`/`fmax` may return either, and the device lane
accepts either sign. `BINARY_REM` is the truncating remainder (`precise::fmod`), matching
`fmodf` on the CUDA side, so a zero divisor is `NaN` on both.

## Verification

Every claim above is a test, not a reading:

- Host lane (`tests/metal_planning.rs`, no GPU): the tag tables against the MSL
  `#define`s, the encoded length of both mirrors and their byte layout against the
  Rust structs, every planner's domain and `bytes`, the `DivComplex` half-width
  cliff, the `rank > 8` refusal, the strategy split at 256, the unserved entries'
  errors, `THREADGROUP` against the MSL `#define`, and the "no kernel beyond
  `SYMBOLS`" scan of the composed text.
- Device lane (`tests/metal_kernels.rs`, `gpu-tests-metal`, M5 Max, fail-loud when
  there is no Apple GPU): each family over `{1, 3, 255, 256, 257, 4096, 100_003}`
  where the family has a flat domain, a sentinel-filled destination on each entry whose
  whole allocation is compared (a write past the domain, or a threadgroup that stops at
  the first one, fails), and a reference computed on the host — including the rank-3
  diagonal, the `k = i64::MIN/MAX` triangles, the byte-per-position `compare`, a
  `select` whose predicate is set on every third byte, the `clamp` NaN bounds, the
  reduction strategies over `{1, 2, 255, 256}` / `{256, 257, 1024, 100_003}`, a
  NaN/zero/infinity fold, and `stats().launches` advancing by exactly one per dispatch.
  A separate case puts every structural entry and the flat op family at 300–900
  positions — two to four threadgroups, the last one partial — in both dtypes wherever
  the entry admits them, including a 300x300 embedded diagonal.
- The A100 CUDA lane is untouched: no file under `cuda/`, `src/lib.rs` unchanged
  since slice 1's one line, `T4A_CUDA_ABI_REVISION` still 14.

## Deferred

- A compile-options seam for `fastMathEnabled = NO` (the subnormal flush), in
  `t4a-metal-runtime` (U4's row).
- `TiledTranspose`, `Gather`/`ScatterAdd`, `Slice`/`Pad`, `ConcatenateCopy`,
  `CombinedStrided`, the linalg families, the fusion emitter (U6) and the `_any`
  dynamic-rank twins, each with the parameter block it needs.
- The split reduction (`ReducePartials` + the combine pass) once a Metal caller
  reaches it.
- `expm1`, `log1p`, `erf`, unchanged from slice 1: MSL has no such function.
