# 2026-11-16 — the Metal structural entries (U3, slices 2–4)

Unit U3, slices 2–4, of tensor4all/tenferro-rs#2402, following
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§7 and §10. Slice 1 (`e870458`, #35) is merged and is not rewritten here beyond the
one-line `pub mod metal;` it already added to `src/lib.rs`.

Slice design: [`../design/slice-metal-structural.md`](../design/slice-metal-structural.md).
Evidence: [`evidence/2026-11-16-metal-structural/`](evidence/2026-11-16-metal-structural/).

## What this adds

- `metal/include/t4a/abi.msl`: `T4A_METAL_ABI_REVISION 2`; the by-value layout and
  dimension-map mirrors (`t4a_metal_layout` 144 B, `t4a_metal_axis_values` 72 B) with
  `static_assert`s; the binary, compare, scalar-side, reduce and strategy tags;
  `t4a_metal_decode`/`t4a_metal_reaches`; bit-pattern NaN/infinity tests; the
  `nan_max`/`nan_min`, `clamp` and scaled Baudin-Smith complex-division helpers; and
  the reduce fold helpers. `metal::precise::*` on every transcendental and on every
  division whose rounding is part of the contract.
- `metal/src/movement.msl`, `diagonal.msl`, `convert.msl`, `reduce.msl` — five
  movement entries, six diagonal and triangular entries, the conversion pair and the
  reduce entry — and the four op-tag entries appended to `elementwise.msl`.
- `src/metal.rs`: `ABI_REVISION = 2`, 24 `SYMBOLS`, the `FLAT_ENTRIES` set, the
  planner for each new entry, `layout_mirror`/`layout_bytes` and
  `axis_values_bytes` (manual little-endian, because the crate is
  `#![forbid(unsafe_code)]`), and the five-module `source` registry.
- `tests/metal_planning.rs` (host) and `tests/metal_kernels.rs` (device, behind
  `gpu-tests-metal`): the tag tables against the MSL `#define`s, the mirror's length
  and byte order, every planner's domain and `bytes`, the 32-bit cliffs, the
  strategy split, and a device case per family with a host reference.
- `docs/abi/c-kernel-abi.md`: the Metal section rewritten for revision 2 (entry and
  tag tables, structural semantics, numerics, overlap, geometry, and what the set
  deliberately does not contain).
- CI: the clippy step now compiles the device test suite too
  (`--features gpu-tests-metal`), because that target is where the last two lint
  findings lived and a hosted job should not be blind to it.

## Decisions

1. **`Plan::bytes` is the destination's allocation.** The engine allocates exactly
   that many bytes (`crates/tenferro-gpu/src/metal/engine.rs:199,269,346` in the
   consumer draft), so an entry whose position count differs from its output length
   plans the output (`embed_diagonal_copy`, `tril`/`triu`), and an entry that writes
   through a view plans the caller's allocation, not `element_count()` — a strided
   destination with an offset reaches past its logical count. Both halves were review
   findings; the host lane pins each with a case whose allocation is bigger than the
   element count.
2. **Rank is checked before the mirror is built.** `layout_mirror` refuses a rank
   above `MAX_RANK`, so a planner that admitted one would hand the caller a plan whose
   encode fails later. Every layout-taking planner now refuses it first, including an
   embed whose output is one rank above its input — again a review finding.
3. **32-bit positions, 64-bit lengths.** Domains above `u32::MAX` are `LaunchLimit`;
   `BinaryOp::DivComplex` additionally caps at `u32::MAX / 2` because it addresses two
   interleaved components per element; the reduce axis is capped too, since a plane
   lane walks it with a 32-bit step.
4. **One vocabulary, one tag space.** `Entry`, `DType`, `BinaryOp`, `CompareOp`,
   `ScalarSide`, `ReduceOp`, `ReduceStrategy`, `CLayout`, `CAxisValues` and `Error`
   are the crate's existing types, and a host test compares every served MSL tag
   against its `tag()`. No Metal-only enum and no second layout type.
5. **`plan_elementwise` stays the six flat entries.** `is_elementwise` is a genre: it
   also covers `FillOne`, `Binary`, `Compare`, `Select` and `Clamp` (larger parameter
   blocks) and `FillZeroView` (a view entry with a layout argument, so it answers
   `NotElementwise`). A plan whose dispatch the MSL does not implement is the failure
   mode slice 1's review named, and the consumer draft reaches only the six.
6. **Six entries were deleted from the draft plan.** `FillOne`, `ReducePartials` and
   the four `Mixed*` tags have no caller: the engine's `ElementwiseReadOp` is
   Add/Subtract/Multiply/Negate/Conj/Divide, no Metal caller reaches `FillOne`, and a
   split reduction waits for a Metal split policy (the consumer uses MPS where it
   can). `TiledTranspose`, `CombinedStrided` and `ConcatenateCopy` are collapse into
   entries that already exist (a stride relabelling and two address maps).
7. **The diagonal map is the corrected one.** `cuda/src/diagonal.cu`'s rank ≥ 3 fix —
   the diagonal output axis is `axis_a - (axis_b < axis_a)` and is skipped while the
   remaining coordinates are walked — is inherited, and the device lane pins a rank-3
   case that the old map fails.
8. **Triangles compare `col - row` with `k`.** The baseline's `col - k` misbehaves at
   an extreme `k`; the device case now covers `i64::MIN` and `i64::MAX`. A masked
   element stores the dtype's `+0`, and no input address is read for it in the `*_view`
   twins.
9. **`Reduce` has two strategies and one rule.** `Unit` is one thread per output
   folding serially from the first element — the CUDA unit body's order, so
   bit-identical where the subnormal flush cannot reach; `Plane` is one threadgroup of
   256 lanes with a `threadgroup` tree, compared to the exact sum at `1e-5` relative
   because the tree order is this vendor's. `Unit` is legal only when the axis fits one
   threadgroup, `Plane` only when it does not; the host lane pins the split at 256.
10. **`Convert` serves f32 ↔ Complex32 only.** An identity pair is `CopyExact`; a
    second way to copy would be a second code path for one operation. The widened
    imaginary part is `+0`, the narrowed real part is kept, and the device lane checks
    both components.
11. **Provenance.** `metal/src/reduce.msl` carries the CubeK Reduce notice its CUDA
    twin carries, and `THIRD_PARTY_NOTICES.md` now names the Metal reduction sources.
    A review finding: the derivation existed but the notice's scope still said CUDA.
12. **bf16 is out** (design §2/§8 supersede the dispatch's "f32/bf16/c32"), and the
    subnormal flush stays out of scope here: it closes in `t4a-metal-runtime` with a
    compile-options seam (`fastMathEnabled = NO`), not in a kernel.

## Verification

Hardware and OS: Apple M5 Max (MacBook Pro, Mac17,6), 40 GPU cores, Metal 4,
macOS 26.7.2 (25G332), MSL compiler from `GPUCompiler.framework/Versions/32023`,
rustc 1.100.0. Identity and commands:
[`evidence/2026-11-16-metal-structural/machine.log`](evidence/2026-11-16-metal-structural/machine.log).

| Command | Result |
| --- | --- |
| `cargo fmt --check` | no diff |
| `cargo clippy -j 4 --locked --all-targets -- -D warnings` | clean |
| `cargo clippy -j 4 --locked --all-targets --features gpu-tests-metal -- -D warnings` | clean |
| `cargo test -j 4 --locked` | 7 + 26 + 91 pass, 0 failed (CUDA host suites unchanged) |
| `cargo test -j 4 --locked --features gpu-tests-metal --no-run` | compiles |
| `cargo test -j 4 --locked --features gpu-tests-metal --test metal_kernels` | 22 device tests pass on the M5 Max, 0 failed, none skipped |

The device lane runs every family against a host reference, with a sentinel-filled
destination whose whole allocation is compared, so a write past the domain — or a
threadgroup that stops at the first one — fails. Bit-exact: `fill_zero_view`, all five
movement entries, `convert`'s kept component, `compare` (one byte per position),
`clamp` (statement order without NaN folds), and `Unit`-strategy f32 sum over the serial
fold. Within 8 ulp: the transcendentals, `Pow`, `DivComplex`. Within 4 ulp: Complex32
`Mul`. A `Plane` fold is compared to the exact sum at `1e-5` relative, except `Max`/`Min`
and a `100_003`-halves case, which are exact. Every structural entry and the flat op
family also run at 300–900 positions — two to four threadgroups with a partial last one,
in both dtypes wherever the entry admits them, including a 300x300 embedded diagonal —
and the reductions at `{1, 2, 255, 256}` (`Unit`) and `{256, 257, 1024, 100_003}`
(`Plane`), each with its own tail assertion.

The subnormal flush measured in slice 1 still holds and is re-asserted for a new entry
(`BINARY_MUL` of a subnormal), not assumed: fast math is on because the runtime compiles
with `nil` compile options, and `MTLCompileOptions.fastMathEnabled = NO` is a U4 seam.

`MAX`/`MIN` follow `t4a_cuda_nan_max/min`; the sign of a zero result is not part of the
contract (C's `fmin`/`fmax` may return either), and the device lane accepts either — the
device disagreed with Rust's `min` on `(-0, +0)` for exactly this reason, and the test
records the rule instead of pinning one vendor's choice.

## Reviews

- Pre-implementation, the mandatory gate of the task: **GLM-5.3-flash**
  (`ollama-cloud/glm-5.3-flash`, read-only), verdict **CHANGES** on the first plan.
  Two blockers and three importants, all applied to the plan before any code:
  `Plan::bytes` under-allocated an axis-inserting output; `DivComplex` needed the
  half-width cap because it addresses components, not elements; the reduce parameter
  block was missing; "component-wise" mis-described the mixed tags (the CUDA twins keep
  zero cross terms, so the tags were deleted instead); and `fill_zero_view` cannot have a
  stride-zero destination. Its delete list (`ReducePartials`, `FillOne`, `Mixed*` and
  two of four cliff families) is decision 6.
- Post-implementation, the repository's gate: **GPT-6.1 Sol** (`openai/gpt-6.1-sol`,
  read-only, separate process), verdict **CHANGES**, one blocker and eight importants.
  Each was verified against the source before anything changed:
  - BLOCKER a view destination's `bytes` used `element_count()`, so `[2]` with stride 2
    and offset 1 planned 8 bytes and wrote element 3 → every planner that writes through
    a view now plans the caller's allocation (`plan_into`), and the host lane pins three
    of them plus `embed` and `reduce` (decision 1).
  - IMPORTANT `plan_reduce` capped only the output length, so a `Plane` axis of
    `i64::MAX` planned a walk that overflows its 32-bit step → the axis is capped at
    `u32::MAX` with `LaunchLimit`, and the host lane pins both sides of the cliff.
  - IMPORTANT the planners admitted rank 9 although the mirror has eight slots → every
    layout-taking planner refuses it first, embed's enlarged output included
    (decision 2).
  - IMPORTANT the Complex32 reduce reference read the layout as row-major and compared
    only the real component, and the fixture contained a zero so a product that always
    returned zero passed → the reference follows `row + 2 * column`, both components are
    compared, and the product fixture is unit-circle factors whose result is exactly `i`.
  - IMPORTANT the f32 `Plane` product's fixture contained a zero → the product case now
    uses alternating ±1, so the exact answer is ±1 and cannot be reached by returning
    zero.
  - IMPORTANT no structural entry crossed a threadgroup, so a first-threadgroup-only
    implementation passed → a 600-position case now covers all five movement entries,
    broadcast, extraction, both triangle pairs, and `compare`/`select`/`clamp`, in both
    dtypes where the entry admits them.
  - IMPORTANT the sentinel slack was allocated but not compared for several entries →
    every destination comparison now covers the whole allocation (`with_sentinel_slack`
    or an explicit word-for-word expectation).
  - IMPORTANT the evidence logs carried no GPU, OS or toolchain identity → `machine.log`
    records `sw_vers`, the hardware and Metal profile, and the rustc/cargo versions, and
    the design's evidence claim now says what the lane actually asserts.
  - IMPORTANT the reduction module lacked the CubeK notice its CUDA twin carries →
    `metal/src/reduce.msl` carries the derivation and `THIRD_PARTY_NOTICES.md` names it.
  - Three MINORs were real documentation defects and are fixed against the code:
    "IEEE remainder" (it is `fmod`), "every division uses `precise::*`" (the complex
    division used bare `/`; those four sites now use `precise::divide`, so the claim is
    true and greppable), and "a `Unit` reduction is bit-identical" without the
    normal-range qualifier the flush requires. Four MINORs corrected the slice design
    (a helper signature that does not exist, `FillZeroView` miscounted as
    `is_elementwise`, `i32` where the tests use `i64`, and a `select` fixture described
    as four patterns when it is one), one corrected the plan's `Select` dtype row (f32
    only: the shared table has no Complex32 select), and the README's Metal status now
    says revision 2. Its two delete-list items are applied: the duplicate-axis `seen`
    vector is gone (the borrowed `dims` prefix already answers it) and the two
    identity/forwarding Complex32 reduce wrappers are deleted.

- Verification pass on those fixes, same reviewer and model: verdict **CHANGES**, two
  importants and five minors, no blocker. Both importants were test-coverage gaps in the
  fix itself and are closed rather than argued: the 300–900-position case now also covers
  Complex32 triangles, Complex32 `materialize_strided`, and a 300x300 embedded diagonal,
  and every reduction case, the short `select`/`clamp` cases and the Complex32 → f32
  `convert` now assert their sentinel tail (a store past the domain would have passed).
  The minors were stale prose: the plan's `bytes` table still described views as compact,
  the slice design kept "IEEE remainder" and `i32::MIN/MAX` after the ABI table was fixed,
  the plan still said `select` serves both dtypes, and a host-test fixture had three
  unused bindings. All corrected.
- Final pass on the pushed head, same reviewer and model: `CRITICAL: 0`,
  `NEEDS_CHANGES: 0`, **APPROVE**, with three stale-text minors (the `Plan::bytes` doc
  comment still gave the element-count formula, this plan still called the `Unit` fold
  bit-identical to CUDA without the flush qualifier, and its unchanged-file row listed
  `README.md` and the CI workflow that this change does edit). All three were fixed
  before the head was pinned.

## Deferred

- `fastMathEnabled = NO` (the subnormal flush), in `t4a-metal-runtime` (U4).
- The split reduction (`ReducePartials` and the combine pass) once a Metal caller
  reaches it; `TiledTranspose`, `ConcatenateCopy`, `CombinedStrided` and the indexing
  family as their own entries; linalg and the fusion emitter (U6); the `_any`
  dynamic-rank twins that need a device-resident layout.
- `expm1`, `log1p`, `erf`, unchanged from slice 1: MSL has no such function.
