# 2026-11-14 — the Metal source set and the Metal planning target (U3, slice 1)

Unit U3, slice 1, of tensor4all/tenferro-rs#2402, following
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§7 and §10. U1 (`83fe982d`, tenferro-rs) and U2 (`b8dee5e9`, t4a-gpu-backend#16)
are merged and are not touched here. The sequence deviation — U3 before the
`rocm-backend.md` M1a rename — is recorded in the design of this slice.

Slice design: [`../design/slice-metal-elementwise.md`](../design/slice-metal-elementwise.md).
Evidence: [`evidence/2026-11-14-metal-elementwise/`](evidence/2026-11-14-metal-elementwise/).

## What this adds

- `metal/include/t4a/abi.msl`: the MSL header — `T4A_METAL_ABI_REVISION 1`, the
  dtype tags (0 `F32`, 5 `C32`), the fourteen unary tags (the values of
  `UnaryOp::tag()`), `struct t4a_metal_c32 { float re; float im; }` with
  `static_assert` size 8 / alignment 4, and the arithmetic, complex, sign and
  guarded-index helpers.
- `metal/src/elementwise.msl`: `t4a_metal_fill_zero`, `copy_exact`, `add`, `sub`,
  `mul`, `div` (f32 only) and `unary`, all tail-guarded.
- `src/metal.rs`: `SIMD_WIDTH = 32` (the planning input) and
  `THREADGROUP = 8 * SIMD_WIDTH`, `ABI_REVISION`, the seven-row `SYMBOLS` table,
  `DTYPES`, `UNARY_OPS`, `Dispatch`/`Plan`, `plan_elementwise`/`plan_fill_zero`/
  `plan_copy_exact`/`plan_unary`, and the `metal::source` registry with `compose`.
- `src/lib.rs`: one line, `pub mod metal;`. `ABI_REVISION` stays 14 and no CUDA
  entry, planner, tag or source text changes.
- `tests/metal_planning.rs` (host) and `tests/metal_kernels.rs` (device, behind
  `gpu-tests-metal`); the ABI document's Metal section; the slice design; CI.
- `Cargo.toml`: `/metal/**` in `include`, the `gpu-tests-metal` feature, the
  optional git-pinned `t4a-metal-runtime`, and the two `[[test]]` rows.

## Decisions

1. **Reuse, not a second vocabulary.** The first draft introduced `metal::Entry`,
   `metal::UnaryOp` and a Metal tag space. The plan review pointed at
   `Entry::is_elementwise`/`supports`, `UnaryOp::tag`/`accepts`, `DType::tag`, and
   the existing `plan_*` names; the slice now adds one seven-row `SYMBOLS` table and
   nothing else that duplicates a table. The MSL `T4A_METAL_UNARY_*` values are
   checked against `UnaryOp::tag()` by a host test rather than restated freely.
2. **Complex32 is two scalars, not a `float2`.** MSL gives `float2` alignment 8;
   the host layout is `num_complex::Complex32` at alignment 4, and a binding offset
   of 4 would then be a misaligned 8-aligned load. `cuda/include/t4a/abi.cuh:34-40`
   makes the same choice and asserts it at `:277-278`. The MSL header asserts size 8
   and alignment 4, and the host lane asserts the Rust mirror.
3. **`EXPM1`, `LOG1P`, `ERF` are refused, not approximated.** Measured against this
   OS's GPU compiler headers, MSL defines none of them, nor `hypot`
   (`evidence/…/msl-math-headers.log`). They keep their tag numbers so the table
   reads against the CUDA one, and `metal::plan_unary` returns `UnsupportedOp`. The
   Complex32 magnitude is a scaled form instead of `sqrt(re²+im²)`, which underflows
   at 1e-30 and overflows at 1e30.
4. **`metal::precise::*` for every transcendental, division and comparison that the
   contract pins.** The U2 runtime compiles with `nil` `MTLCompileOptions`, so Metal's
   default `fastMathEnabled = YES` applies and the caller has no seam to change it.
   The precise entry point is the higher-accuracy form of that one call, not a rounding
   guarantee (the device lane's bar is 8 ulp for a transcendental), and it does not
   change the mode of the surrounding expression, which is why NaN and infinity are
   classified by bit pattern and why the subnormal flush below is measured and stated
   instead of denied. A compile-options seam is a U4 follow-up.
5. **Every kernel guards its tail.** The tiled dispatch (`threadgroups × 256`)
   covers more positions than elements whenever `len` is not a multiple of 256, and
   Metal does not bound-check an index. The device lane proves it with a
   sentinel-filled destination tail rather than with the guard's presence.
6. **An elementwise entry with no kernel is `UnsupportedOp`.** `FillOne`, `Binary`,
   `Compare`, `Select` and `Clamp` are `Entry::is_elementwise()` and exist in the
   CUDA catalogue, so a Metal `plan_elementwise` that returned a plan would hand the
   caller a dispatch no MSL kernel implements. Found by the plan review.
7. **No MPS call, no contraction kernel.** GEMM/contraction is U5 through
   `MPSMatrixMultiplication` in the engine, FFT is a later unit; the ABI document
   states the split so a future contributor does not add an MSL GEMM beside it. The
   host lane fails if the Metal text declares any kernel beyond `SYMBOLS`.
8. **bf16 is out.** The task text says f32/bf16/c32; the design lists bf16 as a
   non-goal for the first version (§2) with f32 + Complex32 as the dtype set (§8),
   and the maintainer's 2026-11-14 decision on #2402 makes the document take
   precedence over the dispatch wording.

## CI

`t4a-metal-runtime` is unpublished (0.1.0), so the device lane pins it by git
revision. A git dependency makes the offline lock check fail on a cold cache, which
was measured, not assumed:

```
$ CARGO_HOME=<fresh> cargo metadata --locked --offline …
error: failed to get `t4a-metal-runtime` as a dependency …
  can't checkout from 'https://github.com/tensor4all/t4a-gpu-backend': you are in the offline mode (--offline)
$ CARGO_HOME=<fresh> cargo fetch --locked && CARGO_HOME=<fresh> cargo metadata --locked --offline …
METADATA OFFLINE OK
```

So the `rust` job gains `cargo fetch --locked` before the lock check, and the
compile-only device step now covers `--features gpu-tests-metal --no-run` beside
`--features gpu-tests --no-run`. `t4a-metal-sys` stubs `dlopen` off macOS, so the
Metal crates build on the `ubuntu-latest` runner (t4a-gpu-backend's own CI builds
them there with `--all-features`). No hosted job executes a Metal kernel.

`[lints.cargo] unused_dependencies = "allow"` is in the manifest because both GPU
dependencies are used only by the test targets their features select, which is
exactly the configuration the lint cannot see.

## Verification

Hardware and OS: Apple M5 Max (MacBook Pro, Mac17,6), macOS 26.7.2 (25G332), MSL
from `GPUCompiler.framework/Versions/32023`. Commands and observed results are in
[`evidence/2026-11-14-metal-elementwise/`](evidence/2026-11-14-metal-elementwise/).

| Command | Result |
| --- | --- |
| `cargo fmt --check` | no diff |
| `cargo clippy -j 8 --locked --all-targets -- -D warnings` | clean |
| `cargo clippy -j 8 --locked --all-targets --features gpu-tests-metal -- -D warnings` | clean |
| `cargo test -j 8 --locked` | 7 + 15 + 91 pass (CUDA host suites unchanged), 0 failed |
| `cargo test -j 8 --locked --features gpu-tests-metal` | + 12 device tests pass on the M5 Max, 0 failed, none skipped |
| `otool -L <device test binary>` | `libiconv`, `libSystem` only — no Metal, MPS or Accelerate entry |
| `cargo metadata --locked --offline` after `cargo fetch --locked` | exit 0 |

The device lane runs every served entry × `{F32, C32}` ×
`{1, 3, 255, 256, 257, 4096, 100_003}` elements against a CPU reference, unary
included, with a sentinel-filled destination tail on each: `fill_zero` and
`copy_exact` bit-exactly (including a NaN payload, a signed zero and a subnormal in
the copy source), the f32 arithmetic and sign entries at a zero-ulp bar over
normal-range operands, the transcendentals within 8 ulp, `div` bit-identical to the
correctly rounded quotient over a wide dynamic range, Complex32 `mul` within 4 ulp,
`abs` of Complex32 at 1e-30 and 1e30 finite with `hypotf`'s NaN/infinity precedence,
`sign` of `±0`, NaN/inf propagation case by case, and `stats().launches` advancing
by exactly one per entry call.

The transcendental tolerance is only the committed bar (8 ulp against the host's
libm); no uncommitted harness number is quoted, because a reviewer could not
reproduce it. The committed device lane is the evidence.

**Subnormals flush to zero, measured.** `smallest + smallest` returns `0`, not
`2 * smallest`, because the runtime compiles with Metal's default options where fast
math is on; the CUDA entries do not flush, since nvcc runs without `--use_fast_math`.
This is a numerical difference between a Metal entry and its CUDA twin on otherwise
identical input (the transcendental tolerance is another, by a few ulp). It is asserted in the device lane (`fast_math_flushes_subnormals_and_the_normals_stay_exact`),
stated in the ABI document, and recorded as a deferred item rather than papered over:
removing it needs `MTLCompileOptions.fastMathEnabled = NO`, i.e. a compile-options
seam in `t4a-metal-runtime`, which this slice does not own. The same measurement
drove two code changes — NaN/infinity classification by bit pattern instead of
`x != x`, and `metal::precise::divide` in the Complex32 magnitude and sign — because
fast math may fold the former and approximate the latter.

## Reviews

- Pre-implementation, mandatory gate: **GLM-5.3-flash** (`ollama-cloud/glm-5.3-flash`,
  read-only `glm-brainstormer`), verdict **CHANGES** on the first plan and **CHANGES**
  on the revision, no blockers. Its four IMPORTANT findings are in the slice's
  history: an unprovable geometry assertion (`Runtime::stats` counts dispatches
  only), the self-contradictory unary matrix versus `UnaryOp::accepts`, the missing
  derivation notices (review checklist §6), and the missing per-entry alias rules
  (checklist §3). Its six MINOR findings (c32 alignment, the `otool` pattern, the
  `t4a_cuda_` text assertion, `SIMD_WIDTH` unused, the `Cargo.lock` entry, the
  review-model record) are all applied.
- Post-implementation, repository gate: **GPT-6.1 Sol** (`reviewer-gpt`, read-only),
  verdict **CHANGES**, no blockers, four IMPORTANT and two MINOR findings, all
  verified against the code and all fixed in this branch:
  - IMPORTANT the planner admitted lengths above the kernels' 32-bit
    `[[thread_position_in_grid]]`, and the Complex32 copy scaled the word index in
    32-bit arithmetic (`2 * index` wraps at 2^31) → `plan_len` now refuses anything
    above `u32::MAX` with `LaunchLimit`, the copy widens first
    (`2 * ulong(index)`), and the host lane pins both ends.
  - IMPORTANT `abs` returned `0` for `(NaN, 0)`, because `fmax(NaN, 0)` is 0 under
    fast math and the early zero shortcut took it → the three `hypotf` outcomes are
    now explicit and ordered (infinity, NaN, zero), with device cases for
    `(NaN, 0)`, `(0, NaN)` and `(inf, NaN)`.
  - IMPORTANT `metal::precise::*` does not protect the surrounding arithmetic, so the
    fast-math claim in the design was too strong → classification is by bit pattern,
    the magnitude and the Complex32 sign divide through `precise::divide`, the
    subnormal flush is measured and documented instead of denied, and the design's
    table now says what fast math actually does.
  - IMPORTANT the unary device tests used five or six elements and checked no tail,
    so a kernel that returned past the first threadgroup would have passed → every
    unary op, f32 and Complex32 (both destination routes, to Complex32 and to f32 for
    `abs`), now runs over all seven lengths with a sentinel tail.
  - MINOR the ulp comparator accepted an infinity where a finite value was expected →
    classification (NaN, infinity, sign) is compared before the tolerance.
  - MINOR six documentation statements exceeded the code (the `CpuEngine` name in the
    shared error message, the `Div` dtype-table derivation, the README's "silent"
    device lane, the include-line wording) → each corrected against the source.
- Post-implementation, verification of those fixes (same reviewer, same model; the
  recorded verdicts before approval were **CHANGES** with no blocker, and the final
  one was **APPROVE**). Each finding was resolved rather than argued:
  the Complex32 unary paths ran a nine-element fixture with no tail check (now over
  all seven lengths for both destination dtypes); the numerical prose still
  overreached (the `precise::*` flag-independence wording, `CONJ` in the f32 exact
  list where `UnaryOp::accepts` refuses it, "the one place the backends differ", and
  `precise::*` described as "correctly rounded" where the committed bar is 8 ulp);
  and this review record itself miscounted its passes.
- Post-approval review by the organisation's gate reviewer (**GLM-5.3-flash**,
  `NEEDS_CHANGES`), on the open pull request. Each finding was checked against the
  hardware before anything changed:
  - `sign` of a Complex32 whose magnitude flushes: reported as returning `inf`/`NaN`
    because `abs` flushes a subnormal magnitude to zero. **Measured and not
    reproducible as reported**: under fast math the comparison `value == 0.0f` is
    itself flush-to-zero (`sign` of a subnormal f32 returns `+0`), so the exact-zero
    guard already caught the pair. The guard on the *magnitude* was added anyway, as
    one branch that cannot misfire and does not depend on comparison semantics, and a
    device case now pins the behaviour both entries owe: `hypotf` says a subnormal
    pair is nonzero, the device flushes it, and `abs` and `sign` must agree on zero.
  - `0/0` was claimed but not tested: only `x/0` existed. Real `0/0`, `-1/0` and
    `smallest/smallest` cases are now in the device lane (`NaN`, `-inf`, `NaN`).
  - The ABI document counted "six" f32 entries where it listed seven.
  - The `1.06e-7` relative-error figure came from an uncommitted harness and is
    dropped everywhere rather than quoted; the committed 8 ulp bar is the evidence.

## Deferred (and why)

Slice 2: the op-tag elementwise entry with `scalar_side`, `fill_one`, `compare`,
`select`, `clamp`. Slice 3: structural movement and transpose/broadcast/concatenate.
Slice 4: reductions. Later: gather/scatter, linalg. Then U5 (MPS GEMM) and U6 (the
MSL fusion emitter) in their own units, and the macOS CI lane in U4. Complex32
division stays out until the baseline's scaled Baudin-Smith form is ported; no MSL
GEMM will ever be added, by design.
