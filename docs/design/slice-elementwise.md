# Slice F: remaining elementwise, comparison and strided-combination operations

Status: design, 2026-10-21. Sixth implementation slice of
[the migration design](cubecl-kernel-port.md), after
[slice E](slice-conversions.md).

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-gpu/src/kernels/{elementwise,structural}.rs`.

## 1. Scope

This slice finishes the baseline's elementwise inventory and the four structural
movement extras. It is the last slice before the linalg/RRQR helpers
(a separate slice, because those are the only remaining custom kernels in
`tenferro-linalg`).

| Baseline symbols | This slice |
| --- | --- |
| `neg_float`, `neg_complex`, `neg_int`, `exp_float`, `log_float`, `sin_float`, `cos_float`, `tanh_float`, `sqrt_float`, `rsqrt_float`, `expm1_float`, `log1p_float`, `erf_float`, `abs_float`, `abs_int`, `abs_complex32/64`, `sign_float`, `sign_int`, `sign_complex`, `conj_complex` | `t4a_cuda_unary` |
| `maximum_*`, `minimum_*`, `pow_float`, `rem_float`, `pow_int_checked`, `div_int_checked`, `rem_int_checked`, `scalar_pow_int_checked`, `scalar_div_int_checked`, `scalar_rem_int_checked`, `scalar_div_float`, `scalar_pow_float`, `scalar_rem_float`, `div_complex_parts`, `scalar_real_complex_binary`, `scale_in_place_*` | `t4a_cuda_binary` |
| `compare_float_bool`, `compare_int_bool` | `t4a_cuda_compare` |
| `select_bool_float`, `select_bool_int` | `t4a_cuda_select` |
| `clamp_float` | `t4a_cuda_clamp` |
| `broadcast_multiply_*`, `axpby_strided_source_*` | `t4a_cuda_combined_strided` |
| `broadcast_in_dim_kernel`, `reverse_kernel`, `concatenate_copy_kernel` | **no new entry**: expressed with the revision-1 movement entries (§3.6) |

Six entries and no new structs, for 46 baseline entry symbols. The next slice
(linalg auxiliaries and RRQR primitives) is not part of this one.

## 2. Consolidation decisions

The baseline has one CUDA entry per `(operation, dtype)` pair. Every previous
slice replaced that fan-out with a runtime dtype/op tag; this slice keeps the
rule, and also removes four entries whose semantics the revision-1 movement
entries already cover with an explicit layout:

| Baseline entry | Replacement |
| --- | --- |
| `broadcast_in_dim_kernel` (insert size-1 axes) | `t4a_cuda_materialize_strided` with an expanded **source read layout**: output rank and output extents, the source stride on each mapped axis, stride 0 on inserted axes and on mapped axes whose source extent is 1, and the source offset. The destination stays compact. |
| `reverse_kernel` (reverse the listed axes) | `t4a_cuda_materialize_strided` (or `t4a_cuda_strided_to_strided` when the destination is strided) with a negative stride on a reversed axis and the offset moved to that axis' far end. |
| `concatenate_copy_kernel` (write at `axis_offset` on one axis) | `t4a_cuda_contiguous_to_view` with the destination layout offset advanced by `axis_offset * dst.strides[axis]` and the input extents. |
| `scale_in_place_*` (`out[i] *= factor[0]`) | `t4a_cuda_binary` op `MUL` with `scalar_side = 2`, `out` aliasing `lhs`. |

This is the "consolidate duplicate helpers" rule of the migration design, and it
is why this slice needs no `axis_map`/broadcast plan struct. A caller that has a
baseline shape plus a `dims` map already builds an expanded shape to compute
broadcast strides; it hands the same information over as a layout.

The revision-1 `t4a_cuda_mul` stays the both-elementwise fast path: `MUL` in
`t4a_cuda_binary` is accepted only with `scalar_side != 0`, so no two entries
claim the same launch.

## 3. ABI additions (revision 6)

`T4A_CUDA_ABI_REVISION == 6`. Revisions 1-5 are unchanged. No new struct, no new
dtype tag.

```c
void t4a_cuda_unary(void* out, const void* in, int32_t dst_dtype, int32_t src_dtype,
                    int32_t op, int64_t len);

void t4a_cuda_binary(void* out, const void* lhs, const void* rhs, int32_t dtype,
                     int32_t op, int32_t scalar_side, int64_t len, unsigned int* err);

void t4a_cuda_compare(void* out, const void* lhs, const void* rhs, int32_t dtype,
                      int32_t cmp, int64_t len);

void t4a_cuda_select(void* out, const void* pred, const void* on_true, const void* on_false,
                     int32_t dtype, int64_t len);

void t4a_cuda_clamp(void* out, const void* in, const void* lower, const void* upper,
                    int32_t dtype, int64_t len);

void t4a_cuda_combined_strided(void* out, const void* a, const void* b, const void* coefficients,
                               t4a_cuda_layout out_layout, t4a_cuda_layout a_layout,
                               t4a_cuda_layout b_layout, int32_t dtype, int32_t op, int64_t len);
```

All six are flat launches over `len` logical elements (`block = 256`,
`grid = ceil(len / 256)`), except `t4a_cuda_combined_strided`, which decodes each
flat index through `out_layout.extents` and re-addresses the three operands.
`len == 0` launches nothing; a rank-0 layout has `len == 1`.

### 3.1 `t4a_cuda_unary`

`len` elements of `src_dtype` are read and `len` elements of `dst_dtype` written;
`dst_dtype != src_dtype` only where the table says so. Op tags are stable and
append-only.

| Tag | Op | `src → dst` | Semantics |
| --- | --- | --- | --- |
| 0 | `NEG` | F32→F32, F64→F64, I32→I32, I64→I64, C32→C32, C64→C64 | `-x`; integer negation wraps |
| 1 | `EXP` | F32, F64 | `expf`/`exp` |
| 2 | `LOG` | F32, F64 | natural logarithm |
| 3 | `SIN` | F32, F64 | |
| 4 | `COS` | F32, F64 | |
| 5 | `TANH` | F32, F64 | |
| 6 | `SQRT` | F32, F64 | |
| 7 | `RSQRT` | F32, F64 | `rsqrtf`/`rsqrt`, ~2 ulp, not `1/sqrt` |
| 8 | `EXPM1` | F32, F64 | `expm1f`/`expm1` |
| 9 | `LOG1P` | F32, F64 | `log1pf`/`log1p` |
| 10 | `ERF` | F32, F64 | `erff`/`erf` |
| 11 | `ABS` | F32, F64, I32→I32, I64→I64, C32→F32, C64→F64 | `fabs`; integers wrap on `INT_MIN`; complex is `hypotf`/`hypot` |
| 12 | `SIGN` | F32, F64, I32, I64, C32, C64 | float: NaN→NaN, **either zero → `+0`** (the baseline arm, not `f32::signum`), else `±1`; integer: `0`/`±1`; complex: `0` or `z/|z|` |
| 13 | `CONJ` | C32, C64 | imaginary part negated; `+0.0 → -0.0` |

An unsupported `(op, src, dst)` triple is rejected by the facade planner and is
caller UB at the raw entry, like every other tag mismatch in this ABI.

### 3.2 `t4a_cuda_binary`

| Tag | Op | `dtype` | `scalar_side` | Semantics |
| --- | --- | --- | --- | --- |
| 0 | `MAX` | F32, F64, I32, I64 | 0,1,2 | float: `lhs` if `lhs` NaN, else `rhs` if `rhs` NaN, else `max(lhs,rhs)`; integer: `lhs >= rhs ? lhs : rhs` |
| 1 | `MIN` | F32, F64, I32, I64 | 0,1,2 | as `MAX` with `min`/`<=` |
| 2 | `POW` | F32, F64 | 0,1,2 | `powf`/`pow` |
| 3 | `REM` | F32, F64 | 0,1,2 | `fmodf`/`fmod`: a zero result keeps the dividend's sign |
| 4 | `MUL` | F32, F64, I32, I64, C32, C64 | 1,2 only | `lhs*rhs`, integers wrap; `out` may alias `lhs` (in-place scaling) |
| 5 | `DIV_COMPLEX` | C32, C64 | 0 only | scale-robust Baudin–Smith division over parts, §3.3; `len` is the complex element count |
| 6 | `DIV_CHECKED` | I32, I64 | 0,1,2 | `y==0` → `err=1`, `out=0`; `y==-1` → wrapping `-x`; else `x/y` |
| 7 | `REM_CHECKED` | I32, I64 | 0,1,2 | `y==0` → `err=1`, `out=0`; `y==-1` → `out=0`; else `x - (x/y)*y` with wrapping multiply |
| 8 | `POW_CHECKED` | I32, I64 | 0,1,2 | `rhs < 0` → `err=1`, `out=0`; else square-and-multiply with wrapping multiply (`0^0 == 1`) |
| 9 | `MIXED_ADD` | F32, F64 | 1,2 | complex parts `+` real scalar, §3.4 |
| 10 | `MIXED_SUB` | F32, F64 | 1,2 | |
| 11 | `MIXED_MUL` | F32, F64 | 1,2 | |
| 12 | `MIXED_DIV` | F32, F64 | 1,2 | robust division with the real operand promoted to `(x, +0)` |
| 13 | `DIV` | F32, F64 | 1,2 only | ordinary componentwise division; the scalar side may be either operand; non-scalar float division stays with revision-1 `t4a_cuda_div` |

`scalar_side`: `0` = both operands elementwise (`len` elements each); `1` = `lhs`
is one element read at index 0; `2` = `rhs` is one element read at index 0. For
`MIXED_*`, `1` = the real operand is `lhs`, `2` = the real operand is `rhs`; the
real operand buffer holds one element, the complex operand buffer holds `2 * len`
components of the component float type, and **`out` also holds `2 * len`
components** (the `dtype` tag names the component type, not the output width).

`err` is one 4-byte, 4-byte-aligned word that the caller initializes to zero
before the first checked launch of a batch. Every thread that rejects its operands
stores `1` with `atomicExch`, so concurrent rejections are not a data race, and
the kernel never clears it: the word means "at least one element was invalid",
not "which one". That is deliberately not the revision-2 flag convention, which
starts at `UINT32_MAX` and records the lowest invalid position; the baseline's
checked integer ops record only the boolean, and a caller that wants the position
re-runs its own diagnostics. `err` is required for ops 6-8, must not overlap the
operands, and is untouched by the other ops (NULL is allowed there). Observation
and translation stay with the caller; no entry synchronizes.

### 3.3 Robust complex division

`DIV_COMPLEX` is the baseline `div_complex_parts` body: Baudin–Smith scaling with
`ab/cd` magnitude guards, the subnormal/near-overflow rescale by `2/eps^2`, the
`complex_div_term` numerator forms, and the sign-preserving zero result for a
finite numerator over an infinite divisor. `out`, `lhs`, `rhs` hold `2 * len`
interleaved components. `len` therefore counts complex elements while the buffers
hold twice that; the planner only rejects a `2 * len` overflow, and the doubled
capacity obligation is documented rather than checked (§5).

### 3.4 Mixed real/complex ordering

`MIXED_ADD/SUB/MUL` keep the baseline's unsimplified component expressions,
including the zero cross terms (`0 * im`, `0 * re`), because they decide NaN,
infinity, overflow and the sign of zero after the real operand is promoted to
`Complex(real, +0.0)`. Only `MIXED_DIV` uses the §3.3 division helper, with the
real operand promoted to `(x, +0)`; the add/sub/mul arms never call it.

### 3.5 `t4a_cuda_compare`, `t4a_cuda_select`, `t4a_cuda_clamp`

- `compare`: `cmp` ∈ {0 `EQ`, 1 `LT`, 2 `LE`, 3 `GT`, 4 `GE`}, `dtype` ∈ {F32, F64,
  I32, I64}; `out` is one byte per element (Bool). An unknown `cmp` writes 0, as
  the baseline's `_ => false` arm does.
- `select`: `pred` is one byte per element; `dtype` ∈ {F32, F64, I32, I64} (the
  baseline ternary dispatch rejects complex). Same-index aliasing of `out` with
  `on_true`/`on_false` is safe under the rule in §3.6; a partially overlapping
  predicate is not, because a thread's write can reach a byte another thread has
  not read.
- `clamp`: `dtype` ∈ {F32, F64}; `out = x < lo ? lo : (x > hi ? hi : x)` with
  ordered comparisons, so a NaN `x` propagates and a NaN bound disables that
  comparison. That literal expression is the reference the tests compare against;
  `f32::clamp` is not, because it panics on a NaN or inverted bound. Inverted
  bounds (`lo > hi`) are not rejected: the statement order decides, so a value
  below `lo` yields `lo` and any other value yields `hi`; this is documented rather
  than validated so a caller cannot fail late. **Deliberate:**
  the baseline relied on CubeCL's `clamp` expansion, whose NaN behaviour was never
  pinned; this ABI fixes the ordered form.

### 3.6 `t4a_cuda_combined_strided`

One logical domain (`len = product(out_layout.extents)`), decoded once per thread
through `out_layout.extents`; each operand address is
`offset + Σ coordinate[a] * strides[a]`.

| Tag | Op | `dtype` | Semantics |
| --- | --- | --- | --- |
| 0 | `MUL` | F32, F64, I32, I64, C32, C64 | `out = a * b` (the baseline broadcast multiply); integers wrap |
| 1 | `AXPBY` | F32, F64, C32, C64 | `out = coefficients[0] * a + coefficients[1] * b` |

- `out_layout` must be a destination (no stride-0 axis with more than one
  element); `a_layout` and `b_layout` may carry stride-0 broadcast axes and
  negative strides, and have their own offsets. All three layouts share the
  destination extents.
- `coefficients` is a two-element device buffer of `dtype`; `MUL` ignores it and
  may be given NULL, `AXPBY` requires it and reads both elements.
- **Aliasing rule (applies to every entry in this slice).** `out` may alias an
  input only when that input's layout — extents, strides and offset — is
  identical to `out_layout` and the element type matches (a narrower write, as in
  `ABS` C32→F32 or a Bool comparison destination, is always disjoint), so that each element is
  read by the same thread that writes it. Under that rule `AXPBY` may run in place
  on `b` (the baseline `axpby_strided_source` does) and `MUL` may scale in place
  through `a`; an unequal layout, a shifted overlap, or a broadcast/negative-stride
  input sharing the allocation is caller UB, because a thread's write can reach an
  element another thread has not read yet. The scalar operand of
  `t4a_cuda_binary` must be disjoint from `out`. All of `a`, `b` and
  `coefficients` must otherwise be distinct allocations.

## 4. Dtype matrices

| Entry | F32 | F64 | I32 | I64 | Bool | C32 | C64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `unary` | ✓ (op-dependent) | ✓ | neg/abs/sign | neg/abs/sign | – | neg/abs/sign/conj | neg/abs/sign/conj |
| `binary` | ✓ | ✓ | max/min/mul/checked | max/min/mul/checked | – | mul/div_complex | mul/div_complex |
| `compare` | ✓ | ✓ | ✓ | ✓ | out only | – | – |
| `select` | ✓ | ✓ | ✓ | ✓ | pred only | – | – |
| `clamp` | ✓ | ✓ | – | – | – | – | – |
| `combined_strided` | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |

Bool appears only as a byte payload (`compare` output, `select` predicate), which
matches the baseline's `u8` handling. Complex never appears in `compare`,
`select` or `clamp`, matching the baseline dispatch.

## 5. Facade

```rust
pub enum UnaryOp { Neg, Exp, Log, Sin, Cos, Tanh, Sqrt, Rsqrt, Expm1, Log1p, Erf, Abs, Sign, Conj }
pub enum BinaryOp { Max, Min, Pow, Rem, Mul, DivComplex, DivChecked, RemChecked, PowChecked,
                    MixedAdd, MixedSub, MixedMul, MixedDiv }
pub enum ScalarSide { Both, Left, Right }
pub enum CompareOp { Eq, Lt, Le, Gt, Ge }
pub enum StridedOp { Mul, Axpby }

pub fn plan_unary(op: UnaryOp, dst: DType, src: DType, len: i64) -> Result<Launch, Error>;
pub fn plan_binary(op: BinaryOp, dtype: DType, scalar_side: ScalarSide, len: i64) -> Result<Launch, Error>;
pub fn plan_compare(dtype: DType, len: i64) -> Result<Launch, Error>;
pub fn plan_select(dtype: DType, len: i64) -> Result<Launch, Error>;
pub fn plan_clamp(dtype: DType, len: i64) -> Result<Launch, Error>;
pub fn plan_combined_strided(op: StridedOp, dtype: DType, out: &Layout, out_len: i64,
                             a: &Layout, a_len: i64, b: &Layout, b_len: i64) -> Result<Launch, Error>;
```

The comparison tag needs no planner argument: every `CompareOp` serves every
accepted dtype, so the caller passes it straight to the kernel. No new `#[repr(C)]`
mirror is needed; the existing `CLayout` covers `combined_strided`. `Error` gains
`UnsupportedOp` for an undefined `(op, dtype, side)` triple. Each planner:

- rejects unsupported `(op, dtype)` pairs, including `MUL`/`DIV` with
  `scalar_side = 0` and the `DIV_COMPLEX` scalar-side restriction, with a typed
  `Error`;
- checks `len >= 0` and, for `combined_strided`, the layout span of all three
  operands and the equal-extents rule. The flat planners cannot see the caller's
  capacities, so they state the required spans instead of checking them: `out`
  and each elementwise operand hold `len` elements of their tag, a scalar operand
  holds one, `DIV_COMPLEX` and `MIXED_*` hold `2 * len` components (checked for
  overflow), `err` holds one 4-byte word, and `coefficients` holds two elements;
- requires `out` to be a destination layout for `combined_strided`;
- returns the flat launch geometry, unchanged from the other flat entries.

## 6. Numerical contract

- No `--use_fast_math`, no flush-to-zero; FMA contraction is the compiler default,
  so transcendental and complex results match a CPU reference within tolerance,
  not bit-for-bit. `REM` is `fmod`, not a `x - trunc(x/y)*y` rewrite.
- Integer add/sub/mul/neg wrap; `INT_MIN / -1` and `y == -1` in `REM_CHECKED` and
  `POW_CHECKED` keep the baseline's explicit arms instead of raising the
  hardware exception. Division by zero sets `err` instead of trapping.
- `MUL`/`AXPBY` complex arithmetic uses the ordinary component forms (the baseline
  does too); only division is scale-robust.
- `SIGN` on a float returns `+0.0` for either zero and the input for NaN, which is
  the baseline's explicit arm (`sign_float`) and not `f32::signum` (`-0.0 → -1.0`).
  The tests use the baseline rule as the reference and the doc says so.
- NaN: `MAX`/`MIN` propagate in `lhs`-then-`rhs` order; `SIGN` propagates; `CLAMP`
  propagates on `x`; `ABS` and the transcendentals follow the CUDA math library.

## 7. Tests

Planning (`tests/planning.rs`):

- the full `(op, dtype, scalar_side)` acceptance matrix per entry, and rejection
  of every unsupported combination, including `Bool` and complex where the table
  says "–";
- `DIV_COMPLEX`/`MIXED_*` element-vs-component lengths, including an overflowing
  `2 * len`;
- `combined_strided`: extents mismatch, broadcast axis in `out`, negative-stride
  `a`/`b`, offset out of span, empty domain;
- launch geometry for sizes across the block boundary, and `len == 0`.

GPU (`tests/gpu_elementwise.rs`, local A100 only, `--features gpu-tests`):

- every op against a CPU reference or an explicit residual, on a size above one
  block (for example 100_003) and on an empty and a single-element domain;
- `SIGN`/`ABS` signed-zero and NaN cases; `MAX`/`MIN` NaN ordering; `CLAMP` NaN and
  `±0`; `REM` dividend-signed zero;
- integer wrapping (`INT_MIN - 1`, `INT_MIN * -1`), `INT_MIN / -1` in
  `DIV_CHECKED`, `y == -1` in `REM_CHECKED`, `0^0`, negative exponent `err`,
  divide-by-zero `err` with the output zeroed, and the flag staying clear for a
  valid checked run;
- `DIV_COMPLEX` against the CPU `strided_basic::complex_div` port: `2/(1e38+1e38i)`,
  `(1e-320, 1e-320)`, infinite and NaN components, `±0` and subnormal results;
- `MIXED_ADD/SUB/MUL` byte-for-byte component comparison against the promoted CPU
  form, including the zero cross terms; `MIXED_DIV` against the robust CPU port
  with the same extreme cases as `DIV_COMPLEX`, since the ordinary promoted form is
  exactly what the baseline calls unsuitable there;
- `combined_strided`: broadcast multiply with extent-1 and stride-0 axes,
  negative-stride `a`, `AXPBY` in place on `b` with the baseline's coefficients,
  and a strided `a`/`b` pair that a compact BLAS-1 form cannot address;
- physical launch counts (one per entry call) through the harness counter.

## 8. Evidence to record

GPU model, driver, toolkit, the exact `cargo test -j 16 --features gpu-tests`
command and its result, plus the per-family numerical outcomes, in
`docs/worklogs/2026-10-21-elementwise-slice.md`. No hosted CI, no performance
comparison against CubeCL (R-ci-pr-60).

## 9. Attribution

`DIV_COMPLEX` and `MIXED_DIV` port tenferro-rs's `complex_div_parts_at` and its
helpers, which tenferro records as derived from Julia Base `complex.jl`
(`cdiv`, `robust_cdiv1/2`, `scaling_cdiv`, `scaleargs_cdiv`).
`THIRD_PARTY_NOTICES.md` currently defers that notice; this slice carries Julia's
MIT notice and copyright line over, names the port chain (Julia Base →
tenferro-rs → this crate) and states that the upstream revision is not recorded.
The `.cu` file keeps the per-function derivation notice.

## 10. Deliberate exclusions

- The linalg auxiliary kernels and RRQR primitives: next slice.
- The fused-expression emitter and the NVRTC prepare/cache path: still design-only.
- `broadcast_in_dim`, `reverse` and `concatenate_copy` as entries of their own: see
  §2. A consumer that wants a fused reverse-copy keeps using the movement entries.
- Bool arithmetic, complex comparison and complex `clamp`: not in the baseline.
- WebGPU/Metal-only `*_comptime_layout` twins: excluded by the migration design.
