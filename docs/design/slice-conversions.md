# Slice E: dtype conversions

Status: implemented and GPU-validated, 2026-10-20 (ABI revision 5). Fifth implementation slice of
[the migration design](cubecl-kernel-port.md).

Baseline source: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
`crates/tenferro-gpu/src/kernels/structural.rs` conversion entries and the
caller's cast path in `crates/tenferro-gpu/src/cubecl/{mod,ops}.rs`.

## 1. Scope

One entry with runtime `(dst_dtype, src_dtype)` tags covers the baseline's ~20
per-dtype conversion twins. The pair determines the semantics, so a single entry
is enough; there is no execution difference to keep apart.

| Baseline symbols | Dtype pairs |
| --- | --- |
| `convert_float_to_float`, `convert_numeric`, `convert_bool_to_numeric` | numeric↔numeric (F32/F64/I32/I64), Bool→numeric |
| `convert_complex_raw` (the kernel the caller's `convert_complex_to_complex` launches over `2*n` scalars) | C32↔C64, componentwise |
| `convert_numeric_to_complex_raw`, `convert_bool_to_complex_raw`, `convert_{f32,f64}_to_{c32,c64}_raw` | F32/F64/I32/I64/Bool → C32/C64 |
| `convert_complex_to_numeric`, `convert_complex_raw_to_bool`, `convert_{c32,c64}_to_{f32,f64}` | C32/C64 → F32/F64/I32/I64/Bool |
| `convert_numeric_to_bool` | F32/F64/I32/I64 → Bool |

Together these cover all 42 unequal pairs of the seven dtypes with a meaningful
conversion. Identity pairs are rejected: a same-dtype copy belongs to
`t4a_cuda_copy_exact`.

Still unported from the same source file, and therefore *not* claimed as
completed by this slice: `broadcast_in_dim`, `reverse`, `concatenate_copy` and
`scale_in_place_*` (structural movement extras), and the elementwise families
that revision 1 did not take (broadcast multiply, unary transcendentals, scalar
and mixed real/complex operators, checked integer division/remainder/power,
compare/select/clamp, sign, extrema, complex division). Those are a separate
slice.

## 2. ABI additions (revision 5)

```c
void t4a_cuda_convert(void* out, const void* in, int32_t dst_dtype, int32_t src_dtype,
                      int64_t len);
```

Flat (`block = 256`, `grid = ceil(len/256)`), `len` logical elements, source
capacity `len * sizeof(src)`, destination capacity `len * sizeof(dst)`.

### Floating→signed-integer precondition

C++ `static_cast<float→int>` is undefined for NaN, infinity and out-of-range
values. The caller must run the revision-2 validator on **every**
floating-component→signed-integer route before converting, and only convert when
validation reported no invalid position:

| Route | Buffer length | Stride | Baseline acceptance |
| --- | --- | --- | --- |
| F32/F64 → I32 | `len` | 1 | finite, `[-2^31, 2^31)` for F32 sources and `[-2^31, 2^31-1]` for F64 |
| F32/F64 → I64 | `len` | 1 | finite, `[-2^63, 2^63)` |
| C32/C64 → I32/I64 | `2 * len` | 2 | as above on the real component; the imaginary component is not read by the cast and lies in the skipped stride slots |

These bounds are the baseline's own acceptance policy (`F::bounds`), which is
stricter than pure truncation safety (for example `F64 2147483647.5 → I32` is
rejected even though truncation would fit). Bool destinations and Bool sources
have no such precondition.

The kernels do not re-validate, and the planner cannot inspect values.

### Aliasing

Source and destination byte spans must be disjoint: a conversion can change the
element width, and overlapping spans let one thread overwrite bytes another has
not read. The caller proves the allocation and alias conditions; the planner
checks the metadata only.

## 3. Facade additions

`plan_convert(dst_dtype, src_dtype, len)` returns a flat launch and rejects
identity pairs with `ShapeMismatch`, negative lengths, and grid limits; every
representable unequal pair is supported, so it has no unsupported-pair branch.
`ABI_REVISION = 5`; `Entry::Convert`. The cast and precondition semantics are
canonical in the ABI document; this plan does not restate them.

## 4. Tests

CPU: the full 42-pair matrix accepted, identity pairs and Bool→Bool rejected,
negative length, grid limit, manifest/ABI consistency.

GPU (A100), each pair against a CPU reference that implements the documented
rules (truncation toward zero, nearest-even widening, low-bit narrowing):

- numeric↔numeric: F32↔F64 (including rounding ties and subnormals), I32↔I64,
  I32→F32/F64, F32/F64→I32/I64 with fixtures inside the validated domain
  (including `-1.75 → -1`, the representable predecessor of `2^63`, and values
  just below an integer boundary); never a NaN/Inf/out-of-range float→int cast.
- C32↔C64 (both components), Bool→numeric, numeric→complex (imaginary zero),
  numeric→Bool (`-0.0`, `0.0`, NaN and Inf), complex→Bool (real only, imaginary
  only, both zero), complex→numeric (real part) where the imaginary part is
  nonfinite and must be ignored.
- Sizes 0, 1 and `65_537` (a partial block), with sentinel-guarded
  destination allocations compared in full; complex subspans aligned to 4/8 but
  not 8/16 bytes; zero launches for an empty conversion and one otherwise.
- The integer-cast precondition exercised end to end: validate a 2^16-element
  F64→I32 domain (including an invalid value late in the buffer) with
  `t4a_cuda_validate_real_cast`, then convert the valid fixture and read back the
  flag; and a complex case where the real components are valid while an imaginary
  value is nonfinite.

## 5. Provenance

tenferro-rs translation notice in `cuda/src/convert.cu` and the host planner
region. No additional third-party notice applies to these entries.

## 6. Acceptance

- Local gates (fmt, clippy, CPU tests) pass without CUDA; GPU tests pass on the
  local A100 with hardware/driver/toolkit/command recorded. CUDA tests stay
  local-only.
- ABI document, manifest and `.cu` symbols agree.
- [Review checklist](../review-checklist.md) sections 1–7 satisfied.
