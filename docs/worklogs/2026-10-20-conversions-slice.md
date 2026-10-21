# Conversions slice — local evidence

Date: 2026-10-20. Plan: [slice-conversions.md](../design/slice-conversions.md).

## What landed

ABI revision 5 adds one entry, `t4a_cuda_convert`, with runtime `(dst_dtype,
src_dtype)` tags. The pair determines the semantics, so the ~20 per-dtype baseline
twins collapse into a single kernel and a single `plan_convert(dst, src, len)`
that accepts every unequal pair of the seven dtypes (42) and rejects identity
pairs (a same-dtype copy belongs to `copy_exact`).

Semantics: language casts (low-bit narrowing, sign-extending widening,
nearest-even floats, truncating float→integer), real→complex `(x, 0)`,
complex→real `re` only, componentwise C32↔C64, `x != 0` and `re != 0 || im != 0`
Bool predicates, and Bool→numeric/complex byte casts. Floating-component→signed
integer routes document the baseline's validator precondition and its acceptance
bounds, including the stride-2 real-component sampling for complex sources.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| CUDA toolkit | 12.6 installed; the tests invoke `nvcc` 12.0.140 from `PATH` |
| Test compile flags | `nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include` |
| Rust | 1.97.1 |

## Commands and results

```bash
cargo fmt --check                                                      # pass
cargo clippy -j 16 --all-targets --features gpu-tests -- -D warnings   # pass
cargo test -j 16                                                       # 44 planning tests pass, no CUDA
cargo test -j 16 --features gpu-tests                                  # 44 planning + 7 + 16 + 13 + 16 + 11 GPU tests pass
```

GPU coverage: F64→F32 nearest-even and back over 65 537 elements; I64→I32
low-bit narrowing over the same length; float→integer truncation toward zero
(`-1.75 → -1`, `-0.5 → 0`, `4.999999 → 4`); integer→float nearest-even at the F32
precision boundary; the routes the first fixtures did not reach (I32→I64/I32→F32/
I32→F64/I64→F64/I64→I32, F64→I64 including the representable predecessor of 2^63,
F64→F32 with a subnormal, C64→C32, C64→I32, I32→C32, Bool→F64) each against a CPU
reference; real→complex with a signed zero and a zero imaginary part;
complex→real ignoring a NaN/infinite imaginary part; Bool destinations for
`-0.0`, `0.0`, NaN, Infinity and complex real-only/imaginary-only/zero sources;
empty (zero launches), length 1, 257 with a sentinel guard; C32 and C64 input
subspans at base+4 and base+8 proving the scalar-alignment claim; and the
validator-then-convert path using the real-cast validator with the I32 bounds (a
fraction accepted, `4.29e9` rejected at its position, the corrected fixture
validated and then converted), plus a complex case where the real components
validate while an imaginary value is nonfinite.

## Findings

- Two test fixtures were wrong before they were right: the float-index validation
  launch had the flag buffer where the index buffer belongs, and the
  `validate_real_cast` re-initialization reused the validation argument slots for
  the initialization call. Both produced "valid" flags for invalid data, which is
  exactly the failure mode the pre-review warned about for launch-argument
  marshalling; the assertions on the reported flag position caught them.

## Still not ported from the inventory

Single authoritative list (the README and the slice plan point here):

- `structural.rs` movement extras: `broadcast_in_dim`, `reverse`,
  `concatenate_copy`, `scale_in_place_float/complex`.
- `elementwise.rs` families revision 1 did not take: broadcast multiply,
  `axpby_strided_source`, unary transcendentals/analytica (`exp`, `log`, `sin`,
  `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, `log1p`, `erf`), `neg`, `abs`, `sign`,
  `maximum`/`minimum`, non-scalar `pow`/`rem`, scalar div/pow/rem and
  mixed real/complex operators, checked integer div/rem/pow, compare/select/clamp,
  `conj`, and the robust complex division.
- The linalg auxiliary kernels and RRQR primitives.
- The fused-expression emitter, and the NVRTC prepare/cache path.

No hosted CI was used; CUDA tests are local-only, and no performance comparison
against the CubeCL baseline is claimed.
