# Elementwise slice — local evidence

Date: 2026-10-21. Plan:
[slice-elementwise.md](../design/slice-elementwise.md).

## What landed

ABI revision 6 adds six entries covering the remaining elementwise inventory and
the strided combination, collapsing 46 baseline `(operation, dtype)` entry symbols
into runtime tags:

- `t4a_cuda_unary` — 14 op tags (neg, ten transcendentals, abs, sign, conj), with
  `C32→F32` / `C64→F64` as the only unequal dtype pair (complex magnitude).
- `t4a_cuda_binary` — 14 op tags: NaN-ordered extrema, `powf`/`fmod`, scalar-side
  multiply and float division, the two scale-robust complex division routes
  (parts and mixed real/complex), and the checked integer division, remainder and
  power with a sticky device status word.
- `t4a_cuda_compare`, `t4a_cuda_select`, `t4a_cuda_clamp`.
- `t4a_cuda_combined_strided` — broadcast multiply and strided axpby over one
  logical domain with three operand layouts.

Four baseline kernels needed no entry of their own: `broadcast_in_dim`, `reverse`
and `concatenate_copy` are expressed with a caller-computed layout (expanded
broadcast source, negative stride with the far-end offset, shifted destination
view) and `scale_in_place_*` is the binary scalar multiply with `out` aliasing
`lhs`. `tests/planning.rs` proves each of those mappings instead of asserting it
in prose.

Provenance: `cuda/src/elementwise.cu` names the baseline functions and lines; the
complex division helpers in `cuda/include/t4a/abi.cuh` carry the Julia Base
derivation, and `THIRD_PARTY_NOTICES.md` now reproduces Julia's MIT notice (the
previous revision deferred it).

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| Toolkit | `nvcc` 12.0.140 (`/usr/bin/nvcc`), harness default `-arch=sm_80` |
| Rust | `cargo` 1.97 toolchain, `-j 16` |

## Commands and results

| Command | Result |
| --- | --- |
| `cargo fmt --check` | clean |
| `cargo clippy -j 16 --all-targets -- -D warnings` | clean |
| `cargo test -j 16` | 51 planning tests pass |
| `cargo doc -j 16 --no-deps`, `cargo test -j 16 --doc` | clean |
| `cargo test -j 16 --features gpu-tests` | 11 `gpu_elementwise` + 63 pre-existing GPU tests pass, 0 failed, 0 ignored |

The full GPU run after the final fix: `gpu_convert` 7, `gpu_elementwise` 11,
`gpu_gather_scatter` 16, `gpu_kernels` 13, `gpu_reduce` 16, `gpu_slice_b` 11.

## Coverage

`tests/gpu_elementwise.rs` (2231 lines) checks, per entry, against a CPU
reference or an explicitly stated residual:

- unary: all 14 tags on 100_003 elements in f64 (relative 1e-13) and f32
  (1e-5, `rsqrt` ~4 ulp), ERF past `|x| > 2.2`, bit-exact NEG/ABS/CONJ with both
  zeros, NaN and infinities, SIGN's either-zero `+0` and NaN, integer wrapping on
  `i32::MIN`/`i64::MIN`, complex magnitude and complex sign, and len 0/1 domains.
- binary: NaN-in-lhs-then-rhs extrema order, plain integer order, `powf` including
  `0^0 == 1` and `(-2)^0.5`, `fmod` with the dividend-signed zero, scalar division
  both sides, scalar multiply in place, complex multiply with the scalar side,
  `INT_MIN / -1`, `y == -1` remainder, negative-exponent and divide-by-zero
  rejection through the flag (and the flag staying clear on a valid run).
- complex division: the scaled Baudin–Smith algorithm against a CPU port for
  `2/(1e38+1e38i)`, subnormal operands, an infinite divisor with a finite
  numerator (signed zeros), and NaN components; mixed real/complex add, sub, mul
  byte-identical to the promoted CPU form including the zero cross terms.
- compare, select and clamp: every tag/dtype, NaN, infinity, signed zero, a NaN
  bound and inverted bounds.
- combined strided: broadcast multiply over a stride-zero operand, a
  negative-stride operand, axpby in place on `b`, integer wrapping, complex
  multiply and axpby, and a domain above one block.
- one physical launch per entry call is asserted through the harness counter, and
  the empty-domain case asserts zero launches.

## Findings

- The parallel test author found a real defect the plan review had not: the complex
  multiply branch of `t4a_cuda_binary` ignored `scalar_side`, so `MUL` with
  `scalar_side = 1/2` read `rhs[index]` instead of `rhs[0]`. Fixed in
  `binary_complex_body`; the case is now part of the suite.
- The pre-implementation review's inverted-bounds sentence was wrong in three
  places (the value above an inverted upper bound is `hi`, not `lo`); the ordered
  expression, not the prose summary, is the contract, and the tests already
  asserted it.
- Two planning assertions in the slice's own first draft were wrong about the
  facade (a doubled length that the launch limit rejects before the overflow, and a
  reversed-layout offset outside its allocation). Both were corrected rather than
  accommodated by the code, and the mapping test for the four consolidated kernels
  was added afterwards.
- The post-implementation review found coverage gaps rather than contract errors:
  compare/select/clamp never exceeded ten elements, the F32, I64 and C32
  strided-combine routes and `MIXED_DIV` on F32 were unexercised, and the mixed
  component comparisons accepted `+0` for `-0`. All four are now covered
  (100_003-element compare/select/clamp fixtures, bit-exact mixed comparisons),
  and the review's simplification — one coordinate walk per element instead of
  three decodes — is in the kernel.
- A second review round on the same diff added: AXPBY on F32 and C32 with a
  strided, offset output and shifted `b`; both-NaN extrema cases with distinct
  payloads compared by bits (so a reversed NaN priority would fail); tail-guard
  assertions inside the shared launch helpers; and the per-file provenance notice
  in the test source. It also caught two documentation errors, now fixed: the
  `combined_strided` prototype repeated the pointer names as its layout
  parameters, and the aliasing rule needed the matching-element-type restriction
  (a narrower write such as `ABS` C64→F64 or a Bool comparison destination must be
  disjoint even when the element counts match).
- Deliberate decisions recorded in the design: `SIGN` returns `+0` for either zero
  (the baseline arm, not `f32::signum`); `clamp` uses the ordered expression, so a
  NaN input propagates and a NaN bound disables its comparison, where the baseline
  left it to CubeCL's expansion; the checked-op flag is a sticky boolean written
  with `atomicExch`, not the revision-2 lowest-position flag.

## Still not ported

- The linalg auxiliary kernels and RRQR primitives (next slice).
- The fused-expression emitter and the NVRTC prepare/cache path.

No hosted CI was used; CUDA tests are local-only (R-ci-pr-60), and no performance
comparison against the CubeCL baseline is claimed.
