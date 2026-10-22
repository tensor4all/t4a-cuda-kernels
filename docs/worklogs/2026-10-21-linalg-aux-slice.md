# Linalg auxiliary slice — local evidence

Date: 2026-10-21. Plan:
[slice-linalg-aux.md](../design/slice-linalg-aux.md).

## What landed

ABI revision 7 adds eleven entries for the supporting kernels tenferro's CUDA
linalg paths run around its vendor calls, collapsing 21 baseline entry symbols:

- `t4a_cuda_fill_one`, `t4a_cuda_matrix_adjoint` (conjugating for the complex
  dtypes), `t4a_cuda_householder_explicit_v`, `t4a_cuda_identity_shifted` (the
  batched RRQR identity at `start = 0` and the shifted Q-column identity above it).
- `t4a_cuda_qr_phase` and `t4a_cuda_qr_apply_phase`, with the phase operand
  carrying one fewer axis than the matrix (its axis 0 is the diagonal index).
- `t4a_cuda_upper_trapezoidal_violation` and `t4a_cuda_householder_assemble`.
- `t4a_cuda_lu_extract_outputs`, `t4a_cuda_lu_parity` and
  `t4a_cuda_lu_apply_pivots`, with the pinned `P = [m,m]`, `L = [m,k]`, `U = [k,n]`,
  `k = min(m,n)` relationships and 1-based `int32_t` pivots.

Two baseline symbols are deliberately not ported: `complex32_magnitude` /
`complex64_magnitude` have no caller in the pinned source, and revision 6's
`unary`/`ABS` already performs the operation. The rank-revealing QR primitives are
the next slice.

Provenance: `cuda/src/linalg_aux.cu` names every baseline function and line.
Unlike the baseline, which drops the operand offset in its explicit stride
arithmetic, every entry here honours the layout offset; the header says so.

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
| `cargo test -j 16` | 62 planning tests pass (pivot-capacity and exact L/U cases included) |
| `cargo doc -j 16 --no-deps`, `cargo test -j 16 --doc` | clean |
| `cargo test -j 16 --features gpu-tests` | 86 pass, 0 failed, 0 ignored (12 new) |

## Coverage

`tests/gpu_linalg_aux.rs` (2600 lines) checks, per entry, against a CPU reference
built from the same layouts, with a sentinel guard behind every output:

- `fill_one` on all seven dtypes, a strided 59x17 view with an offset above one
  block (every addressed element one, every unaddressed element untouched), and an
  empty extent.
- `matrix_adjoint` for F64, I32 and C64 (conjugation) with an asymmetric strided
  operand and a batched rank-3 case.
- `householder_explicit_v` from a strided packed 4x6 into a 4x3 output and into a
  full-width one; `identity_shifted` at `start` 0, 2 (zero complement column) and a
  batched rank-3 complex case.
- `qr_phase` for F64 and C64 with positive, negative, `+0`, `-0`, zero-magnitude,
  pure-imaginary and NaN diagonals, compared bit for bit; `qr_apply_phase` with
  `q_start > 0`, a complement column left untouched, F64 and C64 (R conjugated),
  batched, checking both operands.
- `upper_trapezoidal_violation` for F32, I64 and C64 (a complex value with a zero
  real part counts), NaN and negative zeros in the upper part not flagged, above
  one block.
- `householder_assemble` at `factor_width` 0, 1 and full width with narrower
  sources, plus the flat coefficient prefix and zero tail.
- `lu_extract_outputs` and `lu_parity` on 1x1, 2x2 and a pivot-heavy 8x5
  factorisation: P is a permutation matrix, L unit lower triangular, U upper
  triangular, and the parity matches a direct count of pivot moves and the
  standalone entry; batched and strided.
- `lu_apply_pivots` forward, inverse, and the round trip, for F32, I64 and C64,
  batched with a strided input.
- One physical launch per call through the harness counter, zero for a validated
  empty plan.

The second review round's test findings are covered as well: every entry now has a
domain above one 256-thread block (adjoint 640, explicit-V 360, identity 280,
phase 320, apply-phase 800, assemble 500, LU extraction 800, pivot application 800,
and 300 batches for the standalone parity), the in-place Q/R operands are uploaded
through a guarded buffer with sentinel holes so their guards and unaddressed
elements are actually asserted, every GPU dtype route of every entry is exercised
(20+ routes that had no dispatch coverage), the phase fixtures include
near-overflow and subnormal complex diagonals, the apply-phase case has R larger
than Q, the assemble case has more coefficients than matrix elements and a
`factor_width` above both dimensions, the violation case puts a NaN below the
diagonal, and the copy contracts compare raw bits instead of accepting any NaN
payload.

A third review round asked for four more cases, now in place: a rank-4 adjoint
(two batch axes, so a body that dropped axes above axis 2 fails), a reversed
source for the pivot application (negative row stride and a shifted offset), a
strided/offset P, L and U for the LU extraction case, and bitwise comparison of the
untouched complement columns. It also produced the last pivot-bound fix above.

## Findings

- The post-implementation review found two address-safety bugs in the first
  submitted state, both now fixed:
  1. Every flat entry used the *allocation* length as its logical tail bound while
     the planner launched over the *extent product*. A strided operand with spare
     capacity therefore wrapped coordinates past its logical end and raced earlier
     threads. The kernels now derive every bound from the layout
     (`logical_count`), and the allocation lengths were removed from the C
     signatures: they are a host-side validation input for `plan_*`, never a kernel
     bound. Empty operands in a multi-output launch can no longer reach a zero
     extent, because each branch is guarded by its own layout count.
  2. `lu_extract_outputs` bounded its parity branch by the parity allocation length
     instead of `batches`, so a larger parity buffer made the kernel read pivots it
     was not given. The branch is now bounded by `batches`.
- The same review found that the pivot-array bound was unvalidated:
  `plan_lu_parity(F64, 2, i64::MAX)` planned a kernel that walks an absent pivot
  array. Both LU planners now take the pivot-array length and require
  `k.checked_mul(batches) <= pivots_len`. `L` and `U` are also required to be
  exactly `[m, k]` and `[k, n]` as the design states, and the four entries that
  serve the six numeric dtypes are in that manifest arm instead of advertising
  Bool and silently doing nothing.
- A second review round on the fixed state found one remaining hole: 
  `plan_lu_apply_pivots` could not see the pivot array, so a batched permutation
  accepted one pivot for two batches. It now takes `pivots_len` and requires
  `k * product(batch extents)` pivots. The same round added a pivot *byte-span*
  check to both LU planners (an element count alone let a byte address wrap), and
  found a planning test whose oversized-`L` case also passed an invalid `P`, so the
  `L` check was untested; the case now varies one operand at a time.
- No product defect was found by the GPU suite itself; the two safety gaps found
  during implementation were closed before that run: `k` must equal `min(rows, cols)`
  in `t4a_cuda_lu_extract_outputs` (a larger `k` let a `U` row address outside
  `work`), and both in-place operands of `t4a_cuda_qr_apply_phase` are now required
  to be destinations.
- The plan review rejected the first draft's LU pivot type (`int64_t`): the pinned
  caller stores `int32_t` 1-based pivots, and reading that buffer as `int64_t`
  would combine adjacent pivots and read past its allocation.
- The plan review also removed this slice's first-draft status words (`int64_t`
  nonfinite flags) because none of the eleven baseline kernels reports nonfinite
  input; the convention belongs to the RRQR slice that needs it.

## Still not ported

- The rank-revealing QR primitives (`rank_revealing_qr.rs`) and `extract_r`.
- The fused-expression emitter and the NVRTC prepare/cache path.

No hosted CI was used; CUDA tests are local-only (R-ci-pr-60), and no performance
comparison against the CubeCL baseline is claimed.
