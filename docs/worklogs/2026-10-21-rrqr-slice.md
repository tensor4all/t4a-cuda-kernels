# Rank-revealing QR slice — local evidence

Date: 2026-10-21. Plan:
[slice-rrqr.md](../design/slice-rrqr.md).

## What landed

ABI revision 8 adds the eight entries that port
`tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs` (15 baseline symbols):

- `t4a_cuda_initialize_permutation` (`pos % n` over a flat `int64_t` array).
- `t4a_cuda_column_norms` with the baseline two-pass max-scale norm, one 32-lane
  plane per `(batch, column)` task, and the `-1` sentinel for a nonfinite magnitude
  or norm.
- `t4a_cuda_select_pivot` with the baseline's three-stage tie rule (largest norm,
  smaller original column, smaller column index) and the per-batch `status` word.
- `t4a_cuda_swap_columns` (matrix columns plus the permutation entries).
- `t4a_cuda_reflector` and `t4a_cuda_apply_reflector`, with `x0` broadcast from lane
  0 before any store and the complex dot rebuilt from two plane sums as the
  baseline does.
- `t4a_cuda_extract_r` and `t4a_cuda_prefix_rank`.

With this slice the custom-kernel inventory of the migration design is fully
ported. `initialize_q` stays revision 7's `identity_shifted`; the host step loop
stays with the consumer.

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
| `cargo test -j 16` | 72 planning tests pass |
| `cargo doc -j 16 --no-deps`, `cargo test -j 16 --doc` | clean |
| `cargo test -j 16 --features gpu-tests` | 103 pass, 0 failed, 0 ignored (17 new) |

## Coverage

`tests/gpu_rrqr.rs` (1955 lines) checks, per entry, against a CPU reference built
from the same layouts, with sentinel guards and the harness launch counter:

- `initialize_permutation` for `n` = 1, 3 and 7 over 1003 entries.
- `column_norms` for F32, F64, C32 and C64 against the max-scale 2-norm at
  `step = 0` and `step > 0`, including a zero column, a column mixing `1e30` and
  `1e-30`, a NaN column and an infinity column (the `-1` sentinel), and a domain
  above one block.
- `select_pivot` on hand-built norms: a clear maximum, the equal-norm tie broken by
  the smaller original column, the third tie stage on equal columns, a sentinel
  raising `status`, `step > 0`, and more than 32 columns so lanes wrap.
- `swap_columns` batched and strided, with a no-op pivot, a real swap, `step > 0` and
  above one block.
- `reflector` for F64, C32 and C64 with a zero column, a negative and a positive
  `x0`, and a complex phase; the defining identity (`H` applied to the original
  column is `alpha e_step`, `|alpha| = norm`) is checked numerically per case.
- `apply_reflector` as an involution for F64 and C64 with `first_column > 0`,
  several columns and above one block, plus a zero target column staying zero.
- `extract_r` on a batched strided work matrix, NaN payloads preserved bit for bit.
- `prefix_rank` for full rank, rank 0, a limited rank, `rtol` versus `atol`
  dominance, a NaN diagonal raising `status`, and more than 32 diagonals.
- One physical launch per call, zero for a validated empty plan.

A further review round added: a CPU reference for the *first* `apply_reflector`
application (real and a genuinely complex coefficient, so an empty body cannot pass
the involution test), the previously unlaunched dispatch routes (`swap_columns`
F32/I32/C32/C64, `apply_reflector` F32/C32, `extract_r` F32/C32), a nonempty slice
shorter than 32 rows for each plane body so a lane-local early return would fail, and
`prefix_rank` fixtures with an imaginary diagonal and a NaN tolerance.

## Findings

- The GPU suite found a real defect: the real-payload branch of
  `apply_reflector` multiplied the plane-summed dot product by the coefficient
  *slot index* instead of the coefficient value (the complex branch read the value
  correctly). The involution test caught it at `dot * 1` instead of `dot * coeff`;
  fixed, and both branches now load `coeff_values[coefficient]`.
- Implementation-time simplifications: the baseline's one-element imaginary-unit
  buffer is not part of this ABI (the complex dot is rebuilt from the two plane
  sums directly), and `x0` is broadcast from lane 0 before any store so no lane can
  read the overwritten diagonal.
- The plan review's four blockers (prefix-rank diagonal bounds, swap `step`,
  reflector step/coefficient bounds, apply-reflector row/batch/window agreement) and
  its flat-array capacity requirements are all reflected in the planners and their
  tests.

## Still not ported

- The fused-expression emitter and the NVRTC prepare/cache path.

No hosted CI was used; CUDA tests are local-only (R-ci-pr-60), and no performance
comparison against the CubeCL baseline is claimed.
