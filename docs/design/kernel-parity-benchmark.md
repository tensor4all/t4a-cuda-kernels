# Paired kernel performance comparison against the CubeCL baseline

Status: protocol frozen 2026-10-21, before any candidate timing. Plan for the
performance-preservation gate of
[cubecl-performance-parity.md](cubecl-performance-parity.md) §5.

## 1. Question

Do the ported `t4a_cuda_*` kernels, driven by an equivalent minimal caller, run at
least as fast as the CubeCL kernels they replace, for the operations tenferro
dispatches to them today?

## 2. Pinned inputs

| Item | Value |
| --- | --- |
| Baseline source | tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`, worktree `~/tensor4all/tenferro-parity` (unmodified: no patches applied) |
| Baseline CubeCL | the fork pinned by that commit's `Cargo.toml` (`t4a-cubecl-*` crates) |
| Candidate | `t4a-cuda-kernels` at the revision-8 branch head, ABI revision 8 |
| Harness | `~/tensor4all/parity-bench` (standalone crate, own lockfile and target) |

The harness depends on `tenferro-gpu` (feature `cuda`) as a path dependency, so the
baseline kernels are the pinned sources compiled by the pinned CubeCL fork, and on
`t4a-cuda-kernels` as a second path dependency. Both arms run in one process, one
GPU, one stream per arm.

## 3. Arms

**Baseline** — one CUDA operation through
`CudaBackend::with_backend_session(|s| …)`, using the `*_into` form when the
backend trait has one (`add_read_into`, `mul_read_into`, `copy_read_into`,
`dot_general_read_into`, …) so the timed region excludes a destination allocation.
Where no `_into` form exists (`transpose_read`, `reduce_*_read`, `convert`,
`gather`, `scatter`, `slice`, `pad`) the baseline arm allocates its output inside
the timed region; that is recorded per case and never subtracted.

**Candidate** — one `plan_*` + `cuLaunchKernel` of the corresponding `t4a_cuda_*`
entry into a preallocated buffer, with arguments marshalled exactly as the GPU
tests do. A split reduction is two physical launches and is reported as one
operation with `launches = 2`.

## 4. Frozen case list (phase 1)

Every case runs on the A100 with the driver/toolkit recorded in the worklog.

| # | Case | Dtype | Size / shape | Baseline arm | Candidate entry |
| --- | --- | --- | --- | --- | --- |
| 1 | `add_f32_1e4` | F32 | 10 000, 1-D | `add_read_into` | `t4a_cuda_add` |
| 2 | `add_f32_1e6` | F32 | 1 000 000, 1-D | `add_read_into` | `t4a_cuda_add` |
| 3 | `add_c64_1e5` | C64 | 100 000, 1-D | `add_read_into` | `t4a_cuda_add` |
| 4 | `mul_f64_1e6` | F64 | 1 000 000, 1-D | `mul_read_into` | `t4a_cuda_mul` |
| 5 | `copy_f64_1e6` | F64 | 1 000 000, 1-D | `copy_read_into` | `t4a_cuda_copy_exact` |
| 7 | `convert_f64_f32_1e6` | F64→F32 | 1 000 000, 1-D | `convert` (allocates) | `t4a_cuda_convert` |
| 8 | `reduce_sum_f64_1e6_axis0` | F64 | 1 000 000, one output | `reduce_sum_read` (allocates) | `t4a_cuda_reduce` (+ partials) |
| 9 | `reduce_sum_f64_1000x1000_axis1` | F64 | 1 000 000, 1000 outputs | `reduce_sum_read` (allocates) | `t4a_cuda_reduce` |
| 10 | `reduce_max_f32_513_axis0` | F32 | 513, one output | `reduce_max_read` (allocates) | `t4a_cuda_reduce` |

Deferred, and reported as not run rather than inferred:

- **Transpose.** `transpose_read` is the allocating structural route and does not
  expose a kernel-only comparison through the public API; the candidate's
  `t4a_cuda_tiled_transpose` cannot be paired with it honestly. Deferred to a
  visibility-patched measurement, or the tenferro integration workload.
- **Indexing, scatter, linalg and RRQR.** The public backend reaches these only
  through allocating operations that also run vendor libraries or orchestration
  (cuSOLVER, the host step loop), so a kernel-level pair needs the internal
  `kernels` modules exposed in a local measurement patch. Phase 2.

## 5. Protocol

- Release build (`--release`), `-j 16`, one process, one GPU, no other GPU work.
- Operands uploaded and outputs validated once, outside the timed region; a
  mismatching result aborts the run.
- 3 warm-up iterations, then the median of 5 repetitions of each arm:
  - **batch-16**: 16 enqueues followed by one synchronize, reported per operation;
  - **round trip**: one enqueue plus one synchronize;
  - **sync floor**: one synchronize with nothing outstanding, for context.
- No timing barrier inside a batch; no per-iteration allocation on either arm
  except where §3 says the baseline allocates.
- Host wall-clock (`Instant`) on both arms, so both pay the same clock overhead and
  neither benefits from CUDA-event asymmetry.
- A 300 ms spin of one add launch precedes the first measured case so every case
  starts at the sustained device clock, and the whole campaign is run three times;
  per-case verdicts use all three runs.

## 6. Verdict rules (declared before running)

- A case **regresses** when the candidate's median batch-16 time exceeds the
  baseline's by more than 10 %, or when its median round-trip time does. A spread
  (max − min over the 5 repetitions) above 20 % of the median makes that case
  **inconclusive** and it is rerun whole.
- Any regression is reported with the candidate's launch count and the baseline's
  allocation caveat, and is not averaged away.
- A candidate win is reported as an upper bound whenever the baseline arm includes
  work the candidate arm does not (dispatch, allocation, vendor calls), with the
  two numbers stated separately.

## 7. Deliverables

`~/tensor4all/parity-bench/src/main.rs` (harness and case list, path and commit
recorded) and `docs/worklogs/2026-10-21-kernel-parity-benchmark.md` in this
repository with the environment, the per-case table, the verdict per case and the
cases not run.
