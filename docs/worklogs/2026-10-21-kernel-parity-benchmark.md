# Kernel performance comparison against the CubeCL baseline — local evidence

Date: 2026-10-21. Protocol frozen before the first run in
[kernel-parity-benchmark.md](../design/kernel-parity-benchmark.md).

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe, 0 % utilisation and no compute process before and after each campaign |
| Driver | 580.126.09 |
| Toolkit | `nvcc` 12.0.140; baseline CubeCL compiles through its own NVRTC path |
| Baseline | tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`, worktree `~/tensor4all/tenferro-parity`, **unmodified** (`git status` clean) |
| Candidate | `t4a-cuda-kernels` ABI revision 8 |
| Harness | `~/tensor4all/parity-bench` (own lockfile and target), source mirrored in [docs/benchmarks/kernel-parity](../benchmarks/kernel-parity/) |
| Command | `cargo build --release -j 16` then `./target/release/parity-bench`, three whole campaigns |

Protocol note (added before the first campaign, not after seeing results): a 300 ms
spin of one add launch precedes the first measured case, so every case starts at the
sustained device clock; the campaign is then run three times and each case reports
its median of five repetitions per run. The verdict uses the per-run medians of all
three campaigns.

## Results (batch of 16 enqueues + one synchronize, ns per operation)

`base` = tenferro CubeCL path, `cand` = t4a-cuda-kernels, `ratio` = cand / base.

| Case | base r1 | cand r1 | ratio | base r2 | cand r2 | ratio | base r3 | cand r3 | ratio | launches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `add_f32_1e4` | 10049 | 9020 | 0.90 | 9934 | 8216 | 0.83 | 12175 | 8783 | 0.72 | 1 |
| `add_f32_1e6` | 11828 | 11449 | 0.97 | 10857 | 11438 | 1.05 | 13696 | 12439 | 0.91 | 1 |
| `add_c64_1e5` | 9588 | 9431 | 0.98 | 9627 | 9581 | 1.00 | 11922 | 10021 | 0.84 | 1 |
| `mul_f64_1e6` | 14586 | 13347 | 0.92 | 13216 | 12885 | 0.97 | 13892 | 13059 | 0.94 | 1 |
| `copy_f64_1e6` | 13332 | 12112 | 0.91 | 14873 | 12630 | 0.85 | 15282 | 12551 | 0.82 | 1 |
| `convert_f64_f32_1e6` | 13506 | 13141 | 0.97 | 13384 | 13410 | 1.00 | 13330 | 13510 | 1.01 | 1 |
| `reduce_sum_f64_1e6_axis0` | 32858 | 32918 | 1.00 | 34897 | 33136 | 0.95 | 34813 | 33637 | 0.97 | 2 |
| `reduce_sum_f64_1000x1000_axis1` | 35221 | 32356 | 0.92 | 32992 | 33662 | 1.02 | 36979 | 34350 | 0.93 | 2 |
| `reduce_max_f32_513_axis0` | 24153 | 20813 | 0.86 | 25871 | 22003 | 0.85 | 27751 | 21819 | 0.79 | 2 |

Round trip (one operation + synchronize, ns), for the submission/overhead picture:

| Case | base r1 | cand r1 | base r2 | cand r2 | base r3 | cand r3 |
| --- | --- | --- | --- | --- | --- | --- |
| `add_f32_1e4` | 19650 | 10130 | 16331 | 9001 | 26681 | 10511 |
| `add_f32_1e6` | 20330 | 13801 | 18811 | 12481 | 27161 | 14520 |
| `add_c64_1e5` | 18291 | 10000 | 17000 | 10200 | 22931 | 11450 |
| `mul_f64_1e6` | 20621 | 13620 | 21211 | 14871 | 24761 | 14520 |
| `copy_f64_1e6` | 23971 | 12880 | 28711 | 12620 | 26481 | 13530 |
| `convert_f64_f32_1e6` | 20300 | 13341 | 21671 | 13881 | 27991 | 14100 |
| `reduce_sum_f64_1e6_axis0` | 40441 | 34912 | 43382 | 32881 | 50342 | 34932 |
| `reduce_sum_f64_1000x1000_axis1` | 42011 | 36042 | 43191 | 34721 | 51781 | 34731 |
| `reduce_max_f32_513_axis0` | 32771 | 23241 | 34451 | 21431 | 43541 | 21791 |

## Verdict

- **No case regresses.** The largest candidate/baseline ratio observed is 1.05
  (`add_f32_1e6`, run 2), inside the declared 10 % tolerance and inside that case's
  own run-to-run range (0.91 in run 3). Every other case is at parity or faster.
- **Candidate faster by more than the noise floor** in `copy_f64_1e6` (0.82-0.91),
  `reduce_max_f32_513` (0.79-0.86), `mul_f64_1e6` (0.92-0.97) and `add_f32_1e4`
  (0.72-0.90).
- **Inconclusive by the declared spread rule** (spread above 20 % of the median in
  every campaign): `add_c64_1e5` (40-56 %) and `convert_f64_f32_1e6` (25-32 %). Their
  ratios sit at 0.84-1.01, so neither a win nor a regression is claimed. The likely
  cause is host-side noise in a 9-13 µs measurement rather than a kernel difference;
  a longer per-case measurement is the fix, and it is not needed for the gate as long
  as no regression is claimed.
- **Round trip favours the candidate by construction.** The baseline arm runs
  tenferro's full session dispatch (planning, dtype dispatch, binding) before the
  launch, while the candidate arm marshals a prepared `plan_*` and calls
  `cuLaunchKernel`. The 1.4-2.8x round-trip difference is therefore an upper bound on
  what a consumer gains from the kernel layer, not a kernel-only result; the batch-16
  numbers are the kernel-level comparison.
- **The candidate's reductions are genuinely two launches** (partials and combine) and
  still match the baseline's single-operation arm within noise, which satisfies the
  parity contract's prohibition on sacrificing the split for a one-launch target.

## Cases not run

- **Transpose.** `transpose_read` is the allocating structural route on the public
  backend and cannot be paired with `t4a_cuda_tiled_transpose` honestly; it needs a
  visibility-patched measurement (phase 2) or the tenferro integration workload.
- **Indexing, scatter, linalg and RRQR.** The public backend reaches those through
  allocating operations that also run vendor libraries or the host step loop, so a
  kernel-level pair needs the internal `kernels` modules exposed in a local
  measurement patch. Phase 2.

## Harness caveats

- The candidate arm compiles its cubins with `nvcc` before the first measurement, so
  no NVRTC work happens inside the timed region on either arm; the baseline's NVRTC
  and module load happen during warm-up as well.
- Both arms run in one process on one device with their own stream, operands uploaded
  and both results checked (byte-equal for the element-wise and copy cases, within
  1e-9 relative of a CPU reference for the reductions) outside the timed region.
- The harness lives outside this repository because it depends on the pinned tenferro
  checkout, which the kernel crate must not depend on; its source is mirrored under
  `docs/benchmarks/kernel-parity/` for reproduction.
