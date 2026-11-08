# Worklog: vectorized pass-1 lane walk (tenferro-rs #2406, part 2)

Date: 2026-11-08. Plan and outcome:
[slice-reduction-pass1-vector.md](../design/slice-reduction-pass1-vector.md).

## 1. Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe, compute capability 8.0, driver 580.178.04 |
| Toolkit | CUDA 12.6 (`nvcc` V12.6.85); kernels compiled with `-std=c++17 -arch=sm_80`; no fast math, no FTZ |
| Host | primerose; measured processes pinned to CPUs 0-7, one GPU job at a time under the host's bench-slot lock, GPU otherwise idle |
| Baseline | `31b734e`: library `e1f56e4` plus this branch's harness-only commit |
| Candidate | this branch |
| Through tenferro | tenferro-rs `a09c8d72` (`origin/main`), tenferro-benchmark `c3cc783`, PyTorch 2.12.0+cu130 |

**Theoretical bandwidth.** 1935 GB/s, the A100 80GB PCIe's HBM2e specification.

**Bench binaries.** The bench binary compiles `cuda/src/*.cu` at run time from its own crate
directory, so each arm was built in its own worktree. Copying a binary would have run the
other side's kernels.

## 2. Pass 1 alone: device time (nsys)

- Each process ran 100 launches in the profiled region, and the table reports the median kernel
  time.
- Bandwidth is the input bytes divided by the pass-1 time.
- Command per case:
  `nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop <bench_kernels> <case>`,
  followed by `nsys stats --report cuda_gpu_kern_sum`.

| case | pass 1 before µs | GB/s | pass 1 after µs | GB/s | ratio | after / 1935 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 8.9 | 471 | 8.1 | 518 | 0.910 | 27 % |
| `reduce-max-f32-full-1048576` | 10.5 | 401 | 8.9 | 470 | 0.853 | 24 % |
| `reduce-min-f32-full-1048576` | 10.4 | 402 | 8.9 | 470 | 0.856 | 24 % |
| `reduce-sum-f64-full-1048576` | 9.0 | 933 | 8.4 | 1004 | 0.929 | 52 % |
| `reduce-max-f64-full-1048576` | 10.5 | 798 | 8.9 | 943 | 0.846 | 49 % |
| `reduce-min-f64-full-1048576` | 11.0 | 762 | 9.0 | 928 | 0.821 | 48 % |
| `reduce-sum-f32-full-16777216` | 60.9 | 1102 | 46.7 | 1438 | 0.766 | 74 % |
| `reduce-max-f32-full-16777216` | 149.4 | 449 | 49.6 | 1354 | 0.332 | 70 % |
| `reduce-min-f32-full-16777216` | 148.9 | 451 | 49.2 | 1363 | 0.331 | 70 % |
| `reduce-sum-f64-full-16777216` | 94.3 | 1424 | 87.5 | 1534 | 0.928 | 79 % |
| `reduce-max-f64-full-16777216` | 163.3 | 822 | 90.2 | 1487 | 0.553 | 77 % |
| `reduce-min-f64-full-16777216` | 163.7 | 820 | 90.3 | 1486 | 0.552 | 77 % |

Pass 2 (`t4a_cuda_reduce`) is unchanged: 5.2-6.5 µs at 1M on both sides. At 1M, pass 1 is
L2-resident and short (2 to 4 vectors per lane), so it is latency-bound, and the gain there is
1.1-1.2 µs.

## 3. Native/native paired campaign

- Runner: `bench/run-native-pair.sh`, with `REPS=3 CPUS=0-7`.
- Size: 24 cases, 3 repetitions and 2 sides, in AB/BA order, with no failed process.
- Noise: the largest IQR/median is 2.7 %, below the 5 % invalid threshold.
- Metric: host ns per operation over a 100-iteration region, median of the three repetitions.

| case | base µs (IQR%) | cand µs (IQR%) | ratio |
|---|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 15.40 (1.8) | 14.72 (0.3) | 0.956 |
| `reduce-max-f32-full-1048576` | 18.34 (1.2) | 16.52 (1.9) | 0.901 |
| `reduce-min-f32-full-1048576` | 18.12 (0.9) | 16.72 (1.9) | 0.923 |
| `reduce-sum-f64-full-1048576` | 15.84 (1.9) | 15.05 (2.1) | 0.950 |
| `reduce-max-f64-full-1048576` | 18.88 (1.3) | 17.44 (0.9) | 0.924 |
| `reduce-min-f64-full-1048576` | 19.14 (0.3) | 17.54 (1.7) | 0.917 |
| `reduce-sum-f32-full-16777216` | 67.78 (0.5) | 54.00 (0.5) | 0.797 |
| `reduce-max-f32-full-16777216` | 157.89 (0.1) | 59.11 (0.4) | 0.374 |
| `reduce-min-f32-full-16777216` | 156.91 (0.1) | 58.97 (0.5) | 0.376 |
| `reduce-sum-f64-full-16777216` | 101.85 (0.3) | 94.86 (0.3) | 0.931 |
| `reduce-max-f64-full-16777216` | 172.52 (0.1) | 100.90 (0.3) | 0.585 |
| `reduce-min-f64-full-16777216` | 172.78 (0.3) | 100.70 (0.2) | 0.583 |
| `reduce-prod-f32-full-1048576` | 21.67 (1.2) | 21.69 (0.7) | 1.001 |
| `reduce-prod-f64-full-1048576` | 22.14 (2.0) | 22.11 (1.9) | 0.998 |
| `reduce-sumsquares-f32-full-1048576` | 15.69 (0.8) | 14.54 (1.1) | 0.926 |
| `reduce-sum-f32-4096sq-axis1` | 191.66 (2.1) | 193.07 (1.3) | 1.007 |
| `reduce-max-f32-4x131072-axis1` | 14.46 (1.2) | 14.49 (0.9) | 1.002 |
| `reduce-sum-f32-axis-600` | 15.76 (2.7) | 15.47 (1.2) | 0.981 |
| `reduce-sum-f32-4096sq-axis0` | 85.63 (0.1) | 65.17 (0.8) | 0.761 |
| `reduce-sum-f64-4096sq-axis0` | 99.60 (0.6) | 93.67 (0.1) | 0.940 |
| `reduce-max-i32-4096sq-axis0` | 86.78 (0.1) | 66.53 (0.3) | 0.767 |
| `reduce-sum-c64-4096sq-axis0` | 176.83 (0.0) | 176.87 (0.8) | 1.000 |
| `reduce-sum-f32-full-1000000` | 15.33 (2.3) | 14.06 (2.5) | 0.918 |
| `reduce-sum-f32-full-10000000` | 51.06 (0.3) | 38.18 (1.1) | 0.748 |

Rule check: the worst case is 1.007x (the strided `4096sq-axis1`), within the 1.03x limit.
`PROD`, complex and strided cells sit at 0.998-1.007.

The first campaign did not pass that rule:
- `reduce-sum-f32-axis-600` was 1.053x and `reduce-sum-f64-full-1048576` was 1.029x;
- the cause was serial loads for short chunks (design §8);
- the slice was adjusted, and the whole campaign was repeated.

## 4. Through tenferro: `gpu/isolated_ops` contract, paired, one session

- **Contract:** 15 samples, 3 warmups, a 2 ms target batch, host API plus device synchronize,
  no download, CPUs 0-7.
- **Probe:** the 8 published `reduce_*_full_*_n1048576` cells plus 6 cells at `n16777216`.
- **Builds:** both arms from the same tenferro-rs commit. The candidate overrides the kernels
  with `cargo build --release --features cuda --bin benchmark_gpu_public_api --config 'patch."https://github.com/tensor4all/t4a-cuda-kernels".t4a-cuda-kernels.path="<this branch>"'`.
  The candidate binary embeds the new `reduce.cu` source, and the base binary does not.
- **Order:** base, cand, base, cand.
- **Ratio:** the mean of the two adjacent-pair ratios (individual pairs in parentheses).
- **PyTorch drift:** the same ratio for the `pytorch-cuda` arm.
- **Normalised:** ratio ÷ drift.

| cell | direct base µs | direct cand µs | direct ratio | PyTorch drift | normalised | eager ratio | PyTorch µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reduce_sum_full_f32_n1048576` | 28.01 | 27.09 | 0.967 (0.960/0.974) | 1.000 | 0.967 | 0.978 | 9.65 |
| `reduce_sum_full_f64_n1048576` | 28.18 | 27.77 | 0.986 (0.976/0.995) | 1.001 | 0.985 | 0.985 | 11.02 |
| `reduce_prod_full_f32_n1048576` | 34.30 | 34.23 | 0.998 (0.992/1.004) | 1.001 | 0.997 | 1.001 | 9.64 |
| `reduce_prod_full_f64_n1048576` | 34.63 | 34.53 | 0.997 (0.990/1.004) | 1.000 | 0.996 | 0.997 | 11.04 |
| `reduce_max_full_f32_n1048576` | 30.81 | 29.52 | 0.958 (0.953/0.963) | 1.001 | 0.957 | 0.966 | 10.23 |
| `reduce_max_full_f64_n1048576` | 31.92 | 29.63 | 0.929 (0.918/0.939) | 1.001 | 0.927 | 0.950 | 11.56 |
| `reduce_min_full_f32_n1048576` | 30.50 | 29.01 | 0.951 (0.945/0.957) | 1.002 | 0.949 | 0.971 | 10.23 |
| `reduce_min_full_f64_n1048576` | 31.95 | 29.69 | 0.929 (0.918/0.941) | 1.000 | 0.930 | 0.946 | 11.56 |
| `reduce_sum_full_f32_n16777216` | 81.15 | 66.43 | 0.819 (0.815/0.822) | 1.001 | 0.818 | 0.946 | 50.99 |
| `reduce_sum_full_f64_n16777216` | 114.68 | 108.37 | 0.945 (0.948/0.942) | 1.000 | 0.945 | 0.981 | 87.89 |
| `reduce_max_full_f32_n16777216` | 171.28 | 72.20 | 0.422 (0.422/0.421) | 1.000 | 0.421 | 0.703 | 51.63 |
| `reduce_max_full_f64_n16777216` | 186.19 | 113.14 | 0.608 (0.607/0.608) | 1.000 | 0.608 | 0.852 | 88.68 |
| `reduce_min_full_f32_n16777216` | 169.83 | 71.06 | 0.418 (0.418/0.419) | 1.000 | 0.419 | 0.705 | 51.66 |
| `reduce_min_full_f64_n16777216` | 186.34 | 113.52 | 0.609 (0.609/0.610) | 0.998 | 0.610 | 0.848 | 88.66 |
primary (6 x n16777216, direct): median raw 0.608 median normalised 0.609
worst max_error_over_tolerance 0.9873446488113121 worst IQR%  3.5

Verification: every cell of every arm passes `max_error_over_tolerance < 1` against the
NumPy/SciPy `f64` reference.
- tenferro's worst is 0.745, on `reduce_prod_full_f32_n1048576`. It is identical before and
  after, because `PROD` keeps its walk.
- The 0.987 maximum is PyTorch's own `prod` f32.
- `reduce_sum_full_f32_n16777216` goes from 0.070 to below 0.05 of tolerance.

**Gate.** The primary metric (design §5) is 0.609, against the ≤ 0.95 rule, so the slice is
adopted.

## 5. Correctness and mutation evidence

- `cargo test -j 8 --release --features gpu-tests`: all 15 GPU test binaries pass.
- `cargo test -j 8`, `cargo fmt --check` and `cargo clippy -j 8 --all-targets -- -D warnings`
  are clean.
- The new GPU tests add no clippy hit. The gpu-tests feature set has pre-existing clippy
  failures in other files.

Mutation check: each mutant was applied to `reduce.cu`, the new tests were run, and the
original was restored.

| mutant | caught by |
| --- | --- |
| drop the tail fold | `vector_walk_folds_every_element_once…`, `…propagates_a_nan…`, rank-9 twin |
| drop the head fold | same three |
| batch loads `index + b` instead of `index + b*W` | `vector_walk_folds_every_element_once…` |
| first batch folds one vector too few | 2 tests |
| last partial batch folds one vector too few | `vector_walk_folds_every_element_once…` (needs the 4M + 5 length or an 8-lane plane) |
| head ignored (misaligned 16-byte loads) | `CUDA_ERROR_MISALIGNED_ADDRESS` in every walk test |
| `PROD` allowed to regroup | `product_partials_keep_the_serial_lane_walk_bit_for_bit` |

In the first test version, two mutants survived: the whole-batch loop and the 3-left partial
batch never ran on ~256-element chunks. The 4M + 5 length and 8- and 16-lane planes were added
for that.

## 6. Review

- The repository's `independent-review` skill (`openai/gpt-6.1-sol`) is not installed on this
  host.
- A fresh-context reviewer agent was given the diff and `docs/review-checklist.md` instead.
- It found no correctness bug.
- Its two test notes were fixed: a wrong alignment comment in the rank-9 test, and plane
  width 16.
- It confirmed that `c-kernel-abi.md`'s bump rules do not require a revision change.
