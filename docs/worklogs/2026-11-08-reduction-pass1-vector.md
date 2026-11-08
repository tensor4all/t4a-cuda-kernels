# Worklog: vectorized pass-1 lane walk (tenferro-rs #2406, part 2)

Date: 2026-11-08. Plan and outcome:
[slice-reduction-pass1-vector.md](../design/slice-reduction-pass1-vector.md).

The raw evidence for every table below is in
[evidence/2026-11-08-pass1-vector/](evidence/2026-11-08-pass1-vector/).

## 1. Revisions and environment

| Item | Value |
| --- | --- |
| Measured candidate | **code at `65a74d5`** (ABI revision 13). The kernel and library sources (`cuda/`, `src/`) are unchanged after it. Tests changed in `3b65e03` (`tests/planning.rs`, a comment in `tests/gpu_reduce.rs`) and `01dc9b0` (`tests/planning.rs`); every other later commit touches only `docs/`. The GPU suite was re-run on `3b65e03` (§8) and `3d9847e` (§9), which includes both test changes. |
| Baseline | `31b734e`: library `e1f56e4` plus this branch's harness-only commit |
| Through tenferro | tenferro-rs `a09c8d724dfd897ba85de0c06c293b3f8f8d7238`, tenferro-benchmark `c3cc78394667c1e1a99621777db13d7c23ec4232`, PyTorch 2.12.0+cu130 |
| GPU | NVIDIA A100 80GB PCIe, compute capability 8.0, driver 580.178.04 |
| Toolkit | CUDA 12.6 (`nvcc` V12.6.85); kernels compiled with `-std=c++17 -arch=sm_80`; no fast math, no FTZ |
| Host | primerose; measured processes pinned to CPUs 0-7, one GPU job at a time under the host's bench-slot lock, GPU otherwise idle |

- **Theoretical bandwidth:** 1935 GB/s, the A100 80GB PCIe's HBM2e specification.
- **Bench binaries:** the bench binary compiles `cuda/src/*.cu` at run time from its own crate
  directory, so each arm was built in its own worktree. Copying a binary would have run the
  other side's kernels. The tenferro candidate binary was checked to embed the revision-13
  source (`T4A_CUDA_ABI_REVISION 13`, `t4a_cuda_load_group`), and the base binary embeds
  neither.
- **Earlier tables:** the tables in this branch's earlier commits were measured on `3718481`.
  They are superseded by the ones below.

## 2. Pass 1 alone: device time (nsys)

- Each process ran 100 launches in the profiled region, and the table reports the median kernel
  time.
- Bandwidth is the input bytes divided by the pass-1 time.
- Command per case: `scripts/knsys.sh` in the evidence folder, which runs
  `nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop <bench_kernels> <case>`
  and then `nsys stats --report cuda_gpu_kern_sum`.
- Raw data: `b-nsys-65a74d5/{baseline-31b734e,candidate-65a74d5}.csv`.

| case | pass 1 before µs | GB/s | pass 1 after µs | GB/s | ratio | after / 1935 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 8.9 | 471 | 8.1 | 516 | 0.914 | 27 % |
| `reduce-max-f32-full-1048576` | 10.5 | 401 | 8.8 | 477 | 0.841 | 25 % |
| `reduce-min-f32-full-1048576` | 10.4 | 402 | 8.9 | 471 | 0.853 | 24 % |
| `reduce-sum-f64-full-1048576` | 9.0 | 935 | 8.5 | 989 | 0.945 | 51 % |
| `reduce-max-f64-full-1048576` | 10.4 | 807 | 8.8 | 953 | 0.846 | 49 % |
| `reduce-min-f64-full-1048576` | 11.0 | 763 | 8.8 | 953 | 0.801 | 49 % |
| `reduce-sum-f32-full-16777216` | 61.0 | 1100 | 46.6 | 1441 | 0.763 | 74 % |
| `reduce-max-f32-full-16777216` | 149.3 | 450 | 49.2 | 1363 | 0.330 | 70 % |
| `reduce-min-f32-full-16777216` | 148.9 | 451 | 49.4 | 1359 | 0.332 | 70 % |
| `reduce-sum-f64-full-16777216` | 94.4 | 1422 | 87.7 | 1530 | 0.929 | 79 % |
| `reduce-max-f64-full-16777216` | 163.5 | 821 | 90.6 | 1481 | 0.554 | 77 % |
| `reduce-min-f64-full-16777216` | 163.7 | 820 | 90.6 | 1481 | 0.554 | 77 % |

Pass 2 (`t4a_cuda_reduce`) is unchanged on both sides (same CSVs).

## 3. Native/native paired campaign

- Runner: `bench/run-native-pair.sh`, with `REPS=3 CPUS=0-7`.
- Size: 24 cases, 3 repetitions and 2 sides, in AB/BA order.
- Metric: host ns per operation over a 100-iteration region, median of three.
- Raw samples: `b-native-65a74d5-run2/records.jsonl`.

**Run 1 was invalid and is kept, not used.** The baseline side of `reduce-sum-f32-4096sq-axis0`
had an IQR/median of 7.0 %, above the protocol's 5 % threshold, so the whole campaign was
repeated. Run 1's raw data is in `b-native-65a74d5-run1-invalid/`.

Run 2 had no failed process, and its largest IQR/median is 2.2 %:

| case | base µs (IQR%) | cand µs (IQR%) | ratio |
|---|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 15.60 (1.6) | 14.72 (1.7) | 0.944 |
| `reduce-max-f32-full-1048576` | 18.44 (1.5) | 16.92 (0.1) | 0.917 |
| `reduce-min-f32-full-1048576` | 18.09 (0.8) | 16.69 (2.0) | 0.922 |
| `reduce-sum-f64-full-1048576` | 16.00 (1.8) | 15.61 (1.1) | 0.976 |
| `reduce-max-f64-full-1048576` | 18.87 (1.2) | 17.82 (1.2) | 0.944 |
| `reduce-min-f64-full-1048576` | 19.16 (0.4) | 17.31 (1.4) | 0.903 |
| `reduce-sum-f32-full-16777216` | 67.89 (0.5) | 54.06 (0.3) | 0.796 |
| `reduce-max-f32-full-16777216` | 157.97 (0.2) | 59.57 (0.4) | 0.377 |
| `reduce-min-f32-full-16777216` | 156.99 (0.2) | 58.42 (0.2) | 0.372 |
| `reduce-sum-f64-full-16777216` | 101.74 (0.3) | 95.40 (0.2) | 0.938 |
| `reduce-max-f64-full-16777216` | 172.42 (0.0) | 101.14 (0.1) | 0.587 |
| `reduce-min-f64-full-16777216` | 172.59 (0.2) | 100.57 (0.2) | 0.583 |
| `reduce-prod-f32-full-1048576` | 21.67 (1.1) | 21.75 (2.2) | 1.004 |
| `reduce-prod-f64-full-1048576` | 21.97 (0.7) | 22.16 (0.3) | 1.009 |
| `reduce-sumsquares-f32-full-1048576` | 15.46 (1.5) | 14.57 (1.0) | 0.943 |
| `reduce-sum-f32-4096sq-axis1` | 193.77 (2.2) | 193.05 (1.1) | 0.996 |
| `reduce-max-f32-4x131072-axis1` | 14.38 (0.3) | 14.30 (1.9) | 0.994 |
| `reduce-sum-f32-axis-600` | 15.71 (1.7) | 15.50 (1.8) | 0.986 |
| `reduce-sum-f32-4096sq-axis0` | 85.55 (0.2) | 63.35 (0.1) | 0.740 |
| `reduce-sum-f64-4096sq-axis0` | 99.81 (1.0) | 94.68 (0.1) | 0.949 |
| `reduce-max-i32-4096sq-axis0` | 86.79 (0.5) | 64.07 (0.7) | 0.738 |
| `reduce-sum-c64-4096sq-axis0` | 176.85 (0.1) | 176.67 (0.1) | 0.999 |
| `reduce-sum-f32-full-1000000` | 15.31 (1.9) | 14.29 (0.7) | 0.934 |
| `reduce-sum-f32-full-10000000` | 50.97 (0.3) | 39.42 (0.3) | 0.773 |

**Rule check.**
- The worst case is 1.009x (`reduce-prod-f64-full-1048576`), within the 1.03x limit.
- `PROD`, complex and strided cells sit at 0.994-1.009.

**History.** The first campaign of this branch, on `e46e22a` before amendment, did not pass the
rule. `reduce-sum-f32-axis-600` measured 1.053x because short chunks issued serial loads; that
is fixed by the predicated first and last batch.

## 4. Through tenferro: `gpu/isolated_ops` contract, paired, one session

- **Contract:** 15 samples, 3 warmups, a 2 ms target batch, host API plus device synchronize,
  no download, CPUs 0-7.
- **Probe:** `b-tenferro-65a74d5/probe.json` (the 8 published `reduce_*_full_*_n1048576` cells
  plus 6 cells at `n16777216`).
- **Builds:** both arms from the same tenferro-rs commit. The candidate overrides the kernels
  with `cargo build --release --features cuda --bin benchmark_gpu_public_api --config 'patch."https://github.com/tensor4all/t4a-cuda-kernels".t4a-cuda-kernels.path="<this branch at 65a74d5>"'`.
- **Order:** base, cand, base, cand.
- **Raw records:** `b-tenferro-65a74d5/{tf1-base,tf2-cand,tf3-base,tf4-cand}/*.jsonl`. Each
  record holds every sample's `batch_ms` and `operations_per_sample`.
- **Summary:** produced by `scripts/tfsumm.py`. The ratio is the mean of the two adjacent-pair
  ratios (individual pairs in parentheses).
- **PyTorch drift:** the same ratio for the `pytorch-cuda` arm. "Normalised" is ratio ÷ drift.

| cell | direct base µs | direct cand µs | direct ratio | PyTorch drift | normalised | eager ratio | PyTorch µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reduce_sum_full_f32_n1048576` | 27.84 | 27.06 | 0.972 (0.967/0.976) | 1.000 | 0.971 | 0.975 | 9.65 |
| `reduce_sum_full_f64_n1048576` | 27.97 | 27.72 | 0.991 (0.983/0.999) | 1.000 | 0.991 | 0.981 | 11.03 |
| `reduce_prod_full_f32_n1048576` | 34.26 | 34.13 | 0.996 (0.991/1.002) | 1.000 | 0.997 | 0.997 | 9.65 |
| `reduce_prod_full_f64_n1048576` | 34.45 | 34.48 | 1.001 (0.995/1.007) | 1.000 | 1.001 | 1.001 | 11.04 |
| `reduce_max_full_f32_n1048576` | 30.64 | 29.28 | 0.956 (0.949/0.963) | 1.000 | 0.955 | 0.957 | 10.24 |
| `reduce_max_full_f64_n1048576` | 31.67 | 29.43 | 0.929 (0.926/0.932) | 1.001 | 0.928 | 0.951 | 11.57 |
| `reduce_min_full_f32_n1048576` | 30.15 | 29.33 | 0.973 (0.970/0.976) | 1.001 | 0.971 | 0.965 | 10.23 |
| `reduce_min_full_f64_n1048576` | 31.75 | 29.44 | 0.927 (0.922/0.933) | 1.000 | 0.928 | 0.945 | 11.57 |
| `reduce_sum_full_f32_n16777216` | 81.01 | 67.07 | 0.828 (0.830/0.826) | 1.002 | 0.826 | 0.948 | 50.97 |
| `reduce_sum_full_f64_n16777216` | 114.33 | 107.81 | 0.943 (0.942/0.944) | 0.998 | 0.945 | 0.983 | 87.95 |
| `reduce_max_full_f32_n16777216` | 170.90 | 71.82 | 0.420 (0.419/0.421) | 1.000 | 0.420 | 0.701 | 51.70 |
| `reduce_max_full_f64_n16777216` | 185.53 | 113.78 | 0.613 (0.612/0.615) | 1.000 | 0.613 | 0.852 | 88.68 |
| `reduce_min_full_f32_n16777216` | 169.62 | 71.54 | 0.422 (0.420/0.424) | 1.001 | 0.421 | 0.706 | 51.67 |
| `reduce_min_full_f64_n16777216` | 186.21 | 112.84 | 0.606 (0.605/0.607) | 1.001 | 0.606 | 0.849 | 88.65 |
primary (6 x n16777216, direct): median raw 0.61 median normalised 0.609
worst max_error_over_tolerance 0.9873446488113121 worst IQR%  1.5

Verification: every cell of every arm has `max_error_over_tolerance < 1` against the
NumPy/SciPy `f64` reference.
- tenferro's worst is 0.745, on `reduce_prod_full_f32_n1048576`. It is identical before and
  after.
- The 0.987 maximum is PyTorch's own `prod` f32.

**Gate.** The primary metric (design §5) is **0.609**, against the ≤ 0.95 rule.

## 5. Correctness, mutation and instruction evidence (all at code `65a74d5`)

**(a) Which commands run the GPU tests.**
- `cargo test` and `cargo clippy` without features build no GPU test binary.
  - `Cargo.toml` gives each `gpu_*` test `required-features = ["gpu-tests"]`.
  - `a-cargo-test-default.log` shows only `unittests src/lib.rs`, `tests/fusion.rs` and
    `tests/planning.rs` running.
  - `a-cargo-clippy-default.log` exits 0.
- The changed GPU tests were therefore run with the feature:
  - `a-cargo-test-gpu-tests.log`: `cargo test -j 4 --release --features gpu-tests
    --no-fail-fast`. All 15 test binaries pass and the command exits 0. `gpu_reduce` has 28
    tests and `gpu_dynamic_rank` 23.
  - `a-cargo-clippy-gpu-tests.log`: `cargo clippy -j 4 --keep-going --all-targets --features
    gpu-tests -- -D warnings`. It exits 101 on lints that were already present in other files,
    and in `tests/gpu_dynamic_rank.rs` below line 3030; none is in code this branch adds.
    `tests/gpu_reduce.rs` and `tests/planning.rs` have no hit.

**(c) Mutation.** Each mutant patch was applied to `reduce.cu`, the walk tests ran, and the
original was restored and its sha256 checked. Patches and full logs are in `c-mutants/<name>.{patch,log}`,
and the driver is `scripts/gpu-evidence.sh`.

| mutant | result |
| --- | --- |
| `drop-tail` | 5 tests FAILED |
| `batch-overlap` (whole-batch loads `index + b`) | 2 FAILED |
| `first-batch-short` | 2 FAILED |
| `last-batch-short` | 2 FAILED (reached by the 917 507-element case) |
| `aligned-test-too-weak` (16-byte loads on 4-byte-aligned chunks) | `CUDA_ERROR_MISALIGNED_ADDRESS` in 4 places |
| `scalar-group-load-wrong` | 6 FAILED |
| `prod-regrouped` | 1 FAILED (`product_partials_keep_the_serial_lane_walk_bit_for_bit`) |

**(c) Instructions.**
- `c-sass-diff.log` is the raw `scripts/sass-diff.sh 31b734e` output: 92 entries identical, and
  only `t4a_cuda_reduce_partials` and `t4a_cuda_reduce_partials_any` differ.
- `c-ptxas.log` has the `ptxas -v` output for both trees: registers per entry and zero spill
  bytes. Both pass-1 entries use 50 registers (`:7`, `:115`); on `31b734e` they used 40
  (`t4a_cuda_reduce_partials_any`, `:225`) and 39 (`t4a_cuda_reduce_partials`, `:277`).
- `f-sass-ldg128.log` (added after the 14:25Z review) is `cuobjdump --dump-sass` of the same
  `nvcc -std=c++17 -cubin -arch=sm_80` build with CUDA 12.6: 154 `LDG.E.128` in each pass-1
  entry, none in either entry on `31b734e`, and none in the single-pass entries. It includes
  an instruction excerpt of `t4a_cuda_reduce_partials`.

**Not runnable here.** `gpt_review.sh`, the external GPT review driver, is not installed on
this host, and the review states it runs on a Darwin host. It has to be run there.

## 6. Review witnesses (GPT-6.1-sol review of #23)

- **Point 1, the stride checked after the pointer is formed.**
  `tests/planning.rs::pass1_checks_the_axis_stride_before_forming_a_chunk_pointer` fails on
  `3718481` and passes from `469284f`. The GPU test of the review's two layouts (stride 0 over
  one element, stride -1 at offset 511) passes on both: the UB is not observable.
- **Point 2, a topology that depends on the address.**
  `gpu_reduce.rs::pass1_topology_does_not_depend_on_the_input_address` failed on `3718481`
  with `0 vs NaN` and passes from `469284f`.
- **Point 3, the lane assignment.**
  - `gpu_reduce.rs::pass1_overflow_follows_the_revision_13_grouping` fails on the
    revision-12 kernel (`31b734e`) and passes from `65a74d5`.
  - The NaN of `x0 = x1 = A, x32 = x33 = -A` is the `A + A` overflow in one group: with
    `A = 1` the same placement sums to exactly `0`.
  - `pass1_folds_nothing_outside_its_chunk` (NaN guards) and
    `pass1_adds_exactly_the_one_nonzero_element` pin that no empty lane or unloaded batch slot
    contributes.
  - `an_empty_reduction_axis_is_refused_before_any_launch` pins the empty axis.

## 7. Restoring the revision-12 lanes with warp shuffles: measured, not adopted

**What was built.** The prototype is `d-shuffle-prototype/prototype.patch`, against `65a74d5`
and not on this branch.
- Lane `t` of a 32-lane block keeps owning chunk elements `t + 32k` in order.
- Each lane loads the 16-byte vector `stripe * 32 + lane`.
- Each owner receives its element through `__shfl_sync`, `V` shuffles per element.
- It covers a 16-byte-aligned chunk, plane width 32 and `len % (32V) == 0`, which is every
  measured full-reduction cell. Every other case uses the serial walk.

**Bit identity.** Its partials and results are bit-identical to the revision-12 serial walk
(`31b734e`) for F32/F64 `SUM`, `MAX` and `SUM_SQUARES` at 1M and 16M.
- `d-shuffle-prototype/partials-{prototype,serial-31b734e}.txt.gz` are the 4096-partial dumps.
- `cmp` reports them identical, and `partials.sha256` holds the hashes.

**Timing.** Pass-1 device time, nsys, with all three variants measured in the same session:

| case | serial walk, `31b734e` µs | vector walk, `65a74d5` µs | shuffle prototype µs | shuffle − vector | shuffle / vector − 1 |
|---|---:|---:|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 8.9 | 8.1 | 8.7 | +0.5 µs | +6.69 % |
| `reduce-max-f32-full-1048576` | 10.5 | 8.8 | 8.9 | +0.0 µs | +0.36 % |
| `reduce-sum-f64-full-1048576` | 9.0 | 8.5 | 8.4 | -0.1 µs | -0.94 % |
| `reduce-max-f64-full-1048576` | 10.4 | 8.8 | 9.8 | +0.9 µs | +10.49 % |
| `reduce-sum-f32-full-16777216` | 61.0 | 46.7 | 52.4 | +5.6 µs | +12.05 % |
| `reduce-max-f32-full-16777216` | 149.3 | 49.0 | 55.3 | +6.3 µs | +12.86 % |
| `reduce-min-f32-full-16777216` | 148.9 | 49.3 | 55.3 | +6.0 µs | +12.27 % |
| `reduce-sum-f64-full-16777216` | 94.4 | 87.3 | 90.2 | +2.9 µs | +3.30 % |
| `reduce-max-f64-full-16777216` | 163.5 | 90.6 | 98.3 | +7.8 µs | +8.57 % |
| `reduce-min-f64-full-16777216` | 163.7 | 90.6 | 98.4 | +7.8 µs | +8.62 % |

The last two columns are computed from the `Med (ns)` column of
`d-shuffle-prototype/nsys-prototype.csv` and `nsys-vector-65a74d5-same-session.csv`, kernel
`t4a_cuda_reduce_partials`: `prototype / vector − 1`.

**Conclusion: not worth it.**
- **Cost:** at 16M, keeping the revision-12 topology adds +5.6 to +6.3 µs (12.1-12.9 %) to
  the vector walk's pass 1 for F32 `SUM`/`MAX`/`MIN`, +7.8 µs (8.57 % `MAX`, 8.62 % `MIN`) for
  F64, and +2.9 µs (3.30 %) for F64 `SUM`. That is under the 10-15 µs estimated beforehand.
  At 1M it is -0.1 to +0.9 µs.
- **What it would buy:** bit compatibility with revision 12 only. Revision 12's topology is not
  more accurate: the two documented witnesses flip classification in opposite directions.
- **What it would also need:**
  - more code for chunks that start off a 16-byte boundary, for plane widths 8 and 16, and for
    lengths that do not fill a stripe;
  - a fourth shuffle-dependent fold order, alongside the serial, group and combine folds, to
    keep tested.

Revision 13 keeps the cheaper topology and states it.

## 8. Review fixes (GPT-6.1-sol review of #23, 2026-11-08T14:25Z)

**Pointer-guard test (needs-changes 1).**
- `tests/planning.rs::pass1_checks_the_axis_stride_before_forming_a_chunk_pointer` used to
  check only that `axis_stride == 1` came before the pointer expression in the text. Both
  reviews' hoisting mutants passed it (`hoist-old-test.log`, `hoist-folded-guard-old-test.log`:
  `ok. 1 passed`).
- The test now executes the dispatch instead of reading it (`01dc9b0`):
  - it extracts the actual `t4a_cuda_partials_walk` definition from `reduce.cu`;
  - it retypes only the `in` parameter, to `CheckedPtr<T>`, whose `operator+` records every
    result outside `[0, len]` of its allocation;
  - it stubs the callees so they report which walk ran, compiles the result with the host
    `c++ -std=c++17` (`$CXX` if set) and runs it;
  - it runs the review's two witnesses (stride 0 over one element, `start = 256`; stride -1
    at offset 511 over 512) and a contiguous chunk. The witnesses must form no out-of-range
    pointer and take the serial walk; the contiguous chunk must take the vector walk.
- `g-pointer-guard-mutants/` (each patch applied to `reduce.cu`, the test run, the original
  restored and its sha256 checked):

| mutant | result | probe output |
| --- | --- | --- |
| `hoist` (14:25Z review) | `FAILED. 0 passed; 1 failed` | `formed in + 256 over an allocation of 1`, `formed in + 767 over an allocation of 512` |
| `hoist-folded-guard` (14:28Z review) | `FAILED. 0 passed; 1 failed` | the same two |
| `or-guard` (`axis_stride == 1 \|\| ...`) | `FAILED. 0 passed; 1 failed` | the same two, and both witnesses take the vector walk |
| `no-stride-check` | `FAILED. 0 passed; 1 failed` | the same |
| `var-guard` (pointer inside `if (contiguous && ...)`, which is safe) | `ok. 1 passed` | none |
| `negated-guard` (15:30Z review: `if (!(false && axis_stride == 1 && ...))`) | `FAILED. 0 passed; 1 failed` | `formed in + 256 over an allocation of 1`, `formed in + 767 over an allocation of 512` |

- The probe stubs the walks, so it proves where the pointer is formed, not what the walks
  compute; the GPU tests cover that.
- GCC 13 does not diagnose out-of-bounds pointer arithmetic during constant evaluation, so a
  `static_assert`-based probe would not fail on the mutants. That is why the probe runs.

**Numbers corrected against the artifacts (needs-changes 2).**
- Sparse coverage: the single-`1.0` test checks 13 chosen positions (F32, `W = 32`, leads
  0-3), not every chunk position. The design and the test comment now say so.
- Registers: 50 in both pass-1 entries (`c-ptxas.log:7,115`), from 40 and 39
  (`:225`, `:277`). The earlier "52-54 from 39-40" was wrong.
- Shuffle prototype: recomputed from the same-session medians (§7 table, last column). 16M:
  F32 +12.05/+12.86/+12.27 % (`SUM`/`MAX`/`MIN`), F64 `MAX` +8.57 %, `MIN` +8.62 %, `SUM`
  +3.30 %. The earlier "12-16 %" was wrong.

**`LDG.E.128` (unverified 1).** `f-sass-ldg128.log`; see §5(c).

**GPU re-run (unverified 2).** Superseded by §9.
 `e-cargo-test-gpu-tests-3b65e03.log` (sha256
`625b32320585ef0de99beb44531dac6e820c53644fb0c3e8f3c7ea9e273edf44`): `cargo test -j 4
--release --features gpu-tests --no-fail-fast` on primerose (A100 80GB PCIe, driver
580.178.04, CUDA 12.6) at `3b65e03`. All 15 test binaries report `0 failed`, and the command
ends `# exit 0`. `65a74d5..8640857` touches only `docs/`, and `8640857..3b65e03` touches only
`tests/`. No timing was re-measured.

## 9. Re-run on `3d9847e` (2026-11-08T15:14Z onward)

`cuda/` and `src/` at `3d9847e` are identical to `65a74d5`
(`git diff --quiet 65a74d5 3d9847e -- cuda src`). Later commits add only `docs/`.

**GPU tests.** `h-cargo-test-gpu-tests-3d9847e.log` (sha256
`60066c14fd9e6dceb98be376ac81b478778b8c7e84dae7081bd74dda9756cc09`): `cargo test -j 4
--release --features gpu-tests --no-fail-fast` on primerose (A100 80GB PCIe, driver
580.178.04, CUDA 12.6). All 15 test binaries and the doc-tests report `0 failed`, and the
command ends `# exit 0`. `gpu_dynamic_rank` 23 passed, `gpu_reduce` 28, `planning` 89.

**Pass 1 alone, nsys.** `i-nsys-3d9847e/` (method in its `README.md`): `31b734e` against
`3d9847e`, run b1, c1, b2, c2 with `scripts/knsys.sh`. Every bench run reports `check=ok`.
Median `t4a_cuda_reduce_partials` time, mean of the two runs, with each run in parentheses:

| case | before µs (r1/r2) | GB/s | after µs (r1/r2) | GB/s | ratio (r1/r2) | after / 1935 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `reduce-sum-f32-full-1048576` | 8.9 (8.9/8.9) | 471 | 8.2 (8.2/8.2) | 514 | 0.917 (0.917/0.917) | 27% |
| `reduce-max-f32-full-1048576` | 10.5 (10.5/10.5) | 401 | 8.8 (8.8/8.8) | 475 | 0.844 (0.844/0.844) | 25% |
| `reduce-min-f32-full-1048576` | 10.4 (10.4/10.4) | 402 | 8.9 (8.9/8.9) | 471 | 0.853 (0.853/0.853) | 24% |
| `reduce-sum-f64-full-1048576` | 9.0 (9.0/9.0) | 936 | 8.5 (8.5/8.5) | 987 | 0.948 (0.950/0.946) | 51% |
| `reduce-max-f64-full-1048576` | 10.4 (10.5/10.4) | 804 | 8.8 (8.8/8.8) | 950 | 0.847 (0.844/0.849) | 49% |
| `reduce-min-f64-full-1048576` | 11.0 (11.0/11.0) | 760 | 8.8 (8.8/8.8) | 953 | 0.797 (0.797/0.797) | 49% |
| `reduce-sum-f32-full-16777216` | 60.9 (60.9/60.9) | 1102 | 46.7 (46.7/46.6) | 1438 | 0.766 (0.767/0.765) | 74% |
| `reduce-max-f32-full-16777216` | 149.4 (149.3/149.5) | 449 | 49.3 (49.3/49.3) | 1361 | 0.330 (0.330/0.330) | 70% |
| `reduce-min-f32-full-16777216` | 149.0 (149.1/148.9) | 450 | 49.4 (49.3/49.5) | 1358 | 0.332 (0.331/0.332) | 70% |
| `reduce-sum-f64-full-16777216` | 94.2 (94.1/94.4) | 1424 | 87.6 (87.6/87.7) | 1531 | 0.930 (0.931/0.930) | 79% |
| `reduce-max-f64-full-16777216` | 163.4 (163.4/163.4) | 822 | 90.6 (90.7/90.5) | 1481 | 0.555 (0.555/0.554) | 77% |
| `reduce-min-f64-full-16777216` | 163.5 (163.6/163.5) | 821 | 90.7 (90.8/90.7) | 1479 | 0.555 (0.555/0.554) | 76% |

Every cell agrees with §2 (measured on `65a74d5`) to within 0.1 µs, except the baseline of
`reduce-sum-f64-full-16777216` (94.2 µs, was 94.4). The two runs of one side differ by at most
0.61 % (`reduce-max-f64-full-1048576` baseline).

## 10. GPT-6.1-sol review of 2026-11-08T15:30Z (read head `a01544f`)

- **Point 1, the `&&`-split guard check.** The finding is correct for `a01544f`: the negated
  condition `if (!(false && axis_stride == 1 && ...))` passes that text check
  (`g-pointer-guard-mutants/negated-guard-a01544f-test.log`: `ok. 1 passed`). Since `01dc9b0`
  the test runs the dispatch rather than parsing its condition (§8), and the same mutant fails it
  (`negated-guard.log`: `FAILED. 0 passed; 1 failed`, both witnesses form an out-of-range
  pointer and take the vector walk).
- **Point 2, provenance.** §1 claimed that every commit after `65a74d5` touches only `docs/`.
  It now says that the kernel and library sources are unchanged after `65a74d5` and names the two
  commits that changed tests.
