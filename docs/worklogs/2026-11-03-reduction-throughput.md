# Worklog: reduction combine throughput (tenferro-rs #2406, part 1)

Date: 2026-11-03. Plan: [slice-reduction-throughput.md](../design/slice-reduction-throughput.md).
Issue: tenferro-rs #2406 (full-array reductions 5.7-8.0x behind PyTorch on the A100).

Shipped: **mechanism 1 only** — a wide combine block for a split reduction's pass 2.
Withdrawn: mechanism 3 (pass-1 grouped walk), measured and reverted. Not adopted: mechanism 4
(reduction-only extrema fold), rejected by its isolated A/B.

## 1. Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe, driver 580.178.04, toolkit 12.6 (`CUDARC_CUDA_VERSION=12080` for the tests) |
| Host | primerose; measured harness and recorder pinned to CPUs 60-62 inside the 56-63 L3 domain, observer on 48-55 |
| Baseline | `t4a-cuda-kernels@d44b57b` (`origin/main`), same harness source |
| Candidate | this branch, ABI revision 12 |
| Harness | `examples/bench_kernels.rs`, release, `--features gpu-tests`, both arms built from the same source |

Each process: warm-ups, `cuProfilerStart`, 100 iterations, one barrier, `cuProfilerStop`; the
host clock spans exactly that barrier-delimited region, so the reported number is warm
batch-throughput in ns per operation (device time for the same region comes from nsys).

## 2. Protocol

Paired native/native campaign (`/tmp/supk2406/run_native_pair.sh` in this session; the harness
work it relies on is in this branch): 21 cases x 3 repetitions x 2 sides, AB/BA order
alternating per repetition, one process per case per side, fresh campaign directory, GPU
exclusive (refused while another compute process was present), median and IQR per case,
invalid above IQR/median 0.05, a whole-campaign repeat rather than a selective retry.

## 3. A/A noise floor

Same binary on both sides, four representative cells x 3 repetitions: median ratios 0.990,
1.001, 1.008 and 1.012. The noise floor of this host is therefore about 1.2% on these cells,
which is the scale at which the 1.03x non-regression limit operates.

## 4. Result (reported campaign, baseline `d44b57b`)

medians in ns per operation of the measured region. Harness and runner are in this branch
(`examples/bench_kernels.rs`, `bench/run-native-pair.sh`), so the run is reproducible from the
tree. The campaign replaced an earlier one whose whole-campaign repeat the protocol required
(one cell above the noise gate); this one is clean (max IQR/median 0.027, no failed process).

| case | baseline | candidate | ratio |
| --- | ---: | ---: | ---: |
| `reduce-sum-f32-axis-511` | 10 280 | 10 128 | 0.985 |
| `reduce-sum-f32-axis-512` | 16 220 | 15 974 | 0.985 |
| `reduce-sum-f32-axis-513` | 16 307 | 16 063 | 0.985 |
| `reduce-sum-f32-axis-600` | 15 910 | 15 663 | 0.984 |
| `reduce-sum-f32-4096sq-axis0` | 85 536 | 85 978 | 1.005 |
| `reduce-sum-f32-4096sq-axis1` | 192 359 | 194 349 | 1.010 |
| `reduce-sum-f64-4096sq-axis0` | 99 426 | 99 514 | 1.001 |
| `reduce-sum-c64-4096sq-axis0` | 177 365 | 176 716 | 0.996 |
| `reduce-max-i32-4096sq-axis0` | 85 788 | 86 270 | 1.006 |
| `reduce-sum-f32-full-1000000` | 21 214 | 15 438 | 0.728 |
| `reduce-sum-f32-full-10000000` | 57 517 | 51 063 | 0.888 |
| `reduce-sum-f32-full-1048576` | 21 869 | 15 362 | 0.702 |
| `reduce-prod-f32-full-1048576` | 21 579 | 21 816 | 1.011 |
| `reduce-max-f32-full-1048576` | 46 676 | 18 457 | 0.395 |
| `reduce-min-f32-full-1048576` | 46 864 | 18 249 | 0.389 |
| `reduce-sum-f64-full-1048576` | 21 976 | 15 674 | 0.713 |
| `reduce-prod-f64-full-1048576` | 22 141 | 22 070 | 0.997 |
| `reduce-max-f64-full-1048576` | 49 018 | 19 187 | 0.391 |
| `reduce-min-f64-full-1048576` | 48 728 | 19 139 | 0.393 |
| `reduce-sumsquares-f32-full-1048576` | 21 922 | 15 472 | 0.706 |
| `reduce-max-f32-4x131072-axis1` | 16 701 | 14 349 | 0.859 |

- Eight selection cells (the `2406` cells of `gpu_isolated_ops`): median ratio **0.549**.
- Four extrema cells: **0.389, 0.391, 0.393, 0.395**.
- Worst case 1.011, inside the declared 1.03 limit; `Prod` unchanged (1.011 / 0.997).

**Measured scope.** The frozen matrix in the plan named 43 cases; this campaign measured the 21
in the table, and these are **not** measured: the 4M/16M extrema and sum-of-squares cells, the
wide integer and complex full reductions, a rank-9 benchmark case, and a 256-output wide case.
The plan's own extrema-wide and rank-9 *correctness* claims are covered by the GPU tests, not by
timings. Recorded as a scope limitation of this part, not as a passed gate over those cells.

## 5. Mechanism 3 withdrawn, with the numbers

Implemented (four lane-strided loads combined as a tree before folding into one accumulator,
shared by both partials bodies), measured, and removed:

| case | baseline | grouped walk | ratio |
| --- | ---: | ---: | ---: |
| `reduce-sum-f32-4096sq-axis1` | 194.3 us | 217.3 us | 1.118 |
| `reduce-sum-f32-axis-600` | 16.0 us | 16.9 us | 1.056 |

Two attempts to attribute it: restricting the grouped walk to a contiguous axis left the
strided row at 1.131, and a build with the grouped path forced off still measured 1.115 on it.
The regression is pass-1 code generation caused by the branch's presence, not the branch
executing, so the change was withdrawn rather than tuned.

## 6. Mechanism 4 rejected by its isolated A/B

Mechanism 1 fixed, only the reduction-only extrema helper differing, 5 extrema cells x 3
repetitions (`reduce-{max,min}-{f32,f64}-full-1048576` and `reduce-max-i32-4096sq-axis0`):
median ratio **0.998**, worst cell **1.0047**. No measurable win, so the helper is
not in this change and the shared elementwise helpers were never touched (they promise the lhs
NaN payload, which `tests/gpu_elementwise.rs` pins).

## 7. The regression the first version of mechanism 1 caused

Putting the wide step in the fold used by *all* reduce entries regressed a case the wide block
never serves — `reduce-sum-f32-4096sq-axis1` at 1.125 (16 partials, so pass 2 is `Unit` at
width 32; the cost was in pass 1). Splitting the fold so only `t4a_cuda_reduce`/`_any` can widen
restored that case to 0.987-1.000 with every win intact. Neither marking the wide step
`__noinline__` (1.136) nor leaving it inlined (1.125) helped, which is what identified the
separation, not the call, as the fix.

## 8. Verification

- `cargo fmt --all --check`, `cargo clippy -j 4 --all-targets -- -D warnings` (the repository's
  documented gate), `cargo test -j 4` (88 planning/ABI tests among others).
- `cargo test -j 4 --features gpu-tests` on the local A100: every suite green, including the
  four new wide-combine tests, the rank-9 `*_any` split test, and the unchanged extrema and
  dynamic-rank suites.
- ABI revision 11 -> 12 recorded in `src/lib.rs`, `cuda/include/t4a/abi.cuh` and
  `docs/abi/c-kernel-abi.md` (launch geometry, reassociation policy, wide combine rule), with
  the crate's own header/revision agreement test green.

## 9. What is not done

Pass-1 streaming is untouched and is now the largest remaining term for full reductions: at
16M the issue accounts ~112 us to it, which this change does not address (a vectorized
lane-to-element mapping, its own slice). `MAX_SPLITS` stays 4096 — with the combine no longer
scaling with `splits`, re-tuning it bought nothing measurable. The per-call floor and the split
workspace's zero fill remain consumer/runtime work (issue fixes 5 and 6).
