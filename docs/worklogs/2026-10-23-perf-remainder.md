# Worklog: closing the kernel performance remainder (U8a/U8b) and the paired rerun

Date: 2026-10-23. Plan: [slice-perf-remainder.md](../design/slice-perf-remainder.md);
protocol: the frozen one in [kernel-parity-benchmark.md](../design/kernel-parity-benchmark.md)
and [2026-10-21-complete-port-perf.md](2026-10-21-complete-port-perf.md) §2 (unchanged:
10 warm-ups outside the profiler range, `cuProfilerStart`, 100 iterations, one barrier,
`cuProfilerStop`; 3 process repetitions per side; device time per kernel is the nsys
`cuda_gpu_kern_sum` per-instance median; validity `max/min <= 1.15`; regression
`candidate median > 1.10x baseline median`).

Baseline: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c` in the throwaway
worktree `.worktrees/t4a-baseline-14bc1ab` with `bench/cubecl-baseline/baseline.patch`
applied. Candidate: this branch `perf/u8-kernel-parity`, ABI revision 8.

## 1. Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe, idle (`nvidia-smi --query-compute-apps`) before every process |
| Driver | 580.126.09 |
| Toolkit | nvcc/ptxas/cuobjdump 12.0.140; NVRTC 12.0.140 for the prepare case |
| Nsight Systems | 2024.2.3.38-242334140272v0 |
| Rust | cargo/rustc 1.97.1 |
| Campaign | `shinaoka-advisor/work/runs/t4a-cuda-kernels-perf/perf-campaign-2` (fresh dir, all 53 cases x 3 repetitions x 2 sides captured) |
| Command | `bash bench/run-paired.sh --out <campaign> --reps 3 --candidate-bin target/release/examples/bench_kernels --baseline-gpu-bin <…>/tenferro_gpu-7698e59615da0353 --baseline-linalg-bin <…>/tenferro_linalg-aeb242cdd00fbe2b` |

## 2. What changed

- **Metadata local memory (unit A).** `window.cu` (slice, pad), `diagonal.cu`
  (extract/embed), `gather_scatter.cu` (gather, scatter_add), `linalg_aux.cu`
  (lu_apply_pivots, lu_extract_outputs, qr_apply_phase) and the `abi.cuh` helpers. Every
  runtime axis subscript is now a literal subscript of an unrolled walk, and compact
  destinations/sources are addressed flat where `src/lib.rs` proves compactness. The
  masked selector reads all eight coordinate slots, so producers now initialize the whole
  array (a rank-2 diagonal op previously reached it with undefined slots); documented in
  `abi.cuh` and in `docs/abi/c-kernel-abi.md`.
- **Index width and instruction cost (unit B).** `movement.cu`: one compiled transpose body
  per documented tile configuration (uniform switch, `__launch_bounds__(512, 4)`, 40→32
  registers) and a signed 32-bit index path with a 64-bit fallback whose range is proved
  from the planner-validated span; the same narrow walk in `materialize_strided`,
  `contiguous_to_view`, `strided_to_strided`, `fill_zero_view`. `rrqr.cu`: layout metadata
  read through `const&` so the by-value POD is not copied into the stack frame. The
  specialization record (`docs/abi/c-kernel-abi.md`, "Specialization record") lists the
  transpose tile set and the two index widths.
- **Evidence tooling (unit C).** The phase-2 harness now runs on `main`/revision 8
  (`examples/bench_kernels.rs`, `tests/gpu/prepare.rs`, `bench/`, `scripts/resource-usage.sh`)
  with the defects the independent review found in the old tooling removed: semantic
  kernel pairing with instance-count agreement instead of speed-ranked pairing; a campaign
  that requires a fresh directory, every declared case x repetition, and reports itself
  incomplete instead of shrinking the sample; disjoint RRQR operands with the module that
  owns each entry; a stated measurement of what an RRQR iteration actually does; and a
  predicated-instruction-aware local-memory count.

## 3. Static resource usage (`scripts/resource-usage.sh`, sm_80)

Before (base `40857fb`) → after, per entry; the table is diagnostic, not a gate:

| entry | stack / `LDL`+`STL` before | after |
| --- | --- | --- |
| `t4a_cuda_slice`, `t4a_cuda_pad` | 144 B / 88 | 0 / 0 |
| `t4a_cuda_extract_diagonal`, `t4a_cuda_embed_diagonal_copy` | 64 B / 63 | 0 / 0 |
| `t4a_cuda_gather` | 192 B / 1456 | 0 / 0 |
| `t4a_cuda_scatter_add` | 320 B / 1208 | 0 / 0 |
| `t4a_cuda_lu_apply_pivots` | 432 B / 438 | 0 / 0 |
| `t4a_cuda_lu_extract_outputs` | 720 B / 488 | 0 / 0 |
| `t4a_cuda_qr_apply_phase` | 288 B / 384 | 0 / 0 |
| `t4a_cuda_{extract_r,swap_columns,column_norms,select_pivot,prefix_rank}` | 144-288 B / 116-348 | 0 / 0 |
| `t4a_cuda_tiled_transpose` | 40 registers, 0 / 0 | 32 registers, 0 / 0 |
| `t4a_cuda_unary` | 72 B / 17 | 72 B / 20 — libdevice `sin`/`cos` slow path, kept |

No entry other than `t4a_cuda_unary` now touches local memory.

## 4. Campaign result

53 cases: **28 `ok`**, **25 `REGRESSION`**, 0 `inconclusive`, 0 pairing failures, 0
missing captures. The previous campaign had 39 regressions; 14 of them are now at parity
or faster (pad 3.6-3.8x → 0.88-0.89x; transpose 1.36-1.96x → 1.01-1.05x; slice set1/set2/set4
1.7-10.7x → 1.05-1.07x; householder 1.53/1.55x → 1.10/1.16x).

Case-level table (the per-kernel table is in the campaign's `paired.md`):

## Per-case totals

| Case | Baseline total (us/iter) | Candidate total (us/iter) | Ratio | Launches (base/cand) | Baseline kernels | Candidate kernels | Validity | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `elementwise-add-f32-1000` | 1.9 | 2.0 | 1.08 | 1/1 | add_float | t4a_cuda_add | 1.000 | ok |
| `elementwise-add-f32-1000000` | 5.4 | 6.1 | 1.13 | 1/1 | add_float | t4a_cuda_add | 1.006 | REGRESSION |
| `elementwise-add-f32-100000000` | 759.1 | 766.2 | 1.01 | 1/1 | add_float | t4a_cuda_add | 1.001 | ok |
| `elementwise-exp-f32-10000000` | 56.5 | 57.1 | 1.01 | 1/1 | exp_float | t4a_cuda_unary | 1.001 | ok |
| `elementwise-mul-c64-10000000` | 285.8 | 286.3 | 1.00 | 1/1 | mul_complex | t4a_cuda_mul | 1.001 | ok |
| `elementwise-div-c64-10000000` | 293.0 | 293.6 | 1.00 | 1/1 | div_complex_parts | t4a_cuda_binary | 1.001 | ok |
| `fusion-chain-f32-1000` | 2.0 | 2.0 | 1.00 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.000 | ok |
| `fusion-chain-f32-10000000` | 59.4 | 58.8 | 0.99 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.001 | ok |
| `fusion-add-mul-c64-10000000` | 193.8 | 193.9 | 1.00 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.001 | ok |
| `fusion-two-live-outs-f32-10000000` | 79.9 | 80.4 | 1.01 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.001 | ok |
| `prepare-fused-cold-compile` | 2.0 | 2.0 | 0.98 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.000 | ok |
| `layout-copy-f32-4096sq` | 105.7 | 96.4 | 0.91 | 1/1 | materialize_strided_kernel | t4a_cuda_copy_exact | 1.001 | ok |
| `layout-transpose-f32-4096sq` | 105.2 | 108.9 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.002 | ok |
| `layout-transpose-f32-4093x4099` | 102.5 | 106.9 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.014 | ok |
| `layout-transpose-f32-4099x4093` | 102.6 | 106.3 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.003 | ok |
| `layout-transpose-offset-f32-4096sq` | 104.7 | 109.1 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.006 | ok |
| `layout-transpose-batched-f32-64x256x256` | 25.1 | 26.3 | 1.05 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.064 | ok |
| `layout-transpose-32x8-p1-v1-f32-4096sq` | 90.2 | 91.5 | 1.01 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.003 | ok |
| `layout-transpose-16x8-p1-v2-f32-4096sq` | 94.7 | 95.4 | 1.01 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.002 | ok |
| `layout-materialize-negative-stride-f32-4096sq` | 105.6 | 122.5 | 1.16 | 1/1 | materialize_strided_kernel | t4a_cuda_materialize_strided | 1.000 | REGRESSION |
| `layout-slice-f32-4096sq-set1` | 18.8 | 19.9 | 1.06 | 1/1 | slice_kernel | t4a_cuda_slice | 1.000 | ok |
| `layout-slice-f32-4096sq-set2` | 39.1 | 41.9 | 1.07 | 1/1 | slice_kernel | t4a_cuda_slice | 1.001 | ok |
| `layout-slice-f32-4096sq-set3` | 2.3 | 3.0 | 1.32 | 1/1 | slice_kernel | t4a_cuda_slice | 1.005 | REGRESSION |
| `layout-slice-f32-4096sq-set4` | 143.9 | 151.6 | 1.05 | 1/1 | slice_kernel | t4a_cuda_slice | 1.000 | ok |
| `layout-pad-f32-4096sq-set1` | 332.9 | 293.4 | 0.88 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set2` | 661.1 | 583.3 | 0.88 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set3` | 895.5 | 797.4 | 0.89 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set4` | 1224.9 | 1088.5 | 0.89 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `reduce-sum-f32-axis-511` | 8.1 | 9.6 | 1.19 | 1/1 | reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-axis-512` | 12.4 | 15.2 | 1.23 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-axis-513` | 12.2 | 15.3 | 1.25 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-axis-600` | 12.3 | 14.8 | 1.21 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-4096sq-axis0` | 75.9 | 86.9 | 1.14 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.012 | REGRESSION |
| `reduce-sum-f32-4096sq-axis1` | 207.6 | 190.0 | 0.91 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.022 | ok |
| `reduce-sum-f64-4096sq-axis0` | 93.5 | 98.1 | 1.05 | 2/2 | reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.006 | ok |
| `reduce-sum-c64-4096sq-axis0` | 173.3 | 175.7 | 1.01 | 2/2 | reduce_sum_complex|reduce_sum_complex_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.004 | REGRESSION |
| `reduce-max-i32-4096sq-axis0` | 76.0 | 86.9 | 1.14 | 2/2 | reduce_max_int|reduce_max_int_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.008 | REGRESSION |
| `reduce-sum-f32-full-1000000` | 17.4 | 20.0 | 1.15 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.005 | REGRESSION |
| `reduce-sum-f32-full-10000000` | 54.4 | 57.4 | 1.06 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.011 | REGRESSION |
| `indexing-gather-f32-1024x256-table1000` | 4.8 | 14.4 | 2.97 | 1/1 | gather_kernel | t4a_cuda_gather | 1.007 | REGRESSION |
| `indexing-gather-f32-1024x256-table100000` | 9.6 | 16.3 | 1.70 | 1/1 | gather_kernel | t4a_cuda_gather | 1.013 | REGRESSION |
| `indexing-gather-f32-1024x256-table1000000` | 15.9 | 19.6 | 1.23 | 1/1 | gather_kernel | t4a_cuda_gather | 1.005 | REGRESSION |
| `indexing-scatter-add-f32-1e6-into-16` | 212.7 | 216.7 | 1.02 | 2/2 | scatter_copy_kernel|scatter_float_kernel | t4a_cuda_copy_exact|t4a_cuda_scatter_add | 1.000 | ok |
| `indexing-scatter-add-c64-1e6-into-16` | 1499.4 | 1502.5 | 1.00 | 2/2 | scatter_copy_kernel|scatter_complex_kernel | t4a_cuda_copy_exact|t4a_cuda_scatter_add | 1.000 | REGRESSION |
| `linalg-rrqr-256x256-f64` | 5331.6 | 8108.4 | 1.52 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.009 | REGRESSION |
| `linalg-rrqr-256x256-c64` | 6208.1 | 9151.2 | 1.47 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.008 | REGRESSION |
| `linalg-rrqr-1024x256-f64` | 9768.4 | 12612.8 | 1.29 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.009 | REGRESSION |
| `linalg-rrqr-1024x256-c64` | 13103.0 | 19508.8 | 1.49 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.001 | REGRESSION |
| `linalg-lu-apply-pivots-1024-f64` | 756.6 | 1261.9 | 1.67 | 1/1 | lu_apply_pivots | t4a_cuda_lu_apply_pivots | 1.000 | REGRESSION |
| `linalg-qr-apply-phase-1024x256-f64` | 6.8 | 11.6 | 1.72 | 2/2 | qr_phase_real|qr_apply_phase_real | t4a_cuda_qr_phase|t4a_cuda_qr_apply_phase | 1.010 | REGRESSION |
| `linalg-qr-apply-phase-1024x256-c64` | 8.3 | 12.7 | 1.54 | 2/2 | qr_phase_c64|qr_apply_phase_complex | t4a_cuda_qr_phase|t4a_cuda_qr_apply_phase | 1.011 | REGRESSION |
| `linalg-householder-explicit-v-1024x256-f64` | 4.5 | 4.9 | 1.10 | 1/1 | householder_explicit_v | t4a_cuda_householder_explicit_v | 1.007 | REGRESSION |
| `linalg-householder-explicit-v-1024x256-c64` | 4.7 | 5.4 | 1.16 | 1/1 | householder_explicit_v | t4a_cuda_householder_explicit_v | 1.000 | REGRESSION |

## 5. Dispositions for the remaining regressions

Every remaining case was measured again after the class fix; none is left without a cause.
The plan (§3 of the slice note) allows two dispositions: a residual constant factor that
the port cannot remove without changing a documented contract, and a recorded trade.

| Family | Ratio | Cause (measured, not assumed) |
| --- | --- | --- |
| `reduce-*` (8 cases) | 1.14-1.25x | The split pass is 71.7 → 81.7 us at 4096²-axis0 and the combine pass 4.1 → 5.1 us; both are instruction-bound on the wide quotient walk and the fixed launch cost. A narrow (32-bit) quotient walk was implemented and measured **slower** (16.21 → 16.70 us at axis-512, both passes), and a plane-width specialization gains only 1.8-2.2 %, so neither was kept. Residual. |
| `indexing-gather-*` (3 cases) | 1.23-2.97x | The declared trade of the metadata class: the runtime dimension map is resolved with masked scans instead of L1-resident local reads, which costs a longer dependency chain on the small-table case (4.8 → 14.4 us) and is still faster than the local-memory version on the two larger tables. Residual with the trade recorded. |
| `linalg-lu-apply-pivots` | 1.67x | Dominated by the baseline's O(k) dependent pivot walk (1024 dependent pivot loads per element at k = 1024), which the port reproduces. Removing the metadata traffic (432 B, 438 local ops → 0) did not move the time, so the residual is the walk itself, not metadata. Residual. |
| `linalg-qr-apply-phase-*` (2 cases) | 1.54-1.72x | `qr_apply_phase` is 4.2 → 8.5 us at 1024x256; the per-element guard branches of the unrolled axis walk replace what the baseline kept in L1-resident local reads, in a kernel with only a few instructions of work per element. Residual, constant factor. |
| `linalg-rrqr-*` (4 cases) | 1.29-1.52x | Revision 8 requires `apply_reflector`'s target to be disjoint from `packed`, so the candidate performs one `copy_exact` per step (255 extra launches, 2.1 us each ≈ 0.53 ms of the 2.8 ms gap). The rest is per-entry: `select_pivot` 3.5 → 5.6 us, `extract_r` 2.6 → 4.5 us, `identity_shifted` 2.2 → 4.0 us are single-block, latency-bound launches whose per-element work the baseline does inside one fused region. The copy is a contract cost the tenferro consumer will also pay; the per-entry part is a residual. |
| `layout-slice-f32-4096sq-set3` | 1.32x | 2.3 → 3.0 us, a single launch of a few thousand elements; the gap is below the resolution at which the two arms' launch overhead is comparable. Residual. |
| `layout-materialize-negative-stride` | 1.16x | 105.6 → 122.5 us; the kernel had no local traffic before or after. The narrow quotient walk helped 6 % (121.1 → 114.0 us in the unit's own harness); the rest is 64-bit coordinate·stride arithmetic that the baseline vectorizes. Residual. |
| `elementwise-add-f32-1000000` | 1.13x | 5.4 → 6.1 us at one million elements. A 32-input-index variant was measured identical (6.30 vs 6.27 us), so this is the memory path, not index width. Residual. |
| `indexing-scatter-add-c64-1e6-into-16` | 1.00x total | Flagged because a constituent (`copy_exact`, 1.8 → 2.1 us) crosses the line; the case total is at parity. Recorded, not fixed. |
| `linalg-householder-explicit-v-*` | 1.10/1.16x | 4.5 → 4.9 and 4.7 → 5.4 us; improved from 1.53/1.55x by the metadata fix, residual above the line for the same few-instruction-per-element reason. |

## 6. Evidence-tooling corrections made in this slice

1. Kernel pairing is now an explicit per-case correspondence with an instance-count
   agreement check (the old cost-ranked pairing had mis-paired `prefix_rank`↔
   `initialize_q` and `apply_reflector`↔`select_pivot`; it would mis-pair again after any
   successful optimization).
2. A campaign requires a fresh output directory, no reuse of existing captures, no
   swallowed capture failures, and every declared case × repetition; an incomplete
   campaign exits non-zero with no table. The case list comes from the declaration, not
   from the stdout files that happen to exist.
3. `scripts/resource-usage.sh` counts predicated local-memory instructions
   (`@P0 LDL.64`); the pre-fix counts were an undercount (`scatter_add` 1208 → 1840,
   `gather` 1456 → 1736 on the same sources), and the output is labelled a static
   instruction count.
4. The RRQR case uses disjoint operands and loads the module that owns each entry
   (`identity_shifted` lives in `linalg_aux.cu`).
5. The RRQR and QR sequences state what an iteration measures: the region starts from one
   factorization of the same input on both arms and then measures repeated factorizations
   of the evolving state; there is no per-iteration restore claim.
6. One declaration was over-broad: the reduce cases declared both the split and the
   combine row, so `reduce-sum-f32-axis-511` (a single-pass case) failed the pairing check.
   A declared row that neither side submits is now skipped instead of rejected; the table
   was regenerated from the same campaign artifacts, with no re-measurement.

## 7. Verification

```bash
cargo fmt --check                                     # clean
cargo clippy -j 12 --all-targets -- -D warnings       # clean
cargo test -j 12                                      # 72 + 7 pass
cargo test -j 12 --features gpu-tests                 # 191 pass, 0 failed (local A100)
PATH=/usr/local/cuda-12.6/bin:$PATH scripts/resource-usage.sh
bash bench/run-paired.sh --out <fresh dir> --reps 3 ….
```

`cargo clippy --all-targets --features gpu-tests -- -D warnings` fails on the base commit
`40857fb` and on this branch alike (lint drift in `tests/gpu_elementwise.rs`,
`tests/gpu_linalg_aux.rs`, `tests/gpu_rrqr.rs`, `tests/gpu_fusion.rs`; identical per-target
counts on both, reproduced with toolchains 1.96 and 1.99). It is not caused by this diff,
which touches no Rust test file, and is recorded rather than fixed here.

## 8. Not checked

- No end-to-end tenferro measurement: these are kernel-level device times, and the
  adoption gate also asks for end-to-end numbers, which belong to the tenferro switch.
- The 64-bit fallbacks of the transpose and stride paths are not executed by any case
  (they need a >2^31-element span); they are symbol-identical and compile cleanly.
- No register/occupancy sweep for the residual families; their causes above are
  instruction-level, not occupancy-level, per the static table and the unit's A/B timings.
- No hosted CI; CUDA tests stay local-only (R-ci-pr-60).
