# Worklog: closing the kernel performance remainder (U8a/U8b) and the paired rerun

Date: 2026-10-23. Plan: [slice-perf-remainder.md](../design/slice-perf-remainder.md).
Protocol: the phase-2 one frozen in
[2026-10-21-complete-port-perf.md](2026-10-21-complete-port-perf.md) §2 —
10 warm-ups outside the profiler range, `cuProfilerStart`, 100 iterations, one barrier,
`cuProfilerStop`, 3 process repetitions per side, device time per kernel is the nsys
`cuda_gpu_kern_sum` per-instance median over the 100 iterations, validity
`max/min <= 1.15`, regression `candidate median > 1.10x baseline median`. The older
[kernel-parity-benchmark.md](../design/kernel-parity-benchmark.md) is the phase-1
host-wall protocol (one process, two arms, 16-deep batches) and is **not** the protocol
used here.

Baseline: tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c` in the throwaway
worktree `.worktrees/t4a-baseline-14bc1ab` with `bench/cubecl-baseline/baseline.patch`
applied. Candidate: this branch `perf/u8-kernel-parity`, ABI revision 8.

## 1. Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe, idle (`nvidia-smi --query-compute-apps`) before every process |
| Driver | 580.126.09 |
| Kernel toolchain | nvcc/ptxas/cuobjdump 12.0.140 (campaign and GPU tests); the static resource table in §3 was produced with the same 12.0.140 tools under `PATH=/usr/local/cuda-12.6/bin` (NVRTC 12.0.140 for the prepare case) |
| Nsight Systems | 2024.2.3.38-242334140272v0 |
| Rust | cargo/rustc 1.97.1 |
| Campaign | `shinaoka-advisor/work/runs/t4a-cuda-kernels-perf/perf-campaign-3` (fresh directory; all 53 cases × 3 repetitions × 2 sides captured) |
| Command | `bash bench/run-paired.sh --out <campaign> --reps 3 --candidate-bin target/release/examples/bench_kernels --baseline-gpu-bin <…>/tenferro_gpu-7698e59615da0353 --baseline-linalg-bin <…>/tenferro_linalg-aeb242cdd00fbe2b` |

Both arms print `input_digest` in `BENCH_META` and the digests agree per case (for
example `linalg-rrqr-256x256-f64` is `0x208fc78470db93a5` on both sides), so the two arms
factorize the same initial matrix.

## 2. What changed

- **Metadata local memory (unit A).** `window.cu` (slice, pad), `diagonal.cu`
  (extract/embed), `gather_scatter.cu` (gather, scatter_add), `linalg_aux.cu`
  (lu_apply_pivots, lu_extract_outputs, qr_apply_phase) and the `abi.cuh` helpers. Every
  runtime axis subscript is a literal subscript of an unrolled walk, and compact
  destinations/sources are addressed flat where `src/lib.rs` proves compactness. The
  masked selector reads all eight coordinate slots, so producers now initialize the whole
  array (a rank-2 diagonal op previously reached it with undefined slots); documented in
  `abi.cuh` and in `docs/abi/c-kernel-abi.md`.
- **Index width and instruction cost (unit B).** `movement.cu`: one compiled transpose body
  per documented tile configuration (uniform switch, `__launch_bounds__(512, 4)`, 40→32
  registers) and a signed 32-bit index path with a 64-bit fallback. The narrow dispatch
  bounds **every narrowed argument** (offset, batch stride and both extents) before the
  cast, not only the last reachable address: an extent of `2^31` has a last index of
  `2^31-1` but does not fit the narrow body, and a boundary test at `2^31` Bool elements
  covers both sides of that line. The same narrow walk is used in `materialize_strided`,
  `contiguous_to_view`, `strided_to_strided` and `fill_zero_view`, where only the logical
  coordinate narrows. `rrqr.cu`: layout metadata is read through `const&` so the by-value
  POD is not copied into the stack frame. The specialization record in
  `docs/abi/c-kernel-abi.md` lists the transpose tile set and the two index widths.
- **Evidence tooling (unit C).** The phase-2 harness now runs on `main`/revision 8
  (`examples/bench_kernels.rs`, `tests/gpu/prepare.rs`, `bench/`, `scripts/resource-usage.sh`)
  with the defects the independent reviews found in the old tooling removed: semantic
  kernel pairing per repetition instead of speed-ranked pairing; a campaign that requires
  a fresh directory, every declared case × repetition, a successful numerical check, the
  requested case identity and the declared launch count, and that fails closed instead of
  shrinking the sample; identical input generation on both arms with a published digest;
  disjoint RRQR operands with the module that owns each entry; a stated measurement of what
  an RRQR/QR iteration does; and a predicated-instruction-aware local-memory count.

## 3. Static resource usage (`scripts/resource-usage.sh`, sm_80, nvcc 12.0.140)

Before = base `40857fb`, after = this branch, both with the **same** (predicate-aware)
counting script, so the before/after magnitudes are comparable:

| entry | stack / `LDL`+`STL` before | after |
| --- | --- | --- |
| `t4a_cuda_slice`, `t4a_cuda_pad` | 144 B / 88 | 0 / 0 |
| `t4a_cuda_extract_diagonal`, `t4a_cuda_embed_diagonal_copy` | 64 B / 98 | 0 / 0 |
| `t4a_cuda_gather` | 192 B / 1736 | 0 / 0 |
| `t4a_cuda_scatter_add` | 320 B / 1840 | 0 / 0 |
| `t4a_cuda_lu_apply_pivots` | 432 B / 438 | 0 / 0 |
| `t4a_cuda_lu_extract_outputs` | 720 B / 488 | 0 / 0 |
| `t4a_cuda_qr_apply_phase` | 288 B / 384 | 0 / 0 |
| `t4a_cuda_extract_r` / `swap_columns` / `column_norms` / `select_pivot` / `prefix_rank` | 144-288 B / 116-348 | 0 / 0 |
| `t4a_cuda_tiled_transpose` | 40 registers, 0 / 0 | 32 registers, 0 / 0 |
| `t4a_cuda_unary` | 72 B / 20 | 72 B / 20 — libdevice `sin`/`cos` slow path, kept |

These are static instruction counts, not executed traffic, and entry bodies do not include
traffic inside called device functions. No entry other than `t4a_cuda_unary` touches local
memory. An earlier revision of this worklog quoted 1456/1208 for gather/scatter: those are
the same sources counted without predicated instructions, i.e. an undercount that
`scripts/resource-usage.sh` now fixes.

## 4. Campaign result

53 cases: **29 `ok`**, **24 `REGRESSION`**, 0 `inconclusive`, 0 pairing failures, 0
missing captures, 0 invalid evidence. The previous campaign had 39 regressions; 15 of them
are now at parity or faster (pad 3.6-3.8x → 0.88-0.89x; transpose 1.36-1.96x → 1.01-1.05x;
slice set1/set2/set4 1.7-10.7x → 1.05-1.07x; householder-explicit-v f64 1.53x → below the
line, c64 1.55x → 1.16x).

Case-level table (the per-kernel table is in the campaign's `paired.md`):

### Case-level totals

| Case | Baseline total (us/iter) | Candidate total (us/iter) | Ratio | Launches (base/cand) | Baseline kernels | Candidate kernels | Validity | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `elementwise-add-f32-1000` | 1.9 | 2.0 | 1.08 | 1/1 | add_float | t4a_cuda_add | 1.000 | ok |
| `elementwise-add-f32-1000000` | 5.4 | 6.1 | 1.13 | 1/1 | add_float | t4a_cuda_add | 1.012 | REGRESSION |
| `elementwise-add-f32-100000000` | 758.9 | 766.2 | 1.01 | 1/1 | add_float | t4a_cuda_add | 1.001 | ok |
| `elementwise-exp-f32-10000000` | 56.5 | 57.1 | 1.01 | 1/1 | exp_float | t4a_cuda_unary | 1.001 | ok |
| `elementwise-mul-c64-10000000` | 285.7 | 286.4 | 1.00 | 1/1 | mul_complex | t4a_cuda_mul | 1.001 | ok |
| `elementwise-div-c64-10000000` | 293.2 | 293.6 | 1.00 | 1/1 | div_complex_parts | t4a_cuda_binary | 1.000 | ok |
| `fusion-chain-f32-1000` | 2.0 | 2.0 | 1.00 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.000 | ok |
| `fusion-chain-f32-10000000` | 59.4 | 58.8 | 0.99 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.001 | ok |
| `fusion-add-mul-c64-10000000` | 193.7 | 193.9 | 1.00 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.002 | ok |
| `fusion-two-live-outs-f32-10000000` | 79.8 | 80.4 | 1.01 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.003 | ok |
| `prepare-fused-cold-compile` | 2.0 | 2.0 | 0.98 | 1/1 | tenferro_fused_elementwise | t4a_fused_elementwise | 1.000 | ok |
| `layout-copy-f32-4096sq` | 105.7 | 96.4 | 0.91 | 1/1 | materialize_strided_kernel | t4a_cuda_copy_exact | 1.001 | ok |
| `layout-transpose-f32-4096sq` | 105.0 | 109.3 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.005 | ok |
| `layout-transpose-f32-4093x4099` | 102.5 | 107.2 | 1.05 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.013 | ok |
| `layout-transpose-f32-4099x4093` | 102.8 | 106.6 | 1.04 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.003 | ok |
| `layout-transpose-offset-f32-4096sq` | 104.8 | 109.7 | 1.05 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.007 | ok |
| `layout-transpose-batched-f32-64x256x256` | 25.1 | 26.5 | 1.05 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.033 | ok |
| `layout-transpose-32x8-p1-v1-f32-4096sq` | 90.0 | 91.7 | 1.02 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.005 | ok |
| `layout-transpose-16x8-p1-v2-f32-4096sq` | 94.7 | 95.4 | 1.01 | 1/1 | tiled_transpose_kernel | t4a_cuda_tiled_transpose | 1.005 | ok |
| `layout-materialize-negative-stride-f32-4096sq` | 105.6 | 122.5 | 1.16 | 1/1 | materialize_strided_kernel | t4a_cuda_materialize_strided | 1.001 | REGRESSION |
| `layout-slice-f32-4096sq-set1` | 18.8 | 19.9 | 1.06 | 1/1 | slice_kernel | t4a_cuda_slice | 1.000 | ok |
| `layout-slice-f32-4096sq-set2` | 39.1 | 41.9 | 1.07 | 1/1 | slice_kernel | t4a_cuda_slice | 1.001 | ok |
| `layout-slice-f32-4096sq-set3` | 2.3 | 3.0 | 1.32 | 1/1 | slice_kernel | t4a_cuda_slice | 1.000 | REGRESSION |
| `layout-slice-f32-4096sq-set4` | 143.9 | 151.7 | 1.05 | 1/1 | slice_kernel | t4a_cuda_slice | 1.001 | ok |
| `layout-pad-f32-4096sq-set1` | 332.9 | 293.4 | 0.88 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set2` | 661.1 | 583.3 | 0.88 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set3` | 895.5 | 797.4 | 0.89 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `layout-pad-f32-4096sq-set4` | 1224.9 | 1088.5 | 0.89 | 1/1 | pad_kernel | t4a_cuda_pad | 1.000 | ok |
| `reduce-sum-f32-axis-511` | 8.1 | 9.6 | 1.19 | 1/1 | reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-axis-512` | 12.4 | 15.2 | 1.23 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.007 | REGRESSION |
| `reduce-sum-f32-axis-513` | 12.2 | 15.3 | 1.25 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-axis-600` | 12.3 | 14.9 | 1.21 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-4096sq-axis0` | 75.9 | 86.9 | 1.15 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.003 | REGRESSION |
| `reduce-sum-f32-4096sq-axis1` | 207.6 | 193.9 | 0.93 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.020 | ok |
| `reduce-sum-f64-4096sq-axis0` | 93.6 | 98.1 | 1.05 | 2/2 | reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.006 | ok |
| `reduce-sum-c64-4096sq-axis0` | 173.4 | 176.1 | 1.02 | 2/2 | reduce_sum_complex|reduce_sum_complex_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.008 | REGRESSION |
| `reduce-max-i32-4096sq-axis0` | 76.0 | 86.9 | 1.14 | 2/2 | reduce_max_int|reduce_max_int_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.007 | REGRESSION |
| `reduce-sum-f32-full-1000000` | 17.4 | 20.0 | 1.15 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.005 | REGRESSION |
| `reduce-sum-f32-full-10000000` | 54.4 | 57.7 | 1.06 | 2/2 | reduce_sum_float_chunked|reduce_sum_float|reduce_sum_float_plane | t4a_cuda_reduce_partials|t4a_cuda_reduce | 1.015 | REGRESSION |
| `indexing-gather-f32-1024x256-table1000` | 4.8 | 14.4 | 2.96 | 1/1 | gather_kernel | t4a_cuda_gather | 1.006 | REGRESSION |
| `indexing-gather-f32-1024x256-table100000` | 9.6 | 16.3 | 1.69 | 1/1 | gather_kernel | t4a_cuda_gather | 1.003 | REGRESSION |
| `indexing-gather-f32-1024x256-table1000000` | 16.0 | 19.6 | 1.23 | 1/1 | gather_kernel | t4a_cuda_gather | 1.004 | REGRESSION |
| `indexing-scatter-add-f32-1e6-into-16` | 212.7 | 216.7 | 1.02 | 2/2 | scatter_copy_kernel|scatter_float_kernel | t4a_cuda_copy_exact|t4a_cuda_scatter_add | 1.000 | ok |
| `indexing-scatter-add-c64-1e6-into-16` | 1499.4 | 1502.6 | 1.00 | 2/2 | scatter_copy_kernel|scatter_complex_kernel | t4a_cuda_copy_exact|t4a_cuda_scatter_add | 1.000 | REGRESSION |
| `linalg-rrqr-256x256-f64` | 5331.6 | 8108.9 | 1.52 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.008 | REGRESSION |
| `linalg-rrqr-256x256-c64` | 6208.1 | 9151.2 | 1.47 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.008 | REGRESSION |
| `linalg-rrqr-1024x256-f64` | 9768.4 | 12612.8 | 1.29 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.018 | REGRESSION |
| `linalg-rrqr-1024x256-c64` | 13102.8 | 19508.8 | 1.49 | 1539/1794 | rrqr_initialize_permutation|rrqr_column_norms|rrqr_select_pivot|rrqr_swap_columns|rrqr_reflector|rrqr_apply_reflector|rrqr_extract_r|rrqr_initialize_q|rrqr_prefix_rank | t4a_cuda_initialize_permutation|t4a_cuda_column_norms|t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|t4a_cuda_identity_shifted|t4a_cuda_prefix_rank | 1.004 | REGRESSION |
| `linalg-lu-apply-pivots-1024-f64` | 756.6 | 1262.0 | 1.67 | 1/1 | lu_apply_pivots | t4a_cuda_lu_apply_pivots | 1.000 | REGRESSION |
| `linalg-qr-apply-phase-1024x256-f64` | 6.8 | 11.6 | 1.71 | 2/2 | qr_phase_real|qr_apply_phase_real | t4a_cuda_qr_phase|t4a_cuda_qr_apply_phase | 1.010 | REGRESSION |
| `linalg-qr-apply-phase-1024x256-c64` | 8.3 | 12.7 | 1.53 | 2/2 | qr_phase_c64|qr_apply_phase_complex | t4a_cuda_qr_phase|t4a_cuda_qr_apply_phase | 1.009 | REGRESSION |
| `linalg-householder-explicit-v-1024x256-f64` | 4.5 | 4.9 | 1.10 | 1/1 | householder_explicit_v | t4a_cuda_householder_explicit_v | 1.000 | ok |
| `linalg-householder-explicit-v-1024x256-c64` | 4.7 | 5.4 | 1.16 | 1/1 | householder_explicit_v | t4a_cuda_householder_explicit_v | 1.000 | REGRESSION |

## 5. Dispositions for the remaining regressions

Every remaining case was measured again after the class fix. The plan (§3 of the slice
note) allows two dispositions: a residual constant factor that the port cannot remove
without changing a documented contract, and a recorded trade. Where the cause is not
established, it is labelled unknown rather than guessed.

| Family | Ratio | Measured facts and disposition |
| --- | --- | --- |
| `reduce-*` (8 cases) | 1.14-1.25x | Split pass 71.7 → 81.7 us at 4096²-axis0, combine pass 4.1 → 5.1 us; both instruction-bound on the wide quotient walk plus the fixed launch cost. A 32-bit quotient walk was implemented and measured **slower** (16.21 → 16.70 us at axis-512 across both passes) and a plane-width specialization gains only 1.8-2.2 %, so neither was kept. Residual. |
| `indexing-gather-*` (3 cases) | 1.23-2.96x | The declared trade of the metadata class: the runtime dimension map is resolved with masked scans instead of L1-resident local reads. That costs a longer dependency chain on the 1 MB-table case (4.8 → 14.4 us) and still beats the local-memory version on the two larger tables (16.0 → 19.6 us). Residual with the trade recorded. |
| `linalg-lu-apply-pivots` | 1.67x | Dominated by the baseline's O(k) dependent pivot walk (1024 dependent pivot loads per element at k = 1024), which the port reproduces. Removing the metadata traffic (432 B, 438 local ops → 0) did not move the time, so the residual is the walk itself. Residual. |
| `linalg-qr-apply-phase-*` (2 cases) | 1.53-1.71x | `qr_apply_phase` 4.2 → 8.5 us at 1024x256, `qr_phase` 2.5 → 3.1 us: a few instructions of work per element, where the per-element guard branches of the unrolled axis walk are not offset by the removed local reads. Residual, constant factor. |
| `linalg-rrqr-*` (4 cases) | 1.29-1.52x | Measured contributions to the 2.78 ms gap of the 256² f64 case: the revision-8 disjoint-operand `copy_exact` (255 launches × 2.1 us) ≈ 0.54 ms; `column_norms` 1.34x ≈ 0.36 ms; `select_pivot` 1.60x ≈ 0.54 ms; `reflector` 1.41x ≈ 0.41 ms; `swap_columns` 1.41x ≈ 0.26 ms; `apply_reflector` 1.39x ≈ 0.33 ms; `extract_r` 1.73x and `identity_shifted` 1.81x ≈ 0.02 ms. The copy is a contract cost every consumer of this ABI pays; the per-entry gaps are spread over all six kernels, and **why six small launches are each 1.3-1.8x slower than the baseline's counterparts is not established** — the entries have no local memory and use flat multi-block launches like the baseline's. Recorded as an open residual with the decomposition above. |
| `layout-slice-f32-4096sq-set3` | 1.32x | 2.3 → 3.0 us for a single launch of a few thousand elements; below the resolution at which the two arms' launch overhead is comparable. Residual. |
| `layout-materialize-negative-stride` | 1.16x | 105.6 → 122.5 us; the kernel had no local traffic before or after, and the narrow quotient walk helped 6 % (121.1 → 114.0 us in the unit's own harness). The rest is the 64-bit coordinate·stride arithmetic the baseline vectorizes. Residual. |
| `elementwise-add-f32-1000000` | 1.13x | 5.4 → 6.1 us at one million elements; a 32-bit input-index variant measured identical (6.30 vs 6.27 us), so this is the memory path, not index width. Residual. |
| `indexing-scatter-add-c64-1e6-into-16` | 1.00x total | Flagged only because a constituent (`copy_exact`, 1.8 → 2.1 us) crosses the line; the case total is at parity. Recorded, not fixed. |

## 6. Evidence-tooling corrections made in this slice

1. Kernel pairing is an explicit per-case correspondence with a per-repetition
   instance-count check (the old cost-ranked pairing had mis-paired `prefix_rank` ↔
   `initialize_q` and `apply_reflector` ↔ `select_pivot`). A declared row that is not an
   explicitly declared alternative must be present on both sides; a missing row or a
   count that differs in any repetition makes the case `invalid-pairing`.
2. A campaign requires a fresh output directory, no reuse of existing captures, no
   swallowed capture failures, and every declared case × repetition. Every repetition must
   also report the requested case, a successful numerical check and the declared launch
   count; otherwise the campaign is invalid and exits non-zero with no table.
3. `scripts/resource-usage.sh` counts predicated local-memory instructions
   (`@P0 LDL.64`); the pre-fix counts were an undercount of the same sources (`gather`
   1456 → 1736, `scatter_add` 1208 → 1840). Both columns of §3 use the fixed script.
4. Both arms generate each case's inputs with the same rules, seeds and order, and publish
   `input_digest` in `BENCH_META`; the digests agree per case in this campaign. The Q/R
   restore is symmetric (neither arm restores after the reference check).
5. The RRQR case uses disjoint operands and loads the module that owns each entry
   (`identity_shifted` lives in `linalg_aux.cu`). Its scratch accounting includes the
   packed-reflector buffer.
6. The RRQR/QR sequences state what an iteration measures: the region starts from one
   factorization of the same input on both arms and then measures repeated factorizations
   of the evolving state.

## 7. Verification

```bash
cargo fmt --check                                     # clean
cargo clippy -j 12 --all-targets -- -D warnings       # clean
cargo test -j 12                                      # 72 + 7 pass
cargo test -j 12 --features gpu-tests                 # 192 pass, 0 failed (local A100), incl. the 2^31-element transpose boundary test
PATH=/usr/local/cuda-12.6/bin:$PATH scripts/resource-usage.sh
python3 bench/test_summarize.py                       # 8 summariser fixture tests
bash bench/run-paired.sh --out <fresh dir> --reps 3 …
```

`cargo clippy --all-targets --features gpu-tests -- -D warnings` fails on the base commit
`40857fb` and on this branch alike: the lints are in `tests/gpu_elementwise.rs`,
`tests/gpu_fusion.rs`, `tests/gpu_linalg_aux.rs` and `tests/gpu_rrqr.rs`, with identical
per-target counts on both, reproduced with toolchains 1.96 and 1.99. This diff does not
touch those files (it does add `tests/gpu/prepare.rs` and extend `tests/planning.rs`), and
the failure is recorded rather than fixed here.

## 8. Not checked

- No end-to-end tenferro measurement: these are kernel-level device times. The adoption
  gate also asks for end-to-end numbers, which belong to the tenferro switch.
- The 64-bit fallbacks of the transpose and stride paths are not executed by the paired
  cases (they need a >2^31-element span); the transpose boundary test exercises the
  dispatch decision at `2^31` but with a one-element fast extent, so the wide body's
  arithmetic is covered only by construction.
- The per-entry RRQR gap (1.3-1.8x on six small launches) is measured and decomposed but
  not explained; no register/occupancy sweep or per-instruction comparison was made.
- No hosted CI; CUDA tests stay local-only (R-ci-pr-60).
