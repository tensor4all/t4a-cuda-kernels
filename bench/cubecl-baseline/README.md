# CubeCL baseline side of the paired A100 performance evidence

This directory holds the baseline half of the frozen paired experiment
([`docs/design/kernel-parity-benchmark.md`](../../docs/design/kernel-parity-benchmark.md),
case list of `2026-10-21-complete-port-perf.md` §4.1): the patch that turns a
throwaway tenferro-rs worktree at
`14bc1ab933e3c7a9c966e08355670331ab9d993c` into the matched benchmark peer of
`examples/bench_kernels.rs`, and the reproduction steps for it.

The baseline harness calls the **private CubeCL launchers** directly rather than
tenferro's public operation API. That is deliberate: the launchers are
`pub(crate)`, so the harness has to live inside `tenferro-gpu` and
`tenferro-linalg`, and calling them directly is what makes both sides run a
matched kernel sequence (`add_float` vs `t4a_cuda_add`, `tiled_transpose_kernel`
vs `t4a_cuda_tiled_transpose`, and so on). Operation-level numbers are therefore
not used anywhere in the comparison.

## What the patch adds

| File | Content |
| --- | --- |
| `crates/tenferro-gpu/src/cubecl/bench_harness.rs` | elementwise, fusion, layout/transpose, reduction, gather and scatter cases; `#[test] #[ignore] fn bench_tenferro_gpu` |
| `crates/tenferro-linalg/src/gpu/linalg/rank_revealing_qr/bench_harness.rs` | LU apply-pivots, QR phase/apply-phase, Householder explicit-V and the RRQR per-step loop; `#[test] #[ignore] fn bench_tenferro_linalg` |
| `crates/tenferro-gpu/src/cubecl/mod.rs` | `#[cfg(all(test, feature = "cuda"))] mod bench_harness;` |
| `crates/tenferro-linalg/src/gpu/linalg/rank_revealing_qr.rs` | `#[cfg(all(test, feature = "cuda"))] mod bench_harness;` |

Both harnesses implement the frozen protocol: 10 warm-ups outside the profiler
range, `cuProfilerStart`, 100 timed iterations, one barrier at the observation
boundary, `cuProfilerStop` (`nsys --capture-range=cudaProfilerApi`). Buffers,
plans and modules are prepared before the range; the region performs no
transfer, allocation, zero fill or host wait per iteration except where the
baseline's own launcher allocates its outputs (the fusion cases, reported as
`device_allocations_per_iter=1` or `2`).

Case selection is `T4A_BENCH_CASES` (comma-separated, or `all`).

Notes on fidelity, all of them recorded in the worklog:

- **Same inputs on both arms.** Every case's host input is generated with the same
  generator, seed, element order and complex scale as `examples/bench_kernels.rs`
  (the complex generator is `f64() * 4.0` per component on both sides), and both
  harnesses print an `input_digest` — FNV-1a 64 over the case's initial host bytes,
  one length-prefixed part per operand — in `BENCH_META` and in the `BENCH_META` of
  the cold-prepare case. `bench/summarize.py` does not compare the digests; they are
  there so a run can be audited: equal digests on the two sides mean equal data.
  The one case whose digest is not a *numerical* input is not exempt from this:
  `prepare-fused-cold-compile` uses the same `0x5555_6666` chain generator as
  `fusion-chain-f32-*`.
- **RRQR factorization state.** The RRQR arm is a caller-side host loop in the
  baseline too (`rank_revealing_qr_typed`). The harness mirrors that loop launch
  by launch, excludes the input `cuMemcpyDtoDAsync` clone the host function
  performs before the loop on both sides, and runs one reference-checked
  factorization before the measured region. **Neither side restores the work
  buffer between iterations**, so the region starts from the state that
  reference-check factorization left and every further iteration factorizes the
  previous iteration's leftover — the same starting state and the same behaviour
  on the candidate arm. No number in the campaign is a "fresh matrix per
  iteration" measurement.
- **RRQR operands.** The pinned baseline applies each reflector in place
  (`launch_apply(work, work, …)`); its write set (columns `step+1..`) is disjoint
  from the reflector column it reads, which is what its DSL allows. Revision 8 of
  the candidate ABI requires the packed reflector storage and the target to be
  disjoint allocations, so the candidate arm copies each step's reflector vector
  into its own buffer (one `t4a_cuda_copy_exact` launch per step with a trailing
  block). `bench/summarize.py` declares that kernel as a candidate-side
  helper instead of pairing it with a baseline kernel, and the candidate's launch
  count for the RRQR cases is therefore `7k+3` instead of `6k+3`.
- **Q/R state.** `qr_apply_phase` scales Q's columns and R's rows in place, so its
  iterations compound. Neither arm restores Q or R after its reference check: the
  measured region starts from the state one application of the two kernels left, on
  both sides.
- **`check=n/a`.** The cold-prepare case on this side declares that it runs no
  numerical reference (`check=n/a`) because the baseline has no compile-count hook
  and the case only observes cold/warm wall time; `bench/summarize.py` accepts `ok`
  or `n/a` from the baseline arm and only `ok` from the candidate.
- **Compile counts** are not observable on this side: the pinned tenferro-rs
  records the missing NVRTC observation hook
  (`docs/design/cubecl-011-compile-count-hook.md`), so the cold-prepare case
  reports wall-clock compile/load times only (`not_observable` for the counts).
- `nsys` needs `cuProfilerStart`/`cuProfilerStop`; the harness resolves them from
  `libcuda.so.1` with `dlopen`/`dlsym` because cudarc does not bind them.

## Reproduction

The patch must be re-applied whenever this directory changes; a campaign always
starts from a fresh baseline build and a fresh output directory.

```bash
# 1. Throwaway worktree at the pinned baseline commit (never write in the
#    primary tenferro-rs checkout).
git -C ~/tensor4all/tenferro-rs worktree add --detach \
    ~/tensor4all/.worktrees/t4a-baseline-14bc1ab \
    14bc1ab933e3c7a9c966e08355670331ab9d993c

# 2. Apply this patch (check it first: a changed patch must apply cleanly).
git -C ~/tensor4all/.worktrees/t4a-baseline-14bc1ab apply --check \
    <this repository>/bench/cubecl-baseline/baseline.patch
git -C ~/tensor4all/.worktrees/t4a-baseline-14bc1ab apply \
    <this repository>/bench/cubecl-baseline/baseline.patch

# 3. Build only the two lib test binaries, in release, with their own target
#    directory (the crates' `cuda` feature; tenferro-linalg's `cuda` feature
#    pulls tenferro-gpu/cuda in turn).
CARGO_TARGET_DIR=<run dir>/baseline-target \
    cargo test --release --no-run -p tenferro-gpu -p tenferro-linalg --features cuda

# 4. Run one case (the test binaries are `#[ignore]`d; run-paired.sh resolves the
#    newest ones automatically).
gpu_bin=$(ls -t "$CARGO_TARGET_DIR"/release/deps/tenferro_gpu-* | grep -v '\.d$' | head -1)
T4A_BENCH_CASES=elementwise-add-f32-1000000 \
    "$gpu_bin" --exact cubecl::bench_harness::bench_tenferro_gpu --ignored --nocapture

linalg_bin=$(ls -t "$CARGO_TARGET_DIR"/release/deps/tenferro_linalg-* | grep -v '\.d$' | head -1)
T4A_BENCH_CASES=linalg-rrqr-256x256-f64 \
    "$linalg_bin" --exact gpu::linalg::rank_revealing_qr::bench_harness::bench_tenferro_linalg \
    --ignored --nocapture
```

Each case prints a `BENCH_META` line (physical launches per iteration, scratch
bytes, transfers, allocations, zero fills, host waits, `input_digest`,
reference-check verdict) and, for the prepared-fusion cases, a `BENCH_PREPARE` line.

## The paired run

`../run-paired.sh` drives both sides under
`nsys profile --capture-range=cudaProfilerApi`, one process per case per side per
repetition, keeps the raw exports under its `--out` directory (outside this
repository) and prints the paired table via `../summarize.py`. `--out` is
required and must be fresh: nothing is reused from an earlier campaign, a failed
capture is never swallowed, and a campaign that is missing any declared
case × repetition × side reports itself incomplete and produces no verdict.
The worktree and the target directory are disposable and are deleted after
review.

`python3 bench/test_summarize.py -v` runs the CPU-only fixture tests of the
summariser (no GPU, no `nsys`): a complete synthetic campaign, a missing required
kernel row, a legitimate reduction alternative, instance counts that cancel across
repetitions, a failed numerical check, a mismatched case identity and a declared
launch count that disagrees with the capture.
