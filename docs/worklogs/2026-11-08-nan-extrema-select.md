# NaN-propagating extrema as selects: measurement (2026-11-08)

Plan: [../design/slice-nan-extrema-select.md](../design/slice-nan-extrema-select.md).
Evidence: [evidence/2026-11-08-nan-extrema-select/](evidence/2026-11-08-nan-extrema-select/).

## Environment

primerose, NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA 12.6 (`/usr/local/cuda-12.6`).
Kernels compiled with `nvcc -std=c++17 -cubin -arch=sm_80`, no fast math. Baseline is `main`
at `3c8d82c` (#23 merged); the candidate is the same tree plus this branch's `abi.cuh` diff.
Both were built with `cargo build -j 8 --release --features gpu-tests --example
bench_kernels`, and the two binaries ran alternately, three rounds (main, fix) × 3, with
`knsys.sh` (`nsys profile` over 100 launches, `nsys stats --report cuda_gpu_kern_sum`).
Nothing else was running on the GPU. Every bench run reports `check=ok`. The measured
candidate tree is exactly this branch's commit: the diff against `3c8d82c` is `abi.cuh` plus
`docs/`.

## Results

Medians over the three rounds; host time is the bench's `host_ns_per_op` (100 launches,
pass 1 + pass 2, no host wait). The per-round values are in `nsys/table.md`.

| case | pass 1 µs main → fix | pass 2 µs main → fix | host/op µs main → fix | host ratio |
|---|---|---|---|---|
| `reduce-max-f32-full-1048576` | 8.83 → 8.45 | 6.85 → 5.98 | 18.28 → 17.10 | 0.936 |
| `reduce-min-f32-full-1048576` | 8.90 → 8.42 | 6.56 → 5.63 | 18.15 → 16.61 | 0.915 |
| `reduce-max-f64-full-1048576` | 8.83 → 8.48 | 7.42 → 6.46 | 18.95 → 17.62 | 0.930 |
| `reduce-min-f64-full-1048576` | 8.80 → 8.48 | 7.23 → 6.11 | 18.72 → 17.29 | 0.924 |
| `reduce-max-f32-full-16777216` | 49.23 → 47.10 | 9.14 → 8.45 | 61.05 → 58.38 | 0.956 |
| `reduce-min-f32-full-16777216` | 49.30 → 47.04 | 8.00 → 7.20 | 59.99 → 57.12 | 0.952 |
| `reduce-max-f64-full-16777216` | 90.59 → 88.18 | 9.41 → 9.36 | 102.50 → 100.30 | 0.978 |
| `reduce-min-f64-full-16777216` | 90.61 → 88.08 | 8.83 → 7.62 | 102.06 → 98.37 | 0.964 |
| `reduce-max-f32-4x131072-axis1` | 7.20 → 6.75 | 5.50 → 5.63 | 15.31 → 15.11 | 0.987 |
| `reduce-sum-f32-full-1048576` (control) | 8.13 → 8.13 | 5.25 → 5.38 | 16.01 → 16.08 | 1.004 |
| `reduce-sum-f64-full-1048576` (control) | 8.45 → 8.43 | 5.70 → 5.54 | 16.66 → 16.44 | 0.987 |
| `reduce-sum-f32-full-16777216` (control) | 46.64 → 46.59 | 6.30 → 7.23 | 55.63 → 56.42 | 1.014 |
| `reduce-sum-f64-full-16777216` (control) | 87.41 → 87.46 | 7.01 → 6.43 | 96.95 → 96.57 | 0.996 |
| `reduce-prod-f32-full-1048576` (control) | 8.93 → 8.93 | 11.52 → 11.59 | 23.19 → 22.99 | 0.992 |

- **`MAX`/`MIN`:** host time per operation is 0.915-0.936 at 1M and 0.952-0.978 at 16M.
  - Pass 2 drops by 0.69-1.21 µs in every full cell except `max` f64 16M (9.41 → 9.36).
  - Pass 1 drops by 0.32-0.48 µs at 1M and 2.1-2.5 µs at 16M, so `max`/`min` f32 pass 1 at
    16M (47.04-47.10 µs) is now within 0.51 µs of `sum` (46.59 µs).
- **Controls.** These cells run code whose source did not change, but they share the entry,
  and the entry's code changed around them.
  - Worst host ratio: `reduce-sum-f32-full-16777216` at **1.014** (1.018 / 1.014 / 0.996).
  - Its pass 2 is 6.29-6.30 µs on main and 7.23 / 7.23 / 6.43 µs on the fix.
  - The value looks bimodal, about 6.3 or about 7.2 µs, in both trees. An earlier two-round
    campaign (`nsys-campaign1/`) gave main 7.26 / fix 7.23 in round 1 and main 6.30 / fix 6.30
    in round 2. That campaign is not used for the table: an `nvcc` build ran on the host
    during it.
  - `t4a_cuda_reduce` uses 40 registers in both trees (`ptxas.log`).
  - **Whether the change makes the slower mode more likely is not established.**
  - Smaller pass-2 increases: `sum` f32 1M +0.13 µs, `max` f32 `4x131072` axis 1 +0.13 µs,
    `prod` f32 1M +0.07 µs.
- **Registers.** The pass-1 entries go from 50 to 54, with no spills (`ptxas.log`). Pass 1
  runs one-warp blocks, so the 32-blocks-per-SM limit binds before registers do.

## Correctness

- `bit-identity.log`: the new and old functions, compiled into one GPU program, agree bit for
  bit for `max` and `min` over 67 109 120 pairs per type. The pairs are all 16 × 16 pairs of
  special values (±0, ±1, ±Inf, ±smallest subnormal, ±largest finite, and quiet,
  signalling and negative NaNs with distinct payloads), plus 2^26 random bit patterns, one
  third of them pairs of neighbouring values.
- `cargo-test-gpu-tests.log` (run on `3eb8376`, the code of this branch): `cargo test -j 4 --release --features gpu-tests
  --no-fail-fast` passes (all 15 test binaries and the doc-tests, `# exit 0`). This includes
  `binary_extrema_pow_rem_and_scalar_division`, which checks that the lhs NaN payload
  survives.
- `cargo fmt --check`, `cargo clippy -j 8 --all-targets -- -D warnings` and host `cargo test`
  pass.

## Not measured

- Elementwise `MAX`/`MIN` (`t4a_cuda_binary`) timing. Its SASS changed, and `bench_kernels`
  has no case for it.
- Through tenferro. That needs the pin moved past this change.

## Review follow-up (2026-11-09)

The 2026-11-08 sections above stay as they were. This section answers the three items of the
PR #26 review that held back `kernels 0.1.2` (no version change in this PR). All GPU runs were
on primerose (A100 80GB PCIe, driver 580.178.04, CUDA 12.6 at `/usr/local/cuda-12.6`, nsys
2024.2.3.38), one GPU job at a time under `flock /tmp/gpu-slot.lock`, bench processes pinned
with `taskset -c 0-7`, and cores 0-7 checked idle before each job.

**Toolkit on PATH.** `~/work/gpu-env.sh` sets `CUDA_HOME` but does not put
`/usr/local/cuda-12.6/bin` on `PATH`, so a non-interactive shell finds `/usr/bin/nvcc` (12.0.140),
and `bench_kernels` compiles its cubins with whatever `nvcc` is on `PATH`. Every run below put
12.6 first (`knsys.sh` now does it itself). Which `nvcc` the 2026-11-08 runs used is not recorded.

### 1. `bit-identity.cu` fails loudly

- Every CUDA call goes through `CUDA_CHECK`. That covers the allocations, the copies, the
  memset, the launch (`cudaGetLastError`), `cudaDeviceSynchronize`, the download and the frees.
  Any failure exits 2 before a result line is printed.
- The host counters start at a sentinel, not 0, and the kernel also counts the pairs it compared.
  Unless that count equals `n`, the program exits 2. Exit 1 means a mismatch; exit 0 means every
  pair was compared and none differed.
- `evidence/.../bit-identity-failpath.log` has the passing run (`compared 67109120, mismatches 0`
  for f32 and f64, `# exit 0`) and two injections made in scratch copies with `sed` and deleted
  afterwards. Each copy's diff is printed before it runs.
  - **Every pair counted as a mismatch:** `mismatches 67109120` per type, `total mismatches
    134218240`, `# exit 1`.
  - **The review's reproduction** (the mismatch injection plus the download into `nullptr`):
    `CUDA failure at inject-download.cu:60: cudaMemcpy(nullptr, dcounters, ...) -> invalid
    argument`, `# exit 2`. No mismatch line is printed.
  - **For contrast**, the 8c3423d version with the same reproduction prints `mismatches 0, cuda
    invalid argument` and `total mismatches 0`, then exits 0 (`knsys-idempotency.log`, last
    block). That is the failure the review described.
- `bit-identity.log` is the 8c3423d program's output and is kept as it was.

### 2. Elementwise `MAX`/`MIN` (`t4a_cuda_binary`), paired

- **Cases.** `bench_kernels` gains `elementwise-{max,min}-{f32,f64}-{1000,1048576,16777216}`.
  `t4a_cuda_binary` gets two full-length operands. About one element in a thousand of each
  operand is a payload-tagged NaN, so the select paths run. Every element of the result is
  checked against a CPU reference, NaN results bit for bit.
- **Builds.** Baseline is `3c8d82c` plus the same `bench_kernels.rs`; the candidate is this branch.
  Each was built in its own tree and target directory, because the harness compiles `cuda/src`
  from the tree it was built in. Binary sha256: main `309283cc…c050`, fix `97f7375a…db28`.
- **Protocol** (`elementwise-pair.sh`). For each round and each case, the two binaries run back
  to back in separate processes, each under `knsys.sh` (nsys over the 100 timed launches). Main
  runs first in odd rounds and the fix runs first in even rounds. Before each process the GPU
  must have no compute process and cores 0-7 must be under 5% busy over 1 s.
- **Rounds.** The campaign asked for 9 rounds. The guard stopped it in round 7 (`cores 0-7 busy:
  6.0%`, `elementwise/stop.log`). The table uses the 6 complete rounds
  (`elementwise-table.py elementwise 6`, which requires exactly 6 per cell). Round 7's partial
  data stays in `elementwise/r7/` and is not used. All 165 runs report `check=ok`.

Median [Q1-Q3] over 6 rounds, µs. Device time is nsys `Avg` per launch; host time is
`host_ns_per_op`.

| case | device main | device fix | ratio | host main | host fix | ratio |
|---|---|---|---|---|---|---|
| `max` f32 1000 | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | 4.60 [3.84-5.06] | 3.83 [3.74-4.24] | 0.833 |
| `max` f64 1000 | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.980 | 3.90 [3.74-5.16] | 3.77 [3.74-3.79] | 0.965 |
| `min` f32 1000 | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | 3.70 [3.69-3.79] | 3.70 [3.68-4.17] | 1.001 |
| `min` f64 1000 | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.981 | 3.79 [3.75-4.83] | 3.78 [3.71-4.16] | 0.996 |
| `max` f32 1M | 7.79 [7.79-7.80] | 7.63 [7.62-7.63] | 0.979 | 9.32 [9.20-9.36] | 9.20 [9.10-9.41] | 0.986 |
| `max` f64 1M | 9.32 [9.31-9.33] | 9.25 [9.24-9.25] | 0.992 | 10.75 [10.75-10.77] | 10.71 [10.69-10.78] | 0.996 |
| `min` f32 1M | 7.80 [7.79-7.80] | 7.63 [7.62-7.63] | 0.978 | 9.32 [9.18-9.38] | 9.01 [9.01-9.08] | 0.967 |
| `min` f64 1M | 9.33 [9.31-9.34] | 9.24 [9.24-9.24] | 0.991 | 10.82 [10.71-10.93] | 10.69 [10.66-10.71] | 0.988 |
| `max` f32 16M | 139.19 [139.14-139.22] | 138.71 [138.70-138.75] | 0.997 | 140.61 [140.59-140.70] | 140.17 [140.15-140.23] | 0.997 |
| `max` f64 16M | 251.18 [251.13-251.20] | 250.88 [250.86-250.96] | 0.999 | 252.66 [252.56-252.72] | 252.34 [252.27-252.38] | 0.999 |
| `min` f32 16M | 139.20 [139.15-139.21] | 138.74 [138.71-138.80] | 0.997 | 140.65 [140.61-140.65] | 140.20 [140.15-140.26] | 0.997 |
| `min` f64 16M | 251.13 [251.08-251.18] | 250.92 [250.91-250.93] | 0.999 | 252.54 [252.52-252.59] | 252.41 [252.40-252.43] | 0.999 |

- **No cell regresses.**
  - **Device time** is lower on the fix in all 12 cells, and the fix's Q3 is below main's Q1 in
    every one. By how much: 0.04-0.05 µs at 1000, 0.07-0.17 µs at 1M, 0.2-0.5 µs at 16M.
  - **Host time.** The 1000-element cells are launch-bound and noisy on main, and their IQRs
    overlap, so the 0.833 for `max` f32 1000 says nothing either way. The worst host ratio is
    1.001 (`min` f32 1000), with overlapping IQRs.
- **Why it is faster is not established.** The improvement is small and consistent, but I did
  not analyse the `t4a_cuda_binary` SASS to explain it.
- Per-round values are in `elementwise/table.md`; raw `kern.csv` and bench logs are in
  `elementwise/r*/`, without the `.nsys-rep`/`.sqlite` files.

### 3. `knsys.sh` is idempotent

- **What changed.** `knsys.sh` refuses an output directory that is not empty, and a case listed
  twice. One invocation now owns its `kern.csv`, reports and logs, the rule `bench/run-paired.sh`
  already applies.
- **Other changes.** `--force-overwrite` is gone, because the directory is always fresh. The
  kernel filter is widened from `t4a_cuda_reduce` to `t4a_cuda_`; a reduction case launches only
  reduction kernels, so its rows are the same.
- **Evidence** (`knsys-idempotency.log`).
  - **Second run into the same directory:** `knsys.sh: refusing to reuse ...: it is not empty`,
    `# exit 2`. `kern.csv` keeps 2 lines with the same sha256 prefix (`dac7f716116953c1`) before
    and after, and the file list is unchanged.
  - **A case listed twice:** exit 2, and no directory is created.
  - **The 8c3423d script, run twice into one directory,** did not append with this nsys. The
    second run overwrote the report and the log, then `nsys stats` declined the stale `.sqlite`
    ("File is older than input file ... Use --force-export=true"). The `grep` matched nothing
    and the script exited 1. That left `kern.csv` from run 1 next to the log of run 2: the
    mismatch the review described, reached by a different route.

### 4. SASS on the production entry

- **What the review found.** The combine-loop explanation (`FSETP.NAN` → `BRA` → `LDG` per
  element) rested on a standalone probe kernel (`sass-plane-loop.log`), not on the production
  entry, so it was not established for shipped code.
- **What is now there.** `sass-production.log`: `nvcc -lineinfo` + `nvdisasm -g` on `reduce.cu`
  and `elementwise.cu`, both trees, with the section header, the section's address range and a
  contiguous, labelled run of instructions for each excerpt.
  - `t4a_cuda_reduce` (`reduce.cu:657`), the fold at `reduce.cu:252-253` in `plane_reduce_body`:
    `3c8d82c` runs `DSETP.NAN` → `@P2 BRA` → one `LDG.E.64` → `DSETP.MIN` per fold, with the
    label the branch skips to in view; this branch issues four `LDG.E.64` before the folds and
    has no branch on the accumulator.
  - `t4a_cuda_binary` (`elementwise.cu:371`), at `0x12b0`: two `@P0 BRA` per element pair at
    `3c8d82c`, `FSETP.NAN` → `FMNMX` → `FSEL` straight-line here.
  - Registers unchanged: `SHI_REGISTERS=40` for `t4a_cuda_reduce` and `38` for `t4a_cuda_binary`
    in both trees.
- **What it is not.** Step counts, timing or a measurement: it was taken on evo2, which has no
  NVIDIA device. Which compiler built the timed binaries was not recorded either, so this is
  code generation from the source, not the disassembly of those binaries. `sass-diff.log` also
  carries a re-run of `scripts/sass-diff.sh 3c8d82c` on the same host and CUDA 12.6 build: 94
  symbols, 89 identical, the same five differing entries.

### Review

- **The `gpt-6.1-sol` independent review** of this section ran on the branch head `84be777`
  (read-only `pi --print`, `openai/gpt-6.1-sol`): no blocking finding, two wording findings, both
  fixed before the final push (a parenthetical claiming one line annotation per instruction, and
  a stale cross-reference in the provenance caveat). Two earlier rounds on the same file found
  the missing section extents and the incomplete loop; those are what the excerpts now carry.
