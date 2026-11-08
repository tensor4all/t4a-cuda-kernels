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
candidate tree is `3eb8376`: the diff against `3c8d82c` is `abi.cuh` plus `docs/`. (The branch
head has moved since, and the later commits touch only `docs/`, the evidence programs and
`examples/bench_kernels.rs`.)

The evidence revision of 2026-11-09 (checked `bit-identity.cu`, `sass-production.log`, the
`elementwise-max-f32` bench cases) was produced on evo2 (Ryzen AI MAX+ 395, no NVIDIA device,
no driver): `nvcc -cubin` and `nvdisasm` need no GPU, running a kernel does. So nothing in
this revision re-measures timing, and the identity rows below still come from the unchecked
harness of Part 1 in `bit-identity.log`.

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

- `bit-identity.log` Part 1, **unchecked harness**: a zero-mismatch *observation* for `max`
  and `min` over 67 109 120 pairs per type, from a run whose every CUDA return value was
  ignored. The pairs are all 16 × 16 pairs of special values (±0, ±1, ±Inf, ±smallest
  subnormal, ±largest finite, and quiet, signalling and negative NaNs with distinct payloads),
  plus 2^26 random bit patterns, one third of them pairs of neighbouring values. Because the
  harness is what #26 reports, this is not a bit-identity proof: on a host where the copy fails
  it prints the same `mismatches 0` and exits 0 (Part 2, last block).
- `bit-identity.log` Part 2, checked harness (`CUDA_CHECK` around every call, device
  allocations before host ones so a failed `cudaMalloc` is not followed by an uninitialised
  pointer). `./bit-identity.sh` records the exit code of four runs.
  - **Checked bit identity on a device is not measured.** That is the one gap this revision
    leaves: it was built on a host without a device (`--copy-fails`, a null host pointer, is
    the copy failure; `cudaMemcpy` rejects it without reading anything).
  - Without a device (evo2): every checked run exits 2 at `cudaMalloc` with the driver error
    on stderr, and `--legacy --copy-fails` — the pre-fix behaviour — prints `mismatches 0`
    and exits 0 with no comparison made. Both are in Part 2.
- `cargo-test-gpu-tests.log` (run on `3eb8376`, the code of this branch): `cargo test -j 4 --release --features gpu-tests
  --no-fail-fast` passes (all 15 test binaries and the doc-tests, `# exit 0`). This includes
  `binary_extrema_pow_rem_and_scalar_division`, which checks that the lhs NaN payload
  survives.
- `cargo fmt --check`, `cargo clippy -j 8 --all-targets -- -D warnings` and host `cargo test`
  pass.

## SASS

`evidence/2026-11-08-nan-extrema-select/sass-production.log` takes the instruction stream on
the production entries with source-line attribution (`nvcc -lineinfo`, `nvdisasm -g`), both
trees:

- `t4a_cuda_reduce` (`cuda/src/reduce.cu:657`), the combine pass's fold at `reduce.cu:252-253`
  inside `plane_reduce_body`: `3c8d82c` runs `DSETP.NAN` → `BSSY` → `@P2 BRA` → one `LDG.E.64`
  → `DSETP.MIN` per fold; this branch issues four `LDG.E.64` before the folds and has no
  branch on the accumulator.
- `t4a_cuda_binary` (`cuda/src/elementwise.cu:371`), float `MAX`/`MIN` arms at `0x12b0` and
  `0x1310`: two `@P0 BRA` per pair at `3c8d82c`, straight-line `FSETP.NAN`/`FMNMX`/`FSEL` here.

The earlier `sass-plane-loop.log` is `plane_reduce_body<MaxFold, float>` in a probe kernel
(`probe_max`), not the entry; it stays as the loop-body excerpt. Registers are unchanged:
`t4a_cuda_reduce` 40 in both trees, `t4a_cuda_binary` 38 in both.

## Not measured

- Elementwise `MAX`/`MIN` (`t4a_cuda_binary`) timing. Its SASS changed; `bench_kernels` now
  has `elementwise-max-f32-1000000` and `elementwise-max-f32-10000000` (`plan_binary`
  `BinaryOp::Max` on `Entry::Binary`), which compile and pass host tests but have never run:
  the host that produced this revision has no device. No number is claimed for them.
- Through tenferro. That needs the pin moved past this change.
