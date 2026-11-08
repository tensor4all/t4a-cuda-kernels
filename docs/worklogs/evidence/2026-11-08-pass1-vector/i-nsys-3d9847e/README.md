# Pass-1 nsys re-measurement on 3d9847e

- Host primerose, NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA 12.6 (`/usr/local/cuda-12.6`), 2026-11-08.
- before: `31b734e` (`git worktree add --detach <scratch>/base31b 31b734e`); after: `3d9847e` (cuda/ identical to `65a74d5`).
- Both built with `cargo build -j 8 --release --features gpu-tests --example bench_kernels`.
- Order: b1, c1, b2, c2, each running `scripts/knsys.sh <bench_kernels> <dir> <12 cases>` (one GPU job at a time).
- `<run>/kern.csv`: `nsys stats --report cuda_gpu_kern_sum` rows: case, time %, total ns, instances, avg, med, min, max, stddev, kernel.
- `<run>/<case>.log`: the bench's `BENCH_META` line (`check=ok` in all 48).
- Table: med of `t4a_cuda_reduce_partials`, mean of the two runs; GB/s = input bytes / time.
