<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to t4a-cuda-kernels

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `t4a-cuda-kernels`:

## Local facts

- **MSRV**: `rust-version = "1.82"` (`Cargo.toml`). CI does not pin a toolchain — the Rust lane uses the runner's
  preinstalled `rustc` on `ubuntu-latest` and only adds `rustfmt` and `clippy` — so nothing builds on 1.82 and there is
  no `msrv` job.
- **Feature combinations CI builds**: `default` (no features), which is the whole host-side crate and needs no CUDA
  installation; and `--features gpu-tests` with `--no-run`, which compiles the device suites against `cudarc` but
  cannot run them, because the hosted runners have no GPU.
- **Platforms claimed**: `ubuntu-latest` for fmt, clippy and the host tests; `ubuntu-22.04` for a lane that compiles
  every `cuda/src/*.cu` with the CUDA 12.6.3 `nvcc` to `sm_80` cubins. No macOS or Windows job exists, and no hosted
  job executes a kernel: GPU evidence is produced locally on the A100 and recorded in the pull request, with the GPU
  model, driver and toolkit versions, the exact command and the observed result.
- **Local gate**: `cargo test --locked` (the CPU-only suite), plus `cargo test --locked --features gpu-tests` on a
  machine with a device when the change touches a kernel.

## Anything here that departs from the shared rules

One does: `rust-version = "1.82"` is declared but no `msrv` job builds on it, so the claim can drift with a
dependency bump. That is a gap to close, not an exemption from the rest of the gates.

This repository adds a `kernel-compile` lane (every `.cu` compiled with `nvcc`, no device needed) and a check that
the committed `Cargo.lock` is current (`cargo metadata --locked --offline`), and it runs the GPU tests outside CI.
