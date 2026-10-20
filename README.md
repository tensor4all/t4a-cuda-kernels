# t4a-cuda-kernels

Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.

**This repository does not depend on t4a-cuda-runtime, tenferro Tensor/Engine/AD types, CubeCL or CubeK.** Callers provide their own CUDA stream, device buffers, module loading, workspace and completion management.

## Status

Design and source inventory only. No kernel implementation or GPU validation is claimed yet.

- [Migration design](docs/design/cubecl-kernel-port.md)
- [Pinned source/symbol inventory](docs/kernel-inventory.md)
- [Machine-readable inventory](docs/kernel-inventory.json)

The initial port targets elementwise operations, strided data movement/conversions, reductions, indexing/scatter, diagonal operations, and QR/LU/RRQR helper kernels currently expressed in CubeCL in tenferro-rs. Vendor GEMM/SVD/FFT implementations are tracked separately.

## Intended interfaces

Reusable CUDA C++ source/header assets with documented C-compatible kernel ABIs, plus a Rust facade for metadata validation and launch/workspace planning. A thin optional raw-CUDA launcher may be provided. The caller owns execution resources and synchronization; kernel APIs do not allocate device scratch or create runtime objects.

The first crate will be named `t4a-cuda-kernels` (Rust import `t4a_cuda_kernels`). Device compilation/testing is explicit; a source/planning build should not initialize CUDA. Registry publication is deferred.

## Related repositories

- [t4a-gpu-backend](https://github.com/tensor4all/t4a-gpu-backend): execution foundations, including t4a-cuda-runtime.
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs): a future consumer that supplies Tensor, graph and AD semantics.

Development follows the applicable [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md). Before a source translation, preserve and audit the inherited notices described in the design.
