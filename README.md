# t4a-cuda-kernels

Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.

**This repository does not depend on t4a-cuda-runtime, tenferro Tensor/Engine/AD types, CubeCL or CubeK.** Callers provide their own CUDA stream, device buffers, module loading, workspace and completion management.

## Status

First slice implemented and GPU-validated: C ABI revision 1, checked host
planning, and the movement and elementary-arithmetic kernel families.

| Entry | Ported from | Dtypes |
| --- | --- | --- |
| `t4a_cuda_fill_zero`, `_fill_zero_view` | `fill_zero_kernel`, `fill_zero_view_kernel` | all |
| `t4a_cuda_copy_exact` | `copy_bool_kernel` generalised | all |
| `t4a_cuda_materialize_strided`, `_contiguous_to_view`, `_strided_to_strided` | `materialize_strided_kernel`, `contiguous_to_view_kernel`, `strided_to_strided_kernel` | all |
| `t4a_cuda_tiled_transpose` | `tiled_transpose_kernel` | all |
| `t4a_cuda_add`, `_sub`, `_mul`, `_div` | `add_*`, `sub_*`, `mul_*`, `div_float` | F32/F64/I32/I64/C32/C64; div F32/F64 |

Reductions, conversions, indexing/scatter, diagonal, linalg/RRQR helpers, the
fused-expression emitter, NVRTC prepare/cache and complex division are not
implemented yet.

- [C kernel ABI](docs/abi/c-kernel-abi.md)
- [Slice plan](docs/design/slice-abi-movement-arithmetic.md)
- [Local GPU evidence](docs/worklogs/2026-10-20-abi-movement-arithmetic-slice.md)
- [Migration design](docs/design/cubecl-kernel-port.md)
- [CubeCL performance preservation contract](docs/design/cubecl-performance-parity.md)
- [Pinned source/symbol inventory](docs/kernel-inventory.md)
- [Machine-readable inventory](docs/kernel-inventory.json)
- [Review checklist](docs/review-checklist.md)

The initial port targets elementwise operations, strided data movement/conversions, reductions, indexing/scatter, diagonal operations, and QR/LU/RRQR helper kernels currently expressed in CubeCL in tenferro-rs. Vendor GEMM/SVD/FFT implementations are tracked separately.

## Interfaces

Reusable CUDA C++ source/header assets (`cuda/`) with a documented C-compatible
kernel ABI, plus a Rust facade (`t4a_cuda_kernels`) for metadata validation and
launch planning. The caller owns execution resources and synchronization; kernel
APIs do not allocate device scratch or create runtime objects.

The fused-expression emitter, when implemented, receives a small runtime-independent operation description. Axis sizes, strides, offsets and counts are runtime parameters of the current ABI, passed by value for common small-rank paths. Validated rank, operation/axis mappings and algorithm configuration may specialize code for unrolling and branch elimination. The caller compiles or retrieves cached artifacts during prepare, binds modules to its own runtimes, and reuses them during run; ordinary shape changes do not require recompilation. See the [preparation/cache contract](docs/design/cubecl-kernel-port.md#preparation-compilation-and-private-caches).

Device compilation/testing is explicit; the default build and its tests do not
initialize CUDA. Registry publication is deferred.

## Related repositories

- [t4a-gpu-backend](https://github.com/tensor4all/t4a-gpu-backend): execution foundations, including t4a-cuda-runtime.
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs): a future consumer that supplies Tensor, graph and AD semantics.

Development follows the applicable [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md). Before a source translation, preserve and audit the inherited notices described in the design.
