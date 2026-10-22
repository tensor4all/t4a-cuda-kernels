# t4a-cuda-kernels

Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.

**This repository does not depend on t4a-cuda-runtime, tenferro Tensor/Engine/AD types, CubeCL or CubeK.** Callers provide their own CUDA stream, device buffers, module loading, workspace and completion management.

## Status

Seven slices implemented and GPU-validated: C ABI revision 7, checked host
planning, the movement and elementary-arithmetic families, the diagonal,
windowing and validation-status families, single-axis reductions, XLA-style
gather with additive scatter, the dtype conversions, the remaining elementwise,
comparison and strided-combination families, and the linalg auxiliary kernels.

| Entry | Ported from | Dtypes |
| --- | --- | --- |
| `t4a_cuda_fill_zero`, `_fill_zero_view` | `fill_zero_kernel`, `fill_zero_view_kernel` | all |
| `t4a_cuda_copy_exact` | `copy_bool_kernel` generalised | all |
| `t4a_cuda_materialize_strided`, `_contiguous_to_view`, `_strided_to_strided` | `materialize_strided_kernel`, `contiguous_to_view_kernel`, `strided_to_strided_kernel` | all |
| `t4a_cuda_tiled_transpose` | `tiled_transpose_kernel` | all |
| `t4a_cuda_add`, `_sub`, `_mul`, `_div` | `add_*`, `sub_*`, `mul_*`, `div_float` | F32/F64/I32/I64/C32/C64; div F32/F64 |
| `t4a_cuda_extract_diagonal`, `_embed_diagonal_copy` | `extract_diagonal_kernel`, `embed_diagonal_copy_kernel` | all |
| `t4a_cuda_tril`, `_triu` | `tril_kernel`, `triu_kernel` | all |
| `t4a_cuda_slice`, `_dynamic_slice`, `_pad` | `slice_kernel`, `dynamic_slice_kernel`, `pad_kernel` | all |
| `t4a_cuda_init_validation`, `_validate_float_indices`, `_extract_invalid_float_index`, `_validate_real_cast`, `_extract_invalid_real_cast` | `init_float_index_validation_flag`, `validate_float_indices_kernel`, `extract_invalid_float_index_kernel`, `validate_real_cast`, `extract_invalid_real_cast` | F32/F64 |
| `t4a_cuda_reduce`, `_reduce_partials` | `reduce_{sum,prod,sum_squares,max,min}*`, `_plane`, `_chunked` | sum/prod: F32/F64/I32/I64/C32/C64; sum-of-squares F32/F64; max/min F32/F64/I32/I64 |
| `t4a_cuda_gather`, `_scatter_add` | `gather_kernel`, `scatter_{float,complex}_kernel` | all data dtypes with F32/F64/I32/I64 indices; scatter F32/F64/C32/C64 |
| `t4a_cuda_convert` | the `convert_*` family in `structural.rs` | every unequal pair of the seven dtypes |
| `t4a_cuda_unary` | `neg_*`, `exp_float` .. `erf_float`, `abs_*`, `sign_*`, `conj_complex` | F32/F64/I32/I64/C32/C64, op-dependent |
| `t4a_cuda_binary` | extrema, `pow*`, `rem*`, scalar and mixed real/complex forms, `div_complex_parts`, `scale_in_place_*` | F32/F64/I32/I64/C32/C64, op-dependent |
| `t4a_cuda_compare`, `_select`, `_clamp` | `compare_*`, `select_bool_*`, `clamp_float` | F32/F64/I32/I64 (clamp F32/F64) |
| `t4a_cuda_combined_strided` | `broadcast_multiply_*`, `axpby_strided_source_*` | F32/F64/I32/I64/C32/C64 (axpby F32/F64/C32/C64) |
| `t4a_cuda_fill_one`, `_matrix_adjoint`, `_householder_explicit_v`, `_identity_shifted` | `fill_one_kernel`, `matrix_adjoint_*`, `householder_explicit_v`, `householder_q_columns_identity`, `initialize_q` | all / six |
| `t4a_cuda_qr_phase`, `_qr_apply_phase`, `_upper_trapezoidal_violation`, `_householder_assemble` | the QR phase/apply and assembly helpers in `tenferro-linalg/src/gpu/kernels.rs` | F32/F64/C32/C64 (violation and assembly wider) |
| `t4a_cuda_lu_extract_outputs`, `_lu_parity`, `_lu_apply_pivots` | the LU output extraction helpers | F32/F64/C32/C64 (apply wider) |

The baseline's `broadcast_in_dim`, `reverse` and `concatenate_copy` kernels need
no entry of their own: an explicit layout (expanded broadcast source, negative
stride, shifted destination view) expresses the same address map through the
movement entries.

Not implemented yet: the rank-revealing QR primitives (`rank_revealing_qr.rs`), the
fused-expression emitter, and the NVRTC prepare/cache path. See the
[linalg worklog](docs/worklogs/2026-10-21-linalg-aux-slice.md#still-not-ported).

- [C kernel ABI](docs/abi/c-kernel-abi.md)
- [Slice plan: ABI, movement, arithmetic](docs/design/slice-abi-movement-arithmetic.md)
- [Slice plan: diagonal, windowing, validation](docs/design/slice-diagonal-window-validation.md)
- [Slice plan: reductions](docs/design/slice-reductions.md)
- [Slice plan: gather and scatter](docs/design/slice-gather-scatter.md)
- [Slice plan: conversions](docs/design/slice-conversions.md)
- [Local GPU evidence: movement and arithmetic](docs/worklogs/2026-10-20-abi-movement-arithmetic-slice.md)
- [Local GPU evidence: diagonal, windowing, validation](docs/worklogs/2026-10-20-diagonal-window-validation-slice.md)
- [Local GPU evidence: reductions](docs/worklogs/2026-10-20-reductions-slice.md)
- [Local GPU evidence: gather and scatter](docs/worklogs/2026-10-20-gather-scatter-slice.md)
- [Local GPU evidence: conversions](docs/worklogs/2026-10-20-conversions-slice.md)
- [Slice plan: elementwise, comparison, strided combination](docs/design/slice-elementwise.md)
- [Local GPU evidence: elementwise](docs/worklogs/2026-10-21-elementwise-slice.md)
- [Slice plan: linalg auxiliaries](docs/design/slice-linalg-aux.md)
- [Local GPU evidence: linalg auxiliaries](docs/worklogs/2026-10-21-linalg-aux-slice.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
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
