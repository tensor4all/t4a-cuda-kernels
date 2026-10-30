# t4a-cuda-kernels

Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.

**This repository does not depend on t4a-cuda-runtime, tenferro Tensor/Engine/AD types, CubeCL or CubeK.** Callers provide their own CUDA stream, device buffers, module loading, workspace and completion management.

## Status

Eight slices implemented and GPU-validated: C ABI revision 8, checked host
planning, the movement and elementary-arithmetic families, the diagonal,
windowing and validation-status families, single-axis reductions, XLA-style
gather with additive scatter, the dtype conversions, the remaining elementwise,
comparison and strided-combination families, the linalg auxiliary kernels, and the
rank-revealing QR primitives. The custom-kernel inventory of the migration design
is fully ported, and the fused-expression emitter (`fusion` module) replaces the
baseline's run-time CubeCL IR generation with a source emitter plus an artifact key;
compiling and loading remain the caller's service, as the design requires.

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
| `t4a_cuda_initialize_permutation`, `_column_norms`, `_select_pivot`, `_swap_columns` | the RRQR permutation, column-norm and pivot-selection primitives | payload-dependent |
| `t4a_cuda_reflector`, `_apply_reflector`, `_extract_r`, `_prefix_rank` | the RRQR Householder and rank primitives | F32/F64/C32/C64 |

Revision 10 adds the two families that had no entry — `t4a_cuda_broadcast_in_dim`
(rank-mapped broadcast, runtime dimension map) and `t4a_cuda_concatenate_copy`
(copy at an axis offset) — each with a revision 9 general-path twin. `reverse` and
`scale_in_place_*` still need no entry of their own: a reversed layout (negated
strides and a rebased offset) expresses the reversal through the movement entries,
and the in-place factor multiply is `t4a_cuda_binary` with a scalar `rhs` and the
destination aliasing `lhs`.

Not implemented, deliberately: nothing in the kernel inventory. Compilation, module
loading and artifact persistence are the caller's, per
[the prepare contract](docs/design/cubecl-kernel-port.md#preparation-compilation-and-private-caches);
the reference consumer in `tests/gpu_fusion.rs` shows the whole prepare/cache flow
with NVRTC and observed compile counts.

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
- [Slice plan: rank-revealing QR primitives](docs/design/slice-rrqr.md)
- [Slice plan: fused-expression emitter](docs/design/slice-fusion-emitter.md)
- [Local GPU evidence: RRQR primitives](docs/worklogs/2026-10-21-rrqr-slice.md)
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

The fused-expression emitter (`t4a_cuda_kernels::fusion`) receives a small
runtime-independent operation description: a dtype, the input count, a topologically
ordered instruction list and the live-out values. It emits a self-contained CUDA C++
kernel whose only runtime arguments are the buffer pointers and `len`; extents,
strides and offsets never enter the source, and the artifact key is the canonical
key material (source plus the caller's compiler options, target and compiler
identity). The caller compiles or retrieves cached artifacts during prepare, binds
modules to its own runtimes, recomputes the grid from each `len` and reuses the code;
see the [preparation/cache contract](docs/design/cubecl-kernel-port.md#preparation-compilation-and-private-caches)
and `tests/gpu_fusion.rs` for the reference consumer with observed NVRTC counts.

Device compilation/testing is explicit; the default build and its tests do not
initialize CUDA. Published on crates.io as [`t4a-cuda-kernels`](https://crates.io/crates/t4a-cuda-kernels).

## Related repositories

- [t4a-gpu-backend](https://github.com/tensor4all/t4a-gpu-backend): execution foundations, including t4a-cuda-runtime.
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs): the consumer that supplies Tensor, graph and AD semantics (its CUDA backend since tenferro 0.8).

Development follows the applicable [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md). Before a source translation, preserve and audit the inherited notices described in the design.
