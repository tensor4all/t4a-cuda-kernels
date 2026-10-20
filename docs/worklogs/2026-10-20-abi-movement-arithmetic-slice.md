# ABI, movement and elementary arithmetic slice — local evidence

Date: 2026-10-20. Plan:
[slice-abi-movement-arithmetic.md](../design/slice-abi-movement-arithmetic.md).

## What landed

C ABI revision 1 (`docs/abi/c-kernel-abi.md`), the Rust facade
(`Layout`, `DType`, `Entry` manifest, per-entry checked planning, `CLayout` /
`CTransposeConfig` mirrors) and eleven CUDA entries in `cuda/src/`:

`t4a_cuda_fill_zero`, `t4a_cuda_fill_zero_view`, `t4a_cuda_copy_exact`,
`t4a_cuda_materialize_strided`, `t4a_cuda_contiguous_to_view`,
`t4a_cuda_strided_to_strided`, `t4a_cuda_tiled_transpose`, `t4a_cuda_add`,
`t4a_cuda_sub`, `t4a_cuda_mul`, `t4a_cuda_div`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe (`nvidia-smi`) |
| Driver | 580.126.09 |
| CUDA toolkit | 12.6 (`/usr/local/cuda-12.6`), 12.5, 12.0, 12.6 symlink |
| nvcc used by the tests | 12.0.140 (`/usr/bin/nvcc` from `PATH`) |
| Rust | 1.97.1, cargo 1.97.1 |
| Test compile flags | `nvcc -std=c++17 -cubin -arch=sm_80 -I cuda/include` (no `--use_fast_math`, no `-ftz`) |
| cudarc (test-only) | 0.19.8, features `std`, `driver`, `dynamic-loading`, `cuda-12060` |

## Commands and results

```bash
cargo fmt --check                                     # pass
cargo clippy -j 16 --all-targets --features gpu-tests -- -D warnings   # pass, no warnings
cargo test -j 16                                      # 22 planning tests pass, no CUDA needed
cargo test -j 16 --features gpu-tests                 # 13 GPU tests + 22 planning tests pass
```

GPU tests: fill/zero for all seven dtype tags (65 537 elements), fill-zero view
with preserved holes, bit-preserving exact copy over nonfinite payloads for every
dtype, materialize with negative and zero strides, contiguous→view and
strided→strided with full-allocation sentinel comparison, tiled transpose over
1×1, 5×7 batched, 16×8, 33×17 batched and 8×8 with the v1/v2/v4 configurations,
non-zero source offsets, nonfinite complex payloads and NaN payload bits,
add/sub/mul/div against CPU references for F32/F64 (65 536 elements), I32/I64
integer wrap at the boundaries, C32/C64 complex routes, IEEE division edge cases,
zero launches for validated empty plans, and a 2^16-element case for each
strided movement entry.

## Findings

- The first harness used synchronous `cuMemcpyHtoD` on the legacy default stream
  while launching on a non-blocking stream. Those are not ordered against each
  other, so the sentinel upload could land after the kernel and silently pass a
  partially written buffer as correct. Transfers now go through the session's own
  stream. This is exactly the class of defect the review checklist's execution-
  boundary and evidence items are meant to catch; the numerical tests only caught
  it once the buffers were large enough to lose the race.
- An independent review of the diff found the documented transpose formula wrong
  on the source side (`i*N + j*M` instead of `i*N + j`), a non-injective
  `fill_zero_view` destination treated as harmless although two stores to one
  address is a data race, unrepresentable byte offsets accepted for large element
  offsets, `grid.x` limited as if it were `grid.y`, missing C++ alignment/offset
  assertions, an over-permissive test comparator for infinite expectations, and
  thin dtype/large-size coverage. All were fixed here; the test matrix above is
  the extended one.
- All `_comptime_layout_kernel` twins remain excluded (WebGPU/Metal only).

## Not done in this slice

Reductions, conversions/casts, indexing/scatter, diagonal, linalg/RRQR helpers,
the fused-expression source emitter, NVRTC prepare-time compilation with
artifact/binding caches, complex division, and any performance comparison against
the CubeCL baseline. No hosted CI was used; the repository has no workflows yet.
