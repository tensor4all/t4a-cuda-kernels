// t4a-cuda-kernels: diagonal extraction/embedding and triangular masks.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/diagonal.rs: extract_diagonal_kernel (L5),
// embed_diagonal_copy_kernel (L33), tril_kernel (L61), triu_kernel (L75).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// CORRECTION against the baseline: extract_diagonal_kernel walks the output
// coordinate cursor over every non-diagonal input axis but never skips the
// diagonal output axis, so with rank >= 3 and a non-zero diagonal output
// position it reads a wrong input coordinate. This port implements the intended
// map (skip the diagonal output axis) and the caller derives that position as
// axis_a - (axis_b < axis_a).

#include <t4a/abi.cuh>

namespace {

// Logical coordinates of `flat` in a compact layout, axis 0 fastest.
__device__ __forceinline__ void t4a_cuda_compact_coordinates(long long flat,
                                                             const t4a_cuda_layout& layout,
                                                             long long* coordinates) {
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis >= layout.rank) {
      break;
    }
    coordinates[axis] = flat % layout.extents[axis];
    flat /= layout.extents[axis];
  }
}

template <typename T>
__device__ __forceinline__ void extract_diagonal_body(void* out_ptr,
                                                      const void* in_ptr,
                                                      const t4a_cuda_layout& out_layout,
                                                      const t4a_cuda_layout& in_layout,
                                                      int axis_a,
                                                      int axis_b,
                                                      long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long out_coords[T4A_CUDA_MAX_RANK];
  t4a_cuda_compact_coordinates(flat, out_layout, out_coords);

  const int diag_axis = axis_a - (axis_b < axis_a ? 1 : 0);
  const long long diagonal = out_coords[diag_axis];

  long long in_index = in_layout.offset;
  int out_axis = 0;
  for (int axis = 0; axis < in_layout.rank; ++axis) {
    long long coordinate;
    if (axis == axis_a || axis == axis_b) {
      coordinate = diagonal;
    } else {
      if (out_axis == diag_axis) {
        ++out_axis;
      }
      coordinate = out_coords[out_axis];
      ++out_axis;
    }
    in_index += coordinate * in_layout.strides[axis];
  }
  const long long out_index =
      t4a_cuda_decode(flat, out_layout.rank, out_layout.extents, out_layout.strides,
                      out_layout.offset);
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(in_ptr)[in_index];
}

template <typename T>
__device__ __forceinline__ void embed_diagonal_copy_body(void* out_ptr,
                                                         const void* in_ptr,
                                                         const t4a_cuda_layout& out_layout,
                                                         const t4a_cuda_layout& in_layout,
                                                         int axis_a,
                                                         int axis_b,
                                                         long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long in_coords[T4A_CUDA_MAX_RANK];
  t4a_cuda_compact_coordinates(flat, in_layout, in_coords);
  const long long diagonal = in_coords[axis_a];

  long long out_index = out_layout.offset;
  int in_axis = 0;
  for (int out_axis = 0; out_axis < out_layout.rank; ++out_axis) {
    const long long coordinate = out_axis == axis_b ? diagonal : in_coords[in_axis++];
    out_index += coordinate * out_layout.strides[out_axis];
  }
  const long long in_index =
      t4a_cuda_decode(flat, in_layout.rank, in_layout.extents, in_layout.strides,
                      in_layout.offset);
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(in_ptr)[in_index];
}

// Both operands are compact over the same extents; axes 0 and 1 carry the
// triangle and every higher axis batches it. Comparisons are overflow-free:
// the baseline's `col - k` is undefined for extreme `k`.
template <typename T>
__device__ __forceinline__ void triangular_body(void* out_ptr,
                                                const void* in_ptr,
                                                long long k,
                                                long long rows,
                                                long long cols,
                                                long long len,
                                                bool keep_below) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long row = flat % rows;
  const long long col = (flat / rows) % cols;
  const long long offset = col - row;
  const bool keep = keep_below ? offset <= k : offset >= k;
  static_cast<T*>(out_ptr)[flat] =
      keep ? static_cast<const T*>(in_ptr)[flat] : t4a_cuda_zero<T>();
}

}  // namespace

extern "C" __global__ void t4a_cuda_extract_diagonal(void* out, const void* in, int dtype,
                                                      t4a_cuda_layout out_layout,
                                                      t4a_cuda_layout in_layout, int axis_a,
                                                      int axis_b, long long len) {
#define T4A_BODY(T) \
  extract_diagonal_body<T>(out, in, out_layout, in_layout, axis_a, axis_b, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_embed_diagonal_copy(void* out, const void* in, int dtype,
                                                         t4a_cuda_layout out_layout,
                                                         t4a_cuda_layout in_layout, int axis_a,
                                                         int axis_b, long long len) {
#define T4A_BODY(T) \
  embed_diagonal_copy_body<T>(out, in, out_layout, in_layout, axis_a, axis_b, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_tril(void* out, const void* in, int dtype, long long k,
                                         long long rows, long long cols, long long len) {
#define T4A_BODY(T) triangular_body<T>(out, in, k, rows, cols, len, true)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_triu(void* out, const void* in, int dtype, long long k,
                                         long long rows, long long cols, long long len) {
#define T4A_BODY(T) triangular_body<T>(out, in, k, rows, cols, len, false)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}
