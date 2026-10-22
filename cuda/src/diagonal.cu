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

// Logical coordinates of `flat` in a compact layout, axis 0 fastest. Every slot is
// written, including the ones past `layout.rank`: `t4a_cuda_axis_value` reads the
// whole array, and the masked zero does not make an uninitialized slot defined.
// The axis walk is unrolled with literal metadata indices so the coordinate vector
// and the by-value layout stay in registers / the constant bank (see the
// metadata-access note in t4a/abi.cuh).
__device__ __forceinline__ void t4a_cuda_compact_coordinates(long long flat,
                                                            const t4a_cuda_layout& layout,
                                                            long long* coordinates) {
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    long long coordinate = 0;
    if (axis < layout.rank) {
      coordinate = flat % layout.extents[axis];
      flat /= layout.extents[axis];
    }
    coordinates[axis] = coordinate;
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
  const long long diagonal = t4a_cuda_axis_value(out_coords, diag_axis);

  long long in_index = in_layout.offset;
  int out_axis = 0;
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < in_layout.rank) {
      long long coordinate;
      if (axis == axis_a || axis == axis_b) {
        coordinate = diagonal;
      } else {
        if (out_axis == diag_axis) {
          ++out_axis;
        }
        coordinate = t4a_cuda_axis_value(out_coords, out_axis);
        ++out_axis;
      }
      in_index += coordinate * in_layout.strides[axis];
    }
  }
  // The destination is compact (`require_compact` in `plan_extract_diagonal`).
  static_cast<T*>(out_ptr)[flat] = static_cast<const T*>(in_ptr)[in_index];
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
  const long long diagonal = t4a_cuda_axis_value(in_coords, axis_a);

  long long out_index = out_layout.offset;
  int in_axis = 0;
#pragma unroll
  for (int out_axis = 0; out_axis < T4A_CUDA_MAX_RANK; ++out_axis) {
    if (out_axis < out_layout.rank) {
      const bool inserted = out_axis == axis_b;
      const long long coordinate =
          inserted ? diagonal : t4a_cuda_axis_value(in_coords, in_axis);
      if (!inserted) {
        ++in_axis;
      }
      out_index += coordinate * out_layout.strides[out_axis];
    }
  }
  // The source is compact (`require_compact` in `plan_embed_diagonal_copy`).
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(in_ptr)[flat];
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
