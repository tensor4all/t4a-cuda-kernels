// t4a-cuda-kernels: slice, dynamic slice and pad.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/indexing.rs: slice_kernel (L174),
// dynamic_slice_kernel (L193), clamp_window_start (L63), pad_kernel (L215).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// The destination is compact (the baseline stores at the flat position while it
// decodes coordinates from the layout); the source keeps signed strides.
// Starts and steps of the static slice are runtime values here; the baseline
// made steps a compile-time attribute.

#include <t4a/abi.cuh>

namespace {

// `start <= 0` clamps to zero, otherwise the truncated start is used when it is
// below `dim - window`, and `dim - window` (saturating at zero) otherwise.
template <typename I>
__device__ __forceinline__ long long clamp_window_start(I start, long long dim,
                                                        long long window) {
  const long long max_start = dim > window ? dim - window : 0;
  if (!(start > static_cast<I>(0))) {
    return 0;
  }
  const unsigned long long raw = t4a_cuda_index_to_u64<I>(start);
  return raw < static_cast<unsigned long long>(max_start) ? static_cast<long long>(raw)
                                                         : max_start;
}

template <typename T>
__device__ __forceinline__ void slice_body(void* out_ptr,
                                           const void* in_ptr,
                                           const t4a_cuda_layout& out_layout,
                                           const t4a_cuda_layout& in_layout,
                                           const t4a_cuda_axis_values& starts,
                                           const t4a_cuda_axis_values& steps,
                                           long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long in_index = in_layout.offset;
  long long rest = flat;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = rest % extent;
    rest /= extent;
    const long long in_coordinate = starts.values[axis] + coordinate * steps.values[axis];
    in_index += in_coordinate * in_layout.strides[axis];
  }
  const long long out_index =
      t4a_cuda_decode(flat, out_layout.rank, out_layout.extents, out_layout.strides,
                      out_layout.offset);
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(in_ptr)[in_index];
}

template <typename T, typename I>
__device__ __forceinline__ void dynamic_slice_body(void* out_ptr,
                                                   const void* in_ptr,
                                                   const void* starts_ptr,
                                                   const t4a_cuda_layout& out_layout,
                                                   const t4a_cuda_layout& in_layout,
                                                   long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const I* starts = static_cast<const I*>(starts_ptr);
  long long in_index = in_layout.offset;
  long long rest = flat;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = rest % extent;
    rest /= extent;
    // The window size is the output extent on this axis.
    const long long clamped =
        clamp_window_start<I>(starts[axis], in_layout.extents[axis], extent);
    in_index += (clamped + coordinate) * in_layout.strides[axis];
  }
  const long long out_index =
      t4a_cuda_decode(flat, out_layout.rank, out_layout.extents, out_layout.strides,
                      out_layout.offset);
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(in_ptr)[in_index];
}

// Edge padding keeps the baseline's unsigned-magnitude treatment of negative
// lows (`0 - low`), so `i64::MIN` behaves; interior padding is non-negative and
// checked by the facade, so the spacing never vanishes.
template <typename T>
__device__ __forceinline__ void pad_body(void* out_ptr,
                                         const void* in_ptr,
                                         const t4a_cuda_layout& out_layout,
                                         const t4a_cuda_layout& in_layout,
                                         const t4a_cuda_axis_values& edge_low,
                                         const t4a_cuda_axis_values& interior,
                                         long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  bool in_bounds = true;
  long long in_index = in_layout.offset;
  long long rest = flat;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long out_pos = rest % out_layout.extents[axis];
    rest /= out_layout.extents[axis];

    const long long low = edge_low.values[axis];
    const unsigned long long low_bits = static_cast<unsigned long long>(low);
    const unsigned long long low_magnitude = low < 0 ? 0ULL - low_bits : low_bits;
    const unsigned long long spacing =
        static_cast<unsigned long long>(interior.values[axis] + 1);
    const unsigned long long position = static_cast<unsigned long long>(out_pos);

    unsigned long long shifted = 0;
    if (low < 0) {
      shifted = position + low_magnitude;
    } else if (position < low_magnitude) {
      in_bounds = false;
    } else {
      shifted = position - low_magnitude;
    }
    if (shifted % spacing != 0) {
      in_bounds = false;
    } else {
      const unsigned long long candidate = shifted / spacing;
      if (candidate >= static_cast<unsigned long long>(in_layout.extents[axis])) {
        in_bounds = false;
      } else {
        in_index += static_cast<long long>(candidate) * in_layout.strides[axis];
      }
    }
  }
  const long long out_index =
      t4a_cuda_decode(flat, out_layout.rank, out_layout.extents, out_layout.strides,
                      out_layout.offset);
  static_cast<T*>(out_ptr)[out_index] =
      in_bounds ? static_cast<const T*>(in_ptr)[in_index] : t4a_cuda_zero<T>();
}

}  // namespace

extern "C" __global__ void t4a_cuda_slice(void* out, const void* in, int dtype,
                                          t4a_cuda_layout out_layout,
                                          t4a_cuda_layout in_layout,
                                          t4a_cuda_axis_values starts,
                                          t4a_cuda_axis_values steps, long long len) {
#define T4A_BODY(T) slice_body<T>(out, in, out_layout, in_layout, starts, steps, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_dynamic_slice(void* out, const void* in, const void* starts,
                                                  int dtype, int index_dtype,
                                                  t4a_cuda_layout out_layout,
                                                  t4a_cuda_layout in_layout, long long len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
#define T4A_BODY(T) dynamic_slice_body<T, float>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_F64:
#define T4A_BODY(T) dynamic_slice_body<T, double>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I32:
#define T4A_BODY(T) dynamic_slice_body<T, int>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I64:
#define T4A_BODY(T) dynamic_slice_body<T, long long>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_pad(void* out, const void* in, int dtype,
                                        t4a_cuda_layout out_layout, t4a_cuda_layout in_layout,
                                        t4a_cuda_axis_values edge_low,
                                        t4a_cuda_axis_values edge_high,
                                        t4a_cuda_axis_values interior, long long len) {
  // `edge_high` only defines the output extent, which the facade derives and
  // checks; the kernel never reads it.
  (void)edge_high;
#define T4A_BODY(T) pad_body<T>(out, in, out_layout, in_layout, edge_low, interior, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}
