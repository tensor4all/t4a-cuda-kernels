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
//
// Revision 11 adds `t4a_cuda_dynamic_update_slice` and its `_any` twin: the
// window write of StableHLO dynamic_update_slice, addressed through the update's
// own signed layout and the same `clamp_window_start`. The operand copy that
// initializes the destination is still the caller's (see the ABI document).

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

// One thread per output element over a compact destination, which the planner
// enforces (`require_compact` in `plan_slice`/`plan_dynamic_slice`/`plan_pad`), so
// the destination is addressed at the flat logical position and never decoded a
// second time. The axis walk is fully unrolled with literal metadata indices (see
// the metadata-access note in t4a/abi.cuh): a runtime axis would copy both layouts
// and both by-value axis-value PODs into the per-thread stack frame and turn every
// metadata access into local-memory traffic.
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
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < out_layout.rank) {
      const long long extent = out_layout.extents[axis];
      const long long coordinate = rest % extent;
      rest /= extent;
      const long long in_coordinate = starts.values[axis] + coordinate * steps.values[axis];
      in_index += in_coordinate * in_layout.strides[axis];
    }
  }
  static_cast<T*>(out_ptr)[flat] = static_cast<const T*>(in_ptr)[in_index];
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
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < out_layout.rank) {
      const long long extent = out_layout.extents[axis];
      const long long coordinate = rest % extent;
      rest /= extent;
      // The window size is the output extent on this axis.
      const long long clamped =
          clamp_window_start<I>(starts[axis], in_layout.extents[axis], extent);
      in_index += (clamped + coordinate) * in_layout.strides[axis];
    }
  }
  static_cast<T*>(out_ptr)[flat] = static_cast<const T*>(in_ptr)[in_index];
}

// The window write of StableHLO `dynamic_update_slice`: one thread per *update*
// element, never per operand element. The destination is compact
// (`require_compact` in `plan_dynamic_update_slice`), so its address is decoded
// from its own extents and the clamped window origin; the update keeps its signed
// strides. The operand copy that initializes `out` is ordered before this launch
// by the caller; elements outside the window are not written here.
template <typename T, typename I>
__device__ __forceinline__ void dynamic_update_slice_body(
    void* out_ptr, const void* update_ptr, const void* starts_ptr,
    const t4a_cuda_layout& out_layout, const t4a_cuda_layout& update_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const I* starts = static_cast<const I*>(starts_ptr);
  long long out_index = 0;
  long long out_stride = 1;
  long long update_index = update_layout.offset;
  long long rest = flat;
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < update_layout.rank) {
      const long long extent = update_layout.extents[axis];
      const long long coordinate = rest % extent;
      rest /= extent;
      const long long start =
          clamp_window_start<I>(starts[axis], out_layout.extents[axis], extent);
      out_index += (start + coordinate) * out_stride;
      out_stride *= out_layout.extents[axis];
      update_index += coordinate * update_layout.strides[axis];
    }
  }
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(update_ptr)[update_index];
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
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < out_layout.rank) {
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
  }
  static_cast<T*>(out_ptr)[flat] =
      in_bounds ? static_cast<const T*>(in_ptr)[in_index] : t4a_cuda_zero<T>();
}

// --- Revision 9 general path ---------------------------------------------------
//
// The same formulas with pointer geometry and a walk bounded by the operand's own
// runtime rank, so any rank is representable. No per-axis array, no masked select
// and no unrolled T4A_CUDA_MAX_RANK bound.

template <typename T>
__device__ __forceinline__ void slice_body_any(void* out_ptr,
                                               const void* in_ptr,
                                               const t4a_cuda_layout_any& out_layout,
                                               const t4a_cuda_layout_any& in_layout,
                                               const t4a_cuda_axis_values_any& starts,
                                               const t4a_cuda_axis_values_any& steps,
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
    const long long in_coordinate =
        t4a_cuda_axis_value_any(starts.values, axis) +
        coordinate * t4a_cuda_axis_value_any(steps.values, axis);
    in_index += in_coordinate * in_layout.strides[axis];
  }
  static_cast<T*>(out_ptr)[flat] = static_cast<const T*>(in_ptr)[in_index];
}

template <typename T, typename I>
__device__ __forceinline__ void dynamic_slice_body_any(void* out_ptr,
                                                       const void* in_ptr,
                                                       const void* starts_ptr,
                                                       const t4a_cuda_layout_any& out_layout,
                                                       const t4a_cuda_layout_any& in_layout,
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
  static_cast<T*>(out_ptr)[flat] = static_cast<const T*>(in_ptr)[in_index];
}

template <typename T, typename I>
__device__ __forceinline__ void dynamic_update_slice_body_any(
    void* out_ptr, const void* update_ptr, const void* starts_ptr,
    const t4a_cuda_layout_any& out_layout, const t4a_cuda_layout_any& update_layout,
    long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const I* starts = static_cast<const I*>(starts_ptr);
  long long out_index = 0;
  long long out_stride = 1;
  long long update_index = update_layout.offset;
  long long rest = flat;
  for (int axis = 0; axis < update_layout.rank; ++axis) {
    const long long extent = update_layout.extents[axis];
    const long long coordinate = rest % extent;
    rest /= extent;
    const long long start =
        clamp_window_start<I>(starts[axis], out_layout.extents[axis], extent);
    out_index += (start + coordinate) * out_stride;
    out_stride *= out_layout.extents[axis];
    update_index += coordinate * update_layout.strides[axis];
  }
  static_cast<T*>(out_ptr)[out_index] = static_cast<const T*>(update_ptr)[update_index];
}

template <typename T>
__device__ __forceinline__ void pad_body_any(void* out_ptr,
                                             const void* in_ptr,
                                             const t4a_cuda_layout_any& out_layout,
                                             const t4a_cuda_layout_any& in_layout,
                                             const t4a_cuda_axis_values_any& edge_low,
                                             const t4a_cuda_axis_values_any& interior,
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

    const long long low = t4a_cuda_axis_value_any(edge_low.values, axis);
    const unsigned long long low_bits = static_cast<unsigned long long>(low);
    const unsigned long long low_magnitude = low < 0 ? 0ULL - low_bits : low_bits;
    const unsigned long long spacing = static_cast<unsigned long long>(
        t4a_cuda_axis_value_any(interior.values, axis) + 1);
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
  static_cast<T*>(out_ptr)[flat] =
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

extern "C" __global__ void t4a_cuda_dynamic_update_slice(
    void* out, const void* update, const void* starts, int dtype, int index_dtype,
    t4a_cuda_layout out_layout, t4a_cuda_layout update_layout, long long len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
#define T4A_BODY(T) \
  dynamic_update_slice_body<T, float>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_F64:
#define T4A_BODY(T) \
  dynamic_update_slice_body<T, double>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I32:
#define T4A_BODY(T) \
  dynamic_update_slice_body<T, int>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I64:
#define T4A_BODY(T) \
  dynamic_update_slice_body<T, long long>(out, update, starts, out_layout, update_layout, len)
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

extern "C" __global__ void t4a_cuda_slice_any(void* out, const void* in, int dtype,
                                              t4a_cuda_layout_any out_layout,
                                              t4a_cuda_layout_any in_layout,
                                              t4a_cuda_axis_values_any starts,
                                              t4a_cuda_axis_values_any steps, long long len) {
#define T4A_BODY(T) slice_body_any<T>(out, in, out_layout, in_layout, starts, steps, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_dynamic_slice_any(void* out, const void* in,
                                                      const void* starts, int dtype,
                                                      int index_dtype,
                                                      t4a_cuda_layout_any out_layout,
                                                      t4a_cuda_layout_any in_layout,
                                                      long long len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
#define T4A_BODY(T) dynamic_slice_body_any<T, float>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_F64:
#define T4A_BODY(T) dynamic_slice_body_any<T, double>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I32:
#define T4A_BODY(T) dynamic_slice_body_any<T, int>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I64:
#define T4A_BODY(T) \
  dynamic_slice_body_any<T, long long>(out, in, starts, out_layout, in_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_dynamic_update_slice_any(
    void* out, const void* update, const void* starts, int dtype, int index_dtype,
    t4a_cuda_layout_any out_layout, t4a_cuda_layout_any update_layout, long long len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
#define T4A_BODY(T)                    \
  dynamic_update_slice_body_any<T, float>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_F64:
#define T4A_BODY(T)                     \
  dynamic_update_slice_body_any<T, double>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I32:
#define T4A_BODY(T)                      \
  dynamic_update_slice_body_any<T, int>(out, update, starts, out_layout, update_layout, len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_I64:
#define T4A_BODY(T)                            \
  dynamic_update_slice_body_any<T, long long>(out, update, starts, out_layout, update_layout, \
                                              len)
      T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_pad_any(void* out, const void* in, int dtype,
                                            t4a_cuda_layout_any out_layout,
                                            t4a_cuda_layout_any in_layout,
                                            t4a_cuda_axis_values_any edge_low,
                                            t4a_cuda_axis_values_any interior, long long len) {
  // The general path takes only the two axis-value vectors the body reads;
  // `edge_high` is a by-value extent input that the facade folds into
  // `out_layout`.
#define T4A_BODY(T) pad_body_any<T>(out, in, out_layout, in_layout, edge_low, interior, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}
