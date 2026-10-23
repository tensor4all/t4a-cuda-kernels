// t4a-cuda-kernels: XLA-style gather and additive scatter.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/indexing.rs: gather_kernel (L263),
// scatter_float_kernel (L350), scatter_complex_kernel (L455),
// clamp_window_start (L63), index_component (L82), flat_to_index_batch_index
// (L108), flat_to_update_window_index (L139), update_window_len (L160).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0. See THIRD_PARTY_NOTICES.md.
//
// The additive scatter takes no operand: the caller initializes the destination
// with an operand copy first (the baseline does the same), and `out` supplies
// every extent the window-fit check needs. Index tensors are compact; their
// values are the caller's validated integral starts.

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

// Tensor axis carrying batch position `position`: the index vector axis is
// skipped when the index tensor has one, because the start component varies along
// it instead of along a batch coordinate.
__device__ __forceinline__ int index_batch_axis(int position, bool vector_present,
                                                int index_vector_dim) {
  return position + ((vector_present && position >= index_vector_dim) ? 1 : 0);
}

// One start component. `batch_base` is the index-tensor address of the element at
// the batch position, already carrying the batch coordinates; `component_stride`
// is the stride of the index vector axis, or zero when the index tensor has none
// (then every start component reads the same element, as the baseline does).
template <typename I>
__device__ __forceinline__ I index_component(const I* indices, long long batch_base,
                                             long long component_stride, int component) {
  return indices[batch_base + static_cast<long long>(component) * component_stride];
}

template <typename T, typename I>
__device__ __forceinline__ void gather_body(void* out_ptr,
                                            const void* operand_ptr,
                                            const void* indices_ptr,
                                            const t4a_cuda_layout& out_layout,
                                            const t4a_cuda_layout& operand_layout,
                                            const t4a_cuda_layout& indices_layout,
                                            const t4a_cuda_gather_plan& plan,
                                            long long output_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, output_len)) {
    return;
  }
  const T* operand = static_cast<const T*>(operand_ptr);
  const I* indices = static_cast<const I*>(indices_ptr);

  // Window offsets and clamped starts fold straight into the operand address and
  // the batch coordinates fold straight into the index-tensor address, so the
  // kernel carries no per-axis metadata array: every subscript of a by-value POD
  // is a loop constant, and the runtime axes (a dimension map or a plan entry) go
  // through `t4a_cuda_axis_value` (see t4a/abi.cuh). Registers therefore stay
  // where occupancy wants them.
  const bool vector_present = plan.index_vector_dim < indices_layout.rank;
  const long long component_stride =
      vector_present ? t4a_cuda_axis_value(indices_layout.strides, plan.index_vector_dim) : 0;

  long long in_index = operand_layout.offset;
  long long out_index = out_layout.offset;
  long long indices_base = indices_layout.offset;
  long long rest = flat;
  int batch_position = 0;
#pragma unroll
  for (int out_axis = 0; out_axis < T4A_CUDA_MAX_RANK; ++out_axis) {
    if (out_axis < out_layout.rank) {
      const long long extent = out_layout.extents[out_axis];
      const long long coordinate = rest % extent;
      // No axis reads the quotient of the last axis, so it is not formed (a
      // 64-bit division is a multi-instruction sequence).
      if (out_axis + 1 < out_layout.rank) {
        rest /= extent;
      }
      out_index += coordinate * out_layout.strides[out_axis];
      int window_pos = -1;
#pragma unroll
      for (int pos = 0; pos < T4A_CUDA_MAX_RANK; ++pos) {
        if (pos < plan.window_rank && window_pos < 0 && plan.offset_dims[pos] == out_axis) {
          window_pos = pos;
        }
      }
      if (window_pos >= 0) {
        const int operand_axis =
            static_cast<int>(t4a_cuda_axis_value(plan.window_dims, window_pos));
        in_index += coordinate * t4a_cuda_axis_value(operand_layout.strides, operand_axis);
      } else {
        const int tensor_axis = index_batch_axis(batch_position, vector_present,
                                                 plan.index_vector_dim);
        indices_base += coordinate * t4a_cuda_axis_value(indices_layout.strides, tensor_axis);
        ++batch_position;
      }
    }
  }

#pragma unroll
  for (int component = 0; component < T4A_CUDA_MAX_RANK; ++component) {
    if (component < plan.start_map_len) {
      const I start = index_component(indices, indices_base, component_stride, component);
      const int operand_axis = static_cast<int>(plan.start_index_map[component]);
      const long long dim = t4a_cuda_axis_value(operand_layout.extents, operand_axis);
      const long long window = t4a_cuda_axis_value(plan.slice_sizes, operand_axis);
      in_index += clamp_window_start<I>(start, dim, window) *
                  t4a_cuda_axis_value(operand_layout.strides, operand_axis);
    }
  }

  static_cast<T*>(out_ptr)[out_index] = operand[in_index];
}

// Shared scatter addressing: decodes one update element into a destination
// offset, or returns false when the window does not fit the operand.
template <typename I>
__device__ __forceinline__ bool scatter_destination(long long flat,
                                                    long long* destination,
                                                    const t4a_cuda_layout& out_layout,
                                                    const t4a_cuda_layout& indices_layout,
                                                    const t4a_cuda_layout& updates_layout,
                                                    const t4a_cuda_scatter_plan& plan,
                                                    const I* indices) {
  // Clamped starts accumulate in `base`, which is addressed with unrolled literal
  // subscripts plus an arithmetic mask (see t4a_cuda_axis_add in t4a/abi.cuh) and
  // therefore stays in registers. Everything else folds into scalars: the update
  // coordinate is decoded inside the axis walk, the batch coordinates fold into
  // the index-tensor address, the window offset into one accumulator and the
  // window shape is derived where it is needed. The live metadata therefore stays
  // small enough not to cost occupancy.
  long long base[T4A_CUDA_MAX_RANK];
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    base[axis] = 0;
  }

  const bool vector_present = plan.index_vector_dim < indices_layout.rank;
  const long long component_stride =
      vector_present ? t4a_cuda_axis_value(indices_layout.strides, plan.index_vector_dim) : 0;

  long long window_offset = 0;
  long long indices_base = indices_layout.offset;
  long long rest = flat;
  int batch_position = 0;
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < updates_layout.rank) {
      const long long extent = updates_layout.extents[axis];
      const long long coordinate = rest % extent;
      rest /= extent;
      int window_pos = -1;
#pragma unroll
      for (int pos = 0; pos < T4A_CUDA_MAX_RANK; ++pos) {
        if (pos < plan.window_rank && window_pos < 0 && plan.update_window_dims[pos] == axis) {
          window_pos = pos;
        }
      }
      if (window_pos >= 0) {
        const int operand_axis =
            static_cast<int>(t4a_cuda_axis_value(plan.window_dims, window_pos));
        window_offset += coordinate * t4a_cuda_axis_value(out_layout.strides, operand_axis);
      } else {
        const int tensor_axis = index_batch_axis(batch_position, vector_present,
                                                 plan.index_vector_dim);
        indices_base += coordinate * t4a_cuda_axis_value(indices_layout.strides, tensor_axis);
        ++batch_position;
      }
    }
  }

  // The window shape comes from the updates extents; every other operand axis is
  // covered by the scattered start alone.
#pragma unroll
  for (int component = 0; component < T4A_CUDA_MAX_RANK; ++component) {
    if (component < plan.scatter_map_len) {
      const I start = index_component(indices, indices_base, component_stride, component);
      const int axis = static_cast<int>(plan.scatter_dims[component]);
      long long shape = 1;
#pragma unroll
      for (int pos = 0; pos < T4A_CUDA_MAX_RANK; ++pos) {
        if (pos < plan.window_rank && plan.window_dims[pos] == axis) {
          shape = t4a_cuda_axis_value(updates_layout.extents, plan.update_window_dims[pos]);
        }
      }
      t4a_cuda_axis_add(base, axis,
                        clamp_window_start<I>(
                            start, t4a_cuda_axis_value(out_layout.extents, axis), shape));
    }
  }

  // A window that does not fit the operand skips this update.
  long long offset = out_layout.offset;
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < out_layout.rank) {
      long long shape = 1;
#pragma unroll
      for (int pos = 0; pos < T4A_CUDA_MAX_RANK; ++pos) {
        if (pos < plan.window_rank && plan.window_dims[pos] == axis) {
          shape = t4a_cuda_axis_value(updates_layout.extents, plan.update_window_dims[pos]);
        }
      }
      if (base[axis] + shape > out_layout.extents[axis]) {
        return false;
      }
      offset += base[axis] * out_layout.strides[axis];
    }
  }
  *destination = offset + window_offset;
  return true;
}

template <typename T, typename I>
__device__ __forceinline__ void scatter_add_body(void* out_ptr,
                                                 const void* indices_ptr,
                                                 const void* updates_ptr,
                                                 const t4a_cuda_layout& out_layout,
                                                 const t4a_cuda_layout& indices_layout,
                                                 const t4a_cuda_layout& updates_layout,
                                                 const t4a_cuda_scatter_plan& plan,
                                                 long long update_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, update_len)) {
    return;
  }
  long long destination;
  if (!scatter_destination(flat, &destination, out_layout, indices_layout, updates_layout, plan,
                           static_cast<const I*>(indices_ptr))) {
    return;
  }
  const long long source = t4a_cuda_decode(flat, updates_layout.rank, updates_layout.extents,
                                           updates_layout.strides, updates_layout.offset);
  atomicAdd(static_cast<T*>(out_ptr) + destination,
            static_cast<const T*>(updates_ptr)[source]);
}

template <typename F, typename C, typename I>
__device__ __forceinline__ void scatter_add_complex_body(void* out_ptr,
                                                         const void* indices_ptr,
                                                         const void* updates_ptr,
                                                         const t4a_cuda_layout& out_layout,
                                                         const t4a_cuda_layout& indices_layout,
                                                         const t4a_cuda_layout& updates_layout,
                                                         const t4a_cuda_scatter_plan& plan,
                                                         long long update_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, update_len)) {
    return;
  }
  long long destination;
  if (!scatter_destination(flat, &destination, out_layout, indices_layout, updates_layout, plan,
                           static_cast<const I*>(indices_ptr))) {
    return;
  }
  const long long source = t4a_cuda_decode(flat, updates_layout.rank, updates_layout.extents,
                                           updates_layout.strides, updates_layout.offset);
  const C update = static_cast<const C*>(updates_ptr)[source];
  F* out = reinterpret_cast<F*>(out_ptr) + destination * 2;
  atomicAdd(out, update.re);
  atomicAdd(out + 1, update.im);
}

// --- Revision 9 general path ---------------------------------------------------
//
// The same maps with pointer geometry and loops bounded by the operand's own
// runtime rank, window rank and map lengths. Unlike the fast bodies these keep no
// `[T4A_CUDA_MAX_RANK]` base array and no masked select: the per-axis base is
// recomputed where it is checked, so the general path's state is O(1) whatever the
// rank.

template <typename T, typename I>
__device__ __forceinline__ void gather_body_any(void* out_ptr,
                                               const void* operand_ptr,
                                               const void* indices_ptr,
                                               const t4a_cuda_layout_any& out_layout,
                                               const t4a_cuda_layout_any& operand_layout,
                                               const t4a_cuda_layout_any& indices_layout,
                                               const t4a_cuda_gather_plan_any& plan,
                                               long long output_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, output_len)) {
    return;
  }
  const T* operand = static_cast<const T*>(operand_ptr);
  const I* indices = static_cast<const I*>(indices_ptr);

  const bool vector_present = plan.index_vector_dim < indices_layout.rank;
  const long long component_stride =
      vector_present ? indices_layout.strides[plan.index_vector_dim] : 0;

  long long in_index = operand_layout.offset;
  long long out_index = out_layout.offset;
  long long indices_base = indices_layout.offset;
  long long rest = flat;
  int batch_position = 0;
  for (int out_axis = 0; out_axis < out_layout.rank; ++out_axis) {
    const long long extent = out_layout.extents[out_axis];
    const long long coordinate = rest % extent;
    rest /= extent;
    out_index += coordinate * out_layout.strides[out_axis];
    int window_pos = -1;
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (window_pos < 0 && plan.offset_dims[pos] == out_axis) {
        window_pos = pos;
      }
    }
    if (window_pos >= 0) {
      const int operand_axis = static_cast<int>(plan.window_dims[window_pos]);
      in_index += coordinate * operand_layout.strides[operand_axis];
    } else {
      const int tensor_axis =
          index_batch_axis(batch_position, vector_present, plan.index_vector_dim);
      indices_base += coordinate * indices_layout.strides[tensor_axis];
      ++batch_position;
    }
  }

  for (int component = 0; component < plan.start_map_len; ++component) {
    const I start = index_component(indices, indices_base, component_stride, component);
    const int operand_axis = static_cast<int>(plan.start_index_map[component]);
    in_index += clamp_window_start<I>(start, operand_layout.extents[operand_axis],
                                      plan.slice_sizes[operand_axis]) *
                operand_layout.strides[operand_axis];
  }

  static_cast<T*>(out_ptr)[out_index] = operand[in_index];
}

// Destination offset of one update element, or false when the window does not fit
// the operand. The per-axis base is recomputed per output axis instead of being
// kept in a rank-sized array.
template <typename I>
__device__ __forceinline__ bool scatter_destination_any(
    long long flat, long long* destination, const t4a_cuda_layout_any& out_layout,
    const t4a_cuda_layout_any& indices_layout, const t4a_cuda_layout_any& updates_layout,
    const t4a_cuda_scatter_plan_any& plan, const I* indices) {
  const bool vector_present = plan.index_vector_dim < indices_layout.rank;
  const long long component_stride =
      vector_present ? indices_layout.strides[plan.index_vector_dim] : 0;

  long long window_offset = 0;
  long long indices_base = indices_layout.offset;
  long long rest = flat;
  int batch_position = 0;
  for (int axis = 0; axis < updates_layout.rank; ++axis) {
    const long long extent = updates_layout.extents[axis];
    const long long coordinate = rest % extent;
    rest /= extent;
    int window_pos = -1;
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (window_pos < 0 && plan.update_window_dims[pos] == axis) {
        window_pos = pos;
      }
    }
    if (window_pos >= 0) {
      const int operand_axis = static_cast<int>(plan.window_dims[window_pos]);
      window_offset += coordinate * out_layout.strides[operand_axis];
    } else {
      const int tensor_axis =
          index_batch_axis(batch_position, vector_present, plan.index_vector_dim);
      indices_base += coordinate * indices_layout.strides[tensor_axis];
      ++batch_position;
    }
  }

  // The window shape comes from the updates extents; every other operand axis is
  // covered by the scattered start alone. Both are recomputed per axis.
  long long offset = out_layout.offset;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    long long shape = 1;
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (plan.window_dims[pos] == axis) {
        shape = updates_layout.extents[plan.update_window_dims[pos]];
      }
    }
    long long base = 0;
    for (int component = 0; component < plan.scatter_map_len; ++component) {
      if (plan.scatter_dims[component] != axis) {
        continue;
      }
      const I start = index_component(indices, indices_base, component_stride, component);
      base += clamp_window_start<I>(start, out_layout.extents[axis], shape);
    }
    if (base + shape > out_layout.extents[axis]) {
      return false;
    }
    offset += base * out_layout.strides[axis];
  }
  *destination = offset + window_offset;
  return true;
}

template <typename T, typename I>
__device__ __forceinline__ void scatter_add_body_any(
    void* out_ptr, const void* indices_ptr, const void* updates_ptr,
    const t4a_cuda_layout_any& out_layout, const t4a_cuda_layout_any& indices_layout,
    const t4a_cuda_layout_any& updates_layout, const t4a_cuda_scatter_plan_any& plan,
    long long update_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, update_len)) {
    return;
  }
  long long destination;
  if (!scatter_destination_any(flat, &destination, out_layout, indices_layout, updates_layout,
                               plan, static_cast<const I*>(indices_ptr))) {
    return;
  }
  const long long source = t4a_cuda_decode_any(flat, updates_layout.rank, updates_layout.extents,
                                               updates_layout.strides, updates_layout.offset);
  atomicAdd(static_cast<T*>(out_ptr) + destination,
            static_cast<const T*>(updates_ptr)[source]);
}

template <typename F, typename C, typename I>
__device__ __forceinline__ void scatter_add_complex_body_any(
    void* out_ptr, const void* indices_ptr, const void* updates_ptr,
    const t4a_cuda_layout_any& out_layout, const t4a_cuda_layout_any& indices_layout,
    const t4a_cuda_layout_any& updates_layout, const t4a_cuda_scatter_plan_any& plan,
    long long update_len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, update_len)) {
    return;
  }
  long long destination;
  if (!scatter_destination_any(flat, &destination, out_layout, indices_layout, updates_layout,
                               plan, static_cast<const I*>(indices_ptr))) {
    return;
  }
  const long long source = t4a_cuda_decode_any(flat, updates_layout.rank, updates_layout.extents,
                                               updates_layout.strides, updates_layout.offset);
  const C update = static_cast<const C*>(updates_ptr)[source];
  F* out = reinterpret_cast<F*>(out_ptr) + destination * 2;
  atomicAdd(out, update.re);
  atomicAdd(out + 1, update.im);
}

}  // namespace

extern "C" __global__ void t4a_cuda_gather(void* out, const void* operand, const void* indices,
                                           int dtype, int index_dtype, t4a_cuda_layout out_layout,
                                           t4a_cuda_layout operand_layout,
                                           t4a_cuda_layout indices_layout,
                                           t4a_cuda_gather_plan plan, long long output_len) {
#define T4A_GATHER_DATA(T) \
  gather_body<T, INDEX>(out, operand, indices, out_layout, operand_layout, indices_layout, plan, output_len)
#define T4A_GATHER_INDEX(I)                       \
  do {                                            \
    using INDEX = I;                              \
    T4A_CUDA_DTYPE_SWITCH(dtype, T4A_GATHER_DATA) \
  } while (0)
  switch (index_dtype) {
    case T4A_CUDA_F32:
      T4A_GATHER_INDEX(float);
      break;
    case T4A_CUDA_F64:
      T4A_GATHER_INDEX(double);
      break;
    case T4A_CUDA_I32:
      T4A_GATHER_INDEX(int);
      break;
    case T4A_CUDA_I64:
      T4A_GATHER_INDEX(long long);
      break;
    default:
      break;
  }
#undef T4A_GATHER_INDEX
#undef T4A_GATHER_DATA
}

extern "C" __global__ void t4a_cuda_scatter_add(void* out, const void* indices,
                                                const void* updates, int dtype,
                                                int index_dtype, t4a_cuda_layout out_layout,
                                                t4a_cuda_layout indices_layout,
                                                t4a_cuda_layout updates_layout,
                                                t4a_cuda_scatter_plan plan,
                                                long long update_len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body<float, float>(out, indices, updates, out_layout, indices_layout,
                                         updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body<double, float>(out, indices, updates, out_layout, indices_layout,
                                          updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body<float, t4a_cuda_c32, float>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body<double, t4a_cuda_c64, float>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_F64:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body<float, double>(out, indices, updates, out_layout, indices_layout,
                                          updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body<double, double>(out, indices, updates, out_layout, indices_layout,
                                           updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body<float, t4a_cuda_c32, double>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body<double, t4a_cuda_c64, double>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_I32:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body<float, int>(out, indices, updates, out_layout, indices_layout,
                                       updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body<double, int>(out, indices, updates, out_layout, indices_layout,
                                        updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body<float, t4a_cuda_c32, int>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body<double, t4a_cuda_c64, int>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_I64:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body<float, long long>(out, indices, updates, out_layout, indices_layout,
                                             updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body<double, long long>(out, indices, updates, out_layout, indices_layout,
                                              updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body<float, t4a_cuda_c32, long long>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body<double, t4a_cuda_c64, long long>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_gather_any(void* out, const void* operand, const void* indices,
                                           int dtype, int index_dtype, t4a_cuda_layout_any out_layout,
                                           t4a_cuda_layout_any operand_layout,
                                           t4a_cuda_layout_any indices_layout,
                                           t4a_cuda_gather_plan_any plan, long long output_len) {
#define T4A_GATHER_DATA(T) \
  gather_body_any<T, INDEX>(out, operand, indices, out_layout, operand_layout, indices_layout, plan, output_len)
#define T4A_GATHER_INDEX(I)                       \
  do {                                            \
    using INDEX = I;                              \
    T4A_CUDA_DTYPE_SWITCH(dtype, T4A_GATHER_DATA) \
  } while (0)
  switch (index_dtype) {
    case T4A_CUDA_F32:
      T4A_GATHER_INDEX(float);
      break;
    case T4A_CUDA_F64:
      T4A_GATHER_INDEX(double);
      break;
    case T4A_CUDA_I32:
      T4A_GATHER_INDEX(int);
      break;
    case T4A_CUDA_I64:
      T4A_GATHER_INDEX(long long);
      break;
    default:
      break;
  }
#undef T4A_GATHER_INDEX
#undef T4A_GATHER_DATA
}

extern "C" __global__ void t4a_cuda_scatter_add_any(void* out, const void* indices,
                                                const void* updates, int dtype,
                                                int index_dtype, t4a_cuda_layout_any out_layout,
                                                t4a_cuda_layout_any indices_layout,
                                                t4a_cuda_layout_any updates_layout,
                                                t4a_cuda_scatter_plan_any plan,
                                                long long update_len) {
  switch (index_dtype) {
    case T4A_CUDA_F32:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body_any<float, float>(out, indices, updates, out_layout, indices_layout,
                                         updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body_any<double, float>(out, indices, updates, out_layout, indices_layout,
                                          updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body_any<float, t4a_cuda_c32, float>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body_any<double, t4a_cuda_c64, float>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_F64:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body_any<float, double>(out, indices, updates, out_layout, indices_layout,
                                          updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body_any<double, double>(out, indices, updates, out_layout, indices_layout,
                                           updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body_any<float, t4a_cuda_c32, double>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body_any<double, t4a_cuda_c64, double>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_I32:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body_any<float, int>(out, indices, updates, out_layout, indices_layout,
                                       updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body_any<double, int>(out, indices, updates, out_layout, indices_layout,
                                        updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body_any<float, t4a_cuda_c32, int>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body_any<double, t4a_cuda_c64, int>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    case T4A_CUDA_I64:
      switch (dtype) {
        case T4A_CUDA_F32:
          scatter_add_body_any<float, long long>(out, indices, updates, out_layout, indices_layout,
                                             updates_layout, plan, update_len);
          break;
        case T4A_CUDA_F64:
          scatter_add_body_any<double, long long>(out, indices, updates, out_layout, indices_layout,
                                              updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C32:
          scatter_add_complex_body_any<float, t4a_cuda_c32, long long>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        case T4A_CUDA_C64:
          scatter_add_complex_body_any<double, t4a_cuda_c64, long long>(
              out, indices, updates, out_layout, indices_layout, updates_layout, plan, update_len);
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
}
