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

// One start component: the batch coordinates fill every index axis except the
// index vector axis, where the component selects. When the index vector axis is
// absent (`index_vector_dim == rank`), the batch coordinates address the tensor
// directly.
template <typename I>
__device__ __forceinline__ I index_component(const I* indices,
                                             const long long* batch_coords,
                                             int batch_rank,
                                             const t4a_cuda_layout& indices_layout,
                                             int index_vector_dim,
                                             int component) {
  long long offset = indices_layout.offset;
  if (index_vector_dim == indices_layout.rank) {
    for (int axis = 0; axis < batch_rank; ++axis) {
      offset += batch_coords[axis] * indices_layout.strides[axis];
    }
  } else {
    int batch_axis = 0;
    for (int axis = 0; axis < indices_layout.rank; ++axis) {
      const long long coordinate =
          axis == index_vector_dim ? component : batch_coords[batch_axis++];
      offset += coordinate * indices_layout.strides[axis];
    }
  }
  return indices[offset];
}

__device__ __forceinline__ void decode_coordinates(long long flat,
                                                   const t4a_cuda_layout& layout,
                                                   long long* coordinates) {
  long long rest = flat;
  for (int axis = 0; axis < layout.rank; ++axis) {
    coordinates[axis] = rest % layout.extents[axis];
    rest /= layout.extents[axis];
  }
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

  long long out_coords[T4A_CUDA_MAX_RANK];
  decode_coordinates(flat, out_layout, out_coords);

  // Window offsets and clamped starts accumulate into the operand coordinates,
  // matching the baseline's zero-init + add structure.
  long long operand_coords[T4A_CUDA_MAX_RANK];
  for (int axis = 0; axis < operand_layout.rank; ++axis) {
    operand_coords[axis] = 0;
  }

  long long batch_coords[T4A_CUDA_MAX_RANK];
  int batch_rank = 0;
  for (int out_axis = 0; out_axis < out_layout.rank; ++out_axis) {
    int window_pos = -1;
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (plan.offset_dims[pos] == out_axis) {
        window_pos = pos;
        break;
      }
    }
    if (window_pos >= 0) {
      operand_coords[plan.window_dims[window_pos]] += out_coords[out_axis];
    } else {
      batch_coords[batch_rank++] = out_coords[out_axis];
    }
  }

  for (int component = 0; component < plan.start_map_len; ++component) {
    const I start = index_component(indices, batch_coords, batch_rank, indices_layout,
                                    plan.index_vector_dim, component);
    const int operand_axis = static_cast<int>(plan.start_index_map[component]);
    operand_coords[operand_axis] +=
        clamp_window_start<I>(start, operand_layout.extents[operand_axis],
                              plan.slice_sizes[operand_axis]);
  }

  long long in_index = operand_layout.offset;
  for (int axis = 0; axis < operand_layout.rank; ++axis) {
    in_index += operand_coords[axis] * operand_layout.strides[axis];
  }
  const long long out_index = t4a_cuda_decode(flat, out_layout.rank, out_layout.extents,
                                              out_layout.strides, out_layout.offset);
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
  long long update_coords[T4A_CUDA_MAX_RANK];
  decode_coordinates(flat, updates_layout, update_coords);

  long long window_coords[T4A_CUDA_MAX_RANK];
  long long batch_coords[T4A_CUDA_MAX_RANK];
  int batch_rank = 0;
  for (int pos = 0; pos < plan.window_rank; ++pos) {
    window_coords[pos] = 0;
  }
  for (int axis = 0; axis < updates_layout.rank; ++axis) {
    int window_pos = -1;
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (plan.update_window_dims[pos] == axis) {
        window_pos = pos;
        break;
      }
    }
    if (window_pos >= 0) {
      window_coords[window_pos] = update_coords[axis];
    } else {
      batch_coords[batch_rank++] = update_coords[axis];
    }
  }

  // The window shape comes from the updates extents; every other operand axis is
  // covered by the scattered start alone.
  long long window_shape[T4A_CUDA_MAX_RANK];
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    window_shape[axis] = 1;
  }
  for (int pos = 0; pos < plan.window_rank; ++pos) {
    window_shape[plan.window_dims[pos]] =
        updates_layout.extents[plan.update_window_dims[pos]];
  }

  long long base[T4A_CUDA_MAX_RANK];
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    base[axis] = 0;
  }
  for (int component = 0; component < plan.scatter_map_len; ++component) {
    const I start = index_component(indices, batch_coords, batch_rank, indices_layout,
                                    plan.index_vector_dim, component);
    const int axis = static_cast<int>(plan.scatter_dims[component]);
    base[axis] = clamp_window_start<I>(start, out_layout.extents[axis], window_shape[axis]);
  }

  for (int axis = 0; axis < out_layout.rank; ++axis) {
    if (base[axis] + window_shape[axis] > out_layout.extents[axis]) {
      return false;
    }
  }

  long long offset = out_layout.offset;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    long long coordinate = base[axis];
    for (int pos = 0; pos < plan.window_rank; ++pos) {
      if (plan.window_dims[pos] == axis) {
        coordinate += window_coords[pos];
      }
    }
    offset += coordinate * out_layout.strides[axis];
  }
  *destination = offset;
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
