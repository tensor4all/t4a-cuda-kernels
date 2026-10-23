// t4a-cuda-kernels: single-axis reductions (sum, product, sum of squares,
// maximum, minimum) with unit, plane and split-partials strategies.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/reduce/{kernels,launch,routines}.rs and
// kernels/helpers.rs (nan-propagating extrema, wrapping arithmetic).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// The launch strategy and split policy are adapted from CubeK Reduce
// (https://github.com/tracel-ai/cubek at commit
// 9cf90b797107d46829e1c9d9355ce801c3dd4a7d, crates/cubek-reduce/src/launch/{base,strategy}.rs
// and routines/{unit,blueprint}.rs):
// Copyright (c) 2022 Nathaniel Simard & CubeCL Framework Contributors,
// MIT OR Apache-2.0. See THIRD_PARTY_NOTICES.md.
//
// Tenferro changes carried over: single-axis keepdims output, first-element
// accumulator seeding, floor split whose last chunk absorbs the remainder, and
// no recursive split of the partials axis.

#include <t4a/abi.cuh>

namespace {

// A fold seeds the accumulator from the axis' first element, so the seed is the
// operation's own identity applied to that element (sum-of-squares must square
// it, unlike the others).
struct SumFold {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_add<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_add<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T seed(T value) const {
    return value;
  }
};

struct ProdFold {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_mul<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_mul<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T seed(T value) const {
    return value;
  }
};

struct SumSquaresFold {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_add<T>(lhs, t4a_cuda_square<T>(rhs));
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_add<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T seed(T value) const {
    return t4a_cuda_square<T>(value);
  }
};

struct MaxFold {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_max<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_max<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T seed(T value) const {
    return value;
  }
};

struct MinFold {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_min<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T combine(T lhs, T rhs) const {
    return t4a_cuda_min<T>(lhs, rhs);
  }
  template <typename T>
  __device__ __forceinline__ T seed(T value) const {
    return value;
  }
};

// Base offset in the input and offset in the output for one keepdims output
// element. The reduce axis has output extent 1, so its coordinate is zero and
// contributes nothing to either offset.
__device__ __forceinline__ void t4a_cuda_reduce_offsets(const t4a_cuda_layout& out_layout,
                                                        const t4a_cuda_layout& in_layout,
                                                        long long output_index,
                                                        long long* in_base,
                                                        long long* out_offset) {
  long long rest = output_index;
  long long base = in_layout.offset;
  long long offset = out_layout.offset;
  for (int dim = 0; dim < out_layout.rank; ++dim) {
    const long long extent = out_layout.extents[dim];
    const long long coordinate = rest % extent;
    rest /= extent;
    base += coordinate * in_layout.strides[dim];
    offset += coordinate * out_layout.strides[dim];
  }
  *in_base = base;
  *out_offset = offset;
}

template <typename Fold, typename T>
__device__ __forceinline__ T t4a_cuda_plane_fold(T value, int plane_width, Fold fold) {
  const unsigned int mask = t4a_cuda_plane_mask(plane_width);
  // The plane combines lane accumulators, so it uses the fold's combine step:
  // sum-of-squares must not square an accumulator again.
  for (int offset = 1; offset < plane_width; offset <<= 1) {
    value = fold.combine(value, t4a_cuda_shfl_xor(value, offset, mask));
  }
  return value;
}

// One thread per output, walking the axis serially. The planner only allows this
// when the axis is no longer than the plane, so the serial walk stays bounded.
template <typename Fold, typename T>
__device__ __forceinline__ void unit_reduce_body(const void* in_ptr,
                                                 void* out_ptr,
                                                 const t4a_cuda_layout& out_layout,
                                                 const t4a_cuda_layout& in_layout,
                                                 int axis,
                                                 long long output_len,
                                                 Fold fold) {
  const long long output_index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* out = static_cast<T*>(out_ptr);
  long long in_base;
  long long out_offset;
  t4a_cuda_reduce_offsets(out_layout, in_layout, output_index, &in_base, &out_offset);

  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  T acc = fold.seed(in[in_base]);
  for (long long index = 1; index < axis_len; ++index) {
    acc = fold(acc, in[in_base + index * axis_stride]);
  }
  out[out_offset] = acc;
}

// One block (one plane) per output. Every lane participates in the shuffles: the
// planner guarantees `axis_len >= plane_width`, so no lane returns early, and
// only lane 0 writes.
template <typename Fold, typename T>
__device__ __forceinline__ void plane_reduce_body(const void* in_ptr,
                                                  void* out_ptr,
                                                  const t4a_cuda_layout& out_layout,
                                                  const t4a_cuda_layout& in_layout,
                                                  int axis,
                                                  long long output_len,
                                                  Fold fold) {
  const long long output_index = blockIdx.x;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* out = static_cast<T*>(out_ptr);
  long long in_base;
  long long out_offset;
  t4a_cuda_reduce_offsets(out_layout, in_layout, output_index, &in_base, &out_offset);

  const int plane_width = static_cast<int>(blockDim.x);
  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  long long reduce_index = threadIdx.x;
  T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
  for (reduce_index += plane_width; reduce_index < axis_len; reduce_index += plane_width) {
    acc = fold(acc, in[in_base + reduce_index * axis_stride]);
  }
  const T reduced = t4a_cuda_plane_fold(acc, plane_width, fold);
  if (threadIdx.x == 0) {
    out[out_offset] = reduced;
  }
}

// Pass 1 of a split reduction: one block per (output, split). The planner
// guarantees `chunk_len >= plane_width` and `chunk_len == axis_len / splits`, so
// every lane seeds from inside its own chunk and only the last split absorbs the
// remainder.
template <typename Fold, typename T>
__device__ __forceinline__ void partials_body(const void* in_ptr,
                                              void* partials_ptr,
                                              const t4a_cuda_layout& partials_layout,
                                              const t4a_cuda_layout& in_layout,
                                              int axis,
                                              long long output_len,
                                              long long splits,
                                              long long chunk_len,
                                              Fold fold) {
  const long long cube = blockIdx.x;
  const long long output_index = cube / splits;
  const long long split = cube % splits;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* partials = static_cast<T*>(partials_ptr);

  long long in_base = in_layout.offset;
  long long partial_offset = partials_layout.offset;
  long long rest = output_index;
  for (int dim = 0; dim < partials_layout.rank; ++dim) {
    if (dim == axis) {
      partial_offset += split * partials_layout.strides[dim];
    } else {
      const long long extent = partials_layout.extents[dim];
      const long long coordinate = rest % extent;
      rest /= extent;
      in_base += coordinate * in_layout.strides[dim];
      partial_offset += coordinate * partials_layout.strides[dim];
    }
  }

  const int plane_width = static_cast<int>(blockDim.x);
  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  const long long start = split * chunk_len;
  const long long end = split + 1 == splits ? axis_len : start + chunk_len;
  long long reduce_index = start + threadIdx.x;
  T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
  for (reduce_index += plane_width; reduce_index < end; reduce_index += plane_width) {
    acc = fold(acc, in[in_base + reduce_index * axis_stride]);
  }
  const T reduced = t4a_cuda_plane_fold(acc, plane_width, fold);
  if (threadIdx.x == 0) {
    partials[partial_offset] = reduced;
  }
}

// --- Revision 9 general path ---------------------------------------------------
//
// The same three algorithms with pointer geometry and a walk bounded by the
// operand's own runtime rank. The fast bodies above are unchanged; these keep the
// same fold objects, so the accumulation order per output is identical.

__device__ __forceinline__ void t4a_cuda_reduce_offsets_any(
    const t4a_cuda_layout_any& out_layout, const t4a_cuda_layout_any& in_layout,
    long long output_index, long long* in_base, long long* out_offset) {
  long long rest = output_index;
  long long base = in_layout.offset;
  long long offset = out_layout.offset;
  for (int dim = 0; dim < out_layout.rank; ++dim) {
    const long long extent = out_layout.extents[dim];
    const long long coordinate = rest % extent;
    rest /= extent;
    base += coordinate * in_layout.strides[dim];
    offset += coordinate * out_layout.strides[dim];
  }
  *in_base = base;
  *out_offset = offset;
}

template <typename Fold, typename T>
__device__ __forceinline__ void unit_reduce_body_any(const void* in_ptr,
                                                     void* out_ptr,
                                                     const t4a_cuda_layout_any& out_layout,
                                                     const t4a_cuda_layout_any& in_layout,
                                                     int axis,
                                                     long long output_len,
                                                     Fold fold) {
  const long long output_index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* out = static_cast<T*>(out_ptr);
  long long in_base;
  long long out_offset;
  t4a_cuda_reduce_offsets_any(out_layout, in_layout, output_index, &in_base, &out_offset);

  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  T acc = fold.seed(in[in_base]);
  for (long long index = 1; index < axis_len; ++index) {
    acc = fold(acc, in[in_base + index * axis_stride]);
  }
  out[out_offset] = acc;
}

template <typename Fold, typename T>
__device__ __forceinline__ void plane_reduce_body_any(const void* in_ptr,
                                                      void* out_ptr,
                                                      const t4a_cuda_layout_any& out_layout,
                                                      const t4a_cuda_layout_any& in_layout,
                                                      int axis,
                                                      long long output_len,
                                                      Fold fold) {
  const long long output_index = blockIdx.x;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* out = static_cast<T*>(out_ptr);
  long long in_base;
  long long out_offset;
  t4a_cuda_reduce_offsets_any(out_layout, in_layout, output_index, &in_base, &out_offset);

  const int plane_width = static_cast<int>(blockDim.x);
  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  long long reduce_index = threadIdx.x;
  T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
  for (reduce_index += plane_width; reduce_index < axis_len; reduce_index += plane_width) {
    acc = fold(acc, in[in_base + reduce_index * axis_stride]);
  }
  const T reduced = t4a_cuda_plane_fold(acc, plane_width, fold);
  if (threadIdx.x == 0) {
    out[out_offset] = reduced;
  }
}

template <typename Fold, typename T>
__device__ __forceinline__ void partials_body_any(const void* in_ptr,
                                                  void* partials_ptr,
                                                  const t4a_cuda_layout_any& partials_layout,
                                                  const t4a_cuda_layout_any& in_layout,
                                                  int axis,
                                                  long long output_len,
                                                  long long splits,
                                                  long long chunk_len,
                                                  Fold fold) {
  const long long cube = blockIdx.x;
  const long long output_index = cube / splits;
  const long long split = cube % splits;
  if (output_index >= output_len) {
    return;
  }
  const T* in = static_cast<const T*>(in_ptr);
  T* partials = static_cast<T*>(partials_ptr);

  long long in_base = in_layout.offset;
  long long partial_offset = partials_layout.offset;
  long long rest = output_index;
  for (int dim = 0; dim < partials_layout.rank; ++dim) {
    if (dim == axis) {
      partial_offset += split * partials_layout.strides[dim];
    } else {
      const long long extent = partials_layout.extents[dim];
      const long long coordinate = rest % extent;
      rest /= extent;
      in_base += coordinate * in_layout.strides[dim];
      partial_offset += coordinate * partials_layout.strides[dim];
    }
  }

  const int plane_width = static_cast<int>(blockDim.x);
  const long long axis_stride = in_layout.strides[axis];
  const long long axis_len = in_layout.extents[axis];
  const long long start = split * chunk_len;
  const long long end = split + 1 == splits ? axis_len : start + chunk_len;
  long long reduce_index = start + threadIdx.x;
  T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
  for (reduce_index += plane_width; reduce_index < end; reduce_index += plane_width) {
    acc = fold(acc, in[in_base + reduce_index * axis_stride]);
  }
  const T reduced = t4a_cuda_plane_fold(acc, plane_width, fold);
  if (threadIdx.x == 0) {
    partials[partial_offset] = reduced;
  }
}

}  // namespace

extern "C" __global__ void t4a_cuda_reduce(void* out, const void* in, int dtype, int op,
                                           int strategy, t4a_cuda_layout out_layout,
                                           t4a_cuda_layout in_layout, int axis,
                                           long long output_len) {
  const bool plane = strategy == T4A_CUDA_REDUCE_PLANE;
#define T4A_REDUCE_DISPATCH(FOLD, TYPE)                                                  \
  do {                                                                                   \
    if (plane) {                                                                         \
      plane_reduce_body<FOLD, TYPE>(in, out, out_layout, in_layout, axis, output_len,     \
                                    FOLD());                                             \
    } else {                                                                             \
      unit_reduce_body<FOLD, TYPE>(in, out, out_layout, in_layout, axis, output_len,       \
                                   FOLD());                                              \
    }                                                                                    \
  } while (0)

  switch (op) {
    case T4A_CUDA_REDUCE_SUM:
#define T4A_BODY(T) T4A_REDUCE_DISPATCH(SumFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_PROD:
#define T4A_BODY(T) T4A_REDUCE_DISPATCH(ProdFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_SUM_SQUARES:
#define T4A_BODY(T) T4A_REDUCE_DISPATCH(SumSquaresFold, T)
      T4A_CUDA_FLOAT_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MAX:
#define T4A_BODY(T) T4A_REDUCE_DISPATCH(MaxFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MIN:
#define T4A_BODY(T) T4A_REDUCE_DISPATCH(MinFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
#undef T4A_REDUCE_DISPATCH
}

extern "C" __global__ void t4a_cuda_reduce_partials(void* partials, const void* in, int dtype,
                                                    int op, t4a_cuda_layout partials_layout,
                                                    t4a_cuda_layout in_layout, int axis,
                                                    long long output_len, long long splits,
                                                    long long chunk_len) {
#define T4A_PARTIALS_DISPATCH(FOLD, TYPE)                                                 \
  partials_body<FOLD, TYPE>(in, partials, partials_layout, in_layout, axis, output_len,    \
                            splits, chunk_len, FOLD())

  switch (op) {
    case T4A_CUDA_REDUCE_SUM:
#define T4A_BODY(T) T4A_PARTIALS_DISPATCH(SumFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_PROD:
#define T4A_BODY(T) T4A_PARTIALS_DISPATCH(ProdFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_SUM_SQUARES:
#define T4A_BODY(T) T4A_PARTIALS_DISPATCH(SumSquaresFold, T)
      T4A_CUDA_FLOAT_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MAX:
#define T4A_BODY(T) T4A_PARTIALS_DISPATCH(MaxFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MIN:
#define T4A_BODY(T) T4A_PARTIALS_DISPATCH(MinFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
#undef T4A_PARTIALS_DISPATCH
}

extern "C" __global__ void t4a_cuda_reduce_any(void* out, const void* in, int dtype, int op,
                                               int strategy, t4a_cuda_layout_any out_layout,
                                               t4a_cuda_layout_any in_layout, int axis,
                                               long long output_len) {
  const bool plane = strategy == T4A_CUDA_REDUCE_PLANE;
#define T4A_REDUCE_ANY_DISPATCH(FOLD, TYPE)                                              \
  do {                                                                                   \
    if (plane) {                                                                         \
      plane_reduce_body_any<FOLD, TYPE>(in, out, out_layout, in_layout, axis, output_len, \
                                        FOLD());                                          \
    } else {                                                                             \
      unit_reduce_body_any<FOLD, TYPE>(in, out, out_layout, in_layout, axis, output_len,  \
                                       FOLD());                                          \
    }                                                                                    \
  } while (0)

  switch (op) {
    case T4A_CUDA_REDUCE_SUM:
#define T4A_BODY(T) T4A_REDUCE_ANY_DISPATCH(SumFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_PROD:
#define T4A_BODY(T) T4A_REDUCE_ANY_DISPATCH(ProdFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_SUM_SQUARES:
#define T4A_BODY(T) T4A_REDUCE_ANY_DISPATCH(SumSquaresFold, T)
      T4A_CUDA_FLOAT_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MAX:
#define T4A_BODY(T) T4A_REDUCE_ANY_DISPATCH(MaxFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MIN:
#define T4A_BODY(T) T4A_REDUCE_ANY_DISPATCH(MinFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
#undef T4A_REDUCE_ANY_DISPATCH
}

extern "C" __global__ void t4a_cuda_reduce_partials_any(
    void* partials, const void* in, int dtype, int op, t4a_cuda_layout_any partials_layout,
    t4a_cuda_layout_any in_layout, int axis, long long output_len, long long splits,
    long long chunk_len) {
#define T4A_PARTIALS_ANY_DISPATCH(FOLD, TYPE)                                             \
  partials_body_any<FOLD, TYPE>(in, partials, partials_layout, in_layout, axis,           \
                                output_len, splits, chunk_len, FOLD())

  switch (op) {
    case T4A_CUDA_REDUCE_SUM:
#define T4A_BODY(T) T4A_PARTIALS_ANY_DISPATCH(SumFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_PROD:
#define T4A_BODY(T) T4A_PARTIALS_ANY_DISPATCH(ProdFold, T)
      T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_SUM_SQUARES:
#define T4A_BODY(T) T4A_PARTIALS_ANY_DISPATCH(SumSquaresFold, T)
      T4A_CUDA_FLOAT_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MAX:
#define T4A_BODY(T) T4A_PARTIALS_ANY_DISPATCH(MaxFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    case T4A_CUDA_REDUCE_MIN:
#define T4A_BODY(T) T4A_PARTIALS_ANY_DISPATCH(MinFold, T)
      T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
      break;
    default:
      break;
  }
#undef T4A_PARTIALS_ANY_DISPATCH
}
