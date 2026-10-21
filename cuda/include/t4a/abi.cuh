// t4a-cuda-kernels: shared kernel ABI and device helpers.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// (crates/tenferro-gpu/src/kernels/helpers.rs, structural.rs element indexing
// and elementwise.rs arithmetic helper forms). Copyright (c) 2026 Hiroshi
// Shinaoka and tenferro-rs contributors; MIT OR Apache-2.0.
//
// The header deliberately includes nothing: the same text must compile under
// NVCC and NVRTC, where standard headers are not guaranteed to be available.

#ifndef T4A_CUDA_ABI_CUH_
#define T4A_CUDA_ABI_CUH_

#if !defined(__cplusplus)
#error "t4a/abi.cuh requires C++"
#endif

#define T4A_CUDA_MAX_RANK 8
#define T4A_CUDA_ABI_REVISION 5

static_assert(sizeof(int) == 4, "the C ABI requires 32-bit int");
static_assert(sizeof(long long) == 8, "the C ABI requires 64-bit long long");

typedef enum {
  T4A_CUDA_F32 = 0,
  T4A_CUDA_F64 = 1,
  T4A_CUDA_I32 = 2,
  T4A_CUDA_I64 = 3,
  T4A_CUDA_BOOL = 4,
  T4A_CUDA_C32 = 5,
  T4A_CUDA_C64 = 6
} t4a_cuda_dtype;

// Interleaved complex with the scalar's alignment, matching
// num_complex::Complex32/64 rather than CUDA's float2/double2.
typedef struct {
  float re;
  float im;
} t4a_cuda_c32;

typedef struct {
  double re;
  double im;
} t4a_cuda_c64;

// Logical strided layout. Axis 0 is the fastest-varying axis: a flat index
// decodes with coordinate = flat % extent, flat /= extent, and the address is
// offset + sum(coordinate * stride). Strides and the offset are signed element
// counts; stride zero is a broadcast axis.
typedef struct {
  int rank;
  int reserved;
  long long offset;
  long long extents[T4A_CUDA_MAX_RANK];
  long long strides[T4A_CUDA_MAX_RANK];
} t4a_cuda_layout;

// Reduction operation and strategy tags (revision 3).
typedef enum {
  T4A_CUDA_REDUCE_SUM = 0,
  T4A_CUDA_REDUCE_PROD = 1,
  T4A_CUDA_REDUCE_SUM_SQUARES = 2,
  T4A_CUDA_REDUCE_MAX = 3,
  T4A_CUDA_REDUCE_MIN = 4
} t4a_cuda_reduce_op;

typedef enum {
  T4A_CUDA_REDUCE_UNIT = 0,
  T4A_CUDA_REDUCE_PLANE = 1
} t4a_cuda_reduce_strategy;

// Per-axis runtime values of the windowing operations (slice starts and steps,
// padding amounts). `rank` must equal the operation rank.
typedef struct {
  int rank;
  int reserved;
  long long values[T4A_CUDA_MAX_RANK];
} t4a_cuda_axis_values;

// Dimension maps of the XLA-style gather (revision 4).
typedef struct {
  int operand_rank;
  int out_rank;
  int index_vector_dim;
  int window_rank;
  int start_map_len;
  int reserved;
  long long window_dims[T4A_CUDA_MAX_RANK];
  long long offset_dims[T4A_CUDA_MAX_RANK];
  long long start_index_map[T4A_CUDA_MAX_RANK];
  long long slice_sizes[T4A_CUDA_MAX_RANK];
} t4a_cuda_gather_plan;

// Dimension maps of the additive scatter. The window shape comes from the
// updates extents at `update_window_dims`, so the plan carries no slice sizes.
typedef struct {
  int operand_rank;
  int updates_rank;
  int index_vector_dim;
  int window_rank;
  int scatter_map_len;
  int reserved;
  long long window_dims[T4A_CUDA_MAX_RANK];
  long long update_window_dims[T4A_CUDA_MAX_RANK];
  long long scatter_dims[T4A_CUDA_MAX_RANK];
} t4a_cuda_scatter_plan;

// Algorithm configuration of the tiled transpose, selected at runtime.
typedef struct {
  int tile;
  int block_rows;
  int padding;
  int vector_width;
} t4a_cuda_transpose_config;

static_assert(sizeof(t4a_cuda_layout) == 144, "t4a_cuda_layout ABI size");
static_assert(alignof(t4a_cuda_layout) == 8, "t4a_cuda_layout ABI alignment");
static_assert(__builtin_offsetof(t4a_cuda_layout, offset) == 8, "t4a_cuda_layout.offset offset");
static_assert(__builtin_offsetof(t4a_cuda_layout, extents) == 16, "t4a_cuda_layout.extents offset");
static_assert(__builtin_offsetof(t4a_cuda_layout, strides) == 80, "t4a_cuda_layout.strides offset");
static_assert(alignof(t4a_cuda_transpose_config) == 4, "t4a_cuda_transpose_config ABI alignment");
static_assert(sizeof(t4a_cuda_transpose_config) == 16, "t4a_cuda_transpose_config ABI size");
static_assert(sizeof(t4a_cuda_axis_values) == 72, "t4a_cuda_axis_values ABI size");
static_assert(sizeof(t4a_cuda_gather_plan) == 280, "t4a_cuda_gather_plan ABI size");
static_assert(alignof(t4a_cuda_gather_plan) == 8, "t4a_cuda_gather_plan ABI alignment");
static_assert(__builtin_offsetof(t4a_cuda_gather_plan, window_dims) == 24,
              "t4a_cuda_gather_plan.window_dims offset");
static_assert(sizeof(t4a_cuda_scatter_plan) == 216, "t4a_cuda_scatter_plan ABI size");
static_assert(alignof(t4a_cuda_scatter_plan) == 8, "t4a_cuda_scatter_plan ABI alignment");
static_assert(__builtin_offsetof(t4a_cuda_scatter_plan, window_dims) == 24,
              "t4a_cuda_scatter_plan.window_dims offset");
static_assert(alignof(t4a_cuda_axis_values) == 8, "t4a_cuda_axis_values ABI alignment");
static_assert(__builtin_offsetof(t4a_cuda_axis_values, values) == 8, "t4a_cuda_axis_values.values offset");
static_assert(sizeof(t4a_cuda_c32) == 8, "t4a_cuda_c32 ABI size");
static_assert(alignof(t4a_cuda_c32) == 4, "t4a_cuda_c32 ABI alignment");
static_assert(sizeof(t4a_cuda_c64) == 16, "t4a_cuda_c64 ABI size");
static_assert(alignof(t4a_cuda_c64) == 8, "t4a_cuda_c64 ABI alignment");

// Expands one templated body per supported dtype. Unknown tags are callers'
// responsibility; the Rust facade rejects them before planning.
#define T4A_CUDA_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                         \
    case T4A_CUDA_F32:                     \
      BODY(float);                          \
      break;                               \
    case T4A_CUDA_F64:                     \
      BODY(double);                         \
      break;                               \
    case T4A_CUDA_I32:                     \
      BODY(int);                            \
      break;                               \
    case T4A_CUDA_I64:                     \
      BODY(long long);                      \
      break;                               \
    case T4A_CUDA_BOOL:                    \
      BODY(unsigned char);                  \
      break;                               \
    case T4A_CUDA_C32:                     \
      BODY(t4a_cuda_c32);                   \
      break;                               \
    case T4A_CUDA_C64:                     \
      BODY(t4a_cuda_c64);                   \
      break;                               \
    default:                               \
      break;                               \
  }

// Dtype subsets for the reduction entries.
#define T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                                 \
    case T4A_CUDA_F32:                             \
      BODY(float);                                 \
      break;                                       \
    case T4A_CUDA_F64:                             \
      BODY(double);                                \
      break;                                       \
    case T4A_CUDA_I32:                             \
      BODY(int);                                   \
      break;                                       \
    case T4A_CUDA_I64:                             \
      BODY(long long);                             \
      break;                                       \
    case T4A_CUDA_C32:                             \
      BODY(t4a_cuda_c32);                          \
      break;                                       \
    case T4A_CUDA_C64:                             \
      BODY(t4a_cuda_c64);                          \
      break;                                       \
    default:                                       \
      break;                                       \
  }

#define T4A_CUDA_ORDERED_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                                 \
    case T4A_CUDA_F32:                             \
      BODY(float);                                 \
      break;                                       \
    case T4A_CUDA_F64:                             \
      BODY(double);                                \
      break;                                       \
    case T4A_CUDA_I32:                             \
      BODY(int);                                   \
      break;                                       \
    case T4A_CUDA_I64:                             \
      BODY(long long);                             \
      break;                                       \
    default:                                       \
      break;                                       \
  }

#define T4A_CUDA_FLOAT_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                               \
    case T4A_CUDA_F32:                           \
      BODY(float);                               \
      break;                                     \
    case T4A_CUDA_F64:                           \
      BODY(double);                              \
      break;                                     \
    default:                                     \
      break;                                     \
  }

// Address of logical element `flat` in `layout`, or the address of the same
// logical coordinate in a layout that shares `extents_from`'s extents and uses
// `strides_from`'s strides and offset.
__device__ __forceinline__ long long t4a_cuda_decode(long long flat,
                                                     int rank,
                                                     const long long* extents,
                                                     const long long* strides,
                                                     long long offset) {
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis >= rank) {
      break;
    }
    const long long extent = extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    offset += coordinate * strides[axis];
  }
  return offset;
}

__device__ __forceinline__ bool t4a_cuda_tail_index(long long* index, long long len) {
  *index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  return *index < len;
}

template <typename T>
__device__ __forceinline__ T t4a_cuda_zero() {
  return T(0);
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_zero<t4a_cuda_c32>() {
  t4a_cuda_c32 value = {0.0f, 0.0f};
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_zero<t4a_cuda_c64>() {
  t4a_cuda_c64 value = {0.0, 0.0};
  return value;
}

// Integer arithmetic keeps the baseline's defined wrapping behavior; CUDA C++
// signed overflow would otherwise be undefined.
template <typename T>
__device__ __forceinline__ T t4a_cuda_add(T lhs, T rhs) {
  return lhs + rhs;
}

template <>
__device__ __forceinline__ int t4a_cuda_add<int>(int lhs, int rhs) {
  return static_cast<int>(static_cast<unsigned>(lhs) + static_cast<unsigned>(rhs));
}

template <>
__device__ __forceinline__ long long t4a_cuda_add<long long>(long long lhs, long long rhs) {
  return static_cast<long long>(static_cast<unsigned long long>(lhs) +
                                static_cast<unsigned long long>(rhs));
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_add<t4a_cuda_c32>(t4a_cuda_c32 lhs,
                                                                 t4a_cuda_c32 rhs) {
  t4a_cuda_c32 value = {lhs.re + rhs.re, lhs.im + rhs.im};
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_add<t4a_cuda_c64>(t4a_cuda_c64 lhs,
                                                                 t4a_cuda_c64 rhs) {
  t4a_cuda_c64 value = {lhs.re + rhs.re, lhs.im + rhs.im};
  return value;
}

template <typename T>
__device__ __forceinline__ T t4a_cuda_sub(T lhs, T rhs) {
  return lhs - rhs;
}

template <>
__device__ __forceinline__ int t4a_cuda_sub<int>(int lhs, int rhs) {
  return static_cast<int>(static_cast<unsigned>(lhs) - static_cast<unsigned>(rhs));
}

template <>
__device__ __forceinline__ long long t4a_cuda_sub<long long>(long long lhs, long long rhs) {
  return static_cast<long long>(static_cast<unsigned long long>(lhs) -
                                static_cast<unsigned long long>(rhs));
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_sub<t4a_cuda_c32>(t4a_cuda_c32 lhs,
                                                                 t4a_cuda_c32 rhs) {
  t4a_cuda_c32 value = {lhs.re - rhs.re, lhs.im - rhs.im};
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_sub<t4a_cuda_c64>(t4a_cuda_c64 lhs,
                                                                 t4a_cuda_c64 rhs) {
  t4a_cuda_c64 value = {lhs.re - rhs.re, lhs.im - rhs.im};
  return value;
}

template <typename T>
__device__ __forceinline__ T t4a_cuda_mul(T lhs, T rhs) {
  return lhs * rhs;
}

template <>
__device__ __forceinline__ int t4a_cuda_mul<int>(int lhs, int rhs) {
  return static_cast<int>(static_cast<unsigned>(lhs) * static_cast<unsigned>(rhs));
}

template <>
__device__ __forceinline__ long long t4a_cuda_mul<long long>(long long lhs, long long rhs) {
  return static_cast<long long>(static_cast<unsigned long long>(lhs) *
                                static_cast<unsigned long long>(rhs));
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_mul<t4a_cuda_c32>(t4a_cuda_c32 lhs,
                                                                 t4a_cuda_c32 rhs) {
  t4a_cuda_c32 value = {lhs.re * rhs.re - lhs.im * rhs.im, lhs.re * rhs.im + lhs.im * rhs.re};
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_mul<t4a_cuda_c64>(t4a_cuda_c64 lhs,
                                                                 t4a_cuda_c64 rhs) {
  t4a_cuda_c64 value = {lhs.re * rhs.re - lhs.im * rhs.im, lhs.re * rhs.im + lhs.im * rhs.re};
  return value;
}

// Index values of the dynamic-slice starts tensor. Truncation matches the
// baseline's cast, but the baseline validates float starts before the launch, so
// a fractional or nonfinite start is a caller error rather than data.
template <typename I>
__device__ __forceinline__ unsigned long long t4a_cuda_index_to_u64(I value) {
  return static_cast<unsigned long long>(value);
}

// Nonfinite detection by bit pattern: exact for NaN and +/-Infinity without math
// headers, and independent of any x - x folding.
__device__ __forceinline__ bool t4a_cuda_is_finite(float value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ bool t4a_cuda_is_finite(double value) {
  return (__double_as_longlong(value) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}

// Lowest invalid position wins; the flag domain is capped at UINT32_MAX by the
// facade, matching the 32-bit status word.
__device__ __forceinline__ void t4a_cuda_note_invalid(unsigned int* flag, long long position) {
  atomicMin(flag, static_cast<unsigned int>(position));
}

// Warp-plane collectives. Offsets ascend (1, 2, 4, ... < plane_width) to keep the
// baseline's combination order; `mask` names exactly the participating lanes.
__device__ __forceinline__ unsigned int t4a_cuda_plane_mask(int plane_width) {
  return plane_width >= 32 ? 0xFFFFFFFFu : ((1u << plane_width) - 1u);
}

__device__ __forceinline__ float t4a_cuda_shfl_xor(float value, int offset, unsigned int mask) {
  return __shfl_xor_sync(mask, value, offset);
}

__device__ __forceinline__ double t4a_cuda_shfl_xor(double value, int offset, unsigned int mask) {
  return __shfl_xor_sync(mask, value, offset);
}

__device__ __forceinline__ int t4a_cuda_shfl_xor(int value, int offset, unsigned int mask) {
  return __shfl_xor_sync(mask, value, offset);
}

__device__ __forceinline__ long long t4a_cuda_shfl_xor(long long value, int offset,
                                                      unsigned int mask) {
  return __shfl_xor_sync(mask, value, offset);
}

__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_shfl_xor(t4a_cuda_c32 value, int offset,
                                                          unsigned int mask) {
  t4a_cuda_c32 result = {__shfl_xor_sync(mask, value.re, offset),
                         __shfl_xor_sync(mask, value.im, offset)};
  return result;
}

__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_shfl_xor(t4a_cuda_c64 value, int offset,
                                                          unsigned int mask) {
  t4a_cuda_c64 result = {__shfl_xor_sync(mask, value.re, offset),
                         __shfl_xor_sync(mask, value.im, offset)};
  return result;
}

// NaN-propagating extrema: a NaN operand wins, otherwise the ordered maximum.
__device__ __forceinline__ float t4a_cuda_nan_max(float lhs, float rhs) {
  if (lhs != lhs) {
    return lhs;
  }
  if (rhs != rhs) {
    return rhs;
  }
  return fmaxf(lhs, rhs);
}

__device__ __forceinline__ double t4a_cuda_nan_max(double lhs, double rhs) {
  if (lhs != lhs) {
    return lhs;
  }
  if (rhs != rhs) {
    return rhs;
  }
  return fmax(lhs, rhs);
}

__device__ __forceinline__ float t4a_cuda_nan_min(float lhs, float rhs) {
  if (lhs != lhs) {
    return lhs;
  }
  if (rhs != rhs) {
    return rhs;
  }
  return fminf(lhs, rhs);
}

__device__ __forceinline__ double t4a_cuda_nan_min(double lhs, double rhs) {
  if (lhs != lhs) {
    return lhs;
  }
  if (rhs != rhs) {
    return rhs;
  }
  return fmin(lhs, rhs);
}

// Integer extrema: plain ordering.
template <typename T>
__device__ __forceinline__ T t4a_cuda_max(T lhs, T rhs) {
  return lhs > rhs ? lhs : rhs;
}

template <>
__device__ __forceinline__ float t4a_cuda_max<float>(float lhs, float rhs) {
  return t4a_cuda_nan_max(lhs, rhs);
}

template <>
__device__ __forceinline__ double t4a_cuda_max<double>(double lhs, double rhs) {
  return t4a_cuda_nan_max(lhs, rhs);
}

template <typename T>
__device__ __forceinline__ T t4a_cuda_min(T lhs, T rhs) {
  return lhs < rhs ? lhs : rhs;
}

template <>
__device__ __forceinline__ float t4a_cuda_min<float>(float lhs, float rhs) {
  return t4a_cuda_nan_min(lhs, rhs);
}

template <>
__device__ __forceinline__ double t4a_cuda_min<double>(double lhs, double rhs) {
  return t4a_cuda_nan_min(lhs, rhs);
}

// Square through fma so the square rounds before any later addition and the
// compiler cannot contract it into that addition.
template <typename T>
__device__ __forceinline__ T t4a_cuda_square(T value) {
  return value * value;
}

template <>
__device__ __forceinline__ float t4a_cuda_square<float>(float value) {
  return __fmaf_rn(value, value, 0.0f);
}

template <>
__device__ __forceinline__ double t4a_cuda_square<double>(double value) {
  return __fma_rn(value, value, 0.0);
}

// Division is float-only in this ABI revision; complex division keeps the
// baseline's scaled Baudin-Smith form and is not ported here.
template <typename T>
__device__ __forceinline__ T t4a_cuda_div(T lhs, T rhs) {
  return lhs / rhs;
}

#endif  // T4A_CUDA_ABI_CUH_
