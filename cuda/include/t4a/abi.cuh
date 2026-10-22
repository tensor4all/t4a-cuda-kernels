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
#define T4A_CUDA_ABI_REVISION 7

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

// Operation tags of the elementwise entries (revision 6). The tags are stable
// and append-only; an unsupported (tag, dtype) pair is the caller's error and
// is rejected by the Rust facade before planning.
typedef enum {
  T4A_CUDA_UNARY_NEG = 0,
  T4A_CUDA_UNARY_EXP = 1,
  T4A_CUDA_UNARY_LOG = 2,
  T4A_CUDA_UNARY_SIN = 3,
  T4A_CUDA_UNARY_COS = 4,
  T4A_CUDA_UNARY_TANH = 5,
  T4A_CUDA_UNARY_SQRT = 6,
  T4A_CUDA_UNARY_RSQRT = 7,
  T4A_CUDA_UNARY_EXPM1 = 8,
  T4A_CUDA_UNARY_LOG1P = 9,
  T4A_CUDA_UNARY_ERF = 10,
  T4A_CUDA_UNARY_ABS = 11,
  T4A_CUDA_UNARY_SIGN = 12,
  T4A_CUDA_UNARY_CONJ = 13
} t4a_cuda_unary_op;

typedef enum {
  T4A_CUDA_BINARY_MAX = 0,
  T4A_CUDA_BINARY_MIN = 1,
  T4A_CUDA_BINARY_POW = 2,
  T4A_CUDA_BINARY_REM = 3,
  T4A_CUDA_BINARY_MUL = 4,
  T4A_CUDA_BINARY_DIV_COMPLEX = 5,
  T4A_CUDA_BINARY_DIV_CHECKED = 6,
  T4A_CUDA_BINARY_REM_CHECKED = 7,
  T4A_CUDA_BINARY_POW_CHECKED = 8,
  T4A_CUDA_BINARY_MIXED_ADD = 9,
  T4A_CUDA_BINARY_MIXED_SUB = 10,
  T4A_CUDA_BINARY_MIXED_MUL = 11,
  T4A_CUDA_BINARY_MIXED_DIV = 12,
  T4A_CUDA_BINARY_DIV = 13
} t4a_cuda_binary_op;

// Which operand is the single-element scalar: both are elementwise, or one side
// is read at index 0. For the mixed real/complex tags, `LHS`/`RHS` names the real
// operand.
typedef enum {
  T4A_CUDA_SCALAR_BOTH = 0,
  T4A_CUDA_SCALAR_LHS = 1,
  T4A_CUDA_SCALAR_RHS = 2
} t4a_cuda_scalar_side;

typedef enum {
  T4A_CUDA_COMPARE_EQ = 0,
  T4A_CUDA_COMPARE_LT = 1,
  T4A_CUDA_COMPARE_LE = 2,
  T4A_CUDA_COMPARE_GT = 3,
  T4A_CUDA_COMPARE_GE = 4
} t4a_cuda_compare_op;

typedef enum {
  T4A_CUDA_STRIDED_MUL = 0,
  T4A_CUDA_STRIDED_AXPBY = 1
} t4a_cuda_strided_op;

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

// Float and complex dtypes: the linalg entries whose phase and LU payloads have no
// integer or Bool form.
#define T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                                 \
    case T4A_CUDA_F32:                             \
      BODY(float);                                 \
      break;                                       \
    case T4A_CUDA_F64:                             \
      BODY(double);                                \
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

template <typename T>
__device__ __forceinline__ T t4a_cuda_one() {
  return T(1);
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_one<t4a_cuda_c32>() {
  t4a_cuda_c32 value = {1.0f, 0.0f};
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_one<t4a_cuda_c64>() {
  t4a_cuda_c64 value = {1.0, 0.0};
  return value;
}

// Conjugation is the identity for the real and integer element types, so the
// linalg entries can share one templated body.
template <typename T>
__device__ __forceinline__ T t4a_cuda_conj(T value) {
  return value;
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_conj<t4a_cuda_c32>(t4a_cuda_c32 value) {
  t4a_cuda_c32 result = {value.re, -value.im};
  return result;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_conj<t4a_cuda_c64>(t4a_cuda_c64 value) {
  t4a_cuda_c64 result = {value.re, -value.im};
  return result;
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

// Plain division. Complex division does not come through here: it keeps the
// baseline's scaled Baudin-Smith form in t4a_cuda_div_complex_parts_at below.
template <typename T>
__device__ __forceinline__ T t4a_cuda_div(T lhs, T rhs) {
  return lhs / rhs;
}

// Integer negation wraps; CUDA C++ signed overflow would otherwise be undefined.
template <typename T>
__device__ __forceinline__ T t4a_cuda_neg(T value) {
  return -value;
}

template <>
__device__ __forceinline__ int t4a_cuda_neg<int>(int value) {
  return t4a_cuda_sub<int>(0, value);
}

template <>
__device__ __forceinline__ long long t4a_cuda_neg<long long>(long long value) {
  return t4a_cuda_sub<long long>(0, value);
}

template <>
__device__ __forceinline__ t4a_cuda_c32 t4a_cuda_neg<t4a_cuda_c32>(t4a_cuda_c32 value) {
  t4a_cuda_c32 result = {-value.re, -value.im};
  return result;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 t4a_cuda_neg<t4a_cuda_c64>(t4a_cuda_c64 value) {
  t4a_cuda_c64 result = {-value.re, -value.im};
  return result;
}

// Scale-robust complex division over interleaved parts, ported from tenferro-rs
// `elementwise.rs:div_complex_parts_at`, which derives from Julia Base
// `complex.jl` (`cdiv`, `robust_cdiv1/2`, `scaling_cdiv`, `scaleargs_cdiv`).
// Copyright (c) 2009-2026 Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and
// other contributors; MIT. See THIRD_PARTY_NOTICES.md.
//
// `out[index]`/`out[index + 1]` receive the real and imaginary parts of
// `(a0 + b0 i) / (c0 + d0 i)`. An extreme divisor stays finite, a zero result
// keeps the signed-zero sign of the operands, and a nonfinite result is NaN.
template <typename F>
struct t4a_cuda_float_limits;

template <>
struct t4a_cuda_float_limits<float> {
  static constexpr float max_value = 3.40282346638528859812e+38f;
  static constexpr float min_positive = 1.17549435082228750797e-38f;
  static constexpr float epsilon = 1.1920928955078125e-07f;
  static __device__ __forceinline__ float nan() { return __int_as_float(0x7fc00000); }
};

template <>
struct t4a_cuda_float_limits<double> {
  static constexpr double max_value = 1.79769313486231570815e+308;
  static constexpr double min_positive = 2.22507385850720138309e-308;
  static constexpr double epsilon = 2.22044604925031308085e-16;
  static __device__ __forceinline__ double nan() { return __longlong_as_double(0x7ff8000000000000ll); }
};

// |x| without the Abs expression, whose result type cannot be recombined.
template <typename F>
__device__ __forceinline__ F t4a_cuda_abs(F value) {
  return value < F(0) ? -value : value;
}

// sign(x) preserving the sign of zero, as Julia Base uses it.
template <typename F>
__device__ __forceinline__ F t4a_cuda_signed_unit(F value) {
  F result = F(1);
  if (value == F(0)) {
    result = value;
  } else if (value < F(0)) {
    result = F(-1);
  }
  return result;
}

// One Baudin-Smith numerator/denominator term, ported from Julia Base
// `robust_cdiv2`.
template <typename F>
__device__ __forceinline__ F t4a_cuda_complex_div_term(F a, F b, F c, F d, F r, F t) {
  if (r != F(0)) {
    const F br = b * r;
    if (br != F(0)) {
      return (a + br) * t;
    }
    return a * t + (b * t) * r;
  }
  return (a + d * (b / c)) * t;
}

template <typename F>
__device__ __forceinline__ void t4a_cuda_div_complex_parts_at(F a0, F b0, F c0, F d0, F* out,
                                                             long long index) {
  constexpr F max_value = t4a_cuda_float_limits<F>::max_value;
  constexpr F min_positive = t4a_cuda_float_limits<F>::min_positive;
  constexpr F epsilon = t4a_cuda_float_limits<F>::epsilon;
  F a = a0;
  F b = b0;
  F c = c0;
  F d = d0;
  if (t4a_cuda_abs(c) > max_value || t4a_cuda_abs(d) > max_value) {
    const bool a_finite = a == a && t4a_cuda_abs(a) <= max_value;
    const bool b_finite = b == b && t4a_cuda_abs(b) <= max_value;
    if (a_finite && b_finite) {
      out[index] = F(0) * t4a_cuda_signed_unit(a) * t4a_cuda_signed_unit(c);
      out[index + 1] = -F(0) * t4a_cuda_signed_unit(b) * t4a_cuda_signed_unit(d);
    } else {
      // F::NAN codegens to an undefined `NaN` identifier in the baseline's dialect.
      const F nan = t4a_cuda_float_limits<F>::nan();
      out[index] = nan;
      out[index + 1] = nan;
    }
    return;
  }
  const F abs_a = t4a_cuda_abs(a);
  const F abs_b = t4a_cuda_abs(b);
  const F abs_c = t4a_cuda_abs(c);
  const F abs_d = t4a_cuda_abs(d);
  const F ab = abs_a >= abs_b ? abs_a : abs_b;
  const F cd = abs_c >= abs_d ? abs_c : abs_d;
  const F half = F(0.5);
  const F two = F(2);
  const F half_over = max_value * half;
  const F two_under_eps = min_positive * two / epsilon;
  F scale = F(1);
  if (ab >= half_over || ab <= two_under_eps || cd >= half_over || cd <= two_under_eps) {
    const F big = two / (epsilon * epsilon);
    if (ab >= half_over) {
      a = a * half;
      b = b * half;
      scale = scale * two;
    } else if (ab <= two_under_eps) {
      a = a * big;
      b = b * big;
      scale = scale / big;
    }
    if (cd >= half_over) {
      c = c * half;
      d = d * half;
      scale = scale * half;
    } else if (cd <= two_under_eps) {
      c = c * big;
      d = d * big;
      scale = scale * big;
    }
  }
  if (t4a_cuda_abs(d) <= t4a_cuda_abs(c)) {
    const F r = d / c;
    const F t = F(1) / (c + d * r);
    out[index] = t4a_cuda_complex_div_term<F>(a, b, c, d, r, t) * scale;
    out[index + 1] = t4a_cuda_complex_div_term<F>(b, -a, c, d, r, t) * scale;
  } else {
    const F r = c / d;
    const F t = F(1) / (c * r + d);
    out[index] = t4a_cuda_complex_div_term<F>(b, a, d, c, r, t) * scale;
    out[index + 1] = -t4a_cuda_complex_div_term<F>(a, -b, d, c, r, t) * scale;
  }
}

#endif  // T4A_CUDA_ABI_CUH_
