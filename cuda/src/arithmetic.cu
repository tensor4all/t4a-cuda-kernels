// t4a-cuda-kernels: element-wise add, sub, mul and float division.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/elementwise.rs: the binary_float_kernel (L33),
// binary_float_complex_kernel (L46), add_int (L210), sub_int (L217),
// mul_int (L224) and div_float (L208) entries, with the wrapping and complex
// value forms of crates/tenferro-gpu/src/kernels/helpers.rs. Copyright (c) 2026
// Hiroshi Shinaoka and tenferro-rs contributors; MIT OR Apache-2.0.
//
// Bool arithmetic is not part of the baseline dispatch and is not exported.
// Complex division keeps the baseline's scaled Baudin-Smith form and is not
// ported in this revision.

#include <t4a/abi.cuh>

namespace {

// One lane of each entry: the same scalar expression for every vector width.
struct add_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_add<T>(lhs, rhs);
  }
};

struct sub_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_sub<T>(lhs, rhs);
  }
};

struct mul_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_mul<T>(lhs, rhs);
  }
};

struct div_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return t4a_cuda_div<T>(lhs, rhs);
  }
};

}  // namespace

#define T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, BODY) \
  switch (dtype) {                                    \
    case T4A_CUDA_F32:                                \
      BODY(float);                                     \
      break;                                          \
    case T4A_CUDA_F64:                                \
      BODY(double);                                    \
      break;                                          \
    case T4A_CUDA_I32:                                \
      BODY(int);                                       \
      break;                                          \
    case T4A_CUDA_I64:                                \
      BODY(long long);                                 \
      break;                                          \
    case T4A_CUDA_C32:                                \
      BODY(t4a_cuda_c32);                              \
      break;                                          \
    case T4A_CUDA_C64:                                \
      BODY(t4a_cuda_c64);                              \
      break;                                          \
    default:                                          \
      break;                                          \
  }

extern "C" __global__ void t4a_cuda_add(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len, int vector_width) {
#define T4A_BODY(T) t4a_cuda_flat_binary<T, T, T>(vector_width, out, lhs, rhs, len, add_op{})
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_sub(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len, int vector_width) {
#define T4A_BODY(T) t4a_cuda_flat_binary<T, T, T>(vector_width, out, lhs, rhs, len, sub_op{})
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_mul(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len, int vector_width) {
#define T4A_BODY(T) t4a_cuda_flat_binary<T, T, T>(vector_width, out, lhs, rhs, len, mul_op{})
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_div(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len, int vector_width) {
  switch (dtype) {
    case T4A_CUDA_F32:
      t4a_cuda_flat_binary<float, float, float>(vector_width, out, lhs, rhs, len, div_op{});
      break;
    case T4A_CUDA_F64:
      t4a_cuda_flat_binary<double, double, double>(vector_width, out, lhs, rhs, len, div_op{});
      break;
    default:
      break;
  }
}
