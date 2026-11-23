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

template <typename T>
__device__ __forceinline__ void add_body(void* out, const void* lhs, const void* rhs,
                                         long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(out)[index] =
        t4a_cuda_add<T>(static_cast<const T*>(lhs)[index], static_cast<const T*>(rhs)[index]);
  }
}

template <typename T>
__device__ __forceinline__ void sub_body(void* out, const void* lhs, const void* rhs,
                                         long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(out)[index] =
        t4a_cuda_sub<T>(static_cast<const T*>(lhs)[index], static_cast<const T*>(rhs)[index]);
  }
}

template <typename T>
__device__ __forceinline__ void mul_body(void* out, const void* lhs, const void* rhs,
                                         long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(out)[index] =
        t4a_cuda_mul<T>(static_cast<const T*>(lhs)[index], static_cast<const T*>(rhs)[index]);
  }
}

template <typename T>
__device__ __forceinline__ void div_body(void* out, const void* lhs, const void* rhs,
                                         long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(out)[index] =
        t4a_cuda_div<T>(static_cast<const T*>(lhs)[index], static_cast<const T*>(rhs)[index]);
  }
}

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
                                        long long len) {
#define T4A_BODY(T) add_body<T>(out, lhs, rhs, len)
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_sub(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len) {
#define T4A_BODY(T) sub_body<T>(out, lhs, rhs, len)
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_mul(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len) {
#define T4A_BODY(T) mul_body<T>(out, lhs, rhs, len)
  T4A_CUDA_ARITHMETIC_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_div(void* out, const void* lhs, const void* rhs, int dtype,
                                        long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      div_body<float>(out, lhs, rhs, len);
      break;
    case T4A_CUDA_F64:
      div_body<double>(out, lhs, rhs, len);
      break;
    default:
      break;
  }
}
