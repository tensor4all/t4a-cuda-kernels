// t4a-cuda-kernels: device-side validation status for index tensors and real
// casts.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/indexing.rs: init_float_index_validation_flag
// (L12), validate_float_indices_kernel (L22), extract_invalid_float_index_kernel
// (L47); and crates/tenferro-gpu/src/kernels/structural.rs: validate_real_cast
// (L419), extract_invalid_real_cast (L447). Copyright (c) 2026 Hiroshi Shinaoka
// and tenferro-rs contributors; MIT OR Apache-2.0.
//
// The flag is one 32-bit device word holding the lowest invalid position, so the
// caller caps the validation domain at UINT32_MAX elements. Initialization and
// extraction are single-thread work; the caller decides when to read the flag.
// Nonfinite and NaN detection uses comparisons only (no math headers), which is
// equivalent to the baseline's isnan/isinf/range predicate because the bounds are
// finite.

#include <t4a/abi.cuh>

namespace {

constexpr unsigned int kNoInvalid = 0xFFFFFFFFu;

template <typename T>
__device__ __forceinline__ void init_validation_body(unsigned int* flag, void* value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    flag[0] = kNoInvalid;
    *static_cast<T*>(value) = t4a_cuda_zero<T>();
  }
}

template <typename T>
__device__ __forceinline__ void validate_float_indices_body(const void* indices_ptr,
                                                            unsigned int* flag,
                                                            T max_exact_integer,
                                                            long long len) {
  long long position;
  if (!t4a_cuda_tail_index(&position, len)) {
    return;
  }
  const T value = static_cast<const T*>(indices_ptr)[position];
  // NaN and +/-Infinity are rejected explicitly; a fraction differs from its
  // floor, and the bound is a finite exact-integer limit.
  const bool in_range = value >= -max_exact_integer && value <= max_exact_integer;
  const bool integral = value == floor(value);
  if (!t4a_cuda_is_finite(value) || !in_range || !integral) {
    t4a_cuda_note_invalid(flag, position);
  }
}

template <typename T>
__device__ __forceinline__ void extract_invalid_float_index_body(const void* indices_ptr,
                                                                 const unsigned int* flag,
                                                                 void* value_ptr) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned int position = flag[0];
    if (position != kNoInvalid) {
      *static_cast<T*>(value_ptr) = static_cast<const T*>(indices_ptr)[position];
    }
  }
}

template <typename T>
__device__ __forceinline__ void validate_real_cast_body(const void* input_ptr,
                                                        unsigned int* flag,
                                                        T min_value,
                                                        T max_value,
                                                        long long stride,
                                                        int max_inclusive,
                                                        long long len) {
  long long position;
  if (!t4a_cuda_tail_index(&position, len)) {
    return;
  }
  // Bounds-check by division so an accepted plan with a huge stride cannot
  // overflow the position-times-stride product.
  if (position > (len - 1) / stride) {
    return;
  }
  const long long sampled = position * stride;
  const T value = static_cast<const T*>(input_ptr)[sampled];
  const bool invalid_max = max_inclusive ? value > max_value : value >= max_value;
  // The baseline rejects NaN and Infinity explicitly; finite bounds alone cannot
  // reject +Infinity when the caller passes an infinite maximum.
  if (!t4a_cuda_is_finite(value) || value < min_value || invalid_max) {
    t4a_cuda_note_invalid(flag, position);
  }
}

template <typename T>
__device__ __forceinline__ void extract_invalid_real_cast_body(const void* input_ptr,
                                                               const unsigned int* flag,
                                                               void* value_ptr,
                                                               long long stride) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned int position = flag[0];
    if (position != kNoInvalid) {
      *static_cast<T*>(value_ptr) = static_cast<const T*>(input_ptr)[position * stride];
    }
  }
}

}  // namespace

extern "C" __global__ void t4a_cuda_init_validation(unsigned int* flag, void* value, int dtype) {
  switch (dtype) {
    case T4A_CUDA_F32:
      init_validation_body<float>(flag, value);
      break;
    case T4A_CUDA_F64:
      init_validation_body<double>(flag, value);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_validate_float_indices(const void* indices,
                                                           unsigned int* flag, int dtype,
                                                           double max_exact_integer, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      validate_float_indices_body<float>(indices, flag, static_cast<float>(max_exact_integer),
                                         len);
      break;
    case T4A_CUDA_F64:
      validate_float_indices_body<double>(indices, flag, max_exact_integer, len);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_extract_invalid_float_index(const void* indices,
                                                                const unsigned int* flag,
                                                                void* value, int dtype) {
  switch (dtype) {
    case T4A_CUDA_F32:
      extract_invalid_float_index_body<float>(indices, flag, value);
      break;
    case T4A_CUDA_F64:
      extract_invalid_float_index_body<double>(indices, flag, value);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_validate_real_cast(const void* input, unsigned int* flag,
                                                       int dtype, double min_value,
                                                       double max_value, long long stride,
                                                       int max_inclusive, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      validate_real_cast_body<float>(input, flag, static_cast<float>(min_value),
                                     static_cast<float>(max_value), stride, max_inclusive, len);
      break;
    case T4A_CUDA_F64:
      validate_real_cast_body<double>(input, flag, min_value, max_value, stride, max_inclusive,
                                      len);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_extract_invalid_real_cast(const void* input,
                                                              const unsigned int* flag,
                                                              void* value, int dtype,
                                                              long long stride) {
  switch (dtype) {
    case T4A_CUDA_F32:
      extract_invalid_real_cast_body<float>(input, flag, value, stride);
      break;
    case T4A_CUDA_F64:
      extract_invalid_real_cast_body<double>(input, flag, value, stride);
      break;
    default:
      break;
  }
}
