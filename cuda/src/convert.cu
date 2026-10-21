// t4a-cuda-kernels: dtype conversions.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/structural.rs: convert_float_to_float (L351),
// convert_numeric (L358), convert_numeric_to_bool (L365), convert_bool_to_numeric
// (L376), convert_numeric_to_complex_raw (L383), convert_bool_to_complex_raw
// (L391), convert_complex_to_numeric (L399), convert_complex_raw_to_bool (L406),
// convert_c32_to_f32 .. convert_f64_to_c64_raw (L462-L521),
// convert_complex_to_complex (L530) and convert_complex_raw (L540).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// One entry covers the whole pair matrix: the (dst_dtype, src_dtype) tags select
// a templated body, so no mode flag reaches the element loop. Floating
// component -> signed integer routes require the revision-2 validator first
// (C++ static_cast is undefined for NaN, infinity and out-of-range values).

#include <t4a/abi.cuh>

namespace {

// Identity cast between two representations.
template <typename Out, typename In>
__device__ __forceinline__ void convert_cast_body(void* out_ptr, const void* in_ptr,
                                                  long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  static_cast<Out*>(out_ptr)[index] = static_cast<Out>(static_cast<const In*>(in_ptr)[index]);
}

// Complex-to-complex: both components cast independently.
template <typename Out, typename In>
__device__ __forceinline__ void convert_complex_body(void* out_ptr, const void* in_ptr,
                                                     long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const In value = static_cast<const In*>(in_ptr)[index];
  Out converted;
  converted.re = static_cast<decltype(converted.re)>(value.re);
  converted.im = static_cast<decltype(converted.im)>(value.im);
  static_cast<Out*>(out_ptr)[index] = converted;
}

// Real (or Bool) source into complex: the imaginary part is exactly zero.
template <typename C, typename In>
__device__ __forceinline__ void convert_to_complex_body(void* out_ptr, const void* in_ptr,
                                                        long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  C value;
  value.re = static_cast<decltype(value.re)>(static_cast<const In*>(in_ptr)[index]);
  value.im = static_cast<decltype(value.im)>(0);
  static_cast<C*>(out_ptr)[index] = value;
}

// Complex source into a real type: the real part, never the magnitude.
template <typename Out, typename C>
__device__ __forceinline__ void convert_from_complex_body(void* out_ptr, const void* in_ptr,
                                                          long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const C value = static_cast<const C*>(in_ptr)[index];
  static_cast<Out*>(out_ptr)[index] = static_cast<Out>(value.re);
}

// Complex source into Bool: both components zero means false.
template <typename C>
__device__ __forceinline__ void convert_complex_to_bool_body(void* out_ptr, const void* in_ptr,
                                                             long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const C value = static_cast<const C*>(in_ptr)[index];
  static_cast<unsigned char*>(out_ptr)[index] =
      (value.re != static_cast<decltype(value.re)>(0) ||
       value.im != static_cast<decltype(value.im)>(0))
          ? 1u
          : 0u;
}

// Numeric source into Bool: zero means false (NaN is not zero).
template <typename In>
__device__ __forceinline__ void convert_to_bool_body(void* out_ptr, const void* in_ptr,
                                                     long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const In value = static_cast<const In*>(in_ptr)[index];
  static_cast<unsigned char*>(out_ptr)[index] = value != static_cast<In>(0) ? 1u : 0u;
}

}  // namespace

#define T4A_CONVERT_CASE(DST_TAG, SRC_TAG, ...) \
  if (dst_dtype == DST_TAG && src_dtype == SRC_TAG) { \
    __VA_ARGS__;                                     \
    return;                                          \
  }

// Numeric destination: casts from the four numeric types and Bool, plus the
// complex real part.
#define T4A_CONVERT_NUMERIC_DST(TAG, CTYPE)                                          \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_F32,                                     \
                   convert_cast_body<CTYPE, float>(out, in, len))                     \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_F64,                                     \
                   convert_cast_body<CTYPE, double>(out, in, len))                    \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_I32,                                     \
                   convert_cast_body<CTYPE, int>(out, in, len))                       \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_I64,                                     \
                   convert_cast_body<CTYPE, long long>(out, in, len))                 \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_BOOL,                                    \
                   convert_cast_body<CTYPE, unsigned char>(out, in, len))             \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_C32,                                     \
                   convert_from_complex_body<CTYPE, t4a_cuda_c32>(out, in, len))      \
  T4A_CONVERT_CASE(T4A_CUDA_##TAG, T4A_CUDA_C64,                                     \
                   convert_from_complex_body<CTYPE, t4a_cuda_c64>(out, in, len))

extern "C" __global__ void t4a_cuda_convert(void* out, const void* in, int dst_dtype,
                                            int src_dtype, long long len) {
  // Complex destinations: real/Bool sources write a zero imaginary part, and the
  // complex pair casts both components.
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_F32,
                   convert_to_complex_body<t4a_cuda_c32, float>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_F64,
                   convert_to_complex_body<t4a_cuda_c32, double>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_I32,
                   convert_to_complex_body<t4a_cuda_c32, int>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_I64,
                   convert_to_complex_body<t4a_cuda_c32, long long>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_BOOL,
                   convert_to_complex_body<t4a_cuda_c32, unsigned char>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_F32,
                   convert_to_complex_body<t4a_cuda_c64, float>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_F64,
                   convert_to_complex_body<t4a_cuda_c64, double>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_I32,
                   convert_to_complex_body<t4a_cuda_c64, int>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_I64,
                   convert_to_complex_body<t4a_cuda_c64, long long>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_BOOL,
                   convert_to_complex_body<t4a_cuda_c64, unsigned char>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C32, T4A_CUDA_C64,
                   convert_complex_body<t4a_cuda_c32, t4a_cuda_c64>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_C64, T4A_CUDA_C32,
                   convert_complex_body<t4a_cuda_c64, t4a_cuda_c32>(out, in, len))

  // Bool destination: a predicate, not a cast.
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_F32, convert_to_bool_body<float>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_F64, convert_to_bool_body<double>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_I32, convert_to_bool_body<int>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_I64,
                   convert_to_bool_body<long long>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_C32,
                   convert_complex_to_bool_body<t4a_cuda_c32>(out, in, len))
  T4A_CONVERT_CASE(T4A_CUDA_BOOL, T4A_CUDA_C64,
                   convert_complex_to_bool_body<t4a_cuda_c64>(out, in, len))

  // Numeric destinations: casts, plus the complex real part below.
  T4A_CONVERT_NUMERIC_DST(F32, float)
  T4A_CONVERT_NUMERIC_DST(F64, double)
  T4A_CONVERT_NUMERIC_DST(I32, int)
  T4A_CONVERT_NUMERIC_DST(I64, long long)
}
