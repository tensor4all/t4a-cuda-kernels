// t4a-cuda-kernels: fill, copy, strided movement and tiled transpose.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/structural.rs: fill_zero_kernel (L27),
// fill_zero_view_kernel (L40), copy_bool_kernel (L63), materialize_strided_kernel
// (L102), tiled_transpose_kernel (L164), contiguous_to_view_kernel (L284),
// strided_to_strided_kernel (L299). Copyright (c) 2026 Hiroshi Shinaoka and
// tenferro-rs contributors; MIT OR Apache-2.0.
//
// The *_comptime_layout_kernel twins at L133/L227 are the WebGPU/Metal-only
// exception and are deliberately not ported.

#include <t4a/abi.cuh>

namespace {

template <typename T>
__device__ __forceinline__ void fill_zero_body(void* out, long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(out)[index] = t4a_cuda_zero<T>();
  }
}

template <typename T>
__device__ __forceinline__ void fill_zero_view_body(void* out,
                                                    const t4a_cuda_layout& layout,
                                                    long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode(flat, layout.rank, layout.extents,
                                            layout.strides, layout.offset);
    static_cast<T*>(out)[index] = t4a_cuda_zero<T>();
  }
}

template <typename T>
__device__ __forceinline__ void copy_exact_body(void* dst, const void* src, long long len) {
  long long index;
  if (t4a_cuda_tail_index(&index, len)) {
    static_cast<T*>(dst)[index] = static_cast<const T*>(src)[index];
  }
}

template <typename T>
__device__ __forceinline__ void materialize_strided_body(void* dst,
                                                         const void* src,
                                                         const t4a_cuda_layout& layout,
                                                         long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode(flat, layout.rank, layout.extents,
                                            layout.strides, layout.offset);
    static_cast<T*>(dst)[flat] = static_cast<const T*>(src)[index];
  }
}

template <typename T>
__device__ __forceinline__ void contiguous_to_view_body(void* dst,
                                                        const void* src,
                                                        const t4a_cuda_layout& layout,
                                                        long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode(flat, layout.rank, layout.extents,
                                            layout.strides, layout.offset);
    static_cast<T*>(dst)[index] = static_cast<const T*>(src)[flat];
  }
}

template <typename T>
__device__ __forceinline__ void strided_to_strided_body(void* dst,
                                                        const void* src,
                                                        const t4a_cuda_layout& src_layout,
                                                        const t4a_cuda_layout& dst_layout,
                                                        long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    // Both layouts share the source's extents; only the strides and offsets
    // differ.
    const long long src_index = t4a_cuda_decode(flat, src_layout.rank, src_layout.extents,
                                                src_layout.strides, src_layout.offset);
    const long long dst_index = t4a_cuda_decode(flat, src_layout.rank, src_layout.extents,
                                                dst_layout.strides, dst_layout.offset);
    static_cast<T*>(dst)[dst_index] = static_cast<const T*>(src)[src_index];
  }
}

// Shared-memory tiled transpose of a compact row-major (M, N) matrix per batch
// into a compact destination at offset zero: dst[b*S + i + j*M] =
// src[src_offset + b*S + i*N + j], S = M*N. The destination tiles advance
// along x so neighbouring blocks stream the destination. Extents, offsets and
// the batch stride are runtime values; only the configuration is selected by
// the caller.
template <typename T>
__device__ __forceinline__ void tiled_transpose_body(void* dst_ptr,
                                                     const void* src_ptr,
                                                     long long src_offset,
                                                     long long batch_stride,
                                                     long long dst_fast_extent,
                                                     long long src_fast_extent,
                                                     const t4a_cuda_transpose_config& config) {
  extern __shared__ unsigned char t4a_cuda_transpose_shared[];
  T* shared = reinterpret_cast<T*>(t4a_cuda_transpose_shared);
  const T* src = static_cast<const T*>(src_ptr);
  T* dst = static_cast<T*>(dst_ptr);

  const int tile = config.tile;
  const int pitch = tile + config.padding;
  const int unit_x = static_cast<int>(threadIdx.x);
  const int unit_y = static_cast<int>(threadIdx.y);
  const int vector_width = config.vector_width;
  const int block_rows = config.block_rows;

  const long long tile_dst_fast = static_cast<long long>(blockIdx.x) * tile;
  const long long tile_src_fast = static_cast<long long>(blockIdx.y) * tile;
  const long long batch_base = static_cast<long long>(blockIdx.z) * batch_stride;

  for (int row = unit_y; row < tile; row += block_rows) {
    const long long dst_fast = tile_dst_fast + row;
    for (int lane = 0; lane < vector_width; ++lane) {
      const int local_src_fast = unit_x * vector_width + lane;
      const long long src_fast = tile_src_fast + local_src_fast;
      if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {
        const long long src_index =
            src_offset + batch_base + dst_fast * src_fast_extent + src_fast;
        shared[row * pitch + local_src_fast] = src[src_index];
      }
    }
  }

  __syncthreads();

  for (int row = unit_y; row < tile; row += block_rows) {
    const long long src_fast = tile_src_fast + row;
    for (int lane = 0; lane < vector_width; ++lane) {
      const int local_dst_fast = unit_x * vector_width + lane;
      const long long dst_fast = tile_dst_fast + local_dst_fast;
      if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {
        const long long dst_index = batch_base + dst_fast + src_fast * dst_fast_extent;
        dst[dst_index] = shared[local_dst_fast * pitch + row];
      }
    }
  }
}

}  // namespace

extern "C" __global__ void t4a_cuda_fill_zero(void* out, int dtype, long long len) {
#define T4A_BODY(T) fill_zero_body<T>(out, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_fill_zero_view(void* out, int dtype,
                                                   t4a_cuda_layout layout, long long len) {
#define T4A_BODY(T) fill_zero_view_body<T>(out, layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_copy_exact(void* dst, const void* src, int dtype,
                                               long long len) {
#define T4A_BODY(T) copy_exact_body<T>(dst, src, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_materialize_strided(void* dst, const void* src, int dtype,
                                                        t4a_cuda_layout src_layout,
                                                        long long len) {
#define T4A_BODY(T) materialize_strided_body<T>(dst, src, src_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_contiguous_to_view(void* dst, const void* src, int dtype,
                                                       t4a_cuda_layout dst_layout,
                                                       long long len) {
#define T4A_BODY(T) contiguous_to_view_body<T>(dst, src, dst_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_strided_to_strided(void* dst, const void* src, int dtype,
                                                       t4a_cuda_layout src_layout,
                                                       t4a_cuda_layout dst_layout,
                                                       long long len) {
#define T4A_BODY(T) strided_to_strided_body<T>(dst, src, src_layout, dst_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_tiled_transpose(void* dst, const void* src, int dtype,
                                                    long long src_offset, long long batch_stride,
                                                    long long dst_fast_extent,
                                                    long long src_fast_extent,
                                                    t4a_cuda_transpose_config config) {
#define T4A_BODY(T) \
  tiled_transpose_body<T>(dst, src, src_offset, batch_stride, dst_fast_extent, src_fast_extent, config)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}
