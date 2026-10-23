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

// Narrow form of the generic signed-stride decoder: the quotient walk runs in
// 32-bit unsigned arithmetic while the coordinate*stride products and the
// offset stay 64-bit, which is the baseline's `usize` index with `i64` strides
// (`structural.rs:materialize_strided_kernel`).
//
// Precondition, proved by the facade before the launch: `flat < len` and
// `len <= INT32_MAX`; `len` is the layout's logical count, i.e. the extent
// product, so every extent is at most `len` and every quotient fits 32 bits. The
// address arithmetic is unchanged, so a negative stride or a large offset still
// lands in the same 64-bit element index.
__device__ __forceinline__ long long t4a_cuda_decode_narrow(unsigned int flat, int rank,
                                                           const long long* extents,
                                                           const long long* strides,
                                                           long long offset) {
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis >= rank) {
      break;
    }
    const unsigned int extent = static_cast<unsigned int>(extents[axis]);
    const unsigned int coordinate = flat % extent;
    flat /= extent;
    offset += static_cast<long long>(coordinate) * strides[axis];
  }
  return offset;
}

// Decode `flat` with the narrow quotient walk when the launch proved it fits,
// and with the wide decoder otherwise. The decision is launch-uniform.
__device__ __forceinline__ long long t4a_cuda_decode_indexed(long long flat, bool narrow,
                                                             int rank,
                                                             const long long* extents,
                                                             const long long* strides,
                                                             long long offset) {
  if (narrow) {
    return t4a_cuda_decode_narrow(static_cast<unsigned int>(flat), rank, extents, strides,
                                  offset);
  }
  return t4a_cuda_decode(flat, rank, extents, strides, offset);
}

// Whether the narrow decoder may run. `len` is the layout's logical count, i.e.
// the product of its non-negative extents, so `len <= INT32_MAX` bounds every
// extent as well, and the quotient and coordinate of each axis fit 32 bits before
// anything is cast; strides, offsets and the element addresses stay 64-bit in the
// narrow decoder, so a signed stride or a large offset is unaffected. Testing every
// extent explicitly instead costs 13 % on the per-element materialize case, and it
// cannot differ for a layout whose count is the extent product.
__device__ __forceinline__ bool t4a_cuda_decode_fits_narrow(long long len) {
  return len >= 0 && len <= 0x7fffffffLL;
}

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
                                                    long long len, bool narrow) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_indexed(flat, narrow, layout.rank, layout.extents,
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
                                                         long long len,
                                                         bool narrow) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_indexed(flat, narrow, layout.rank, layout.extents,
                                                    layout.strides, layout.offset);
    static_cast<T*>(dst)[flat] = static_cast<const T*>(src)[index];
  }
}

template <typename T>
__device__ __forceinline__ void contiguous_to_view_body(void* dst,
                                                        const void* src,
                                                        const t4a_cuda_layout& layout,
                                                        long long len,
                                                        bool narrow) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_indexed(flat, narrow, layout.rank, layout.extents,
                                                    layout.strides, layout.offset);
    static_cast<T*>(dst)[index] = static_cast<const T*>(src)[flat];
  }
}

template <typename T>
__device__ __forceinline__ void strided_to_strided_body(void* dst,
                                                        const void* src,
                                                        const t4a_cuda_layout& src_layout,
                                                        const t4a_cuda_layout& dst_layout,
                                                        long long len,
                                                        bool narrow) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    // Both layouts share the source's extents; only the strides and offsets
    // differ.
    const long long src_index =
        t4a_cuda_decode_indexed(flat, narrow, src_layout.rank, src_layout.extents,
                                src_layout.strides, src_layout.offset);
    const long long dst_index =
        t4a_cuda_decode_indexed(flat, narrow, src_layout.rank, src_layout.extents,
                                dst_layout.strides, dst_layout.offset);
    static_cast<T*>(dst)[dst_index] = static_cast<const T*>(src)[src_index];
  }
}

// --- Revision 9 general path ---------------------------------------------------
//
// These bodies take the pointer-geometry layout and walk the operand's own
// runtime rank, so any rank is representable. They keep no per-axis array and no
// masked select, and they do not use the fast path's narrow quotient walk, which
// is an optimisation for the ranks that dominate, not a contract.

template <typename T>
__device__ __forceinline__ void fill_zero_view_body_any(void* out,
                                                        const t4a_cuda_layout_any& layout,
                                                        long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_any(flat, layout.rank, layout.extents,
                                                layout.strides, layout.offset);
    static_cast<T*>(out)[index] = t4a_cuda_zero<T>();
  }
}

template <typename T>
__device__ __forceinline__ void materialize_strided_body_any(void* dst,
                                                             const void* src,
                                                             const t4a_cuda_layout_any& layout,
                                                             long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_any(flat, layout.rank, layout.extents,
                                                layout.strides, layout.offset);
    static_cast<T*>(dst)[flat] = static_cast<const T*>(src)[index];
  }
}

template <typename T>
__device__ __forceinline__ void contiguous_to_view_body_any(
    void* dst, const void* src, const t4a_cuda_layout_any& layout, long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    const long long index = t4a_cuda_decode_any(flat, layout.rank, layout.extents,
                                                layout.strides, layout.offset);
    static_cast<T*>(dst)[index] = static_cast<const T*>(src)[flat];
  }
}

template <typename T>
__device__ __forceinline__ void strided_to_strided_body_any(
    void* dst, const void* src, const t4a_cuda_layout_any& src_layout,
    const t4a_cuda_layout_any& dst_layout, long long len) {
  long long flat;
  if (t4a_cuda_tail_index(&flat, len)) {
    // Both layouts share the source's extents; only the strides and offsets
    // differ.
    const long long src_index = t4a_cuda_decode_any(flat, src_layout.rank, src_layout.extents,
                                                    src_layout.strides, src_layout.offset);
    const long long dst_index = t4a_cuda_decode_any(flat, src_layout.rank, src_layout.extents,
                                                    dst_layout.strides, dst_layout.offset);
    static_cast<T*>(dst)[dst_index] = static_cast<const T*>(src)[src_index];
  }
}

// Shared-memory tiled transpose of a compact row-major (M, N) matrix per batch
// into a compact destination at offset zero: dst[b*S + i + j*M] =
// src[src_offset + b*S + i*N + j], S = M*N. The destination tiles advance
// along x so neighbouring blocks stream the destination. Extents and offsets
// stay runtime scalars; the tile configuration is selected by a uniform switch
// so ptxas sees the baseline's compile-time tile, pitch and vector width and
// can unroll the lane and row loops.
template <int kTile, int kBlockRows, int kPadding, int kVectorWidth, int kRowUnroll,
          typename Index, typename T>
__device__ __forceinline__ void tiled_transpose_tile(void* dst_ptr,
                                                     const void* src_ptr,
                                                     Index src_offset,
                                                     Index batch_stride,
                                                     Index dst_fast_extent,
                                                     Index src_fast_extent) {
  extern __shared__ unsigned char t4a_cuda_transpose_shared[];
  T* shared = reinterpret_cast<T*>(t4a_cuda_transpose_shared);
  const T* src = static_cast<const T*>(src_ptr);
  T* dst = static_cast<T*>(dst_ptr);

  constexpr int pitch = kTile + kPadding;
  const int unit_x = static_cast<int>(threadIdx.x);
  const int unit_y = static_cast<int>(threadIdx.y);

  const Index tile_dst_fast = static_cast<Index>(blockIdx.x) * kTile;
  const Index tile_src_fast = static_cast<Index>(blockIdx.y) * kTile;
  const Index batch_base = static_cast<Index>(blockIdx.z) * batch_stride;

#pragma unroll kRowUnroll
  for (int row = unit_y; row < kTile; row += kBlockRows) {
    const Index dst_fast = tile_dst_fast + row;
    for (int lane = 0; lane < kVectorWidth; ++lane) {
      const int local_src_fast = unit_x * kVectorWidth + lane;
      const Index src_fast = tile_src_fast + local_src_fast;
      if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {
        const Index src_index =
            src_offset + batch_base + dst_fast * src_fast_extent + src_fast;
        shared[row * pitch + local_src_fast] = src[src_index];
      }
    }
  }

  __syncthreads();

#pragma unroll kRowUnroll
  for (int row = unit_y; row < kTile; row += kBlockRows) {
    const Index src_fast = tile_src_fast + row;
    for (int lane = 0; lane < kVectorWidth; ++lane) {
      const int local_dst_fast = unit_x * kVectorWidth + lane;
      const Index dst_fast = tile_dst_fast + local_dst_fast;
      if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {
        const Index dst_index = batch_base + dst_fast + src_fast * dst_fast_extent;
        dst[dst_index] = shared[local_dst_fast * pitch + row];
      }
    }
  }
}

// Every configuration accepted by `TransposeConfig::new` (the facade rejects
// anything else before planning) selects a compiled tile variant. An unknown
// configuration performs no work, as with an unknown dtype or op tag.
template <typename Index, typename T>
__device__ __forceinline__ void tiled_transpose_config(void* dst, const void* src,
                                                       Index src_offset, Index batch_stride,
                                                       Index dst_fast_extent,
                                                       Index src_fast_extent, int tile,
                                                       int block_rows, int padding,
                                                       int vector_width) {
#define T4A_TRANSPOSE_CONFIG(TILE, ROWS, PAD, WIDTH)                                    \
  case WIDTH:                                                                          \
    tiled_transpose_tile<TILE, ROWS, PAD, WIDTH, (TILE == 32 ? 4 : 2), Index, T>(       \
        dst, src, src_offset, batch_stride, dst_fast_extent, src_fast_extent);         \
    return;
// Each nested switch returns on every path (its `default` arm included), so no
// configuration falls through to a neighbouring body.
#define T4A_TRANSPOSE_WIDTHS(TILE, ROWS, PAD)   \
  switch (vector_width) {                       \
    T4A_TRANSPOSE_CONFIG(TILE, ROWS, PAD, 1)    \
    T4A_TRANSPOSE_CONFIG(TILE, ROWS, PAD, 2)    \
    T4A_TRANSPOSE_CONFIG(TILE, ROWS, PAD, 4)    \
    default:                                    \
      return;                                   \
  }
#define T4A_TRANSPOSE_PADDING(TILE, ROWS)   \
  switch (padding) {                        \
    case 0:                                 \
      T4A_TRANSPOSE_WIDTHS(TILE, ROWS, 0)   \
      return;                               \
    case 1:                                 \
      T4A_TRANSPOSE_WIDTHS(TILE, ROWS, 1)   \
      return;                               \
    default:                                \
      return;                               \
  }
#define T4A_TRANSPOSE_ROWS(TILE)       \
  switch (block_rows) {                \
    case 4:                            \
      T4A_TRANSPOSE_PADDING(TILE, 4)   \
      return;                          \
    case 8:                            \
      T4A_TRANSPOSE_PADDING(TILE, 8)   \
      return;                          \
    case 16:                           \
      T4A_TRANSPOSE_PADDING(TILE, 16)  \
      return;                          \
    default:                           \
      return;                          \
  }
  switch (tile) {
    case 8:
      T4A_TRANSPOSE_ROWS(8)
      return;
    case 16:
      T4A_TRANSPOSE_ROWS(16)
      return;
    case 32:
      T4A_TRANSPOSE_ROWS(32)
      return;
    default:
      return;
  }
#undef T4A_TRANSPOSE_ROWS
#undef T4A_TRANSPOSE_PADDING
#undef T4A_TRANSPOSE_WIDTHS
#undef T4A_TRANSPOSE_CONFIG
}

// Wide (64-bit) fallback for a source/destination span above 2^31 elements. It
// keeps the runtime configuration generic so the common narrow launch does not
// pay the wide body's register footprint. `__noinline__` keeps ptxas from
// merging that footprint into the narrow entry.
__device__ __noinline__ void tiled_transpose_wide(void* dst_ptr, const void* src_ptr,
                                                 int dtype, long long src_offset,
                                                 long long batch_stride,
                                                 long long dst_fast_extent,
                                                 long long src_fast_extent,
                                                 t4a_cuda_transpose_config config) {
  extern __shared__ unsigned char t4a_cuda_transpose_shared[];
  const int tile = config.tile;
  const int pitch = tile + config.padding;
  const int unit_x = static_cast<int>(threadIdx.x);
  const int unit_y = static_cast<int>(threadIdx.y);
  const int vector_width = config.vector_width;
  const int block_rows = config.block_rows;
  const long long tile_dst_fast = static_cast<long long>(blockIdx.x) * tile;
  const long long tile_src_fast = static_cast<long long>(blockIdx.y) * tile;
  const long long batch_base = static_cast<long long>(blockIdx.z) * batch_stride;

#define T4A_WIDE_BODY(T)                                                               \
  do {                                                                                 \
    T* shared = reinterpret_cast<T*>(t4a_cuda_transpose_shared);                       \
    const T* src = static_cast<const T*>(src_ptr);                                     \
    T* dst = static_cast<T*>(dst_ptr);                                                 \
    for (int row = unit_y; row < tile; row += block_rows) {                             \
      const long long dst_fast = tile_dst_fast + row;                                  \
      for (int lane = 0; lane < vector_width; ++lane) {                                \
        const int local_src_fast = unit_x * vector_width + lane;                       \
        const long long src_fast = tile_src_fast + local_src_fast;                     \
        if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {                \
          const long long src_index =                                                  \
              src_offset + batch_base + dst_fast * src_fast_extent + src_fast;         \
          shared[row * pitch + local_src_fast] = src[src_index];                       \
        }                                                                              \
      }                                                                                \
    }                                                                                  \
    __syncthreads();                                                                   \
    for (int row = unit_y; row < tile; row += block_rows) {                             \
      const long long src_fast = tile_src_fast + row;                                  \
      for (int lane = 0; lane < vector_width; ++lane) {                                \
        const int local_dst_fast = unit_x * vector_width + lane;                       \
        const long long dst_fast = tile_dst_fast + local_dst_fast;                     \
        if (dst_fast < dst_fast_extent && src_fast < src_fast_extent) {                \
          const long long dst_index = batch_base + dst_fast + src_fast * dst_fast_extent; \
          dst[dst_index] = shared[local_dst_fast * pitch + row];                       \
        }                                                                              \
      }                                                                                \
    }                                                                                  \
  } while (0)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_WIDE_BODY)
#undef T4A_WIDE_BODY
}

}  // namespace

extern "C" __global__ void t4a_cuda_fill_zero(void* out, int dtype, long long len) {
#define T4A_BODY(T) fill_zero_body<T>(out, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_fill_zero_view(void* out, int dtype,
                                                   t4a_cuda_layout layout, long long len) {
  const bool narrow = t4a_cuda_decode_fits_narrow(len);
#define T4A_BODY(T) fill_zero_view_body<T>(out, layout, len, narrow)
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
  const bool narrow = t4a_cuda_decode_fits_narrow(len);
#define T4A_BODY(T) materialize_strided_body<T>(dst, src, src_layout, len, narrow)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_contiguous_to_view(void* dst, const void* src, int dtype,
                                                       t4a_cuda_layout dst_layout,
                                                       long long len) {
  const bool narrow = t4a_cuda_decode_fits_narrow(len);
#define T4A_BODY(T) contiguous_to_view_body<T>(dst, src, dst_layout, len, narrow)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_strided_to_strided(void* dst, const void* src, int dtype,
                                                       t4a_cuda_layout src_layout,
                                                       t4a_cuda_layout dst_layout,
                                                       long long len) {
  // Both layouts share the source's extents, so the source layout carries the
  // quotient walk's range proof.
  const bool narrow = t4a_cuda_decode_fits_narrow(len);
#define T4A_BODY(T) strided_to_strided_body<T>(dst, src, src_layout, dst_layout, len, narrow)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

// The single ABI symbol serves every tile configuration and dtype. Cap the
// register file at the 32-register occupancy point for the largest legal block
// (32x16 = 512 threads): without this cap ptxas spends 40 registers, which costs
// the default 16x8 tile 8-12 % of bandwidth on the A100. The executed narrow
// branch keeps that footprint without spilling; the wide fallback is a separate
// device function.
extern "C" __global__ void __launch_bounds__(512, 4) t4a_cuda_tiled_transpose(
    void* dst, const void* src, int dtype, long long src_offset, long long batch_stride,
    long long dst_fast_extent, long long src_fast_extent, t4a_cuda_transpose_config config) {
  // The narrow body declares every one of its `Index` values as a signed 32-bit
  // int, so each argument that is narrowed must itself fit that range, not only
  // the last reachable address: an extent of 2^31 with the other extent one has a
  // last index of 2^31-1 while its own value is out of range, and the casts below
  // would then produce a negative bound and skip every store.
  //
  // All four arguments are non-negative here, so OR-ing them yields their maximum
  // and one comparison bounds the whole set; it also runs before the span products
  // so a hostile argument cannot overflow them. With
  // `batch_stride == dst_fast_extent * src_fast_extent` (the facade proves it),
  // non-negative arguments and the total span below 2^31, every intermediate of
  // `src_offset + batch_base + dst_fast * src_fast_extent + src_fast` and of
  // `batch_base + dst_fast + src_fast * dst_fast_extent` is a partial sum of
  // non-negative terms that the total bounds, so the int arithmetic cannot wrap.
  const long long narrowed = src_offset | batch_stride | dst_fast_extent | src_fast_extent;
  bool narrow = narrowed >= 0 && narrowed <= 0x7fffffffLL;
  if (narrow) {
    // Safe: all four factors are below 2^31, so both products stay below 2^62.
    const long long matrix = dst_fast_extent * src_fast_extent;
    const long long last_batch = static_cast<long long>(gridDim.z) - 1;
    const long long last_index = last_batch * batch_stride + matrix - 1;
    narrow = last_index <= 0x7fffffffLL && src_offset + last_index <= 0x7fffffffLL;
  }
  if (narrow) {
#define T4A_BODY(T)                                                                      \
    tiled_transpose_config<int, T>(dst, src, static_cast<int>(src_offset),               \
                                   static_cast<int>(batch_stride),                       \
                                   static_cast<int>(dst_fast_extent),                    \
                                   static_cast<int>(src_fast_extent), config.tile,        \
                                   config.block_rows, config.padding, config.vector_width)
    T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
  } else {
    tiled_transpose_wide(dst, src, dtype, src_offset, batch_stride, dst_fast_extent,
                         src_fast_extent, config);
  }
}

extern "C" __global__ void t4a_cuda_fill_zero_view_any(void* out, int dtype,
                                                       t4a_cuda_layout_any layout,
                                                       long long len) {
#define T4A_BODY(T) fill_zero_view_body_any<T>(out, layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_materialize_strided_any(void* dst, const void* src,
                                                            int dtype,
                                                            t4a_cuda_layout_any src_layout,
                                                            long long len) {
#define T4A_BODY(T) materialize_strided_body_any<T>(dst, src, src_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_contiguous_to_view_any(void* dst, const void* src,
                                                           int dtype,
                                                           t4a_cuda_layout_any dst_layout,
                                                           long long len) {
#define T4A_BODY(T) contiguous_to_view_body_any<T>(dst, src, dst_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}

extern "C" __global__ void t4a_cuda_strided_to_strided_any(void* dst, const void* src,
                                                           int dtype,
                                                           t4a_cuda_layout_any src_layout,
                                                           t4a_cuda_layout_any dst_layout,
                                                           long long len) {
#define T4A_BODY(T) strided_to_strided_body_any<T>(dst, src, src_layout, dst_layout, len)
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_BODY)
#undef T4A_BODY
}
