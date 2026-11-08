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
// it, unlike the others). `kRegroup` says whether pass 1 may combine a lane's
// elements in an order other than the serial lane-strided one (ABI revision 12,
// "Reassociation"); `PROD` may not.
struct SumFold {
  static constexpr bool kRegroup = true;
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
  static constexpr bool kRegroup = false;
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
  static constexpr bool kRegroup = true;
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
  static constexpr bool kRegroup = true;
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
  static constexpr bool kRegroup = true;
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

// A block wider than one warp: the xor tree is warp-local (a full mask does not
// communicate across warps), so each warp's result goes through static shared memory and
// warp 0 folds at most `t4a_cuda_max_warps` of them.
//
// Only the combine-pass kernels may widen, so only they take the wide-capable fold below;
// pass 1 calls `t4a_cuda_plane_fold_narrow` directly. Folding both through the wide-capable
// version measured 12.5% slower on a strided pass-1 case that never widens.
constexpr int t4a_cuda_warp_width = 32;
constexpr int t4a_cuda_max_warps = 16;

template <typename Fold, typename T>
__device__ __noinline__ T t4a_cuda_plane_fold_wide(T value, int plane_width, Fold fold);

// One-warp and sub-warp fold: the xor tree alone, no shared memory. Pass 1 always folds a
// block of `plane_width` (8/16/32) since only the combine pass may widen, so it keeps this
// fold and its generated code stays what it was.
template <typename Fold, typename T>
__device__ __forceinline__ T t4a_cuda_plane_fold_narrow(T value, int plane_width, Fold fold) {
  const unsigned int mask = t4a_cuda_plane_mask(plane_width);
  // The plane combines lane accumulators, so it uses the fold's combine step:
  // sum-of-squares must not square an accumulator again.
  for (int offset = 1; offset < plane_width; offset <<= 1) {
    value = fold.combine(value, t4a_cuda_shfl_xor(value, offset, mask));
  }
  return value;
}

// Fold for the entries the *combine* pass runs (`t4a_cuda_reduce` and its general-rank
// twin): the only place a wide block can be planned, so the wide step lives here and not in
// the pass-1 kernels.
template <typename Fold, typename T>
__device__ __forceinline__ T t4a_cuda_plane_fold(T value, int plane_width, Fold fold) {
  if (plane_width > t4a_cuda_warp_width) {
    return t4a_cuda_plane_fold_wide(value, plane_width, fold);
  }
  return t4a_cuda_plane_fold_narrow(value, plane_width, fold);
}

template <typename Fold, typename T>
__device__ __noinline__ T t4a_cuda_plane_fold_wide(T value, int plane_width, Fold fold) {
  const unsigned int warp_mask = t4a_cuda_plane_mask(t4a_cuda_warp_width);
  for (int offset = 1; offset < t4a_cuda_warp_width; offset <<= 1) {
    value = fold.combine(value, t4a_cuda_shfl_xor(value, offset, warp_mask));
  }
  __shared__ T warp_results[t4a_cuda_max_warps];
  const int lane = threadIdx.x & (t4a_cuda_warp_width - 1);
  const int warp = threadIdx.x / t4a_cuda_warp_width;
  const int warps = plane_width / t4a_cuda_warp_width;
  if (lane == 0) {
    warp_results[warp] = value;
  }
  __syncthreads();
  if (warp != 0) {
    return value;
  }
  // Lanes 0..warps load the initialized entries; the mask names only those lanes.
  T combined = lane < warps ? warp_results[lane] : value;
  const unsigned int warps_mask = t4a_cuda_plane_mask(warps);
  for (int offset = 1; offset < warps; offset <<= 1) {
    combined = fold.combine(combined, t4a_cuda_shfl_xor(combined, offset, warps_mask));
  }
  return combined;
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

// --- Pass-1 chunk walks --------------------------------------------------------
//
// Both pass-1 bodies (`partials_body` and its general-rank twin) take the same two
// walks, so the twins cannot drift. The scalar walk is the baseline's: lane `t` seeds
// from `start + t` and folds `start + t + k*W`. The vector walk serves a contiguous
// chunk of a regroupable fold (`kRegroup`) over a 4- or 8-byte real or integer type.

// The scalar walk: one lane-strided element per step and one loop-carried accumulator.
template <typename Fold, typename T>
__device__ __forceinline__ T t4a_cuda_partials_walk_scalar(const T* in,
                                                           long long in_base,
                                                           long long axis_stride,
                                                           long long start,
                                                           long long end,
                                                           int plane_width,
                                                           Fold fold) {
  long long reduce_index = start + threadIdx.x;
  T acc = fold.seed(in[in_base + reduce_index * axis_stride]);
  for (reduce_index += plane_width; reduce_index < end; reduce_index += plane_width) {
    acc = fold(acc, in[in_base + reduce_index * axis_stride]);
  }
  return acc;
}

// Elements per 16-byte vector for the element types the vector walk serves. Zero keeps
// the scalar walk: the complex types are only component-aligned (`t4a_cuda_c32` is
// 4-byte aligned, not a `float2`), and no reduction uses a narrower type.
template <typename T>
struct t4a_cuda_vector_lanes {
  static constexpr int value = 0;
};
template <>
struct t4a_cuda_vector_lanes<float> {
  static constexpr int value = 4;
};
template <>
struct t4a_cuda_vector_lanes<double> {
  static constexpr int value = 2;
};
template <>
struct t4a_cuda_vector_lanes<int> {
  static constexpr int value = 4;
};
template <>
struct t4a_cuda_vector_lanes<long long> {
  static constexpr int value = 2;
};

template <typename T, int V>
struct alignas(16) t4a_cuda_vector {
  T lane[V];
};

// Vectors a lane loads before it folds any of them. The loads of one batch are
// independent, so a fold whose result selects between operands (the NaN-propagating
// extrema) no longer serializes them behind one DRAM latency each.
constexpr int t4a_cuda_vector_batch = 4;
static_assert(t4a_cuda_vector_batch == 4, "the whole-batch fold below is a fixed four-way tree");

template <typename Fold, typename T>
__host__ __device__ constexpr bool t4a_cuda_vector_walk_enabled() {
  return Fold::kRegroup && t4a_cuda_vector_lanes<T>::value != 0;
}

// Whether a contiguous chunk of `len` elements takes the vector walk. It must hold at least
// one whole group of `V` elements per lane to seed from; the planner's
// `chunk_len >= 8 * plane_width` already guarantees that, and this check keeps it true
// independently of the planner. The values are uniform across the block, so every lane takes
// the same walk.
template <typename T>
__device__ __forceinline__ bool t4a_cuda_vector_walk_applies(long long len, int plane_width) {
  constexpr int V = t4a_cuda_vector_lanes<T>::value;
  return len / V >= plane_width;
}

// Group `group` of the chunk: elements `group*V .. group*V + V`. The groups are fixed by
// the logical position in the chunk, never by the address, so the fold topology is the same
// wherever the input sits; the address only decides whether a group is one 16-byte load
// (`Aligned`: `chunk` is 16-byte aligned) or `V` element loads.
template <typename T, int V, bool Aligned>
__device__ __forceinline__ t4a_cuda_vector<T, V> t4a_cuda_load_group(const T* chunk,
                                                                     long long group) {
  if constexpr (Aligned) {
    return reinterpret_cast<const t4a_cuda_vector<T, V>*>(chunk)[group];
  } else {
    t4a_cuda_vector<T, V> x;
#pragma unroll
    for (int c = 0; c < V; ++c) {
      x.lane[c] = chunk[group * V + c];
    }
    return x;
  }
}

// One vector's contribution: a pairwise tree of its seeded elements, so a sum of
// squares squares each element exactly once.
template <typename Fold, typename T, int V>
__device__ __forceinline__ T t4a_cuda_vector_value(const t4a_cuda_vector<T, V>& x, Fold fold) {
  if constexpr (V == 4) {
    return fold.combine(fold.combine(fold.seed(x.lane[0]), fold.seed(x.lane[1])),
                        fold.combine(fold.seed(x.lane[2]), fold.seed(x.lane[3])));
  } else {
    return fold.combine(fold.seed(x.lane[0]), fold.seed(x.lane[1]));
  }
}

// The vector walk over `chunk[0..len)`. With `groups = len / V` whole groups of `V`
// elements counted from the chunk start, and `tail = len - groups * V < V` elements after
// them:
//   - lane `t` folds groups `t, t + W, t + 2W, ...` (one warp instruction reads `16*W`
//     contiguous bytes when the chunk is 16-byte aligned), in batches of
//     `t4a_cuda_vector_batch` loads issued before any of them is folded (the first and last
//     batches predicated per group);
//   - lane `t < tail` folds tail element `groups * V + t`.
// Every element of the chunk is folded exactly once, and which lane folds it, in which order,
// depends only on `len`, `W`, the element type and the operation. A separate, non-inlined
// function so the scalar walk of the same kernel keeps its own code (a grouped walk inlined
// into pass 1 slowed the strided path even when it never ran;
// slice-reduction-throughput.md §9.2).
template <typename Fold, typename T, bool Aligned>
__device__ __noinline__ T t4a_cuda_partials_walk_vector(const T* chunk,
                                                        long long len,
                                                        int plane_width,
                                                        Fold fold) {
  constexpr int V = t4a_cuda_vector_lanes<T>::value;
  constexpr int B = t4a_cuda_vector_batch;
  using Vector = t4a_cuda_vector<T, V>;
  const int lane = static_cast<int>(threadIdx.x);
  // INVARIANT: `groups >= plane_width` (t4a_cuda_vector_walk_applies), so group `lane`
  // exists; with `Aligned`, `chunk` is 16-byte aligned (t4a_cuda_partials_walk).
  const long long groups = len / V;
  const long long stride = plane_width;

  // The first batch: group `lane` always exists, and the batch's other loads are predicated
  // on their own index, so a chunk with only a few groups per lane still issues its loads
  // together.
  Vector batch[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    if (b == 0 || lane + b * stride < groups) {
      batch[b] = t4a_cuda_load_group<T, V, Aligned>(chunk, lane + b * stride);
    }
  }
  T acc = t4a_cuda_vector_value<Fold, T, V>(batch[0], fold);
#pragma unroll
  for (int b = 1; b < B; ++b) {
    if (lane + b * stride < groups) {
      acc = fold.combine(acc, t4a_cuda_vector_value<Fold, T, V>(batch[b], fold));
    }
  }
  // Whole batches.
  long long index = lane + B * stride;
  for (; index + (B - 1) * stride < groups; index += B * stride) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      batch[b] = t4a_cuda_load_group<T, V, Aligned>(chunk, index + b * stride);
    }
    const T low = fold.combine(t4a_cuda_vector_value<Fold, T, V>(batch[0], fold),
                               t4a_cuda_vector_value<Fold, T, V>(batch[1], fold));
    const T high = fold.combine(t4a_cuda_vector_value<Fold, T, V>(batch[2], fold),
                                t4a_cuda_vector_value<Fold, T, V>(batch[3], fold));
    acc = fold.combine(acc, fold.combine(low, high));
  }
  // The last, partial batch (fewer than `B` groups left for this lane), predicated the same
  // way.
#pragma unroll
  for (int b = 0; b < B - 1; ++b) {
    if (index + b * stride < groups) {
      batch[b] = t4a_cuda_load_group<T, V, Aligned>(chunk, index + b * stride);
    }
  }
#pragma unroll
  for (int b = 0; b < B - 1; ++b) {
    if (index + b * stride < groups) {
      acc = fold.combine(acc, t4a_cuda_vector_value<Fold, T, V>(batch[b], fold));
    }
  }
  const long long tail_start = groups * V;
  if (lane < len - tail_start) {
    acc = fold(acc, chunk[tail_start + lane]);
  }
  return acc;
}

// One lane's fold over its share of the chunk `[start, end)` of the axis at `in_base`.
template <typename Fold, typename T>
__device__ __forceinline__ T t4a_cuda_partials_walk(const T* in,
                                                    long long in_base,
                                                    long long axis_stride,
                                                    long long start,
                                                    long long end,
                                                    int plane_width,
                                                    Fold fold) {
  if constexpr (t4a_cuda_vector_walk_enabled<Fold, T>()) {
    // The stride comes first: `in + in_base + start` is an address inside the input only on
    // a contiguous axis (a broadcast or reversed axis can put it outside the allocation, and
    // forming such a pointer is undefined behaviour).
    if (axis_stride == 1 && t4a_cuda_vector_walk_applies<T>(end - start, plane_width)) {
      const T* chunk = in + in_base + start;
      // INVARIANT: element pointers are aligned to their dtype (c-kernel-abi.md, "Types");
      // the 16-byte test only selects the load instruction, never the topology.
      if ((reinterpret_cast<unsigned long long>(chunk) & 15ull) == 0) {
        return t4a_cuda_partials_walk_vector<Fold, T, true>(chunk, end - start, plane_width,
                                                            fold);
      }
      return t4a_cuda_partials_walk_vector<Fold, T, false>(chunk, end - start, plane_width,
                                                           fold);
    }
  }
  return t4a_cuda_partials_walk_scalar<Fold, T>(in, in_base, axis_stride, start, end,
                                                plane_width, fold);
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
  const T acc = t4a_cuda_partials_walk<Fold, T>(in, in_base, axis_stride, start, end,
                                                plane_width, fold);
  const T reduced = t4a_cuda_plane_fold_narrow(acc, plane_width, fold);
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
  const T acc = t4a_cuda_partials_walk<Fold, T>(in, in_base, axis_stride, start, end,
                                                plane_width, fold);
  const T reduced = t4a_cuda_plane_fold_narrow(acc, plane_width, fold);
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
