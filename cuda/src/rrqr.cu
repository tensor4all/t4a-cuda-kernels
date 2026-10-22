// t4a-cuda-kernels: rank-revealing QR primitives.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs:
// initialize_permutation (L24), column_norms_real (L32), column_norms_c32 (L79),
// column_norms_c64 (L128), select_pivot (L177), swap_columns (L232),
// reflector_real (L265), reflector_c32 (L335), reflector_c64 (L394),
// c32_from_real/c64_from_real (L325, L330), apply_reflector_real (L453),
// apply_reflector_c32 (L499), apply_reflector_c64 (L547), extract_r (L609),
// prefix_rank_real (L630), prefix_rank_c32 (L676) and prefix_rank_c64 (L720).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// initialize_q (L595) is served by revision 7's t4a_cuda_identity_shifted with
// start = 0. Every reduction entry runs one 32-lane plane per block
// (block = (32,1,1)), matching the baseline's PLANE_DIM and the caller's
// RRQR_PLANE_WIDTH; flat entries use block = (256,1,1).
//
// INVARIANT: a lane that owns no rows contributes its zero accumulator and never
// returns before a shuffle, so every collective runs for the whole plane. The
// reflector broadcasts x0 from lane 0 before any thread stores over it.

#include <t4a/abi.cuh>

namespace {

constexpr int kPlane = 32;
constexpr unsigned int kWarpMask = 0xFFFFFFFFu;
constexpr long long kNone = 0x7fffffffffffffffLL;

// Magnitude of a payload element; the norms/rank entries write the real type, so
// these are overloads rather than one template with a changing return type.
__device__ __forceinline__ float magnitude(float value) {
  return value < 0.0f ? -value : value;
}

__device__ __forceinline__ double magnitude(double value) {
  return value < 0.0 ? -value : value;
}

__device__ __forceinline__ float magnitude(t4a_cuda_c32 value) {
  return hypotf(value.re, value.im);
}

__device__ __forceinline__ double magnitude(t4a_cuda_c64 value) {
  return hypot(value.re, value.im);
}

// Ascending-XOR plane collectives over the full warp.
template <typename T>
__device__ __forceinline__ T plane_sum(T value) {
  for (int offset = 1; offset < kPlane; offset <<= 1) {
    value += t4a_cuda_shfl_xor(value, offset, kWarpMask);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T plane_max(T value) {
  for (int offset = 1; offset < kPlane; offset <<= 1) {
    const T other = t4a_cuda_shfl_xor(value, offset, kWarpMask);
    value = other > value ? other : value;
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T plane_min(T value) {
  for (int offset = 1; offset < kPlane; offset <<= 1) {
    const T other = t4a_cuda_shfl_xor(value, offset, kWarpMask);
    value = other < value ? other : value;
  }
  return value;
}

// Any lane true.
__device__ __forceinline__ bool plane_any(bool value) {
  return plane_sum(value ? 1 : 0) > 0;
}

// Element and batch offsets of the one-axis-shorter operand (axis 0 is the column
// or step index, axes 1.. are the matrix's axes 2..) for a linear batch index.
__device__ __forceinline__ long long shorter_batch_offset(long long batch, t4a_cuda_layout layout) {
  long long offset = 0;
  long long remaining = batch;
  for (int axis = 1; axis < layout.rank; ++axis) {
    const long long extent = layout.extents[axis];
    const long long coordinate = remaining % extent;
    remaining /= extent;
    offset += coordinate * layout.strides[axis];
  }
  return offset;
}

// Batch offset of a full matrix operand (axes 2..) for a linear batch index.
__device__ __forceinline__ long long matrix_batch_offset(long long batch, t4a_cuda_layout layout) {
  long long offset = 0;
  long long remaining = batch;
  for (int axis = 2; axis < layout.rank; ++axis) {
    const long long extent = layout.extents[axis];
    const long long coordinate = remaining % extent;
    remaining /= extent;
    offset += coordinate * layout.strides[axis];
  }
  return offset;
}

template <typename T>
__device__ __forceinline__ T complex_div_ordinary(T numerator, T denominator) {
  return numerator / denominator;
}

template <>
__device__ __forceinline__ t4a_cuda_c32 complex_div_ordinary<t4a_cuda_c32>(t4a_cuda_c32 numerator,
                                                                         t4a_cuda_c32 denominator) {
  const float norm = denominator.re * denominator.re + denominator.im * denominator.im;
  t4a_cuda_c32 result = {(numerator.re * denominator.re + numerator.im * denominator.im) / norm,
                         (numerator.im * denominator.re - numerator.re * denominator.im) / norm};
  return result;
}

template <>
__device__ __forceinline__ t4a_cuda_c64 complex_div_ordinary<t4a_cuda_c64>(t4a_cuda_c64 numerator,
                                                                         t4a_cuda_c64 denominator) {
  const double norm = denominator.re * denominator.re + denominator.im * denominator.im;
  t4a_cuda_c64 result = {(numerator.re * denominator.re + numerator.im * denominator.im) / norm,
                         (numerator.im * denominator.re - numerator.re * denominator.im) / norm};
  return result;
}

// --- column norms -----------------------------------------------------------

// One block per (batch, column): two passes over the magnitudes (plane maximum,
// then the scaled squares) so no intermediate overflows.
template <typename T, typename F>
__device__ __forceinline__ void column_norms_body(const void* work_ptr, void* norms_ptr,
                                                  t4a_cuda_layout work, t4a_cuda_layout norms,
                                                  long long step) {
  const long long rows = work.extents[0];
  const long long columns = work.extents[1];
  const long long trailing = columns - step;
  const long long task = blockIdx.x;
  const long long batch = task / trailing;
  const long long column = step + task % trailing;
  const long long lane = threadIdx.x;

  const T* work_values = static_cast<const T*>(work_ptr);
  const long long work_base = work.offset + column * work.strides[1] +
                              matrix_batch_offset(batch, work);
  const long long row_stride = work.strides[0];

  F local_max = F(0);
  bool invalid = false;
  for (long long row = step + lane; row < rows; row += kPlane) {
    const F value = magnitude(work_values[work_base + row * row_stride]);
    if (value - value != F(0)) {
      invalid = true;
    } else if (value > local_max) {
      local_max = value;
    }
  }
  const F scale = plane_max(local_max);
  F local_sum = F(0);
  if (scale > F(0)) {
    for (long long row = step + lane; row < rows; row += kPlane) {
      const F scaled =
          magnitude(work_values[work_base + row * row_stride]) / scale;
      local_sum += scaled * scaled;
    }
  }
  const F norm = scale * sqrt(plane_sum(local_sum));
  // A computed norm that is itself nonfinite cannot order columns, so it joins the
  // nonfinite-magnitude case in the negative sentinel.
  const bool bad = plane_any(invalid) || norm - norm != F(0);
  if (lane == 0) {
    F* norms_values = static_cast<F*>(norms_ptr);
    const long long norms_index = norms.offset + column * norms.strides[0] +
                                  shorter_batch_offset(batch, norms);
    norms_values[norms_index] = bad ? F(-1) : norm;
  }
}

// --- pivot selection --------------------------------------------------------

// One block per batch. Lanes scan columns, the plane combines the local best with
// the baseline's deterministic tie rule: largest norm, then smallest original
// column, then smallest column index.
template <typename F>
__device__ __forceinline__ void select_pivot_body(const void* norms_ptr,
                                                  const void* permutation_ptr, void* pivots_ptr,
                                                  void* status_ptr, t4a_cuda_layout norms,
                                                  t4a_cuda_layout permutation, long long step) {
  const long long batch = blockIdx.x;
  const long long columns = norms.extents[0];
  const long long lane = threadIdx.x;
  const F* norms_values = static_cast<const F*>(norms_ptr);
  const long long* permutation_values = static_cast<const long long*>(permutation_ptr);
  const long long norms_base = norms.offset + shorter_batch_offset(batch, norms);
  const long long permutation_base =
      permutation.offset + shorter_batch_offset(batch, permutation);

  F local_max = F(0);
  long long local_original = kNone;
  long long local_column = kNone;
  bool invalid = false;
  for (long long column = step + lane; column < columns; column += kPlane) {
    const F norm = norms_values[norms_base + column * norms.strides[0]];
    const long long original = permutation_values[permutation_base + column * permutation.strides[0]];
    if (norm < F(0)) {
      invalid = true;
    } else if (norm > local_max || (norm == local_max && original < local_original)) {
      local_max = norm;
      local_original = original;
      local_column = column;
    }
  }
  const F maximum = plane_max(local_max);
  const long long original =
      plane_min(local_max == maximum ? local_original : kNone);
  const long long selected = plane_min(
      local_max == maximum && local_original == original ? local_column : kNone);
  const bool any_invalid = plane_any(invalid);
  if (lane == 0) {
    static_cast<long long*>(pivots_ptr)[batch] = selected == kNone ? step : selected;
    if (any_invalid) {
      static_cast<long long*>(status_ptr)[batch] = 1;
    }
  }
}

// --- column swap ------------------------------------------------------------

// Flat over rows * batches. The row-0 thread also exchanges the permutation entries.
template <typename T>
__device__ __forceinline__ void swap_columns_body(void* work_ptr, void* permutation_ptr,
                                                  const void* pivots_ptr, t4a_cuda_layout work,
                                                  t4a_cuda_layout permutation, long long step) {
  const long long rows = work.extents[0];
  const long long batches = 1;
  long long batch_count = batches;
  for (int axis = 2; axis < work.rank; ++axis) {
    batch_count *= work.extents[axis];
  }
  const long long pos = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pos >= rows * batch_count) {
    return;
  }
  const long long row = pos % rows;
  const long long batch = pos / rows;
  const long long pivot = static_cast<const long long*>(pivots_ptr)[batch];
  if (pivot == step) {
    return;
  }
  T* work_values = static_cast<T*>(work_ptr);
  const long long work_base = work.offset + row * work.strides[0] +
                              matrix_batch_offset(batch, work);
  const long long left = work_base + step * work.strides[1];
  const long long right = work_base + pivot * work.strides[1];
  const T value = work_values[left];
  work_values[left] = work_values[right];
  work_values[right] = value;
  if (row == 0) {
    long long* permutation_values = static_cast<long long*>(permutation_ptr);
    const long long permutation_base =
        permutation.offset + shorter_batch_offset(batch, permutation);
    const long long first = permutation_base + step * permutation.strides[0];
    const long long second = permutation_base + pivot * permutation.strides[0];
    const long long stored = permutation_values[first];
    permutation_values[first] = permutation_values[second];
    permutation_values[second] = stored;
  }
}

// --- Householder reflector --------------------------------------------------

// One block per batch: build the reflector for column `step` in place. x0 is
// broadcast from lane 0 before any store overwrites its slot.
template <typename T, typename F>
__device__ __forceinline__ F reflector_norm(const T* work_values, long long column_base,
                                            long long row_stride, long long step, long long rows,
                                            long long lane) {
  F local_max = F(0);
  for (long long row = step + lane; row < rows; row += kPlane) {
    const F value = magnitude(work_values[column_base + row * row_stride]);
    if (value > local_max) {
      local_max = value;
    }
  }
  const F scale = plane_max(local_max);
  F local_sum = F(0);
  if (scale > F(0)) {
    for (long long row = step + lane; row < rows; row += kPlane) {
      const F scaled = magnitude(work_values[column_base + row * row_stride]) / scale;
      local_sum += scaled * scaled;
    }
  }
  return scale * sqrt(plane_sum(local_sum));
}

template <typename F>
__device__ __forceinline__ void reflector_real_body(F* work_values, F* coeff_values,
                                                    long long column_base, long long row_stride,
                                                    long long coeff_index, long long step,
                                                    long long rows, long long lane) {
  const F norm = reflector_norm<F, F>(work_values, column_base, row_stride, step, rows, lane);
  F x0 = lane == 0 ? work_values[column_base + step * row_stride] : F(0);
  x0 = __shfl_sync(kWarpMask, x0, 0);
  if (norm == F(0)) {
    if (lane == 0) {
      coeff_values[coeff_index] = F(0);
    }
    return;
  }
  const F phase = x0 < F(0) ? F(-1) : F(1);
  const F alpha = -phase * norm;
  const F denominator = x0 - alpha;
  for (long long row = step + 1 + lane; row < rows; row += kPlane) {
    work_values[column_base + row * row_stride] /= denominator;
  }
  if (lane == 0) {
    work_values[column_base + step * row_stride] = alpha;
    coeff_values[coeff_index] = F(1) + magnitude(x0) / norm;
  }
}

template <typename C, typename F>
__device__ __forceinline__ void reflector_complex_body(C* work_values, C* coeff_values,
                                                       long long column_base, long long row_stride,
                                                       long long coeff_index, long long step,
                                                       long long rows, long long lane) {
  const F norm = reflector_norm<C, F>(work_values, column_base, row_stride, step, rows, lane);
  C x0;
  x0.re = lane == 0 ? work_values[column_base + step * row_stride].re : F(0);
  x0.im = lane == 0 ? work_values[column_base + step * row_stride].im : F(0);
  x0.re = __shfl_sync(kWarpMask, x0.re, 0);
  x0.im = __shfl_sync(kWarpMask, x0.im, 0);
  if (norm == F(0)) {
    if (lane == 0) {
      C zero = {F(0), F(0)};
      coeff_values[coeff_index] = zero;
    }
    return;
  }
  const F x0_abs = magnitude(x0);
  C phase;
  if (x0_abs == F(0)) {
    phase.re = F(1);
    phase.im = F(0);
  } else {
    phase.re = x0.re / x0_abs;
    phase.im = x0.im / x0_abs;
  }
  C alpha;
  alpha.re = -phase.re * norm;
  alpha.im = -phase.im * norm;
  C denominator;
  denominator.re = x0.re - alpha.re;
  denominator.im = x0.im - alpha.im;
  for (long long row = step + 1 + lane; row < rows; row += kPlane) {
    work_values[column_base + row * row_stride] =
        complex_div_ordinary<C>(work_values[column_base + row * row_stride], denominator);
  }
  if (lane == 0) {
    work_values[column_base + step * row_stride] = alpha;
    C coefficient = {F(1) + x0_abs / norm, F(0)};
    coeff_values[coeff_index] = coefficient;
  }
}

// --- reflector application --------------------------------------------------

// One block per (batch, column): dot product over the reflector's rows, then the
// rank-one update.
template <typename F>
__device__ __forceinline__ void apply_reflector_real_body(
    const F* packed_values, F* target_values, const F* coeff_values, long long packed_base,
    long long target_base, long long coefficient, long long step, long long rows,
    long long packed_row_stride, long long target_row_stride, long long lane) {
  F dot = F(0);
  for (long long row = step + lane; row < rows; row += kPlane) {
    const F vector = row == step ? F(1) : packed_values[packed_base + row * packed_row_stride];
    dot += vector * target_values[target_base + row * target_row_stride];
  }
  // `coefficient` is the slot index in `coeff_values`, not the value.
  dot = plane_sum(dot) * coeff_values[coefficient];
  for (long long row = step + lane; row < rows; row += kPlane) {
    const F vector = row == step ? F(1) : packed_values[packed_base + row * packed_row_stride];
    target_values[target_base + row * target_row_stride] -= vector * dot;
  }
}

template <typename C, typename F>
__device__ __forceinline__ void apply_reflector_complex_body(
    const C* packed_values, C* target_values, const C* coeff_values, long long packed_base,
    long long target_base, long long coefficient, long long step, long long rows,
    long long packed_row_stride, long long target_row_stride, long long lane) {
  F dot_re = F(0);
  F dot_im = F(0);
  for (long long row = step + lane; row < rows; row += kPlane) {
    C vector;
    if (row == step) {
      vector.re = F(1);
      vector.im = F(0);
    } else {
      vector = packed_values[packed_base + row * packed_row_stride];
    }
    const C value = target_values[target_base + row * target_row_stride];
    // conj(vector) * value
    dot_re += vector.re * value.re + vector.im * value.im;
    dot_im += vector.re * value.im - vector.im * value.re;
  }
  const C coefficient_value = coeff_values[coefficient];
  // (plane_sum(re) + i plane_sum(im)) * coeff, the baseline's complex product; the
  // reflector itself stores a real coefficient, but a caller-supplied one need not be.
  const F summed_re = plane_sum(dot_re);
  const F summed_im = plane_sum(dot_im);
  C dot;
  dot.re = summed_re * coefficient_value.re - summed_im * coefficient_value.im;
  dot.im = summed_re * coefficient_value.im + summed_im * coefficient_value.re;
  for (long long row = step + lane; row < rows; row += kPlane) {
    C vector;
    if (row == step) {
      vector.re = F(1);
      vector.im = F(0);
    } else {
      vector = packed_values[packed_base + row * packed_row_stride];
    }
    const long long index = target_base + row * target_row_stride;
    const C value = target_values[index];
    const C product = t4a_cuda_mul<C>(vector, dot);
    C updated;
    updated.re = value.re - product.re;
    updated.im = value.im - product.im;
    target_values[index] = updated;
  }
}

// --- R extraction -----------------------------------------------------------

template <typename T>
__device__ __forceinline__ void extract_r_body(const void* work_ptr, void* r_ptr,
                                               t4a_cuda_layout work, t4a_cuda_layout r,
                                               long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long row = flat % r.extents[0];
  const long long column = (flat / r.extents[0]) % r.extents[1];
  const long long batch = flat / (r.extents[0] * r.extents[1]);
  const long long r_index = r.offset + row * r.strides[0] + column * r.strides[1] +
                            matrix_batch_offset(batch, r);
  T value = t4a_cuda_zero<T>();
  if (row <= column) {
    value = static_cast<const T*>(work_ptr)[work.offset + row * work.strides[0] +
                                            column * work.strides[1] +
                                            matrix_batch_offset(batch, work)];
  }
  static_cast<T*>(r_ptr)[r_index] = value;
}

// --- prefix rank ------------------------------------------------------------

template <typename T, typename F>
__device__ __forceinline__ void prefix_rank_body(const void* r_ptr, void* rank_ptr,
                                                 void* status_ptr, t4a_cuda_layout r, F rtol,
                                                 F atol) {
  const long long batch = blockIdx.x;
  const long long k = r.extents[0];
  const long long lane = threadIdx.x;
  const T* values = static_cast<const T*>(r_ptr);
  const long long base = r.offset + matrix_batch_offset(batch, r);
  const F leading = magnitude(values[base]);
  const F relative = rtol * leading;
  // The baseline's comparison, not fmax: a NaN tolerance propagates the same way.
  const F threshold = atol > relative ? atol : relative;
  long long first_failure = kNone;
  bool invalid = false;
  for (long long diagonal = lane; diagonal < k; diagonal += kPlane) {
    const F value = magnitude(
        values[base + diagonal * r.strides[0] + diagonal * r.strides[1]]);
    if (value - value != F(0)) {
      invalid = true;
    }
    if (!(value > threshold) && diagonal < first_failure) {
      first_failure = diagonal;
    }
  }
  const long long failure = plane_min(first_failure);
  const bool any_invalid = plane_any(invalid);
  if (lane == 0) {
    static_cast<long long*>(rank_ptr)[batch] = failure == kNone ? k : failure;
    if (any_invalid) {
      static_cast<long long*>(status_ptr)[batch] = 1;
    }
  }
}

}  // namespace

extern "C" __global__ void t4a_cuda_initialize_permutation(void* permutation, long long n,
                                                           long long len) {
  long long pos;
  if (!t4a_cuda_tail_index(&pos, len)) {
    return;
  }
  static_cast<long long*>(permutation)[pos] = pos % n;
}

#define T4A_COLUMN_NORMS_CASE(T, F) \
  column_norms_body<T, F>(work, norms, work_layout, norms_layout, step)
extern "C" __global__ void t4a_cuda_column_norms(const void* work, void* norms, int dtype,
                                                 t4a_cuda_layout work_layout,
                                                 t4a_cuda_layout norms_layout, long long step) {
  switch (dtype) {
    case T4A_CUDA_F32:
      T4A_COLUMN_NORMS_CASE(float, float);
      break;
    case T4A_CUDA_F64:
      T4A_COLUMN_NORMS_CASE(double, double);
      break;
    case T4A_CUDA_C32:
      T4A_COLUMN_NORMS_CASE(t4a_cuda_c32, float);
      break;
    case T4A_CUDA_C64:
      T4A_COLUMN_NORMS_CASE(t4a_cuda_c64, double);
      break;
    default:
      break;
  }
}
#undef T4A_COLUMN_NORMS_CASE

extern "C" __global__ void t4a_cuda_select_pivot(const void* norms, const void* permutation,
                                                 void* pivots, void* status, int dtype,
                                                 t4a_cuda_layout norms_layout,
                                                 t4a_cuda_layout permutation_layout,
                                                 long long step) {
  switch (dtype) {
    case T4A_CUDA_F32:
      select_pivot_body<float>(norms, permutation, pivots, status, norms_layout,
                               permutation_layout, step);
      break;
    case T4A_CUDA_F64:
      select_pivot_body<double>(norms, permutation, pivots, status, norms_layout,
                                permutation_layout, step);
      break;
    default:
      break;
  }
}

#define T4A_SWAP_COLUMNS_CASE(T) \
  swap_columns_body<T>(work, permutation, pivots, work_layout, permutation_layout, step)
extern "C" __global__ void t4a_cuda_swap_columns(void* work, void* permutation,
                                                 const void* pivots, int dtype,
                                                 t4a_cuda_layout work_layout,
                                                 t4a_cuda_layout permutation_layout,
                                                 long long step) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_SWAP_COLUMNS_CASE);
}
#undef T4A_SWAP_COLUMNS_CASE

extern "C" __global__ void t4a_cuda_reflector(void* work, void* coeff, int dtype,
                                              t4a_cuda_layout work_layout,
                                              t4a_cuda_layout coeff_layout, long long step) {
  const long long batch = blockIdx.x;
  const long long column_base =
      work_layout.offset + step * work_layout.strides[1] + matrix_batch_offset(batch, work_layout);
  switch (dtype) {
    case T4A_CUDA_F32: {
      const long long coeff_index = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      reflector_real_body<float>(static_cast<float*>(work), static_cast<float*>(coeff),
                                 column_base, work_layout.strides[0], coeff_index, step,
                                 work_layout.extents[0], threadIdx.x);
      break;
    }
    case T4A_CUDA_F64: {
      const long long coeff_index = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      reflector_real_body<double>(static_cast<double*>(work), static_cast<double*>(coeff),
                                  column_base, work_layout.strides[0], coeff_index, step,
                                  work_layout.extents[0], threadIdx.x);
      break;
    }
    case T4A_CUDA_C32: {
      const long long coeff_index = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      reflector_complex_body<t4a_cuda_c32, float>(
          static_cast<t4a_cuda_c32*>(work), static_cast<t4a_cuda_c32*>(coeff), column_base,
          work_layout.strides[0], coeff_index, step, work_layout.extents[0], threadIdx.x);
      break;
    }
    case T4A_CUDA_C64: {
      const long long coeff_index = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      reflector_complex_body<t4a_cuda_c64, double>(
          static_cast<t4a_cuda_c64*>(work), static_cast<t4a_cuda_c64*>(coeff), column_base,
          work_layout.strides[0], coeff_index, step, work_layout.extents[0], threadIdx.x);
      break;
    }
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_apply_reflector(
    const void* packed, void* target, const void* coeff, int dtype,
    t4a_cuda_layout packed_layout, t4a_cuda_layout target_layout, t4a_cuda_layout coeff_layout,
    long long step, long long first_column, long long column_count) {
  const long long task = blockIdx.x;
  const long long batch = task / column_count;
  const long long column = first_column + task % column_count;
  const long long packed_base =
      packed_layout.offset + step * packed_layout.strides[1] + matrix_batch_offset(batch, packed_layout);
  const long long target_base =
      target_layout.offset + column * target_layout.strides[1] + matrix_batch_offset(batch, target_layout);
  switch (dtype) {
    case T4A_CUDA_F32: {
      const long long coefficient = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      apply_reflector_real_body<float>(
          static_cast<const float*>(packed), static_cast<float*>(target),
          static_cast<const float*>(coeff), packed_base, target_base, coefficient, step,
          packed_layout.extents[0], packed_layout.strides[0], target_layout.strides[0],
          threadIdx.x);
      break;
    }
    case T4A_CUDA_F64: {
      const long long coefficient = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      apply_reflector_real_body<double>(
          static_cast<const double*>(packed), static_cast<double*>(target),
          static_cast<const double*>(coeff), packed_base, target_base, coefficient, step,
          packed_layout.extents[0], packed_layout.strides[0], target_layout.strides[0],
          threadIdx.x);
      break;
    }
    case T4A_CUDA_C32: {
      const long long coefficient = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      apply_reflector_complex_body<t4a_cuda_c32, float>(
          static_cast<const t4a_cuda_c32*>(packed), static_cast<t4a_cuda_c32*>(target),
          static_cast<const t4a_cuda_c32*>(coeff), packed_base, target_base, coefficient, step,
          packed_layout.extents[0], packed_layout.strides[0], target_layout.strides[0],
          threadIdx.x);
      break;
    }
    case T4A_CUDA_C64: {
      const long long coefficient = coeff_layout.offset + step * coeff_layout.strides[0] +
                                    shorter_batch_offset(batch, coeff_layout);
      apply_reflector_complex_body<t4a_cuda_c64, double>(
          static_cast<const t4a_cuda_c64*>(packed), static_cast<t4a_cuda_c64*>(target),
          static_cast<const t4a_cuda_c64*>(coeff), packed_base, target_base, coefficient, step,
          packed_layout.extents[0], packed_layout.strides[0], target_layout.strides[0],
          threadIdx.x);
      break;
    }
    default:
      break;
  }
}

#define T4A_EXTRACT_R_CASE(T) extract_r_body<T>(work, r, work_layout, r_layout, t4a_cuda_logical_count(r_layout))
extern "C" __global__ void t4a_cuda_extract_r(const void* work, void* r, int dtype,
                                              t4a_cuda_layout work_layout, t4a_cuda_layout r_layout) {
  T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, T4A_EXTRACT_R_CASE);
}
#undef T4A_EXTRACT_R_CASE

extern "C" __global__ void t4a_cuda_prefix_rank(const void* r, void* rank, void* status, int dtype,
                                                t4a_cuda_layout r_layout, double rtol,
                                                double atol) {
  switch (dtype) {
    case T4A_CUDA_F32:
      prefix_rank_body<float, float>(r, rank, status, r_layout, static_cast<float>(rtol),
                                     static_cast<float>(atol));
      break;
    case T4A_CUDA_F64:
      prefix_rank_body<double, double>(r, rank, status, r_layout, rtol, atol);
      break;
    case T4A_CUDA_C32:
      prefix_rank_body<t4a_cuda_c32, float>(r, rank, status, r_layout, static_cast<float>(rtol),
                                            static_cast<float>(atol));
      break;
    case T4A_CUDA_C64:
      prefix_rank_body<t4a_cuda_c64, double>(r, rank, status, r_layout, rtol, atol);
      break;
    default:
      break;
  }
}
