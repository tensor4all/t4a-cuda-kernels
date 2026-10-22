// t4a-cuda-kernels: linalg auxiliary kernels.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-linalg/src/gpu/kernels.rs: fill_one_kernel (L73),
// matrix_adjoint_real (L81), matrix_adjoint_complex (L93),
// householder_explicit_v (L119), householder_q_columns_identity (L135),
// positive_phase_real/c32/c64 (L152-L183), qr_diagonal_offset (L185),
// qr_phase_offset (L201), qr_phase_real/c32/c64 (L217-L251),
// qr_apply_phase_real (L253), qr_apply_phase_complex (L276),
// upper_trapezoidal_violation (L298), householder_from_factors_assemble (L315),
// batch_linear_index (L19), matching_work_offset (L37), lu_extract_outputs (L345),
// lu_parity (L420) and lu_apply_pivots (L442); and the identity initialiser
// initialize_q from crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs
// (L595). Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// Unlike the baseline, which drops the operand offset in its explicit
// stride arithmetic, every entry here honours the layout offset. The baseline's
// complex32_magnitude/complex64_magnitude are not ported: they have no caller and
// revision 6's unary ABS covers them. The rank-revealing QR primitives are a
// later slice.

#include <t4a/abi.cuh>

namespace {

// Phase of a QR diagonal: `-1` for a negative real diagonal and `+1` otherwise, so a
// NaN diagonal yields `+1` because the comparison is false. Complex dtypes take the
// unit direction, with `1` for a zero magnitude.
__device__ __forceinline__ float positive_phase(float diagonal) {
  return diagonal < 0.0f ? -1.0f : 1.0f;
}

__device__ __forceinline__ double positive_phase(double diagonal) {
  return diagonal < 0.0 ? -1.0 : 1.0;
}

__device__ __forceinline__ t4a_cuda_c32 positive_phase(t4a_cuda_c32 diagonal) {
  const float magnitude = hypotf(diagonal.re, diagonal.im);
  if (magnitude == 0.0f) {
    t4a_cuda_c32 one = {1.0f, 0.0f};
    return one;
  }
  t4a_cuda_c32 phase = {diagonal.re / magnitude, diagonal.im / magnitude};
  return phase;
}

__device__ __forceinline__ t4a_cuda_c64 positive_phase(t4a_cuda_c64 diagonal) {
  const double magnitude = hypot(diagonal.re, diagonal.im);
  if (magnitude == 0.0) {
    t4a_cuda_c64 one = {1.0, 0.0};
    return one;
  }
  t4a_cuda_c64 phase = {diagonal.re / magnitude, diagonal.im / magnitude};
  return phase;
}

// Batch linear index over axes 2.. of `layout` for logical position `flat`,
// accumulated in ascending axis order like the baseline's batch_linear_index.
__device__ __forceinline__ long long batch_linear_index(long long flat,
                                                       const t4a_cuda_layout& layout) {
  long long batch = 0;
  long long stride = 1;
  for (int axis = 0; axis < layout.rank; ++axis) {
    const long long extent = layout.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    if (axis >= 2) {
      batch += coordinate * stride;
      stride *= extent;
    }
  }
  return batch;
}

// Element offset contributed by axes 2.. of `out_layout`, applied to the matching
// axes of another layout of the same rank (the baseline's matching_work_offset).
__device__ __forceinline__ long long batch_axis_offset(long long flat,
                                                       const t4a_cuda_layout& out_layout,
                                                       const t4a_cuda_layout& target_layout) {
  long long remaining = flat / (out_layout.extents[0] * out_layout.extents[1]);
  long long offset = 0;
  for (int axis = 2; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = remaining % extent;
    remaining /= extent;
    offset += coordinate * target_layout.strides[axis];
  }
  return offset;
}

// Address of logical element `flat` in `layout`, with the axis walk unrolled over
// literal metadata indices. Handing `extents`/`strides` to `t4a_cuda_decode` makes
// their addresses escape to pointer parameters, and the bodies below walk a second
// layout or a plan array as well, which is what copies the by-value layouts into
// the per-thread stack frame and turns every metadata access into `LDL`/`STL` (see
// the metadata-access note in t4a/abi.cuh).
__device__ __forceinline__ long long linalg_address(long long flat,
                                                   const t4a_cuda_layout& layout) {
  long long offset = layout.offset;
  long long rest = flat;
#pragma unroll
  for (int axis = 0; axis < T4A_CUDA_MAX_RANK; ++axis) {
    if (axis < layout.rank) {
      const long long extent = layout.extents[axis];
      offset += (rest % extent) * layout.strides[axis];
      rest /= extent;
    }
  }
  return offset;
}

// Non-zero test shared by the template bodies; a complex value is non-zero when
// either component is, as the baseline's complex comparison does.
template <typename T>
__device__ __forceinline__ bool nonzero(T value) {
  return value != t4a_cuda_zero<T>();
}

template <>
__device__ __forceinline__ bool nonzero<t4a_cuda_c32>(t4a_cuda_c32 value) {
  return value.re != 0.0f || value.im != 0.0f;
}

template <>
__device__ __forceinline__ bool nonzero<t4a_cuda_c64>(t4a_cuda_c64 value) {
  return value.re != 0.0 || value.im != 0.0;
}

__device__ __forceinline__ long long larger(long long lhs, long long rhs) {
  return lhs > rhs ? lhs : rhs;
}


template <typename T>
__device__ __forceinline__ void fill_one_body(void* out, t4a_cuda_layout layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long index =
      t4a_cuda_decode(flat, layout.rank, layout.extents, layout.strides, layout.offset);
  static_cast<T*>(out)[index] = t4a_cuda_one<T>();
}

template <typename T>
__device__ __forceinline__ void matrix_adjoint_body(void* out, const void* v,
                                                    t4a_cuda_layout out_layout,
                                                    t4a_cuda_layout v_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long out_index = out_layout.offset;
  long long v_index = v_layout.offset;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    out_index += coordinate * out_layout.strides[axis];
    if (axis == 0) {
      v_index += coordinate * v_layout.strides[1];
    } else if (axis == 1) {
      v_index += coordinate * v_layout.strides[0];
    } else {
      v_index += coordinate * v_layout.strides[axis];
    }
  }
  static_cast<T*>(out)[out_index] = t4a_cuda_conj<T>(static_cast<const T*>(v)[v_index]);
}

// Rank-2 explicit Householder vector: below the diagonal from `packed`, the unit
// diagonal and zeros above it.
template <typename T>
__device__ __forceinline__ void explicit_v_body(void* out, const void* packed,
                                                t4a_cuda_layout out_layout,
                                                t4a_cuda_layout packed_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long row = flat % out_layout.extents[0];
  const long long col = (flat / out_layout.extents[0]) % out_layout.extents[1];
  const long long out_index =
      out_layout.offset + row * out_layout.strides[0] + col * out_layout.strides[1];
  T value;
  if (row < col) {
    value = t4a_cuda_zero<T>();
  } else if (row == col) {
    value = t4a_cuda_one<T>();
  } else {
    value = static_cast<const T*>(packed)[packed_layout.offset +
                                           row * packed_layout.strides[0] +
                                           col * packed_layout.strides[1]];
  }
  static_cast<T*>(out)[out_index] = value;
}

template <typename T>
__device__ __forceinline__ void identity_shifted_body(void* out, t4a_cuda_layout layout,
                                                      long long len, long long start) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long row = flat % layout.extents[0];
  const long long col = (flat / layout.extents[0]) % layout.extents[1];
  const long long index =
      t4a_cuda_decode(flat, layout.rank, layout.extents, layout.strides, layout.offset);
  static_cast<T*>(out)[index] = row == start + col ? t4a_cuda_one<T>() : t4a_cuda_zero<T>();
}

// One phase entry per phase position; `r` is the matrix, whose axes 0/1 are the row
// and column while the phase's axes 1.. are the matrix's axes 2...
template <typename T>
__device__ __forceinline__ void qr_phase_body(void* phase, const void* r,
                                              t4a_cuda_layout phase_layout,
                                              t4a_cuda_layout r_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long index = flat % phase_layout.extents[0];
  long long remaining = flat / phase_layout.extents[0];
  long long phase_index = phase_layout.offset + index * phase_layout.strides[0];
  long long r_index =
      r_layout.offset + index * r_layout.strides[0] + index * r_layout.strides[1];
  for (int axis = 1; axis < phase_layout.rank; ++axis) {
    const long long extent = phase_layout.extents[axis];
    const long long coordinate = remaining % extent;
    remaining /= extent;
    phase_index += coordinate * phase_layout.strides[axis];
    r_index += coordinate * r_layout.strides[axis + 1];
  }
  static_cast<T*>(phase)[phase_index] = positive_phase(static_cast<const T*>(r)[r_index]);
}

// Scale the trailing Q columns and the R rows by the phase; the R side is
// conjugated for the complex dtypes, exactly as the baseline does.
template <typename T>
__device__ __forceinline__ void qr_apply_phase_body(void* q, void* r, const void* phase,
                                                    t4a_cuda_layout q_layout,
                                                    t4a_cuda_layout r_layout,
                                                    t4a_cuda_layout phase_layout,
                                                    long long q_start) {
  const long long q_count = t4a_cuda_logical_count(q_layout);
  const long long r_count = t4a_cuda_logical_count(r_layout);
  long long flat;
  if (!t4a_cuda_tail_index(&flat, larger(q_count, r_count))) {
    return;
  }
  if (flat < q_count) {
    const long long column = (flat / q_layout.extents[0]) % q_layout.extents[1];
    const long long index = q_start + column;
    if (index < phase_layout.extents[0]) {
      long long remaining = flat / (q_layout.extents[0] * q_layout.extents[1]);
      const long long q_index = linalg_address(flat, q_layout);
      long long phase_index = phase_layout.offset + index * phase_layout.strides[0];
      for (int axis = 2; axis < q_layout.rank; ++axis) {
        const long long extent = q_layout.extents[axis];
        const long long coordinate = remaining % extent;
        remaining /= extent;
        phase_index += coordinate * phase_layout.strides[axis - 1];
      }
      T* q_ptr = static_cast<T*>(q);
      q_ptr[q_index] = t4a_cuda_mul<T>(q_ptr[q_index], static_cast<const T*>(phase)[phase_index]);
    }
  }
  if (flat < r_count) {
    const long long index = flat % r_layout.extents[0];
    long long remaining = flat / (r_layout.extents[0] * r_layout.extents[1]);
    const long long r_index = linalg_address(flat, r_layout);
    long long phase_index = phase_layout.offset + index * phase_layout.strides[0];
    for (int axis = 2; axis < r_layout.rank; ++axis) {
      const long long extent = r_layout.extents[axis];
      const long long coordinate = remaining % extent;
      remaining /= extent;
      phase_index += coordinate * phase_layout.strides[axis - 1];
    }
    T* r_ptr = static_cast<T*>(r);
    r_ptr[r_index] = t4a_cuda_mul<T>(r_ptr[r_index],
                                    t4a_cuda_conj<T>(static_cast<const T*>(phase)[phase_index]));
  }
}

template <typename T>
__device__ __forceinline__ void violation_body(void* violation, const void* input,
                                                t4a_cuda_layout violation_layout,
                                                t4a_cuda_layout input_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  const long long row = flat % violation_layout.extents[0];
  const long long col = (flat / violation_layout.extents[0]) % violation_layout.extents[1];
  const long long violation_index =
      t4a_cuda_decode(flat, violation_layout.rank, violation_layout.extents,
                      violation_layout.strides, violation_layout.offset);
  const long long input_index = t4a_cuda_decode(flat, input_layout.rank, input_layout.extents,
                                                input_layout.strides, input_layout.offset);
  const T value = static_cast<const T*>(input)[input_index];
  static_cast<int*>(violation)[violation_index] = row > col && nonzero<T>(value) ? 1 : 0;
}

// Rank-2 factor assembly with flat coefficient arrays.
template <typename T>
__device__ __forceinline__ void assemble_body(void* packed_out, void* coeff_out,
                                               const void* packed_q, const void* coeff_q,
                                               const void* folded_r, t4a_cuda_layout out_layout,
                                               long long out_len, t4a_cuda_layout q_layout,
                                               t4a_cuda_layout r_layout, long long coeff_out_len,
                                               long long factor_width) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, larger(out_len, coeff_out_len))) {
    return;
  }
  if (flat < out_len) {
    const long long row = flat % out_layout.extents[0];
    const long long col = (flat / out_layout.extents[0]) % out_layout.extents[1];
    const long long out_index =
        out_layout.offset + row * out_layout.strides[0] + col * out_layout.strides[1];
    T value;
    if (row < factor_width && row <= col) {
      value = static_cast<const T*>(folded_r)[r_layout.offset + row * r_layout.strides[0] +
                                              col * r_layout.strides[1]];
    } else if (col < factor_width && row > col) {
      value = static_cast<const T*>(packed_q)[q_layout.offset + row * q_layout.strides[0] +
                                              col * q_layout.strides[1]];
    } else {
      value = t4a_cuda_zero<T>();
    }
    static_cast<T*>(packed_out)[out_index] = value;
  }
  if (flat < coeff_out_len) {
    static_cast<T*>(coeff_out)[flat] = flat < factor_width
                                           ? static_cast<const T*>(coeff_q)[flat]
                                           : t4a_cuda_zero<T>();
  }
}

// LU outputs: P from the pivot transpositions, L and U from the packed factors, and
// the permutation parity, in one launch whose domain is the largest of the four
// outputs. The parity loop is serial over the `k` steps per batch, with batches in
// parallel threads.
template <typename T>
__device__ __forceinline__ void lu_extract_body(void* p_out, void* l_out, void* u_out,
                                                 void* parity_out, const void* work,
                                                 const void* pivots, t4a_cuda_layout p_layout,
                                                 long long p_len, t4a_cuda_layout l_layout,
                                                 long long l_len, t4a_cuda_layout u_layout,
                                                 long long u_len, t4a_cuda_layout work_layout,
                                                 long long k, long long batches) {
  long long pos;
  if (!t4a_cuda_tail_index(&pos, larger(larger(p_len, l_len), larger(u_len, batches)))) {
    return;
  }
  const int* pivot_values = static_cast<const int*>(pivots);
  if (pos < p_len) {
    const long long row = pos % p_layout.extents[0];
    const long long col = (pos / p_layout.extents[0]) % p_layout.extents[1];
    const long long batch = batch_linear_index(pos, p_layout);
    long long final_row = col;
    for (long long step = 0; step < k; ++step) {
      const long long pivot = static_cast<long long>(pivot_values[step + batch * k]) - 1;
      if (final_row == step) {
        final_row = pivot;
      } else if (final_row == pivot) {
        final_row = step;
      }
    }
    const long long index = linalg_address(pos, p_layout);
    static_cast<T*>(p_out)[index] = final_row == row ? t4a_cuda_one<T>() : t4a_cuda_zero<T>();
  }
  if (pos < l_len) {
    const long long row = pos % l_layout.extents[0];
    const long long col = (pos / l_layout.extents[0]) % l_layout.extents[1];
    const long long index = linalg_address(pos, l_layout);
    T value;
    if (row < col) {
      value = t4a_cuda_zero<T>();
    } else if (row == col) {
      value = t4a_cuda_one<T>();
    } else {
      value = static_cast<const T*>(work)[work_layout.offset + row * work_layout.strides[0] +
                                          col * work_layout.strides[1] +
                                          batch_axis_offset(pos, l_layout, work_layout)];
    }
    static_cast<T*>(l_out)[index] = value;
  }
  if (pos < u_len) {
    const long long row = pos % u_layout.extents[0];
    const long long col = (pos / u_layout.extents[0]) % u_layout.extents[1];
    const long long index = linalg_address(pos, u_layout);
    static_cast<T*>(u_out)[index] =
        row <= col ? static_cast<const T*>(work)[work_layout.offset +
                                                 row * work_layout.strides[0] +
                                                 col * work_layout.strides[1] +
                                                 batch_axis_offset(pos, u_layout, work_layout)]
                   : t4a_cuda_zero<T>();
  }
  if (pos < batches) {
    T sign = t4a_cuda_one<T>();
    for (long long step = 0; step < k; ++step) {
      const long long pivot = static_cast<long long>(pivot_values[step + pos * k]) - 1;
      if (pivot != step) {
        sign = t4a_cuda_neg<T>(sign);
      }
    }
    static_cast<T*>(parity_out)[pos] = sign;
  }
}

template <typename T>
__device__ __forceinline__ void lu_parity_body(void* parity_out, const void* pivots, long long k,
                                               long long len) {
  long long batch;
  if (!t4a_cuda_tail_index(&batch, len)) {
    return;
  }
  const int* pivot_values = static_cast<const int*>(pivots);
  T sign = t4a_cuda_one<T>();
  for (long long step = 0; step < k; ++step) {
    const long long pivot = static_cast<long long>(pivot_values[step + batch * k]) - 1;
    if (pivot != step) {
      sign = t4a_cuda_neg<T>(sign);
    }
  }
  static_cast<T*>(parity_out)[batch] = sign;
}

template <typename T>
__device__ __forceinline__ void lu_apply_pivots_body(void* out, const void* input,
                                                     const void* pivots, t4a_cuda_layout out_layout,
                                                     t4a_cuda_layout in_layout, long long k,
                                                     bool inverse, long long len) {
  long long pos;
  if (!t4a_cuda_tail_index(&pos, len)) {
    return;
  }
  const long long row = pos % out_layout.extents[0];
  const long long col = (pos / out_layout.extents[0]) % out_layout.extents[1];
  const long long batch = batch_linear_index(pos, out_layout);
  const int* pivot_values = static_cast<const int*>(pivots);
  long long source_row = row;
  if (inverse) {
    for (long long step = 0; step < k; ++step) {
      const long long pivot = static_cast<long long>(pivot_values[step + batch * k]) - 1;
      if (source_row == step) {
        source_row = pivot;
      } else if (source_row == pivot) {
        source_row = step;
      }
    }
  } else {
    for (long long step = k; step > 0; --step) {
      const long long index = step - 1;
      const long long pivot = static_cast<long long>(pivot_values[index + batch * k]) - 1;
      if (source_row == index) {
        source_row = pivot;
      } else if (source_row == pivot) {
        source_row = index;
      }
    }
  }
  const long long out_index = linalg_address(pos, out_layout);
  const long long in_index = in_layout.offset + source_row * in_layout.strides[0] +
                             col * in_layout.strides[1] +
                             batch_axis_offset(pos, out_layout, in_layout);
  static_cast<T*>(out)[out_index] = static_cast<const T*>(input)[in_index];
}

}  // namespace

// Every entry takes the operands as layouts and derives its logical bounds from
// their extents; the caller's allocation lengths are a host-side validation input
// (see the ABI document), so an operand whose capacity exceeds its extent is never
// addressed past its logical end.

#define T4A_FILL_ONE_CASE(T) fill_one_body<T>(out, out_layout, t4a_cuda_logical_count(out_layout))
extern "C" __global__ void t4a_cuda_fill_one(void* out, int dtype, t4a_cuda_layout out_layout) {
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_FILL_ONE_CASE);
}
#undef T4A_FILL_ONE_CASE

#define T4A_MATRIX_ADJOINT_CASE(T) \
  matrix_adjoint_body<T>(out, v, out_layout, v_layout, t4a_cuda_logical_count(out_layout))
extern "C" __global__ void t4a_cuda_matrix_adjoint(void* out, const void* v, int dtype,
                                                   t4a_cuda_layout out_layout,
                                                   t4a_cuda_layout v_layout) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_MATRIX_ADJOINT_CASE);
}
#undef T4A_MATRIX_ADJOINT_CASE

#define T4A_EXPLICIT_V_CASE(T) \
  explicit_v_body<T>(out, packed, out_layout, packed_layout, t4a_cuda_logical_count(out_layout))
extern "C" __global__ void t4a_cuda_householder_explicit_v(void* out, const void* packed,
                                                           int dtype,
                                                           t4a_cuda_layout out_layout,
                                                           t4a_cuda_layout packed_layout) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_EXPLICIT_V_CASE);
}
#undef T4A_EXPLICIT_V_CASE

#define T4A_IDENTITY_SHIFTED_CASE(T) \
  identity_shifted_body<T>(out, out_layout, t4a_cuda_logical_count(out_layout), start)
extern "C" __global__ void t4a_cuda_identity_shifted(void* out, int dtype,
                                                     t4a_cuda_layout out_layout,
                                                     long long start) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_IDENTITY_SHIFTED_CASE);
}
#undef T4A_IDENTITY_SHIFTED_CASE

#define T4A_QR_PHASE_CASE(T) \
  qr_phase_body<T>(phase, r, phase_layout, r_layout, t4a_cuda_logical_count(phase_layout))
extern "C" __global__ void t4a_cuda_qr_phase(void* phase, const void* r, int dtype,
                                             t4a_cuda_layout phase_layout,
                                             t4a_cuda_layout r_layout) {
  T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, T4A_QR_PHASE_CASE);
}
#undef T4A_QR_PHASE_CASE

#define T4A_QR_APPLY_PHASE_CASE(T) \
  qr_apply_phase_body<T>(q, r, phase, q_layout, r_layout, phase_layout, q_start)
extern "C" __global__ void t4a_cuda_qr_apply_phase(void* q, void* r, const void* phase, int dtype,
                                                   t4a_cuda_layout q_layout,
                                                   t4a_cuda_layout r_layout,
                                                   t4a_cuda_layout phase_layout,
                                                   long long q_start) {
  T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, T4A_QR_APPLY_PHASE_CASE);
}
#undef T4A_QR_APPLY_PHASE_CASE

#define T4A_VIOLATION_CASE(T) \
  violation_body<T>(violation, input, violation_layout, input_layout, \
                    t4a_cuda_logical_count(violation_layout))
extern "C" __global__ void t4a_cuda_upper_trapezoidal_violation(
    void* violation, const void* input, int dtype, t4a_cuda_layout violation_layout,
    t4a_cuda_layout input_layout) {
  T4A_CUDA_DTYPE_SWITCH(dtype, T4A_VIOLATION_CASE);
}
#undef T4A_VIOLATION_CASE

#define T4A_ASSEMBLE_CASE(T) \
  assemble_body<T>(packed_out, coeff_out, packed_q, coeff_q, folded_r, packed_out_layout, \
                   t4a_cuda_logical_count(packed_out_layout), packed_q_layout, folded_r_layout, \
                   coeff_out_len, factor_width)
extern "C" __global__ void t4a_cuda_householder_assemble(
    void* packed_out, void* coeff_out, const void* packed_q, const void* coeff_q,
    const void* folded_r, int dtype, t4a_cuda_layout packed_out_layout,
    t4a_cuda_layout packed_q_layout, t4a_cuda_layout folded_r_layout, long long coeff_out_len,
    long long factor_width) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_ASSEMBLE_CASE);
}
#undef T4A_ASSEMBLE_CASE

#define T4A_LU_EXTRACT_CASE(T) \
  lu_extract_body<T>(p_out, l_out, u_out, parity_out, work, pivots, p_layout, \
                     t4a_cuda_logical_count(p_layout), l_layout, t4a_cuda_logical_count(l_layout), u_layout, \
                     t4a_cuda_logical_count(u_layout), work_layout, k, batches)
extern "C" __global__ void t4a_cuda_lu_extract_outputs(
    void* p_out, void* l_out, void* u_out, void* parity_out, const void* work, const void* pivots,
    int dtype, t4a_cuda_layout p_layout, t4a_cuda_layout l_layout, t4a_cuda_layout u_layout,
    t4a_cuda_layout work_layout, long long k, long long batches) {
  T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, T4A_LU_EXTRACT_CASE);
}
#undef T4A_LU_EXTRACT_CASE

#define T4A_LU_PARITY_CASE(T) lu_parity_body<T>(parity_out, pivots, k, batches)
extern "C" __global__ void t4a_cuda_lu_parity(void* parity_out, const void* pivots, int dtype,
                                              long long k, long long batches) {
  T4A_CUDA_ALGEBRA_DTYPE_SWITCH(dtype, T4A_LU_PARITY_CASE);
}
#undef T4A_LU_PARITY_CASE

#define T4A_LU_APPLY_PIVOTS_CASE(T) \
  lu_apply_pivots_body<T>(out, input, pivots, out_layout, in_layout, k, inverse != 0, \
                          t4a_cuda_logical_count(out_layout))
extern "C" __global__ void t4a_cuda_lu_apply_pivots(void* out, const void* input,
                                                    const void* pivots, int dtype,
                                                    t4a_cuda_layout out_layout,
                                                    t4a_cuda_layout in_layout, long long k,
                                                    int inverse) {
  T4A_CUDA_NUMERIC_DTYPE_SWITCH(dtype, T4A_LU_APPLY_PIVOTS_CASE);
}
#undef T4A_LU_APPLY_PIVOTS_CASE
