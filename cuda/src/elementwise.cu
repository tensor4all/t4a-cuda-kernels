// t4a-cuda-kernels: elementwise, comparison and strided-combination kernels.
//
// Translated from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
// crates/tenferro-gpu/src/kernels/elementwise.rs: unary_float_kernel and
// unary_both_kernel (L123, L139), neg_int (L444), exp_float .. erf_float
// (L453-L488), abs_float/abs_int/abs_complex32/abs_complex64 (L488-L516),
// sign_float/sign_int/sign_complex (L523-L560), maximum/minimum
// (L569-L598), pow_float/pow_int_checked/scalar_pow_int_checked (L601-L655),
// scalar_div_float/scalar_pow_float/scalar_rem_float (L254-L262),
// scalar_real_complex_binary (L264), scalar_div_int_checked/scalar_rem_int_checked
// (L332-L366), div_int_checked/rem_float/rem_int_checked (L401-L443),
// compare_float_bool/compare_int_bool (L682-L711),
// select_bool_float/select_bool_int (L721-L741), clamp_float (L743),
// conj_complex (L750), div_complex_parts (L803) with its helpers
// complex_div_abs/complex_div_sign/complex_div_term/complex_div_parts_at
// (L759-L831), broadcast_multiply_kernel (L89-L120) and
// axpby_strided_source_kernel (L166-L199).
// Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
// MIT OR Apache-2.0.
//
// The complex-division helpers in t4a/abi.cuh additionally derive from Julia
// Base `complex.jl` through tenferro-rs; see THIRD_PARTY_NOTICES.md.
//
// The baseline's `structural.rs` broadcast_in_dim_kernel (L327),
// reverse_kernel (L548) and concatenate_copy_kernel (L571) are not entries of
// their own: an explicit layout expresses the same address map through
// t4a_cuda_materialize_strided / t4a_cuda_contiguous_to_view, and
// scale_in_place_* (L74, L92) is `t4a_cuda_binary` with op MUL, rhs scalar.
//
// Every entry is a flat launch over `len` logical elements except
// t4a_cuda_combined_strided and the revision-15 t4a_cuda_strided_{unary,binary,
// compare,select}, which decode each flat index through the destination extents
// and re-address every operand through its own layout.

#include <t4a/abi.cuh>

namespace {

// --- unary ------------------------------------------------------------------

template <typename F>
__device__ __forceinline__ F unary_float(int op, F x) {
  switch (op) {
    case T4A_CUDA_UNARY_NEG:
      return -x;
    case T4A_CUDA_UNARY_EXP:
      return t4a_cuda_exp(x);
    case T4A_CUDA_UNARY_LOG:
      return t4a_cuda_log(x);
    case T4A_CUDA_UNARY_SIN:
      return t4a_cuda_sin(x);
    case T4A_CUDA_UNARY_COS:
      return t4a_cuda_cos(x);
    case T4A_CUDA_UNARY_TANH:
      return t4a_cuda_tanh(x);
    case T4A_CUDA_UNARY_SQRT:
      return t4a_cuda_sqrt(x);
    case T4A_CUDA_UNARY_RSQRT:
      return t4a_cuda_rsqrt(x);
    case T4A_CUDA_UNARY_EXPM1:
      return t4a_cuda_expm1(x);
    case T4A_CUDA_UNARY_LOG1P:
      return t4a_cuda_log1p(x);
    case T4A_CUDA_UNARY_ERF:
      return t4a_cuda_erf(x);
    case T4A_CUDA_UNARY_ABS:
      return t4a_cuda_fabs(x);
    case T4A_CUDA_UNARY_SIGN:
      // INVARIANT: the baseline returns the literal +0 for either signed zero
      // (elementwise.rs:523), not the input, so this is not f32::signum.
      if (x != x) {
        return x;
      }
      if (x == F(0)) {
        return F(0);
      }
      return x > F(0) ? F(1) : F(-1);
    default:
      return x;
  }
}

template <typename I>
__device__ __forceinline__ I unary_int(int op, I x) {
  switch (op) {
    case T4A_CUDA_UNARY_NEG:
      return t4a_cuda_neg<I>(x);
    case T4A_CUDA_UNARY_ABS:
      return x < I(0) ? t4a_cuda_neg<I>(x) : x;
    case T4A_CUDA_UNARY_SIGN:
      if (x == I(0)) {
        return I(0);
      }
      return x > I(0) ? I(1) : I(-1);
    default:
      return x;
  }
}

template <typename C, typename F>
__device__ __forceinline__ C unary_complex(int op, C value) {
  if (op == T4A_CUDA_UNARY_NEG) {
    return t4a_cuda_neg<C>(value);
  }
  if (op == T4A_CUDA_UNARY_CONJ) {
    C result = {value.re, -value.im};
    return result;
  }
  if (op == T4A_CUDA_UNARY_SIGN) {
    if (value.re == F(0) && value.im == F(0)) {
      C zero = {F(0), F(0)};
      return zero;
    }
    const F magnitude = t4a_cuda_abs_complex(value);
    C result = {value.re / magnitude, value.im / magnitude};
    return result;
  }
  return value;
}

}  // namespace

extern "C" __global__ void t4a_cuda_unary(void* out, const void* in, int dst_dtype, int src_dtype,
                                          int op, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  if (dst_dtype == T4A_CUDA_F32 && src_dtype == T4A_CUDA_F32) {
    static_cast<float*>(out)[index] = unary_float<float>(op, static_cast<const float*>(in)[index]);
    return;
  }
  if (dst_dtype == T4A_CUDA_F64 && src_dtype == T4A_CUDA_F64) {
    static_cast<double*>(out)[index] = unary_float<double>(op, static_cast<const double*>(in)[index]);
    return;
  }
  if (dst_dtype == T4A_CUDA_I32 && src_dtype == T4A_CUDA_I32) {
    static_cast<int*>(out)[index] = unary_int<int>(op, static_cast<const int*>(in)[index]);
    return;
  }
  if (dst_dtype == T4A_CUDA_I64 && src_dtype == T4A_CUDA_I64) {
    static_cast<long long*>(out)[index] =
        unary_int<long long>(op, static_cast<const long long*>(in)[index]);
    return;
  }
  if (dst_dtype == T4A_CUDA_C32 && src_dtype == T4A_CUDA_C32) {
    static_cast<t4a_cuda_c32*>(out)[index] =
        unary_complex<t4a_cuda_c32, float>(op, static_cast<const t4a_cuda_c32*>(in)[index]);
    return;
  }
  if (dst_dtype == T4A_CUDA_C64 && src_dtype == T4A_CUDA_C64) {
    static_cast<t4a_cuda_c64*>(out)[index] =
        unary_complex<t4a_cuda_c64, double>(op, static_cast<const t4a_cuda_c64*>(in)[index]);
    return;
  }
  // ABS of a complex operand is the only unequal dtype pair (abs_complex32/64).
  if (dst_dtype == T4A_CUDA_F32 && src_dtype == T4A_CUDA_C32) {
    const t4a_cuda_c32 value = static_cast<const t4a_cuda_c32*>(in)[index];
    static_cast<float*>(out)[index] = t4a_cuda_abs_complex(value);
    return;
  }
  if (dst_dtype == T4A_CUDA_F64 && src_dtype == T4A_CUDA_C64) {
    const t4a_cuda_c64 value = static_cast<const t4a_cuda_c64*>(in)[index];
    static_cast<double*>(out)[index] = t4a_cuda_abs_complex(value);
    return;
  }
}

namespace {

// --- binary -----------------------------------------------------------------

// One element of the real binary tags; shared with t4a_cuda_strided_binary so the
// two entries cannot drift.
template <typename F>
__device__ __forceinline__ F binary_float_value(int op, F l, F r) {
  F result = l;
  switch (op) {
    case T4A_CUDA_BINARY_MAX:
      result = t4a_cuda_max<F>(l, r);
      break;
    case T4A_CUDA_BINARY_MIN:
      result = t4a_cuda_min<F>(l, r);
      break;
    case T4A_CUDA_BINARY_POW:
      result = t4a_cuda_pow(l, r);
      break;
    case T4A_CUDA_BINARY_REM:
      result = t4a_cuda_fmod(l, r);
      break;
    case T4A_CUDA_BINARY_MUL:
      result = l * r;
      break;
    case T4A_CUDA_BINARY_DIV:
      result = l / r;
      break;
    case T4A_CUDA_BINARY_ADD:
      result = t4a_cuda_add<F>(l, r);
      break;
    case T4A_CUDA_BINARY_SUB:
      result = t4a_cuda_sub<F>(l, r);
      break;
    default:
      break;
  }
  return result;
}

template <typename F>
__device__ __forceinline__ void binary_float_body(void* out, const void* lhs, const void* rhs,
                                                  int op, int scalar_side, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const long long li = scalar_side == T4A_CUDA_SCALAR_LHS ? 0 : index;
  const long long ri = scalar_side == T4A_CUDA_SCALAR_RHS ? 0 : index;
  const F l = static_cast<const F*>(lhs)[li];
  const F r = static_cast<const F*>(rhs)[ri];
  static_cast<F*>(out)[index] = binary_float_value<F>(op, l, r);
}

// Integer division/remainder/power keep the baseline's explicit exceptional arms
// so that INT_MIN / -1, a zero divisor and a negative exponent never trap; the
// rejection is reported through the sticky status word instead.
template <typename I>
__device__ __forceinline__ I binary_int_value(int op, I x, I y, unsigned int* err) {
  I result = x;
  switch (op) {
    case T4A_CUDA_BINARY_MAX:
      result = x >= y ? x : y;
      break;
    case T4A_CUDA_BINARY_MIN:
      result = x <= y ? x : y;
      break;
    case T4A_CUDA_BINARY_MUL:
      result = t4a_cuda_mul<I>(x, y);
      break;
    case T4A_CUDA_BINARY_ADD:
      result = t4a_cuda_add<I>(x, y);
      break;
    case T4A_CUDA_BINARY_SUB:
      result = t4a_cuda_sub<I>(x, y);
      break;
    case T4A_CUDA_BINARY_DIV_CHECKED:
      if (y == I(0)) {
        atomicExch(err, 1u);
        result = I(0);
      } else if (y == I(-1)) {
        result = t4a_cuda_neg<I>(x);
      } else {
        result = x / y;
      }
      break;
    case T4A_CUDA_BINARY_REM_CHECKED:
      if (y == I(0)) {
        atomicExch(err, 1u);
        result = I(0);
      } else if (y == I(-1)) {
        result = I(0);
      } else {
        result = t4a_cuda_sub<I>(x, t4a_cuda_mul<I>(x / y, y));
      }
      break;
    case T4A_CUDA_BINARY_POW_CHECKED: {
      I exponent = y;
      if (exponent < I(0)) {
        atomicExch(err, 1u);
        result = I(0);
        break;
      }
      I base = x;
      I accumulator = I(1);
      while (exponent > I(0)) {
        const I quotient = exponent / I(2);
        const I remainder = t4a_cuda_sub<I>(exponent, t4a_cuda_mul<I>(quotient, I(2)));
        if (remainder != I(0)) {
          accumulator = t4a_cuda_mul<I>(accumulator, base);
        }
        exponent = quotient;
        if (exponent > I(0)) {
          base = t4a_cuda_mul<I>(base, base);
        }
      }
      result = accumulator;
      break;
    }
    default:
      break;
  }
  return result;
}

template <typename I>
__device__ __forceinline__ void binary_int_body(void* out, const void* lhs, const void* rhs, int op,
                                                int scalar_side, long long len, unsigned int* err) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const long long li = scalar_side == T4A_CUDA_SCALAR_LHS ? 0 : index;
  const long long ri = scalar_side == T4A_CUDA_SCALAR_RHS ? 0 : index;
  const I x = static_cast<const I*>(lhs)[li];
  const I y = static_cast<const I*>(rhs)[ri];
  static_cast<I*>(out)[index] = binary_int_value<I>(op, x, y, err);
}

// `len` counts complex elements; MUL reads and writes whole values and honours the
// scalar side, DIV_COMPLEX addresses the interleaved parts of element `index` at
// 2 * index.
template <typename C, typename F>
__device__ __forceinline__ void binary_complex_body(void* out, const void* lhs, const void* rhs,
                                                    int op, int scalar_side, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  if (op == T4A_CUDA_BINARY_MUL || op == T4A_CUDA_BINARY_ADD || op == T4A_CUDA_BINARY_SUB) {
    const long long li = scalar_side == T4A_CUDA_SCALAR_LHS ? 0 : index;
    const long long ri = scalar_side == T4A_CUDA_SCALAR_RHS ? 0 : index;
    const C l = static_cast<const C*>(lhs)[li];
    const C r = static_cast<const C*>(rhs)[ri];
    C result;
    if (op == T4A_CUDA_BINARY_ADD) {
      result = t4a_cuda_add<C>(l, r);
    } else if (op == T4A_CUDA_BINARY_SUB) {
      result = t4a_cuda_sub<C>(l, r);
    } else {
      result = t4a_cuda_mul<C>(l, r);
    }
    static_cast<C*>(out)[index] = result;
    return;
  }
  const long long part = index * 2;
  t4a_cuda_div_complex_parts_at<F>(static_cast<const F*>(lhs)[part],
                                   static_cast<const F*>(lhs)[part + 1],
                                   static_cast<const F*>(rhs)[part],
                                   static_cast<const F*>(rhs)[part + 1],
                                   static_cast<F*>(out), part);
}

// Mixed real/complex: one real operand of a single element and one complex
// operand of `len` values (2 * len components on each of the two buffers).
template <typename F>
__device__ __forceinline__ void mixed_real_complex_body(void* out, const void* lhs, const void* rhs,
                                                        int op, int scalar_side, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const bool real_lhs = scalar_side == T4A_CUDA_SCALAR_LHS;
  const void* real_ptr = real_lhs ? lhs : rhs;
  const void* complex_ptr = real_lhs ? rhs : lhs;
  const F scalar = static_cast<const F*>(real_ptr)[0];
  const F* parts = static_cast<const F*>(complex_ptr);
  F* out_parts = static_cast<F*>(out);
  const long long part = index * 2;
  const F re = parts[part];
  const F im = parts[part + 1];
  const F zero = F(0);
  if (op == T4A_CUDA_BINARY_MIXED_DIV) {
    if (real_lhs) {
      t4a_cuda_div_complex_parts_at<F>(scalar, zero, re, im, out_parts, part);
    } else {
      t4a_cuda_div_complex_parts_at<F>(re, im, scalar, zero, out_parts, part);
    }
    return;
  }
  // INVARIANT: keep the zero cross terms. They decide NaN, infinity, overflow and
  // the sign of zero once the real operand is promoted to Complex(real, +0.0), and
  // simplify away only if the compiler may assume the other component is finite.
  F out_re;
  F out_im;
  if (op == T4A_CUDA_BINARY_MIXED_ADD) {
    if (real_lhs) {
      out_re = scalar + re;
      out_im = zero + im;
    } else {
      out_re = re + scalar;
      out_im = im + zero;
    }
  } else if (op == T4A_CUDA_BINARY_MIXED_SUB) {
    if (real_lhs) {
      out_re = scalar - re;
      out_im = zero - im;
    } else {
      out_re = re - scalar;
      out_im = im - zero;
    }
  } else {
    if (real_lhs) {
      out_re = scalar * re - zero * im;
      out_im = scalar * im + zero * re;
    } else {
      out_re = re * scalar - im * zero;
      out_im = re * zero + im * scalar;
    }
  }
  out_parts[part] = out_re;
  out_parts[part + 1] = out_im;
}

}  // namespace

extern "C" __global__ void t4a_cuda_binary(void* out, const void* lhs, const void* rhs, int dtype,
                                           int op, int scalar_side, long long len,
                                           unsigned int* err) {
  switch (dtype) {
    case T4A_CUDA_F32:
      if (op >= T4A_CUDA_BINARY_MIXED_ADD && op <= T4A_CUDA_BINARY_MIXED_DIV) {
        mixed_real_complex_body<float>(out, lhs, rhs, op, scalar_side, len);
      } else {
        binary_float_body<float>(out, lhs, rhs, op, scalar_side, len);
      }
      break;
    case T4A_CUDA_F64:
      if (op >= T4A_CUDA_BINARY_MIXED_ADD && op <= T4A_CUDA_BINARY_MIXED_DIV) {
        mixed_real_complex_body<double>(out, lhs, rhs, op, scalar_side, len);
      } else {
        binary_float_body<double>(out, lhs, rhs, op, scalar_side, len);
      }
      break;
    case T4A_CUDA_I32:
      binary_int_body<int>(out, lhs, rhs, op, scalar_side, len, err);
      break;
    case T4A_CUDA_I64:
      binary_int_body<long long>(out, lhs, rhs, op, scalar_side, len, err);
      break;
    case T4A_CUDA_C32:
      binary_complex_body<t4a_cuda_c32, float>(out, lhs, rhs, op, scalar_side, len);
      break;
    case T4A_CUDA_C64:
      binary_complex_body<t4a_cuda_c64, double>(out, lhs, rhs, op, scalar_side, len);
      break;
    default:
      break;
  }
}

// --- comparison, selection and clamping -------------------------------------

namespace {

template <typename T>
__device__ __forceinline__ bool compare_value(int cmp, T x, T y) {
  bool predicate = false;
  switch (cmp) {
    case T4A_CUDA_COMPARE_EQ:
      predicate = x == y;
      break;
    case T4A_CUDA_COMPARE_LT:
      predicate = x < y;
      break;
    case T4A_CUDA_COMPARE_LE:
      predicate = x <= y;
      break;
    case T4A_CUDA_COMPARE_GT:
      predicate = x > y;
      break;
    case T4A_CUDA_COMPARE_GE:
      predicate = x >= y;
      break;
    default:
      predicate = false;
      break;
  }
  return predicate;
}

template <typename T>
__device__ __forceinline__ void compare_body(void* out, const void* lhs, const void* rhs, int cmp,
                                             long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  const T x = static_cast<const T*>(lhs)[index];
  const T y = static_cast<const T*>(rhs)[index];
  static_cast<unsigned char*>(out)[index] = compare_value<T>(cmp, x, y) ? 1u : 0u;
}

template <typename T>
__device__ __forceinline__ void select_body(void* out, const void* pred, const void* on_true,
                                            const void* on_false, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  static_cast<T*>(out)[index] = static_cast<const unsigned char*>(pred)[index] != 0
                                    ? static_cast<const T*>(on_true)[index]
                                    : static_cast<const T*>(on_false)[index];
}

// Ordered clamp: a NaN input propagates and a NaN bound disables its comparison.
// Inverted bounds are not rejected: the statement order decides, so a value below
// the lower bound yields the lower bound and every other value the upper one. The
// baseline left this to CubeCL's expansion; see the slice design.
template <typename F>
__device__ __forceinline__ void clamp_body(void* out, const void* in, const void* lower,
                                           const void* upper, long long len) {
  long long index;
  if (!t4a_cuda_tail_index(&index, len)) {
    return;
  }
  static_cast<F*>(out)[index] = t4a_cuda_clamp(static_cast<const F*>(in)[index],
                                               static_cast<const F*>(lower)[index],
                                               static_cast<const F*>(upper)[index]);
}

}  // namespace

extern "C" __global__ void t4a_cuda_compare(void* out, const void* lhs, const void* rhs, int dtype,
                                            int cmp, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      compare_body<float>(out, lhs, rhs, cmp, len);
      break;
    case T4A_CUDA_F64:
      compare_body<double>(out, lhs, rhs, cmp, len);
      break;
    case T4A_CUDA_I32:
      compare_body<int>(out, lhs, rhs, cmp, len);
      break;
    case T4A_CUDA_I64:
      compare_body<long long>(out, lhs, rhs, cmp, len);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_select(void* out, const void* pred, const void* on_true,
                                           const void* on_false, int dtype, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      select_body<float>(out, pred, on_true, on_false, len);
      break;
    case T4A_CUDA_F64:
      select_body<double>(out, pred, on_true, on_false, len);
      break;
    case T4A_CUDA_I32:
      select_body<int>(out, pred, on_true, on_false, len);
      break;
    case T4A_CUDA_I64:
      select_body<long long>(out, pred, on_true, on_false, len);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_clamp(void* out, const void* in, const void* lower,
                                          const void* upper, int dtype, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      clamp_body<float>(out, in, lower, upper, len);
      break;
    case T4A_CUDA_F64:
      clamp_body<double>(out, in, lower, upper, len);
      break;
    default:
      break;
  }
}

// --- strided combination ----------------------------------------------------

namespace {

// One logical domain, three operand layouts: the baseline's broadcast multiply and
// its strided axpby with the axis maps already folded into per-axis strides (a
// broadcast axis has stride 0 in the operand's own layout).
template <typename T>
__device__ __forceinline__ void combined_strided_body(void* out, const void* a, const void* b,
                                                     const void* coefficients, int op,
                                                     t4a_cuda_layout out_layout,
                                                     t4a_cuda_layout a_layout,
                                                     t4a_cuda_layout b_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  // The three layouts share the destination extents, so one coordinate walk
  // serves all three addresses.
  long long out_index = out_layout.offset;
  long long a_index = a_layout.offset;
  long long b_index = b_layout.offset;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    out_index += coordinate * out_layout.strides[axis];
    a_index += coordinate * a_layout.strides[axis];
    b_index += coordinate * b_layout.strides[axis];
  }
  const T a_value = static_cast<const T*>(a)[a_index];
  const T b_value = static_cast<const T*>(b)[b_index];
  T result;
  if (op == T4A_CUDA_STRIDED_MUL) {
    result = t4a_cuda_mul<T>(a_value, b_value);
  } else {
    const T* coefficient = static_cast<const T*>(coefficients);
    result = t4a_cuda_add<T>(t4a_cuda_mul<T>(coefficient[0], a_value),
                             t4a_cuda_mul<T>(coefficient[1], b_value));
  }
  static_cast<T*>(out)[out_index] = result;
}

// --- Revision 9 general path ---------------------------------------------------
//
// The same formula with pointer geometry and a walk bounded by the destination's
// runtime rank; the fast body above is unchanged.

template <typename T>
__device__ __forceinline__ void combined_strided_body_any(
    void* out, const void* a, const void* b, const void* coefficients, int op,
    const t4a_cuda_layout_any& out_layout, const t4a_cuda_layout_any& a_layout,
    const t4a_cuda_layout_any& b_layout, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  // The three layouts share the destination extents, so one coordinate walk
  // serves all three addresses.
  long long out_index = out_layout.offset;
  long long a_index = a_layout.offset;
  long long b_index = b_layout.offset;
  for (int axis = 0; axis < out_layout.rank; ++axis) {
    const long long extent = out_layout.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    out_index += coordinate * out_layout.strides[axis];
    a_index += coordinate * a_layout.strides[axis];
    b_index += coordinate * b_layout.strides[axis];
  }
  const T a_value = static_cast<const T*>(a)[a_index];
  const T b_value = static_cast<const T*>(b)[b_index];
  T result;
  if (op == T4A_CUDA_STRIDED_MUL) {
    result = t4a_cuda_mul<T>(a_value, b_value);
  } else {
    const T* coefficient = static_cast<const T*>(coefficients);
    result = t4a_cuda_add<T>(t4a_cuda_mul<T>(coefficient[0], a_value),
                             t4a_cuda_mul<T>(coefficient[1], b_value));
  }
  static_cast<T*>(out)[out_index] = result;
}

}  // namespace

extern "C" __global__ void t4a_cuda_combined_strided(
    void* out, const void* a, const void* b, const void* coefficients, t4a_cuda_layout out_layout,
    t4a_cuda_layout a_layout, t4a_cuda_layout b_layout, int dtype, int op, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      combined_strided_body<float>(out, a, b, coefficients, op, out_layout, a_layout, b_layout,
                                   len);
      break;
    case T4A_CUDA_F64:
      combined_strided_body<double>(out, a, b, coefficients, op, out_layout, a_layout, b_layout,
                                    len);
      break;
    case T4A_CUDA_I32:
      combined_strided_body<int>(out, a, b, coefficients, op, out_layout, a_layout, b_layout, len);
      break;
    case T4A_CUDA_I64:
      combined_strided_body<long long>(out, a, b, coefficients, op, out_layout, a_layout,
                                        b_layout, len);
      break;
    case T4A_CUDA_C32:
      combined_strided_body<t4a_cuda_c32>(out, a, b, coefficients, op, out_layout, a_layout,
                                          b_layout, len);
      break;
    case T4A_CUDA_C64:
      combined_strided_body<t4a_cuda_c64>(out, a, b, coefficients, op, out_layout, a_layout,
                                          b_layout, len);
      break;
    default:
      break;
  }
}

extern "C" __global__ void t4a_cuda_combined_strided_any(
    void* out, const void* a, const void* b, const void* coefficients,
    t4a_cuda_layout_any out_layout, t4a_cuda_layout_any a_layout,
    t4a_cuda_layout_any b_layout, int dtype, int op, long long len) {
  // The same six dtypes as the fast entry: Bool is not a strided-combination
  // element type.
  switch (dtype) {
    case T4A_CUDA_F32:
      combined_strided_body_any<float>(out, a, b, coefficients, op, out_layout, a_layout,
                                       b_layout, len);
      break;
    case T4A_CUDA_F64:
      combined_strided_body_any<double>(out, a, b, coefficients, op, out_layout, a_layout,
                                        b_layout, len);
      break;
    case T4A_CUDA_I32:
      combined_strided_body_any<int>(out, a, b, coefficients, op, out_layout, a_layout,
                                     b_layout, len);
      break;
    case T4A_CUDA_I64:
      combined_strided_body_any<long long>(out, a, b, coefficients, op, out_layout, a_layout,
                                           b_layout, len);
      break;
    case T4A_CUDA_C32:
      combined_strided_body_any<t4a_cuda_c32>(out, a, b, coefficients, op, out_layout,
                                              a_layout, b_layout, len);
      break;
    case T4A_CUDA_C64:
      combined_strided_body_any<t4a_cuda_c64>(out, a, b, coefficients, op, out_layout,
                                              a_layout, b_layout, len);
      break;
    default:
      break;
  }
}

// --- layout-taking elementwise (revision 15) ---------------------------------
//
// t4a_cuda_strided_unary / _binary / _compare / _select read every operand through
// its own layout, so a permuted, offset, broadcast (stride zero) or reversed view
// is read where it lies instead of being staged into a compact copy. Each element
// is computed by the same value function as the flat entry it mirrors
// (unary_float / unary_int / unary_complex / t4a_cuda_abs_complex,
// t4a_cuda_add / sub / mul / div, binary_float_value, binary_int_value,
// t4a_cuda_div_complex, compare_value), so a strided launch is bit-identical to
// the flat launch on a compact copy of the same operands. Every operand layout
// shares the destination extents; one coordinate walk serves all addresses. The
// fast symbols take by-value t4a_cuda_layout (rank <= T4A_CUDA_MAX_RANK); the
// `_any` symbols take t4a_cuda_layout_any (device-resident geometry, any rank).

namespace {

template <typename L>
__device__ __forceinline__ void strided_index2(long long flat, const L& out, const L& a,
                                               long long* out_index, long long* a_index) {
  long long o = out.offset;
  long long x = a.offset;
  for (int axis = 0; axis < out.rank; ++axis) {
    const long long extent = out.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    o += coordinate * out.strides[axis];
    x += coordinate * a.strides[axis];
  }
  *out_index = o;
  *a_index = x;
}

template <typename L>
__device__ __forceinline__ void strided_index3(long long flat, const L& out, const L& a, const L& b,
                                               long long* out_index, long long* a_index,
                                               long long* b_index) {
  long long o = out.offset;
  long long x = a.offset;
  long long y = b.offset;
  for (int axis = 0; axis < out.rank; ++axis) {
    const long long extent = out.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    o += coordinate * out.strides[axis];
    x += coordinate * a.strides[axis];
    y += coordinate * b.strides[axis];
  }
  *out_index = o;
  *a_index = x;
  *b_index = y;
}

template <typename L>
__device__ __forceinline__ void strided_index4(long long flat, const L& out, const L& a, const L& b,
                                               const L& c, long long* out_index, long long* a_index,
                                               long long* b_index, long long* c_index) {
  long long o = out.offset;
  long long x = a.offset;
  long long y = b.offset;
  long long z = c.offset;
  for (int axis = 0; axis < out.rank; ++axis) {
    const long long extent = out.extents[axis];
    const long long coordinate = flat % extent;
    flat /= extent;
    o += coordinate * out.strides[axis];
    x += coordinate * a.strides[axis];
    y += coordinate * b.strides[axis];
    z += coordinate * c.strides[axis];
  }
  *out_index = o;
  *a_index = x;
  *b_index = y;
  *c_index = z;
}

template <typename L>
__device__ __forceinline__ void strided_unary_body(void* out, const void* in, const L& out_layout,
                                                   const L& in_layout, int dst_dtype,
                                                   int src_dtype, int op, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long o;
  long long i;
  strided_index2(flat, out_layout, in_layout, &o, &i);
  if (dst_dtype == T4A_CUDA_F32 && src_dtype == T4A_CUDA_F32) {
    static_cast<float*>(out)[o] = unary_float<float>(op, static_cast<const float*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_F64 && src_dtype == T4A_CUDA_F64) {
    static_cast<double*>(out)[o] = unary_float<double>(op, static_cast<const double*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_I32 && src_dtype == T4A_CUDA_I32) {
    static_cast<int*>(out)[o] = unary_int<int>(op, static_cast<const int*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_I64 && src_dtype == T4A_CUDA_I64) {
    static_cast<long long*>(out)[o] =
        unary_int<long long>(op, static_cast<const long long*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_C32 && src_dtype == T4A_CUDA_C32) {
    static_cast<t4a_cuda_c32*>(out)[o] =
        unary_complex<t4a_cuda_c32, float>(op, static_cast<const t4a_cuda_c32*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_C64 && src_dtype == T4A_CUDA_C64) {
    static_cast<t4a_cuda_c64*>(out)[o] =
        unary_complex<t4a_cuda_c64, double>(op, static_cast<const t4a_cuda_c64*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_F32 && src_dtype == T4A_CUDA_C32) {
    static_cast<float*>(out)[o] = t4a_cuda_abs_complex(static_cast<const t4a_cuda_c32*>(in)[i]);
  } else if (dst_dtype == T4A_CUDA_F64 && src_dtype == T4A_CUDA_C64) {
    static_cast<double*>(out)[o] = t4a_cuda_abs_complex(static_cast<const t4a_cuda_c64*>(in)[i]);
  }
}

template <typename F>
__device__ __forceinline__ F strided_binary_float(int op, F l, F r) {
  switch (op) {
    case T4A_CUDA_STRIDED_BINARY_ADD:
      return t4a_cuda_add<F>(l, r);
    case T4A_CUDA_STRIDED_BINARY_SUB:
      return t4a_cuda_sub<F>(l, r);
    case T4A_CUDA_STRIDED_BINARY_MUL:
      return t4a_cuda_mul_elementwise<F>(l, r);
    case T4A_CUDA_STRIDED_BINARY_DIV:
      return t4a_cuda_div<F>(l, r);
    case T4A_CUDA_STRIDED_BINARY_MAX:
      return binary_float_value<F>(T4A_CUDA_BINARY_MAX, l, r);
    case T4A_CUDA_STRIDED_BINARY_MIN:
      return binary_float_value<F>(T4A_CUDA_BINARY_MIN, l, r);
    case T4A_CUDA_STRIDED_BINARY_POW:
      return binary_float_value<F>(T4A_CUDA_BINARY_POW, l, r);
    case T4A_CUDA_STRIDED_BINARY_REM:
      return binary_float_value<F>(T4A_CUDA_BINARY_REM, l, r);
    default:
      return l;
  }
}

template <typename I>
__device__ __forceinline__ I strided_binary_int(int op, I x, I y, unsigned int* err) {
  switch (op) {
    case T4A_CUDA_STRIDED_BINARY_ADD:
      return t4a_cuda_add<I>(x, y);
    case T4A_CUDA_STRIDED_BINARY_SUB:
      return t4a_cuda_sub<I>(x, y);
    case T4A_CUDA_STRIDED_BINARY_MUL:
      return t4a_cuda_mul_elementwise<I>(x, y);
    case T4A_CUDA_STRIDED_BINARY_MAX:
      return binary_int_value<I>(T4A_CUDA_BINARY_MAX, x, y, err);
    case T4A_CUDA_STRIDED_BINARY_MIN:
      return binary_int_value<I>(T4A_CUDA_BINARY_MIN, x, y, err);
    case T4A_CUDA_STRIDED_BINARY_DIV_CHECKED:
      return binary_int_value<I>(T4A_CUDA_BINARY_DIV_CHECKED, x, y, err);
    case T4A_CUDA_STRIDED_BINARY_REM_CHECKED:
      return binary_int_value<I>(T4A_CUDA_BINARY_REM_CHECKED, x, y, err);
    case T4A_CUDA_STRIDED_BINARY_POW_CHECKED:
      return binary_int_value<I>(T4A_CUDA_BINARY_POW_CHECKED, x, y, err);
    default:
      return x;
  }
}

template <typename C>
__device__ __forceinline__ C strided_binary_complex(int op, C l, C r) {
  switch (op) {
    case T4A_CUDA_STRIDED_BINARY_ADD:
      return t4a_cuda_add<C>(l, r);
    case T4A_CUDA_STRIDED_BINARY_SUB:
      return t4a_cuda_sub<C>(l, r);
    case T4A_CUDA_STRIDED_BINARY_MUL:
      return t4a_cuda_mul_elementwise<C>(l, r);
    case T4A_CUDA_STRIDED_BINARY_DIV:
      return t4a_cuda_div_complex(l, r);
    default:
      return l;
  }
}

template <typename T, typename L, typename Op>
__device__ __forceinline__ void strided_binary_apply(void* out, const void* lhs, const void* rhs,
                                                     const L& out_layout, const L& lhs_layout,
                                                     const L& rhs_layout, long long len, Op f) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long o;
  long long a;
  long long b;
  strided_index3(flat, out_layout, lhs_layout, rhs_layout, &o, &a, &b);
  static_cast<T*>(out)[o] = f(static_cast<const T*>(lhs)[a], static_cast<const T*>(rhs)[b]);
}

template <typename L>
__device__ __forceinline__ void strided_binary_dispatch(void* out, const void* lhs, const void* rhs,
                                                        const L& out_layout, const L& lhs_layout,
                                                        const L& rhs_layout, int dtype, int op,
                                                        long long len, unsigned int* err) {
  switch (dtype) {
    case T4A_CUDA_F32:
      strided_binary_apply<float>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
                                  [op](float l, float r) { return strided_binary_float<float>(op, l, r); });
      break;
    case T4A_CUDA_F64:
      strided_binary_apply<double>(
          out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
          [op](double l, double r) { return strided_binary_float<double>(op, l, r); });
      break;
    case T4A_CUDA_I32:
      strided_binary_apply<int>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
                                [op, err](int x, int y) { return strided_binary_int<int>(op, x, y, err); });
      break;
    case T4A_CUDA_I64:
      strided_binary_apply<long long>(
          out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
          [op, err](long long x, long long y) { return strided_binary_int<long long>(op, x, y, err); });
      break;
    case T4A_CUDA_C32:
      strided_binary_apply<t4a_cuda_c32>(
          out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
          [op](t4a_cuda_c32 l, t4a_cuda_c32 r) { return strided_binary_complex(op, l, r); });
      break;
    case T4A_CUDA_C64:
      strided_binary_apply<t4a_cuda_c64>(
          out, lhs, rhs, out_layout, lhs_layout, rhs_layout, len,
          [op](t4a_cuda_c64 l, t4a_cuda_c64 r) { return strided_binary_complex(op, l, r); });
      break;
    default:
      break;
  }
}

template <typename T, typename L>
__device__ __forceinline__ void strided_compare_body(void* out, const void* lhs, const void* rhs,
                                                     const L& out_layout, const L& lhs_layout,
                                                     const L& rhs_layout, int cmp, long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long o;
  long long a;
  long long b;
  strided_index3(flat, out_layout, lhs_layout, rhs_layout, &o, &a, &b);
  static_cast<unsigned char*>(out)[o] =
      compare_value<T>(cmp, static_cast<const T*>(lhs)[a], static_cast<const T*>(rhs)[b]) ? 1u
                                                                                          : 0u;
}

template <typename L>
__device__ __forceinline__ void strided_compare_dispatch(void* out, const void* lhs,
                                                         const void* rhs, const L& out_layout,
                                                         const L& lhs_layout, const L& rhs_layout,
                                                         int dtype, int cmp, long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      strided_compare_body<float>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, cmp, len);
      break;
    case T4A_CUDA_F64:
      strided_compare_body<double>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, cmp, len);
      break;
    case T4A_CUDA_I32:
      strided_compare_body<int>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, cmp, len);
      break;
    case T4A_CUDA_I64:
      strided_compare_body<long long>(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, cmp,
                                      len);
      break;
    default:
      break;
  }
}

template <typename T, typename L>
__device__ __forceinline__ void strided_select_body(void* out, const void* pred,
                                                    const void* on_true, const void* on_false,
                                                    const L& out_layout, const L& pred_layout,
                                                    const L& true_layout, const L& false_layout,
                                                    long long len) {
  long long flat;
  if (!t4a_cuda_tail_index(&flat, len)) {
    return;
  }
  long long o;
  long long p;
  long long t;
  long long f;
  strided_index4(flat, out_layout, pred_layout, true_layout, false_layout, &o, &p, &t, &f);
  static_cast<T*>(out)[o] = static_cast<const unsigned char*>(pred)[p] != 0
                                ? static_cast<const T*>(on_true)[t]
                                : static_cast<const T*>(on_false)[f];
}

template <typename L>
__device__ __forceinline__ void strided_select_dispatch(void* out, const void* pred,
                                                        const void* on_true, const void* on_false,
                                                        const L& out_layout, const L& pred_layout,
                                                        const L& true_layout,
                                                        const L& false_layout, int dtype,
                                                        long long len) {
  switch (dtype) {
    case T4A_CUDA_F32:
      strided_select_body<float>(out, pred, on_true, on_false, out_layout, pred_layout,
                                 true_layout, false_layout, len);
      break;
    case T4A_CUDA_F64:
      strided_select_body<double>(out, pred, on_true, on_false, out_layout, pred_layout,
                                  true_layout, false_layout, len);
      break;
    case T4A_CUDA_I32:
      strided_select_body<int>(out, pred, on_true, on_false, out_layout, pred_layout,
                               true_layout, false_layout, len);
      break;
    case T4A_CUDA_I64:
      strided_select_body<long long>(out, pred, on_true, on_false, out_layout, pred_layout,
                                     true_layout, false_layout, len);
      break;
    default:
      break;
  }
}

}  // namespace

extern "C" __global__ void t4a_cuda_strided_unary(void* out, const void* in,
                                                  t4a_cuda_layout out_layout,
                                                  t4a_cuda_layout in_layout, int dst_dtype,
                                                  int src_dtype, int op, long long len) {
  strided_unary_body(out, in, out_layout, in_layout, dst_dtype, src_dtype, op, len);
}

extern "C" __global__ void t4a_cuda_strided_unary_any(void* out, const void* in,
                                                      t4a_cuda_layout_any out_layout,
                                                      t4a_cuda_layout_any in_layout,
                                                      int dst_dtype, int src_dtype, int op,
                                                      long long len) {
  strided_unary_body(out, in, out_layout, in_layout, dst_dtype, src_dtype, op, len);
}

extern "C" __global__ void t4a_cuda_strided_binary(void* out, const void* lhs, const void* rhs,
                                                   t4a_cuda_layout out_layout,
                                                   t4a_cuda_layout lhs_layout,
                                                   t4a_cuda_layout rhs_layout, int dtype, int op,
                                                   long long len, unsigned int* err) {
  strided_binary_dispatch(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, dtype, op, len, err);
}

extern "C" __global__ void t4a_cuda_strided_binary_any(
    void* out, const void* lhs, const void* rhs, t4a_cuda_layout_any out_layout,
    t4a_cuda_layout_any lhs_layout, t4a_cuda_layout_any rhs_layout, int dtype, int op,
    long long len, unsigned int* err) {
  strided_binary_dispatch(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, dtype, op, len, err);
}

extern "C" __global__ void t4a_cuda_strided_compare(void* out, const void* lhs, const void* rhs,
                                                    t4a_cuda_layout out_layout,
                                                    t4a_cuda_layout lhs_layout,
                                                    t4a_cuda_layout rhs_layout, int dtype,
                                                    int cmp, long long len) {
  strided_compare_dispatch(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, dtype, cmp, len);
}

extern "C" __global__ void t4a_cuda_strided_compare_any(
    void* out, const void* lhs, const void* rhs, t4a_cuda_layout_any out_layout,
    t4a_cuda_layout_any lhs_layout, t4a_cuda_layout_any rhs_layout, int dtype, int cmp,
    long long len) {
  strided_compare_dispatch(out, lhs, rhs, out_layout, lhs_layout, rhs_layout, dtype, cmp, len);
}

extern "C" __global__ void t4a_cuda_strided_select(
    void* out, const void* pred, const void* on_true, const void* on_false,
    t4a_cuda_layout out_layout, t4a_cuda_layout pred_layout, t4a_cuda_layout true_layout,
    t4a_cuda_layout false_layout, int dtype, long long len) {
  strided_select_dispatch(out, pred, on_true, on_false, out_layout, pred_layout, true_layout,
                          false_layout, dtype, len);
}

extern "C" __global__ void t4a_cuda_strided_select_any(
    void* out, const void* pred, const void* on_true, const void* on_false,
    t4a_cuda_layout_any out_layout, t4a_cuda_layout_any pred_layout,
    t4a_cuda_layout_any true_layout, t4a_cuda_layout_any false_layout, int dtype, long long len) {
  strided_select_dispatch(out, pred, on_true, on_false, out_layout, pred_layout, true_layout,
                          false_layout, dtype, len);
}
