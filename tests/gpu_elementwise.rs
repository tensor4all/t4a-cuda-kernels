//! Numerical GPU tests for the revision-6 elementwise, comparison and
//! strided-combination entries.
//!
//! Every entry is launched through the raw CUDA driver API and compared against a
//! CPU reference: the standard math library where it has the function, the
//! documented promoted component form for the mixed real/complex tags, and a
//! literal port of `t4a_cuda_div_complex_parts_at` for the robust divisions.
//!
//! The kernels under test are translated from tenferro-rs
//! `14bc1ab933e3c7a9c966e08355670331ab9d993c`
//! (`crates/tenferro-gpu/src/kernels/elementwise.rs`, `structural.rs`), Copyright
//! (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors, MIT OR Apache-2.0. The
//! CPU division reference and the kernel helper derive from Julia Base
//! `complex.jl` (`cdiv`, `robust_cdiv1/2`, `scaling_cdiv`, `scaleargs_cdiv`),
//! Copyright (c) 2009-2026 Jeff Bezanson, Stefan Karpinski, Viral B. Shah and
//! other Julia contributors, MIT; the notice is in `THIRD_PARTY_NOTICES.md`. The
//! bulk of each family runs on a domain above one 256-thread block, so a
//! single-threaded or mis-mapped body cannot pass.
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_elementwise`;
//! requires a CUDA device.

mod gpu;

use gpu::{
    as_bytes, from_bytes, launch, launches, read_bytes, read_prefix, upload, written, Buf, Session,
    Slot,
};
use t4a_cuda_kernels::{
    plan_binary, plan_clamp, plan_combined_strided, plan_compare, plan_select, plan_unary,
    BinaryOp, CLayout, CompareOp, DType, Layout, ScalarSide, StridedOp, UnaryOp,
};

const SENTINEL: u8 = 0xCD;
/// Bytes reserved behind a logical output to catch a write past `len`.
const GUARD: usize = 13;
/// Above one 256-thread block, and not a multiple of it.
const BIG: usize = 100_003;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct C32 {
    re: f32,
    im: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct C64 {
    re: f64,
    im: f64,
}

fn sentinel(session: &Session, bytes: usize) -> Buf {
    written(session, &vec![SENTINEL; bytes])
}

/// A destination buffer of `len` elements plus a guard region.
fn guard(session: &Session, len: i64, dtype: DType) -> Buf {
    sentinel(session, len as usize * dtype.size() + GUARD)
}

fn assert_guard(bytes: &[u8], payload: usize, what: &str) {
    assert!(
        bytes[payload..].iter().all(|&byte| byte == SENTINEL),
        "{what}: the guard region behind the logical output was written"
    );
}

/// Relative closeness, with NaN matching NaN and infinities matching exactly.
fn relative(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    if expected == 0.0 {
        return actual == 0.0;
    }
    (actual - expected).abs() <= tolerance * expected.abs()
}

/// Bit-level equality up to the sign of zero, for results the ABI reproduces
/// exactly rather than within a tolerance.
fn same(actual: f64, expected: f64) -> bool {
    actual == expected || (actual.is_nan() && expected.is_nan())
}

/// Bit-exact equality with NaN considered equal to NaN: the promoted component
/// forms decide the sign of a zero result, so `==` would accept `+0` for `-0`.
fn same_bits64(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

fn same_bits32(actual: f32, expected: f32) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

/// A deterministic, well-spread sample of `[lo, hi)`.
fn spread(count: usize, lo: f64, hi: f64) -> Vec<f64> {
    const PHI: f64 = 0.618_033_988_749_894_9;
    (0..count)
        .map(|index| lo + (hi - lo) * (index as f64 * PHI).fract())
        .collect()
}

/// Element indices addressed by a layout, in flat logical order.
fn view_indices(extents: &[i64], strides: &[i64], offset: i64) -> Vec<i64> {
    let count: i64 = extents.iter().product();
    (0..count)
        .map(|flat| {
            let mut flat = flat;
            let mut index = offset;
            for axis in 0..extents.len() {
                let coordinate = flat % extents[axis];
                flat /= extents[axis];
                index += coordinate * strides[axis];
            }
            index
        })
        .collect()
}

// --- entry wrappers ---------------------------------------------------------

/// Run one unary launch and return the whole destination buffer, guard included.
fn unary_bytes(
    session: &Session,
    op: UnaryOp,
    dst: DType,
    src: DType,
    input: &Buf,
    out_bytes: usize,
    len: i64,
) -> Vec<u8> {
    let plan = plan_unary(op, dst, src, len).unwrap();
    let out = sentinel(session, out_bytes);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(input),
        Slot::i32(dst.tag()),
        Slot::i32(src.tag()),
        Slot::i32(op.tag()),
        Slot::i64(len),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_unary"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{op:?}"
    );
    read_bytes(session, &out)
}

fn unary_run<S: Copy, D: Copy>(
    session: &Session,
    op: UnaryOp,
    dst: DType,
    src: DType,
    input: &[S],
    len: i64,
) -> Vec<D> {
    let buffer = upload(session, input);
    let bytes = unary_bytes(
        session,
        op,
        dst,
        src,
        &buffer,
        len as usize * std::mem::size_of::<D>(),
        len,
    );
    from_bytes(&bytes)
}

/// Run one binary launch and return the logical output plus the sticky `err`
/// word the kernel may have written.
fn binary_run<S: Copy, D: Copy>(
    session: &Session,
    op: BinaryOp,
    dtype: DType,
    side: ScalarSide,
    lhs: &[S],
    rhs: &[S],
    len: i64,
) -> (Vec<D>, u32) {
    let lhs_buf = upload(session, lhs);
    let rhs_buf = upload(session, rhs);
    let err = written(session, &[0u8; 4]);
    let plan = plan_binary(op, dtype, side, len).unwrap();
    let out = guard(session, len, dtype);
    let bytes = len as usize * std::mem::size_of::<D>();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&lhs_buf),
        Slot::ptr(&rhs_buf),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        Slot::i32(side.tag()),
        Slot::i64(len),
        Slot::ptr(&err),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_binary"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{op:?}"
    );
    let flag = from_bytes::<u32>(&read_bytes(session, &err))[0];
    let all = read_bytes(session, &out);
    assert_guard(&all, bytes, "binary");
    (from_bytes(&all[..bytes]), flag)
}

/// Run one mixed real/complex launch: `real` is a single element, `complex` holds
/// `len` values, the output holds `len` values of the same complex type.
fn mixed64_run(
    session: &Session,
    op: BinaryOp,
    side: ScalarSide,
    real: f64,
    complex: &[C64],
    len: i64,
) -> Vec<C64> {
    let real_buf = upload(session, &[real]);
    let complex_buf = upload(session, complex);
    let err = written(session, &[0u8; 4]);
    let plan = plan_binary(op, DType::F64, side, len).unwrap();
    let out = sentinel(session, len as usize * 16);
    let (lhs, rhs) = match side {
        ScalarSide::Left => (&real_buf, &complex_buf),
        ScalarSide::Right => (&complex_buf, &real_buf),
        ScalarSide::Both => panic!("the mixed tags need a scalar side"),
    };
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(lhs),
        Slot::ptr(rhs),
        Slot::i32(DType::F64.tag()),
        Slot::i32(op.tag()),
        Slot::i32(side.tag()),
        Slot::i64(len),
        Slot::ptr(&err),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_binary"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{op:?}"
    );
    from_bytes(&read_prefix(session, &out, len as usize * 16))
}

fn mixed32_run(
    session: &Session,
    op: BinaryOp,
    side: ScalarSide,
    real: f32,
    complex: &[C32],
    len: i64,
) -> Vec<C32> {
    let real_buf = upload(session, &[real]);
    let complex_buf = upload(session, complex);
    let err = written(session, &[0u8; 4]);
    let plan = plan_binary(op, DType::F32, side, len).unwrap();
    let out = sentinel(session, len as usize * 8);
    let (lhs, rhs) = match side {
        ScalarSide::Left => (&real_buf, &complex_buf),
        ScalarSide::Right => (&complex_buf, &real_buf),
        ScalarSide::Both => panic!("the mixed tags need a scalar side"),
    };
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(lhs),
        Slot::ptr(rhs),
        Slot::i32(DType::F32.tag()),
        Slot::i32(op.tag()),
        Slot::i32(side.tag()),
        Slot::i64(len),
        Slot::ptr(&err),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_binary"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{op:?}"
    );
    from_bytes(&read_prefix(session, &out, len as usize * 8))
}

fn compare_run<T: Copy>(
    session: &Session,
    cmp: CompareOp,
    dtype: DType,
    lhs: &[T],
    rhs: &[T],
) -> Vec<u8> {
    let len = lhs.len() as i64;
    let lhs_buf = upload(session, lhs);
    let rhs_buf = upload(session, rhs);
    let plan = plan_compare(dtype, len).unwrap();
    let out = guard(session, len, DType::Bool);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&lhs_buf),
        Slot::ptr(&rhs_buf),
        Slot::i32(dtype.tag()),
        Slot::i32(cmp.tag()),
        Slot::i64(len),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_compare"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{cmp:?}"
    );
    read_bytes(session, &out)
}

fn select_run<T: Copy>(
    session: &Session,
    dtype: DType,
    pred: &[u8],
    on_true: &[T],
    on_false: &[T],
) -> Vec<u8> {
    let len = pred.len() as i64;
    let pred_buf = written(session, pred);
    let true_buf = upload(session, on_true);
    let false_buf = upload(session, on_false);
    let plan = plan_select(dtype, len).unwrap();
    let out = guard(session, len, dtype);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&pred_buf),
        Slot::ptr(&true_buf),
        Slot::ptr(&false_buf),
        Slot::i32(dtype.tag()),
        Slot::i64(len),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_select"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + if plan.is_empty() { 0 } else { 1 });
    read_bytes(session, &out)
}

fn clamp_run<T: Copy>(
    session: &Session,
    dtype: DType,
    input: &[T],
    lower: &[T],
    upper: &[T],
) -> Vec<u8> {
    let len = input.len() as i64;
    let in_buf = upload(session, input);
    let low_buf = upload(session, lower);
    let high_buf = upload(session, upper);
    let plan = plan_clamp(dtype, len).unwrap();
    let out = guard(session, len, dtype);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&in_buf),
        Slot::ptr(&low_buf),
        Slot::ptr(&high_buf),
        Slot::i32(dtype.tag()),
        Slot::i64(len),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_clamp"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + if plan.is_empty() { 0 } else { 1 });
    read_bytes(session, &out)
}

/// Run one strided-combination launch into a fresh destination and return the
/// destination bytes.
#[allow(clippy::too_many_arguments)]
fn combined_run(
    session: &Session,
    op: StridedOp,
    dtype: DType,
    out: &Layout,
    a: &Layout,
    b: &Layout,
    a_values: &[u8],
    b_values: &[u8],
    coefficients: Option<&[u8]>,
) -> Vec<u8> {
    let len = out.element_count().unwrap();
    let a_len = (a_values.len() / dtype.size()) as i64;
    let b_len = (b_values.len() / dtype.size()) as i64;
    let plan = plan_combined_strided(op, dtype, out, len, a, a_len, b, b_len).unwrap();
    let a_buf = written(session, a_values);
    let b_buf = written(session, b_values);
    let coefficient_buf = written(session, coefficients.unwrap_or(&[0u8; 8]));
    let out_buf = sentinel(session, len as usize * dtype.size());
    let mut slots = [
        Slot::ptr(&out_buf),
        Slot::ptr(&a_buf),
        Slot::ptr(&b_buf),
        Slot::ptr(&coefficient_buf),
        Slot::layout(&CLayout::new(out)),
        Slot::layout(&CLayout::new(a)),
        Slot::layout(&CLayout::new(b)),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        Slot::i64(len),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_combined_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + if plan.is_empty() { 0 } else { 1 },
        "{op:?}"
    );
    let all = read_bytes(session, &out_buf);
    assert_guard(&all, len as usize * dtype.size(), "combined strided");
    all[..len as usize * dtype.size()].to_vec()
}

// --- CPU references ---------------------------------------------------------

/// `erf` is not in Rust's standard library. The Maclaurin series is accurate to
/// f64 rounding on `|x| <= 2.2` (verified against `math.erf`); larger arguments
/// are compared against tabulated values instead.
fn erf(x: f64) -> f64 {
    let square = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..400 {
        term *= -square / n as f64;
        let contribution = term / (2 * n + 1) as f64;
        sum += contribution;
        if contribution.abs() <= 1e-18 * sum.abs() {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

fn reference_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Erf => erf(x),
        other => panic!("not a transcendental: {other:?}"),
    }
}

fn reference_f32(op: UnaryOp, x: f32) -> f32 {
    match op {
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Erf => erf(f64::from(x)) as f32,
        other => panic!("not a transcendental: {other:?}"),
    }
}

/// The documented promoted component form of the mixed tags, zero cross terms
/// included: they decide NaN, infinity and the sign of zero.
fn mixed_reference(op: BinaryOp, side: ScalarSide, real: f64, value: C64) -> (f64, f64) {
    let zero = 0.0f64;
    let (re, im) = (value.re, value.im);
    match (op, side) {
        (BinaryOp::MixedAdd, ScalarSide::Left) => (real + re, zero + im),
        (BinaryOp::MixedAdd, ScalarSide::Right) => (re + real, im + zero),
        (BinaryOp::MixedSub, ScalarSide::Left) => (real - re, zero - im),
        (BinaryOp::MixedSub, ScalarSide::Right) => (re - real, im - zero),
        (BinaryOp::MixedMul, ScalarSide::Left) => (real * re - zero * im, real * im + zero * re),
        (BinaryOp::MixedMul, ScalarSide::Right) => (re * real - im * zero, re * zero + im * real),
        (op, side) => panic!("not a mixed add/sub/mul: {op:?} {side:?}"),
    }
}

macro_rules! complex_division {
    ($name:ident, $f:ty, $max:expr, $min_positive:expr, $epsilon:expr, $nan:expr) => {
        /// Literal port of `t4a_cuda_div_complex_parts_at` (Baudin-Smith scaling
        /// with the subnormal/near-overflow rescale), in the element precision.
        fn $name(a0: $f, b0: $f, c0: $f, d0: $f) -> ($f, $f) {
            fn magnitude(value: $f) -> $f {
                if value < 0.0 {
                    -value
                } else {
                    value
                }
            }
            fn signed_unit(value: $f) -> $f {
                if value == 0.0 {
                    value
                } else if value < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
            fn term(a: $f, b: $f, c: $f, d: $f, r: $f, t: $f) -> $f {
                if r != 0.0 {
                    let br = b * r;
                    if br != 0.0 {
                        (a + br) * t
                    } else {
                        a * t + (b * t) * r
                    }
                } else {
                    (a + d * (b / c)) * t
                }
            }
            let max_value: $f = $max;
            let min_positive: $f = $min_positive;
            let epsilon: $f = $epsilon;
            let mut a = a0;
            let mut b = b0;
            let mut c = c0;
            let mut d = d0;
            if magnitude(c) > max_value || magnitude(d) > max_value {
                let a_finite = a == a && magnitude(a) <= max_value;
                let b_finite = b == b && magnitude(b) <= max_value;
                if a_finite && b_finite {
                    return (
                        0.0 * signed_unit(a) * signed_unit(c),
                        -0.0 * signed_unit(b) * signed_unit(d),
                    );
                }
                return ($nan, $nan);
            }
            let abs_a = magnitude(a);
            let abs_b = magnitude(b);
            let abs_c = magnitude(c);
            let abs_d = magnitude(d);
            let ab = if abs_a >= abs_b { abs_a } else { abs_b };
            let cd = if abs_c >= abs_d { abs_c } else { abs_d };
            let half: $f = 0.5;
            let two: $f = 2.0;
            let half_over = max_value * half;
            let two_under_eps = min_positive * two / epsilon;
            let mut scale: $f = 1.0;
            if ab >= half_over || ab <= two_under_eps || cd >= half_over || cd <= two_under_eps {
                let big: $f = two / (epsilon * epsilon);
                if ab >= half_over {
                    a = a * half;
                    b = b * half;
                    scale = scale * two;
                } else if ab <= two_under_eps {
                    a = a * big;
                    b = b * big;
                    scale = scale / big;
                }
                if cd >= half_over {
                    c = c * half;
                    d = d * half;
                    scale = scale * half;
                } else if cd <= two_under_eps {
                    c = c * big;
                    d = d * big;
                    scale = scale * big;
                }
            }
            if magnitude(d) <= magnitude(c) {
                let r = d / c;
                let t = 1.0 / (c + d * r);
                (
                    term(a, b, c, d, r, t) * scale,
                    term(b, -a, c, d, r, t) * scale,
                )
            } else {
                let r = c / d;
                let t = 1.0 / (c * r + d);
                (
                    term(b, a, d, c, r, t) * scale,
                    -term(a, -b, d, c, r, t) * scale,
                )
            }
        }
    };
}

complex_division!(
    cdiv32,
    f32,
    f32::MAX,
    f32::MIN_POSITIVE,
    f32::EPSILON,
    f32::from_bits(0x7fc0_0000)
);
complex_division!(
    cdiv64,
    f64,
    f64::MAX,
    f64::MIN_POSITIVE,
    f64::EPSILON,
    f64::from_bits(0x7ff8_0000_0000_0000)
);

// --- unary ------------------------------------------------------------------

#[test]
fn unary_transcendentals_match_the_cpu_reference() {
    let session = Session::new("elementwise");
    let cases: [(UnaryOp, f64, f64); 10] = [
        (UnaryOp::Exp, -80.0, 80.0),
        (UnaryOp::Log, 1e-6, 1e6),
        (UnaryOp::Sin, -100.0, 100.0),
        (UnaryOp::Cos, -100.0, 100.0),
        (UnaryOp::Tanh, -12.0, 12.0),
        (UnaryOp::Sqrt, 0.0, 1e6),
        (UnaryOp::Rsqrt, 1e-3, 1e6),
        (UnaryOp::Expm1, -10.0, 10.0),
        (UnaryOp::Log1p, -0.5, 1e6),
        (UnaryOp::Erf, -2.2, 2.2),
    ];
    for (op, lo, hi) in cases {
        let values = spread(BIG, lo, hi);
        let wide: Vec<f64> = unary_run(&session, op, DType::F64, DType::F64, &values, BIG as i64);
        for (index, (&got, &x)) in wide.iter().zip(values.iter()).enumerate() {
            let expected = reference_f64(op, x);
            assert!(
                relative(got, expected, 1e-13),
                "{op:?} f64[{index}]: got {got:?}, expected {expected:?} for {x:?}"
            );
        }
        // The same arguments at single precision, so both dtypes of the tag run.
        let single: Vec<f32> = values.iter().map(|&value| value as f32).collect();
        let narrow: Vec<f32> = unary_run(&session, op, DType::F32, DType::F32, &single, BIG as i64);
        for (index, (&got, &x)) in narrow.iter().zip(single.iter()).enumerate() {
            let expected = reference_f32(op, x);
            assert!(
                relative(f64::from(got), f64::from(expected), 1e-5),
                "{op:?} f32[{index}]: got {got:?}, expected {expected:?} for {x:?}"
            );
        }
    }

    // ERF past the series' domain: tabulated f64 values.
    let values = vec![3.0f64, 4.0, 5.0, -4.0];
    let got: Vec<f64> = unary_run(&session, UnaryOp::Erf, DType::F64, DType::F64, &values, 4);
    let expected = [
        0.999_977_909_503_001_4,
        0.999_999_984_582_742_1,
        0.999_999_999_998_462_6,
        -0.999_999_984_582_742_1,
    ];
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            relative(got, expected, 1e-12),
            "erf[{index}]: got {got:?}, expected {expected:?}"
        );
    }
}

#[test]
fn unary_neg_abs_sign_and_conj_cover_their_edge_cases() {
    let session = Session::new("elementwise");

    // Float NEG flips the sign bit, including both zeros and a NaN payload.
    let values = vec![0.0f64, -0.0, f64::INFINITY, -1.5, f64::NAN];
    let got: Vec<f64> = unary_run(&session, UnaryOp::Neg, DType::F64, DType::F64, &values, 5);
    assert_eq!(got[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(got[1].to_bits(), 0.0f64.to_bits());
    assert_eq!(got[2], f64::NEG_INFINITY);
    assert_eq!(got[3], 1.5);
    assert!(got[4].is_nan());
    let values32 = vec![0.0f32, -0.0, f32::NEG_INFINITY];
    let got: Vec<f32> = unary_run(&session, UnaryOp::Neg, DType::F32, DType::F32, &values32, 3);
    assert_eq!(got[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(got[1].to_bits(), 0.0f32.to_bits());
    assert_eq!(got[2], f32::INFINITY);

    // Integer NEG and ABS wrap on the minimum instead of overflowing.
    let ints = vec![i32::MIN, -1, 0, 1, i32::MAX];
    let got: Vec<i32> = unary_run(&session, UnaryOp::Neg, DType::I32, DType::I32, &ints, 5);
    assert_eq!(got, vec![i32::MIN, 1, 0, -1, -i32::MAX]);
    let got: Vec<i32> = unary_run(&session, UnaryOp::Abs, DType::I32, DType::I32, &ints, 5);
    assert_eq!(got, vec![i32::MIN, 1, 0, 1, i32::MAX]);
    let longs = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let got: Vec<i64> = unary_run(&session, UnaryOp::Neg, DType::I64, DType::I64, &longs, 5);
    assert_eq!(got, vec![i64::MIN, 1, 0, -1, -i64::MAX]);
    let got: Vec<i64> = unary_run(&session, UnaryOp::Abs, DType::I64, DType::I64, &longs, 5);
    assert_eq!(got, vec![i64::MIN, 1, 0, 1, i64::MAX]);

    // SIGN: NaN propagates, either zero is +0, everything else is +/-1.
    let values = vec![
        f32::NAN,
        0.0f32,
        -0.0,
        2.5,
        -2.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let got: Vec<f32> = unary_run(&session, UnaryOp::Sign, DType::F32, DType::F32, &values, 7);
    assert!(got[0].is_nan(), "SIGN propagates NaN");
    assert_eq!(got[1].to_bits(), 0.0f32.to_bits());
    assert_eq!(got[2].to_bits(), 0.0f32.to_bits(), "either zero is +0");
    assert_eq!(got[3], 1.0);
    assert_eq!(got[4], -1.0);
    assert_eq!(got[5], 1.0);
    assert_eq!(got[6], -1.0);
    let values64 = vec![f64::NAN, -0.0, -3.5];
    let got: Vec<f64> = unary_run(
        &session,
        UnaryOp::Sign,
        DType::F64,
        DType::F64,
        &values64,
        3,
    );
    assert!(got[0].is_nan());
    assert_eq!(got[1].to_bits(), 0.0f64.to_bits());
    assert_eq!(got[2], -1.0);
    let ints = vec![0i64, 7, -7, i64::MIN];
    let got: Vec<i64> = unary_run(&session, UnaryOp::Sign, DType::I64, DType::I64, &ints, 4);
    assert_eq!(got, vec![0, 1, -1, -1]);

    // ABS of a complex operand writes the magnitude at the component precision.
    let values = vec![
        C64 { re: 3.0, im: -4.0 },
        C64 { re: -0.0, im: 0.0 },
        C64 {
            re: 1e300,
            im: 1e300,
        },
        C64 {
            re: f64::NAN,
            im: 1.0,
        },
    ];
    let got: Vec<f64> = unary_run(&session, UnaryOp::Abs, DType::F64, DType::C64, &values, 4);
    assert_eq!(got[0], 5.0);
    assert_eq!(got[1].to_bits(), 0.0f64.to_bits());
    assert!(relative(got[2], 1e300 * 2.0f64.sqrt(), 1e-13));
    assert!(got[3].is_nan());
    let values32 = vec![C32 { re: 3.0, im: -4.0 }, C32 { re: 0.0, im: 0.0 }];
    let got: Vec<f32> = unary_run(&session, UnaryOp::Abs, DType::F32, DType::C32, &values32, 2);
    assert_eq!(got, vec![5.0, 0.0]);

    // Complex SIGN: zero stays zero, anything else is z / |z|.
    let values = vec![
        C64 { re: 3.0, im: -4.0 },
        C64 { re: 0.0, im: 0.0 },
        C64 { re: -0.0, im: 0.0 },
    ];
    let got: Vec<C64> = unary_run(&session, UnaryOp::Sign, DType::C64, DType::C64, &values, 3);
    assert_eq!(got[0], C64 { re: 0.6, im: -0.8 });
    assert_eq!(got[1], C64 { re: 0.0, im: 0.0 });
    assert_eq!(got[2], C64 { re: 0.0, im: 0.0 });
    let values32 = vec![C32 { re: 0.0, im: 2.0 }];
    let got: Vec<C32> = unary_run(
        &session,
        UnaryOp::Sign,
        DType::C32,
        DType::C32,
        &values32,
        1,
    );
    assert_eq!(got[0], C32 { re: 0.0, im: 1.0 });

    // CONJ negates the imaginary part, so +0 becomes -0; the real part is kept.
    let values = vec![C64 { re: 1.5, im: 0.0 }, C64 { re: -2.0, im: -3.0 }];
    let got: Vec<C64> = unary_run(&session, UnaryOp::Conj, DType::C64, DType::C64, &values, 2);
    assert_eq!(got[0].re.to_bits(), 1.5f64.to_bits());
    assert_eq!(got[0].im.to_bits(), (-0.0f64).to_bits());
    assert_eq!(got[1], C64 { re: -2.0, im: 3.0 });
    let values32 = vec![C32 { re: 4.0, im: 0.0 }];
    let got: Vec<C32> = unary_run(
        &session,
        UnaryOp::Conj,
        DType::C32,
        DType::C32,
        &values32,
        1,
    );
    assert_eq!(got[0].im.to_bits(), (-0.0f32).to_bits());

    // The same four tags over a domain above one block, element by element.
    let values = spread(BIG, -5.0, 5.0);
    let negated: Vec<f64> = unary_run(
        &session,
        UnaryOp::Neg,
        DType::F64,
        DType::F64,
        &values,
        BIG as i64,
    );
    let magnitude: Vec<f64> = unary_run(
        &session,
        UnaryOp::Abs,
        DType::F64,
        DType::F64,
        &values,
        BIG as i64,
    );
    let units: Vec<f64> = unary_run(
        &session,
        UnaryOp::Sign,
        DType::F64,
        DType::F64,
        &values,
        BIG as i64,
    );
    for (index, &value) in values.iter().enumerate() {
        assert_eq!(negated[index].to_bits(), (-value).to_bits(), "NEG[{index}]");
        assert_eq!(
            magnitude[index].to_bits(),
            value.abs().to_bits(),
            "ABS[{index}]"
        );
        let expected = if value > 0.0 { 1.0 } else { -1.0 };
        assert_eq!(units[index], expected, "SIGN[{index}]");
    }
    let complex: Vec<C64> = values
        .iter()
        .map(|&value| C64 {
            re: value,
            im: -value,
        })
        .collect();
    let conjugated: Vec<C64> = unary_run(
        &session,
        UnaryOp::Conj,
        DType::C64,
        DType::C64,
        &complex,
        BIG as i64,
    );
    let sizes: Vec<f64> = unary_run(
        &session,
        UnaryOp::Abs,
        DType::F64,
        DType::C64,
        &complex,
        BIG as i64,
    );
    for (index, &value) in complex.iter().enumerate() {
        assert_eq!(
            conjugated[index].re.to_bits(),
            value.re.to_bits(),
            "CONJ real[{index}]"
        );
        assert_eq!(
            conjugated[index].im.to_bits(),
            (-value.im).to_bits(),
            "CONJ imaginary[{index}]"
        );
        let expected = value.re.hypot(value.im);
        assert!(
            relative(sizes[index], expected, 1e-13),
            "complex ABS[{index}]: {} != {expected}",
            sizes[index]
        );
    }
}

// --- binary -----------------------------------------------------------------

#[test]
fn binary_extrema_pow_rem_and_scalar_division() {
    let session = Session::new("elementwise");

    // MAX/MIN propagate NaN, lhs first, and keep the operand order for ties.
    let lhs = vec![f64::NAN, 1.0, 1.0, 3.0, -3.0];
    let rhs = vec![2.0, f64::NAN, f64::NAN, 3.0, 3.0];
    let (got, flag): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Max,
        DType::F64,
        ScalarSide::Both,
        &lhs,
        &rhs,
        5,
    );
    assert!(got[0].is_nan() && got[1].is_nan() && got[2].is_nan());
    assert_eq!(got[3], 3.0);
    assert_eq!(got[4], 3.0);
    assert_eq!(flag, 0, "MAX/MIN do not touch err");
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Min,
        DType::F64,
        ScalarSide::Both,
        &lhs,
        &rhs,
        5,
    );
    assert!(got[0].is_nan() && got[1].is_nan() && got[2].is_nan());
    assert_eq!(got[3], 3.0);
    assert_eq!(got[4], -3.0);
    let lhs32 = vec![f32::NAN, 2.0];
    let rhs32 = vec![1.0, f32::NAN];
    let (got, _): (Vec<f32>, u32) = binary_run(
        &session,
        BinaryOp::Max,
        DType::F32,
        ScalarSide::Both,
        &lhs32,
        &rhs32,
        2,
    );
    assert!(got[0].is_nan() && got[1].is_nan());

    // Both operands NaN with different payloads: the lhs payload must survive, so
    // reversing the comparison order would change the bits even though both are NaN.
    let lhs_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let rhs_nan = f64::from_bits(0x7ff8_0000_0000_0002);
    for op in [BinaryOp::Max, BinaryOp::Min] {
        let (got, _): (Vec<f64>, u32) = binary_run(
            &session,
            op,
            DType::F64,
            ScalarSide::Both,
            &[lhs_nan],
            &[rhs_nan],
            1,
        );
        assert_eq!(
            got[0].to_bits(),
            lhs_nan.to_bits(),
            "{op:?} keeps the lhs NaN payload"
        );
        let (got, _): (Vec<f64>, u32) = binary_run(
            &session,
            op,
            DType::F64,
            ScalarSide::Both,
            &[rhs_nan],
            &[lhs_nan],
            1,
        );
        assert_eq!(
            got[0].to_bits(),
            rhs_nan.to_bits(),
            "{op:?} keeps the new lhs payload"
        );
    }

    // Integer extrema use plain ordering, ties included.
    let lhs = vec![3i32, -3, i32::MIN, i32::MAX];
    let rhs = vec![-3i32, 3, i32::MAX, i32::MIN];
    let (got, _): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::Max,
        DType::I32,
        ScalarSide::Both,
        &lhs,
        &rhs,
        4,
    );
    assert_eq!(got, vec![3, 3, i32::MAX, i32::MAX]);
    let (got, _): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::Min,
        DType::I32,
        ScalarSide::Both,
        &lhs,
        &rhs,
        4,
    );
    assert_eq!(got, vec![-3, -3, i32::MIN, i32::MIN]);
    let lhs = vec![i64::MIN, i64::MAX];
    let rhs = vec![i64::MAX, i64::MIN];
    let (got, _): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::Max,
        DType::I64,
        ScalarSide::Both,
        &lhs,
        &rhs,
        2,
    );
    assert_eq!(got, vec![i64::MAX, i64::MAX]);
    let (got, _): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::Min,
        DType::I64,
        ScalarSide::Both,
        &lhs,
        &rhs,
        2,
    );
    assert_eq!(got, vec![i64::MIN, i64::MIN]);

    // POW against the standard library, plus 0^0 == 1 and a NaN branch.
    let base = spread(BIG, 0.5, 4.0);
    let exponent = spread(BIG, -3.0, 3.0);
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Pow,
        DType::F64,
        ScalarSide::Both,
        &base,
        &exponent,
        BIG as i64,
    );
    for (index, ((&got, &base), &exponent)) in
        got.iter().zip(base.iter()).zip(exponent.iter()).enumerate()
    {
        let expected = base.powf(exponent);
        assert!(
            relative(got, expected, 1e-13),
            "POW[{index}]: got {got:?}, expected {expected:?} for {base:?}^{exponent:?}"
        );
    }
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Pow,
        DType::F64,
        ScalarSide::Both,
        &[0.0, -2.0, 0.0, 2.0],
        &[0.0, 0.5, 2.0, 10.0],
        4,
    );
    assert_eq!(got[0], 1.0, "0^0 == 1");
    assert!(
        got[1].is_nan(),
        "a fractional exponent of a negative base is NaN"
    );
    assert_eq!(got[2], 0.0);
    assert_eq!(got[3], 1024.0);

    // REM is fmod: a zero result keeps the dividend's sign.
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Rem,
        DType::F64,
        ScalarSide::Both,
        &[5.5, -5.5, 4.0, -4.0, 1.0, 5.0],
        &[2.0, 2.0, 2.0, 2.0, 0.0, f64::NAN],
        6,
    );
    assert_eq!(got[0], 1.5);
    assert_eq!(got[1], -1.5);
    assert_eq!(got[2].to_bits(), 0.0f64.to_bits());
    assert_eq!(got[3].to_bits(), (-0.0f64).to_bits(), "the dividend's sign");
    assert!(got[4].is_nan() && got[5].is_nan());

    // DIV with the scalar on either side; IEEE division reproduces bit for bit.
    let values = spread(1024, -100.0, 100.0);
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Div,
        DType::F64,
        ScalarSide::Right,
        &values,
        &[3.0],
        1024,
    );
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        let expected = value / 3.0;
        assert!(
            same(got, expected),
            "DIV right[{index}]: {got:?} != {expected:?}"
        );
    }
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Div,
        DType::F64,
        ScalarSide::Left,
        &[7.0],
        &values,
        1024,
    );
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        let expected = 7.0 / value;
        assert!(
            same(got, expected),
            "DIV left[{index}]: {got:?} != {expected:?}"
        );
    }
}

#[test]
fn binary_mul_covers_scalar_sides_wrapping_and_in_place() {
    let session = Session::new("elementwise");

    // Scalar on the right, on a domain above one block.
    let values = spread(BIG, -10.0, 10.0);
    let (got, flag): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::F64,
        ScalarSide::Right,
        &values,
        &[2.5],
        BIG as i64,
    );
    assert_eq!(flag, 0, "MUL does not touch err");
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        assert!(same(got, value * 2.5), "MUL right[{index}]: {got:?}");
    }
    // Scalar on the left.
    let (got, _): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::F64,
        ScalarSide::Left,
        &[-2.5],
        &values,
        BIG as i64,
    );
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        assert!(same(got, -2.5 * value), "MUL left[{index}]: {got:?}");
    }

    // In place: `out` is the `lhs` allocation, so each thread reads and writes one
    // element of the same buffer.
    let scale = written(&session, as_bytes(&[2.5f64]));
    let buffer = upload(&session, &values);
    let err = written(&session, &[0u8; 4]);
    let plan = plan_binary(BinaryOp::Mul, DType::F64, ScalarSide::Right, BIG as i64).unwrap();
    let mut slots = [
        Slot::ptr(&buffer),
        Slot::ptr(&buffer),
        Slot::ptr(&scale),
        Slot::i32(DType::F64.tag()),
        Slot::i32(BinaryOp::Mul.tag()),
        Slot::i32(ScalarSide::Right.tag()),
        Slot::i64(BIG as i64),
        Slot::ptr(&err),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_binary"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 1);
    let got: Vec<f64> = from_bytes(&read_bytes(&session, &buffer));
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        assert!(same(got, value * 2.5), "in-place MUL[{index}]: {got:?}");
    }

    // Integer multiplication wraps.
    let lhs = vec![i32::MIN, i32::MAX, 0x4000_0000, -1];
    let (got, _): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::I32,
        ScalarSide::Left,
        &[3i32],
        &lhs,
        4,
    );
    let expected: Vec<i32> = lhs.iter().map(|&value| 3i32.wrapping_mul(value)).collect();
    assert_eq!(got, expected);
    let longs = vec![i64::MIN, i64::MAX, -1];
    let (got, _): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::I64,
        ScalarSide::Right,
        &longs,
        &[-1i64],
        3,
    );
    let expected: Vec<i64> = longs.iter().map(|&value| value.wrapping_mul(-1)).collect();
    assert_eq!(got, expected);
    assert_eq!(got[0], i64::MIN, "INT_MIN * -1 wraps");
}

/// `MUL` for C32/C64 with `scalar_side` 1 and 2: the scalar operand is read at
/// index 0 and the other operand elementwise.
#[test]
fn binary_complex_mul_honours_the_scalar_side() {
    let session = Session::new("elementwise");
    let values = vec![
        C64 { re: 1.0, im: 2.0 },
        C64 { re: -3.0, im: 0.0 },
        C64 { re: 0.0, im: -1.0 },
    ];
    let scalar = C64 { re: 2.0, im: -1.0 };
    // `expected` is the component form of `value * scalar`.
    let reference = |value: C64| C64 {
        re: value.re * scalar.re - value.im * scalar.im,
        im: value.re * scalar.im + value.im * scalar.re,
    };

    let (got, _): (Vec<C64>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::C64,
        ScalarSide::Right,
        &values,
        &[scalar],
        values.len() as i64,
    );
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        let expected = reference(value);
        assert!(
            same(got.re, expected.re) && same(got.im, expected.im),
            "complex MUL right[{index}]: got {got:?}, expected {expected:?} for {value:?}"
        );
    }

    let (got, _): (Vec<C64>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::C64,
        ScalarSide::Left,
        &[scalar],
        &values,
        values.len() as i64,
    );
    for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
        let expected = reference(value);
        assert!(
            same(got.re, expected.re) && same(got.im, expected.im),
            "complex MUL left[{index}]: got {got:?}, expected {expected:?} for {value:?}"
        );
    }

    let values32 = [C32 { re: 1.0, im: 2.0 }, C32 { re: -3.0, im: 0.0 }];
    let scalar32 = C32 { re: 2.0, im: -1.0 };
    let (got, _): (Vec<C32>, u32) = binary_run(
        &session,
        BinaryOp::Mul,
        DType::C32,
        ScalarSide::Right,
        &values32,
        &[scalar32],
        2,
    );
    for (index, (&got, &value)) in got.iter().zip(values32.iter()).enumerate() {
        let expected = C32 {
            re: value.re * scalar32.re - value.im * scalar32.im,
            im: value.re * scalar32.im + value.im * scalar32.re,
        };
        assert!(
            same(f64::from(got.re), f64::from(expected.re))
                && same(f64::from(got.im), f64::from(expected.im)),
            "complex MUL right C32[{index}]: got {got:?}, expected {expected:?}"
        );
    }
}

#[test]
fn checked_integer_ops_report_rejections_through_the_flag() {
    let session = Session::new("elementwise");

    // DIV_CHECKED: INT_MIN / -1 wraps, y == 0 rejects and zeroes its output.
    let (got, flag): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::DivChecked,
        DType::I32,
        ScalarSide::Both,
        &[i32::MIN, 7, -7],
        &[-1, 0, 2],
        3,
    );
    assert_eq!(got, vec![i32::MIN, 0, -3]);
    assert_eq!(flag, 1, "the zero divisor must set err");
    // A valid run leaves the flag clear.
    let (got, flag): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::DivChecked,
        DType::I64,
        ScalarSide::Right,
        &[9i64, -9],
        &[3i64],
        2,
    );
    assert_eq!(got, vec![3, -3]);
    assert_eq!(flag, 0, "a valid checked run leaves err clear");
    let (got, flag): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::DivChecked,
        DType::I64,
        ScalarSide::Left,
        &[i64::MIN],
        &[-1, 2],
        2,
    );
    assert_eq!(got, vec![i64::MIN, i64::MIN / 2]);
    assert_eq!(flag, 0);

    // REM_CHECKED: y == -1 yields zero, y == 0 rejects.
    let (got, flag): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::RemChecked,
        DType::I32,
        ScalarSide::Both,
        &[7, i32::MIN, 9],
        &[-1, -1, 0],
        3,
    );
    assert_eq!(got, vec![0, 0, 0]);
    assert_eq!(flag, 1, "the zero divisor must set err");
    let (got, flag): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::RemChecked,
        DType::I32,
        ScalarSide::Both,
        &[7, -7, 9],
        &[2, 2, 4],
        3,
    );
    assert_eq!(got, vec![1, -1, 1]);
    assert_eq!(flag, 0);
    let (got, flag): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::RemChecked,
        DType::I64,
        ScalarSide::Right,
        &[i64::MIN, 5],
        &[-1],
        2,
    );
    assert_eq!(got, vec![0, 0]);
    assert_eq!(flag, 0, "y == -1 is an explicit arm, not a rejection");

    // POW_CHECKED: 0^0 == 1, square-and-multiply wraps, a negative exponent rejects.
    let (got, flag): (Vec<i32>, u32) = binary_run(
        &session,
        BinaryOp::PowChecked,
        DType::I32,
        ScalarSide::Both,
        &[0, 2, 3, 0x4000_0000],
        &[0, 10, 3, 2],
        4,
    );
    assert_eq!(got, vec![1, 1024, 27, 0], "0x40000000^2 wraps to zero");
    assert_eq!(flag, 0);
    let (got, flag): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::PowChecked,
        DType::I64,
        ScalarSide::Left,
        &[-2i64],
        &[3i64, 0, -1],
        3,
    );
    assert_eq!(got, vec![-8, 1, 0]);
    assert_eq!(flag, 1, "a negative exponent must set err");
    let (got, flag): (Vec<i64>, u32) = binary_run(
        &session,
        BinaryOp::PowChecked,
        DType::I64,
        ScalarSide::Both,
        &[7i64, 2],
        &[2i64, 40],
        2,
    );
    assert_eq!(got, vec![49, 1_099_511_627_776]);
    assert_eq!(flag, 0);
}

#[test]
fn complex_division_is_scale_robust() {
    let session = Session::new("elementwise");

    // len counts complex elements; the buffers hold 2 * len components.
    let lhs = vec![
        C64 { re: 2.0, im: 0.0 },
        C64 {
            re: 1e-320,
            im: 1e-320,
        },
        C64 { re: 3.0, im: 4.0 },
        C64 { re: 3.0, im: -4.0 },
        C64 {
            re: f64::NAN,
            im: 1.0,
        },
        C64 { re: 1.0, im: 1.0 },
        C64 {
            re: 1e-320,
            im: 0.0,
        },
        C64 {
            re: 1e308,
            im: 1e308,
        },
    ];
    let rhs = vec![
        C64 { re: 1e38, im: 1e38 },
        C64 {
            re: 1e-320,
            im: 1e-320,
        },
        C64 {
            re: f64::INFINITY,
            im: 0.0,
        },
        C64 {
            re: f64::NEG_INFINITY,
            im: f64::INFINITY,
        },
        C64 { re: 2.0, im: 3.0 },
        C64 {
            re: f64::NAN,
            im: 0.0,
        },
        C64 { re: 1.0, im: 0.0 },
        C64 {
            re: 1e300,
            im: -1e300,
        },
    ];
    let len = lhs.len() as i64;
    let (got, flag): (Vec<C64>, u32) = binary_run(
        &session,
        BinaryOp::DivComplex,
        DType::C64,
        ScalarSide::Both,
        &lhs,
        &rhs,
        len,
    );
    assert_eq!(flag, 0, "DIV_COMPLEX does not touch err");

    for (index, ((&got, &lhs), &rhs)) in got.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
        let expected = cdiv64(lhs.re, lhs.im, rhs.re, rhs.im);
        assert!(
            relative(got.re, expected.0, 1e-13) && relative(got.im, expected.1, 1e-13),
            "DIV_COMPLEX[{index}]: got {got:?}, expected {expected:?} for {lhs:?} / {rhs:?}"
        );
    }
    // 2 / (1e38 + 1e38 i) = 1e-38 - 1e-38 i, still finite.
    assert!(got[0].re.is_finite() && got[0].re > 0.0 && got[0].im < 0.0);
    assert!(relative(got[0].re, 1e-38, 1e-12) && relative(got[0].im, -1e-38, 1e-12));
    // A subnormal divisor that equals its numerator divides to one.
    assert!(got[1].re.is_finite() && relative(got[1].re, 1.0, 1e-13));
    // A finite numerator over an infinite divisor: signed zeros, real then -0.
    assert_eq!(got[2].re.to_bits(), 0.0f64.to_bits());
    assert_eq!(got[2].im.to_bits(), (-0.0f64).to_bits());
    assert_eq!(got[3].re.to_bits(), (-0.0f64).to_bits());
    assert_eq!(got[3].im.to_bits(), 0.0f64.to_bits());
    // NaN components in either operand produce NaN in both parts.
    assert!(got[4].re.is_nan() && got[4].im.is_nan());
    assert!(got[5].re.is_nan() && got[5].im.is_nan());
    // A subnormal numerator over one divides to that subnormal.
    assert_eq!(got[6].re.to_bits(), 1e-320f64.to_bits());
    assert!(got[7].re.is_finite() && got[7].im.is_finite());

    // The same extreme shapes at single precision.
    let lhs = vec![
        C32 { re: 2.0, im: 0.0 },
        C32 {
            re: 1e-42,
            im: 1e-42,
        },
        C32 { re: 3.0, im: 4.0 },
        C32 {
            re: 1.0,
            im: f32::NAN,
        },
    ];
    let rhs = vec![
        C32 { re: 1e30, im: 1e30 },
        C32 {
            re: 1e-42,
            im: 1e-42,
        },
        C32 {
            re: f32::INFINITY,
            im: 0.0,
        },
        C32 { re: 2.0, im: 0.0 },
    ];
    let (got, _): (Vec<C32>, u32) = binary_run(
        &session,
        BinaryOp::DivComplex,
        DType::C32,
        ScalarSide::Both,
        &lhs,
        &rhs,
        4,
    );
    for (index, ((&got, &lhs), &rhs)) in got.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
        let expected = cdiv32(lhs.re, lhs.im, rhs.re, rhs.im);
        assert!(
            relative(f64::from(got.re), f64::from(expected.0), 1e-5)
                && relative(f64::from(got.im), f64::from(expected.1), 1e-5),
            "DIV_COMPLEX f32[{index}]: got {got:?}, expected {expected:?}"
        );
    }
    assert!(relative(f64::from(got[0].re), 1e-30, 1e-6));
    assert!(relative(f64::from(got[1].re), 1.0, 1e-6));
    assert_eq!(got[2].re.to_bits(), 0.0f32.to_bits());
    assert_eq!(got[2].im.to_bits(), (-0.0f32).to_bits());
    assert!(got[3].re.is_nan() && got[3].im.is_nan());
}

#[test]
fn mixed_real_complex_keeps_the_promoted_component_forms() {
    let session = Session::new("elementwise");

    // Components that force the zero cross terms to decide the result: 0 * inf is
    // NaN, so a simplified form would pass where the promoted form must not.
    let values = vec![
        C64 { re: 1.0, im: 0.0 },
        C64 {
            re: 1.0,
            im: f64::INFINITY,
        },
        C64 {
            re: f64::INFINITY,
            im: 1.0,
        },
        C64 { re: -0.0, im: 0.0 },
        C64 {
            re: f64::NAN,
            im: 2.0,
        },
        C64 { re: 3.0, im: -4.0 },
    ];
    let len = values.len() as i64;
    for op in [BinaryOp::MixedAdd, BinaryOp::MixedSub, BinaryOp::MixedMul] {
        for side in [ScalarSide::Left, ScalarSide::Right] {
            let got = mixed64_run(&session, op, side, 2.0, &values, len);
            for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
                let (want_re, want_im) = mixed_reference(op, side, 2.0, value);
                assert!(
                    same_bits64(got.re, want_re) && same_bits64(got.im, want_im),
                    "{op:?} {side:?}[{index}]: got {got:?}, expected ({want_re:?}, {want_im:?}) for {value:?}"
                );
            }
        }
    }
    // The single-precision route keeps the same component forms.
    let values32 = vec![
        C32 { re: 1.0, im: 0.0 },
        C32 {
            re: 0.0,
            im: f32::INFINITY,
        },
        C32 { re: -2.5, im: 0.5 },
    ];
    let got = mixed32_run(
        &session,
        BinaryOp::MixedMul,
        ScalarSide::Left,
        1.5,
        &values32,
        3,
    );
    for (index, (&got, &value)) in got.iter().zip(values32.iter()).enumerate() {
        let (want_re, want_im) = mixed_reference(
            BinaryOp::MixedMul,
            ScalarSide::Left,
            1.5,
            C64 {
                re: f64::from(value.re),
                im: f64::from(value.im),
            },
        );
        assert!(
            same_bits32(got.re, want_re as f32) && same_bits32(got.im, want_im as f32),
            "MIXED_MUL f32[{index}]: got {got:?}, expected ({want_re:?}, {want_im:?})"
        );
    }

    // MIXED_DIV on the single-precision route goes through the same robust helper.
    let values32 = vec![
        C32 { re: 1e38, im: 1e38 },
        C32 {
            re: 1e-30,
            im: -1e-30,
        },
        C32 { re: -0.0, im: 0.0 },
    ];
    for side in [ScalarSide::Left, ScalarSide::Right] {
        let got = mixed32_run(&session, BinaryOp::MixedDiv, side, 2.0, &values32, 3);
        for (index, (&got, &value)) in got.iter().zip(values32.iter()).enumerate() {
            let expected = match side {
                ScalarSide::Left => cdiv32(2.0, 0.0, value.re, value.im),
                _ => cdiv32(value.re, value.im, 2.0, 0.0),
            };
            assert!(
                relative(f64::from(got.re), f64::from(expected.0), 1e-4)
                    && relative(f64::from(got.im), f64::from(expected.1), 1e-4),
                "MIXED_DIV f32 {side:?}[{index}]: got {got:?}, expected {expected:?}"
            );
        }
    }

    // MIXED_DIV promotes the real operand to (x, +0) and divides robustly.
    let values = vec![
        C64 { re: 1e38, im: 1e38 },
        C64 {
            re: 1e-320,
            im: 1e-320,
        },
        C64 {
            re: f64::INFINITY,
            im: 0.0,
        },
        C64 {
            re: f64::NAN,
            im: 1.0,
        },
        C64 { re: 3.0, im: -4.0 },
        C64 { re: -6.0, im: 8.0 },
    ];
    let len = values.len() as i64;
    for side in [ScalarSide::Left, ScalarSide::Right] {
        let got = mixed64_run(&session, BinaryOp::MixedDiv, side, 2.0, &values, len);
        for (index, (&got, &value)) in got.iter().zip(values.iter()).enumerate() {
            let expected = match side {
                ScalarSide::Left => cdiv64(2.0, 0.0, value.re, value.im),
                _ => cdiv64(value.re, value.im, 2.0, 0.0),
            };
            assert!(
                relative(got.re, expected.0, 1e-13) && relative(got.im, expected.1, 1e-13),
                "MIXED_DIV {side:?}[{index}]: got {got:?}, expected {expected:?} for {value:?}"
            );
        }
    }
    let got = mixed64_run(
        &session,
        BinaryOp::MixedDiv,
        ScalarSide::Left,
        2.0,
        &values,
        len,
    );
    assert!(got[0].re.is_finite() && got[0].im < 0.0);
    assert!(
        got[1].re.is_infinite() && got[1].im.is_infinite(),
        "2 / (1e-320 + 1e-320 i) overflows rather than going NaN"
    );
    assert_eq!(got[2].re.to_bits(), 0.0f64.to_bits());
    assert_eq!(got[2].im.to_bits(), (-0.0f64).to_bits());
    assert!(got[3].re.is_nan() && got[3].im.is_nan());
    // A subnormal real numerator over a subnormal complex divisor is 1 / (1 + i).
    let got = mixed64_run(
        &session,
        BinaryOp::MixedDiv,
        ScalarSide::Left,
        1e-320,
        &values,
        len,
    );
    assert!(
        relative(got[1].re, 0.5, 1e-13) && relative(got[1].im, -0.5, 1e-13),
        "subnormal MIXED_DIV: got {:?}",
        got[1]
    );
}

// --- comparison, selection and clamping -------------------------------------

fn compare_reference<T: PartialEq + PartialOrd>(cmp: CompareOp, lhs: &[T], rhs: &[T]) -> Vec<u8> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| {
            let predicate = match cmp {
                CompareOp::Eq => x == y,
                CompareOp::Lt => x < y,
                CompareOp::Le => x <= y,
                CompareOp::Gt => x > y,
                CompareOp::Ge => x >= y,
            };
            predicate as u8
        })
        .collect()
}

fn clamp_reference<T: PartialOrd + Copy>(input: &[T], lower: &[T], upper: &[T]) -> Vec<T> {
    input
        .iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|((&value, &low), &high)| {
            if value < low {
                low
            } else if value > high {
                high
            } else {
                value
            }
        })
        .collect()
}

#[test]
fn compare_select_and_clamp_follow_their_ordered_rules() {
    let session = Session::new("elementwise");

    let cmps = [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ];
    // Float comparison is unordered on NaN, ordered on signed zeros and infinities.
    let lhs = vec![
        1.0f32,
        -1.0,
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        f32::NAN,
    ];
    let rhs = vec![
        1.0f32,
        1.0,
        -0.0,
        0.0,
        1.0,
        f32::INFINITY,
        f32::INFINITY,
        -2.0,
        f32::NAN,
    ];
    for cmp in cmps {
        let bytes = compare_run(&session, cmp, DType::F32, &lhs, &rhs);
        let expected = compare_reference(cmp, &lhs, &rhs);
        assert_eq!(&bytes[..lhs.len()], &expected[..], "compare {cmp:?} f32");
        assert_guard(&bytes, lhs.len(), "compare f32");
    }
    let lhs64 = vec![1.0f64, f64::NAN, f64::INFINITY, -0.0, 2.0];
    let rhs64 = vec![-1.0f64, f64::NAN, f64::MAX, 0.0, 2.0];
    for cmp in cmps {
        let bytes = compare_run(&session, cmp, DType::F64, &lhs64, &rhs64);
        let expected = compare_reference(cmp, &lhs64, &rhs64);
        assert_eq!(&bytes[..lhs64.len()], &expected[..], "compare {cmp:?} f64");
        assert_guard(&bytes, lhs64.len(), "compare f64");
    }
    let lhs32 = vec![i32::MIN, -1, 0, 1, i32::MAX];
    let rhs32 = vec![i32::MAX, 1, 0, -1, i32::MIN];
    for cmp in cmps {
        let bytes = compare_run(&session, cmp, DType::I32, &lhs32, &rhs32);
        let expected = compare_reference(cmp, &lhs32, &rhs32);
        assert_eq!(&bytes[..lhs32.len()], &expected[..], "compare {cmp:?} i32");
    }
    let lhs64 = vec![i64::MIN, -1, 0, 1, i64::MAX];
    for cmp in cmps {
        let bytes = compare_run(
            &session,
            cmp,
            DType::I64,
            &lhs64,
            &rhs32.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        );
        let expected = compare_reference(
            cmp,
            &lhs64,
            &rhs32.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        );
        assert_eq!(&bytes[..lhs64.len()], &expected[..], "compare {cmp:?} i64");
    }

    // SELECT: any nonzero predicate byte selects `on_true`.
    let pred = [1u8, 0, 2, 0, 255];
    let on_true = vec![1.5f32, -2.0, f32::NAN, 4.0, -0.0];
    let on_false = vec![-1.0f32, 2.0, 3.0, f32::NEG_INFINITY, 0.0];
    let bytes = select_run(&session, DType::F32, &pred, &on_true, &on_false);
    let got: Vec<f32> = from_bytes(&bytes[..pred.len() * 4]);
    for (index, &got) in got.iter().enumerate() {
        let expected = if pred[index] != 0 {
            on_true[index]
        } else {
            on_false[index]
        };
        assert!(
            same(f64::from(got), f64::from(expected)),
            "select f32[{index}]: {got:?}"
        );
    }
    assert_guard(&bytes, pred.len() * 4, "select f32");
    let on_true = vec![1.5f64, -2.0, f64::INFINITY, 4.0, -0.0];
    let on_false = vec![-1.0f64, 2.0, 3.0, f64::NAN, 0.0];
    let bytes = select_run(&session, DType::F64, &pred, &on_true, &on_false);
    let got: Vec<f64> = from_bytes(&bytes[..pred.len() * 8]);
    for (index, &got) in got.iter().enumerate() {
        let expected = if pred[index] != 0 {
            on_true[index]
        } else {
            on_false[index]
        };
        assert!(same(got, expected), "select f64[{index}]: {got:?}");
    }
    let on_true = vec![1i32, -2, i32::MIN, 4, 0];
    let on_false = vec![-1i32, 2, 3, i32::MAX, -0];
    let bytes = select_run(&session, DType::I32, &pred, &on_true, &on_false);
    let got: Vec<i32> = from_bytes(&bytes[..pred.len() * 4]);
    for (index, &got) in got.iter().enumerate() {
        let expected = if pred[index] != 0 {
            on_true[index]
        } else {
            on_false[index]
        };
        assert_eq!(got, expected, "select i32[{index}]");
    }
    let on_true = vec![1i64, -2, i64::MIN, 4, 0];
    let on_false = vec![-1i64, 2, 3, i64::MAX, 0];
    let bytes = select_run(&session, DType::I64, &pred, &on_true, &on_false);
    let got: Vec<i64> = from_bytes(&bytes[..pred.len() * 8]);
    for (index, &got) in got.iter().enumerate() {
        let expected = if pred[index] != 0 {
            on_true[index]
        } else {
            on_false[index]
        };
        assert_eq!(got, expected, "select i64[{index}]");
    }

    // CLAMP is the ordered expression x < lo ? lo : (x > hi ? hi : x): a NaN input
    // propagates, a NaN bound disables its comparison, inverted bounds clamp by
    // statement order, and a signed zero survives when it is in range.
    let input = [0.0f64, -0.0, 1.0, -5.0, 5.0, f64::NAN, 2.0, 10.0, 3.0, 10.0];
    let lower = [
        -1.0f64,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        f64::NAN,
        5.0,
        5.0,
        -1.0,
    ];
    let upper = [1.0f64, 1.0, 3.0, 1.0, 3.0, 1.0, 9.0, 1.0, 1.0, f64::NAN];
    let bytes = clamp_run(&session, DType::F64, &input, &lower, &upper);
    let got: Vec<f64> = from_bytes(&bytes[..input.len() * 8]);
    let expected = clamp_reference(&input, &lower, &upper);
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == expected.to_bits(),
            "clamp[{index}]: got {got:?}, expected {expected:?}"
        );
    }
    assert_guard(&bytes, input.len() * 8, "clamp f64");
    assert_eq!(
        got[1].to_bits(),
        (-0.0f64).to_bits(),
        "an in-range signed zero survives"
    );
    assert!(got[5].is_nan(), "a NaN input propagates");
    assert_eq!(got[6], 2.0, "a NaN lower bound disables that comparison");
    assert_eq!(got[7], 1.0, "inverted bounds clamp by statement order");
    assert_eq!(got[8], 5.0, "an inverted lower bound wins while x < lo");
    assert_eq!(got[9], 10.0, "a NaN upper bound disables that comparison");

    let input32: Vec<f32> = input.iter().map(|&value| value as f32).collect();
    let lower32: Vec<f32> = lower.iter().map(|&value| value as f32).collect();
    let upper32: Vec<f32> = upper.iter().map(|&value| value as f32).collect();
    let bytes = clamp_run(&session, DType::F32, &input32, &lower32, &upper32);
    let got: Vec<f32> = from_bytes(&bytes[..input.len() * 4]);
    let expected = clamp_reference(&input32, &lower32, &upper32);
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == expected.to_bits(),
            "clamp f32[{index}]: got {got:?}, expected {expected:?}"
        );
    }
    assert_guard(&bytes, input.len() * 4, "clamp f32");

    // A domain above one block: a body that only ran block 0 would pass the small
    // fixtures above and fail here.
    let lhs: Vec<f32> = (0..BIG).map(|index| index as f32 * 0.5 - 7.0).collect();
    let rhs: Vec<f32> = (0..BIG).map(|index| index as f32 * 0.25).collect();
    for cmp in cmps {
        let bytes = compare_run(&session, cmp, DType::F32, &lhs, &rhs);
        assert_eq!(
            &bytes[..BIG],
            &compare_reference(cmp, &lhs, &rhs)[..],
            "compare {cmp:?} f32 large"
        );
        assert_guard(&bytes, BIG, "compare f32 large");
    }
    let pred: Vec<u8> = (0..BIG).map(|index| u8::from(index % 3 == 0)).collect();
    let on_true: Vec<f64> = (0..BIG).map(|index| index as f64).collect();
    let on_false: Vec<f64> = (0..BIG).map(|index| -(index as f64)).collect();
    let bytes = select_run(&session, DType::F64, &pred, &on_true, &on_false);
    let got: Vec<f64> = from_bytes(&bytes[..BIG * 8]);
    for (index, &got) in got.iter().enumerate() {
        let expected = if pred[index] != 0 {
            on_true[index]
        } else {
            on_false[index]
        };
        assert_eq!(got, expected, "select f64 large[{index}]");
    }
    assert_guard(&bytes, BIG * 8, "select f64 large");
    let input: Vec<f32> = (0..BIG).map(|index| index as f32 * 0.01 - 5.0).collect();
    let lower = vec![-1.0f32; BIG];
    let upper: Vec<f32> = (0..BIG).map(|index| index as f32 * 0.001).collect();
    let bytes = clamp_run(&session, DType::F32, &input, &lower, &upper);
    let got: Vec<f32> = from_bytes(&bytes[..BIG * 4]);
    let expected = clamp_reference(&input, &lower, &upper);
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "clamp f32 large[{index}]"
        );
    }
    assert_guard(&bytes, BIG * 4, "clamp f32 large");
}

// --- strided combination ----------------------------------------------------

#[test]
fn combined_strided_covers_broadcast_negative_stride_and_in_place() {
    let session = Session::new("elementwise");

    // The baseline broadcast multiply composed with a reverse: `a` is a stride-0
    // broadcast and `b` a negative stride, over a domain above one block.
    let out = Layout::compact(&[BIG as i64]).unwrap();
    let a_layout = Layout::new(&[BIG as i64], &[0], 0).unwrap();
    let b_layout = Layout::new(&[BIG as i64], &[-1], BIG as i64 - 1).unwrap();
    let a_values = [2.0f64];
    let b_values: Vec<f64> = (0..BIG).map(|index| index as f64).collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::F64,
        &out,
        &a_layout,
        &b_layout,
        as_bytes(&a_values),
        as_bytes(&b_values),
        None,
    );
    let got: Vec<f64> = from_bytes(&bytes);
    for (index, &got) in got.iter().enumerate() {
        let expected = 2.0 * b_values[BIG - 1 - index];
        assert_eq!(got, expected, "broadcast * reverse[{index}]");
    }

    // Rank three: a broadcast axis, and a negative stride on `b`.
    let out = Layout::new(&[3, 4, 2], &[1, 3, 12], 0).unwrap();
    let a_layout = Layout::new(&[3, 4, 2], &[1, 0, 3], 0).unwrap();
    let b_layout = Layout::new(&[3, 4, 2], &[-1, 3, 12], 2).unwrap();
    let a_values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_values: Vec<f64> = (1..=24).map(|value| value as f64).collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::F64,
        &out,
        &a_layout,
        &b_layout,
        as_bytes(&a_values),
        as_bytes(&b_values),
        None,
    );
    let got: Vec<f64> = from_bytes(&bytes);
    let a_indices = view_indices(&[3, 4, 2], &a_layout.strides().to_vec(), a_layout.offset());
    let b_indices = view_indices(&[3, 4, 2], &b_layout.strides().to_vec(), b_layout.offset());
    for index in 0..24 {
        let expected = a_values[a_indices[index] as usize] * b_values[b_indices[index] as usize];
        assert_eq!(got[index], expected, "rank-3 strided MUL[{index}]");
    }

    // AXPBY in place on `b` with identical layouts, coefficients (2, -1).
    let out = Layout::compact(&[256, 4]).unwrap();
    let a_layout = Layout::new(&[256, 4], &[0, 1], 0).unwrap();
    let a_values = [1.5f64, -2.0, 3.0, 0.25];
    let b_values: Vec<f64> = (0..1024).map(|index| index as f64 * 0.5).collect();
    let a_buf = upload(&session, &a_values);
    let b_buf = upload(&session, &b_values);
    let coefficients = written(&session, as_bytes(&[2.0f64, -1.0]));
    let plan = plan_combined_strided(
        StridedOp::Axpby,
        DType::F64,
        &out,
        1024,
        &a_layout,
        4,
        &out,
        1024,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&b_buf),
        Slot::ptr(&a_buf),
        Slot::ptr(&b_buf),
        Slot::ptr(&coefficients),
        Slot::layout(&CLayout::new(&out)),
        Slot::layout(&CLayout::new(&a_layout)),
        Slot::layout(&CLayout::new(&out)),
        Slot::i32(DType::F64.tag()),
        Slot::i32(StridedOp::Axpby.tag()),
        Slot::i64(1024),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_combined_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 1);
    let got: Vec<f64> = from_bytes(&read_bytes(&session, &b_buf));
    for index in 0..1024 {
        let expected = 2.0 * a_values[index / 256] + -1.0 * b_values[index];
        assert_eq!(got[index], expected, "in-place AXPBY[{index}]");
    }

    // Integer MUL wraps.
    let out = Layout::compact(&[4, 3]).unwrap();
    let a_layout = Layout::new(&[4, 3], &[1, 0], 0).unwrap();
    let a_values = [0x4000_0000i32, -1, i32::MIN, 3];
    let b_values: Vec<i32> = (0..12)
        .map(|index| (index as i32).wrapping_mul(0x1000_0000))
        .collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::I32,
        &out,
        &a_layout,
        &out,
        as_bytes(&a_values),
        as_bytes(&b_values),
        None,
    );
    let got: Vec<i32> = from_bytes(&bytes);
    for index in 0..12 {
        let expected = a_values[index % 4].wrapping_mul(b_values[index]);
        assert_eq!(got[index], expected, "wrapping strided MUL[{index}]");
    }

    // Complex MUL and AXPBY use the ordinary component forms.
    let out = Layout::compact(&[4, 3]).unwrap();
    let a_layout = Layout::new(&[4, 3], &[1, 0], 0).unwrap();
    let a_values = [
        C64 { re: 1.0, im: 2.0 },
        C64 { re: -1.0, im: 0.5 },
        C64 { re: 0.0, im: -0.0 },
        C64 { re: 3.0, im: 1.0 },
    ];
    let b_values: Vec<C64> = (0..12)
        .map(|index| C64 {
            re: index as f64 - 6.0,
            im: 0.5 * index as f64,
        })
        .collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::C64,
        &out,
        &a_layout,
        &out,
        as_bytes(&a_values),
        as_bytes(&b_values),
        None,
    );
    let got: Vec<C64> = from_bytes(&bytes);
    for index in 0..12 {
        let left = a_values[index % 4];
        let right = b_values[index];
        let expected = C64 {
            re: left.re * right.re - left.im * right.im,
            im: left.re * right.im + left.im * right.re,
        };
        assert!(
            same(got[index].re, expected.re) && same(got[index].im, expected.im),
            "complex strided MUL[{index}]: got {:?}, expected {expected:?}",
            got[index]
        );
    }
    let coefficients = [C64 { re: 2.0, im: 0.0 }, C64 { re: -1.0, im: 1.0 }];
    let bytes = combined_run(
        &session,
        StridedOp::Axpby,
        DType::C64,
        &out,
        &a_layout,
        &out,
        as_bytes(&a_values),
        as_bytes(&b_values),
        Some(as_bytes(&coefficients)),
    );
    let got: Vec<C64> = from_bytes(&bytes);
    for index in 0..12 {
        let left = a_values[index % 4];
        let right = b_values[index];
        let first = C64 {
            re: coefficients[0].re * left.re - coefficients[0].im * left.im,
            im: coefficients[0].re * left.im + coefficients[0].im * left.re,
        };
        let second = C64 {
            re: coefficients[1].re * right.re - coefficients[1].im * right.im,
            im: coefficients[1].re * right.im + coefficients[1].im * right.re,
        };
        let expected = C64 {
            re: first.re + second.re,
            im: first.im + second.im,
        };
        assert!(
            same(got[index].re, expected.re) && same(got[index].im, expected.im),
            "complex AXPBY[{index}]: got {:?}, expected {expected:?}",
            got[index]
        );
    }

    // F32, I64 and C32 multiply, so every documented dtype route is exercised.
    let out = Layout::compact(&[3, 4]).unwrap();
    let a_layout = Layout::new(&[3, 4], &[1, 0], 0).unwrap();
    let a_values = [0.5f32, -2.0, 1e30];
    let b_values: Vec<f32> = (0..12).map(|index| index as f32 * 0.5).collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::F32,
        &out,
        &a_layout,
        &out,
        as_bytes(&a_values),
        as_bytes(&b_values),
        None,
    );
    let got: Vec<f32> = from_bytes(&bytes);
    for index in 0..12 {
        let expected = a_values[index % 3] * b_values[index];
        assert_eq!(got[index].to_bits(), expected.to_bits(), "f32 MUL[{index}]");
    }

    let longs: Vec<i64> = vec![i64::MAX / 2 + 1, -3, i64::MIN, 7, i64::MAX];
    let b_longs: Vec<i64> = (0..12).map(|index| index as i64 - 4).collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::I64,
        &out,
        &a_layout,
        &out,
        as_bytes(&longs),
        as_bytes(&b_longs),
        None,
    );
    let got: Vec<i64> = from_bytes(&bytes);
    for index in 0..12 {
        let expected = longs[index % 3].wrapping_mul(b_longs[index]);
        assert_eq!(got[index], expected, "i64 wrapping MUL[{index}]");
    }

    let a_c32 = [
        C32 { re: 1.0, im: -1.0 },
        C32 { re: 0.0, im: 0.0 },
        C32 { re: -0.0, im: 2.0 },
    ];
    let b_c32: Vec<C32> = (0..12)
        .map(|index| C32 {
            re: index as f32 - 2.0,
            im: 0.5,
        })
        .collect();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::C32,
        &out,
        &a_layout,
        &out,
        as_bytes(&a_c32),
        as_bytes(&b_c32),
        None,
    );
    let got: Vec<C32> = from_bytes(&bytes);
    for index in 0..12 {
        let left = a_c32[index % 3];
        let right = b_c32[index];
        let expected = C32 {
            re: left.re * right.re - left.im * right.im,
            im: left.re * right.im + left.im * right.re,
        };
        assert!(
            same_bits32(got[index].re, expected.re) && same_bits32(got[index].im, expected.im),
            "c32 MUL[{index}]: got {:?}, expected {expected:?}",
            got[index]
        );
    }

    // AXPBY on F32 and C32 with a strided output and a shifted `b`, so a body that
    // assumed compact operands would write the wrong elements.
    let out_layout = Layout::new(&[2, 3], &[1, 2], 4).unwrap();
    let a_layout = Layout::new(&[2, 3], &[1, 0], 0).unwrap();
    let b_layout = Layout::new(&[2, 3], &[1, 2], 1).unwrap();
    let out_len = 10usize;
    let b_len = 8usize;
    let a_values = [2.0f32, -3.0];
    let b_values: Vec<f32> = (0..b_len).map(|index| index as f32).collect();
    let coefficients = [1.5f32, -0.5];
    let a_buf = upload(&session, &a_values);
    let b_buf = upload(&session, &b_values);
    let coefficient_buf = upload(&session, &coefficients);
    let out_buf = sentinel(&session, out_len * 4);
    let plan = plan_combined_strided(
        StridedOp::Axpby,
        DType::F32,
        &out_layout,
        out_len as i64,
        &a_layout,
        a_values.len() as i64,
        &b_layout,
        b_len as i64,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&out_buf),
        Slot::ptr(&a_buf),
        Slot::ptr(&b_buf),
        Slot::ptr(&coefficient_buf),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&a_layout)),
        Slot::layout(&CLayout::new(&b_layout)),
        Slot::i32(DType::F32.tag()),
        Slot::i32(StridedOp::Axpby.tag()),
        Slot::i64(6),
    ];
    launch(
        session.function("t4a_cuda_combined_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let got: Vec<f32> = from_bytes(&read_bytes(&session, &out_buf));
    for col in 0..3 {
        for row in 0..2 {
            let expected =
                coefficients[0] * a_values[row] + coefficients[1] * b_values[1 + row + 2 * col];
            assert_eq!(
                got[4 + row + 2 * col].to_bits(),
                expected.to_bits(),
                "f32 strided AXPBY[{row},{col}]"
            );
        }
    }

    // The same shape on C32, with the components checked bit for bit.
    let a_values = [C32 { re: 1.0, im: -2.0 }, C32 { re: -0.5, im: 0.0 }];
    let b_values: Vec<C32> = (0..b_len)
        .map(|index| C32 {
            re: index as f32,
            im: 0.5 * index as f32,
        })
        .collect();
    let coefficients = [C32 { re: 1.5, im: 0.0 }, C32 { re: -0.5, im: 0.0 }];
    let a_buf = upload(&session, &a_values);
    let b_buf = upload(&session, &b_values);
    let coefficient_buf = upload(&session, &coefficients);
    let out_buf = sentinel(&session, out_len * 8);
    let plan = plan_combined_strided(
        StridedOp::Axpby,
        DType::C32,
        &out_layout,
        out_len as i64,
        &a_layout,
        a_values.len() as i64,
        &b_layout,
        b_len as i64,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&out_buf),
        Slot::ptr(&a_buf),
        Slot::ptr(&b_buf),
        Slot::ptr(&coefficient_buf),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&a_layout)),
        Slot::layout(&CLayout::new(&b_layout)),
        Slot::i32(DType::C32.tag()),
        Slot::i32(StridedOp::Axpby.tag()),
        Slot::i64(6),
    ];
    launch(
        session.function("t4a_cuda_combined_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let got: Vec<C32> = from_bytes(&read_bytes(&session, &out_buf));
    for col in 0..3 {
        for row in 0..2 {
            let left = a_values[row];
            let right = b_values[1 + row + 2 * col];
            let first = C32 {
                re: coefficients[0].re * left.re - coefficients[0].im * left.im,
                im: coefficients[0].re * left.im + coefficients[0].im * left.re,
            };
            let second = C32 {
                re: coefficients[1].re * right.re - coefficients[1].im * right.im,
                im: coefficients[1].re * right.im + coefficients[1].im * right.re,
            };
            let expected = C32 {
                re: first.re + second.re,
                im: first.im + second.im,
            };
            let got = got[4 + row + 2 * col];
            assert!(
                same_bits32(got.re, expected.re) && same_bits32(got.im, expected.im),
                "c32 strided AXPBY[{row},{col}]: got {got:?}, expected {expected:?}"
            );
        }
    }
}

// --- empty and single-element domains ---------------------------------------

#[test]
fn empty_and_single_element_domains_launch_once_or_not_at_all() {
    let session = Session::new("elementwise");

    // Empty domains plan a zero grid, so the caller's launch is a no-op and the
    // guard region behind the output stays untouched. `plan.is_empty()` and the
    // harness counter both see zero physical launches.
    let empty = written(&session, &[]);
    let out = sentinel(&session, 64);
    let err = written(&session, &[0u8; 4]);
    let scratch = written(&session, &[0u8; 64]);

    let unary = plan_unary(UnaryOp::Exp, DType::F64, DType::F64, 0).unwrap();
    assert!(unary.is_empty());
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&empty),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::F64.tag()),
        Slot::i32(UnaryOp::Exp.tag()),
        Slot::i64(0),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_unary"),
        &unary,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before);

    let binary = plan_binary(BinaryOp::Mul, DType::F32, ScalarSide::Right, 0).unwrap();
    assert!(binary.is_empty());
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&empty),
        Slot::ptr(&scratch),
        Slot::i32(DType::F32.tag()),
        Slot::i32(BinaryOp::Mul.tag()),
        Slot::i32(ScalarSide::Right.tag()),
        Slot::i64(0),
        Slot::ptr(&err),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_binary"),
        &binary,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before);

    for (plan, function, mut slots) in [
        (
            plan_compare(DType::F32, 0).unwrap(),
            "t4a_cuda_compare",
            vec![
                Slot::ptr(&out),
                Slot::ptr(&empty),
                Slot::ptr(&empty),
                Slot::i32(DType::F32.tag()),
                Slot::i32(CompareOp::Eq.tag()),
                Slot::i64(0),
            ],
        ),
        (
            plan_select(DType::F32, 0).unwrap(),
            "t4a_cuda_select",
            vec![
                Slot::ptr(&out),
                Slot::ptr(&empty),
                Slot::ptr(&scratch),
                Slot::ptr(&scratch),
                Slot::i32(DType::F32.tag()),
                Slot::i64(0),
            ],
        ),
        (
            plan_clamp(DType::F32, 0).unwrap(),
            "t4a_cuda_clamp",
            vec![
                Slot::ptr(&out),
                Slot::ptr(&empty),
                Slot::ptr(&scratch),
                Slot::ptr(&scratch),
                Slot::i32(DType::F32.tag()),
                Slot::i64(0),
            ],
        ),
    ] {
        assert!(plan.is_empty(), "{function}");
        let before = launches();
        launch(
            session.function(function),
            &plan,
            session.stream(),
            &mut slots,
        );
        assert_eq!(launches(), before, "{function}");
    }

    let empty_layout = Layout::new(&[0], &[1], 0).unwrap();
    let combined = plan_combined_strided(
        StridedOp::Mul,
        DType::F32,
        &empty_layout,
        0,
        &empty_layout,
        0,
        &empty_layout,
        0,
    )
    .unwrap();
    assert!(combined.is_empty());
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&empty),
        Slot::ptr(&empty),
        Slot::ptr(&scratch),
        Slot::layout(&CLayout::new(&empty_layout)),
        Slot::layout(&CLayout::new(&empty_layout)),
        Slot::layout(&CLayout::new(&empty_layout)),
        Slot::i32(DType::F32.tag()),
        Slot::i32(StridedOp::Mul.tag()),
        Slot::i64(0),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_combined_strided"),
        &combined,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before);
    assert!(read_bytes(&session, &out)
        .iter()
        .all(|&byte| byte == SENTINEL));

    // A single-element domain launches exactly once per entry, and the guard behind
    // each output survives.
    let input = upload(&session, &[4.0f64]);
    let bytes = unary_bytes(
        &session,
        UnaryOp::Sqrt,
        DType::F64,
        DType::F64,
        &input,
        8 + GUARD,
        1,
    );
    assert_eq!(from_bytes::<f64>(&bytes[..8]), vec![2.0]);
    assert_guard(&bytes, 8, "unary singleton");

    let (got, flag): (Vec<f64>, u32) = binary_run(
        &session,
        BinaryOp::Pow,
        DType::F64,
        ScalarSide::Right,
        &[3.0],
        &[2.0],
        1,
    );
    assert_eq!(got, vec![9.0]);
    assert_eq!(flag, 0);

    let bytes = compare_run(&session, CompareOp::Lt, DType::F64, &[1.0f64], &[2.0f64]);
    assert_eq!(&bytes[..1], &[1]);
    assert_guard(&bytes, 1, "compare singleton");

    let bytes = select_run(&session, DType::I64, &[1], &[7i64], &[9i64]);
    assert_eq!(from_bytes::<i64>(&bytes[..8]), vec![7]);
    assert_guard(&bytes, 8, "select singleton");

    let bytes = clamp_run(&session, DType::F64, &[5.0f64], &[0.0f64], &[1.0f64]);
    assert_eq!(from_bytes::<f64>(&bytes[..8]), vec![1.0]);
    assert_guard(&bytes, 8, "clamp singleton");

    let layout = Layout::compact(&[1]).unwrap();
    let bytes = combined_run(
        &session,
        StridedOp::Mul,
        DType::F64,
        &layout,
        &layout,
        &layout,
        as_bytes(&[6.0f64]),
        as_bytes(&[7.0f64]),
        None,
    );
    assert_eq!(from_bytes::<f64>(&bytes), vec![42.0]);
}
