//! Numerical GPU tests for the revision-5 dtype conversion entry.
//! Run with `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{from_bytes, launch, launches, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{plan_convert, plan_validate_real_cast, DType};

const SENTINEL: u8 = 0xCD;

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

/// Run one conversion and return the raw destination bytes.
fn convert(
    session: &Session,
    dst: DType,
    src: DType,
    input: &Buf,
    len: i64,
    guard_bytes: usize,
) -> Vec<u8> {
    let plan = plan_convert(dst, src, len).unwrap();
    let out = sentinel(session, guard_bytes.max(len as usize * dst.size()));
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(input),
        Slot::i32(dst.tag()),
        Slot::i32(src.tag()),
        Slot::i64(len),
    ];
    let function = session.function("t4a_cuda_convert");
    let before = launches();
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + if plan.is_empty() { 0 } else { 1 });
    read_bytes(session, &out)
}

#[test]
fn numeric_conversions_follow_the_cast_rules() {
    let session = Session::new("convert");
    let len = 65_537i64;

    // Float rounding: F64 -> F32 nearest-even, then back to F64 exactly.
    let values: Vec<f64> = (0..len)
        .map(|index| (index as f64 - 32_768.0) * 0.1 + 1.0 / 3.0)
        .collect();
    let input = upload(&session, &values);
    let bytes = convert(&session, DType::F32, DType::F64, &input, len, 0);
    let narrowed: Vec<f32> = from_bytes(&bytes[..(len as usize) * 4]);
    for (index, (&got, &want)) in narrowed.iter().zip(values.iter()).enumerate() {
        assert_eq!(got.to_bits(), (want as f32).to_bits(), "position {index}");
    }
    let widened = upload(&session, &narrowed);
    let bytes = convert(&session, DType::F64, DType::F32, &widened, len, 0);
    let back: Vec<f64> = from_bytes(&bytes[..(len as usize) * 8]);
    for (got, want) in back.iter().zip(narrowed.iter()) {
        assert_eq!(got.to_bits(), f64::from(*want).to_bits());
    }

    // Integer narrowing keeps the low bits; widening sign-extends.
    let ints: Vec<i64> = (0..len)
        .map(|index| index.wrapping_mul(0x0123_4567_89AB_CDEF) ^ i64::MIN)
        .collect();
    let input = upload(&session, &ints);
    let bytes = convert(&session, DType::I32, DType::I64, &input, len, 0);
    let narrowed: Vec<i32> = from_bytes(&bytes[..(len as usize) * 4]);
    for (got, &want) in narrowed.iter().zip(ints.iter()) {
        assert_eq!(*got, want as i32);
    }
    let input = upload(&session, &narrowed);
    let bytes = convert(&session, DType::I64, DType::I32, &input, len, 0);
    let widened: Vec<i64> = from_bytes(&bytes[..(len as usize) * 8]);
    for (got, want) in widened.iter().zip(narrowed.iter()) {
        assert_eq!(*got, i64::from(*want));
    }

    // Float -> integer truncates toward zero, inside the validated domain.
    let floats: Vec<f64> = vec![-1.75, 1.75, 0.0, -0.5, 4.999_999, -4.999_999, 2.0];
    let input = upload(&session, &floats);
    let bytes = convert(
        &session,
        DType::I32,
        DType::F64,
        &input,
        floats.len() as i64,
        0,
    );
    let ints: Vec<i32> = from_bytes(&bytes);
    let expected: Vec<i32> = floats.iter().map(|&value| value.trunc() as i32).collect();
    assert_eq!(ints, expected);

    // Integer -> float rounds to nearest-even.
    let ints: Vec<i64> = vec![16_777_217, -16_777_217, 3, 0];
    let input = upload(&session, &ints);
    let bytes = convert(&session, DType::F32, DType::I64, &input, 4, 0);
    let floats: Vec<f32> = from_bytes(&bytes);
    let expected: Vec<f32> = ints.iter().map(|&value| value as f32).collect();
    assert_eq!(floats, expected);
}

#[test]
fn complex_and_bool_routes_follow_their_rules() {
    let session = Session::new("convert");

    // Complex -> complex casts both components.
    let values: Vec<C64> = vec![
        C64 { re: 1.5, im: -2.25 },
        C64 {
            re: f64::INFINITY,
            im: 0.0,
        },
    ];
    let input = upload(&session, &values);
    let bytes = convert(&session, DType::C32, DType::C64, &input, 2, 0);
    let narrowed: Vec<C32> = from_bytes(&bytes);
    assert_eq!(narrowed[0], C32 { re: 1.5, im: -2.25 });
    assert_eq!(narrowed[1].re, f32::INFINITY);
    assert_eq!(narrowed[1].im, 0.0);

    // Real -> complex writes a zero imaginary part, at scalar alignment.
    let reals: Vec<f32> = vec![1.0, -0.0, 3.5];
    let input = upload(&session, &reals);
    let bytes = convert(&session, DType::C32, DType::F32, &input, 3, 0);
    let complex: Vec<C32> = from_bytes(&bytes);
    assert_eq!(complex[0], C32 { re: 1.0, im: 0.0 });
    assert_eq!(complex[1].re.to_bits(), (-0.0f32).to_bits());
    assert_eq!(complex[1].im.to_bits(), 0.0f32.to_bits());
    assert_eq!(complex[2], C32 { re: 3.5, im: 0.0 });

    // Complex -> real keeps the real part and ignores a nonfinite imaginary part.
    let complex: Vec<C64> = vec![
        C64 {
            re: 2.5,
            im: f64::NAN,
        },
        C64 {
            re: -1.0,
            im: f64::INFINITY,
        },
    ];
    let input = upload(&session, &complex);
    let bytes = convert(&session, DType::F64, DType::C64, &input, 2, 0);
    assert_eq!(gpu::from_bytes::<f64>(&bytes), vec![2.5, -1.0]);

    // Bool destinations are predicates.
    let reals: Vec<f32> = vec![0.0, -0.0, 2.0, f32::NAN, f32::INFINITY];
    let input = upload(&session, &reals);
    let bytes = convert(&session, DType::Bool, DType::F32, &input, 5, 0);
    assert_eq!(bytes, vec![0, 0, 1, 1, 1]);
    let complex: Vec<C32> = vec![
        C32 { re: 0.0, im: 0.0 },
        C32 { re: 1.0, im: 0.0 },
        C32 { re: 0.0, im: -1.0 },
        C32 {
            re: 0.0,
            im: f32::NAN,
        },
    ];
    let input = upload(&session, &complex);
    let bytes = convert(&session, DType::Bool, DType::C32, &input, 4, 0);
    assert_eq!(bytes, vec![0, 1, 1, 1]);

    // Bool sources cast the 0/1 byte, into numeric and complex destinations.
    let flags: Vec<u8> = vec![1, 0, 1];
    let input = upload(&session, &flags);
    let bytes = convert(&session, DType::I32, DType::Bool, &input, 3, 0);
    assert_eq!(gpu::from_bytes::<i32>(&bytes), vec![1, 0, 1]);
    let bytes = convert(&session, DType::C64, DType::Bool, &input, 3, 0);
    let complex: Vec<C64> = from_bytes(&bytes);
    assert_eq!(complex[0], C64 { re: 1.0, im: 0.0 });
    assert_eq!(complex[1], C64 { re: 0.0, im: 0.0 });
}

#[test]
fn conversions_cover_empty_singleton_and_partial_blocks() {
    let session = Session::new("convert");
    // Empty: no launch at all.
    let empty = written(&session, &[]);
    let plan = plan_convert(DType::F64, DType::F32, 0).unwrap();
    assert!(plan.is_empty());
    let out = sentinel(&session, 8);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&empty),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::F32.tag()),
        Slot::i64(0),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_convert"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before);

    // Singleton.
    let input = upload(&session, &[7.25f64]);
    let bytes = convert(&session, DType::F32, DType::F64, &input, 1, 4);
    assert_eq!(gpu::from_bytes::<f32>(&bytes), vec![7.25]);

    // One element past a block boundary, with a sentinel guard that must survive.
    let len = 257i64;
    let values: Vec<i32> = (0..len).map(|index| index as i32 * 3).collect();
    let input = upload(&session, &values);
    let bytes = convert(
        &session,
        DType::I64,
        DType::I32,
        &input,
        len,
        (len as usize + 2) * 8,
    );
    let converted: Vec<i64> = from_bytes(&bytes[..(len as usize) * 8]);
    for (got, &want) in converted.iter().zip(values.iter()) {
        assert_eq!(*got, i64::from(want));
    }
    // The guard region behind the logical output is untouched.
    assert!(bytes[(len as usize) * 8..]
        .iter()
        .all(|&byte| byte == SENTINEL));
}

#[test]
fn validated_float_to_int_conversion_is_well_defined() {
    let session = Session::with_modules(&["validation", "convert"]);
    let len = 1i64 << 16;
    // One out-of-range value late in the buffer; the fraction is a *valid*
    // truncating conversion, and the conversion only runs once validation passes.
    let mut values: Vec<f64> = (0..len)
        .map(|index| (index % 1000) as f64 - 500.0)
        .collect();
    values[(len - 3) as usize] = 2.5;
    let input = upload(&session, &values);
    let flag = sentinel(&session, 4);
    let value_slot = sentinel(&session, 8);

    // Fractions pass this validator: truncation is defined.
    validate_real_cast_fixture(&session, &input, &flag, &value_slot, len, 1);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag))[0],
        u32::MAX,
        "a fraction is a valid truncating input"
    );

    // An out-of-range value is rejected at its position.
    values[(len - 3) as usize] = 4_294_967_296.0;
    let input = upload(&session, &values);
    validate_real_cast_fixture(&session, &input, &flag, &value_slot, len, 1);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag))[0],
        (len - 3) as u32,
        "4.3e9 exceeds the I32 acceptance range"
    );

    // The corrected fixture validates, and the conversion then truncates toward
    // zero inside the documented domain.
    values[(len - 3) as usize] = -7.75;
    let input = upload(&session, &values);
    validate_real_cast_fixture(&session, &input, &flag, &value_slot, len, 1);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag))[0],
        u32::MAX,
        "the corrected fixture must validate"
    );
    let bytes = convert(&session, DType::I32, DType::F64, &input, len, 0);
    let converted: Vec<i32> = from_bytes(&bytes[..(len as usize) * 4]);
    for (got, &want) in converted.iter().zip(values.iter()) {
        assert_eq!(*got, want.trunc() as i32);
    }
}

/// Initialize the flag and run the real-cast validator over `len` I32-bound
/// samples (`stride = 1`).
fn validate_real_cast_fixture(
    session: &Session,
    input: &Buf,
    flag: &Buf,
    value_slot: &Buf,
    len: i64,
    stride: i64,
) {
    let mut init_slots = [
        Slot::ptr(flag),
        Slot::ptr(value_slot),
        Slot::i32(DType::F64.tag()),
    ];
    launch(
        session.function("t4a_cuda_init_validation"),
        &t4a_cuda_kernels::plan_init_validation(DType::F64).unwrap(),
        session.stream(),
        &mut init_slots,
    );
    let plan = plan_validate_real_cast(DType::F64, len, stride).unwrap();
    let mut slots = [
        Slot::ptr(input),
        Slot::ptr(flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(-2_147_483_648.0),
        Slot::f64(2_147_483_647.0),
        Slot::i64(stride),
        Slot::i32(1),
        Slot::i64(len * stride),
    ];
    launch(
        session.function("t4a_cuda_validate_real_cast"),
        &plan,
        session.stream(),
        &mut slots,
    );
}

#[test]
fn complex_validation_samples_the_real_components_before_converting() {
    let session = Session::with_modules(&["validation", "convert"]);
    // Valid real components; the imaginary part of one entry is nonfinite and
    // must be skipped by the stride-2 validation and ignored by the cast.
    let complex: Vec<C64> = vec![
        C64 {
            re: 2.0,
            im: f64::NAN,
        },
        C64 {
            re: -3.0,
            im: f64::INFINITY,
        },
    ];
    let input = upload(&session, &complex);
    let flag = sentinel(&session, 4);
    let value_slot = sentinel(&session, 8);
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value_slot),
        Slot::i32(DType::F64.tag()),
    ];
    launch(
        session.function("t4a_cuda_init_validation"),
        &t4a_cuda_kernels::plan_init_validation(DType::F64).unwrap(),
        session.stream(),
        &mut slots,
    );
    let validate = plan_validate_real_cast(DType::F64, 4, 2).unwrap();
    let mut slots = [
        Slot::ptr(&input),
        Slot::ptr(&flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(-2_147_483_648.0),
        Slot::f64(2_147_483_647.0),
        Slot::i64(2),
        Slot::i32(0),
        Slot::i64(4),
    ];
    launch(
        session.function("t4a_cuda_validate_real_cast"),
        &validate,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        gpu::from_bytes::<u32>(&read_bytes(&session, &flag))[0],
        u32::MAX,
        "the imaginary components must not be sampled"
    );
    let bytes = convert(&session, DType::I32, DType::C64, &input, 2, 0);
    assert_eq!(gpu::from_bytes::<i32>(&bytes), vec![2, -3]);
}

#[test]
fn remaining_pairs_execute_with_reference_values() {
    let session = Session::new("convert");
    // A small fixture per route that the other tests do not cover, each checked
    // against a CPU reference. Five elements keep the expectations readable.
    let ints32: Vec<i32> = vec![7, -9, 0, i32::MAX, i32::MIN];
    let ints64: Vec<i64> = vec![7, -9, 0, i64::MAX, i64::MIN];
    // The representable predecessor of 2^63 and a subnormal F64.
    let edge_floats: Vec<f64> = vec![
        -9_223_372_036_854_775_808.0,
        f64::from_bits(0x0000_0000_0000_0001),
        2_147_483_647.0,
        -2_147_483_648.0,
        1.0,
    ];

    // Integer -> integer and integer -> float routes.
    let input = upload(&session, &ints32);
    let bytes = convert(&session, DType::I64, DType::I32, &input, 5, 0);
    assert_eq!(
        from_bytes::<i64>(&bytes),
        ints32.iter().map(|&v| i64::from(v)).collect::<Vec<_>>()
    );
    let bytes = convert(&session, DType::F32, DType::I32, &input, 5, 0);
    assert_eq!(
        from_bytes::<f32>(&bytes),
        ints32.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
    let bytes = convert(&session, DType::F64, DType::I32, &input, 5, 0);
    assert_eq!(
        from_bytes::<f64>(&bytes),
        ints32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
    );
    let input = upload(&session, &ints64);
    let bytes = convert(&session, DType::F64, DType::I64, &input, 5, 0);
    assert_eq!(
        from_bytes::<f64>(&bytes),
        ints64.iter().map(|&v| v as f64).collect::<Vec<_>>()
    );
    let bytes = convert(&session, DType::I32, DType::I64, &input, 5, 0);
    assert_eq!(
        from_bytes::<i32>(&bytes),
        ints64.iter().map(|&v| v as i32).collect::<Vec<_>>()
    );

    // Float -> I64 inside the validated domain, including the 2^63 predecessor.
    let input = upload(&session, &edge_floats);
    let bytes = convert(&session, DType::I64, DType::F64, &input, 5, 0);
    assert_eq!(
        from_bytes::<i64>(&bytes),
        edge_floats
            .iter()
            .map(|&v| v.trunc() as i64)
            .collect::<Vec<_>>()
    );
    let bytes = convert(&session, DType::F32, DType::F64, &input, 5, 0);
    assert_eq!(
        from_bytes::<f32>(&bytes),
        edge_floats.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );

    // Complex -> C32/C64 pairs and complex -> integer.
    let complex: Vec<C64> = vec![
        C64 { re: 1.5, im: -2.5 },
        C64 { re: -0.0, im: 0.0 },
        C64 { re: 3.25, im: 4.75 },
        C64 { re: 0.0, im: 1.0 },
        C64 { re: -7.5, im: 0.5 },
    ];
    let input = upload(&session, &complex);
    let bytes = convert(&session, DType::I32, DType::C64, &input, 5, 0);
    assert_eq!(
        from_bytes::<i32>(&bytes),
        complex
            .iter()
            .map(|v| v.re.trunc() as i32)
            .collect::<Vec<_>>()
    );
    let input = upload(
        &session,
        &complex
            .iter()
            .map(|v| C32 {
                re: v.re as f32,
                im: v.im as f32,
            })
            .collect::<Vec<_>>(),
    );
    let bytes = convert(&session, DType::C64, DType::C32, &input, 5, 0);
    let widened: Vec<C64> = from_bytes(&bytes);
    for (got, want) in widened.iter().zip(complex.iter()) {
        assert_eq!(got.re.to_bits(), f64::from(want.re as f32).to_bits());
        assert_eq!(got.im.to_bits(), f64::from(want.im as f32).to_bits());
    }

    // Integer -> complex and Bool -> float.
    let input = upload(&session, &ints32);
    let bytes = convert(&session, DType::C32, DType::I32, &input, 5, 0);
    let as_complex: Vec<C32> = from_bytes(&bytes);
    for (got, &want) in as_complex.iter().zip(ints32.iter()) {
        assert_eq!(got.re, want as f32);
        assert_eq!(got.im.to_bits(), 0.0f32.to_bits());
    }
    let flags: Vec<u8> = vec![1, 0, 1, 1, 0];
    let input = upload(&session, &flags);
    let bytes = convert(&session, DType::F64, DType::Bool, &input, 5, 0);
    assert_eq!(from_bytes::<f64>(&bytes), vec![1.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn complex_subspans_need_only_scalar_alignment() {
    let session = Session::new("convert");
    // C32 subspan starting four bytes into an eight-byte-aligned allocation: the
    // documented alignment is the scalar's, not float2's.
    let mut c32_values = vec![0.0f32; 4];
    c32_values[1] = 9.0;
    c32_values[2] = -9.0;
    let input = upload(&session, &c32_values);
    let plan = plan_convert(DType::F32, DType::C32, 1).unwrap();
    let out = sentinel(&session, 16);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr_offset(&input, 4),
        Slot::i32(DType::F32.tag()),
        Slot::i32(DType::C32.tag()),
        Slot::i64(1),
    ];
    launch(
        session.function("t4a_cuda_convert"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let bytes = read_bytes(&session, &out);
    assert_eq!(from_bytes::<f32>(&bytes[..4]), vec![9.0]);
    assert!(bytes[4..].iter().all(|&byte| byte == SENTINEL));

    // C64 subspan starting eight bytes into a sixteen-byte-aligned allocation.
    let mut c64_values = vec![0.0f64; 4];
    c64_values[1] = 2.5;
    c64_values[2] = f64::NAN;
    let input = upload(&session, &c64_values);
    let plan = plan_convert(DType::F64, DType::C64, 1).unwrap();
    let out = sentinel(&session, 24);
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr_offset(&input, 8),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::C64.tag()),
        Slot::i64(1),
    ];
    launch(
        session.function("t4a_cuda_convert"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let bytes = read_bytes(&session, &out);
    // The real part is 2.5; the NaN imaginary part is ignored.
    assert_eq!(from_bytes::<f64>(&bytes[..8]), vec![2.5]);
    assert!(bytes[8..].iter().all(|&byte| byte == SENTINEL));
}
