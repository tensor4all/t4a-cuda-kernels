//! Numerical GPU tests for every movement and arithmetic entry.
//!
//! Compiled with `nvcc` and launched through the raw CUDA driver API, so the
//! kernels are exercised the way an external caller would use them. Run with
//! `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{as_bytes, download, launch, launches, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_contiguous_to_view, plan_copy_exact, plan_elementwise, plan_fill_zero,
    plan_fill_zero_view, plan_materialize_strided, plan_strided_to_strided, plan_tiled_transpose,
    CLayout, CTransposeConfig, DType, Entry, Layout, TransposeConfig,
};

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

/// Relative tolerance of an arithmetic route at its element precision. Copies
/// are compared bitwise instead.
fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 1e-6,
        _ => 1e-12,
    }
}

fn approx(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual == expected || (actual.is_nan() && expected.is_nan()) {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    let scale = expected.abs().max(1.0);
    (actual - expected).abs() <= tolerance * scale
}

fn launch_binary(
    session: &Session,
    entry: Entry,
    dtype: DType,
    lhs: &Buf,
    rhs: &Buf,
    out: &Buf,
    len: i64,
) {
    let plan = plan_elementwise(entry, dtype, len).unwrap();
    let mut slots = [
        Slot::ptr(out),
        Slot::ptr(lhs),
        Slot::ptr(rhs),
        Slot::i32(dtype.tag()),
        Slot::i64(len),
    ];
    launch(
        session.function(entry.name()),
        &plan,
        session.stream(),
        &mut slots,
    );
}

#[test]
fn fill_zero_covers_every_dtype() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_fill_zero");
    // Far larger than one 256-thread block: a single-threaded body cannot pass.
    let len = 65_537i64;
    for dtype in DType::ALL {
        let bytes = len as usize * dtype.size();
        let out = sentinel(&session, bytes);
        let plan = plan_fill_zero(dtype, len).unwrap();
        let mut slots = [Slot::ptr(&out), Slot::i32(dtype.tag()), Slot::i64(len)];
        let before = launches();
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(launches(), before + 1, "{dtype:?}");
        assert!(
            read_bytes(&session, &out).iter().all(|&byte| byte == 0),
            "{dtype:?}"
        );
    }
}

#[test]
fn fill_zero_view_preserves_holes() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_fill_zero_view");
    for dtype in [DType::F32, DType::C64, DType::Bool] {
        let extents = [3i64, 2];
        let strides = [2i64, 1];
        let offset = 1i64;
        let alloc = 12usize;
        let out = sentinel(&session, alloc * dtype.size());
        let layout = Layout::new(&extents, &strides, offset).unwrap();
        let plan = plan_fill_zero_view(dtype, &layout, alloc as i64).unwrap();
        let mirror = CLayout::new(&layout);
        let mut slots = [
            Slot::ptr(&out),
            Slot::i32(dtype.tag()),
            Slot::layout(&mirror),
            Slot::i64(layout.element_count().unwrap()),
        ];
        launch(function, &plan, session.stream(), &mut slots);

        let touched: Vec<usize> = view_indices(&extents, &strides, offset)
            .into_iter()
            .map(|index| index as usize)
            .collect();
        let bytes = read_bytes(&session, &out);
        for element in 0..alloc {
            let window = &bytes[element * dtype.size()..(element + 1) * dtype.size()];
            if touched.contains(&element) {
                assert!(window.iter().all(|&byte| byte == 0), "{dtype:?} {element}");
            } else {
                assert!(
                    window.iter().all(|&byte| byte == SENTINEL),
                    "{dtype:?} {element}"
                );
            }
        }
    }
}

#[test]
fn copy_exact_is_bit_preserving_for_every_dtype() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_copy_exact");
    let len = 4_097i64;
    for dtype in DType::ALL {
        let bytes = len as usize * dtype.size();
        // Nonfinite payloads, signed zero and 0xFF bytes: a bit-preserving copy
        // must reproduce them exactly, and arithmetic would not.
        let pattern: Vec<u8> = (0..bytes)
            .map(|index| match index % 11 {
                0 => 0x00,
                1 => 0xFF,
                2 => 0x7F,
                3 => 0x80,
                4 => 0x01,
                _ => (index * 37 % 251) as u8,
            })
            .collect();
        let src = written(&session, &pattern);
        let dst = sentinel(&session, bytes);
        let plan = plan_copy_exact(dtype, len).unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(dtype.tag()),
            Slot::i64(len),
        ];
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(read_bytes(&session, &dst), pattern, "{dtype:?}");
    }
}

#[test]
fn materialize_strided_handles_negative_and_broadcast_strides() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_materialize_strided");

    // Reversed source view over a nine-element buffer.
    let src_values: [f32; 9] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let src = upload(&session, &src_values);
    let extents = [4i64];
    let strides = [-2i64];
    let offset = 6i64;
    let layout = Layout::new(&extents, &strides, offset).unwrap();
    let dst = sentinel(&session, 4 * 4);
    let plan = plan_materialize_strided(DType::F32, &layout, 9).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(4),
    ];
    launch(function, &plan, session.stream(), &mut slots);
    let expected: Vec<f32> = view_indices(&extents, &strides, offset)
        .into_iter()
        .map(|index| src_values[index as usize])
        .collect();
    assert_eq!(download::<f32>(&session, &dst, 4), expected);

    // Broadcast read with nonfinite payloads: axis 1 has stride zero and the
    // values must be reproduced bit for bit.
    let c64_values = [
        C64 { re: 1.0, im: -0.0 },
        C64 {
            re: f64::INFINITY,
            im: f64::NAN,
        },
        C64 {
            re: -1.5,
            im: f64::from_bits(0x7ff8_0000_0000_0001),
        },
    ];
    let src = upload(&session, &c64_values);
    let layout = Layout::new(&[3i64, 2], &[1i64, 0], 0).unwrap();
    let dst = sentinel(&session, 6 * std::mem::size_of::<C64>());
    let plan = plan_materialize_strided(DType::C64, &layout, 3).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::C64.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(6),
    ];
    launch(function, &plan, session.stream(), &mut slots);
    let expected_bits: Vec<(u64, u64)> = [0usize, 1, 2, 0, 1, 2]
        .iter()
        .map(|&index| {
            (
                c64_values[index].re.to_bits(),
                c64_values[index].im.to_bits(),
            )
        })
        .collect();
    let actual = download::<C64>(&session, &dst, 6);
    for (got, want) in actual.iter().zip(expected_bits.iter()) {
        assert_eq!((got.re.to_bits(), got.im.to_bits()), *want);
    }
}

#[test]
fn contiguous_to_view_preserves_the_rest_of_the_allocation() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_contiguous_to_view");
    let src_values: [f64; 4] = [1.0, f64::INFINITY, f64::NAN, -0.0];
    let src = upload(&session, &src_values);
    let extents = [2i64, 2];
    let strides = [1i64, 4];
    let offset = 2i64;
    let layout = Layout::new(&extents, &strides, offset).unwrap();
    let alloc = 10usize;
    let dst = sentinel(&session, alloc * 8);
    let plan = plan_contiguous_to_view(DType::F64, &layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(4),
    ];
    launch(function, &plan, session.stream(), &mut slots);

    let mut expected = vec![SENTINEL; alloc * 8];
    for (flat, index) in view_indices(&extents, &strides, offset)
        .into_iter()
        .enumerate()
    {
        let index = index as usize;
        expected[index * 8..(index + 1) * 8].copy_from_slice(as_bytes(&[src_values[flat]]));
    }
    assert_eq!(read_bytes(&session, &dst), expected);
}

#[test]
fn strided_to_strided_writes_only_its_destination_region() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_strided_to_strided");
    let src_values: [f32; 9] = [0.0, -0.0, f32::INFINITY, f32::NAN, 4.0, 5.0, 6.0, 7.0, 8.0];
    let src = upload(&session, &src_values);
    let extents = [2i64, 2];
    let src_strides = [3i64, 1];
    let dst_strides = [1i64, 4];
    let dst_offset = 2i64;
    let src_layout = Layout::new(&extents, &src_strides, 0).unwrap();
    let dst_layout = Layout::new(&extents, &dst_strides, dst_offset).unwrap();
    let alloc = 10usize;
    let dst = sentinel(&session, alloc * 4);
    let plan =
        plan_strided_to_strided(DType::F32, &src_layout, 9, &dst_layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&src_layout)),
        Slot::layout(&CLayout::new(&dst_layout)),
        Slot::i64(4),
    ];
    launch(function, &plan, session.stream(), &mut slots);

    let mut expected = vec![SENTINEL; alloc * 4];
    let source = view_indices(&extents, &src_strides, 0);
    let destination = view_indices(&extents, &dst_strides, dst_offset);
    for (src_index, dst_index) in source.into_iter().zip(destination) {
        let value = src_values[src_index as usize];
        expected[dst_index as usize * 4..(dst_index as usize + 1) * 4]
            .copy_from_slice(as_bytes(&[value]));
    }
    assert_eq!(read_bytes(&session, &dst), expected);
}

fn reference_transpose<T: Copy>(src: &[T], dst_fast: usize, src_fast: usize) -> Vec<T> {
    let matrix = dst_fast * src_fast;
    let mut out = vec![src[0]; matrix];
    for i in 0..dst_fast {
        for j in 0..src_fast {
            out[i + j * dst_fast] = src[i * src_fast + j];
        }
    }
    out
}

#[test]
fn tiled_transpose_matches_the_coordinate_map() {
    let session = Session::new("movement");
    let function = session.function("t4a_cuda_tiled_transpose");
    let cases = [
        (5usize, 7usize, 2i64, 0i64, TransposeConfig::DEFAULT),
        (16, 8, 1, 3, TransposeConfig::new(16, 8, 1, 2).unwrap()),
        (33, 17, 3, 0, TransposeConfig::new(32, 8, 1, 4).unwrap()),
        (8, 8, 1, 1, TransposeConfig::new(8, 4, 0, 1).unwrap()),
        (1, 1, 1, 0, TransposeConfig::DEFAULT),
    ];
    for (m, n, batches, src_offset, config) in cases {
        let per_batch = m * n;
        let total = per_batch * batches as usize;

        // Nonfinite payloads, signed zeros and NaN payloads must survive a
        // movement route untouched.
        let c64_values: Vec<C64> = (0..total)
            .map(|index| match index % 7 {
                0 => C64 { re: -0.0, im: 0.0 },
                1 => C64 {
                    re: f64::INFINITY,
                    im: f64::NEG_INFINITY,
                },
                2 => C64 {
                    re: f64::from_bits(0x7ff8_0000_0000_0002),
                    im: f64::NAN,
                },
                _ => C64 {
                    re: index as f64 * 0.5,
                    im: -(index as f64),
                },
            })
            .collect();
        let c32_values: Vec<C32> = c64_values
            .iter()
            .map(|value| C32 {
                re: value.re as f32,
                im: value.im as f32,
            })
            .collect();

        // Both the source allocation and the destination are byte-compared, so
        // holes, tails and payload bits are covered.
        for dtype in [DType::C32, DType::C64] {
            let total_bytes = total * dtype.size();
            let dst = sentinel(&session, total_bytes);
            let pad = src_offset as usize;
            let (src, expected): (Buf, Vec<u8>) = if dtype == DType::C32 {
                let mut padded = vec![C32 { re: -7.0, im: 7.0 }; pad];
                padded.extend_from_slice(&c32_values);
                let mut expected = Vec::with_capacity(total_bytes);
                for batch in 0..batches as usize {
                    let start = batch * per_batch;
                    expected.extend_from_slice(as_bytes(&reference_transpose(
                        &c32_values[start..start + per_batch],
                        m,
                        n,
                    )));
                }
                (upload(&session, &padded), expected)
            } else {
                let mut padded = vec![C64 { re: -7.0, im: 7.0 }; pad];
                padded.extend_from_slice(&c64_values);
                let mut expected = Vec::with_capacity(total_bytes);
                for batch in 0..batches as usize {
                    let start = batch * per_batch;
                    expected.extend_from_slice(as_bytes(&reference_transpose(
                        &c64_values[start..start + per_batch],
                        m,
                        n,
                    )));
                }
                (upload(&session, &padded), expected)
            };

            let plan = plan_tiled_transpose(
                dtype,
                config,
                src_offset,
                per_batch as i64,
                m as i64,
                n as i64,
                batches,
                (pad + total) as i64,
                total as i64,
            )
            .unwrap();
            let mut slots = [
                Slot::ptr(&dst),
                Slot::ptr(&src),
                Slot::i32(dtype.tag()),
                Slot::i64(src_offset),
                Slot::i64(per_batch as i64),
                Slot::i64(m as i64),
                Slot::i64(n as i64),
                Slot::config(&CTransposeConfig::from(config)),
            ];
            launch(function, &plan, session.stream(), &mut slots);
            assert_eq!(
                read_bytes(&session, &dst),
                expected,
                "M={m} N={n} batches={batches} offset={src_offset} {dtype:?}"
            );
        }
    }
}

#[test]
fn float_arithmetic_matches_cpu_references() {
    let session = Session::new("arithmetic");
    // Far larger than one 256-thread block: a single-threaded body cannot pass.
    let len = 65_536i64;
    type BinOp = fn(f64, f64) -> f64;
    let entries: [(Entry, BinOp); 4] = [
        (Entry::Add, |a, b| a + b),
        (Entry::Sub, |a, b| a - b),
        (Entry::Mul, |a, b| a * b),
        (Entry::Div, |a, b| a / b),
    ];
    for dtype in [DType::F32, DType::F64] {
        let a: Vec<f64> = (0..len as usize)
            .map(|index| (index as f64 - 512.0) * 0.03125)
            .collect();
        let b: Vec<f64> = (0..len as usize)
            .map(|index| ((index * 7) as f64 % 89.0) - 44.0)
            .collect();
        for (entry, reference) in entries {
            let (lhs, rhs) = if dtype == DType::F32 {
                let a: Vec<f32> = a.iter().map(|&value| value as f32).collect();
                let b: Vec<f32> = b.iter().map(|&value| value as f32).collect();
                (upload(&session, &a), upload(&session, &b))
            } else {
                (upload(&session, &a), upload(&session, &b))
            };
            let out = sentinel(&session, len as usize * dtype.size());
            launch_binary(&session, entry, dtype, &lhs, &rhs, &out, len);
            let actual = if dtype == DType::F32 {
                download::<f32>(&session, &out, len as usize)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<f64>>()
            } else {
                download::<f64>(&session, &out, len as usize)
            };
            for (index, got) in actual.iter().enumerate() {
                let expected = reference(a[index], b[index]);
                assert!(
                    approx(*got, expected, tolerance(dtype)),
                    "{} {dtype:?} at {index}: {got} vs {expected}",
                    entry.name()
                );
            }
        }
    }
}

#[test]
fn integer_arithmetic_wraps_at_the_boundaries() {
    let session = Session::new("arithmetic");

    let i32_values: [i32; 5] = [i32::MAX, i32::MIN, -1, 7, 0];
    let i32_others: [i32; 5] = [1, -1, i32::MAX, i32::MIN, 0];
    let i64_values: [i64; 5] = [i64::MAX, i64::MIN, -1, 7, 0];
    let i64_others: [i64; 5] = [1, -1, i64::MAX, i64::MIN, 0];
    for (entry, reference) in [
        (
            Entry::Add,
            (|a: i32, b: i32| a.wrapping_add(b)) as fn(i32, i32) -> i32,
        ),
        (Entry::Sub, |a, b| a.wrapping_sub(b)),
        (Entry::Mul, |a, b| a.wrapping_mul(b)),
    ] {
        let lhs = upload(&session, &i32_values);
        let rhs = upload(&session, &i32_others);
        let out = sentinel(&session, 5 * 4);
        launch_binary(&session, entry, DType::I32, &lhs, &rhs, &out, 5);
        let expected: Vec<i32> = i32_values
            .iter()
            .zip(i32_others.iter())
            .map(|(&a, &b)| reference(a, b))
            .collect();
        assert_eq!(
            download::<i32>(&session, &out, 5),
            expected,
            "{} I32",
            entry.name()
        );

        let lhs = upload(&session, &i64_values);
        let rhs = upload(&session, &i64_others);
        let out = sentinel(&session, 5 * 8);
        launch_binary(&session, entry, DType::I64, &lhs, &rhs, &out, 5);
        let expected: Vec<i64> = match entry {
            Entry::Add => i64_values
                .iter()
                .zip(i64_others.iter())
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect(),
            Entry::Sub => i64_values
                .iter()
                .zip(i64_others.iter())
                .map(|(&a, &b)| a.wrapping_sub(b))
                .collect(),
            _ => i64_values
                .iter()
                .zip(i64_others.iter())
                .map(|(&a, &b)| a.wrapping_mul(b))
                .collect(),
        };
        assert_eq!(
            download::<i64>(&session, &out, 5),
            expected,
            "{} I64",
            entry.name()
        );
    }
}

#[test]
fn complex_arithmetic_matches_cpu_references() {
    let session = Session::new("arithmetic");
    let c64_values: [C64; 5] = [
        C64 { re: 1.0, im: 2.0 },
        C64 { re: -0.0, im: 0.0 },
        C64 {
            re: f64::INFINITY,
            im: 1.0,
        },
        C64 { re: 3.0, im: -4.0 },
        C64 {
            re: f64::NAN,
            im: 1.0,
        },
    ];
    let c64_others: [C64; 5] = [
        C64 { re: 5.0, im: -6.0 },
        C64 { re: 1.0, im: -1.0 },
        C64 { re: 0.0, im: 1.0 },
        C64 { re: -2.0, im: 0.5 },
        C64 { re: 1.0, im: 1.0 },
    ];
    let c32_values: Vec<C32> = c64_values
        .iter()
        .map(|value| C32 {
            re: value.re as f32,
            im: value.im as f32,
        })
        .collect();
    let c32_others: Vec<C32> = c64_others
        .iter()
        .map(|value| C32 {
            re: value.re as f32,
            im: value.im as f32,
        })
        .collect();

    for entry in [Entry::Add, Entry::Sub, Entry::Mul] {
        for dtype in [DType::C32, DType::C64] {
            let (lhs, rhs) = if dtype == DType::C32 {
                (upload(&session, &c32_values), upload(&session, &c32_others))
            } else {
                (upload(&session, &c64_values), upload(&session, &c64_others))
            };
            let out = sentinel(&session, 5 * dtype.size());
            launch_binary(&session, entry, dtype, &lhs, &rhs, &out, 5);
            let actual: Vec<(f64, f64)> = if dtype == DType::C32 {
                download::<C32>(&session, &out, 5)
                    .into_iter()
                    .map(|value| (f64::from(value.re), f64::from(value.im)))
                    .collect()
            } else {
                download::<C64>(&session, &out, 5)
                    .into_iter()
                    .map(|value| (value.re, value.im))
                    .collect()
            };
            for (index, ((a, b), got)) in c64_values
                .iter()
                .zip(c64_others.iter())
                .zip(actual.iter())
                .enumerate()
            {
                let (re, im) = match entry {
                    Entry::Add => (a.re + b.re, a.im + b.im),
                    Entry::Sub => (a.re - b.re, a.im - b.im),
                    Entry::Mul => (a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re),
                    _ => unreachable!(),
                };
                assert!(
                    approx(got.0, re, tolerance(dtype)) && approx(got.1, im, tolerance(dtype)),
                    "{} {dtype:?} at {index}: {got:?} vs ({re}, {im})",
                    entry.name()
                );
            }
        }
    }
}

#[test]
fn division_follows_ieee_edge_cases() {
    let session = Session::new("arithmetic");
    let lhs_values: [f64; 6] = [1.0, -1.0, 0.0, f64::INFINITY, f64::NAN, 1.0];
    let rhs_values: [f64; 6] = [0.0, 0.0, 0.0, f64::INFINITY, 1.0, f64::NAN];
    let lhs32: Vec<f32> = lhs_values.iter().map(|&value| value as f32).collect();
    let rhs32: Vec<f32> = rhs_values.iter().map(|&value| value as f32).collect();
    for dtype in [DType::F32, DType::F64] {
        let (lhs, rhs) = if dtype == DType::F32 {
            (upload(&session, &lhs32), upload(&session, &rhs32))
        } else {
            (upload(&session, &lhs_values), upload(&session, &rhs_values))
        };
        let out = sentinel(&session, 6 * dtype.size());
        launch_binary(&session, Entry::Div, dtype, &lhs, &rhs, &out, 6);
        let actual: Vec<f64> = if dtype == DType::F32 {
            download::<f32>(&session, &out, 6)
                .into_iter()
                .map(f64::from)
                .collect()
        } else {
            download::<f64>(&session, &out, 6)
        };
        assert_eq!(actual[0], f64::INFINITY, "{dtype:?}");
        assert_eq!(actual[1], f64::NEG_INFINITY, "{dtype:?}");
        assert!(actual[2].is_nan(), "{dtype:?}");
        assert!(actual[3].is_nan(), "{dtype:?}");
        assert!(actual[4].is_nan(), "{dtype:?} {actual:?}");
        assert!(actual[5].is_nan(), "{dtype:?}");
    }
}

#[test]
fn empty_requests_plan_and_launch_nothing() {
    let session = Session::new("movement");
    let before = launches();
    for plan in [
        plan_fill_zero(DType::F32, 0).unwrap(),
        plan_copy_exact(DType::C64, 0).unwrap(),
        plan_elementwise(Entry::Add, DType::F32, 0).unwrap(),
        plan_tiled_transpose(DType::F32, TransposeConfig::DEFAULT, 0, 0, 0, 4, 2, 0, 0).unwrap(),
    ] {
        assert!(plan.is_empty());
    }
    assert_eq!(launches(), before);
    // The session is still usable afterwards.
    let out = sentinel(&session, 4);
    let plan = plan_fill_zero(DType::F32, 1).unwrap();
    let function = session.function("t4a_cuda_fill_zero");
    let mut slots = [Slot::ptr(&out), Slot::i32(DType::F32.tag()), Slot::i64(1)];
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(read_bytes(&session, &out), vec![0u8; 4]);
    assert_eq!(launches(), before + 1);
}

#[test]
fn large_domains_cover_every_movement_entry() {
    let session = Session::new("movement");
    // 2^16 elements is 256 blocks: a single-threaded or block-limited body fails
    // here even though it passes on a handful of elements.
    let count = 1usize << 16;
    let extents = [count as i64];
    let strides = [2i64];
    let layout = Layout::new(&extents, &strides, 0).unwrap();
    let alloc = count * 2;

    // fill_zero_view over every other element.
    let out = sentinel(&session, alloc * 4);
    let plan = plan_fill_zero_view(DType::F32, &layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(count as i64),
    ];
    launch(
        session.function("t4a_cuda_fill_zero_view"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let bytes = read_bytes(&session, &out);
    for element in 0..alloc {
        let window = &bytes[element * 4..(element + 1) * 4];
        let expected = if element % 2 == 0 { 0u8 } else { SENTINEL };
        assert!(window.iter().all(|&byte| byte == expected), "{element}");
    }

    // contiguous_to_view and materialize_strided over the same layout.
    let src = upload(&session, &vec![7.5f32; count]);
    let plan = plan_contiguous_to_view(DType::F32, &layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(count as i64),
    ];
    launch(
        session.function("t4a_cuda_contiguous_to_view"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let back = sentinel(&session, count * 4);
    let plan = plan_materialize_strided(DType::F32, &layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&back),
        Slot::ptr(&out),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(count as i64),
    ];
    launch(
        session.function("t4a_cuda_materialize_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert!(download::<f32>(&session, &back, count)
        .iter()
        .all(|&value| value == 7.5));

    // strided_to_strided from a compact source into the strided destination.
    let compact = Layout::new(&extents, &[1], 0).unwrap();
    let plan =
        plan_strided_to_strided(DType::F32, &compact, count as i64, &layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&compact)),
        Slot::layout(&CLayout::new(&layout)),
        Slot::i64(count as i64),
    ];
    launch(
        session.function("t4a_cuda_strided_to_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let bytes = read_bytes(&session, &out);
    for element in 0..alloc {
        let window = &bytes[element * 4..(element + 1) * 4];
        let expected = if element % 2 == 0 {
            7.5f32
        } else {
            f32::from_bits(u32::from_le_bytes([SENTINEL; 4]))
        };
        assert_eq!(
            f32::from_le_bytes(window.try_into().unwrap()).to_bits(),
            expected.to_bits(),
            "{element}"
        );
    }
}
