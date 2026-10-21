//! Numerical GPU tests for the revision-3 reduction entries. Run with
//! `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{
    download, from_bytes, launch, launch_async, launches, pub_sync, read_bytes, upload, written,
    Buf, Session, Slot,
};
use t4a_cuda_kernels::{
    plan_reduce, plan_reduce_partials, plan_reduction, split_policy, CLayout, DType, Layout,
    ReduceOp, ReduceStrategy, ReductionPlan,
};

const SENTINEL: u8 = 0xCD;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct C64 {
    re: f64,
    im: f64,
}

fn sentinel(session: &Session, bytes: usize) -> Buf {
    written(session, &vec![SENTINEL; bytes])
}

fn coordinates(flat: i64, extents: &[i64]) -> Vec<i64> {
    let mut rest = flat;
    extents
        .iter()
        .map(|&extent| {
            let coordinate = rest % extent;
            rest /= extent;
            coordinate
        })
        .collect()
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 1e-5,
        _ => 1e-12,
    }
}

fn approx_within(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual == expected || (actual.is_nan() && expected.is_nan()) {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= tolerance * expected.abs().max(1.0)
}

fn approx(actual: f64, expected: f64) -> bool {
    approx_within(actual, expected, 1e-12)
}

/// Input values in flat logical order of a strided view.
fn gathered(values: &[f64], extents: &[i64], strides: &[i64], offset: i64) -> Vec<f64> {
    let count: i64 = extents.iter().product();
    (0..count)
        .map(|flat| {
            let coords = coordinates(flat, extents);
            let index: i64 = coords
                .iter()
                .zip(strides.iter())
                .map(|(coordinate, stride)| coordinate * stride)
                .sum::<i64>()
                + offset;
            values[index as usize]
        })
        .collect()
}

/// CPU reference for one axis reduction of the gathered values.
fn reference(values: &[f64], extents: &[i64], axis: usize, op: ReduceOp) -> Vec<f64> {
    let mut out_extents = extents.to_vec();
    let axis_len = out_extents[axis];
    out_extents[axis] = 1;
    let output_len: i64 = out_extents.iter().product();
    (0..output_len)
        .map(|out_flat| {
            let mut coords = coordinates(out_flat, &out_extents);
            let mut acc: Option<f64> = None;
            for index in 0..axis_len {
                coords[axis] = index;
                let mut flat = 0i64;
                let mut stride = 1i64;
                for dim in 0..extents.len() {
                    flat += coords[dim] * stride;
                    stride *= extents[dim];
                }
                let value = values[flat as usize];
                acc = Some(match acc {
                    None => match op {
                        ReduceOp::SumSquares => value * value,
                        _ => value,
                    },
                    Some(current) => match op {
                        ReduceOp::Sum => current + value,
                        ReduceOp::Prod => current * value,
                        ReduceOp::SumSquares => current + value * value,
                        ReduceOp::Max => {
                            if current.is_nan() {
                                current
                            } else if value.is_nan() {
                                value
                            } else {
                                current.max(value)
                            }
                        }
                        ReduceOp::Min => {
                            if current.is_nan() {
                                current
                            } else if value.is_nan() {
                                value
                            } else {
                                current.min(value)
                            }
                        }
                    },
                });
            }
            acc.unwrap()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reduce_slots<'a>(
    out: &'a Buf,
    input: &'a Buf,
    dtype: DType,
    op: ReduceOp,
    strategy: ReduceStrategy,
    out_layout: &'a CLayout,
    in_layout: &'a CLayout,
    axis: i32,
    output_len: i64,
) -> [Slot; 9] {
    [
        Slot::ptr(out),
        Slot::ptr(input),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        Slot::i32(strategy.tag()),
        Slot::layout(out_layout),
        Slot::layout(in_layout),
        Slot::i32(axis),
        Slot::i64(output_len),
    ]
}

#[allow(clippy::too_many_arguments)]
fn partials_slots<'a>(
    partials: &'a Buf,
    input: &'a Buf,
    dtype: DType,
    op: ReduceOp,
    partials_layout: &'a CLayout,
    in_layout: &'a CLayout,
    axis: i32,
    output_len: i64,
    splits: i64,
    chunk_len: i64,
) -> [Slot; 10] {
    [
        Slot::ptr(partials),
        Slot::ptr(input),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        Slot::layout(partials_layout),
        Slot::layout(in_layout),
        Slot::i32(axis),
        Slot::i64(output_len),
        Slot::i64(splits),
        Slot::i64(chunk_len),
    ]
}

#[test]
fn reduction_matches_cpu_for_every_route() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    let extents = [4i64, 40, 2];
    let axis = 1usize;
    let total: i64 = extents.iter().product();
    let values: Vec<f64> = (0..total)
        .map(|index| ((index % 97) as f64 - 48.0) * 0.25)
        .collect();
    let input_layout = Layout::compact(&extents).unwrap();
    let mut out_extents = extents.to_vec();
    out_extents[axis] = 1;
    let out_layout = Layout::compact(&out_extents).unwrap();
    let output_len = out_layout.element_count().unwrap();

    for op in [
        ReduceOp::Sum,
        ReduceOp::Prod,
        ReduceOp::SumSquares,
        ReduceOp::Max,
        ReduceOp::Min,
    ] {
        for dtype in [DType::F32, DType::F64] {
            let src = if dtype == DType::F32 {
                let narrowed: Vec<f32> = values.iter().map(|&value| value as f32).collect();
                upload(&session, &narrowed)
            } else {
                upload(&session, &values)
            };
            let dst = sentinel(&session, output_len as usize * dtype.size());
            let strategy = if extents[axis] <= 32 {
                ReduceStrategy::Unit
            } else {
                ReduceStrategy::Plane
            };
            let plan = plan_reduce(
                dtype,
                op,
                strategy,
                &out_layout,
                output_len,
                &input_layout,
                total,
                axis as u32,
                32,
            )
            .unwrap();
            let mut slots = reduce_slots(
                &dst,
                &src,
                dtype,
                op,
                strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&input_layout),
                axis as i32,
                output_len,
            );
            let before = launches();
            launch(function, &plan, session.stream(), &mut slots);
            assert_eq!(launches(), before + 1, "{op:?} {dtype:?}");

            let actual: Vec<f64> = if dtype == DType::F32 {
                download::<f32>(&session, &dst, output_len as usize)
                    .into_iter()
                    .map(f64::from)
                    .collect()
            } else {
                download::<f64>(&session, &dst, output_len as usize)
            };
            let expected = reference(&values, &extents, axis, op);
            for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    approx_within(*got, *want, tolerance(dtype)),
                    "{op:?} {dtype:?} at {index}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn reductions_read_a_strided_input() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    // [8, 40] with strides [16, 1] and offset 2 inside a 160-element buffer:
    // the farthest reachable index is 7*16 + 39 + 2 = 153.
    let extents = [8i64, 40];
    let strides = [16i64, 1];
    let offset = 2i64;
    let buffer: Vec<f64> = (0..160).map(|index| (index % 71) as f64 - 35.0).collect();
    let src = upload(&session, &buffer);
    let in_layout = Layout::new(&extents, &strides, offset).unwrap();
    let out_layout = Layout::compact(&[8, 1]).unwrap();
    let values = gathered(&buffer, &extents, &strides, offset);

    for op in [ReduceOp::Sum, ReduceOp::Max] {
        let dst = sentinel(&session, 8 * 8);
        let plan = plan_reduce(
            DType::F64,
            op,
            ReduceStrategy::Plane,
            &out_layout,
            8,
            &in_layout,
            160,
            1,
            32,
        )
        .unwrap();
        let mut slots = reduce_slots(
            &dst,
            &src,
            DType::F64,
            op,
            ReduceStrategy::Plane,
            &CLayout::new(&out_layout),
            &CLayout::new(&in_layout),
            1,
            8,
        );
        launch(function, &plan, session.stream(), &mut slots);
        let actual = download::<f64>(&session, &dst, 8);
        let expected = reference(&values, &extents, 1, op);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(approx(*got, *want), "{op:?} at {index}: {got} vs {want}");
        }
    }
}

#[test]
fn integer_and_complex_routes_hold_their_contract() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");

    // Integer wrap over a wrapped sum and product.
    let ints: [i32; 4] = [i32::MAX, 1, i32::MIN, -1];
    let src = upload(&session, &ints);
    let in_layout = Layout::compact(&[4]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    for (op, expected) in [
        (
            ReduceOp::Sum,
            ints.iter().fold(0i32, |acc, &v| acc.wrapping_add(v)),
        ),
        (
            ReduceOp::Prod,
            ints.iter().fold(1i32, |acc, &v| acc.wrapping_mul(v)),
        ),
        (ReduceOp::Max, *ints.iter().max().unwrap()),
        (ReduceOp::Min, *ints.iter().min().unwrap()),
    ] {
        let dst = sentinel(&session, 4);
        let plan = plan_reduce(
            DType::I32,
            op,
            ReduceStrategy::Unit,
            &out_layout,
            1,
            &in_layout,
            4,
            0,
            32,
        )
        .unwrap();
        let mut slots = reduce_slots(
            &dst,
            &src,
            DType::I32,
            op,
            ReduceStrategy::Unit,
            &CLayout::new(&out_layout),
            &CLayout::new(&in_layout),
            0,
            1,
        );
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(download::<i32>(&session, &dst, 1), vec![expected], "{op:?}");
    }

    // Complex sum and product with a nonfinite component.
    let complex: [C64; 3] = [
        C64 { re: 1.0, im: 2.0 },
        C64 {
            re: f64::INFINITY,
            im: 1.0,
        },
        C64 { re: 0.5, im: -1.0 },
    ];
    let src = upload(&session, &complex);
    let in_layout = Layout::compact(&[3]).unwrap();
    for op in [ReduceOp::Sum, ReduceOp::Prod] {
        let dst = sentinel(&session, 16);
        let plan = plan_reduce(
            DType::C64,
            op,
            ReduceStrategy::Unit,
            &out_layout,
            1,
            &in_layout,
            3,
            0,
            32,
        )
        .unwrap();
        let mut slots = reduce_slots(
            &dst,
            &src,
            DType::C64,
            op,
            ReduceStrategy::Unit,
            &CLayout::new(&out_layout),
            &CLayout::new(&in_layout),
            0,
            1,
        );
        launch(function, &plan, session.stream(), &mut slots);
        let got = download::<C64>(&session, &dst, 1)[0];
        let (re, im) = if op == ReduceOp::Sum {
            (
                complex.iter().map(|v| v.re).sum::<f64>(),
                complex.iter().map(|v| v.im).sum::<f64>(),
            )
        } else {
            complex.iter().fold((1.0, 0.0), |(ar, ai), v| {
                (ar * v.re - ai * v.im, ar * v.im + ai * v.re)
            })
        };
        assert!(
            approx(got.re, re) && approx(got.im, im),
            "{op:?}: {got:?} vs ({re}, {im})"
        );
    }
}

#[test]
fn plane_and_unit_strategies_meet_at_the_plane_width() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    for (len, strategy) in [
        (31i64, ReduceStrategy::Unit),
        (32, ReduceStrategy::Unit),
        (32, ReduceStrategy::Plane),
        (33, ReduceStrategy::Plane),
        (64, ReduceStrategy::Plane),
        (65, ReduceStrategy::Plane),
    ] {
        let values: Vec<f64> = (0..len).map(|index| index as f64).collect();
        let src = upload(&session, &values);
        let in_layout = Layout::compact(&[len]).unwrap();
        let out_layout = Layout::compact(&[1]).unwrap();
        let dst = sentinel(&session, 8);
        let plan = plan_reduce(
            DType::F64,
            ReduceOp::Sum,
            strategy,
            &out_layout,
            1,
            &in_layout,
            len,
            0,
            32,
        )
        .unwrap();
        let mut slots = reduce_slots(
            &dst,
            &src,
            DType::F64,
            ReduceOp::Sum,
            strategy,
            &CLayout::new(&out_layout),
            &CLayout::new(&in_layout),
            0,
            1,
        );
        launch(function, &plan, session.stream(), &mut slots);
        let expected: f64 = values.iter().sum();
        assert!(
            approx(download::<f64>(&session, &dst, 1)[0], expected),
            "len {len} {strategy:?}"
        );
    }

    // Subwarp participation.
    for (plane_width, len) in [(8u32, 8i64), (8, 9), (16, 16), (16, 17)] {
        let values: Vec<f64> = (1..=len).map(|index| index as f64).collect();
        let src = upload(&session, &values);
        let in_layout = Layout::compact(&[len]).unwrap();
        let out_layout = Layout::compact(&[1]).unwrap();
        let dst = sentinel(&session, 8);
        let strategy = if len <= i64::from(plane_width) {
            ReduceStrategy::Unit
        } else {
            ReduceStrategy::Plane
        };
        let plan = plan_reduce(
            DType::F64,
            ReduceOp::Sum,
            strategy,
            &out_layout,
            1,
            &in_layout,
            len,
            0,
            plane_width,
        )
        .unwrap();
        let mut slots = reduce_slots(
            &dst,
            &src,
            DType::F64,
            ReduceOp::Sum,
            strategy,
            &CLayout::new(&out_layout),
            &CLayout::new(&in_layout),
            0,
            1,
        );
        launch(function, &plan, session.stream(), &mut slots);
        let expected: f64 = values.iter().sum();
        assert!(
            approx(download::<f64>(&session, &dst, 1)[0], expected),
            "plane {plane_width} len {len}"
        );
    }
}

#[test]
fn split_reduction_submits_two_ordered_passes() {
    let session = Session::new("reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");
    let reduce_kernel = session.function("t4a_cuda_reduce");

    for len in [512i64, 513, 1000, 8448] {
        // Two outputs: the compact [len, 2] layout holds 2 * len elements.
        let values: Vec<f64> = (0..len * 2)
            .map(|index| ((index % 89) as f64 - 44.0) * 0.5)
            .collect();
        let src = upload(&session, &values);
        let in_layout = Layout::compact(&[len, 2]).unwrap();
        let out_layout = Layout::compact(&[1, 2]).unwrap();
        let split = split_policy(len, 32)
            .unwrap()
            .expect("policy splits these lengths");
        let partials_layout = Layout::compact(&[split.splits, 2]).unwrap();
        let partials_len = partials_layout.element_count().unwrap();

        let out = sentinel(&session, 2 * 8);
        // Sentinel-filled partials: an omitted store or a hole stays visible.
        let partials = sentinel(&session, partials_len as usize * 8);

        let plan = plan_reduction(
            DType::F64,
            ReduceOp::Sum,
            &out_layout,
            2,
            &in_layout,
            len * 2,
            0,
            32,
        )
        .unwrap();
        let ReductionPlan::Split {
            split: planned,
            pass1,
            pass2,
            ..
        } = plan
        else {
            panic!("len {len} must split");
        };
        assert_eq!(planned, split);

        let before = launches();
        let mut slots1 = partials_slots(
            &partials,
            &src,
            DType::F64,
            ReduceOp::Sum,
            &CLayout::new(&partials_layout),
            &CLayout::new(&in_layout),
            0,
            2,
            split.splits,
            split.chunk_len,
        );
        launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
        let mut slots2 = reduce_slots(
            &out,
            &partials,
            DType::F64,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            &CLayout::new(&out_layout),
            &CLayout::new(&partials_layout),
            0,
            2,
        );
        // Both passes are enqueued before any observation.
        launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
        assert_eq!(launches(), before + 2, "len {len} must be two launches");
        pub_sync(session.stream());

        // Every partial equals the sum of its own chunk.
        let partial_values = download::<f64>(&session, &partials, partials_len as usize);
        for output in 0..2usize {
            for chunk in 0..split.splits as usize {
                let start = chunk as i64 * split.chunk_len;
                let end = if chunk as i64 + 1 == split.splits {
                    len
                } else {
                    start + split.chunk_len
                };
                let expected: f64 = (start..end)
                    .map(|index| values[index as usize + len as usize * output])
                    .sum();
                let got = partial_values[chunk + split.splits as usize * output];
                assert!(
                    approx(got, expected),
                    "len {len} output {output} chunk {chunk}: {got} vs {expected}"
                );
            }
        }
        // And the final result matches the whole-axis sum.
        let actual = download::<f64>(&session, &out, 2);
        for (output, got) in actual.iter().enumerate() {
            let expected: f64 = (0..len)
                .map(|index| values[index as usize + len as usize * output])
                .sum();
            assert!(
                approx(*got, expected),
                "len {len} output {output}: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn sum_of_squares_combines_as_a_plain_sum() {
    let session = Session::new("reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");
    let reduce_kernel = session.function("t4a_cuda_reduce");
    let len = 1000i64;
    let values: Vec<f32> = (0..len)
        .map(|index| (index as f32 * 0.125) - 60.0)
        .collect();
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&[len]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let split = split_policy(len, 32).unwrap().unwrap();
    let partials_layout = Layout::compact(&[split.splits]).unwrap();

    let out = sentinel(&session, 4);
    let partials = sentinel(&session, split.splits as usize * 4);
    let plan = plan_reduction(
        DType::F32,
        ReduceOp::SumSquares,
        &out_layout,
        1,
        &in_layout,
        len,
        0,
        32,
    )
    .unwrap();
    let ReductionPlan::Split {
        pass1,
        pass2,
        combine_op,
        combine_strategy,
        ..
    } = plan
    else {
        panic!("must split");
    };
    assert_eq!(combine_op, ReduceOp::Sum);

    let mut slots1 = partials_slots(
        &partials,
        &src,
        DType::F32,
        ReduceOp::SumSquares,
        &CLayout::new(&partials_layout),
        &CLayout::new(&in_layout),
        0,
        1,
        split.splits,
        split.chunk_len,
    );
    launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
    let mut slots2 = reduce_slots(
        &out,
        &partials,
        DType::F32,
        ReduceOp::Sum,
        combine_strategy,
        &CLayout::new(&out_layout),
        &CLayout::new(&partials_layout),
        0,
        1,
    );
    launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
    pub_sync(session.stream());

    let expected: f64 = values
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let got = f64::from(download::<f32>(&session, &out, 1)[0]);
    assert!(
        (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
        "sum of squares: {got} vs {expected}"
    );
}

#[test]
fn nan_in_one_split_survives_the_combine() {
    let session = Session::new("reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");
    let reduce_kernel = session.function("t4a_cuda_reduce");
    let len = 2048i64;
    let mut values = vec![1.0f64; len as usize];
    values[1500] = f64::NAN;
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&[len]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let split = split_policy(len, 32).unwrap().unwrap();
    let partials_layout = Layout::compact(&[split.splits]).unwrap();
    let out = sentinel(&session, 8);
    let partials = sentinel(&session, split.splits as usize * 8);

    for op in [ReduceOp::Max, ReduceOp::Min] {
        let plan = plan_reduction(DType::F64, op, &out_layout, 1, &in_layout, len, 0, 32).unwrap();
        let ReductionPlan::Split {
            pass1,
            pass2,
            combine_op,
            combine_strategy,
            ..
        } = plan
        else {
            panic!("must split");
        };
        let mut slots1 = partials_slots(
            &partials,
            &src,
            DType::F64,
            op,
            &CLayout::new(&partials_layout),
            &CLayout::new(&in_layout),
            0,
            1,
            split.splits,
            split.chunk_len,
        );
        launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
        let mut slots2 = reduce_slots(
            &out,
            &partials,
            DType::F64,
            combine_op,
            combine_strategy,
            &CLayout::new(&out_layout),
            &CLayout::new(&partials_layout),
            0,
            1,
        );
        launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
        pub_sync(session.stream());
        let got = download::<f64>(&session, &out, 1)[0];
        assert!(got.is_nan(), "{op:?} must propagate the NaN, got {got}");
    }
}

#[test]
fn single_element_axes_seed_from_the_input() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    let in_layout = Layout::compact(&[1]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();

    // SUM keeps the sign of a single -0.0.
    let values = [-0.0f64];
    let src = upload(&session, &values);
    let dst = sentinel(&session, 8);
    let plan = plan_reduce(
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &out_layout,
        1,
        &in_layout,
        1,
        0,
        32,
    )
    .unwrap();
    let mut slots = reduce_slots(
        &dst,
        &src,
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &CLayout::new(&out_layout),
        &CLayout::new(&in_layout),
        0,
        1,
    );
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(
        download::<f64>(&session, &dst, 1)[0].to_bits(),
        (-0.0f64).to_bits()
    );

    // PROD of a single (Inf, 1) keeps its imaginary component.
    let complex = [C64 {
        re: f64::INFINITY,
        im: 1.0,
    }];
    let src = upload(&session, &complex);
    let dst = sentinel(&session, 16);
    let plan = plan_reduce(
        DType::C64,
        ReduceOp::Prod,
        ReduceStrategy::Unit,
        &out_layout,
        1,
        &in_layout,
        1,
        0,
        32,
    )
    .unwrap();
    let mut slots = reduce_slots(
        &dst,
        &src,
        DType::C64,
        ReduceOp::Prod,
        ReduceStrategy::Unit,
        &CLayout::new(&out_layout),
        &CLayout::new(&in_layout),
        0,
        1,
    );
    launch(function, &plan, session.stream(), &mut slots);
    let got = download::<C64>(&session, &dst, 1)[0];
    assert_eq!(got.re.to_bits(), f64::INFINITY.to_bits());
    assert_eq!(got.im.to_bits(), 1.0f64.to_bits());
}

#[test]
fn float_extrema_follow_nan_and_signed_zero_rules() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    let in_layout = Layout::compact(&[40]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();

    let cases: [(Vec<f64>, Option<f64>); 3] = [
        // NaN in one lane only.
        (
            (0..40)
                .map(|i| if i == 33 { f64::NAN } else { i as f64 })
                .collect(),
            None,
        ),
        // All NaN.
        (vec![f64::NAN; 40], None),
        // Signed zeros: fmaxf(-0.0, +0.0) is +0.0, fminf is -0.0.
        (
            (0..40)
                .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
                .collect(),
            Some(0.0),
        ),
    ];
    for (index, (values, expected_max)) in cases.iter().enumerate() {
        let src = upload(&session, values);
        for op in [ReduceOp::Max, ReduceOp::Min] {
            let dst = sentinel(&session, 8);
            let plan = plan_reduce(
                DType::F64,
                op,
                ReduceStrategy::Plane,
                &out_layout,
                1,
                &in_layout,
                40,
                0,
                32,
            )
            .unwrap();
            let mut slots = reduce_slots(
                &dst,
                &src,
                DType::F64,
                op,
                ReduceStrategy::Plane,
                &CLayout::new(&out_layout),
                &CLayout::new(&in_layout),
                0,
                1,
            );
            launch(function, &plan, session.stream(), &mut slots);
            let got = download::<f64>(&session, &dst, 1)[0];
            match expected_max {
                None => assert!(got.is_nan(), "case {index} {op:?}: {got}"),
                Some(zero) => {
                    // Both extrema of the signed-zero pair collapse to +0.0 for
                    // max and -0.0 for min under fmax/fmin ordering.
                    let want: f64 = if op == ReduceOp::Max { *zero } else { -0.0 };
                    assert_eq!(got.to_bits(), want.to_bits(), "case {index} {op:?}");
                }
            }
        }
    }
}

#[test]
fn launch_counts_follow_the_plan() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    // A direct reduction is one launch, a split reduction two.
    let short: Vec<f64> = (0..64).map(|index| index as f64).collect();
    let long: Vec<f64> = (0..2048).map(|index| index as f64).collect();
    let in_short = Layout::compact(&[64]).unwrap();
    let in_long = Layout::compact(&[2048]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();

    let src = upload(&session, &short);
    let dst = sentinel(&session, 8);
    let plan = plan_reduce(
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &out_layout,
        1,
        &in_short,
        64,
        0,
        32,
    )
    .unwrap();
    let mut slots = reduce_slots(
        &dst,
        &src,
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &CLayout::new(&out_layout),
        &CLayout::new(&in_short),
        0,
        1,
    );
    let before = launches();
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);

    let src = upload(&session, &long);
    let plan = plan_reduction(
        DType::F64,
        ReduceOp::Sum,
        &out_layout,
        1,
        &in_long,
        2048,
        0,
        32,
    )
    .unwrap();
    match plan {
        ReductionPlan::Split {
            pass1,
            pass2,
            combine_strategy,
            ..
        } => {
            let before = launches();
            let partials = sentinel(&session, 8 * 8);
            let split = split_policy(2048, 32).unwrap().unwrap();
            let partials_layout = Layout::compact(&[split.splits]).unwrap();
            let mut slots1 = partials_slots(
                &partials,
                &src,
                DType::F64,
                ReduceOp::Sum,
                &CLayout::new(&partials_layout),
                &CLayout::new(&in_long),
                0,
                1,
                split.splits,
                split.chunk_len,
            );
            launch_async(
                session.function("t4a_cuda_reduce_partials"),
                &pass1,
                session.stream(),
                &mut slots1,
            );
            let mut slots2 = reduce_slots(
                &dst,
                &partials,
                DType::F64,
                ReduceOp::Sum,
                combine_strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&partials_layout),
                0,
                1,
            );
            launch_async(function, &pass2, session.stream(), &mut slots2);
            assert_eq!(launches(), before + 2);
            pub_sync(session.stream());
            let expected: f64 = long.iter().sum();
            assert!(approx(download::<f64>(&session, &dst, 1)[0], expected));
        }
        other => panic!("expected a split plan, got {other:?}"),
    }
}

#[test]
fn large_domains_cover_both_reduction_shapes() {
    let session = Session::new("reduce");
    let reduce_kernel = session.function("t4a_cuda_reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");

    // Few outputs over a long axis: the policy splits into 256 chunks.
    let len = 1i64 << 16;
    let values: Vec<f64> = (0..len).map(|index| (index % 13) as f64).collect();
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&[len]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let out = sentinel(&session, 8);
    let plan = plan_reduction(
        DType::F64,
        ReduceOp::Sum,
        &out_layout,
        1,
        &in_layout,
        len,
        0,
        32,
    )
    .unwrap();
    match plan {
        ReductionPlan::Split {
            split,
            pass1,
            pass2,
            ..
        } => {
            let partials_layout = Layout::compact(&[split.splits]).unwrap();
            let partials = sentinel(&session, split.splits as usize * 8);
            let before = launches();
            let mut slots1 = partials_slots(
                &partials,
                &src,
                DType::F64,
                ReduceOp::Sum,
                &CLayout::new(&partials_layout),
                &CLayout::new(&in_layout),
                0,
                1,
                split.splits,
                split.chunk_len,
            );
            launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
            let mut slots2 = reduce_slots(
                &out,
                &partials,
                DType::F64,
                ReduceOp::Sum,
                ReduceStrategy::Plane,
                &CLayout::new(&out_layout),
                &CLayout::new(&partials_layout),
                0,
                1,
            );
            launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
            assert_eq!(launches(), before + 2);
            pub_sync(session.stream());
            let expected: f64 = values.iter().sum();
            assert!(approx(download::<f64>(&session, &out, 1)[0], expected));
        }
        other => panic!("expected a split plan, got {other:?}"),
    }

    // Many outputs over a short axis: one plane launch.
    let outputs = 1i64 << 16;
    let values: Vec<f64> = (0..outputs * 4).map(|index| (index % 7) as f64).collect();
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&[outputs, 4]).unwrap();
    let out_layout = Layout::compact(&[outputs, 1]).unwrap();
    let out = sentinel(&session, outputs as usize * 8);
    let plan = plan_reduce(
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &out_layout,
        outputs,
        &in_layout,
        outputs * 4,
        1,
        32,
    )
    .unwrap();
    let before = launches();
    let mut slots = reduce_slots(
        &out,
        &src,
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &CLayout::new(&out_layout),
        &CLayout::new(&in_layout),
        1,
        outputs,
    );
    launch(reduce_kernel, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);
    let actual = download::<f64>(&session, &out, outputs as usize);
    for (index, got) in actual.iter().enumerate() {
        let expected: f64 = (0..4).map(|j| values[index + outputs as usize * j]).sum();
        assert!(approx(*got, expected), "output {index}");
    }
}

#[test]
fn empty_output_plans_no_launch() {
    let session = Session::new("reduce");
    let function = session.function("t4a_cuda_reduce");
    let in_layout = Layout::compact(&[0, 64]).unwrap();
    let out_layout = Layout::compact(&[0, 1]).unwrap();
    let plan = plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &out_layout,
        0,
        &in_layout,
        0,
        1,
        32,
    )
    .unwrap();
    assert!(plan.is_empty());
    let src = sentinel(&session, 4);
    let dst = sentinel(&session, 4);
    let before = launches();
    let mut slots = reduce_slots(
        &dst,
        &src,
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &CLayout::new(&out_layout),
        &CLayout::new(&in_layout),
        1,
        0,
    );
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before);
    let _ = from_bytes::<u8>(&read_bytes(&session, &dst));
}

/// Runs one reduction on the GPU and returns the downloaded values.
#[allow(clippy::too_many_arguments)]
fn run_reduction(
    session: &Session,
    dtype: DType,
    op: ReduceOp,
    strategy: ReduceStrategy,
    src: &Buf,
    in_layout: &Layout,
    in_len: i64,
    out_layout: &Layout,
    output_len: i64,
    axis: u32,
    plane_width: u32,
) -> Vec<u8> {
    let out = sentinel(session, output_len as usize * dtype.size());
    let plan = plan_reduce(
        dtype,
        op,
        strategy,
        out_layout,
        output_len,
        in_layout,
        in_len,
        axis,
        plane_width,
    )
    .unwrap();
    let mut slots = reduce_slots(
        &out,
        src,
        dtype,
        op,
        strategy,
        &CLayout::new(out_layout),
        &CLayout::new(in_layout),
        axis as i32,
        output_len,
    );
    launch(
        session.function("t4a_cuda_reduce"),
        &plan,
        session.stream(),
        &mut slots,
    );
    read_bytes(session, &out)
}

#[test]
fn integer_and_complex_routes_cover_both_strategies() {
    let session = Session::new("reduce");
    // 16 fits the unit strategy, 40 needs the plane strategy at width 32.
    for len in [16i64, 40] {
        let strategy = if len <= 32 {
            ReduceStrategy::Unit
        } else {
            ReduceStrategy::Plane
        };
        let in_layout = Layout::compact(&[len]).unwrap();
        let out_layout = Layout::compact(&[1]).unwrap();

        let ints: Vec<i32> = (0..len)
            .map(|index| (index as i32 * 7919 - 3) ^ 0x5555_5555)
            .collect();
        let src = upload(&session, &ints);
        for (op, expected) in [
            (
                ReduceOp::Sum,
                ints.iter().fold(0i32, |acc, &v| acc.wrapping_add(v)),
            ),
            (
                ReduceOp::Prod,
                ints.iter().fold(1i32, |acc, &v| acc.wrapping_mul(v)),
            ),
            (ReduceOp::Max, *ints.iter().max().unwrap()),
            (ReduceOp::Min, *ints.iter().min().unwrap()),
        ] {
            let bytes = run_reduction(
                &session,
                DType::I32,
                op,
                strategy,
                &src,
                &in_layout,
                len,
                &out_layout,
                1,
                0,
                32,
            );
            assert_eq!(
                from_bytes::<i32>(&bytes)[0],
                expected,
                "{op:?} I32 len {len}"
            );
        }

        let longs: Vec<i64> = (0..len)
            .map(|index| {
                index.wrapping_mul(6364136223846793005).wrapping_sub(7) ^ 0x0BAD_F00D_DEAD_BEEF
            })
            .collect();
        let src = upload(&session, &longs);
        for (op, expected) in [
            (
                ReduceOp::Sum,
                longs.iter().fold(0i64, |acc, &v| acc.wrapping_add(v)),
            ),
            (
                ReduceOp::Prod,
                longs.iter().fold(1i64, |acc, &v| acc.wrapping_mul(v)),
            ),
        ] {
            let bytes = run_reduction(
                &session,
                DType::I64,
                op,
                strategy,
                &src,
                &in_layout,
                len,
                &out_layout,
                1,
                0,
                32,
            );
            assert_eq!(
                from_bytes::<i64>(&bytes)[0],
                expected,
                "{op:?} I64 len {len}"
            );
        }

        // Complex sum and product, C32 and C64.
        let complex: Vec<C64> = (0..len)
            .map(|index| C64 {
                re: (index as f64 - 8.0) * 0.25,
                im: (index as f64 * 0.5) - 4.0,
            })
            .collect();
        for dtype in [DType::C32, DType::C64] {
            let src = if dtype == DType::C32 {
                let narrowed: Vec<(f32, f32)> = complex
                    .iter()
                    .map(|value| (value.re as f32, value.im as f32))
                    .collect();
                upload(&session, &narrowed)
            } else {
                upload(&session, &complex)
            };
            for op in [ReduceOp::Sum, ReduceOp::Prod] {
                let bytes = run_reduction(
                    &session,
                    dtype,
                    op,
                    strategy,
                    &src,
                    &in_layout,
                    len,
                    &out_layout,
                    1,
                    0,
                    32,
                );
                let (re, im) = if op == ReduceOp::Sum {
                    (
                        complex.iter().map(|v| v.re).sum::<f64>(),
                        complex.iter().map(|v| v.im).sum::<f64>(),
                    )
                } else {
                    complex.iter().fold((1.0, 0.0), |(ar, ai), v| {
                        (ar * v.re - ai * v.im, ar * v.im + ai * v.re)
                    })
                };
                let (got_re, got_im) = if dtype == DType::C32 {
                    let values = from_bytes::<(f32, f32)>(&bytes);
                    (f64::from(values[0].0), f64::from(values[0].1))
                } else {
                    let values = from_bytes::<C64>(&bytes);
                    (values[0].re, values[0].im)
                };
                assert!(
                    approx_within(got_re, re, tolerance(dtype))
                        && approx_within(got_im, im, tolerance(dtype)),
                    "{op:?} {dtype:?} len {len}: ({got_re}, {got_im}) vs ({re}, {im})"
                );
            }
        }
    }
}

#[test]
fn split_product_and_complex_splits_match_the_direct_result() {
    let session = Session::new("reduce");
    let reduce_kernel = session.function("t4a_cuda_reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");
    let len = 1024i64;
    // Small magnitudes keep the product finite; the point is split equivalence.
    let values: Vec<f64> = (0..len)
        .map(|index| 1.0 + (index % 5) as f64 * 0.125)
        .collect();
    let in_layout = Layout::compact(&[len]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let split = split_policy(len, 32).unwrap().unwrap();
    let partials_layout = Layout::compact(&[split.splits]).unwrap();
    let partials_len = partials_layout.element_count().unwrap();
    let src = upload(&session, &values);

    for op in [ReduceOp::Prod, ReduceOp::Sum] {
        let direct = from_bytes::<f64>(&run_reduction(
            &session,
            DType::F64,
            op,
            ReduceStrategy::Plane,
            &src,
            &in_layout,
            len,
            &out_layout,
            1,
            0,
            32,
        ))[0];

        let out = sentinel(&session, 8);
        let partials = sentinel(&session, partials_len as usize * 8);
        let plan = plan_reduction(DType::F64, op, &out_layout, 1, &in_layout, len, 0, 32).unwrap();
        let ReductionPlan::Split {
            pass1,
            pass2,
            combine_op,
            combine_strategy,
            ..
        } = plan
        else {
            panic!("must split");
        };
        let mut slots1 = partials_slots(
            &partials,
            &src,
            DType::F64,
            op,
            &CLayout::new(&partials_layout),
            &CLayout::new(&in_layout),
            0,
            1,
            split.splits,
            split.chunk_len,
        );
        launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
        let mut slots2 = reduce_slots(
            &out,
            &partials,
            DType::F64,
            combine_op,
            combine_strategy,
            &CLayout::new(&out_layout),
            &CLayout::new(&partials_layout),
            0,
            1,
        );
        launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
        pub_sync(session.stream());
        let split_result = download::<f64>(&session, &out, 1)[0];
        assert!(
            approx(split_result, direct),
            "{op:?}: split {split_result} vs direct {direct}"
        );
    }
}

#[test]
fn middle_axis_split_handles_gapped_partials_and_a_strided_output() {
    let session = Session::new("reduce");
    let reduce_kernel = session.function("t4a_cuda_reduce");
    let partials_kernel = session.function("t4a_cuda_reduce_partials");
    // Reduce axis 1 of [3, 512, 2] with a non-compact input and a gapped,
    // offset partials allocation whose holes must stay untouched.
    let extents = [3i64, 512, 2];
    let in_strides = [1i64, 6, 3072];
    let in_offset = 1i64;
    let total = 3 * 512 * 2;
    let buffer: Vec<f64> = (0..total * 2 + 8)
        .map(|index| ((index % 53) as f64 - 26.0) * 0.125)
        .collect();
    let in_layout = Layout::new(&extents, &in_strides, in_offset).unwrap();
    let src = upload(&session, &buffer);
    let values = gathered(&buffer, &extents, &in_strides, in_offset);

    // Non-compact output: [3, 1, 2] with strides [1, 3, 3] and offset 4.
    let mut out_extents = extents.to_vec();
    out_extents[1] = 1;
    let out_layout = Layout::new(&out_extents, &[1, 3, 3], 4).unwrap();
    let output_len = out_layout.element_count().unwrap();
    let split = split_policy(extents[1], 32).unwrap().unwrap();
    // Gapped partials: axis 1 is the fastest axis of the scratch allocation.
    let partials_extents = [3i64, split.splits, 2];
    let partials_strides = [2i64, 6, 6 * split.splits];
    let partials_offset = 3i64;
    let partials_layout =
        Layout::new(&partials_extents, &partials_strides, partials_offset).unwrap();
    let partials_span = partials_layout.max_reachable_offset().unwrap() + 1;
    let alloc = partials_span as usize + 4;

    let out = sentinel(&session, 32 * 8);
    let partials = sentinel(&session, alloc * 8);
    // The per-pass planners validate this test's own gapped partials layout.
    let pass1 = plan_reduce_partials(
        DType::F64,
        ReduceOp::Sum,
        &partials_layout,
        alloc as i64,
        &in_layout,
        buffer.len() as i64,
        1,
        split,
        32,
    )
    .unwrap();
    let pass2 = plan_reduce(
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &out_layout,
        32,
        &partials_layout,
        alloc as i64,
        1,
        32,
    )
    .unwrap();
    let mut slots1 = partials_slots(
        &partials,
        &src,
        DType::F64,
        ReduceOp::Sum,
        &CLayout::new(&partials_layout),
        &CLayout::new(&in_layout),
        1,
        output_len,
        split.splits,
        split.chunk_len,
    );
    launch_async(partials_kernel, &pass1, session.stream(), &mut slots1);
    let mut slots2 = reduce_slots(
        &out,
        &partials,
        DType::F64,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &CLayout::new(&out_layout),
        &CLayout::new(&partials_layout),
        1,
        output_len,
    );
    launch_async(reduce_kernel, &pass2, session.stream(), &mut slots2);
    pub_sync(session.stream());

    // Every logical partial equals the sum of its chunk.
    let partial_bytes = read_bytes(&session, &partials);
    let partial_values: Vec<f64> = from_bytes(&partial_bytes);
    for i in 0..3usize {
        for k in 0..2usize {
            for chunk in 0..split.splits as usize {
                let start = chunk as i64 * split.chunk_len;
                let end = if chunk as i64 + 1 == split.splits {
                    extents[1]
                } else {
                    start + split.chunk_len
                };
                let mut expected = 0.0;
                for j in start..end {
                    let mut flat = 0i64;
                    let mut stride = 1i64;
                    for (coordinate, dim) in [(i as i64, 0usize), (j, 1), (k as i64, 2)] {
                        flat += coordinate * stride;
                        stride *= extents[dim];
                    }
                    expected += values[flat as usize];
                }
                let index = partials_offset
                    + i as i64 * partials_strides[0]
                    + chunk as i64 * partials_strides[1]
                    + k as i64 * partials_strides[2];
                assert!(
                    approx(partial_values[index as usize], expected),
                    "partial ({i},{chunk},{k}): {} vs {expected}",
                    partial_values[index as usize]
                );
            }
        }
    }
    // Holes of the partials allocation remain the sentinel.
    let touched: std::collections::HashSet<usize> = (0..3i64)
        .flat_map(|i| {
            (0..split.splits).flat_map(move |chunk| {
                (0..2i64).map(move |k| {
                    (partials_offset
                        + i * partials_strides[0]
                        + chunk * partials_strides[1]
                        + k * partials_strides[2]) as usize
                })
            })
        })
        .collect();
    for element in 0..alloc {
        if touched.contains(&element) {
            continue;
        }
        assert_eq!(
            &partial_bytes[element * 8..(element + 1) * 8],
            &[SENTINEL; 8],
            "hole {element}"
        );
    }

    // And the final result matches the whole-axis sums at the strided output.
    let out_bytes = read_bytes(&session, &out);
    let out_values: Vec<f64> = from_bytes(&out_bytes);
    for i in 0..3usize {
        for k in 0..2usize {
            let mut expected = 0.0;
            for j in 0..extents[1] {
                let mut flat = 0i64;
                let mut stride = 1i64;
                for (coordinate, dim) in [(i as i64, 0usize), (j, 1), (k as i64, 2)] {
                    flat += coordinate * stride;
                    stride *= extents[dim];
                }
                expected += values[flat as usize];
            }
            let index = out_layout.offset()
                + i as i64 * out_layout.strides()[0]
                + k as i64 * out_layout.strides()[2];
            assert!(
                approx(out_values[index as usize], expected),
                "output ({i},{k})"
            );
        }
    }
}

#[test]
fn sum_squares_rounds_before_adding() {
    let session = Session::new("reduce");
    // 0x3fb504f3 ≈ 1.4142135 and 0x3f800001 ≈ 1.0000001. Squaring each value and
    // then adding gives exactly 3.0 with separately rounded squares; contracting
    // the second square into the addition changes the result.
    let values: Vec<f32> = vec![f32::from_bits(0x3fb5_04f3), f32::from_bits(0x3f80_0001)];
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&[2]).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let bytes = run_reduction(
        &session,
        DType::F32,
        ReduceOp::SumSquares,
        ReduceStrategy::Unit,
        &src,
        &in_layout,
        2,
        &out_layout,
        1,
        0,
        32,
    );
    let got = from_bytes::<f32>(&bytes)[0];
    let square0 = f32::from_bits(0x3fb5_04f3) * f32::from_bits(0x3fb5_04f3);
    let square1 = f32::from_bits(0x3f80_0001) * f32::from_bits(0x3f80_0001);
    let expected = square0 + square1;
    assert_eq!(got.to_bits(), expected.to_bits(), "{got} vs {expected}");
}
