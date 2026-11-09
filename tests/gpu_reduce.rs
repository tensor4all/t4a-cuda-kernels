//! Numerical GPU tests for the revision-3 reduction entries. Run with
//! `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{
    download, from_bytes, launch, launch_async, launches, pub_sync, read_bytes, upload, written,
    Buf, Session, Slot,
};
use t4a_cuda_kernels::{
    plan_reduce, plan_reduce_partials, plan_reduction, split_policy, CLayout, CLayoutAny, DType,
    Layout, ReduceOp, ReduceStrategy, ReductionPlan,
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

/// A dtype the wide-combine tests drive: the tag the kernel takes.
trait Probe: Copy {
    const DTYPE: DType;
}

impl Probe for f64 {
    const DTYPE: DType = DType::F64;
}

impl Probe for f32 {
    const DTYPE: DType = DType::F32;
}

impl Probe for i32 {
    const DTYPE: DType = DType::I32;
}

impl Probe for C64 {
    const DTYPE: DType = DType::C64;
}

/// Run a whole reduction exactly as its plan declares it — both passes, including the
/// combine pass's own strategy and block width — and return the outputs.
fn run_planned<T: Probe>(
    session: &Session,
    values: &[T],
    extents: &[i64],
    axis: usize,
    op: ReduceOp,
) -> (Vec<T>, ReductionPlan) {
    let reduce = session.function("t4a_cuda_reduce");
    let partials_fn = session.function("t4a_cuda_reduce_partials");
    let in_layout = Layout::compact(extents).unwrap();
    let mut out_extents = extents.to_vec();
    out_extents[axis] = 1;
    let out_layout = Layout::compact(&out_extents).unwrap();
    let output_len = out_layout.element_count().unwrap();
    let in_len = in_layout.element_count().unwrap();
    let plan = plan_reduction(
        T::DTYPE,
        op,
        &out_layout,
        output_len,
        &in_layout,
        in_len,
        axis as u32,
        32,
    )
    .unwrap();
    let src = upload(session, values);
    let out = sentinel(session, output_len as usize * std::mem::size_of::<T>());
    match &plan {
        ReductionPlan::Split {
            split,
            pass1,
            pass2,
            combine_strategy,
            combine_op,
        } => {
            let partials_extents =
                t4a_cuda_kernels::reduction_partials_layout(&in_layout, axis as u32, split.splits)
                    .unwrap()
                    .extents()
                    .to_vec();
            let partials_layout = Layout::compact(&partials_extents).unwrap();
            let partials_len = partials_layout.element_count().unwrap();
            let partials = sentinel(session, partials_len as usize * std::mem::size_of::<T>());
            let mut slots1 = partials_slots(
                &partials,
                &src,
                T::DTYPE,
                op,
                &CLayout::new(&partials_layout),
                &CLayout::new(&in_layout),
                axis as i32,
                output_len,
                split.splits,
                split.chunk_len,
            );
            launch_async(partials_fn, pass1, session.stream(), &mut slots1);
            let mut slots2 = reduce_slots(
                &out,
                &partials,
                T::DTYPE,
                *combine_op,
                *combine_strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&partials_layout),
                axis as i32,
                output_len,
            );
            launch_async(reduce, pass2, session.stream(), &mut slots2);
            pub_sync(session.stream());
        }
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = reduce_slots(
                &out,
                &src,
                T::DTYPE,
                op,
                *strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&in_layout),
                axis as i32,
                output_len,
            );
            launch_async(reduce, launch, session.stream(), &mut slots);
            pub_sync(session.stream());
        }
    }
    (download::<T>(session, &out, output_len as usize), plan)
}

#[test]
fn wide_combine_folds_every_partial_for_every_wide_width() {
    // One axis length per wide block width. Each output's only large value sits in a different
    // late partial, so a lane or a warp the combine skipped would change that output's result.
    let session = Session::new("reduce");
    let outputs = 4i64;
    // The width follows `COMBINE_TARGET_FOLDS_PER_LANE = 8`, so it steps at 1024/2048/4096
    // partials; 512 partials still use the narrowest wide block.
    for (splits, want_block) in [(512u32, 64u32), (1_024, 128), (2_048, 256), (4_096, 512)] {
        let axis_len = i64::from(splits) * 256;
        let extents = [axis_len, outputs];
        let mut values = vec![0.0f32; (axis_len * outputs) as usize];
        for output in 0..outputs {
            // Late in the partials order, and a different partial per output.
            let last = axis_len - 1 - 40 * output;
            values[(last + axis_len * output) as usize] = 100.0 + output as f32;
        }
        let (got, plan) = run_planned(&session, &values, &extents, 0, ReduceOp::Max);
        let ReductionPlan::Split {
            split,
            combine_strategy,
            pass2,
            ..
        } = plan
        else {
            panic!("a {axis_len}-long axis must split");
        };
        assert_eq!(split.splits, i64::from(splits));
        assert_eq!(combine_strategy, ReduceStrategy::Plane);
        assert_eq!(
            pass2.block,
            [want_block, 1, 1],
            "{splits} partials must select a {want_block}-thread combine block"
        );
        for (output, value) in got.iter().enumerate() {
            assert_eq!(
                *value,
                100.0 + output as f32,
                "{splits} partials, output {output}: every partial must reach the combine"
            );
        }
    }
}

#[test]
fn wide_combine_matches_the_cpu_reference_for_each_dtype() {
    let session = Session::new("reduce");
    let axis_len = 131_072i64;
    let f64_values: Vec<f64> = (0..axis_len)
        .map(|index| ((index % 89) as f64 - 44.0) * 0.5)
        .collect();
    let (got, _) = run_planned(&session, &f64_values, &[axis_len], 0, ReduceOp::Sum);
    let expected: f64 = f64_values.iter().sum();
    assert!(approx_within(got[0], expected, 1e-12), "{}", got[0]);

    let i32_values: Vec<i32> = (0..axis_len)
        .map(|index| (index % 23) as i32 - 11)
        .collect();
    let (got, _) = run_planned(&session, &i32_values, &[axis_len], 0, ReduceOp::Sum);
    let expected: i64 = i32_values.iter().map(|&v| i64::from(v)).sum();
    assert_eq!(i64::from(got[0]), expected, "integer wide combine");

    let c64_values: Vec<C64> = (0..axis_len)
        .map(|index| C64 {
            re: ((index % 31) as f64 - 15.0) * 0.25,
            im: ((index % 17) as f64 - 8.0) * 0.5,
        })
        .collect();
    let (got, _) = run_planned(&session, &c64_values, &[axis_len], 0, ReduceOp::Sum);
    let re: f64 = c64_values.iter().map(|v| v.re).sum();
    let im: f64 = c64_values.iter().map(|v| v.im).sum();
    assert!(
        approx(got[0].re, re) && approx(got[0].im, im),
        "complex wide combine"
    );
}

// --- Vectorized pass-1 walk (docs/design/slice-reduction-pass1-vector.md) ------------------
//
// Pass 1 reads a contiguous chunk of a `SUM`/`SUM_SQUARES`/`MAX`/`MIN` reduction over
// F32/F64/I32/I64 with 16-byte loads. Every chunk must still fold each of its elements
// exactly once, wherever the chunk starts relative to a 16-byte boundary and however many
// elements are left over after the last whole vector.

impl Probe for i64 {
    const DTYPE: DType = DType::I64;
}

/// An element type of the vector walk with an exact `f64` view.
trait WalkValue: Probe + std::fmt::Debug {
    fn from_index(index: i64) -> Self;
    fn widen(self) -> f64;
    fn floating() -> bool;
}

impl WalkValue for f32 {
    fn from_index(index: i64) -> f32 {
        ((index * 7_919) % 1_009) as f32 / 1_009.0 - 0.5
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn floating() -> bool {
        true
    }
}

impl WalkValue for f64 {
    fn from_index(index: i64) -> f64 {
        ((index * 7_919) % 1_009) as f64 / 1_009.0 - 0.5
    }
    fn widen(self) -> f64 {
        self
    }
    fn floating() -> bool {
        true
    }
}

impl WalkValue for i32 {
    fn from_index(index: i64) -> i32 {
        ((index * 7_919) % 1_009) as i32 - 504
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn floating() -> bool {
        false
    }
}

impl WalkValue for i64 {
    fn from_index(index: i64) -> i64 {
        (index * 7_919) % 1_009 - 504
    }
    fn widen(self) -> f64 {
        self as f64
    }
    fn floating() -> bool {
        false
    }
}

/// The CPU reference of one fold over `values`, in `f64`. The integer inputs above keep
/// every sum far inside the range where `f64` is exact.
fn walk_reference(values: &[f64], op: ReduceOp) -> f64 {
    let nan_aware = |pick: fn(f64, f64) -> f64| {
        move |acc: f64, value: f64| {
            if acc.is_nan() || value.is_nan() {
                f64::NAN
            } else {
                pick(acc, value)
            }
        }
    };
    match op {
        ReduceOp::Sum => values.iter().sum(),
        ReduceOp::SumSquares => values.iter().map(|value| value * value).sum(),
        ReduceOp::Max => values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, nan_aware(f64::max)),
        ReduceOp::Min => values
            .iter()
            .copied()
            .fold(f64::INFINITY, nan_aware(f64::min)),
        ReduceOp::Prod => values.iter().product(),
    }
}

/// Whether a device value matches the reference: exactly for integers and extrema (which
/// select one of their inputs), and within the a-priori `n * u * sum|x|` bound any
/// association obeys for a floating sum.
fn walk_matches<T: WalkValue>(got: T, values: &[f64], op: ReduceOp) -> bool {
    let expected = walk_reference(values, op);
    let got = got.widen();
    if !T::floating() || matches!(op, ReduceOp::Max | ReduceOp::Min) {
        return got == expected || (got.is_nan() && expected.is_nan());
    }
    let unit = if T::DTYPE == DType::F32 {
        f64::from(f32::EPSILON) / 2.0
    } else {
        f64::EPSILON / 2.0
    };
    let magnitude: f64 = match op {
        ReduceOp::SumSquares => values.iter().map(|value| value * value).sum(),
        _ => values.iter().map(|value| value.abs()).sum(),
    };
    (got - expected).abs() <= (values.len() as f64 + 1.0) * unit * magnitude
}

/// A rank-1 reduction of `len` elements starting `offset` elements into `values`, run as
/// its plan declares. Returns the pass-1 partials (empty for a direct plan), the split and
/// the final value.
fn run_walk<T: WalkValue>(
    session: &Session,
    src: &Buf,
    buffer_len: i64,
    offset: i64,
    len: i64,
    op: ReduceOp,
    plane_width: u32,
) -> (Vec<T>, Option<(i64, i64)>, T) {
    let in_layout = Layout::new(&[len], &[1], offset).unwrap();
    let out_layout = Layout::compact(&[1]).unwrap();
    let size = std::mem::size_of::<T>();
    let out = sentinel(session, size);
    let plan = plan_reduction(
        T::DTYPE,
        op,
        &out_layout,
        1,
        &in_layout,
        buffer_len,
        0,
        plane_width,
    )
    .unwrap();
    match plan {
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = reduce_slots(
                &out,
                src,
                T::DTYPE,
                op,
                strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&in_layout),
                0,
                1,
            );
            launch_async(
                session.function("t4a_cuda_reduce"),
                &launch,
                session.stream(),
                &mut slots,
            );
            pub_sync(session.stream());
            (Vec::new(), None, download::<T>(session, &out, 1)[0])
        }
        ReductionPlan::Split {
            split,
            pass1,
            pass2,
            combine_strategy,
            combine_op,
        } => {
            let partials_layout = Layout::compact(&[split.splits]).unwrap();
            // Sentinel-filled: a partial pass 1 never wrote stays visible.
            let partials = sentinel(session, split.splits as usize * size);
            let mut slots1 = partials_slots(
                &partials,
                src,
                T::DTYPE,
                op,
                &CLayout::new(&partials_layout),
                &CLayout::new(&in_layout),
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
                &out,
                &partials,
                T::DTYPE,
                combine_op,
                combine_strategy,
                &CLayout::new(&out_layout),
                &CLayout::new(&partials_layout),
                0,
                1,
            );
            launch_async(
                session.function("t4a_cuda_reduce"),
                &pass2,
                session.stream(),
                &mut slots2,
            );
            pub_sync(session.stream());
            (
                download::<T>(session, &partials, split.splits as usize),
                Some((split.splits, split.chunk_len)),
                download::<T>(session, &out, 1)[0],
            )
        }
    }
}

/// Lengths on both sides of the split threshold (512 with a 32-lane plane), with tails of
/// every size modulo the 4- and 2-element vectors and the 128-element warp stripe, a
/// 257-element floor chunk (1028 = 4 x 257, so split starts fall off the 16-byte grid),
/// and the `MAX_SPLITS` cap with a remainder. At the cap the chunk grows with the length
/// (1024 elements at 4M + 5), so lanes run whole load batches. 917 507 gives 224-element
/// chunks at an 8-lane plane: 56 groups of four F32, 7 per lane, so the last partial batch
/// holds three groups (F64 reaches the same at a 16-lane plane).
const WALK_LENGTHS: [i64; 13] = [
    1,
    3,
    511,
    512,
    515,
    1_028,
    1_031,
    4_099,
    65_549,
    917_507,
    1 << 20,
    (1 << 20) + 7,
    (1 << 22) + 5,
];

fn check_walk<T: WalkValue>(session: &Session, ops: &[ReduceOp]) {
    // Three spare elements after the largest view, so offsets 1..3 stay in bounds.
    let buffer_len = (1i64 << 22) + 5 + 3;
    let host: Vec<T> = (0..buffer_len).map(T::from_index).collect();
    let widened: Vec<f64> = host.iter().map(|value| value.widen()).collect();
    let src = upload(session, &host);
    assert_eq!(
        src.as_ptr() as usize % 16,
        0,
        "device allocations are 16-byte aligned"
    );
    for &len in &WALK_LENGTHS {
        // Offsets 1..3 move the first element off the 16-byte grid by 4/8/12 bytes (F32/I32)
        // or 8/0/8 bytes (F64/I64), so groups load as one 16-byte vector or as elements.
        for (offset, plane_width) in
            (0..4i64).flat_map(|offset| [(offset, 8u32), (offset, 16), (offset, 32)])
        {
            let view = &widened[offset as usize..(offset + len) as usize];
            for &op in ops {
                let (partials, split, total) =
                    run_walk::<T>(session, &src, buffer_len, offset, len, op, plane_width);
                let context = format!(
                    "{:?} {op:?} len {len} offset {offset} plane {plane_width}",
                    T::DTYPE
                );
                if let Some((splits, chunk_len)) = split {
                    for (chunk, &got) in partials.iter().enumerate() {
                        let start = chunk as i64 * chunk_len;
                        let end = if chunk as i64 + 1 == splits {
                            len
                        } else {
                            start + chunk_len
                        };
                        assert!(
                            walk_matches(got, &view[start as usize..end as usize], op),
                            "{context}: partial {chunk} of {splits} is {got:?}"
                        );
                    }
                }
                assert!(walk_matches(total, view, op), "{context}: result {total:?}");
            }
        }
    }
}

#[test]
fn vector_walk_folds_every_element_once_for_every_op_and_dtype() {
    let session = Session::new("reduce");
    let floating = [
        ReduceOp::Sum,
        ReduceOp::SumSquares,
        ReduceOp::Max,
        ReduceOp::Min,
    ];
    let integer = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min];
    check_walk::<f32>(&session, &floating);
    check_walk::<f64>(&session, &floating);
    check_walk::<i32>(&session, &integer);
    check_walk::<i64>(&session, &integer);
}

#[test]
fn vector_walk_propagates_a_nan_from_the_first_group_the_body_and_the_tail() {
    // 1028 = 4 chunks of 257 elements: 64 groups of four and a one-element tail (F32), and
    // with offsets 1..3 the chunk starts are off the 16-byte grid. A NaN at a chunk's first
    // element lands in its first group, one in the middle in a body group, and one at its
    // last element in its tail; every placement must reach that chunk's partial and the
    // result, and no other partial.
    let session = Session::new("reduce");
    let len = 1_028i64;
    for offset in [0i64, 1, 2, 3] {
        for chunk in 0..4i64 {
            for position in [0i64, 128, 256] {
                let nan_at = offset + chunk * 257 + position;
                let host32: Vec<f32> = (0..len + 4)
                    .map(|index| {
                        if index == nan_at {
                            f32::NAN
                        } else {
                            f32::from_index(index)
                        }
                    })
                    .collect();
                let host64: Vec<f64> = host32.iter().map(|&value| f64::from(value)).collect();
                for op in [ReduceOp::Max, ReduceOp::Min] {
                    let src = upload(&session, &host32);
                    let (partials, _, total) =
                        run_walk::<f32>(&session, &src, len + 4, offset, len, op, 32);
                    let src = upload(&session, &host64);
                    let (partials64, _, total64) =
                        run_walk::<f64>(&session, &src, len + 4, offset, len, op, 32);
                    let context =
                        format!("{op:?} offset {offset} chunk {chunk} position {position}");
                    assert!(
                        total.is_nan() && total64.is_nan(),
                        "{context}: result drops the NaN"
                    );
                    for other in 0..4usize {
                        assert_eq!(
                            partials[other].is_nan(),
                            other as i64 == chunk,
                            "{context}: f32 partial {other}"
                        );
                        assert_eq!(
                            partials64[other].is_nan(),
                            other as i64 == chunk,
                            "{context}: f64 partial {other}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn vector_walk_keeps_a_negative_zero_sum() {
    // The vector walk seeds each lane from its first vector rather than from an identity,
    // so a sum of negative zeros stays -0.0 (an identity of +0.0 would turn it into +0.0).
    let session = Session::new("reduce");
    let len = 1_031i64;
    let src = upload(&session, &vec![-0.0f32; len as usize + 1]);
    for offset in [0i64, 1] {
        let (partials, _, total) =
            run_walk::<f32>(&session, &src, len + 1, offset, len, ReduceOp::Sum, 32);
        for (chunk, value) in partials.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                (-0.0f32).to_bits(),
                "offset {offset} partial {chunk}"
            );
        }
        assert_eq!(
            total.to_bits(),
            (-0.0f32).to_bits(),
            "offset {offset} result"
        );
    }
}

#[test]
fn product_partials_keep_the_serial_lane_walk_bit_for_bit() {
    // `PROD` is never regrouped (c-kernel-abi.md, "Reassociation"): each lane multiplies
    // `start + t + k*32` in order, then the plane folds lanes with the xor tree. Emulate
    // exactly that in `f32` and require the same bits, on a contiguous axis where the
    // regroupable ops take the vector walk.
    let session = Session::new("reduce");
    for (len, offset) in [(1_028i64, 1i64), (65_549, 0), (65_549, 3)] {
        let host: Vec<f32> = (0..len + offset)
            .map(|index| 1.0 + (f32::from_index(index)) * 1e-3)
            .collect();
        let src = upload(&session, &host);
        let (partials, split, _) = run_walk::<f32>(
            &session,
            &src,
            len + offset,
            offset,
            len,
            ReduceOp::Prod,
            32,
        );
        let (splits, chunk_len) = split.expect("these lengths split");
        let view = &host[offset as usize..];
        for (chunk, got) in partials.iter().enumerate() {
            let start = chunk as i64 * chunk_len;
            let end = if chunk as i64 + 1 == splits {
                len
            } else {
                start + chunk_len
            };
            let mut lanes: Vec<f32> = (0..32i64)
                .map(|lane| {
                    let mut index = start + lane;
                    let mut acc = view[index as usize];
                    index += 32;
                    while index < end {
                        acc *= view[index as usize];
                        index += 32;
                    }
                    acc
                })
                .collect();
            let mut step = 1;
            while step < 32 {
                lanes = (0..32)
                    .map(|lane| lanes[lane] * lanes[lane ^ step])
                    .collect();
                step <<= 1;
            }
            assert_eq!(
                got.to_bits(),
                lanes[0].to_bits(),
                "len {len} offset {offset} partial {chunk}: {got} vs {}",
                lanes[0]
            );
        }
    }
}

/// One rank-1 reduction of `values` placed `lead` elements into a fresh buffer, so the
/// input pointer the kernel sees is `lead * size_of::<T>()` bytes past a 16-byte boundary.
fn reduce_at_lead<T: WalkValue>(
    session: &Session,
    values: &[T],
    lead: usize,
    op: ReduceOp,
    plane_width: u32,
) -> (Vec<T>, T) {
    let mut host = vec![values[0]; lead];
    host.extend_from_slice(values);
    let src = upload(session, &host);
    assert_eq!(src.as_ptr() as usize % 16, 0);
    let len = values.len() as i64;
    let (partials, _, total) = run_walk::<T>(
        session,
        &src,
        host.len() as i64,
        lead as i64,
        len,
        op,
        plane_width,
    );
    (partials, total)
}

#[test]
fn pass1_topology_does_not_depend_on_the_input_address() {
    // Revision 12 promises a topology fixed by input, layout, plan and build. Two inputs
    // that differ only in where their first element sits relative to a 16-byte boundary
    // must therefore give the same bits.
    let session = Session::new("reduce");
    // GPT-6.1-sol review of #23: `[A, A, -A, -A, 0, ...]` with A = 2^127 over 1024 F32
    // elements. A walk whose vector grid follows the address pairs `A + A` on one
    // alignment (+Inf, then NaN against -Inf) and `A + -A` on another (0).
    let big = 2f32.powi(127);
    let mut witness = vec![0.0f32; 1_024];
    witness[..4].copy_from_slice(&[big, big, -big, -big]);
    let (partials0, total0) = reduce_at_lead(&session, &witness, 0, ReduceOp::Sum, 32);
    for lead in 1..4 {
        let (partials, total) = reduce_at_lead(&session, &witness, lead, ReduceOp::Sum, 32);
        assert_eq!(
            total.to_bits(),
            total0.to_bits(),
            "lead {lead}: {total} vs {total0} at lead 0"
        );
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&partials), bits(&partials0), "lead {lead}: partials");
    }

    // Ordinary data at several lengths, plane widths and ops: every lead gives the bits of
    // lead 0, for every partial and the result.
    for &len in &[1_028usize, 4_099, 65_549] {
        let f32_values: Vec<f32> = (0..len as i64).map(f32::from_index).collect();
        let f64_values: Vec<f64> = (0..len as i64).map(f64::from_index).collect();
        for plane_width in [8u32, 32] {
            for op in [ReduceOp::Sum, ReduceOp::SumSquares, ReduceOp::Max] {
                let reference = reduce_at_lead(&session, &f32_values, 0, op, plane_width);
                for lead in 1..4 {
                    let got = reduce_at_lead(&session, &f32_values, lead, op, plane_width);
                    assert_eq!(
                        (
                            got.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            got.1.to_bits()
                        ),
                        (
                            reference.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            reference.1.to_bits()
                        ),
                        "F32 {op:?} len {len} plane {plane_width} lead {lead}"
                    );
                }
                let reference = reduce_at_lead(&session, &f64_values, 0, op, plane_width);
                let got = reduce_at_lead(&session, &f64_values, 1, op, plane_width);
                assert_eq!(
                    (
                        got.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        got.1.to_bits()
                    ),
                    (
                        reference.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        reference.1.to_bits()
                    ),
                    "F64 {op:?} len {len} plane {plane_width} lead 1"
                );
            }
        }
    }
}

#[test]
fn pass1_reads_broadcast_and_reversed_axes_inside_their_allocation() {
    // The two layouts of the review's first point, run through both passes on allocations
    // that hold exactly the reachable elements: a one-element input broadcast to 512
    // (stride 0) and a reversed 512-element view (stride -1, offset 511). Both take the
    // scalar walk.
    let session = Session::new("reduce");
    let out_layout = Layout::compact(&[1]).unwrap();
    let one = upload(&session, &[0.25f32]);
    let reversed_values: Vec<f32> = (0..512).map(f32::from_index).collect();
    let reversed = upload(&session, &reversed_values);
    for (src, buffer_len, layout, expected) in [
        (
            &one,
            1i64,
            Layout::new(&[512], &[0], 0).unwrap(),
            0.25f64 * 512.0,
        ),
        (
            &reversed,
            512,
            Layout::new(&[512], &[-1], 511).unwrap(),
            reversed_values.iter().map(|&v| f64::from(v)).sum::<f64>(),
        ),
    ] {
        let plan = plan_reduction(
            DType::F32,
            ReduceOp::Sum,
            &out_layout,
            1,
            &layout,
            buffer_len,
            0,
            32,
        )
        .unwrap();
        let ReductionPlan::Split {
            split,
            pass1,
            pass2,
            combine_strategy,
            combine_op,
        } = plan
        else {
            panic!("a 512-long axis splits");
        };
        let partials_layout = Layout::compact(&[split.splits]).unwrap();
        let partials = sentinel(&session, split.splits as usize * 4);
        let out = sentinel(&session, 4);
        let mut slots1 = partials_slots(
            &partials,
            src,
            DType::F32,
            ReduceOp::Sum,
            &CLayout::new(&partials_layout),
            &CLayout::new(&layout),
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
            &out,
            &partials,
            DType::F32,
            combine_op,
            combine_strategy,
            &CLayout::new(&out_layout),
            &CLayout::new(&partials_layout),
            0,
            1,
        );
        launch_async(
            session.function("t4a_cuda_reduce"),
            &pass2,
            session.stream(),
            &mut slots2,
        );
        pub_sync(session.stream());
        let got = f64::from(download::<f32>(&session, &out, 1)[0]);
        assert!(
            (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "stride {:?}: {got} vs {expected}",
            layout.strides()
        );
    }
}

#[test]
fn pass1_folds_nothing_outside_its_chunk() {
    // NaN guards on both sides of the view: a lane that folds a predicated-off batch entry, a
    // tail element it does not own, or an element of a neighbouring chunk or of the padding
    // would turn the result (or that chunk's partial) into NaN. Every partial must equal its
    // own chunk's reference, so no lane without work contributes anything.
    let session = Session::new("reduce");
    let guard = 64usize;
    for &len in &[512usize, 515, 1_028, 1_031, 4_099, 65_549, 917_507] {
        let f32_view: Vec<f32> = (0..len as i64).map(f32::from_index).collect();
        let f64_view: Vec<f64> = (0..len as i64).map(f64::from_index).collect();
        for lead in [0usize, 1] {
            let mut f32_host = vec![f32::NAN; guard + lead];
            f32_host.extend_from_slice(&f32_view);
            f32_host.extend(std::iter::repeat_n(f32::NAN, guard));
            let mut f64_host = vec![f64::NAN; guard + lead];
            f64_host.extend_from_slice(&f64_view);
            f64_host.extend(std::iter::repeat_n(f64::NAN, guard));
            let f32_src = upload(&session, &f32_host);
            let f64_src = upload(&session, &f64_host);
            let widened32: Vec<f64> = f32_view.iter().map(|&v| f64::from(v)).collect();
            for plane_width in [8u32, 16, 32] {
                for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
                    let offset = (guard + lead) as i64;
                    let (partials, split, total) = run_walk::<f32>(
                        &session,
                        &f32_src,
                        f32_host.len() as i64,
                        offset,
                        len as i64,
                        op,
                        plane_width,
                    );
                    let context = format!("F32 {op:?} len {len} lead {lead} plane {plane_width}");
                    assert!(walk_matches(total, &widened32, op), "{context}: {total}");
                    if let Some((splits, chunk_len)) = split {
                        for (chunk, &got) in partials.iter().enumerate() {
                            let start = chunk * chunk_len as usize;
                            let end = if chunk as i64 + 1 == splits {
                                len
                            } else {
                                start + chunk_len as usize
                            };
                            assert!(
                                walk_matches(got, &widened32[start..end], op),
                                "{context}: partial {chunk} is {got}"
                            );
                        }
                    }
                    let (_, _, total) = run_walk::<f64>(
                        &session,
                        &f64_src,
                        f64_host.len() as i64,
                        offset,
                        len as i64,
                        op,
                        plane_width,
                    );
                    assert!(
                        walk_matches(total, &f64_view, op),
                        "F64 {op:?} len {len} lead {lead} plane {plane_width}: {total}"
                    );
                }
            }
        }
    }
}

#[test]
fn pass1_adds_exactly_the_one_nonzero_element() {
    // All zeros except one 1.0, placed at 13 chosen positions (F32, plane width 32, input
    // leads 0-3; not an exhaustive sweep) that cover the first group, a group interior, the
    // last whole group and the tail of a chunk: the sum is exactly 1, the partial of the chunk that
    // holds the element is exactly 1 and every other partial is exactly +0.0. An empty lane or
    // an unloaded batch slot that contributed anything would break the zeros.
    let session = Session::new("reduce");
    let len = 1_031usize; // 4 chunks: 257, 257, 257, 260 elements.
    for lead in [0usize, 1, 2, 3] {
        for position in [
            0usize, 1, 3, 4, 127, 252, 255, 256, 257, 770, 1_027, 1_028, 1_030,
        ] {
            let mut host = vec![0.0f32; lead + len];
            host[lead + position] = 1.0;
            let src = upload(&session, &host);
            let (partials, split, total) = run_walk::<f32>(
                &session,
                &src,
                host.len() as i64,
                lead as i64,
                len as i64,
                ReduceOp::Sum,
                32,
            );
            let (_, chunk_len) = split.expect("1031 elements split");
            let owner = (position / chunk_len as usize).min(partials.len() - 1);
            let context = format!("lead {lead} position {position}");
            assert_eq!(
                total.to_bits(),
                1.0f32.to_bits(),
                "{context}: result {total}"
            );
            for (chunk, value) in partials.iter().enumerate() {
                let want = if chunk == owner { 1.0f32 } else { 0.0 };
                assert_eq!(
                    value.to_bits(),
                    want.to_bits(),
                    "{context}: partial {chunk}"
                );
            }
        }
    }
}

#[test]
fn an_empty_reduction_axis_is_refused_before_any_launch() {
    // The kernels seed from the axis' first element and have no identity value, so an empty
    // axis never reaches them: the plan is refused, and the caller supplies the identity (0
    // for a sum; tenferro-rs does this as `EmptyFoldUnit::Zero`).
    let in_layout = Layout::compact(&[0, 4]).unwrap();
    let out_layout = Layout::compact(&[1, 4]).unwrap();
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        assert!(
            plan_reduction(DType::F32, op, &out_layout, 4, &in_layout, 0, 0, 32).is_err(),
            "{op:?}: an empty axis must not plan a launch"
        );
    }
}

#[test]
fn pass1_overflow_follows_the_revision_13_grouping() {
    // The two documented witnesses of c-kernel-abi.md "Pass-1 lane assignment (revision
    // 13)": F32 SUM of 1024 elements at plane width 32, A = 2^127, everything else zero.
    // Which pairs overflow depends on which elements share a lane, so these pin the
    // revision-13 grouping (groups of four consecutive elements) against the revision-12 one
    // (elements 32 apart), in both directions. The result is the same at every input lead.
    let session = Session::new("reduce");
    let big = 2f32.powi(127);
    let witness = |placement: [(usize, f32); 4]| {
        let mut values = vec![0.0f32; 1_024];
        for (index, value) in placement {
            values[index] = value;
        }
        values
    };
    // Same-group pairs overflow: A + A in group 0, -A + -A in group 8.
    let same_group = witness([(0, big), (1, big), (32, -big), (33, -big)]);
    // Same-group pairs cancel: A + -A in groups 0 and 8 (revision 12 put A with A).
    let cancelling = witness([(0, big), (1, -big), (32, big), (33, -big)]);
    for lead in 0..4 {
        let (_, total) = reduce_at_lead(&session, &same_group, lead, ReduceOp::Sum, 32);
        assert!(
            total.is_nan(),
            "lead {lead}: same-group overflow gives NaN, got {total}"
        );
        let (_, total) = reduce_at_lead(&session, &cancelling, lead, ReduceOp::Sum, 32);
        assert_eq!(
            total.to_bits(),
            0.0f32.to_bits(),
            "lead {lead}: cancelling groups give +0"
        );
    }
}

// --- Combine-pass serial walk, bit for bit (docs/design/slice-combine-load-ahead.md) -------
//
// The combine pass (`t4a_cuda_reduce`, strategy `PLANE`) may issue its loads ahead of the
// folds, but must fold exactly as the serial walk does: lane `t` seeds from element `t` and
// folds `t + k*W` in increasing `k`, then the xor tree combines the lanes (through shared
// memory across warps for a block wider than one warp). The emulation below is that order in
// IEEE arithmetic, with the device's observable NaN conventions: an f32 arithmetic NaN result
// is the canonical NaN, an f64 one keeps the payload of its first NaN operand (quieted), and
// the NaN-propagating extrema select an operand (payload kept, `lhs` first) and order
// `-0.0 < +0.0`. The f64 rule makes the operand order of every add and multiply observable. The inputs are chosen so that a different order changes the
// bits: NaN payloads, infinities, signed zeros, overflow and underflow mid-walk, subnormals.

/// A floating element type of the combine emulation.
trait CombineValue: Probe + PartialEq + std::fmt::Debug {
    const CANONICAL_NAN_BITS: u64;
    /// Whether an arithmetic NaN result keeps an operand's payload (f64) or is canonical (f32).
    const KEEPS_PAYLOAD: bool;
    const QUIET_BIT: u64;
    fn bits(self) -> u64;
    fn from_f64(value: f64) -> Self;
    fn from_bits64(bits: u64) -> Self;
    fn is_nan(self) -> bool;
    fn is_zero(self) -> bool;
    fn is_sign_positive(self) -> bool;
    fn raw_add(self, other: Self) -> Self;
    fn raw_mul(self, other: Self) -> Self;
    fn less(self, other: Self) -> bool;
    /// Largest finite value, smallest positive subnormal and the exponent range the
    /// near-overflow inputs use.
    fn max_finite() -> Self;
    fn min_subnormal() -> Self;
    fn max_exponent() -> i32;
    /// NaN with a distinct payload per `tag` (sign bit set when `negative`).
    fn payload_nan(tag: u64, negative: bool) -> Self;
}

impl CombineValue for f32 {
    const CANONICAL_NAN_BITS: u64 = 0x7fff_ffff;
    const KEEPS_PAYLOAD: bool = false;
    const QUIET_BIT: u64 = 0x0040_0000;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
    fn from_bits64(bits: u64) -> f32 {
        f32::from_bits(bits as u32)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn is_sign_positive(self) -> bool {
        f32::is_sign_positive(self)
    }
    fn raw_add(self, other: f32) -> f32 {
        self + other
    }
    fn raw_mul(self, other: f32) -> f32 {
        self * other
    }
    fn less(self, other: f32) -> bool {
        self < other
    }
    fn max_finite() -> f32 {
        f32::MAX
    }
    fn min_subnormal() -> f32 {
        f32::from_bits(1)
    }
    fn max_exponent() -> i32 {
        127
    }
    fn payload_nan(tag: u64, negative: bool) -> f32 {
        let sign = if negative { 0x8000_0000 } else { 0 };
        f32::from_bits(sign | 0x7fc0_0000 | (0x1234 + tag as u32))
    }
}

impl CombineValue for f64 {
    const CANONICAL_NAN_BITS: u64 = 0xfff8_0000_0000_0000;
    const KEEPS_PAYLOAD: bool = true;
    const QUIET_BIT: u64 = 0x0008_0000_0000_0000;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_f64(value: f64) -> f64 {
        value
    }
    fn from_bits64(bits: u64) -> f64 {
        f64::from_bits(bits)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn is_sign_positive(self) -> bool {
        f64::is_sign_positive(self)
    }
    fn raw_add(self, other: f64) -> f64 {
        self + other
    }
    fn raw_mul(self, other: f64) -> f64 {
        self * other
    }
    fn less(self, other: f64) -> bool {
        self < other
    }
    fn max_finite() -> f64 {
        f64::MAX
    }
    fn min_subnormal() -> f64 {
        f64::from_bits(1)
    }
    fn max_exponent() -> i32 {
        1023
    }
    fn payload_nan(tag: u64, negative: bool) -> f64 {
        let sign = if negative { 1u64 << 63 } else { 0 };
        f64::from_bits(sign | 0x7ff8_0000_0000_0000 | (0x1234_5678 + tag))
    }
}

/// Arithmetic as the device rounds it: IEEE round-to-nearest; a NaN result is canonical (f32)
/// or keeps the first NaN operand's payload, quieted (f64; canonical when neither operand is
/// NaN, as in `inf - inf`).
fn device_arith<T: CombineValue>(lhs: T, rhs: T, value: T) -> T {
    if !value.is_nan() {
        return value;
    }
    if T::KEEPS_PAYLOAD {
        for operand in [lhs, rhs] {
            if operand.is_nan() {
                return T::from_bits64(operand.bits() | T::QUIET_BIT);
            }
        }
    }
    T::from_bits64(T::CANONICAL_NAN_BITS)
}

/// `t4a_cuda_nan_max`/`t4a_cuda_nan_min`: a NaN `lhs`, then a NaN `rhs`, else
/// `fmax`/`fmin` with `-0.0 < +0.0`.
fn device_extremum<T: CombineValue>(lhs: T, rhs: T, max: bool) -> T {
    if lhs.is_nan() {
        return lhs;
    }
    if rhs.is_nan() {
        return rhs;
    }
    if lhs.is_zero() && rhs.is_zero() {
        let positive = lhs.is_sign_positive() || rhs.is_sign_positive();
        let negative = !lhs.is_sign_positive() || !rhs.is_sign_positive();
        return match (max, positive, negative) {
            (true, true, _) | (false, true, false) => T::from_f64(0.0),
            _ => T::from_f64(-0.0),
        };
    }
    if max == rhs.less(lhs) {
        lhs
    } else {
        rhs
    }
}

/// `SumSquaresFold::seed`: the square rounded once (`__fmaf_rn(x, x, 0)`).
fn device_square<T: CombineValue>(value: T) -> T {
    device_arith(value, value, value.raw_mul(value).raw_add(T::from_f64(0.0)))
}

/// `Fold::operator()`: the accumulator folds one more element.
fn device_fold<T: CombineValue>(op: ReduceOp, acc: T, value: T) -> T {
    match op {
        ReduceOp::Sum => device_arith(acc, value, acc.raw_add(value)),
        ReduceOp::Prod => device_arith(acc, value, acc.raw_mul(value)),
        ReduceOp::SumSquares => {
            let square = device_square(value);
            device_arith(acc, square, acc.raw_add(square))
        }
        ReduceOp::Max => device_extremum(acc, value, true),
        ReduceOp::Min => device_extremum(acc, value, false),
    }
}

/// `Fold::combine`: two lane accumulators (sum of squares adds, it does not square again).
fn device_combine<T: CombineValue>(op: ReduceOp, lhs: T, rhs: T) -> T {
    match op {
        ReduceOp::SumSquares => device_arith(lhs, rhs, lhs.raw_add(rhs)),
        _ => device_fold(op, lhs, rhs),
    }
}

/// The xor tree over `lanes` with offsets `1, 2, ... < width`: every lane combines its own
/// value (left) with its partner's (right), as `t4a_cuda_plane_fold_narrow` does.
fn device_xor_tree<T: CombineValue>(op: ReduceOp, mut lanes: Vec<T>, width: usize) -> Vec<T> {
    let mut offset = 1;
    while offset < width {
        lanes = (0..lanes.len())
            .map(|lane| device_combine(op, lanes[lane], lanes[lane ^ offset]))
            .collect();
        offset <<= 1;
    }
    lanes
}

/// The combine pass's result for one output: the serial lane walk over `axis`, then the
/// plane fold of `t4a_cuda_plane_fold` (narrow, or wide through the per-warp results).
fn combine_reference<T: CombineValue>(op: ReduceOp, axis: &[T], width: usize) -> T {
    let lanes: Vec<T> = (0..width)
        .map(|lane| {
            let seed = axis[lane];
            let mut acc = if op == ReduceOp::SumSquares {
                device_square(seed)
            } else {
                seed
            };
            let mut index = lane + width;
            while index < axis.len() {
                acc = device_fold(op, acc, axis[index]);
                index += width;
            }
            acc
        })
        .collect();
    if width <= 32 {
        return device_xor_tree(op, lanes, width)[0];
    }
    let warps = width / 32;
    let warp_results: Vec<T> = lanes
        .chunks(32)
        .map(|warp| device_xor_tree(op, warp.to_vec(), 32)[0])
        .collect();
    device_xor_tree(op, warp_results, warps)[0]
}

/// A deterministic hash in `[0, 1)` of `(column, index)`.
fn unit_hash(column: usize, index: usize) -> f64 {
    let mut state = (column as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 31;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 29;
    (state >> 11) as f64 / (1u64 << 53) as f64
}

/// Number of input patterns, one per output of a launch.
const COMBINE_PATTERNS: usize = 9;

/// The pattern whose distinct NaN payloads meet inside one lane's walk.
const COLLIDING_NANS: usize = 1;

/// Element `index` of an axis of `len` under input pattern `column`.
fn combine_input<T: CombineValue>(column: usize, index: usize, len: usize, width: usize) -> T {
    let h = unit_hash(column, index);
    // Near 1 with an occasional tiny addend: the sum's rounding depends on the order, the
    // product stays finite.
    let rounding = 1.0 + (h - 0.5) * 2f64.powi(-7);
    let base = if index % 7 == 3 {
        T::from_f64(rounding * 2f64.powi(-30))
    } else {
        T::from_f64(rounding)
    };
    match column {
        0 => base,
        // NaNs with distinct payloads, all in lane 7's walk (its seed and its elements 1 and 3):
        // which payload survives depends on the walk folding in order with the accumulator on
        // the left. Keeping them in one lane leaves the xor tree a single NaN lane, so the
        // result does not depend on the operand order the compiler chose inside the tree.
        1 => match index {
            7 => T::payload_nan(1, false),
            _ if index == width + 7 => T::payload_nan(2, false),
            _ if index == 3 * width + 7 => T::payload_nan(3, true),
            _ => base,
        },
        // Both infinities and a zero: the sum is NaN, the product is NaN (`inf * 0`).
        2 => match index {
            7 => T::from_f64(f64::INFINITY),
            11 => T::from_f64(0.0),
            _ if index == len - 2 => T::from_f64(f64::NEG_INFINITY),
            _ => base,
        },
        // Negative zeros only: the sum stays -0.0, the product's sign is the count's parity.
        3 => T::from_f64(-0.0),
        // Mixed signed zeros.
        4 => T::from_f64(if h < 0.5 { -0.0 } else { 0.0 }),
        // Near-overflow magnitudes with random signs: partial sums overflow or not depending
        // on the order.
        5 => {
            let magnitude = T::max_finite();
            let scale = T::from_f64((0.5 + h * 0.5) / 16.0);
            let value = magnitude.raw_mul(scale);
            if unit_hash(column + 100, index) < 0.5 {
                value.raw_mul(T::from_f64(-1.0))
            } else {
                value
            }
        }
        // Wide random exponents: partial products overflow to inf or underflow to zero
        // depending on the order (and `inf * 0` makes NaN).
        6 => {
            let span = f64::from(T::max_exponent() / 5);
            let exponent = ((h - 0.5) * 2.0 * span).round() as i32;
            let mantissa = 1.0 + unit_hash(column + 200, index);
            T::from_f64(mantissa * 2f64.powi(exponent))
        }
        // A single NaN with a payload, in lane 5's second element: one NaN operand, so its
        // payload (quieted) is the result whatever the operand order of the folds.
        8 => {
            if index == width + 5 {
                T::payload_nan(4, true)
            } else {
                base
            }
        }
        // Subnormals of both signs: the sum and the extrema stay subnormal, the product
        // underflows to a signed zero.
        _ => {
            let value = T::min_subnormal().raw_mul(T::from_f64((1 + index % 97) as f64));
            if h < 0.5 {
                value.raw_mul(T::from_f64(-1.0))
            } else {
                value
            }
        }
    }
}

/// How the `COMBINE_PATTERNS` reduced axes sit in memory.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CombineLayout {
    /// `[len, outputs]`, the reduced axis contiguous (the partials of a full reduction).
    AxisMajor,
    /// `[outputs, len]` reduced over axis 1: the reduced axis has stride `outputs`.
    Strided,
    /// `[len, outputs]` with axis stride -1 from offset `len - 1`: element `k` of output `o`
    /// sits at `(len - 1 - k) + o * len`.
    Reversed,
    /// `[len, outputs]` with axis stride 0: one element per output, read `len` times.
    Broadcast,
}

/// The input layout, the reduced axis and the host buffer holding `axes` under `kind`.
fn combine_operand<T: CombineValue>(
    axes: &[Vec<T>],
    kind: CombineLayout,
) -> (Layout, usize, Vec<T>) {
    let outputs = axes.len();
    let n = axes[0].len();
    let (len, count) = (n as i64, outputs as i64);
    match kind {
        CombineLayout::AxisMajor => (
            Layout::compact(&[len, count]).unwrap(),
            0,
            (0..n * outputs)
                .map(|flat| axes[flat / n][flat % n])
                .collect(),
        ),
        CombineLayout::Strided => (
            Layout::compact(&[count, len]).unwrap(),
            1,
            (0..n * outputs)
                .map(|flat| axes[flat % outputs][flat / outputs])
                .collect(),
        ),
        CombineLayout::Reversed => (
            Layout::new(&[len, count], &[-1, len], len - 1).unwrap(),
            0,
            (0..n * outputs)
                .map(|flat| axes[flat / n][n - 1 - flat % n])
                .collect(),
        ),
        CombineLayout::Broadcast => (
            Layout::new(&[len, count], &[0, 1], 0).unwrap(),
            0,
            axes.iter().map(|axis| axis[0]).collect(),
        ),
    }
}

/// Launch the combine entry (`t4a_cuda_reduce` or `t4a_cuda_reduce_any`) with strategy
/// `PLANE` and `width` lanes over the outputs of `in_layout` reduced along `axis`, and return
/// the outputs.
#[allow(clippy::too_many_arguments)]
fn run_combine<T: CombineValue>(
    session: &Session,
    host: &[T],
    in_layout: &Layout,
    axis: usize,
    width: u32,
    op: ReduceOp,
    general: bool,
) -> Vec<T> {
    let mut out_extents = in_layout.extents().to_vec();
    let len = out_extents[axis];
    out_extents[axis] = 1;
    let out_layout = Layout::compact(&out_extents).unwrap();
    let outputs = out_layout.element_count().unwrap();
    let plan = plan_reduce(
        T::DTYPE,
        op,
        ReduceStrategy::Plane,
        &out_layout,
        outputs,
        in_layout,
        len * outputs,
        axis as u32,
        width,
    )
    .unwrap();
    assert_eq!(plan.block, [width, 1, 1]);
    assert_eq!(plan.grid, [outputs as u32, 1, 1]);
    let src = upload(session, host);
    let dst = sentinel(session, outputs as usize * std::mem::size_of::<T>());
    if general {
        let geometry = [
            upload(session, out_layout.extents()),
            upload(session, out_layout.strides()),
            upload(session, in_layout.extents()),
            upload(session, in_layout.strides()),
        ];
        let out_any = CLayoutAny::new(
            out_layout.as_ref(),
            geometry[0].as_ptr(),
            geometry[1].as_ptr(),
        )
        .unwrap();
        let in_any = CLayoutAny::new(
            in_layout.as_ref(),
            geometry[2].as_ptr(),
            geometry[3].as_ptr(),
        )
        .unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(T::DTYPE.tag()),
            Slot::i32(op.tag()),
            Slot::i32(ReduceStrategy::Plane.tag()),
            Slot::layout_any(&out_any),
            Slot::layout_any(&in_any),
            Slot::i32(axis as i32),
            Slot::i64(outputs),
        ];
        launch(
            session.function("t4a_cuda_reduce_any"),
            &plan,
            session.stream(),
            &mut slots,
        );
    } else {
        let out_c = CLayout::new(&out_layout);
        let in_c = CLayout::new(in_layout);
        let mut slots = reduce_slots(
            &dst,
            &src,
            T::DTYPE,
            op,
            ReduceStrategy::Plane,
            &out_c,
            &in_c,
            axis as i32,
            outputs,
        );
        launch(
            session.function("t4a_cuda_reduce"),
            &plan,
            session.stream(),
            &mut slots,
        );
    }
    download::<T>(session, &dst, outputs as usize)
}

/// Whether the emulation pins `pattern`'s result bit for bit under `op`. The one exception:
/// two distinct f64 NaN payloads meeting in one lane's `SUM`/`PROD`/`SUM_SQUARES` walk. An
/// f64 add or multiply of two NaNs keeps one operand's payload, and which one is the
/// compiler's choice of operand order per instruction: at `ab7395c` it already differs between
/// the loop's remainder and unrolled iterations, so the same lane keeps a different payload
/// at lengths 3001 and 4096. The ABI leaves the payload out of the contract
/// (docs/abi/c-kernel-abi.md, "NaN"), so there only "is NaN" is checked. The extrema select
/// their NaN in source order and stay bit-exact on that pattern too.
fn combine_bits_pinned<T: CombineValue>(op: ReduceOp, pattern: usize) -> bool {
    pattern != COLLIDING_NANS || !T::KEEPS_PAYLOAD || matches!(op, ReduceOp::Max | ReduceOp::Min)
}

fn check_combine_serial_walk<T: CombineValue>(session: &Session) {
    // (axis length, lanes): the planner's combine of 4096 partials (`PROD` on one warp, the
    // others on 512 lanes), ragged partial counts, lanes holding only their seed
    // (`axis_len == plane_width`) or one or two elements, and short direct-plane axes.
    let geometries: [(i64, u32); 10] = [
        (4_096, 32),
        (4_096, 512),
        (3_001, 32),
        (3_001, 512),
        (1_000, 128),
        (513, 512),
        (512, 512),
        (511, 32),
        (40, 32),
        (32, 32),
    ];
    let ops = [
        ReduceOp::Sum,
        ReduceOp::Prod,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::SumSquares,
    ];
    let kinds = [
        CombineLayout::AxisMajor,
        CombineLayout::Strided,
        CombineLayout::Reversed,
        CombineLayout::Broadcast,
    ];
    let mut compared = 0usize;
    for (len, width) in geometries {
        let n = len as usize;
        for kind in kinds {
            let axes: Vec<Vec<T>> = (0..COMBINE_PATTERNS)
                .map(|column| {
                    let axis: Vec<T> = (0..n)
                        .map(|index| combine_input::<T>(column, index, n, width as usize))
                        .collect();
                    if kind == CombineLayout::Broadcast {
                        vec![axis[0]; n]
                    } else {
                        axis
                    }
                })
                .collect();
            let (in_layout, axis, host) = combine_operand(&axes, kind);
            for op in ops {
                let expected: Vec<T> = axes
                    .iter()
                    .map(|values| combine_reference(op, values, width as usize))
                    .collect();
                for general in [false, true] {
                    let got = run_combine(session, &host, &in_layout, axis, width, op, general);
                    for (pattern, (got, want)) in got.iter().zip(&expected).enumerate() {
                        let context = format!(
                            "{:?} {op:?} len {len} lanes {width} {kind:?} general {general} \
                             pattern {pattern}: got {got:?}, serial walk {want:?}",
                            T::DTYPE
                        );
                        if combine_bits_pinned::<T>(op, pattern) || !want.is_nan() {
                            assert_eq!(got.bits(), want.bits(), "{context}");
                        } else {
                            assert!(got.is_nan() && want.is_nan(), "{context}");
                        }
                        compared += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        compared,
        geometries.len() * kinds.len() * ops.len() * 2 * COMBINE_PATTERNS
    );
}

#[test]
fn combine_keeps_the_serial_lane_walk_bit_for_bit() {
    let session = Session::new("reduce");
    check_combine_serial_walk::<f32>(&session);
    check_combine_serial_walk::<f64>(&session);
}

#[test]
fn combine_inputs_exercise_every_special_class() {
    // The bit-for-bit test is only as strong as its inputs: make sure the emulated results
    // actually contain each class the order can change.
    let len = 4_096usize;
    // NaN payload kept, canonical NaN, +-inf, -0.0, subnormal, an order-sensitive near-overflow
    // sum (it differs from the left-to-right sum of the same values).
    let mut seen = [false; 6];
    for (op, width) in [
        (ReduceOp::Sum, 512usize),
        (ReduceOp::Prod, 32),
        (ReduceOp::Max, 512),
        (ReduceOp::Min, 512),
    ] {
        for column in 0..COMBINE_PATTERNS {
            let axis: Vec<f64> = (0..len)
                .map(|index| combine_input::<f64>(column, index, len, width))
                .collect();
            let result = combine_reference(op, &axis, width);
            seen[0] |= result.is_nan() && result.to_bits() != f64::CANONICAL_NAN_BITS;
            seen[1] |= result.to_bits() == f64::CANONICAL_NAN_BITS;
            seen[2] |= result.is_infinite();
            seen[3] |= result.to_bits() == (-0.0f64).to_bits();
            seen[4] |= result.is_subnormal();
            if column == 5 && op == ReduceOp::Sum {
                let sequential = axis.iter().skip(1).fold(axis[0], |acc, &x| acc + x);
                seen[5] |= sequential.to_bits() != result.to_bits();
            }
        }
    }
    assert_eq!(seen, [true; 6], "classes the combine inputs reach");
}
