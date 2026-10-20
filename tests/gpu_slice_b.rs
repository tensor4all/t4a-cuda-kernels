//! Numerical GPU tests for the revision-2 diagonal, windowing and
//! validation-status entries. Run with
//! `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{
    download, from_bytes, launch, launches, read_bytes, upload, written, Buf, Session, Slot,
};
use t4a_cuda_kernels::{
    plan_dynamic_slice, plan_embed_diagonal_copy, plan_extract_diagonal,
    plan_extract_invalid_float_index, plan_extract_invalid_real_cast, plan_fill_zero,
    plan_fill_zero_view, plan_init_validation, plan_pad, plan_slice, plan_tril, plan_triu,
    plan_validate_float_indices, plan_validate_real_cast, CAxisValues, CLayout, DType, Layout,
};

const SENTINEL: u8 = 0xCD;
const NO_INVALID: u32 = u32::MAX;

fn sentinel(session: &Session, bytes: usize) -> Buf {
    written(session, &vec![SENTINEL; bytes])
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    value / divisor + i64::from(value % divisor != 0)
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

/// Independent forward model of `extract_diagonal`: enumerate every input
/// coordinate, keep the ones whose `axis_a` and `axis_b` coordinates agree, and
/// place each value at its output coordinate. This never walks an output cursor,
/// so it cannot share the kernel's cursor mistake.
fn extract_diagonal_reference(
    input_extents: &[i64],
    axis_a: usize,
    axis_b: usize,
    values: &[f64],
) -> (Vec<i64>, Vec<f64>) {
    let mut out_extents = input_extents.to_vec();
    let diagonal = input_extents[axis_a].min(input_extents[axis_b]);
    out_extents.remove(axis_b);
    let diag_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
    out_extents[diag_axis] = diagonal;
    let out_len: i64 = out_extents.iter().product();
    let mut out = vec![f64::NAN; out_len as usize];
    let input_len: i64 = input_extents.iter().product();
    for in_flat in 0..input_len {
        let coords = coordinates(in_flat, input_extents);
        if coords[axis_a] != coords[axis_b] {
            continue;
        }
        let mut out_coords = coords.clone();
        out_coords.remove(axis_b);
        out_coords[diag_axis] = coords[axis_a];
        let mut out_flat = 0i64;
        let mut stride = 1i64;
        for axis in 0..out_coords.len() {
            out_flat += out_coords[axis] * stride;
            stride *= out_extents[axis];
        }
        out[out_flat as usize] = values[in_flat as usize];
    }
    (out_extents, out)
}

#[test]
fn extract_diagonal_matches_an_independent_oracle() {
    let session = Session::new("diagonal");
    let function = session.function("t4a_cuda_extract_diagonal");

    for (input_extents, axis_a, axis_b) in [
        (vec![2i64, 3, 4], 0usize, 2usize),
        (vec![2, 3, 4], 2, 0),
        (vec![3, 2, 2], 1, 2),
        (vec![4, 4], 0, 1),
        (vec![2, 2, 3, 2], 1, 3),
    ] {
        let input_len: i64 = input_extents.iter().product();
        let values: Vec<f64> = (0..input_len)
            .map(|index| match index % 13 {
                0 => f64::NAN,
                1 => -0.0,
                2 => f64::NEG_INFINITY,
                _ => index as f64 * 0.25,
            })
            .collect();
        let (out_extents, expected) =
            extract_diagonal_reference(&input_extents, axis_a, axis_b, &values);

        let src = upload(&session, &values);
        let in_layout = Layout::compact(&input_extents).unwrap();
        let out_layout = Layout::compact(&out_extents).unwrap();
        let out_len = out_layout.element_count().unwrap();
        let dst = sentinel(&session, out_len as usize * 8);

        let plan = plan_extract_diagonal(
            DType::F64,
            &in_layout,
            input_len,
            &out_layout,
            out_len,
            axis_a as u32,
            axis_b as u32,
        )
        .unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F64.tag()),
            Slot::layout(&CLayout::new(&out_layout)),
            Slot::layout(&CLayout::new(&in_layout)),
            Slot::i32(axis_a as i32),
            Slot::i32(axis_b as i32),
            Slot::i64(out_len),
        ];
        let before = launches();
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(launches(), before + 1);

        let actual = download::<f64>(&session, &dst, out_len as usize);
        for out_flat in 0..out_len as usize {
            let got = actual[out_flat];
            let want = expected[out_flat];
            assert!(
                (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits(),
                "axes ({axis_a},{axis_b}) input {input_extents:?} out {out_flat}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn extract_diagonal_reads_a_strided_input() {
    let session = Session::new("diagonal");
    let function = session.function("t4a_cuda_extract_diagonal");
    // Input extents [2, 3] viewed with strides [10, 2] and offset 1 inside a
    // 40-element buffer. Every address holds a distinct value, so ignoring the
    // offset or the strides cannot pass.
    let input_extents = [2i64, 3];
    let input_strides = [10i64, 2];
    let in_layout = Layout::new(&input_extents, &input_strides, 1).unwrap();
    let mut buffer: Vec<f64> = (0..40).map(|index| index as f64 * 0.5 + 0.125).collect();
    buffer[13] = f64::from_bits(0x7ff8_0000_0000_0005);
    let src = upload(&session, &buffer);
    let out_layout = Layout::compact(&[2]).unwrap();
    let dst = sentinel(&session, 2 * 8);
    let plan = plan_extract_diagonal(DType::F64, &in_layout, 40, &out_layout, 2, 0, 1).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::i32(0),
        Slot::i32(1),
        Slot::i64(2),
    ];
    let before = launches();
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);
    let actual = download::<f64>(&session, &dst, 2);
    // Coordinate (0,0) -> 1 + 0, coordinate (1,1) -> 1 + 10 + 2 = 13.
    for (index, expected) in [buffer[1], buffer[13]].iter().enumerate() {
        assert_eq!(
            actual[index].to_bits(),
            expected.to_bits(),
            "position {index}"
        );
    }
}

#[test]
fn embed_writes_only_the_diagonal_of_a_strided_destination() {
    let session = Session::new("diagonal");
    let movement = Session::new("movement");
    // Output extents [2, 3, 3] with gaps between the logical positions, so both
    // the holes and the allocation tail are observable.
    let out_layout = Layout::new(&[2i64, 3, 3], &[1i64, 4, 16], 0).unwrap();
    let alloc = 42usize;
    let dst = sentinel(&session, alloc * 4);
    let in_layout = Layout::compact(&[2, 3]).unwrap();
    let values: [f32; 6] = [1.5, -0.0, f32::NAN, f32::INFINITY, 5.25, -6.5];
    let src = upload(&session, &values);

    // Zero only the logical view; holes and the tail keep the sentinel.
    let fill = movement.function("t4a_cuda_fill_zero_view");
    let fill_plan = plan_fill_zero_view(DType::F32, &out_layout, alloc as i64).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::i64(18),
    ];
    launch(fill, &fill_plan, movement.stream(), &mut slots);

    let embed = session.function("t4a_cuda_embed_diagonal_copy");
    let plan = plan_embed_diagonal_copy(DType::F32, &in_layout, 6, &out_layout, alloc as i64, 1, 2)
        .unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::i32(1),
        Slot::i32(2),
        Slot::i64(6),
    ];
    let before = launches();
    launch(embed, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);

    let mut expected: Vec<u8> = vec![SENTINEL; alloc * 4];
    // Every logical position is zero from the fill.
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..3usize {
                let index = i + 4 * j + 16 * k;
                expected[index * 4..(index + 1) * 4].copy_from_slice(&[0u8; 4]);
            }
        }
    }
    // Diagonal positions carry the input.
    for i in 0..2usize {
        for j in 0..3usize {
            let index = i + 4 * j + 16 * j;
            expected[index * 4..(index + 1) * 4]
                .copy_from_slice(gpu::as_bytes(&[values[i + 2 * j]]));
        }
    }
    assert_eq!(read_bytes(&session, &dst), expected);
}

#[test]
fn embed_diagonal_composes_with_fill_zero() {
    let session = Session::new("diagonal");
    let movement = Session::new("movement");
    let fill = movement.function("t4a_cuda_fill_zero");
    let embed = session.function("t4a_cuda_embed_diagonal_copy");

    // [2, 3] -> [2, 3, 3] by inserting the diagonal at output axis 2.
    let in_layout = Layout::compact(&[2, 3]).unwrap();
    let out_layout = Layout::compact(&[2, 3, 3]).unwrap();
    let values: [f32; 6] = [1.0, -0.0, f32::NAN, f32::INFINITY, 5.0, 6.0];
    let src = upload(&session, &values);
    let dst = sentinel(&session, 18 * 4);

    let fill_plan = plan_fill_zero(DType::F32, 18).unwrap();
    let mut slots = [Slot::ptr(&dst), Slot::i32(DType::F32.tag()), Slot::i64(18)];
    launch(fill, &fill_plan, movement.stream(), &mut slots);

    let before = launches();
    let plan = plan_embed_diagonal_copy(DType::F32, &in_layout, 6, &out_layout, 18, 1, 2).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::i32(1),
        Slot::i32(2),
        Slot::i64(6),
    ];
    launch(embed, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);

    let actual = download::<f32>(&session, &dst, 18);
    // Off-diagonal positions are zero from the fill, never the sentinel.
    let mut expected = [0.0f32; 18];
    for i in 0..2usize {
        for j in 0..3usize {
            // out coordinate (i, j, j) in a compact [2,3,3] layout.
            expected[i + 2 * j + 6 * j] = values[i + 2 * j];
        }
    }
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "position {index}: {got} vs {want}"
        );
    }
}

#[test]
fn tril_and_triu_match_a_cpu_mask() {
    let session = Session::new("diagonal");
    for (rows, cols, batches, k, entry) in [
        (5usize, 7usize, 1usize, 0i64, "tril"),
        (5, 7, 1, -2, "tril"),
        (5, 7, 1, 3, "tril"),
        (5, 7, 1, i64::MIN, "tril"),
        (5, 7, 1, i64::MAX, "tril"),
        (4, 4, 3, 0, "tril"),
        (5, 7, 1, 0, "triu"),
        (5, 7, 1, 2, "triu"),
        (5, 7, 1, -3, "triu"),
        (5, 7, 1, i64::MAX, "triu"),
        (5, 7, 1, i64::MIN, "triu"),
        (3, 5, 2, -1, "triu"),
    ] {
        let count = rows * cols * batches;
        let values: Vec<i64> = (0..count).map(|index| index as i64 + 1).collect();
        let src = upload(&session, &values);
        let dst = sentinel(&session, count * 8);
        let plan = if entry == "tril" {
            plan_tril(DType::I64, rows as i64, cols as i64, count as i64).unwrap()
        } else {
            plan_triu(DType::I64, rows as i64, cols as i64, count as i64).unwrap()
        };
        let function = session.function(if entry == "tril" {
            "t4a_cuda_tril"
        } else {
            "t4a_cuda_triu"
        });
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::I64.tag()),
            Slot::i64(k),
            Slot::i64(rows as i64),
            Slot::i64(cols as i64),
            Slot::i64(count as i64),
        ];
        let before = launches();
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(launches(), before + 1, "{entry} k={k}");

        let actual = download::<i64>(&session, &dst, count);
        for flat in 0..count {
            let row = flat % rows;
            let col = (flat / rows) % cols;
            let offset = col as i64 - row as i64;
            let keep = if entry == "tril" {
                offset <= k
            } else {
                offset >= k
            };
            let want = if keep { values[flat] } else { 0 };
            assert_eq!(
                actual[flat],
                want,
                "{entry} k={k} {rows}x{cols} batch {} flat {flat}",
                flat / (rows * cols)
            );
        }
    }
}

#[test]
fn slice_matches_a_cpu_reference() {
    let session = Session::new("window");
    let function = session.function("t4a_cuda_slice");
    let input_extents = [7i64, 5];
    let input_len = 35i64;
    let values: Vec<f32> = (0..input_len).map(|index| index as f32).collect();
    let src = upload(&session, &values);
    let starts = [1i64, 0i64];
    let steps = [2i64, 3i64];
    let out_extents = [
        ceil_div(input_extents[0] - starts[0], steps[0]),
        ceil_div(input_extents[1] - starts[1], steps[1]),
    ];
    let out_layout = Layout::compact(&out_extents).unwrap();
    let out_len = out_layout.element_count().unwrap();
    let dst = sentinel(&session, out_len as usize * 4);
    let in_layout = Layout::compact(&input_extents).unwrap();
    let plan = plan_slice(
        DType::F32,
        &out_layout,
        out_len,
        &in_layout,
        input_len,
        &starts,
        &steps,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::axis_values(&CAxisValues::new(&starts).unwrap()),
        Slot::axis_values(&CAxisValues::new(&steps).unwrap()),
        Slot::i64(out_len),
    ];
    let before = launches();
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1);

    let actual = download::<f32>(&session, &dst, out_len as usize);
    for (out_flat, &got) in actual.iter().enumerate() {
        let out_coords = coordinates(out_flat as i64, &out_extents);
        let in_index: i64 = out_coords
            .iter()
            .enumerate()
            .map(|(axis, coordinate)| (starts[axis] + coordinate * steps[axis]) * [1, 7][axis])
            .sum();
        assert_eq!(got, values[in_index as usize], "position {out_flat}");
    }
}

#[test]
fn dynamic_slice_clamps_device_starts() {
    let session = Session::new("window");
    let function = session.function("t4a_cuda_dynamic_slice");
    let input_extents = [6i64, 5];
    let input_len = 30i64;
    let values: Vec<i64> = (0..input_len).collect();
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&input_extents).unwrap();

    // Both axes clamp: window 3 of 6 and window 2 of 5.
    let out_extents = [3i64, 2];
    let out_layout = Layout::compact(&out_extents).unwrap();
    let out_len = 6i64;

    for (index_dtype, starts) in [
        (DType::I64, vec![-5i64, 100]),
        (DType::I32, vec![2, 1]),
        (DType::F32, vec![-2, 7]),
        (DType::F64, vec![1, 3]),
    ] {
        let start_buf = match index_dtype {
            DType::I64 => upload(&session, &starts),
            DType::I32 => {
                let narrowed: Vec<i32> = starts.iter().map(|&value| value as i32).collect();
                upload(&session, &narrowed)
            }
            DType::F32 => {
                let narrowed: Vec<f32> = starts.iter().map(|&value| value as f32).collect();
                upload(&session, &narrowed)
            }
            _ => {
                let widened: Vec<f64> = starts.iter().map(|&value| value as f64).collect();
                upload(&session, &widened)
            }
        };
        let dst = sentinel(&session, out_len as usize * 8);
        let plan = plan_dynamic_slice(
            DType::I64,
            index_dtype,
            &out_layout,
            out_len,
            &in_layout,
            input_len,
            2,
        )
        .unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::ptr(&start_buf),
            Slot::i32(DType::I64.tag()),
            Slot::i32(index_dtype.tag()),
            Slot::layout(&CLayout::new(&out_layout)),
            Slot::layout(&CLayout::new(&in_layout)),
            Slot::i64(out_len),
        ];
        let before = launches();
        launch(function, &plan, session.stream(), &mut slots);
        assert_eq!(launches(), before + 1);

        let clamped: Vec<i64> = starts
            .iter()
            .enumerate()
            .map(|(axis, &start)| {
                let max_start = (input_extents[axis] - out_extents[axis]).max(0);
                if start <= 0 {
                    0
                } else {
                    start.min(max_start)
                }
            })
            .collect();
        let actual = download::<i64>(&session, &dst, out_len as usize);
        for (out_flat, &got) in actual.iter().enumerate() {
            let out_coords = coordinates(out_flat as i64, &out_extents);
            let in_index = (clamped[0] + out_coords[0]) + (clamped[1] + out_coords[1]) * 6;
            assert_eq!(got, values[in_index as usize], "{index_dtype:?} {out_flat}");
        }
    }
}

#[test]
fn pad_matches_a_cpu_reference() {
    let session = Session::new("window");
    let function = session.function("t4a_cuda_pad");
    let input_extents = [3i64, 4];
    let input_len = 12i64;
    let values: Vec<f64> = (0..input_len).map(|index| index as f64 + 0.5).collect();
    let src = upload(&session, &values);
    let in_layout = Layout::compact(&input_extents).unwrap();

    for (low, high, interior, out_extents) in [
        (vec![1i64, 0], vec![2i64, 1], vec![0i64, 1], vec![6i64, 8]),
        (vec![-1, 0], vec![0, 0], vec![0, 0], vec![2i64, 4]),
        (vec![0, 2], vec![0, 2], vec![1, 0], vec![5i64, 8]),
    ] {
        let out_layout = Layout::compact(&out_extents).unwrap();
        let out_len = out_layout.element_count().unwrap();
        let dst = sentinel(&session, out_len as usize * 8);
        let plan = plan_pad(
            DType::F64,
            &out_layout,
            out_len,
            &in_layout,
            input_len,
            &low,
            &high,
            &interior,
        )
        .unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F64.tag()),
            Slot::layout(&CLayout::new(&out_layout)),
            Slot::layout(&CLayout::new(&in_layout)),
            Slot::axis_values(&CAxisValues::new(&low).unwrap()),
            Slot::axis_values(&CAxisValues::new(&high).unwrap()),
            Slot::axis_values(&CAxisValues::new(&interior).unwrap()),
            Slot::i64(out_len),
        ];
        launch(function, &plan, session.stream(), &mut slots);

        let actual = download::<f64>(&session, &dst, out_len as usize);
        for (out_flat, &got) in actual.iter().enumerate() {
            let out_coords = coordinates(out_flat as i64, &out_extents);
            let mut in_coords = Vec::new();
            let mut in_bounds = true;
            for axis in 0..2 {
                let shifted = out_coords[axis] - low[axis];
                let spacing = interior[axis] + 1;
                if shifted < 0 || shifted % spacing != 0 {
                    in_bounds = false;
                    break;
                }
                let candidate = shifted / spacing;
                if candidate >= input_extents[axis] {
                    in_bounds = false;
                    break;
                }
                in_coords.push(candidate);
            }
            let want = if in_bounds {
                values[(in_coords[0] + 3 * in_coords[1]) as usize]
            } else {
                0.0
            };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "low {low:?} interior {interior:?} out {out_flat}"
            );
        }
    }
}

#[test]
fn validation_reports_the_earliest_invalid_position() {
    let session = Session::new("validation");
    let init = session.function("t4a_cuda_init_validation");
    let validate = session.function("t4a_cuda_validate_float_indices");
    let extract = session.function("t4a_cuda_extract_invalid_float_index");

    let count = 1usize << 16;
    let plan = plan_init_validation(DType::F64).unwrap();
    let flag = sentinel(&session, 4);
    let value = sentinel(&session, 8);
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(init, &plan, session.stream(), &mut slots);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag)),
        vec![NO_INVALID]
    );
    assert_eq!(from_bytes::<f64>(&read_bytes(&session, &value)), vec![0.0]);

    // Invalid values at several positions; the earliest wins and its source
    // value is extracted.
    let mut values: Vec<f64> = vec![2.0; count];
    values[40_000] = 7.0; // integral and in range: valid
    values[30_000] = 2.5; // non-integral
    values[512] = f64::NAN;
    values[100] = f64::INFINITY;
    let src = upload(&session, &values);
    let plan = plan_validate_float_indices(DType::F64, count as i64).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(9_007_199_254_740_992.0),
        Slot::i64(count as i64),
    ];
    launch(validate, &plan, session.stream(), &mut slots);
    let plan = plan_extract_invalid_float_index(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(extract, &plan, session.stream(), &mut slots);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag)),
        vec![100u32]
    );
    assert_eq!(
        from_bytes::<f64>(&read_bytes(&session, &value)),
        vec![f64::INFINITY]
    );

    // A fully valid domain leaves the flag untouched and the value unwritten.
    let plan = plan_init_validation(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(init, &plan, session.stream(), &mut slots);
    let valid = vec![1.0f64; count];
    let src = upload(&session, &valid);
    let plan = plan_validate_float_indices(DType::F64, count as i64).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(9_007_199_254_740_992.0),
        Slot::i64(count as i64),
    ];
    launch(validate, &plan, session.stream(), &mut slots);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag)),
        vec![NO_INVALID]
    );
    let plan = plan_extract_invalid_float_index(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(extract, &plan, session.stream(), &mut slots);
    assert_eq!(from_bytes::<f64>(&read_bytes(&session, &value)), vec![0.0]);

    // The exact-integer cutoff: 2^24 is valid for F32, 2^24 + 1 is not.
    let cutoff: [f32; 3] = [16_777_216.0, 16_777_218.0, 1.0];
    let src = upload(&session, &cutoff);
    let plan = plan_init_validation(DType::F32).unwrap();
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F32.tag()),
    ];
    launch(init, &plan, session.stream(), &mut slots);
    let plan = plan_validate_float_indices(DType::F32, 3).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::i32(DType::F32.tag()),
        Slot::f64(16_777_216.0),
        Slot::i64(3),
    ];
    launch(validate, &plan, session.stream(), &mut slots);
    assert_eq!(from_bytes::<u32>(&read_bytes(&session, &flag)), vec![1u32]);
}

#[test]
fn real_cast_validation_samples_by_stride() {
    let session = Session::new("validation");
    let init = session.function("t4a_cuda_init_validation");
    let validate = session.function("t4a_cuda_validate_real_cast");
    let extract = session.function("t4a_cuda_extract_invalid_real_cast");

    let flag = sentinel(&session, 4);
    let value = sentinel(&session, 8);
    let plan = plan_init_validation(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(init, &plan, session.stream(), &mut slots);

    // stride 2: only even positions are sampled.
    let count = 4096usize;
    let mut values = vec![1.0f64; count];
    values[3] = f64::NAN; // skipped slot: must not invalidate
    values[100] = 5.0; // exactly max: invalid when exclusive (sampled position 50)
    values[2000] = 6.0; // above max (sampled position 1000)
    let src = upload(&session, &values);
    let plan = plan_validate_real_cast(DType::F64, count as i64, 2).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(-1.0),
        Slot::f64(5.0),
        Slot::i64(2),
        Slot::i32(0),
        Slot::i64(count as i64),
    ];
    launch(validate, &plan, session.stream(), &mut slots);
    let plan = plan_extract_invalid_real_cast(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
        Slot::i64(2),
    ];
    launch(extract, &plan, session.stream(), &mut slots);
    // The flag is the sampling position (element index / stride), matching the
    // baseline's thread-index flag; the extracted value is the source element.
    assert_eq!(from_bytes::<u32>(&read_bytes(&session, &flag)), vec![50u32]);
    assert_eq!(from_bytes::<f64>(&read_bytes(&session, &value)), vec![5.0]);

    // Inclusive max: 5.0 is accepted, 6.0 is still outside.
    let plan = plan_init_validation(DType::F64).unwrap();
    let mut slots = [
        Slot::ptr(&flag),
        Slot::ptr(&value),
        Slot::i32(DType::F64.tag()),
    ];
    launch(init, &plan, session.stream(), &mut slots);
    let plan = plan_validate_real_cast(DType::F64, count as i64, 2).unwrap();
    let mut slots = [
        Slot::ptr(&src),
        Slot::ptr(&flag),
        Slot::i32(DType::F64.tag()),
        Slot::f64(-1.0),
        Slot::f64(5.0),
        Slot::i64(2),
        Slot::i32(1),
        Slot::i64(count as i64),
    ];
    launch(validate, &plan, session.stream(), &mut slots);
    assert_eq!(
        from_bytes::<u32>(&read_bytes(&session, &flag)),
        vec![1000u32]
    );
}

#[test]
fn large_domains_cover_every_revision_two_entry() {
    let session = Session::new("window");
    let diagonal = Session::new("diagonal");
    let movement = Session::new("movement");
    let count = 1usize << 16;

    // tril over 2^16 elements: batched 256x256 mask.
    let values: Vec<f32> = (0..count).map(|index| index as f32).collect();
    let src = upload(&session, &values);
    let dst = sentinel(&session, count * 4);
    let plan = plan_tril(DType::F32, 256, 256, count as i64).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::i64(0),
        Slot::i64(256),
        Slot::i64(256),
        Slot::i64(count as i64),
    ];
    launch(
        diagonal.function("t4a_cuda_tril"),
        &plan,
        diagonal.stream(),
        &mut slots,
    );
    let actual = download::<f32>(&session, &dst, count);
    for (flat, got) in actual.iter().enumerate() {
        let row = flat % 256;
        let col = (flat / 256) % 256;
        let want = if col <= row { values[flat] } else { 0.0 };
        assert_eq!(*got, want, "position {flat}");
    }

    // triu over the same 256x256 domain.
    let dst = sentinel(&session, count * 4);
    let plan = plan_triu(DType::F32, 256, 256, count as i64).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::i64(0),
        Slot::i64(256),
        Slot::i64(256),
        Slot::i64(count as i64),
    ];
    let before = launches();
    launch(
        diagonal.function("t4a_cuda_triu"),
        &plan,
        diagonal.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 1);
    let actual = download::<f32>(&session, &dst, count);
    for (flat, got) in actual.iter().enumerate() {
        let row = flat % 256;
        let col = (flat / 256) % 256;
        let want = if col >= row { values[flat] } else { 0.0 };
        assert_eq!(*got, want, "triu position {flat}");
    }

    // extract diagonal of a 256x256 matrix: 2^8 outputs, but the kernel still
    // maps a whole tensor domain per launch.
    let square = Layout::compact(&[256, 256]).unwrap();
    let diagonal_out = Layout::compact(&[256]).unwrap();
    let extracted = sentinel(&session, 256 * 4);
    let plan =
        plan_extract_diagonal(DType::F32, &square, count as i64, &diagonal_out, 256, 0, 1).unwrap();
    let mut slots = [
        Slot::ptr(&extracted),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&diagonal_out)),
        Slot::layout(&CLayout::new(&square)),
        Slot::i32(0),
        Slot::i32(1),
        Slot::i64(256),
    ];
    launch(
        diagonal.function("t4a_cuda_extract_diagonal"),
        &plan,
        diagonal.stream(),
        &mut slots,
    );
    let extracted_values = download::<f32>(&session, &extracted, 256);
    for index in 0..256usize {
        assert_eq!(extracted_values[index], values[index + 256 * index]);
    }

    // embed a 256-element vector into a 256x256 zero-filled matrix.
    let vector = upload(&session, &values[..256]);
    let embedded = sentinel(&session, count * 4);
    let fill = movement.function("t4a_cuda_fill_zero");
    let fill_plan = plan_fill_zero(DType::F32, count as i64).unwrap();
    let mut slots = [
        Slot::ptr(&embedded),
        Slot::i32(DType::F32.tag()),
        Slot::i64(count as i64),
    ];
    launch(fill, &fill_plan, movement.stream(), &mut slots);
    let vector_layout = Layout::compact(&[256]).unwrap();
    let plan =
        plan_embed_diagonal_copy(DType::F32, &vector_layout, 256, &square, count as i64, 0, 0)
            .unwrap();
    let mut slots = [
        Slot::ptr(&embedded),
        Slot::ptr(&vector),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&square)),
        Slot::layout(&CLayout::new(&vector_layout)),
        Slot::i32(0),
        Slot::i32(0),
        Slot::i64(256),
    ];
    launch(
        diagonal.function("t4a_cuda_embed_diagonal_copy"),
        &plan,
        diagonal.stream(),
        &mut slots,
    );
    let embedded_values = download::<f32>(&session, &embedded, count);
    for (flat, &got) in embedded_values.iter().enumerate() {
        let row = flat % 256;
        let col = flat / 256;
        let want = if row == col { values[row] } else { 0.0 };
        assert_eq!(got, want, "embed position {flat}");
    }

    // dynamic_slice over a 512x256 operand producing 2^16 outputs.
    let wide_extents = [512i64, 256];
    let wide_len = 512 * 256;
    let wide_values: Vec<f32> = (0..wide_len).map(|index| index as f32).collect();
    let wide = upload(&session, &wide_values);
    let wide_layout = Layout::compact(&wide_extents).unwrap();
    let starts = [256i64, 0];
    let start_buf = upload(&session, &starts);
    let out_extents = [256i64, 256];
    let out_layout = Layout::compact(&out_extents).unwrap();
    let sliced = sentinel(&session, count * 4);
    let plan = plan_dynamic_slice(
        DType::F32,
        DType::I64,
        &out_layout,
        count as i64,
        &wide_layout,
        wide_len as i64,
        2,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&sliced),
        Slot::ptr(&wide),
        Slot::ptr(&start_buf),
        Slot::i32(DType::F32.tag()),
        Slot::i32(DType::I64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&wide_layout)),
        Slot::i64(count as i64),
    ];
    launch(
        session.function("t4a_cuda_dynamic_slice"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let sliced_values = download::<f32>(&session, &sliced, count);
    for (out_flat, &got) in sliced_values.iter().enumerate() {
        let out_coords = coordinates(out_flat as i64, &out_extents);
        let in_index = (256 + out_coords[0]) + 512 * out_coords[1];
        assert_eq!(
            got, wide_values[in_index as usize],
            "dynamic position {out_flat}"
        );
    }

    // pad over 2^16 elements with interior padding on one axis.
    let in_extents = [256i64, 256];
    // interior 1 doubles the spacing: (256 - 1) * 2 + 1 = 511.
    let out_extents = [511i64, 256];
    let out_len = out_extents[0] * out_extents[1];
    let in_layout = Layout::compact(&in_extents).unwrap();
    let out_layout = Layout::compact(&out_extents).unwrap();
    let padded = sentinel(&session, out_len as usize * 4);
    let plan = plan_pad(
        DType::F32,
        &out_layout,
        out_len,
        &in_layout,
        count as i64,
        &[0, 0],
        &[0, 0],
        &[1, 0],
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&padded),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::axis_values(&CAxisValues::new(&[0, 0]).unwrap()),
        Slot::axis_values(&CAxisValues::new(&[0, 0]).unwrap()),
        Slot::axis_values(&CAxisValues::new(&[1, 0]).unwrap()),
        Slot::i64(out_len),
    ];
    launch(
        session.function("t4a_cuda_pad"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let padded_values = download::<f32>(&session, &padded, out_len as usize);
    for (out_flat, &got) in padded_values.iter().enumerate() {
        let out_coords = coordinates(out_flat as i64, &out_extents);
        let shifted = out_coords[0];
        let want = if shifted % 2 == 0 && shifted / 2 < 256 {
            values[(shifted / 2 + 256 * out_coords[1]) as usize]
        } else {
            0.0
        };
        assert_eq!(got, want, "position {out_flat}");
    }

    // slice over 2^16 output elements, offset by one row.
    let out_extents = [255i64, 256];
    let out_layout = Layout::compact(&out_extents).unwrap();
    let out_len = out_layout.element_count().unwrap();
    let sliced = sentinel(&session, out_len as usize * 4);
    let starts = [1i64, 0];
    let steps = [1i64, 1];
    let plan = plan_slice(
        DType::F32,
        &out_layout,
        out_len,
        &in_layout,
        count as i64,
        &starts,
        &steps,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&sliced),
        Slot::ptr(&src),
        Slot::i32(DType::F32.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&in_layout)),
        Slot::axis_values(&CAxisValues::new(&starts).unwrap()),
        Slot::axis_values(&CAxisValues::new(&steps).unwrap()),
        Slot::i64(out_len),
    ];
    launch(
        session.function("t4a_cuda_slice"),
        &plan,
        session.stream(),
        &mut slots,
    );
    let sliced_values = download::<f32>(&session, &sliced, out_len as usize);
    for (out_flat, &got) in sliced_values.iter().enumerate() {
        let out_coords = coordinates(out_flat as i64, &out_extents);
        let in_index = (1 + out_coords[0]) + 256 * out_coords[1];
        assert_eq!(got, values[in_index as usize], "position {out_flat}");
    }
}
