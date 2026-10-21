//! Numerical GPU tests for the revision-4 gather and additive-scatter entries.
//! Run with `cargo test -j 16 --features gpu-tests`; requires a CUDA device.

mod gpu;

use gpu::{
    download, from_bytes, launch, launch_async, launches, pub_sync, read_bytes, upload, written,
    Buf, Session, Slot,
};
use t4a_cuda_kernels::{
    plan_copy_exact, plan_gather, plan_scatter_add, window_dims, CGatherPlan, CLayout,
    CScatterPlan, DType, GatherPlan, Layout, ScatterPlan,
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

/// Every coordinate tuple of `extents`, with axis 0 varying fastest (the
/// repository's logical order). Built as a product, not by decoding a flat index.
fn cartesian(extents: &[i64]) -> Vec<Vec<i64>> {
    let mut tuples: Vec<Vec<i64>> = vec![Vec::new()];
    for &extent in extents.iter().rev() {
        let mut next = Vec::with_capacity(tuples.len() * extent as usize);
        for tuple in &tuples {
            for coordinate in 0..extent {
                let mut extended = Vec::with_capacity(tuple.len() + 1);
                extended.push(coordinate);
                extended.extend_from_slice(tuple);
                next.push(extended);
            }
        }
        tuples = next;
    }
    tuples
}

/// Position of a coordinate tuple in a compact buffer with axis 0 fastest.
fn flat_of(coordinates: &[i64], extents: &[i64]) -> i64 {
    let mut offset = 0i64;
    let mut stride = 1i64;
    for axis in 0..extents.len() {
        offset += coordinates[axis] * stride;
        stride *= extents[axis];
    }
    offset
}

fn clamp_start(start: i64, extent: i64, window: i64) -> i64 {
    let max_start = (extent - window).max(0);
    if start <= 0 {
        0
    } else {
        start.min(max_start)
    }
}

/// Independent gather reference: walks the output's cartesian product and
/// assembles the operand coordinate from the batch, window and start pieces.
/// It never decodes a flat output index the way the kernel does.
#[allow(clippy::too_many_arguments)]
fn gather_reference(
    operand_extents: &[i64],
    operand_values: &[f64],
    indices_extents: &[i64],
    indices_values: &[i64],
    plan: &GatherPlan,
    out_extents: &[i64],
) -> Vec<f64> {
    let batch_axes: Vec<usize> = (0..plan.index_vector_dim as usize)
        .chain((plan.index_vector_dim as usize + 1)..indices_extents.len())
        .collect();
    let batch_extents: Vec<i64> = batch_axes
        .iter()
        .map(|&axis| indices_extents[axis])
        .collect();
    let batch_tuples = cartesian(&batch_extents);

    let mut out = Vec::new();
    for out_coords in cartesian(out_extents) {
        // Batch coordinates are the output coordinates on non-offset axes.
        let mut batch_coords = vec![0i64; batch_extents.len()];
        let mut batch_axis = 0;
        let mut window_coords = vec![0i64; plan.window_dims.len()];
        for (out_axis, &coordinate) in out_coords.iter().enumerate() {
            if let Some(position) = plan
                .offset_dims
                .iter()
                .position(|&axis| axis as usize == out_axis)
            {
                window_coords[position] = coordinate;
            } else {
                batch_coords[batch_axis] = coordinate;
                batch_axis += 1;
            }
        }
        // Map the batch coordinates back to the index tensor's own axes.
        let mut operand_coords = vec![0i64; operand_extents.len()];
        for (component, &operand_axis) in plan.start_index_map.iter().enumerate() {
            let mut index_coords = vec![0i64; indices_extents.len()];
            let mut b = 0;
            for (axis, coordinate) in index_coords.iter_mut().enumerate() {
                if axis == plan.index_vector_dim as usize {
                    *coordinate = component as i64;
                } else {
                    *coordinate = batch_coords[b];
                    b += 1;
                }
            }
            let index_flat = flat_of(&index_coords, indices_extents);
            let start = indices_values[index_flat as usize];
            let axis = operand_axis as usize;
            operand_coords[axis] +=
                clamp_start(start, operand_extents[axis], plan.slice_sizes[axis]);
        }
        for (position, &operand_axis) in plan.window_dims.iter().enumerate() {
            operand_coords[operand_axis as usize] += window_coords[position];
        }
        let flat = flat_of(&operand_coords, operand_extents);
        out.push(operand_values[flat as usize]);
    }
    let _ = batch_tuples;
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gather(
    session: &Session,
    dtype: DType,
    index_dtype: DType,
    operand: &Buf,
    indices: &Buf,
    out_layout: &Layout,
    operand_layout: &Layout,
    operand_len: i64,
    indices_layout: &Layout,
    indices_len: i64,
    plan: &GatherPlan,
    out_len: i64,
    guard: usize,
) -> Buf {
    let (launch_plan, output_len) = plan_gather(
        dtype,
        index_dtype,
        out_layout,
        out_len,
        operand_layout,
        operand_len,
        indices_layout,
        indices_len,
        plan,
    )
    .unwrap();
    let mirror = CGatherPlan::new(plan, out_layout.rank()).unwrap();
    let out = sentinel(session, guard * dtype.size());
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(operand),
        Slot::ptr(indices),
        Slot::i32(dtype.tag()),
        Slot::i32(index_dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(operand_layout)),
        Slot::layout(&CLayout::new(indices_layout)),
        Slot::gather_plan(&mirror),
        Slot::i64(output_len),
    ];
    let function = session.function("t4a_cuda_gather");
    let before = launches();
    launch(function, &launch_plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1, "gather is one launch");
    out
}

#[test]
fn gather_matches_an_independent_reference() {
    let session = Session::new("gather_scatter");
    // Operand [4, 3]: axis 0 collapsed, window on axis 1. Index tensor [2, 1].
    let operand_extents = [4i64, 3];
    let operand_values: Vec<f64> = (0..12).map(|index| index as f64 * 0.5 - 2.0).collect();
    let operand_layout = Layout::compact(&operand_extents).unwrap();
    let operand = upload(&session, &operand_values);
    let indices_extents = [2i64, 1];
    let indices_values: Vec<i64> = vec![2, 0];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: window_dims(2, &[0]),
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, 3],
        index_vector_dim: 1,
    };
    let out_layout = Layout::compact(&[2, 3]).unwrap();
    let out = run_gather(
        &session,
        DType::F64,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        12,
        &indices_layout,
        2,
        &plan,
        6,
        6,
    );
    let expected = gather_reference(
        &operand_extents,
        &operand_values,
        &indices_extents,
        &indices_values,
        &plan,
        &[2, 3],
    );
    assert_eq!(download::<f64>(&session, &out, 6), expected);
}

#[test]
fn gather_handles_a_middle_index_vector_and_unsorted_offsets() {
    let session = Session::new("gather_scatter");
    // Operand [3, 4, 2]; gather over axes 0 and 2 with a 1-wide window on axis 1.
    let operand_extents = [3i64, 4, 2];
    let operand_values: Vec<f64> = (0..24).map(|index| (index % 11) as f64 - 5.0).collect();
    let operand_layout = Layout::compact(&operand_extents).unwrap();
    let operand = upload(&session, &operand_values);
    // Index tensor [2, 2, 3]: index_vector_dim = 1, so two components and two
    // batches.
    let indices_extents = [2i64, 2, 3];
    let indices_values: Vec<i64> = vec![
        2, 0, // batch (0,0): starts (2, 0)
        0, 1, // batch (0,1)
        1, 3, // batch (0,2)
        1, 2, // batch (1,0)
        2, 2, // batch (1,1)
        0, 0, // batch (1,2)
    ];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let indices = upload(&session, &indices_values);
    // Window on operand axes 0 and 2 (one element each: collapsed), so window_dims
    // = [1] but slice_sizes are 1 on the gathered axes... The window is axis 1
    // with extent 1, giving output [2 (batch, axis 1), 3 (axis 2)] ordered as
    // offset dims at out axis 0.
    let plan = GatherPlan {
        window_dims: vec![1],
        offset_dims: vec![0],
        start_index_map: vec![0, 2],
        slice_sizes: vec![1, 2, 1],
        index_vector_dim: 1,
    };
    // Batch extents: indices [2, 2, 3] without axis 1 -> [2, 3].
    let out_layout = Layout::compact(&[2, 2, 3]).unwrap();
    let out = run_gather(
        &session,
        DType::F64,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        24,
        &indices_layout,
        12,
        &plan,
        12,
        12,
    );
    let expected = gather_reference(
        &operand_extents,
        &operand_values,
        &indices_extents,
        &indices_values,
        &plan,
        &[2, 2, 3],
    );
    assert_eq!(download::<f64>(&session, &out, 12), expected);
}

#[test]
fn gather_clamps_starts_and_reads_a_strided_operand() {
    let session = Session::new("gather_scatter");
    // Operand [2, 6] viewed with strides [16, 1] and offset 3.
    let operand_extents = [2i64, 6];
    let operand_strides = [16i64, 1];
    let operand_offset = 3i64;
    let buffer: Vec<f64> = (0..40).map(|index| (index % 17) as f64 * 0.25).collect();
    let operand_layout = Layout::new(&operand_extents, &operand_strides, operand_offset).unwrap();
    let operand = upload(&session, &buffer);
    // Starts far below zero and far above the axis end.
    let indices_values: Vec<i64> = vec![-9, 100, 2];
    let indices_extents = [3i64, 1];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![1],
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, 3],
        index_vector_dim: 1,
    };
    // Batch extents [3]; output [3, 3].
    let out_layout = Layout::compact(&[3, 3]).unwrap();
    let out = run_gather(
        &session,
        DType::F64,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        40,
        &indices_layout,
        3,
        &plan,
        9,
        9,
    );
    // Independent expectation: starts clamp to 0, 1 and 1 on axis 0 (extent 2),
    // and the window covers axis 1 from 0.
    // The output is [batch, window] with axis 0 fastest, so the window varies
    // fastest in the flat buffer.
    let clamped: Vec<i64> = [0i64, 3, 2]
        .iter()
        .map(|&start| clamp_start(start, operand_extents[0], 1))
        .collect();
    let mut expected = Vec::new();
    for window in 0..3 {
        for &start in &clamped {
            let flat = operand_offset + start * operand_strides[0] + window * operand_strides[1];
            expected.push(buffer[flat as usize]);
        }
    }
    assert_eq!(download::<f64>(&session, &out, 9), expected);
}

#[test]
fn gather_supports_a_scalar_output_and_float_indices() {
    let session = Session::new("gather_scatter");
    let operand_values: Vec<f64> = (0..5).map(|index| index as f64 + 0.5).collect();
    let operand_layout = Layout::compact(&[5]).unwrap();
    let operand = upload(&session, &operand_values);
    let indices_values: Vec<f64> = vec![3.0];
    let indices_layout = Layout::compact(&[1]).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![],
        offset_dims: vec![],
        start_index_map: vec![0],
        slice_sizes: vec![1],
        index_vector_dim: 0,
    };
    let out_layout = Layout::compact(&[]).unwrap();
    let out = run_gather(
        &session,
        DType::F64,
        DType::F64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        5,
        &indices_layout,
        1,
        &plan,
        1,
        1,
    );
    assert_eq!(download::<f64>(&session, &out, 1), vec![3.5]);
}

#[test]
fn gather_covers_a_large_output_domain() {
    let session = Session::new("gather_scatter");
    let outputs = 1i64 << 16;
    let rows = 512i64;
    let cols = 8i64;
    let operand_values: Vec<f32> = (0..rows * cols).map(|index| index as f32).collect();
    let operand_layout = Layout::compact(&[rows, cols]).unwrap();
    let operand = upload(&session, &operand_values);
    // Scalar starts, one per output row.
    let indices_values: Vec<i64> = (0..outputs).map(|index| (index * 7) % rows).collect();
    let indices_layout = Layout::compact(&[outputs]).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![1],
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, cols],
        index_vector_dim: 1,
    };
    // One contiguous window of `cols` values does not fit a flat output; use an
    // output of [outputs, cols] to keep the comparison trivial.
    let out_layout = Layout::compact(&[outputs, cols]).unwrap();
    let out = run_gather(
        &session,
        DType::F32,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        rows * cols,
        &indices_layout,
        outputs,
        &plan,
        outputs * cols,
        (outputs * cols) as usize,
    );
    let actual = download::<f32>(&session, &out, (outputs * cols) as usize);
    for (index, got) in actual.iter().enumerate() {
        // Compact [outputs, cols]: axis 0 is the fastest-varying axis.
        let row = index as i64 % outputs;
        let col = index as i64 / outputs;
        let operand_row = indices_values[row as usize];
        let want = operand_values[(operand_row + rows * col) as usize];
        assert_eq!(*got, want, "position {index}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_scatter(
    session: &Session,
    dtype: DType,
    index_dtype: DType,
    out: &Buf,
    indices: &Buf,
    updates: &Buf,
    out_layout: &Layout,
    indices_layout: &Layout,
    updates_layout: &Layout,
    out_len: i64,
    indices_len: i64,
    updates_len: i64,
    plan: &ScatterPlan,
) {
    let launch_plan = plan_scatter_add(
        dtype,
        index_dtype,
        out_layout,
        out_len,
        indices_layout,
        indices_len,
        updates_layout,
        updates_len,
        plan,
    )
    .unwrap();
    let mirror = CScatterPlan::new(out_layout.rank(), updates_layout.rank(), plan).unwrap();
    let mut slots = [
        Slot::ptr(out),
        Slot::ptr(indices),
        Slot::ptr(updates),
        Slot::i32(dtype.tag()),
        Slot::i32(index_dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(indices_layout)),
        Slot::layout(&CLayout::new(updates_layout)),
        Slot::scatter_plan(&mirror),
        Slot::i64(updates_layout.element_count().unwrap()),
    ];
    let function = session.function("t4a_cuda_scatter_add");
    let before = launches();
    launch(function, &launch_plan, session.stream(), &mut slots);
    assert_eq!(launches(), before + 1, "scatter add is one launch");
}

/// Independent additive-scatter reference over the updates cartesian product.
fn scatter_reference(
    operand_extents: &[i64],
    operand_values: &[f64],
    indices_extents: &[i64],
    indices_values: &[i64],
    updates_extents: &[i64],
    updates_values: &[f64],
    plan: &ScatterPlan,
) -> Vec<f64> {
    let batch_axes: Vec<usize> = (0..updates_extents.len())
        .filter(|axis| !plan.update_window_dims.contains(&(*axis as u32)))
        .collect();
    let window_shape: Vec<i64> = plan
        .window_dims
        .iter()
        .zip(plan.update_window_dims.iter())
        .map(|(&operand_axis, &updates_axis)| {
            let _ = operand_axis;
            updates_extents[updates_axis as usize]
        })
        .collect();
    let _ = window_shape;
    let mut result = operand_values.to_vec();
    let index_batch_extents: Vec<i64> = (0..indices_extents.len())
        .filter(|axis| *axis != plan.index_vector_dim as usize)
        .map(|axis| indices_extents[axis])
        .collect();
    let _ = batch_axes;
    for update_coords in cartesian(updates_extents) {
        // Batch coordinates come from the non-window updates axes.
        let mut batch_coords = Vec::new();
        let mut window_coords = vec![0i64; plan.window_dims.len()];
        for (axis, &coordinate) in update_coords.iter().enumerate() {
            if let Some(position) = plan
                .update_window_dims
                .iter()
                .position(|&update_axis| update_axis as usize == axis)
            {
                window_coords[position] = coordinate;
            } else {
                batch_coords.push(coordinate);
            }
        }
        let mut operand_coords = vec![0i64; operand_extents.len()];
        let mut fits = true;
        for (component, &operand_axis) in plan.scatter_dims.iter().enumerate() {
            let mut index_coords = vec![0i64; indices_extents.len()];
            let mut b = 0;
            for (axis, coordinate) in index_coords.iter_mut().enumerate() {
                if axis == plan.index_vector_dim as usize {
                    *coordinate = component as i64;
                } else {
                    *coordinate = batch_coords[b];
                    b += 1;
                }
            }
            let index_flat = flat_of(&index_coords, indices_extents);
            let start = indices_values[index_flat as usize];
            let axis = operand_axis as usize;
            let window = plan
                .window_dims
                .iter()
                .position(|&position| position as usize == axis)
                .map(|position| updates_extents[plan.update_window_dims[position] as usize])
                .unwrap_or(1);
            operand_coords[axis] = clamp_start(start, operand_extents[axis], window);
        }
        for (position, &operand_axis) in plan.window_dims.iter().enumerate() {
            operand_coords[operand_axis as usize] += window_coords[position];
        }
        for (axis, &coordinate) in operand_coords.iter().enumerate() {
            let window = plan
                .window_dims
                .iter()
                .position(|&position| position as usize == axis)
                .map(|position| updates_extents[plan.update_window_dims[position] as usize])
                .unwrap_or(1);
            if coordinate >= operand_extents[axis]
                || coordinate < 0
                || window > operand_extents[axis]
            {
                fits = false;
            }
        }
        if !fits {
            continue;
        }
        let flat = flat_of(&update_coords, updates_extents);
        let operand_flat = flat_of(&operand_coords, operand_extents);
        result[operand_flat as usize] += updates_values[flat as usize];
        let _ = index_batch_extents;
    }
    result
}

#[test]
fn scatter_add_matches_an_independent_reference() {
    let session = Session::new("gather_scatter");
    // Destination [6]; four batches of a 2-wide window, with collisions.
    let operand_extents = [6i64];
    let operand_values = vec![1.0f64; 6];
    let out_layout = Layout::compact(&operand_extents).unwrap();
    let indices_values: Vec<i64> = vec![0, 1, 5, 4];
    let indices_extents = [4i64];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_values: Vec<f64> = (0..8).map(|index| index as f64 + 0.25).collect();
    let updates_extents = [4i64, 2];
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![0],
        update_window_dims: vec![1],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let out = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    run_scatter(
        &session,
        DType::F64,
        DType::I64,
        &out,
        &indices,
        &updates,
        &out_layout,
        &indices_layout,
        &updates_layout,
        6,
        4,
        8,
        &plan,
    );
    let expected = scatter_reference(
        &operand_extents,
        &operand_values,
        &indices_extents,
        &indices_values,
        &updates_extents,
        &updates_values,
        &plan,
    );
    let actual = download::<f64>(&session, &out, 6);
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12,
            "position {index}: {got} vs {want}"
        );
    }
}

#[test]
fn scatter_add_supports_complex_updates_and_index_vectors() {
    let session = Session::new("gather_scatter");
    // Destination [3, 4] with an index vector of two components.
    let operand_extents = [3i64, 4];
    let operand_values: Vec<C64> = (0..12)
        .map(|index| C64 {
            re: index as f64,
            im: -(index as f64),
        })
        .collect();
    let out_layout = Layout::compact(&operand_extents).unwrap();
    // Compact [2, 2]: axis 0 is the batch (fastest), axis 1 the vector, so this
    // reads as batch 0 -> (0, 1) and batch 1 -> (2, 3).
    let indices_values: Vec<i64> = vec![0, 2, 1, 3];
    let indices_extents = [2i64, 2];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_values: Vec<C64> = (0..4)
        .map(|index| C64 {
            re: 0.5 * (index as f64 + 1.0),
            im: -(index as f64),
        })
        .collect();
    let updates_extents = [2i64];
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    let out = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    run_scatter(
        &session,
        DType::C64,
        DType::I64,
        &out,
        &indices,
        &updates,
        &out_layout,
        &indices_layout,
        &updates_layout,
        12,
        4,
        2,
        &plan,
    );
    let actual = download::<C64>(&session, &out, 12);
    let mut expected = operand_values.clone();
    // Destination (0,1) += update 0 and (2,3) += update 1; a compact [3, 4]
    // buffer stores (a0, a1) at a0 + 3 * a1.
    let first = 3;
    let second = 11;
    expected[first].re += updates_values[0].re;
    expected[first].im += updates_values[0].im;
    expected[second].re += updates_values[1].re;
    expected[second].im += updates_values[1].im;
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got.re - want.re).abs() <= 1e-12 && (got.im - want.im).abs() <= 1e-12,
            "position {index}: {got:?} vs {want:?}"
        );
    }
}

#[test]
fn scatter_add_skips_windows_that_do_not_fit_and_keeps_holes() {
    let session = Session::new("gather_scatter");
    // Destination [6] with a 4-wide window on a 3-element axis part: the start 5
    // cannot fit a 4-wide window and is skipped.
    let operand_extents = [6i64];
    let operand_values = vec![2.0f64; 6];
    let out_layout = Layout::compact(&operand_extents).unwrap();
    let indices_values: Vec<i64> = vec![0, 5];
    let indices_extents = [2i64, 1];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_values: Vec<f64> = vec![1.0; 14];
    let updates_extents = [2i64, 7];
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![0],
        update_window_dims: vec![1],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let out = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    run_scatter(
        &session,
        DType::F64,
        DType::I64,
        &out,
        &indices,
        &updates,
        &out_layout,
        &indices_layout,
        &updates_layout,
        6,
        2,
        14,
        &plan,
    );
    // A 7-wide window never fits the 6-element destination, so nothing is added.
    let expected = vec![2.0; 6];
    assert_eq!(download::<f64>(&session, &out, 6), expected);
}

#[test]
fn scatter_composition_is_two_ordered_launches() {
    let session = Session::with_modules(&["movement", "gather_scatter"]);
    // Nonzero operand, so a missing or reordered copy is visible.
    let operand_extents = [5i64];
    let operand_values: Vec<f64> = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    let out_layout = Layout::compact(&operand_extents).unwrap();
    let indices_values: Vec<i64> = vec![1];
    let indices_extents = [1i64];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_values: Vec<f64> = vec![10.0];
    let updates_extents = [1i64];
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };

    let operand = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    let out = sentinel(&session, 5 * 8);

    // Copy the operand into the destination, then add, both enqueued before any
    // observation and synchronized once at the end.
    let copy_plan = plan_copy_exact(DType::F64, 5).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&operand),
        Slot::i32(DType::F64.tag()),
        Slot::i64(5),
    ];
    let before = launches();
    launch_async(
        session.function("t4a_cuda_copy_exact"),
        &copy_plan,
        session.stream(),
        &mut slots,
    );

    let add_plan = plan_scatter_add(
        DType::F64,
        DType::I64,
        &out_layout,
        5,
        &indices_layout,
        1,
        &updates_layout,
        1,
        &plan,
    )
    .unwrap();
    let mirror = CScatterPlan::new(1, 1, &plan).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&indices),
        Slot::ptr(&updates),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::I64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&indices_layout)),
        Slot::layout(&CLayout::new(&updates_layout)),
        Slot::scatter_plan(&mirror),
        Slot::i64(1),
    ];
    launch_async(
        session.function("t4a_cuda_scatter_add"),
        &add_plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 2);
    pub_sync(session.stream());
    assert_eq!(
        download::<f64>(&session, &out, 5),
        vec![-1.0, 8.0, -3.0, -4.0, -5.0]
    );
}

#[test]
fn scatter_with_zero_updates_only_copies() {
    let session = Session::with_modules(&["movement", "gather_scatter"]);
    let operand_values: Vec<f64> = vec![1.0, 2.0, 3.0];
    let operand_extents = [3i64];
    let out_layout = Layout::compact(&operand_extents).unwrap();
    let indices_extents = [0i64, 1];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_extents = [0i64];
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let indices = written(&session, &[]);
    let updates = written(&session, &[]);
    let out = sentinel(&session, 3 * 8);
    let add_plan = plan_scatter_add(
        DType::F64,
        DType::I64,
        &out_layout,
        3,
        &indices_layout,
        0,
        &updates_layout,
        0,
        &plan,
    )
    .unwrap();
    assert!(add_plan.is_empty());

    // The copy is the caller's step and still runs for a nonempty destination.
    let operand = upload(&session, &operand_values);
    let copy_plan = plan_copy_exact(DType::F64, 3).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&operand),
        Slot::i32(DType::F64.tag()),
        Slot::i64(3),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_copy_exact"),
        &copy_plan,
        session.stream(),
        &mut slots,
    );
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&indices),
        Slot::ptr(&updates),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::I64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&indices_layout)),
        Slot::layout(&CLayout::new(&updates_layout)),
        Slot::scatter_plan(&CScatterPlan::new(1, 1, &plan).unwrap()),
        Slot::i64(0),
    ];
    launch(
        session.function("t4a_cuda_scatter_add"),
        &add_plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 1);
    assert_eq!(download::<f64>(&session, &out, 3), operand_values);
}

#[test]
fn scatter_collisions_on_a_large_domain_match_the_cpu_sum() {
    let session = Session::new("gather_scatter");
    let count = 1i64 << 16;
    let destinations = 64usize;
    let operand_values = vec![0.0f64; destinations];
    let out_layout = Layout::compact(&[destinations as i64]).unwrap();
    let indices_values: Vec<i64> = (0..count)
        .map(|index| index % destinations as i64)
        .collect();
    let indices_layout = Layout::compact(&[count]).unwrap();
    let updates_values: Vec<f64> = (0..count).map(|index| (index % 5) as f64).collect();
    let updates_layout = Layout::compact(&[count]).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let out = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    run_scatter(
        &session,
        DType::F64,
        DType::I64,
        &out,
        &indices,
        &updates,
        &out_layout,
        &indices_layout,
        &updates_layout,
        destinations as i64,
        count,
        count,
        &plan,
    );
    let mut expected = vec![0.0f64; destinations];
    for (index, &destination) in indices_values.iter().enumerate() {
        expected[destination as usize] += updates_values[index];
    }
    let actual = download::<f64>(&session, &out, destinations);
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-9,
            "position {index}: {got} vs {want}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gather_raw(
    session: &Session,
    dtype: DType,
    index_dtype: DType,
    operand: &Buf,
    indices: &Buf,
    out_layout: &Layout,
    operand_layout: &Layout,
    operand_len: i64,
    indices_layout: &Layout,
    indices_len: i64,
    plan: &GatherPlan,
    out_capacity: i64,
    out: &Buf,
) {
    let out_len = out_layout.element_count().unwrap();
    let (launch_plan, _) = plan_gather(
        dtype,
        index_dtype,
        out_layout,
        out_capacity,
        operand_layout,
        operand_len,
        indices_layout,
        indices_len,
        plan,
    )
    .unwrap();
    let mirror = CGatherPlan::new(plan, out_layout.rank()).unwrap();
    let mut slots = [
        Slot::ptr(out),
        Slot::ptr(operand),
        Slot::ptr(indices),
        Slot::i32(dtype.tag()),
        Slot::i32(index_dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(operand_layout)),
        Slot::layout(&CLayout::new(indices_layout)),
        Slot::gather_plan(&mirror),
        Slot::i64(out_len),
    ];
    launch(
        session.function("t4a_cuda_gather"),
        &launch_plan,
        session.stream(),
        &mut slots,
    );
}

#[test]
fn gather_with_two_window_dims_and_reversed_offsets() {
    let session = Session::new("gather_scatter");
    // Operand [3, 4, 5]; window on axes 0 and 2, axis 1 collapsed. The offset
    // dims are reversed: window position 0 (operand axis 0) maps to output axis 2
    // and window position 1 (operand axis 2) maps to output axis 0.
    let operand_extents = [3i64, 4, 5];
    let operand_values: Vec<f64> = (0..60).map(|index| (index % 23) as f64 * 0.5).collect();
    let operand_layout = Layout::compact(&operand_extents).unwrap();
    let operand = upload(&session, &operand_values);
    // Scalar starts on axes 0 and 1: indices [4, 2] with index_vector_dim = 1.
    let indices_extents = [4i64, 2];
    let indices_values: Vec<i64> = vec![0, 0, 2, 1, 1, 3, 0, 2];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![0, 2],
        offset_dims: vec![2, 0],
        start_index_map: vec![0, 1],
        slice_sizes: vec![2, 1, 3],
        index_vector_dim: 1,
    };
    // Batch extents [4]; window extents: axis0 -> 2, axis2 -> 3, so the output is
    // [3, 4, 2].
    let out_layout = Layout::compact(&[3, 4, 2]).unwrap();
    let out = run_gather(
        &session,
        DType::F64,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        60,
        &indices_layout,
        8,
        &plan,
        24,
        24,
    );
    let expected = gather_reference(
        &operand_extents,
        &operand_values,
        &indices_extents,
        &indices_values,
        &plan,
        &[3, 4, 2],
    );
    assert_eq!(download::<f64>(&session, &out, 24), expected);
}

#[test]
fn gather_into_a_strided_destination_leaves_holes_untouched() {
    let session = Session::new("gather_scatter");
    let operand_extents = [2i64, 3];
    let operand_values: Vec<f32> = (0..6).map(|index| index as f32 + 1.5).collect();
    let operand_layout = Layout::compact(&operand_extents).unwrap();
    let operand = upload(&session, &operand_values);
    let indices_values: Vec<i64> = vec![1, 0];
    let indices_layout = Layout::compact(&[2i64, 1]).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![1],
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, 3],
        index_vector_dim: 1,
    };
    // Logical output [2, 3] with strides [1, 4] and offset 2: the physical span is
    // 2 + 2*4 + 1 + ... = 12 elements, with holes between rows.
    let out_layout = Layout::new(&[2, 3], &[1, 4], 2).unwrap();
    let alloc = 14usize;
    let out = sentinel(&session, alloc * 4);
    run_gather_raw(
        &session,
        DType::F32,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        6,
        &indices_layout,
        2,
        &plan,
        alloc as i64,
        &out,
    );
    let expected = gather_reference(
        &operand_extents,
        &operand_values
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
        &[2, 1],
        &indices_values,
        &plan,
        &[2, 3],
    );
    let bytes = read_bytes(&session, &out);
    let words: Vec<f32> = from_bytes(&bytes);
    let logical = [2usize, 3, 6, 7, 10, 11];
    for (position, &physical) in logical.iter().enumerate() {
        assert_eq!(
            words[physical].to_bits(),
            (expected[position] as f32).to_bits(),
            "logical position {position}"
        );
    }
    let touched: Vec<usize> = logical.to_vec();
    for element in 0..alloc {
        if !touched.contains(&element) {
            assert_eq!(
                &bytes[element * 4..(element + 1) * 4],
                &[SENTINEL; 4],
                "hole {element}"
            );
        }
    }
}

#[test]
fn scatter_with_two_window_dims_and_reversed_update_windows() {
    let session = Session::new("gather_scatter");
    // Destination [4, 3, 5]; window on axes 0 and 2, axis 1 collapsed. The update
    // window dims are reversed: window position 0 (operand axis 2) reads updates
    // axis 0, window position 1 (operand axis 0) reads updates axis 2.
    let operand_extents = [4i64, 3, 5];
    let operand_values: Vec<f64> = vec![0.0; 60];
    let out_layout = Layout::compact(&operand_extents).unwrap();
    let indices_extents = [2i64, 2];
    // Compact [2, 2]: batch 0 -> (0, 1), batch 1 -> (2, 2).
    let indices_values: Vec<i64> = vec![0, 2, 1, 2];
    let indices_layout = Layout::compact(&indices_extents).unwrap();
    let updates_extents = [2i64, 2, 2];
    let updates_values: Vec<f64> = (0..8).map(|index| index as f64 + 1.0).collect();
    let updates_layout = Layout::compact(&updates_extents).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![2, 0],
        update_window_dims: vec![0, 2],
        scatter_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    let out = upload(&session, &operand_values);
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    run_scatter(
        &session,
        DType::F64,
        DType::I64,
        &out,
        &indices,
        &updates,
        &out_layout,
        &indices_layout,
        &updates_layout,
        60,
        4,
        8,
        &plan,
    );
    let expected = scatter_reference(
        &operand_extents,
        &operand_values,
        &indices_extents,
        &indices_values,
        &updates_extents,
        &updates_values,
        &plan,
    );
    let actual = download::<f64>(&session, &out, 60);
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12,
            "position {index}: {got} vs {want}"
        );
    }
}

#[test]
fn scatter_into_a_strided_destination_leaves_holes_untouched() {
    let session = Session::with_modules(&["movement", "gather_scatter"]);
    // Logical destination [4] with stride 3 and offset 1.
    let out_layout = Layout::new(&[4], &[3], 1).unwrap();
    let alloc = 12usize;
    let out = sentinel(&session, alloc * 8);
    let indices_values: Vec<i64> = vec![1, 3];
    let indices_layout = Layout::compact(&[2i64]).unwrap();
    let updates_values: Vec<f64> = vec![10.0, 20.0];
    let updates_layout = Layout::compact(&[2i64]).unwrap();
    let indices = upload(&session, &indices_values);
    let updates = upload(&session, &updates_values);
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    // Initialize the logical view with zeros through the operand copy.
    let zeros = upload(&session, &[0.0f64; 4]);
    let mut copy_slots = [
        Slot::ptr(&out),
        Slot::ptr(&zeros),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::i64(4),
    ];
    let copy_plan =
        t4a_cuda_kernels::plan_contiguous_to_view(DType::F64, &out_layout, alloc as i64).unwrap();
    launch_async(
        session.function("t4a_cuda_contiguous_to_view"),
        &copy_plan,
        session.stream(),
        &mut copy_slots,
    );
    let add_plan = plan_scatter_add(
        DType::F64,
        DType::I64,
        &out_layout,
        alloc as i64,
        &indices_layout,
        2,
        &updates_layout,
        2,
        &plan,
    )
    .unwrap();
    let mirror = CScatterPlan::new(1, 1, &plan).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&indices),
        Slot::ptr(&updates),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::I64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&indices_layout)),
        Slot::layout(&CLayout::new(&updates_layout)),
        Slot::scatter_plan(&mirror),
        Slot::i64(2),
    ];
    launch_async(
        session.function("t4a_cuda_scatter_add"),
        &add_plan,
        session.stream(),
        &mut slots,
    );
    pub_sync(session.stream());
    let bytes = read_bytes(&session, &out);
    let words: Vec<f64> = from_bytes(&bytes);
    let logical = [1usize, 4, 7, 10];
    let expected = [0.0, 10.0, 0.0, 20.0];
    for (position, &physical) in logical.iter().enumerate() {
        assert_eq!(words[physical], expected[position], "logical {position}");
    }
    for element in 0..alloc {
        if !logical.contains(&element) {
            assert_eq!(
                &bytes[element * 8..(element + 1) * 8],
                &[SENTINEL; 8],
                "hole {element}"
            );
        }
    }
}

#[test]
fn gather_moves_bool_data() {
    let session = Session::new("gather_scatter");
    let operand_values: Vec<u8> = vec![1, 0, 1, 1, 0];
    let operand_layout = Layout::compact(&[5]).unwrap();
    let operand = upload(&session, &operand_values);
    let indices_values: Vec<i64> = vec![4, 0, 2];
    let indices_layout = Layout::compact(&[3]).unwrap();
    let indices = upload(&session, &indices_values);
    let plan = GatherPlan {
        window_dims: vec![],
        offset_dims: vec![],
        start_index_map: vec![0],
        slice_sizes: vec![1],
        index_vector_dim: 1,
    };
    let out_layout = Layout::compact(&[3]).unwrap();
    let out = run_gather(
        &session,
        DType::Bool,
        DType::I64,
        &operand,
        &indices,
        &out_layout,
        &operand_layout,
        5,
        &indices_layout,
        3,
        &plan,
        3,
        3,
    );
    assert_eq!(read_bytes(&session, &out), vec![0, 1, 1]);
}
