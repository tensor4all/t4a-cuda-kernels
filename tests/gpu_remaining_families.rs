//! Revision 10: `broadcast_in_dim`, `concatenate_copy` and the two families that fold
//! into existing entries (`reverse` into strided movement, `scale_in_place` into
//! `binary` with a scalar rhs and an aliased destination).
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_remaining_families`;
//! requires a CUDA device.

mod gpu;

use gpu::{download, launch, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_binary, plan_broadcast_in_dim, plan_concatenate_copy, plan_contiguous_to_view,
    plan_materialize_strided, BinaryOp, CAxisValues, CAxisValuesAny, CLayout, CLayoutAny, DType,
    Layout, ScalarSide, WideLayout,
};

const SENTINEL: u8 = 0xCD;
const PAD: i64 = 8;

fn compact_strides(extents: &[i64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(extents.len());
    let mut stride = 1;
    for &extent in extents {
        strides.push(stride);
        stride *= extent;
    }
    strides
}

fn wide(extents: &[i64], strides: &[i64], offset: i64) -> WideLayout {
    WideLayout::new(extents.to_vec(), strides.to_vec(), offset).unwrap()
}

struct Geometry {
    fast: CLayout,
    general: CLayoutAny,
    _buffers: Vec<Buf>,
}

fn geometry(session: &Session, layout: &WideLayout) -> Geometry {
    let fast = if layout.extents().len() <= t4a_cuda_kernels::MAX_RANK {
        CLayout::new(&Layout::new(layout.extents(), layout.strides(), layout.offset()).unwrap())
    } else {
        CLayout::new(&Layout::new(&[], &[], 0).unwrap())
    };
    let extents = upload(session, layout.extents());
    let strides = upload(session, layout.strides());
    let general = CLayoutAny::new(layout.as_ref(), extents.as_ptr(), strides.as_ptr()).unwrap();
    Geometry {
        fast,
        general,
        _buffers: vec![extents, strides],
    }
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

fn target(session: &Session, bytes: usize) -> Buf {
    written(session, &vec![SENTINEL; bytes])
}

fn assert_guard(session: &Session, buffer: &Buf, payload: usize) {
    let bytes = read_bytes(session, buffer);
    assert!(
        bytes[payload..].iter().all(|&byte| byte == SENTINEL),
        "the destination guard after {payload} bytes was overwritten"
    );
}

/// Payload size in bytes of `count` elements of `dtype`.
fn payload(dtype: DType, count: i64) -> usize {
    count as usize * dtype.size()
}

/// A source allocation whose bytes are a deterministic pattern per element index.
fn patterned(session: &Session, dtype: DType, count: i64) -> Buf {
    patterned_from(session, dtype, count, 0)
}

/// As [`patterned`], with the pattern shifted by `base`.
fn patterned_from(session: &Session, dtype: DType, count: i64, base: i64) -> Buf {
    let bytes = match dtype {
        DType::F32 => {
            let values: Vec<f32> = (0..count)
                .map(|index| (base + index) as f32 + 0.5)
                .collect();
            gpu::as_bytes(&values).to_vec()
        }
        DType::F64 => {
            let values: Vec<f64> = (0..count)
                .map(|index| (base + index) as f64 + 0.5)
                .collect();
            gpu::as_bytes(&values).to_vec()
        }
        DType::I32 => {
            let values: Vec<i32> = (0..count).map(|index| (base + index) as i32 - 3).collect();
            gpu::as_bytes(&values).to_vec()
        }
        DType::I64 => {
            let values: Vec<i64> = (0..count).map(|index| (base + index) as i64 - 7).collect();
            gpu::as_bytes(&values).to_vec()
        }
        DType::Bool => (0..count).map(|index| ((base + index) % 2) as u8).collect(),
        DType::C32 => {
            let values: Vec<(f32, f32)> = (0..count)
                .map(|index| {
                    let value = (base + index) as f32;
                    (value + 0.25, -value - 0.75)
                })
                .collect();
            gpu::as_bytes(&values).to_vec()
        }
        DType::C64 => {
            let values: Vec<(f64, f64)> = (0..count)
                .map(|index| {
                    let value = (base + index) as f64;
                    (value + 0.25, -value - 0.75)
                })
                .collect();
            gpu::as_bytes(&values).to_vec()
        }
    };
    // Slack so a read past the logical end is not a fault but a wrong value.
    let mut padded = bytes;
    padded.extend(std::iter::repeat_n(SENTINEL, (PAD as usize) * dtype.size()));
    written(session, &padded)
}

/// Byte image of one element of the pattern at `index + base`, for the CPU reference.
fn element_bytes_at(dtype: DType, index: i64, base: i64) -> Vec<u8> {
    let index = base + index;
    element_bytes(dtype, index)
}

/// Byte image of one element type per index, for the CPU reference.
fn element_bytes(dtype: DType, index: i64) -> Vec<u8> {
    match dtype {
        DType::F32 => ((index as f32) + 0.5).to_le_bytes().to_vec(),
        DType::F64 => ((index as f64) + 0.5).to_le_bytes().to_vec(),
        DType::I32 => (index as i32 - 3).to_le_bytes().to_vec(),
        DType::I64 => (index as i64 - 7).to_le_bytes().to_vec(),
        DType::Bool => vec![(index % 2) as u8],
        DType::C32 => {
            let mut bytes = ((index as f32) + 0.25).to_le_bytes().to_vec();
            bytes.extend((-(index as f32) - 0.75).to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = ((index as f64) + 0.25).to_le_bytes().to_vec();
            bytes.extend((-(index as f64) - 0.75).to_le_bytes());
            bytes
        }
    }
}

const PAYLOAD_DTYPES: [DType; 7] = [
    DType::F32,
    DType::F64,
    DType::I32,
    DType::I64,
    DType::Bool,
    DType::C32,
    DType::C64,
];

/// Run one broadcast launch (fast or general) and return the destination image.
fn broadcast(
    session: &Session,
    general: bool,
    dtype: DType,
    out: &WideLayout,
    input: &WideLayout,
    dims: &[i64],
    source: &Buf,
) -> (Buf, usize) {
    let out_count = out.element_count().unwrap();
    let in_count = input.element_count().unwrap();
    let bytes = payload(dtype, out_count);
    let dst = target(session, bytes + (PAD as usize) * dtype.size());
    let out_geom = geometry(session, out);
    let in_geom = geometry(session, input);
    let plan = plan_broadcast_in_dim(dtype, out, out_count, input, in_count, dims).unwrap();
    let fast_dims = if dims.len() <= t4a_cuda_kernels::MAX_RANK {
        CAxisValues::new(dims).unwrap()
    } else {
        CAxisValues::new(&[0]).unwrap()
    };
    let dims_buf = upload(session, dims);
    let dims_any = CAxisValuesAny::new(dims.len(), dims_buf.as_ptr()).unwrap();
    if general {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout_any(&out_geom.general),
            Slot::layout_any(&in_geom.general),
            Slot::axis_values_any(&dims_any),
            Slot::i64(out_count),
        ];
        launch(
            session.function("t4a_cuda_broadcast_in_dim_any"),
            &plan,
            session.stream(),
            &mut slots,
        );
    } else {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout(&out_geom.fast),
            Slot::layout(&in_geom.fast),
            Slot::axis_values(&fast_dims),
            Slot::i64(out_count),
        ];
        launch(
            session.function("t4a_cuda_broadcast_in_dim"),
            &plan,
            session.stream(),
            &mut slots,
        );
    }
    (dst, bytes)
}

/// The CPU image a rank-mapped broadcast must produce.
fn broadcast_image(dtype: DType, out: &WideLayout, input: &WideLayout, dims: &[i64]) -> Vec<u8> {
    let out_count = out.element_count().unwrap();
    let out_strides = out.strides().to_vec();
    let in_strides = input.strides().to_vec();
    let out_extents = out.extents().to_vec();
    let in_extents = input.extents().to_vec();
    let mut image = Vec::with_capacity(payload(dtype, out_count));
    for flat in 0..out_count {
        // Destination coordinates in flat logical order.
        let mut rest = flat;
        let mut coordinates = vec![0i64; out_extents.len()];
        for axis in 0..out_extents.len() {
            coordinates[axis] = rest % out_extents[axis];
            rest /= out_extents[axis];
        }
        let mut source = input.offset();
        for src_axis in 0..dims.len() {
            let coordinate = if in_extents[src_axis] == 1 {
                0
            } else {
                coordinates[dims[src_axis] as usize]
            };
            source += coordinate * in_strides[src_axis];
        }
        image.extend(element_bytes(dtype, source));
    }
    image
}

#[test]
fn broadcast_in_dim_matches_the_cpu_reference_for_every_dtype() {
    let session = Session::new("movement");
    // Identity map, rank 2.
    let out = wide(&[2, 3], &compact_strides(&[2, 3]), 0);
    let input = wide(&[2, 3], &compact_strides(&[2, 3]), 0);
    for dtype in PAYLOAD_DTYPES {
        let source = patterned(&session, dtype, 6);
        let expected = broadcast_image(dtype, &out, &input, &[0, 1]);
        for general in [false, true] {
            let (dst, bytes) = broadcast(&session, general, dtype, &out, &input, &[0, 1], &source);
            assert_eq!(
                read_bytes(&session, &dst)[..bytes],
                expected[..],
                "identity map {dtype:?} general={general}"
            );
            assert_guard(&session, &dst, bytes);
        }
    }

    // An extent-1 input axis is broadcast, and the input extent differs per axis.
    let out = wide(&[2, 3, 4], &compact_strides(&[2, 3, 4]), 0);
    let input = wide(&[2, 1, 4], &compact_strides(&[2, 1, 4]), 0);
    let source = patterned(&session, DType::F64, 8);
    let expected = broadcast_image(DType::F64, &out, &input, &[0, 1, 2]);
    for general in [false, true] {
        let (dst, bytes) = broadcast(
            &session,
            general,
            DType::F64,
            &out,
            &input,
            &[0, 1, 2],
            &source,
        );
        assert_eq!(
            read_bytes(&session, &dst)[..bytes],
            expected[..],
            "broadcast axis"
        );
    }

    // A non-self-inverse permutation: the input axis 0 lands on the destination axis
    // 2, the input axis 1 on 0 and the input axis 2 on 1.
    let in_extents = [2i64, 3, 2];
    let input = wide(&in_extents, &compact_strides(&in_extents), 0);
    // The destination extents must agree with the input on each mapped axis: the
    // input axis 0 (2) lands on 2, axis 1 (3) on 0 and axis 2 (2) on 1.
    let permuted_out = wide(&[3, 2, 2], &compact_strides(&[3, 2, 2]), 0);
    let input_payload: i64 = 12;
    let source = patterned(&session, DType::F64, input_payload);
    let dims = [2i64, 0, 1];
    let expected = broadcast_image(DType::F64, &permuted_out, &input, &dims);
    for general in [false, true] {
        let (dst, bytes) = broadcast(
            &session,
            general,
            DType::F64,
            &permuted_out,
            &input,
            &dims,
            &source,
        );
        assert_eq!(
            read_bytes(&session, &dst)[..bytes],
            expected[..],
            "permuted map"
        );
    }

    // A scalar input (rank 0): every destination element takes element zero.
    let scalar = wide(&[], &[], 0);
    let source = patterned(&session, DType::F32, 1);
    let expected = broadcast_image(DType::F32, &out, &scalar, &[]);
    for general in [false, true] {
        let (dst, bytes) = broadcast(&session, general, DType::F32, &out, &scalar, &[], &source);
        assert_eq!(
            read_bytes(&session, &dst)[..bytes],
            expected[..],
            "scalar input"
        );
    }

    // Rank 9 and 16 with non-unit extents on the high axes and more than one block.
    for rank in [9usize, 16] {
        let extents: Vec<i64> = (0..rank).map(|axis| 2 + axis as i64 % 2).collect();
        let out = wide(&extents, &compact_strides(&extents), 0);
        let mut in_extents = extents.clone();
        in_extents[1] = 1;
        in_extents[rank - 1] = 1;
        let input = wide(&in_extents, &compact_strides(&in_extents), 0);
        let count = input.element_count().unwrap();
        let source = patterned(&session, DType::F64, count);
        let dims: Vec<i64> = (0..rank as i64).collect();
        let expected = broadcast_image(DType::F64, &out, &input, &dims);
        let (dst, bytes) = broadcast(&session, true, DType::F64, &out, &input, &dims, &source);
        let image = read_bytes(&session, &dst);
        assert_eq!(image[..bytes], expected[..], "rank {rank}");
        assert_guard(&session, &dst, bytes);
    }
}

/// One concatenate launch of `input` into `out` at `axis_offset` along `axis`.
fn concatenate(
    session: &Session,
    general: bool,
    dtype: DType,
    out: &WideLayout,
    input: &WideLayout,
    axis: u32,
    axis_offset: i64,
    out_len: i64,
    destination: &Buf,
    source: &Buf,
) {
    let in_count = input.element_count().unwrap();
    let out_geom = geometry(session, out);
    let in_geom = geometry(session, input);
    let plan =
        plan_concatenate_copy(dtype, out, out_len, input, in_count, axis, axis_offset).unwrap();
    if general {
        let mut slots = [
            Slot::ptr(destination),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout_any(&out_geom.general),
            Slot::layout_any(&in_geom.general),
            Slot::i32(axis as i32),
            Slot::i64(axis_offset),
            Slot::i64(in_count),
        ];
        launch(
            session.function("t4a_cuda_concatenate_copy_any"),
            &plan,
            session.stream(),
            &mut slots,
        );
    } else {
        let mut slots = [
            Slot::ptr(destination),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout(&out_geom.fast),
            Slot::layout(&in_geom.fast),
            Slot::i32(axis as i32),
            Slot::i64(axis_offset),
            Slot::i64(in_count),
        ];
        launch(
            session.function("t4a_cuda_concatenate_copy"),
            &plan,
            session.stream(),
            &mut slots,
        );
    }
}

#[test]
fn concatenate_copy_matches_the_cpu_reference_for_every_dtype() {
    let session = Session::new("movement");
    let out_extents = [2i64, 6];
    let out = wide(&out_extents, &compact_strides(&out_extents), 0);
    let in_extents = [2i64, 3];
    let input = wide(&in_extents, &compact_strides(&in_extents), 0);
    let out_count: i64 = out_extents.iter().product();
    let in_count: i64 = in_extents.iter().product();
    let in_strides = input.strides().to_vec();
    let out_strides = out.strides().to_vec();

    for dtype in PAYLOAD_DTYPES {
        // Three inputs: two distinct payloads, then a third that overwrites the first
        // half again, so the final image proves stream ordering.
        let first = patterned_from(&session, dtype, in_count, 0);
        let second = patterned_from(&session, dtype, in_count, 100);
        let third = patterned_from(&session, dtype, in_count, 200);
        let mut expected = vec![0u8; payload(dtype, out_count)];
        let mut write = |offset: i64, base: i64, image: &mut Vec<u8>| {
            for flat in 0..in_count {
                let mut rest = flat;
                let mut coordinates = vec![0i64; in_extents.len()];
                for axis in 0..in_extents.len() {
                    coordinates[axis] = rest % in_extents[axis];
                    rest /= in_extents[axis];
                }
                let mut source = 0i64;
                for axis in 0..in_extents.len() {
                    source += coordinates[axis] * in_strides[axis];
                }
                coordinates[1] += offset;
                let mut destination = 0i64;
                for axis in 0..out_extents.len() {
                    destination += coordinates[axis] * out_strides[axis];
                }
                let start = destination as usize * dtype.size();
                image[start..start + dtype.size()]
                    .copy_from_slice(&element_bytes_at(dtype, source, base));
            }
        };
        write(0, 0, &mut expected);
        write(3, 100, &mut expected);
        write(0, 200, &mut expected);

        for general in [false, true] {
            let destination = written(&session, &vec![0u8; payload(dtype, out_count)]);
            concatenate(
                &session,
                general,
                dtype,
                &out,
                &input,
                1,
                0,
                out_count,
                &destination,
                &first,
            );
            concatenate(
                &session,
                general,
                dtype,
                &out,
                &input,
                1,
                3,
                out_count,
                &destination,
                &second,
            );
            concatenate(
                &session,
                general,
                dtype,
                &out,
                &input,
                1,
                0,
                out_count,
                &destination,
                &third,
            );
            assert_eq!(
                read_bytes(&session, &destination),
                expected,
                "concatenate_copy {dtype:?} general={general}"
            );
        }
    }

    // A genuinely non-compact destination: extents [2, 4, 2], strides [1, 3, 12]
    // (highest reachable index 1 + 9 + 12 = 22).
    let strided_extents = [2i64, 4, 2];
    let strided_strides = [1i64, 3, 12];
    let strided_out = wide(&strided_extents, &strided_strides, 0);
    let source_extents = [2i64, 2, 2];
    let source_layout = wide(&source_extents, &compact_strides(&source_extents), 0);
    let capacity = 23i64;
    let source_count: i64 = source_extents.iter().product();
    let source = patterned_from(&session, DType::F64, source_count, 0);
    let in_strides = source_layout.strides().to_vec();
    let mut expected = vec![SENTINEL; payload(DType::F64, capacity)];
    for flat in 0..source_count {
        let mut rest = flat;
        let mut coordinates = vec![0i64; 3];
        for axis in 0..3 {
            coordinates[axis] = rest % source_extents[axis];
            rest /= source_extents[axis];
        }
        let mut source_index = 0i64;
        for axis in 0..3 {
            source_index += coordinates[axis] * in_strides[axis];
        }
        coordinates[1] += 1;
        let mut destination_index = 0i64;
        for axis in 0..3 {
            destination_index += coordinates[axis] * strided_strides[axis];
        }
        let start = destination_index as usize * DType::F64.size();
        expected[start..start + DType::F64.size()]
            .copy_from_slice(&element_bytes(DType::F64, source_index));
    }
    for general in [false, true] {
        let destination = written(&session, &vec![SENTINEL; payload(DType::F64, capacity)]);
        concatenate(
            &session,
            general,
            DType::F64,
            &strided_out,
            &source_layout,
            1,
            1,
            capacity,
            &destination,
            &source,
        );
        assert_eq!(
            read_bytes(&session, &destination),
            expected,
            "strided destination general={general}"
        );
    }

    // Rank 16: axes 8 and up carry non-unit extents and the shifted axis is high.
    let rank = 16usize;
    let extents: Vec<i64> = (0..rank).map(|axis| 2 + axis as i64 % 2).collect();
    let out = wide(&extents, &compact_strides(&extents), 0);
    let mut in_extents = extents.clone();
    in_extents[9] = 2;
    let input = wide(&in_extents, &compact_strides(&in_extents), 0);
    let in_count = input.element_count().unwrap();
    let out_count = out.element_count().unwrap();
    let source = patterned_from(&session, DType::F64, in_count, 0);
    let destination = written(&session, &vec![0u8; payload(DType::F64, out_count)]);
    concatenate(
        &session,
        true,
        DType::F64,
        &out,
        &input,
        9,
        0,
        out_count,
        &destination,
        &source,
    );
    let in_strides = input.strides().to_vec();
    let out_strides = out.strides().to_vec();
    let mut expected = vec![0u8; payload(DType::F64, out_count)];
    for flat in 0..in_count {
        let mut rest = flat;
        let mut coordinates = vec![0i64; rank];
        for axis in 0..rank {
            coordinates[axis] = rest % in_extents[axis];
            rest /= in_extents[axis];
        }
        let mut source_index = 0i64;
        for axis in 0..rank {
            source_index += coordinates[axis] * in_strides[axis];
        }
        let mut destination_index = 0i64;
        for axis in 0..rank {
            destination_index += coordinates[axis] * out_strides[axis];
        }
        let start = destination_index as usize * DType::F64.size();
        expected[start..start + DType::F64.size()]
            .copy_from_slice(&element_bytes(DType::F64, source_index));
    }
    assert_eq!(
        read_bytes(&session, &destination),
        expected,
        "concatenate_copy rank 16"
    );
}

#[test]
fn copy_entries_preserve_nonfinite_payloads_bitwise() {
    let session = Session::new("movement");
    // NaN payloads, signed zeros and infinities must survive a copy unchanged, which a
    // value comparison would miss.
    let f64_values: [f64; 6] = [
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0002),
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let bytes = gpu::as_bytes(&f64_values).to_vec();
    let source = written(&session, &bytes);
    let out = wide(&[6], &compact_strides(&[6]), 0);
    let input = wide(&[6], &compact_strides(&[6]), 0);
    for general in [false, true] {
        let (dst, size) = broadcast(&session, general, DType::F64, &out, &input, &[0], &source);
        assert_eq!(
            read_bytes(&session, &dst)[..size],
            bytes[..],
            "broadcast nonfinite"
        );
    }
    let destination = written(&session, &vec![0u8; 6 * DType::F64.size()]);
    for general in [false, true] {
        let destination = written(&session, &vec![0u8; 6 * DType::F64.size()]);
        concatenate(
            &session,
            general,
            DType::F64,
            &out,
            &input,
            0,
            0,
            6,
            &destination,
            &source,
        );
        assert_eq!(
            read_bytes(&session, &destination),
            bytes,
            "concatenate nonfinite"
        );
    }
    let _ = destination;

    // A genuinely non-compact broadcast source: extents [2, 3] with strides [1, 5]
    // (capacity 11), so the source addressing is not a flat copy.
    let out = wide(&[2, 3], &compact_strides(&[2, 3]), 0);
    let input = wide(&[2, 3], &[1, 5], 0);
    // Highest reachable index: 1 + 2 * 5 = 11.
    let capacity = 12i64;
    let mut host = vec![SENTINEL; payload(DType::F64, capacity)];
    for flat in 0..6i64 {
        let coordinate0 = flat % 2;
        let coordinate1 = flat / 2;
        let index = (coordinate0 + coordinate1 * 5) as usize;
        host[index * 8..index * 8 + 8].copy_from_slice(&element_bytes(DType::F64, index as i64));
    }
    let source = written(&session, &host);
    for general in [false, true] {
        let (dst, size) = broadcast_source_capacity(
            &session,
            general,
            DType::F64,
            &out,
            &input,
            &[0, 1],
            &source,
            capacity,
        );
        let expected: Vec<u8> = (0..6i64)
            .flat_map(|flat| {
                let coordinate0 = flat % 2;
                let coordinate1 = flat / 2;
                let index = (coordinate0 + coordinate1 * 5) as i64;
                element_bytes(DType::F64, index)
            })
            .collect();
        assert_eq!(
            read_bytes(&session, &dst)[..size],
            expected[..],
            "strided broadcast source"
        );
    }
}

/// Broadcast with an explicit source allocation capacity, for a strided source.
fn broadcast_source_capacity(
    session: &Session,
    general: bool,
    dtype: DType,
    out: &WideLayout,
    input: &WideLayout,
    dims: &[i64],
    source: &Buf,
    in_capacity: i64,
) -> (Buf, usize) {
    let out_count = out.element_count().unwrap();
    let bytes = payload(dtype, out_count);
    let dst = target(session, bytes + (PAD as usize) * dtype.size());
    let out_geom = geometry(session, out);
    let in_geom = geometry(session, input);
    let plan = plan_broadcast_in_dim(dtype, out, out_count, input, in_capacity, dims).unwrap();
    let fast_dims = CAxisValues::new(dims).unwrap();
    let dims_buf = upload(session, dims);
    let dims_any = CAxisValuesAny::new(dims.len(), dims_buf.as_ptr()).unwrap();
    if general {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout_any(&out_geom.general),
            Slot::layout_any(&in_geom.general),
            Slot::axis_values_any(&dims_any),
            Slot::i64(out_count),
        ];
        launch(
            session.function("t4a_cuda_broadcast_in_dim_any"),
            &plan,
            session.stream(),
            &mut slots,
        );
    } else {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(source),
            Slot::i32(dtype.tag()),
            Slot::layout(&out_geom.fast),
            Slot::layout(&in_geom.fast),
            Slot::axis_values(&fast_dims),
            Slot::i64(out_count),
        ];
        launch(
            session.function("t4a_cuda_broadcast_in_dim"),
            &plan,
            session.stream(),
            &mut slots,
        );
    }
    (dst, bytes)
}

#[test]
fn new_entries_agree_between_variants_up_to_rank_eight() {
    let session = Session::new("movement");
    for rank in 1..=8usize {
        let extents: Vec<i64> = (0..rank).map(|axis| 2 + axis as i64 % 3).collect();
        let strides = compact_strides(&extents);
        let out = wide(&extents, &strides, 0);
        let input = wide(&extents, &strides, 0);
        let count = input.element_count().unwrap();
        let source = patterned(&session, DType::F64, count);
        let dims: Vec<i64> = (0..rank as i64).collect();

        let (fast_dst, bytes) =
            broadcast(&session, false, DType::F64, &out, &input, &dims, &source);
        let (general_dst, _) = broadcast(&session, true, DType::F64, &out, &input, &dims, &source);
        assert_eq!(
            read_bytes(&session, &fast_dst),
            read_bytes(&session, &general_dst),
            "broadcast_in_dim rank {rank}"
        );
        let _ = bytes;

        let destination = written(&session, &vec![0u8; payload(DType::F64, count)]);
        concatenate(
            &session,
            false,
            DType::F64,
            &out,
            &input,
            0,
            0,
            count,
            &destination,
            &source,
        );
        let fast_image = read_bytes(&session, &destination);
        let destination = written(&session, &vec![0u8; payload(DType::F64, count)]);
        concatenate(
            &session,
            true,
            DType::F64,
            &out,
            &input,
            0,
            0,
            count,
            &destination,
            &source,
        );
        assert_eq!(
            fast_image,
            read_bytes(&session, &destination),
            "concatenate_copy rank {rank}"
        );
    }
}

#[test]
fn reverse_is_a_strided_copy_in_both_directions() {
    let session = Session::new("movement");
    // Three reversed axes out of four, odd extents, non-zero offset.
    let extents = [3i64, 5, 2, 4];
    let strides = compact_strides(&extents);
    let offset = 0i64;
    let masked = [0usize, 2, 3];
    let mut reversed_strides = strides.to_vec();
    let mut reversed_offset = offset;
    for axis in masked {
        reversed_offset += (extents[axis] - 1) * strides[axis];
        reversed_strides[axis] = -strides[axis];
    }
    let compact = wide(&extents, &strides, offset);
    let reversed = wide(&extents, &reversed_strides, reversed_offset);
    let count: i64 = extents.iter().product();
    let source = patterned(&session, DType::F64, count);
    let expected: Vec<u8> = view_indices(&extents, &reversed_strides, reversed_offset)
        .into_iter()
        .flat_map(|index| element_bytes(DType::F64, index))
        .collect();

    // Reading direction: materialize_strided gathers the reversed view.
    let compact_geom = geometry(&session, &compact);
    let reversed_geom = geometry(&session, &reversed);
    let dst = target(&session, payload(DType::F64, count));
    let plan = plan_materialize_strided(DType::F64, &reversed, count).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&source),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&reversed_geom.fast),
        Slot::i64(count),
    ];
    launch(
        session.function("t4a_cuda_materialize_strided"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        read_bytes(&session, &dst)[..payload(DType::F64, count)],
        expected[..],
        "reverse read"
    );

    // Writing direction: contiguous_to_view scatters a compact source into the
    // reversed destination.
    let dst = target(&session, ((count + PAD) as usize) * DType::F64.size());
    let plan = plan_contiguous_to_view(DType::F64, &reversed, count).unwrap();
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&source),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&reversed_geom.fast),
        Slot::i64(count),
    ];
    launch(
        session.function("t4a_cuda_contiguous_to_view"),
        &plan,
        session.stream(),
        &mut slots,
    );
    // The reversed view is a permutation, so writing the compact payload through it
    // places element i at the reversed index of i.
    let image = read_bytes(&session, &dst);
    let expected_written: Vec<u8> = (0..count)
        .map(|flat| {
            let index = view_indices(&extents, &reversed_strides, reversed_offset)[flat as usize];
            (index, element_bytes(DType::F64, flat))
        })
        .fold(
            vec![SENTINEL; (count as usize) * 8],
            |mut bytes, (index, element)| {
                let start = index as usize * 8;
                bytes[start..start + 8].copy_from_slice(&element);
                bytes
            },
        )
        .into_iter()
        .collect();
    assert_eq!(
        image[..(count as usize) * 8],
        expected_written[..],
        "reverse write"
    );
    let _ = &compact_geom;
}

#[test]
fn scale_in_place_is_the_scalar_mul_of_binary() {
    let session = Session::new("elementwise");
    // The ABI documents that `MUL` may scale in place through `a` (`out` aliases
    // `lhs`), which is tenferro's `scale_in_place_float/complex_kernel`.
    for dtype in [DType::F32, DType::F64, DType::C32, DType::C64] {
        let count = 12i64;
        let buffer = patterned(&session, dtype, count);
        let factor = patterned(&session, dtype, 1);
        let status = written(&session, &vec![0u8; 4]);
        let before = read_bytes(&session, &buffer);
        let plan = plan_binary(BinaryOp::Mul, dtype, ScalarSide::Right, count).unwrap();
        let mut slots = [
            Slot::ptr(&buffer),
            Slot::ptr(&buffer),
            Slot::ptr(&factor),
            Slot::i32(dtype.tag()),
            Slot::i32(BinaryOp::Mul.tag()),
            Slot::i32(ScalarSide::Right.tag()),
            Slot::i64(count),
            Slot::ptr(&status),
        ];
        launch(
            session.function("t4a_cuda_binary"),
            &plan,
            session.stream(),
            &mut slots,
        );
        let after = read_bytes(&session, &buffer);
        assert_ne!(
            before, after,
            "in-place scale {dtype:?} must change the buffer"
        );
        // The status word stays clear: the arithmetic is not a checked integer op.
        assert_eq!(
            read_bytes(&session, &status),
            vec![0u8; 4],
            "status {dtype:?}"
        );
    }
}
