//! Numerical GPU tests for ABI revision 11: the layout-taking triangle views
//! (`t4a_cuda_{tril,triu}_view` and their `_any` twins) and the window write
//! (`t4a_cuda_dynamic_update_slice` and its `_any` twin).
//!
//! Every assertion compares the **whole destination allocation bytes** — the
//! payload plus a sentinel slab after it — so NaN payloads, signed zeros,
//! subnormals, masked zeros and the tail guard are part of the comparison, and
//! the CPU reference never reads a device result.
//!
//! Run with
//! `cargo test -j 4 --features gpu-tests --test gpu_tril_dus -- --test-threads 1`;
//! requires a CUDA device.

mod gpu;

use gpu::{launch, launches, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_dynamic_update_slice, plan_tril_view, plan_triu_view, CLayout, CLayoutAny, DType, Launch,
    Layout, LayoutRef, RankVariant, WideLayout,
};

const SENTINEL: u8 = 0xCD;
/// Elements of sentinel slab after every payload, so an overrun is visible and
/// an under-written tail is caught.
const SLACK: i64 = 4;

/// Deterministic byte payload: an LCG over the whole allocation, so every
/// element holds a different bit pattern and nonfinite values, signed zeros and
/// NaN payloads are exercised by construction.
fn payload_seeded(seed: u64, bytes: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

fn payload(bytes: usize) -> Vec<u8> {
    payload_seeded(0x2545_f491_4f6c_dd1d, bytes)
}

/// A payload of `count` elements followed by the sentinel slab.
fn guarded(seed: u64, count: i64, element: usize) -> Vec<u8> {
    let mut bytes = payload_seeded(seed, count as usize * element);
    bytes.extend(std::iter::repeat_n(SENTINEL, SLACK as usize * element));
    bytes
}

fn product(extents: &[i64]) -> i64 {
    extents.iter().product()
}

/// Compact strides of `extents`, axis 0 fastest.
fn compact_strides(extents: &[i64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(extents.len());
    let mut stride = 1;
    for &extent in extents {
        strides.push(stride);
        stride *= extent;
    }
    strides
}

/// Strides of `extents` in the given axis order, fastest first; a permutation
/// of the compact strides, so the view is injective but non-compact.
fn permuted_strides(extents: &[i64], order: &[usize]) -> Vec<i64> {
    let mut strides = vec![0i64; extents.len()];
    let mut stride = 1;
    for &axis in order {
        strides[axis] = stride;
        stride *= extents[axis];
    }
    strides
}

/// Number of elements a layout can address, i.e. one past its largest index.
fn capacity(extents: &[i64], strides: &[i64], offset: i64) -> i64 {
    let mut max = offset;
    let mut min = offset;
    for axis in 0..extents.len() {
        let step = (extents[axis] - 1) * strides[axis];
        if step > 0 {
            max += step;
        } else {
            min += step;
        }
    }
    assert!(
        min >= 0,
        "test layouts keep the minimum address non-negative"
    );
    max + 1
}

/// The device geometry arrays of one general-path layout ([`LayoutRef::pack`]
/// order: extents, then strides), kept alive for the launches that point at them.
struct Geometry {
    extents: Buf,
    strides: Buf,
}

impl Geometry {
    fn upload(session: &Session, layout: LayoutRef<'_>) -> Geometry {
        Geometry {
            extents: upload(session, layout.extents()),
            strides: upload(session, layout.strides()),
        }
    }

    fn mirror(&self, layout: LayoutRef<'_>) -> CLayoutAny {
        CLayoutAny::new(layout, self.extents.as_ptr(), self.strides.as_ptr()).unwrap()
    }
}

// --- tril/triu views --------------------------------------------------------

/// One input view: extents, signed strides and an element offset.
#[derive(Clone, Debug)]
struct View {
    extents: Vec<i64>,
    strides: Vec<i64>,
    offset: i64,
}

impl View {
    fn count(&self) -> i64 {
        product(&self.extents)
    }

    fn capacity(&self) -> i64 {
        capacity(&self.extents, &self.strides, self.offset)
    }
}

/// The geometries the design lists: compact, transposed, offset, negative and
/// zero stride, batched rank 3 and a non-compact rank 4.
fn view_cases() -> Vec<View> {
    vec![
        View {
            extents: vec![3, 4],
            strides: vec![1, 3],
            offset: 0,
        },
        View {
            extents: vec![3, 4],
            strides: vec![4, 1],
            offset: 5,
        },
        View {
            extents: vec![3, 4],
            strides: vec![1, -4],
            offset: 12,
        },
        View {
            extents: vec![3, 4],
            strides: vec![1, 0],
            offset: 2,
        },
        View {
            extents: vec![3, 4, 2],
            strides: compact_strides(&[3, 4, 2]),
            offset: 0,
        },
        View {
            extents: vec![2, 3, 2, 2],
            strides: permuted_strides(&[2, 3, 2, 2], &[1, 0, 3, 2]),
            offset: 1,
        },
        View {
            extents: vec![2, 3, 2, 2, 2],
            strides: permuted_strides(&[2, 3, 2, 2, 2], &[4, 2, 0, 3, 1]),
            offset: 2,
        },
        View {
            extents: vec![2, 2, 3, 2, 2, 2, 2, 2],
            strides: permuted_strides(&[2, 2, 3, 2, 2, 2, 2, 2], &[1, 3, 5, 7, 0, 2, 4, 6]),
            offset: 3,
        },
    ]
}

/// Independent model of the view mask: enumerate the compact output positions,
/// decode the input coordinate, keep or zero by `col - row` against `k` and copy
/// the element's bytes through the input layout. The destination allocation
/// holds `out_len` elements and is sentinel-filled, so a masked element must
/// become exactly zero and the slab after the payload must stay untouched.
fn triangular_expected(
    view: &View,
    input_bytes: &[u8],
    element: usize,
    keep_below: bool,
    k: i64,
    out_len: i64,
) -> Vec<u8> {
    let mut out = vec![SENTINEL; out_len as usize * element];
    for flat in 0..view.count() {
        let mut rest = flat;
        let mut index = view.offset;
        let mut row = 0;
        let mut col = 0;
        for axis in 0..view.extents.len() {
            let coordinate = rest % view.extents[axis];
            rest /= view.extents[axis];
            if axis == 0 {
                row = coordinate;
            }
            if axis == 1 {
                col = coordinate;
            }
            index += coordinate * view.strides[axis];
        }
        let keep = if keep_below {
            col - row <= k
        } else {
            col - row >= k
        };
        let destination = flat as usize * element;
        if keep {
            let source = index as usize * element;
            out[destination..destination + element]
                .copy_from_slice(&input_bytes[source..source + element]);
        } else {
            out[destination..destination + element].fill(0);
        }
    }
    out
}

/// One fast-path view launch; the destination is compared by the caller.
#[allow(clippy::too_many_arguments)]
fn launch_view(
    session: &Session,
    symbol: &str,
    plan: &Launch,
    dst: &Buf,
    src: &Buf,
    dtype: DType,
    layout: &Layout,
    k: i64,
    len: i64,
) {
    let mut slots = [
        Slot::ptr(dst),
        Slot::ptr(src),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(layout)),
        Slot::i64(k),
        Slot::i64(len),
    ];
    launch(session.function(symbol), plan, session.stream(), &mut slots);
}

/// One general-path view launch.
#[allow(clippy::too_many_arguments)]
fn launch_view_any(
    session: &Session,
    symbol: &str,
    plan: &Launch,
    dst: &Buf,
    src: &Buf,
    dtype: DType,
    layout_any: &CLayoutAny,
    k: i64,
    len: i64,
) {
    let mut slots = [
        Slot::ptr(dst),
        Slot::ptr(src),
        Slot::i32(dtype.tag()),
        Slot::layout_any(layout_any),
        Slot::i64(k),
        Slot::i64(len),
    ];
    launch(session.function(symbol), plan, session.stream(), &mut slots);
}

#[test]
fn triangular_views_match_a_cpu_reference_for_every_dtype_and_k() {
    let session = Session::new("diagonal");
    for dtype in DType::ALL {
        let element = dtype.size();
        for view in view_cases() {
            let count = view.count();
            let capacity = view.capacity();
            let in_len = capacity + SLACK;
            let input_bytes = payload(in_len as usize * element);
            let src = written(&session, &input_bytes);
            let input = Layout::new(&view.extents, &view.strides, view.offset).unwrap();
            let geometry = Geometry::upload(&session, input.as_ref());
            let general = geometry.mirror(input.as_ref());
            // A longer output allocation than the element count is legal, and its
            // slack must keep the sentinel while the payload is fully written.
            let out_len = count + SLACK;
            for k in [i64::MIN, -1, 0, 1, i64::MAX] {
                for keep_below in [true, false] {
                    let expected =
                        triangular_expected(&view, &input_bytes, element, keep_below, k, out_len);
                    let plan = if keep_below {
                        plan_tril_view(dtype, out_len, &input, in_len).unwrap()
                    } else {
                        plan_triu_view(dtype, out_len, &input, in_len).unwrap()
                    };
                    assert_eq!(plan.variant, RankVariant::Fast);
                    assert_eq!(
                        plan.grid,
                        [u32::try_from((count as u64).div_ceil(256)).unwrap(), 1, 1]
                    );
                    // Both symbols must produce the reference bytes.
                    for general_path in [false, true] {
                        let symbol = match (keep_below, general_path) {
                            (true, false) => "t4a_cuda_tril_view",
                            (true, true) => "t4a_cuda_tril_view_any",
                            (false, false) => "t4a_cuda_triu_view",
                            (false, true) => "t4a_cuda_triu_view_any",
                        };
                        let dst = written(&session, &vec![SENTINEL; out_len as usize * element]);
                        let before = launches();
                        if general_path {
                            launch_view_any(
                                &session, symbol, &plan, &dst, &src, dtype, &general, k, count,
                            );
                        } else {
                            launch_view(
                                &session, symbol, &plan, &dst, &src, dtype, &input, k, count,
                            );
                        }
                        assert_eq!(launches(), before + 1);
                        assert_eq!(
                            read_bytes(&session, &dst),
                            expected,
                            "{dtype:?} {view:?} k={k} {symbol}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn triangular_views_reach_rank_nine_through_the_general_symbol() {
    let session = Session::new("diagonal");
    // Rank 9 with a non-unit extent on axis 8 (and non-unit extents below it):
    // the general symbol is the only one that can accept the plan, and its
    // addressing must reach the ninth axis.
    let extents = vec![2i64, 3, 2, 2, 2, 2, 2, 2, 3];
    let strides = permuted_strides(&extents, &[0, 2, 4, 6, 8, 1, 3, 5, 7]);
    let view = View {
        extents,
        strides,
        offset: 0,
    };
    let wide = WideLayout::new(view.extents.clone(), view.strides.clone(), view.offset).unwrap();
    let element = DType::F32.size();
    let in_len = view.capacity();
    assert_eq!(in_len, view.count());
    let input_bytes = payload(in_len as usize * element);
    let src = written(&session, &input_bytes);
    let out_len = view.count() + SLACK;
    let geometry = Geometry::upload(&session, wide.as_ref());
    let general = geometry.mirror(wide.as_ref());

    for (keep_below, k) in [(false, -1i64), (true, i64::MAX)] {
        let expected = triangular_expected(&view, &input_bytes, element, keep_below, k, out_len);
        let plan = if keep_below {
            plan_tril_view(DType::F32, out_len, &wide, in_len).unwrap()
        } else {
            plan_triu_view(DType::F32, out_len, &wide, in_len).unwrap()
        };
        assert_eq!(plan.variant, RankVariant::General);
        let symbol = if keep_below {
            "t4a_cuda_tril_view_any"
        } else {
            "t4a_cuda_triu_view_any"
        };
        let dst = written(&session, &vec![SENTINEL; out_len as usize * element]);
        launch_view_any(
            &session,
            symbol,
            &plan,
            &dst,
            &src,
            DType::F32,
            &general,
            k,
            view.count(),
        );
        assert_eq!(read_bytes(&session, &dst), expected, "rank 9 {symbol}");
    }
}

#[test]
fn triangular_views_cover_one_and_a_million_elements() {
    let session = Session::new("diagonal");
    // Both symbols of the mask, at a one-element domain and at a multi-million
    // element one: the physical launch count follows the plan and is one in
    // every case, while the grid is derived from the element count. `triu` gets
    // its own multi-block sizes, so both bodies are proved to run beyond one
    // block.
    for (keep_below, extents, k, expected_grid) in [
        (true, vec![1i64, 1], 0i64, [1u32, 1, 1]),
        (true, vec![1000i64, 1000], 0, [3907, 1, 1]),
        (false, vec![37i64, 41], 2, [6, 1, 1]),
        (false, vec![1000i64, 1000], -1, [3907, 1, 1]),
    ] {
        let view = View {
            extents: extents.clone(),
            strides: compact_strides(&extents),
            offset: 0,
        };
        let element = DType::F32.size();
        let count = view.count();
        let input_bytes = payload(count as usize * element);
        let src = written(&session, &input_bytes);
        let input = Layout::new(&view.extents, &view.strides, view.offset).unwrap();
        let out_len = count + SLACK;
        let expected = triangular_expected(&view, &input_bytes, element, keep_below, k, out_len);
        let dst = written(&session, &vec![SENTINEL; out_len as usize * element]);
        let (symbol, plan) = if keep_below {
            (
                "t4a_cuda_tril_view",
                plan_tril_view(DType::F32, out_len, &input, count).unwrap(),
            )
        } else {
            (
                "t4a_cuda_triu_view",
                plan_triu_view(DType::F32, out_len, &input, count).unwrap(),
            )
        };
        assert_eq!(plan.grid, expected_grid);
        let before = launches();
        launch_view(
            &session,
            symbol,
            &plan,
            &dst,
            &src,
            DType::F32,
            &input,
            k,
            count,
        );
        assert_eq!(launches(), before + 1);
        assert_eq!(read_bytes(&session, &dst), expected, "{extents:?}");
    }

    // An empty input writes nothing and performs no launch.
    let empty = Layout::compact(&[0, 5]).unwrap();
    assert!(plan_tril_view(DType::F32, 0, &empty, 0).unwrap().is_empty());
}

// --- special values ---------------------------------------------------------

/// The f32 bit patterns the copy contract must preserve exactly: both zeros,
/// both infinities, a subnormal and its largest neighbour, and a quiet NaN whose
/// payload is not the canonical one.
fn special_f32() -> [u32; 7] {
    [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x0000_0001, // smallest subnormal
        0x007f_ffff, // largest subnormal
        0x7fc1_2345, // NaN, payload 0x412345 (not the default quiet NaN)
    ]
}

fn special_f64() -> [u64; 7] {
    [
        0x0000_0000_0000_0000, // +0
        0x8000_0000_0000_0000, // -0
        0x7ff0_0000_0000_0000, // +inf
        0xfff0_0000_0000_0000, // -inf
        0x0000_0000_0000_0001, // smallest subnormal
        0x000f_ffff_ffff_ffff, // largest subnormal
        0x7ff8_0123_4567_89ab, // NaN, payload 0x80123456789ab
    ]
}

/// Element-sized byte images cycling the dtype's special values. Complex
/// elements pair consecutive component patterns, so both components carry
/// special values; the integer and Bool routes get their extremes.
fn special_images(dtype: DType) -> Vec<Vec<u8>> {
    match dtype {
        DType::F32 => special_f32()
            .iter()
            .map(|v| v.to_ne_bytes().to_vec())
            .collect(),
        DType::F64 => special_f64()
            .iter()
            .map(|v| v.to_ne_bytes().to_vec())
            .collect(),
        DType::C32 => {
            let patterns = special_f32();
            (0..patterns.len())
                .map(|index| {
                    let mut bytes = patterns[index].to_ne_bytes().to_vec();
                    bytes.extend_from_slice(&patterns[(index + 1) % patterns.len()].to_ne_bytes());
                    bytes
                })
                .collect()
        }
        DType::C64 => {
            let patterns = special_f64();
            (0..patterns.len())
                .map(|index| {
                    let mut bytes = patterns[index].to_ne_bytes().to_vec();
                    bytes.extend_from_slice(&patterns[(index + 1) % patterns.len()].to_ne_bytes());
                    bytes
                })
                .collect()
        }
        DType::I32 => vec![
            i32::MIN.to_ne_bytes().to_vec(),
            i32::MAX.to_ne_bytes().to_vec(),
            0i32.to_ne_bytes().to_vec(),
            (-1i32).to_ne_bytes().to_vec(),
        ],
        DType::I64 => vec![
            i64::MIN.to_ne_bytes().to_vec(),
            i64::MAX.to_ne_bytes().to_vec(),
            0i64.to_ne_bytes().to_vec(),
            (-1i64).to_ne_bytes().to_vec(),
        ],
        DType::Bool => vec![vec![0u8], vec![1u8], vec![0xffu8]],
    }
}

#[test]
fn copies_preserve_special_values_bitwise() {
    let session = Session::with_modules(&["diagonal", "window"]);
    for dtype in DType::ALL {
        let element = dtype.size();
        let images = special_images(dtype);
        let extents = vec![2i64, 3];
        let count = product(&extents);
        let input_bytes: Vec<u8> = (0..count)
            .flat_map(|flat| images[flat as usize % images.len()].clone())
            .collect();
        assert_eq!(input_bytes.len(), count as usize * element);
        let src = written(&session, &input_bytes);
        let layout = Layout::compact(&extents).unwrap();
        let geometry = Geometry::upload(&session, layout.as_ref());
        let general = geometry.mirror(layout.as_ref());
        let out_len = count + SLACK;

        // `k = i64::MAX` keeps every element of tril, `k = i64::MIN` every
        // element of triu, so the whole output must be the input image bit for
        // bit: no NaN canonicalization, no signed-zero loss, no flush to zero.
        for (keep_below, k) in [(true, i64::MAX), (false, i64::MIN)] {
            let expected = {
                let mut bytes = input_bytes.clone();
                bytes.extend(std::iter::repeat_n(SENTINEL, SLACK as usize * element));
                bytes
            };
            let plan = if keep_below {
                plan_tril_view(dtype, out_len, &layout, count).unwrap()
            } else {
                plan_triu_view(dtype, out_len, &layout, count).unwrap()
            };
            for general_path in [false, true] {
                let dst = written(&session, &vec![SENTINEL; out_len as usize * element]);
                let symbol = match (keep_below, general_path) {
                    (true, false) => "t4a_cuda_tril_view",
                    (true, true) => "t4a_cuda_tril_view_any",
                    (false, false) => "t4a_cuda_triu_view",
                    (false, true) => "t4a_cuda_triu_view_any",
                };
                if general_path {
                    launch_view_any(
                        &session, symbol, &plan, &dst, &src, dtype, &general, k, count,
                    );
                } else {
                    launch_view(
                        &session, symbol, &plan, &dst, &src, dtype, &layout, k, count,
                    );
                }
                assert_eq!(
                    read_bytes(&session, &dst),
                    expected,
                    "{dtype:?} {symbol} kept every element"
                );
            }
        }

        // A masked element is exactly zero bytes even where the input there is
        // `-0.0` or a NaN: tril with `k = -1` keeps only `col - row <= -1`.
        let plan = plan_tril_view(dtype, out_len, &layout, count).unwrap();
        let dst = written(&session, &vec![SENTINEL; out_len as usize * element]);
        launch_view(
            &session,
            "t4a_cuda_tril_view",
            &plan,
            &dst,
            &src,
            dtype,
            &layout,
            -1,
            count,
        );
        let got = read_bytes(&session, &dst);
        for flat in 0..count {
            let row = flat % 2;
            let col = (flat / 2) % 3;
            let start = flat as usize * element;
            if col - row <= -1 {
                assert_eq!(
                    &got[start..start + element],
                    &input_bytes[start..start + element],
                    "{dtype:?} kept {flat}"
                );
            } else {
                assert!(
                    got[start..start + element].iter().all(|&byte| byte == 0),
                    "{dtype:?} masked {flat} is not zero"
                );
            }
        }
        assert!(
            got[(count as usize) * element..]
                .iter()
                .all(|&byte| byte == SENTINEL),
            "{dtype:?} guard"
        );

        // The window write copies the same special values: a full-size window at
        // the origin replaces the whole payload bit for bit.
        let update_layout = Layout::compact(&extents).unwrap();
        let update = written(&session, &input_bytes);
        let starts = upload(&session, &[0i32, 0]);
        let operand = {
            let mut bytes = payload_seeded(0xDEAD_BEEF, count as usize * element);
            bytes.extend(std::iter::repeat_n(SENTINEL, SLACK as usize * element));
            bytes
        };
        let dst = written(&session, &operand);
        let plan = plan_dynamic_update_slice(
            dtype,
            DType::I32,
            &layout,
            out_len,
            &update_layout,
            count,
            2,
        )
        .unwrap();
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&update),
            Slot::ptr(&starts),
            Slot::i32(dtype.tag()),
            Slot::i32(DType::I32.tag()),
            Slot::layout(&CLayout::new(&layout)),
            Slot::layout(&CLayout::new(&update_layout)),
            Slot::i64(count),
        ];
        launch(
            session.function("t4a_cuda_dynamic_update_slice"),
            &plan,
            session.stream(),
            &mut slots,
        );
        let expected = {
            let mut bytes = input_bytes.clone();
            bytes.extend(std::iter::repeat_n(SENTINEL, SLACK as usize * element));
            bytes
        };
        assert_eq!(
            read_bytes(&session, &dst),
            expected,
            "{dtype:?} window copy"
        );
        // The fixture is only meaningful if it actually differs from the operand.
        assert_ne!(expected, operand);
    }
}

// --- dynamic_update_slice ---------------------------------------------------

/// The `dynamic_slice` clamp, as the kernel applies it.
fn clamp_start(start: i64, dim: i64, window: i64) -> i64 {
    let max_start = if dim > window { dim - window } else { 0 };
    if start <= 0 {
        0
    } else {
        start.min(max_start)
    }
}

/// Independent model of the window write: start from the uploaded destination
/// (the caller's operand copy, sentinel slab included) and copy each update
/// element's bytes to its windowed address.
#[allow(clippy::too_many_arguments)]
fn dynamic_update_slice_expected(
    out_extents: &[i64],
    out_bytes: &[u8],
    update_extents: &[i64],
    update_strides: &[i64],
    update_offset: i64,
    update_bytes: &[u8],
    starts: &[i64],
    element: usize,
) -> Vec<u8> {
    let window: Vec<i64> = out_extents
        .iter()
        .zip(update_extents.iter())
        .zip(starts.iter())
        .map(|((&dim, &extent), &start)| clamp_start(start, dim, extent))
        .collect();
    let mut out = out_bytes.to_vec();
    for flat in 0..product(update_extents) {
        let mut rest = flat;
        let mut source = update_offset;
        let mut destination = 0;
        let mut stride = 1;
        for axis in 0..update_extents.len() {
            let coordinate = rest % update_extents[axis];
            rest /= update_extents[axis];
            source += coordinate * update_strides[axis];
            destination += (window[axis] + coordinate) * stride;
            stride *= out_extents[axis];
        }
        let src = source as usize * element;
        let dst = destination as usize * element;
        out[dst..dst + element].copy_from_slice(&update_bytes[src..src + element]);
    }
    out
}

/// Index values that the validator accepts and that stress the clamp: integer
/// extremes for the integer routes and the largest exactly representable integer
/// for the float routes (the validator's `max_exact_integer` domain).
fn extreme_starts(index_dtype: DType) -> Vec<i64> {
    match index_dtype {
        DType::I32 => vec![i32::MIN as i64, i32::MAX as i64],
        DType::I64 => vec![i64::MIN, i64::MAX],
        DType::F32 => vec![-(1i64 << 24), 1i64 << 24],
        DType::F64 => vec![-(1i64 << 53), 1i64 << 53],
        other => panic!("not an index dtype: {other:?}"),
    }
}

fn starts_upload(session: &Session, index_dtype: DType, starts: &[i64]) -> Buf {
    match index_dtype {
        DType::I64 => upload(session, starts),
        DType::I32 => {
            let narrowed: Vec<i32> = starts.iter().map(|&value| value as i32).collect();
            upload(session, &narrowed)
        }
        DType::F32 => {
            let narrowed: Vec<f32> = starts.iter().map(|&value| value as f32).collect();
            upload(session, &narrowed)
        }
        DType::F64 => {
            let widened: Vec<f64> = starts.iter().map(|&value| value as f64).collect();
            upload(session, &widened)
        }
        other => panic!("not an index dtype: {other:?}"),
    }
}

/// The three start vectors of a rank: ordinary, both-sides-clamped and the
/// route's extremes.
fn starts_cases(index_dtype: DType, rank: usize) -> Vec<Vec<i64>> {
    let candidates = vec![vec![2i64, 1], vec![-5i64, 100], extreme_starts(index_dtype)];
    candidates
        .into_iter()
        .map(|values| values[..rank].to_vec())
        .collect()
}

#[derive(Clone, Debug)]
struct UpdateCase {
    out_extents: Vec<i64>,
    update_extents: Vec<i64>,
    update_strides: Vec<i64>,
    update_offset: i64,
}

fn update_cases() -> Vec<UpdateCase> {
    vec![
        // Interior window.
        UpdateCase {
            out_extents: vec![6, 5],
            update_extents: vec![2, 3],
            update_strides: vec![1, 2],
            update_offset: 0,
        },
        // Negative-stride update, window clamped high on both axes.
        UpdateCase {
            out_extents: vec![6, 5],
            update_extents: vec![3, 2],
            update_strides: vec![1, -3],
            update_offset: 3,
        },
        // The window fills the operand extent on one axis.
        UpdateCase {
            out_extents: vec![7, 7],
            update_extents: vec![4, 4],
            update_strides: vec![1, 4],
            update_offset: 1,
        },
        // Odd extents and a rank-2 window with a unit axis.
        UpdateCase {
            out_extents: vec![5, 3],
            update_extents: vec![2, 1],
            update_strides: vec![1, 1],
            update_offset: 0,
        },
        // Length one on both sides.
        UpdateCase {
            out_extents: vec![4, 4],
            update_extents: vec![1, 1],
            update_strides: vec![1, 1],
            update_offset: 0,
        },
        // Rank 1 and an odd length.
        UpdateCase {
            out_extents: vec![9],
            update_extents: vec![3],
            update_strides: vec![1],
            update_offset: 2,
        },
    ]
}

/// One window-write launch of either variant. The destination address and the
/// geometry differ, so the caller builds the slots.
#[allow(clippy::too_many_arguments)]
fn dynamic_update_slice_slots(
    dst: &Buf,
    update: &Buf,
    starts: &Buf,
    dtype: DType,
    index_dtype: DType,
    out_layout: &Layout,
    update_layout: &Layout,
    out_any: &CLayoutAny,
    update_any: &CLayoutAny,
    update_count: i64,
    general_path: bool,
) -> [Slot; 8] {
    if general_path {
        [
            Slot::ptr(dst),
            Slot::ptr(update),
            Slot::ptr(starts),
            Slot::i32(dtype.tag()),
            Slot::i32(index_dtype.tag()),
            Slot::layout_any(out_any),
            Slot::layout_any(update_any),
            Slot::i64(update_count),
        ]
    } else {
        [
            Slot::ptr(dst),
            Slot::ptr(update),
            Slot::ptr(starts),
            Slot::i32(dtype.tag()),
            Slot::i32(index_dtype.tag()),
            Slot::layout(&CLayout::new(out_layout)),
            Slot::layout(&CLayout::new(update_layout)),
            Slot::i64(update_count),
        ]
    }
}

#[test]
fn dynamic_update_slice_matches_a_cpu_reference_for_every_dtype_and_index_dtype() {
    let session = Session::new("window");
    for dtype in DType::ALL {
        let element = dtype.size();
        for case in update_cases() {
            let rank = case.out_extents.len();
            let out_count = product(&case.out_extents);
            let out_len = out_count + SLACK;
            let operand_bytes = guarded(0x5EED, out_count, element);
            let update_capacity = capacity(
                &case.update_extents,
                &case.update_strides,
                case.update_offset,
            );
            let update_count = product(&case.update_extents);
            let update_len = update_capacity + SLACK;
            // A different payload from the operand, so an unwritten destination
            // cannot compare equal.
            let update_bytes =
                payload_seeded(0xC0FFEE ^ element as u64, update_len as usize * element);
            let update_layout = Layout::new(
                &case.update_extents,
                &case.update_strides,
                case.update_offset,
            )
            .unwrap();
            let out_layout = Layout::compact(&case.out_extents).unwrap();
            assert_eq!(update_count, update_layout.element_count().unwrap());
            let out_geometry = Geometry::upload(&session, out_layout.as_ref());
            let update_geometry = Geometry::upload(&session, update_layout.as_ref());
            let out_any = out_geometry.mirror(out_layout.as_ref());
            let update_any = update_geometry.mirror(update_layout.as_ref());

            for index_dtype in [DType::F32, DType::F64, DType::I32, DType::I64] {
                for starts in starts_cases(index_dtype, rank) {
                    let start_buf = starts_upload(&session, index_dtype, &starts);
                    let plan = plan_dynamic_update_slice(
                        dtype,
                        index_dtype,
                        &out_layout,
                        out_len,
                        &update_layout,
                        update_len,
                        rank as u32,
                    )
                    .unwrap();
                    assert_eq!(plan.variant, RankVariant::Fast);
                    let expected = dynamic_update_slice_expected(
                        &case.out_extents,
                        &operand_bytes,
                        &case.update_extents,
                        &case.update_strides,
                        case.update_offset,
                        &update_bytes,
                        &starts,
                        element,
                    );
                    for general_path in [false, true] {
                        let dst = written(&session, &operand_bytes);
                        let update = written(&session, &update_bytes);
                        let symbol = if general_path {
                            "t4a_cuda_dynamic_update_slice_any"
                        } else {
                            "t4a_cuda_dynamic_update_slice"
                        };
                        let mut slots = dynamic_update_slice_slots(
                            &dst,
                            &update,
                            &start_buf,
                            dtype,
                            index_dtype,
                            &out_layout,
                            &update_layout,
                            &out_any,
                            &update_any,
                            update_count,
                            general_path,
                        );
                        let before = launches();
                        launch(
                            session.function(symbol),
                            &plan,
                            session.stream(),
                            &mut slots,
                        );
                        assert_eq!(launches(), before + 1);
                        assert_eq!(
                            read_bytes(&session, &dst),
                            expected,
                            "{dtype:?} {case:?} {index_dtype:?} starts={starts:?} general={general_path}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn empty_and_scalar_updates_plan_the_documented_domains() {
    let session = Session::new("window");
    let out_layout = Layout::compact(&[4, 4]).unwrap();
    let empty = Layout::compact(&[0, 2]).unwrap();
    // An empty update writes nothing: zero launches, and the operand copy is
    // untouched byte for byte (the caller's copy is its own launch).
    let plan =
        plan_dynamic_update_slice(DType::F64, DType::I64, &out_layout, 16, &empty, 0, 2).unwrap();
    assert!(plan.is_empty());
    let operand = guarded(0x0BAD_CAFE, 16, 8);
    let dst = written(&session, &operand);
    let update = written(&session, &payload(8));
    let starts = upload(&session, &[0i64, 0]);
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&update),
        Slot::ptr(&starts),
        Slot::i32(DType::F64.tag()),
        Slot::i32(DType::I64.tag()),
        Slot::layout(&CLayout::new(&out_layout)),
        Slot::layout(&CLayout::new(&empty)),
        Slot::i64(0),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_dynamic_update_slice"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before);
    assert_eq!(read_bytes(&session, &dst), operand);

    // A rank-0 update writes exactly one element and dereferences no start.
    let scalar = Layout::new(&[], &[], 0).unwrap();
    let plan =
        plan_dynamic_update_slice(DType::I64, DType::F64, &scalar, 1, &scalar, 1, 0).unwrap();
    assert_eq!(plan.grid, [1, 1, 1]);
    let operand = guarded(0x1234_5678, 1, 8);
    let dst = written(&session, &operand);
    // The starts buffer is a valid allocation whose single value is never read
    // for a rank-0 window; an empty vector would have no element to upload.
    let starts = upload(&session, &[0.0f64]);
    let update = upload(&session, &[0x0123_4567_89ab_cdefu64]);
    let mut slots = [
        Slot::ptr(&dst),
        Slot::ptr(&update),
        Slot::ptr(&starts),
        Slot::i32(DType::I64.tag()),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&scalar)),
        Slot::layout(&CLayout::new(&scalar)),
        Slot::i64(1),
    ];
    let before = launches();
    launch(
        session.function("t4a_cuda_dynamic_update_slice"),
        &plan,
        session.stream(),
        &mut slots,
    );
    assert_eq!(launches(), before + 1);
    let mut expected = operand;
    expected[..8].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_ne_bytes());
    assert_eq!(read_bytes(&session, &dst), expected);
}

#[test]
fn dynamic_update_slice_agrees_between_variants_up_to_rank_eight() {
    let session = Session::new("window");
    // Ranks 2-8: both variants must produce the reference bytes.
    for rank in 2..=8usize {
        let out_extents: Vec<i64> = (0..rank).map(|axis| 3 + (axis as i64 % 2)).collect();
        let out_layout = Layout::compact(&out_extents).unwrap();
        let out_count = out_layout.element_count().unwrap();
        let out_len = out_count + SLACK;
        let operand_bytes = guarded(0xABCD, out_count, 8);
        // A non-compact update with a non-zero offset and non-unit extents on
        // every axis.
        let update_extents: Vec<i64> = (0..rank).map(|axis| 1 + (axis as i64 % 2)).collect();
        let update_strides =
            permuted_strides(&update_extents, &(0..rank).rev().collect::<Vec<_>>());
        let update_offset = 2;
        let update_layout = Layout::new(&update_extents, &update_strides, update_offset).unwrap();
        let update_count = update_layout.element_count().unwrap();
        let update_len = capacity(&update_extents, &update_strides, update_offset) + SLACK;
        let update_bytes = payload_seeded(0xF00D, update_len as usize * 8);
        let starts = vec![1i64; rank];

        let expected = dynamic_update_slice_expected(
            &out_extents,
            &operand_bytes,
            &update_extents,
            &update_strides,
            update_offset,
            &update_bytes,
            &starts,
            8,
        );
        let plan = plan_dynamic_update_slice(
            DType::F64,
            DType::I64,
            &out_layout,
            out_len,
            &update_layout,
            update_len,
            rank as u32,
        )
        .unwrap();
        assert_eq!(plan.variant, RankVariant::Fast);
        assert!(update_count < out_count);
        let out_geometry = Geometry::upload(&session, out_layout.as_ref());
        let update_geometry = Geometry::upload(&session, update_layout.as_ref());
        let out_any = out_geometry.mirror(out_layout.as_ref());
        let update_any = update_geometry.mirror(update_layout.as_ref());
        let start_buf = upload(&session, &starts);
        for general_path in [false, true] {
            let dst = written(&session, &operand_bytes);
            let update = written(&session, &update_bytes);
            let symbol = if general_path {
                "t4a_cuda_dynamic_update_slice_any"
            } else {
                "t4a_cuda_dynamic_update_slice"
            };
            let mut slots = dynamic_update_slice_slots(
                &dst,
                &update,
                &start_buf,
                DType::F64,
                DType::I64,
                &out_layout,
                &update_layout,
                &out_any,
                &update_any,
                update_count,
                general_path,
            );
            launch(
                session.function(symbol),
                &plan,
                session.stream(),
                &mut slots,
            );
            assert_eq!(read_bytes(&session, &dst), expected, "rank {rank} {symbol}");
        }
    }
}

#[test]
fn dynamic_update_slice_reaches_rank_nine_and_multi_block_windows() {
    let session = Session::new("window");
    // Rank 9 with a non-unit extent on axis 8. The general symbol is the only
    // one that can accept the plan; the operand is 2^8*3 = 768 elements.
    let out_extents = vec![2i64, 2, 2, 2, 2, 2, 2, 2, 3];
    let out_wide = WideLayout::compact(out_extents.clone()).unwrap();
    let out_count = out_wide.element_count().unwrap();
    assert_eq!(out_count, 768);
    let out_len = out_count + SLACK;
    let element = DType::C64.size();
    let out_geometry = Geometry::upload(&session, out_wide.as_ref());
    let out_any = out_geometry.mirror(out_wide.as_ref());

    // (update extents, update strides, update offset, starts, multi-block)
    let cases = vec![
        (
            // Two updates on the ninth axis, at a non-zero offset.
            vec![1i64, 1, 1, 1, 1, 1, 1, 1, 2],
            vec![1i64, 1, 1, 1, 1, 1, 1, 1, 1],
            5i64,
            {
                let mut starts = vec![0i64; 9];
                starts[8] = 100;
                starts
            },
            false,
        ),
        (
            // 384 updates: two blocks, and the window origin is clamped on the
            // axes where the update fills the operand extent.
            vec![1i64, 2, 2, 2, 2, 2, 2, 2, 3],
            compact_strides(&[1, 2, 2, 2, 2, 2, 2, 2, 3]),
            0,
            vec![1i64; 9],
            true,
        ),
        (
            // 19683 updates: the whole operand, 77 blocks.
            vec![2i64, 2, 2, 2, 2, 2, 2, 2, 3],
            compact_strides(&[2, 2, 2, 2, 2, 2, 2, 2, 3]),
            0,
            vec![1i64; 9],
            true,
        ),
    ];

    for (update_extents, update_strides, update_offset, starts, multi_block) in cases {
        let update_wide = WideLayout::new(
            update_extents.clone(),
            update_strides.clone(),
            update_offset,
        )
        .unwrap();
        let update_count = update_wide.element_count().unwrap();
        let update_len = capacity(&update_extents, &update_strides, update_offset) + SLACK;
        let update_bytes =
            payload_seeded(0xBEEF ^ update_count as u64, update_len as usize * element);
        let operand_bytes = guarded(0x2468, out_count, element);
        let expected = dynamic_update_slice_expected(
            &out_extents,
            &operand_bytes,
            &update_extents,
            &update_strides,
            update_offset,
            &update_bytes,
            &starts,
            element,
        );
        let plan = plan_dynamic_update_slice(
            DType::C64,
            DType::F64,
            &out_wide,
            out_len,
            &update_wide,
            update_len,
            9,
        )
        .unwrap();
        assert_eq!(plan.variant, RankVariant::General);
        if multi_block {
            assert!(update_count > 256, "{update_count} updates");
            assert!(plan.grid[0] > 1, "grid {:?}", plan.grid);
        }
        let dst = written(&session, &operand_bytes);
        let update = written(&session, &update_bytes);
        let start_buf = starts_upload(&session, DType::F64, &starts);
        let update_geometry = Geometry::upload(&session, update_wide.as_ref());
        let update_any = update_geometry.mirror(update_wide.as_ref());
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&update),
            Slot::ptr(&start_buf),
            Slot::i32(DType::C64.tag()),
            Slot::i32(DType::F64.tag()),
            Slot::layout_any(&out_any),
            Slot::layout_any(&update_any),
            Slot::i64(update_count),
        ];
        let before = launches();
        launch(
            session.function("t4a_cuda_dynamic_update_slice_any"),
            &plan,
            session.stream(),
            &mut slots,
        );
        assert_eq!(launches(), before + 1);
        assert_eq!(
            read_bytes(&session, &dst),
            expected,
            "rank 9, {update_count} updates, grid {:?}",
            plan.grid
        );
    }
}

#[test]
fn dynamic_update_slice_launches_once_for_one_and_a_million_elements() {
    let session = Session::new("window");
    let out_extents = vec![1000i64, 1000];
    let out_layout = Layout::compact(&out_extents).unwrap();
    let out_count = out_layout.element_count().unwrap();
    let element = DType::F32.size();
    // The operand and the update carry different payloads, so a kernel that
    // writes nothing cannot compare equal to the expected image.
    let operand_bytes = guarded(0x1111_2222, out_count, element);
    let starts = [300i64, 200];

    for update_extents in [vec![1i64, 1], vec![1000i64, 1000]] {
        let update_layout = Layout::compact(&update_extents).unwrap();
        let update_count = update_layout.element_count().unwrap();
        let update_strides = update_layout.strides().to_vec();
        let update_bytes = payload_seeded(0x3333_4444, update_count as usize * element);
        let update = written(&session, &update_bytes);
        let plan = plan_dynamic_update_slice(
            DType::F32,
            DType::F64,
            &out_layout,
            out_count + SLACK,
            &update_layout,
            update_count,
            2,
        )
        .unwrap();
        let expected_grid = [
            u32::try_from((update_count as u64).div_ceil(256)).unwrap(),
            1,
            1,
        ];
        assert_eq!(plan.grid, expected_grid);
        let start_buf = starts_upload(&session, DType::F64, &starts);
        let dst = written(&session, &operand_bytes);
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&update),
            Slot::ptr(&start_buf),
            Slot::i32(DType::F32.tag()),
            Slot::i32(DType::F64.tag()),
            Slot::layout(&CLayout::new(&out_layout)),
            Slot::layout(&CLayout::new(&update_layout)),
            Slot::i64(update_count),
        ];
        let before = launches();
        launch(
            session.function("t4a_cuda_dynamic_update_slice"),
            &plan,
            session.stream(),
            &mut slots,
        );
        assert_eq!(launches(), before + 1);
        let expected = dynamic_update_slice_expected(
            &out_extents,
            &operand_bytes,
            &update_extents,
            &update_strides,
            0,
            &update_bytes,
            &starts,
            element,
        );
        let got = read_bytes(&session, &dst);
        assert_eq!(got, expected, "update {update_extents:?}");
        if update_count > 1 {
            // A write confined to the first block (or to a prefix) cannot pass:
            // the window covers the whole operand and the update payload differs
            // from it in the last block too.
            let last_block = (out_count as usize - 256) * element;
            assert_ne!(
                expected[last_block..(out_count as usize) * element],
                operand_bytes[last_block..(out_count as usize) * element],
                "the full-size window must change the last block"
            );
            assert_eq!(
                got[last_block..(out_count as usize) * element],
                expected[last_block..(out_count as usize) * element]
            );
        }
    }
}
