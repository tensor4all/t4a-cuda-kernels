//! Revision 15: the layout-taking elementwise entries against the flat entries.
//!
//! Every `(op, dtype)` each strided entry accepts is launched on operands read
//! through permuted, padded-and-offset, broadcast (stride zero) and reversed
//! (negative stride) layouts, into a compact or a permuted destination. The same
//! operands are gathered on the host into compact copies (an exact byte copy) and
//! run through the flat entry the strided tag mirrors. The two results must be
//! bit-identical (a NaN matches any NaN: the ABI leaves the NaN sign and payload
//! unspecified), and the status word of a checked integer tag must agree too.
//!
//! Parallelism (review checklist section 2): the domain is 4355 elements (17 blocks,
//! not a multiple of 256) for every case, one case per family runs 103 000
//! elements, every destination starts as a sentinel and is compared over its whole
//! logical domain, and the guard behind it must stay untouched. Rank 9 exercises the
//! `_any` symbols, and the fast and general symbols must agree at rank <= 8.
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_strided_elementwise`;
//! requires a CUDA device.

mod gpu;

use gpu::{launch, launches, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_binary, plan_compare, plan_elementwise, plan_select, plan_strided_binary,
    plan_strided_compare, plan_strided_select, plan_strided_unary, plan_unary, BinaryOp, CLayout,
    CLayoutAny, CompareOp, DType, Entry, Launch, Layout, ScalarSide, StridedBinaryOp, UnaryOp,
    WideLayout, FLAT_BLOCK, MAX_RANK,
};

const SENTINEL: u8 = 0xCD;
/// Guard elements behind every destination, so a write past the domain shows.
const GUARD: usize = 16;
const EXTENTS: [i64; 3] = [67, 13, 5];
const BIG: [i64; 2] = [1000, 103];

// --- layouts ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Kind {
    Compact,
    Permuted,
    Padded,
    Broadcast,
    Reversed,
}

fn compact_strides(extents: &[i64]) -> Vec<i64> {
    let mut stride = 1;
    extents
        .iter()
        .map(|&extent| {
            let current = stride;
            stride *= extent;
            current
        })
        .collect()
}

/// A layout of `extents` and the allocation length (elements) it needs.
fn layout(kind: Kind, extents: &[i64]) -> (WideLayout, usize) {
    let rank = extents.len();
    let mut base = 0i64;
    let strides: Vec<i64> = match kind {
        Kind::Compact => compact_strides(extents),
        Kind::Permuted => {
            let mut strides = vec![0; rank];
            let mut stride = 1;
            for axis in (0..rank).rev() {
                strides[axis] = stride;
                stride *= extents[axis];
            }
            strides
        }
        Kind::Padded => {
            base = 7;
            let mut padded = extents.to_vec();
            padded[0] += 3;
            compact_strides(&padded)
        }
        Kind::Broadcast => {
            let axis = if rank > 1 { 1 } else { 0 };
            let mut kept = extents.to_vec();
            kept[axis] = 1;
            let mut strides = compact_strides(&kept);
            strides[axis] = 0;
            strides
        }
        Kind::Reversed => {
            base = 2;
            let mut strides = compact_strides(extents);
            strides[0] = -strides[0];
            strides
        }
    };
    let low: i64 = strides
        .iter()
        .zip(extents)
        .map(|(&stride, &extent)| stride.min(0) * (extent - 1))
        .sum();
    let high: i64 = strides
        .iter()
        .zip(extents)
        .map(|(&stride, &extent)| stride.max(0) * (extent - 1))
        .sum();
    let offset = base - low;
    let len = (offset + high + 1 + 3) as usize;
    (
        WideLayout::new(extents.to_vec(), strides, offset).unwrap(),
        len,
    )
}

/// Element indices a layout addresses, in flat logical order (axis 0 fastest).
fn indices(layout: &WideLayout) -> Vec<usize> {
    let extents = layout.extents();
    let count: i64 = extents.iter().product();
    (0..count)
        .map(|flat| {
            let mut flat = flat;
            let mut index = layout.offset();
            for (axis, &extent) in extents.iter().enumerate() {
                index += (flat % extent) * layout.strides()[axis];
                flat /= extent;
            }
            index as usize
        })
        .collect()
}

fn gather(bytes: &[u8], at: &[usize], size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(at.len() * size);
    for &index in at {
        out.extend_from_slice(&bytes[index * size..(index + 1) * size]);
    }
    out
}

/// The by-value and the device-resident form of one layout; owns the geometry.
struct Geometry {
    fast: CLayout,
    general: CLayoutAny,
    _buffers: [Buf; 2],
}

fn geometry(session: &Session, layout: &WideLayout) -> Geometry {
    let fast = if layout.extents().len() <= MAX_RANK {
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
        _buffers: [extents, strides],
    }
}

fn slot(geometry: &Geometry, general: bool) -> Slot {
    if general {
        Slot::layout_any(&geometry.general)
    } else {
        Slot::layout(&geometry.fast)
    }
}

// --- data -------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn unit(&mut self) -> f64 {
        (self.next() % 1_000_003) as f64 / 1_000_003.0
    }
}

/// `count` elements of `dtype`: mostly ordinary values, with zeros, signed zeros,
/// infinities and NaN (floats) or zero, -1 and the minimum (integers) mixed in.
fn values(dtype: DType, count: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let real = |rng: &mut Rng| -> f64 {
        match rng.next() % 23 {
            0 => 0.0,
            1 => -0.0,
            2 => f64::INFINITY,
            3 => f64::NAN,
            4 => 1.0e300,
            _ => (rng.unit() - 0.5) * 8.0,
        }
    };
    let mut out = Vec::with_capacity(count * dtype.size());
    for _ in 0..count {
        match dtype {
            DType::F32 => out.extend_from_slice(&(real(&mut rng) as f32).to_ne_bytes()),
            DType::F64 => out.extend_from_slice(&real(&mut rng).to_ne_bytes()),
            DType::C32 => {
                out.extend_from_slice(&(real(&mut rng) as f32).to_ne_bytes());
                out.extend_from_slice(&(real(&mut rng) as f32).to_ne_bytes());
            }
            DType::C64 => {
                out.extend_from_slice(&real(&mut rng).to_ne_bytes());
                out.extend_from_slice(&real(&mut rng).to_ne_bytes());
            }
            DType::I32 | DType::I64 => {
                let value: i64 = match rng.next() % 19 {
                    0 => 0,
                    1 => -1,
                    2 => i64::MIN,
                    _ => (rng.next() % 41) as i64 - 20,
                };
                if dtype == DType::I32 {
                    let value = if value == i64::MIN {
                        i32::MIN
                    } else {
                        value as i32
                    };
                    out.extend_from_slice(&value.to_ne_bytes());
                } else {
                    out.extend_from_slice(&value.to_ne_bytes());
                }
            }
            DType::Bool => out.push((rng.next() % 2) as u8),
        }
    }
    out
}

/// One operand: its layout and its device buffer.
struct Operand {
    layout: WideLayout,
    len: usize,
    host: Vec<u8>,
    dtype: DType,
    buffer: Buf,
}

fn operand(session: &Session, kind: Kind, extents: &[i64], dtype: DType, seed: u64) -> Operand {
    let (layout, len) = layout(kind, extents);
    let host = values(dtype, len, seed);
    let buffer = written(session, &host);
    Operand {
        layout,
        len,
        host,
        dtype,
        buffer,
    }
}

impl Operand {
    /// The compact copy the flat entry reads: an exact byte gather.
    fn compact(&self, session: &Session) -> Buf {
        written(
            session,
            &gather(&self.host, &indices(&self.layout), self.dtype.size()),
        )
    }
}

/// A destination of `kind` prefilled with the sentinel, with a guard behind it.
struct Destination {
    layout: WideLayout,
    len: usize,
    dtype: DType,
    buffer: Buf,
}

fn destination(session: &Session, kind: Kind, extents: &[i64], dtype: DType) -> Destination {
    let (layout, len) = layout(kind, extents);
    let buffer = written(session, &vec![SENTINEL; (len + GUARD) * dtype.size()]);
    Destination {
        layout,
        len,
        dtype,
        buffer,
    }
}

impl Destination {
    /// The logical result, after checking that nothing outside the domain moved.
    fn logical(&self, session: &Session) -> Vec<u8> {
        let bytes = read_bytes(session, &self.buffer);
        let size = self.dtype.size();
        let at = indices(&self.layout);
        let mut reached = vec![false; self.len + GUARD];
        for &index in &at {
            reached[index] = true;
        }
        for (index, hit) in reached.iter().enumerate() {
            if !hit {
                assert!(
                    bytes[index * size..(index + 1) * size]
                        .iter()
                        .all(|&byte| byte == SENTINEL),
                    "element {index} outside the destination domain was written"
                );
            }
        }
        gather(&bytes, &at, size)
    }
}

fn status(session: &Session) -> Buf {
    written(session, &[0u8; 4])
}

fn count(extents: &[i64]) -> i64 {
    extents.iter().product()
}

fn check_plan(plan: &Launch, len: i64) {
    let blocks = (len as u64).div_ceil(u64::from(FLAT_BLOCK));
    assert_eq!(plan.grid, [blocks as u32, 1, 1], "grid covers the domain");
    assert_eq!(plan.block, [FLAT_BLOCK, 1, 1]);
}

/// Whether two elements of `dtype` agree: identical bits, except that a NaN
/// component matches any NaN component. The NaN sign and payload are unspecified by
/// the ABI: the GPU's FMUL/FFMA/DMUL/DFMA propagate one input NaN chosen by operand
/// position, and the compiler may commute a product's operands differently in two
/// bodies (the revision 14 golden documents the same class).
fn same_element(strided: &[u8], flat: &[u8], dtype: DType) -> bool {
    if strided == flat {
        return true;
    }
    let component_nan = |bytes: &[u8]| -> Vec<bool> {
        match dtype {
            DType::F32 | DType::C32 => bytes
                .chunks(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()).is_nan())
                .collect(),
            DType::F64 | DType::C64 => bytes
                .chunks(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()).is_nan())
                .collect(),
            _ => vec![false],
        }
    };
    let width = match dtype {
        DType::F32 | DType::C32 => 4,
        DType::F64 | DType::C64 => 8,
        _ => return false,
    };
    let (a, b) = (component_nan(strided), component_nan(flat));
    strided
        .chunks(width)
        .zip(flat.chunks(width))
        .zip(a.iter().zip(&b))
        .all(|((x, y), (&nan_x, &nan_y))| x == y || (nan_x && nan_y))
}

/// Equality of two logical results under [`same_element`], reporting the first
/// differing element.
fn assert_same(strided: &[u8], flat: &[u8], dtype: DType, context: &str) {
    let size = dtype.size();
    assert_eq!(strided.len(), flat.len(), "{context}");
    let differing: Vec<usize> = (0..flat.len() / size)
        .filter(|&element| {
            let range = element * size..(element + 1) * size;
            !same_element(&strided[range.clone()], &flat[range], dtype)
        })
        .collect();
    if let Some(&first) = differing.first() {
        panic!(
            "{context}: {} of {} elements differ; first at {first}: strided {:02x?} flat {:02x?}",
            differing.len(),
            flat.len() / size,
            &strided[first * size..(first + 1) * size],
            &flat[first * size..(first + 1) * size],
        );
    }
}

// --- unary ------------------------------------------------------------------------

const UNARY: [UnaryOp; 14] = [
    UnaryOp::Neg,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tanh,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Expm1,
    UnaryOp::Log1p,
    UnaryOp::Erf,
    UnaryOp::Abs,
    UnaryOp::Sign,
    UnaryOp::Conj,
];

const NUMERIC: [DType; 6] = [
    DType::F32,
    DType::F64,
    DType::I32,
    DType::I64,
    DType::C32,
    DType::C64,
];

fn unary_pairs() -> Vec<(UnaryOp, DType, DType)> {
    let mut pairs = Vec::new();
    for op in UNARY {
        for dst in NUMERIC {
            for src in NUMERIC {
                if op.accepts(dst, src) {
                    pairs.push((op, dst, src));
                }
            }
        }
    }
    pairs
}

fn run_unary(
    session: &Session,
    (op, dst, src): (UnaryOp, DType, DType),
    extents: &[i64],
    in_kind: Kind,
    out_kind: Kind,
    general: bool,
) {
    let len = count(extents);
    let input = operand(session, in_kind, extents, src, 11 + op.tag() as u64);

    let flat_out = destination(session, Kind::Compact, extents, dst);
    let compact = input.compact(session);
    let mut slots = [
        Slot::ptr(&flat_out.buffer),
        Slot::ptr(&compact),
        Slot::i32(dst.tag()),
        Slot::i32(src.tag()),
        Slot::i32(op.tag()),
        Slot::i64(len),
    ];
    launch(
        session.function("t4a_cuda_unary"),
        &plan_unary(op, dst, src, len).unwrap(),
        session.stream(),
        &mut slots,
    );

    let out = destination(session, out_kind, extents, dst);
    let out_geometry = geometry(session, &out.layout);
    let in_geometry = geometry(session, &input.layout);
    let plan = plan_strided_unary(
        op,
        dst,
        src,
        &out.layout,
        out.len as i64,
        &input.layout,
        input.len as i64,
    )
    .unwrap();
    check_plan(&plan, len);
    let mut slots = [
        Slot::ptr(&out.buffer),
        Slot::ptr(&input.buffer),
        slot(&out_geometry, general),
        slot(&in_geometry, general),
        Slot::i32(dst.tag()),
        Slot::i32(src.tag()),
        Slot::i32(op.tag()),
        Slot::i64(len),
    ];
    let name = if general {
        "t4a_cuda_strided_unary_any"
    } else {
        "t4a_cuda_strided_unary"
    };
    launch(session.function(name), &plan, session.stream(), &mut slots);
    assert_same(
        &out.logical(session),
        &flat_out.logical(session),
        dst,
        &format!(
            "{name} {op:?} {dst:?}<-{src:?} in {in_kind:?} out {out_kind:?} extents {extents:?}"
        ),
    );
}

#[test]
fn strided_unary_is_bit_identical_to_the_flat_entry() {
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    let kinds = [
        (Kind::Permuted, Kind::Compact),
        (Kind::Padded, Kind::Permuted),
        (Kind::Broadcast, Kind::Compact),
        (Kind::Reversed, Kind::Permuted),
        (Kind::Compact, Kind::Compact),
    ];
    for pair in unary_pairs() {
        for &(in_kind, out_kind) in &kinds {
            run_unary(&session, pair, &EXTENTS, in_kind, out_kind, false);
        }
    }
    let before = launches();
    run_unary(
        &session,
        (UnaryOp::Exp, DType::F64, DType::F64),
        &BIG,
        Kind::Permuted,
        Kind::Compact,
        false,
    );
    assert_eq!(launches() - before, 2, "one flat and one strided launch");
}

// --- binary -----------------------------------------------------------------------

const STRIDED_BINARY: [StridedBinaryOp; 11] = [
    StridedBinaryOp::Add,
    StridedBinaryOp::Sub,
    StridedBinaryOp::Mul,
    StridedBinaryOp::Div,
    StridedBinaryOp::Max,
    StridedBinaryOp::Min,
    StridedBinaryOp::Pow,
    StridedBinaryOp::Rem,
    StridedBinaryOp::DivChecked,
    StridedBinaryOp::RemChecked,
    StridedBinaryOp::PowChecked,
];

/// The flat launch a strided binary tag mirrors.
enum Flat {
    Arithmetic(&'static str, Entry),
    Binary(BinaryOp),
}

fn flat_of(op: StridedBinaryOp, dtype: DType) -> Flat {
    let complex = matches!(dtype, DType::C32 | DType::C64);
    match op {
        StridedBinaryOp::Add => Flat::Arithmetic("t4a_cuda_add", Entry::Add),
        StridedBinaryOp::Sub => Flat::Arithmetic("t4a_cuda_sub", Entry::Sub),
        StridedBinaryOp::Mul => Flat::Arithmetic("t4a_cuda_mul", Entry::Mul),
        StridedBinaryOp::Div if complex => Flat::Binary(BinaryOp::DivComplex),
        StridedBinaryOp::Div => Flat::Arithmetic("t4a_cuda_div", Entry::Div),
        StridedBinaryOp::Max => Flat::Binary(BinaryOp::Max),
        StridedBinaryOp::Min => Flat::Binary(BinaryOp::Min),
        StridedBinaryOp::Pow => Flat::Binary(BinaryOp::Pow),
        StridedBinaryOp::Rem => Flat::Binary(BinaryOp::Rem),
        StridedBinaryOp::DivChecked => Flat::Binary(BinaryOp::DivChecked),
        StridedBinaryOp::RemChecked => Flat::Binary(BinaryOp::RemChecked),
        StridedBinaryOp::PowChecked => Flat::Binary(BinaryOp::PowChecked),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_binary(
    session: &Session,
    op: StridedBinaryOp,
    dtype: DType,
    extents: &[i64],
    lhs_kind: Kind,
    rhs_kind: Kind,
    out_kind: Kind,
    general: bool,
) -> Vec<u8> {
    let len = count(extents);
    let seed = 100 + op.tag() as u64 * 7;
    let lhs = operand(session, lhs_kind, extents, dtype, seed);
    let rhs = operand(session, rhs_kind, extents, dtype, seed + 3);

    let flat_out = destination(session, Kind::Compact, extents, dtype);
    let flat_status = status(session);
    let (lhs_compact, rhs_compact) = (lhs.compact(session), rhs.compact(session));
    match flat_of(op, dtype) {
        Flat::Arithmetic(name, entry) => {
            let mut slots = [
                Slot::ptr(&flat_out.buffer),
                Slot::ptr(&lhs_compact),
                Slot::ptr(&rhs_compact),
                Slot::i32(dtype.tag()),
                Slot::i64(len),
            ];
            launch(
                session.function(name),
                &plan_elementwise(entry, dtype, len).unwrap(),
                session.stream(),
                &mut slots,
            );
        }
        Flat::Binary(binary) => {
            let mut slots = [
                Slot::ptr(&flat_out.buffer),
                Slot::ptr(&lhs_compact),
                Slot::ptr(&rhs_compact),
                Slot::i32(dtype.tag()),
                Slot::i32(binary.tag()),
                Slot::i32(ScalarSide::Both.tag()),
                Slot::i64(len),
                Slot::ptr(&flat_status),
            ];
            launch(
                session.function("t4a_cuda_binary"),
                &plan_binary(binary, dtype, ScalarSide::Both, len).unwrap(),
                session.stream(),
                &mut slots,
            );
        }
    }

    let out = destination(session, out_kind, extents, dtype);
    let strided_status = status(session);
    let geometries = [
        geometry(session, &out.layout),
        geometry(session, &lhs.layout),
        geometry(session, &rhs.layout),
    ];
    let plan = plan_strided_binary(
        op,
        dtype,
        &out.layout,
        out.len as i64,
        &lhs.layout,
        lhs.len as i64,
        &rhs.layout,
        rhs.len as i64,
    )
    .unwrap();
    check_plan(&plan, len);
    // Only the checked tags may touch the status word; the others get null.
    let err = if op.uses_status() {
        Slot::ptr(&strided_status)
    } else {
        Slot::i64(0)
    };
    let mut slots = [
        Slot::ptr(&out.buffer),
        Slot::ptr(&lhs.buffer),
        Slot::ptr(&rhs.buffer),
        slot(&geometries[0], general),
        slot(&geometries[1], general),
        slot(&geometries[2], general),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        Slot::i64(len),
        err,
    ];
    let name = if general {
        "t4a_cuda_strided_binary_any"
    } else {
        "t4a_cuda_strided_binary"
    };
    launch(session.function(name), &plan, session.stream(), &mut slots);
    let context = format!(
        "{name} {op:?} {dtype:?} lhs {lhs_kind:?} rhs {rhs_kind:?} out {out_kind:?} extents {extents:?}"
    );
    assert_same(
        &out.logical(session),
        &flat_out.logical(session),
        dtype,
        &context,
    );
    if op.uses_status() {
        assert_eq!(
            read_bytes(session, &strided_status),
            read_bytes(session, &flat_status),
            "status word: {context}"
        );
    }
    read_bytes(session, &strided_status)
}

const BINARY_KINDS: [(Kind, Kind, Kind); 5] = [
    (Kind::Compact, Kind::Compact, Kind::Compact),
    (Kind::Permuted, Kind::Broadcast, Kind::Compact),
    (Kind::Padded, Kind::Reversed, Kind::Permuted),
    (Kind::Broadcast, Kind::Permuted, Kind::Permuted),
    (Kind::Reversed, Kind::Padded, Kind::Compact),
];

#[test]
fn strided_binary_is_bit_identical_to_the_flat_entry() {
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    let mut covered = 0;
    for op in STRIDED_BINARY {
        for dtype in NUMERIC {
            if !op.accepts(dtype) {
                continue;
            }
            covered += 1;
            for &(lhs, rhs, out) in &BINARY_KINDS {
                let _ = run_binary(&session, op, dtype, &EXTENTS, lhs, rhs, out, false);
            }
        }
    }
    // 3 ops x 6 dtypes + Div x 4 + Max/Min x 4 + Pow/Rem x 2 + checked x 2.
    assert_eq!(covered, 18 + 4 + 8 + 4 + 6);
    let _ = run_binary(
        &session,
        StridedBinaryOp::Add,
        DType::F64,
        &BIG,
        Kind::Permuted,
        Kind::Broadcast,
        Kind::Compact,
        false,
    );
}

#[test]
fn checked_integer_division_reports_a_zero_divisor() {
    // The value draw puts zeros among the divisors, so the flat word is set and the
    // strided word must be set too (run_binary compares them).
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    let rhs = values(DType::I32, count(&EXTENTS) as usize, 100 + 8 * 7 + 3);
    assert!(rhs.chunks(4).any(|bytes| bytes == [0, 0, 0, 0]));
    let word = run_binary(
        &session,
        StridedBinaryOp::DivChecked,
        DType::I32,
        &EXTENTS,
        Kind::Compact,
        Kind::Compact,
        Kind::Compact,
        false,
    );
    assert_ne!(word, [0, 0, 0, 0], "a zero divisor sets the status word");
}

// --- compare and select -------------------------------------------------------------

const ORDERED: [DType; 4] = [DType::F32, DType::F64, DType::I32, DType::I64];
const COMPARE: [CompareOp; 5] = [
    CompareOp::Eq,
    CompareOp::Lt,
    CompareOp::Le,
    CompareOp::Gt,
    CompareOp::Ge,
];

#[allow(clippy::too_many_arguments)]
fn run_compare(
    session: &Session,
    cmp: CompareOp,
    dtype: DType,
    extents: &[i64],
    lhs_kind: Kind,
    rhs_kind: Kind,
    out_kind: Kind,
    general: bool,
) {
    let len = count(extents);
    let seed = 300 + cmp.tag() as u64;
    let lhs = operand(session, lhs_kind, extents, dtype, seed);
    let rhs = operand(session, rhs_kind, extents, dtype, seed + 1);

    let flat_out = destination(session, Kind::Compact, extents, DType::Bool);
    let (lhs_compact, rhs_compact) = (lhs.compact(session), rhs.compact(session));
    let mut slots = [
        Slot::ptr(&flat_out.buffer),
        Slot::ptr(&lhs_compact),
        Slot::ptr(&rhs_compact),
        Slot::i32(dtype.tag()),
        Slot::i32(cmp.tag()),
        Slot::i64(len),
    ];
    launch(
        session.function("t4a_cuda_compare"),
        &plan_compare(dtype, len).unwrap(),
        session.stream(),
        &mut slots,
    );

    let out = destination(session, out_kind, extents, DType::Bool);
    let geometries = [
        geometry(session, &out.layout),
        geometry(session, &lhs.layout),
        geometry(session, &rhs.layout),
    ];
    let plan = plan_strided_compare(
        cmp,
        dtype,
        &out.layout,
        out.len as i64,
        &lhs.layout,
        lhs.len as i64,
        &rhs.layout,
        rhs.len as i64,
    )
    .unwrap();
    check_plan(&plan, len);
    let mut slots = [
        Slot::ptr(&out.buffer),
        Slot::ptr(&lhs.buffer),
        Slot::ptr(&rhs.buffer),
        slot(&geometries[0], general),
        slot(&geometries[1], general),
        slot(&geometries[2], general),
        Slot::i32(dtype.tag()),
        Slot::i32(cmp.tag()),
        Slot::i64(len),
    ];
    let name = if general {
        "t4a_cuda_strided_compare_any"
    } else {
        "t4a_cuda_strided_compare"
    };
    launch(session.function(name), &plan, session.stream(), &mut slots);
    assert_same(
        &out.logical(session),
        &flat_out.logical(session),
        DType::Bool,
        &format!("{name} {cmp:?} {dtype:?} lhs {lhs_kind:?} rhs {rhs_kind:?} out {out_kind:?}"),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_select(
    session: &Session,
    dtype: DType,
    extents: &[i64],
    kinds: [Kind; 3],
    out_kind: Kind,
    general: bool,
) {
    let len = count(extents);
    let pred = operand(session, kinds[0], extents, DType::Bool, 401);
    let on_true = operand(session, kinds[1], extents, dtype, 402);
    let on_false = operand(session, kinds[2], extents, dtype, 403);

    let flat_out = destination(session, Kind::Compact, extents, dtype);
    let compacts = [
        pred.compact(session),
        on_true.compact(session),
        on_false.compact(session),
    ];
    let mut slots = [
        Slot::ptr(&flat_out.buffer),
        Slot::ptr(&compacts[0]),
        Slot::ptr(&compacts[1]),
        Slot::ptr(&compacts[2]),
        Slot::i32(dtype.tag()),
        Slot::i64(len),
    ];
    launch(
        session.function("t4a_cuda_select"),
        &plan_select(dtype, len).unwrap(),
        session.stream(),
        &mut slots,
    );

    let out = destination(session, out_kind, extents, dtype);
    let geometries = [
        geometry(session, &out.layout),
        geometry(session, &pred.layout),
        geometry(session, &on_true.layout),
        geometry(session, &on_false.layout),
    ];
    let plan = plan_strided_select(
        dtype,
        &out.layout,
        out.len as i64,
        &pred.layout,
        pred.len as i64,
        &on_true.layout,
        on_true.len as i64,
        &on_false.layout,
        on_false.len as i64,
    )
    .unwrap();
    check_plan(&plan, len);
    let mut slots = [
        Slot::ptr(&out.buffer),
        Slot::ptr(&pred.buffer),
        Slot::ptr(&on_true.buffer),
        Slot::ptr(&on_false.buffer),
        slot(&geometries[0], general),
        slot(&geometries[1], general),
        slot(&geometries[2], general),
        slot(&geometries[3], general),
        Slot::i32(dtype.tag()),
        Slot::i64(len),
    ];
    let name = if general {
        "t4a_cuda_strided_select_any"
    } else {
        "t4a_cuda_strided_select"
    };
    launch(session.function(name), &plan, session.stream(), &mut slots);
    assert_same(
        &out.logical(session),
        &flat_out.logical(session),
        dtype,
        &format!("{name} {dtype:?} {kinds:?} out {out_kind:?}"),
    );
}

#[test]
fn strided_compare_and_select_are_bit_identical_to_the_flat_entries() {
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    for dtype in ORDERED {
        for cmp in COMPARE {
            for &(lhs, rhs, out) in &BINARY_KINDS {
                run_compare(&session, cmp, dtype, &EXTENTS, lhs, rhs, out, false);
            }
        }
        for kinds in [
            [Kind::Compact, Kind::Compact, Kind::Compact],
            [Kind::Broadcast, Kind::Permuted, Kind::Reversed],
            [Kind::Permuted, Kind::Padded, Kind::Broadcast],
            [Kind::Reversed, Kind::Broadcast, Kind::Padded],
        ] {
            run_select(&session, dtype, &EXTENTS, kinds, Kind::Compact, false);
            run_select(&session, dtype, &EXTENTS, kinds, Kind::Permuted, false);
        }
    }
    run_compare(
        &session,
        CompareOp::Lt,
        DType::F64,
        &BIG,
        Kind::Permuted,
        Kind::Broadcast,
        Kind::Compact,
        false,
    );
    run_select(
        &session,
        DType::F64,
        &BIG,
        [Kind::Broadcast, Kind::Permuted, Kind::Reversed],
        Kind::Compact,
        false,
    );
}

// --- general path -------------------------------------------------------------------

#[test]
fn general_symbols_agree_at_low_rank_and_serve_rank_nine() {
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    // Rank 9 has no by-value layout: the planner selects the general variant, and the
    // general symbol must still equal the flat entry on compact copies.
    let rank9: Vec<i64> = vec![3, 2, 2, 3, 2, 2, 2, 3, 2];
    let plan = plan_strided_unary(
        UnaryOp::Exp,
        DType::F64,
        DType::F64,
        &layout(Kind::Compact, &rank9).0,
        count(&rank9),
        &layout(Kind::Permuted, &rank9).0,
        count(&rank9),
    )
    .unwrap();
    assert_eq!(plan.variant, t4a_cuda_kernels::RankVariant::General);
    for extents in [EXTENTS.to_vec(), rank9] {
        run_unary(
            &session,
            (UnaryOp::Sin, DType::F64, DType::F64),
            &extents,
            Kind::Reversed,
            Kind::Permuted,
            true,
        );
        run_unary(
            &session,
            (UnaryOp::Abs, DType::F64, DType::C64),
            &extents,
            Kind::Broadcast,
            Kind::Compact,
            true,
        );
        for op in [
            StridedBinaryOp::Add,
            StridedBinaryOp::Div,
            StridedBinaryOp::PowChecked,
        ] {
            for dtype in [DType::C64, DType::F32, DType::I64] {
                if op.accepts(dtype) {
                    let _ = run_binary(
                        &session,
                        op,
                        dtype,
                        &extents,
                        Kind::Permuted,
                        Kind::Broadcast,
                        Kind::Permuted,
                        true,
                    );
                }
            }
        }
        run_compare(
            &session,
            CompareOp::Ge,
            DType::I32,
            &extents,
            Kind::Reversed,
            Kind::Padded,
            Kind::Compact,
            true,
        );
        run_select(
            &session,
            DType::F32,
            &extents,
            [Kind::Permuted, Kind::Broadcast, Kind::Reversed],
            Kind::Permuted,
            true,
        );
    }
}

// --- revision 15: scalar-side Add / Sub on the flat binary entry ----------------------

#[test]
fn scalar_side_add_and_sub_match_the_elementwise_entries_on_a_splat() {
    let session = Session::with_modules(&["elementwise", "arithmetic"]);
    let len = count(&BIG);
    for (binary, entry, name) in [
        (BinaryOp::Add, Entry::Add, "t4a_cuda_add"),
        (BinaryOp::Sub, Entry::Sub, "t4a_cuda_sub"),
    ] {
        assert!(!binary.accepts(DType::F64, ScalarSide::Both));
        assert!(!binary.accepts(DType::Bool, ScalarSide::Right));
        for dtype in NUMERIC {
            for side in [ScalarSide::Left, ScalarSide::Right] {
                let size = dtype.size();
                let full = values(dtype, len as usize, 500 + binary.tag() as u64);
                // Several scalars, so the specials (NaN, infinity, signed zero,
                // integer minimum) get their turn as the broadcast operand.
                for seed in 0..6u64 {
                    let scalar = values(dtype, 1, 600 + seed * 13);
                    let splat: Vec<u8> = scalar.iter().copied().cycle().take(full.len()).collect();
                    let (full_buf, scalar_buf, splat_buf) = (
                        written(&session, &full),
                        written(&session, &scalar),
                        written(&session, &splat),
                    );
                    let (lhs, rhs, lhs_splat, rhs_splat) = match side {
                        ScalarSide::Left => (&scalar_buf, &full_buf, &splat_buf, &full_buf),
                        _ => (&full_buf, &scalar_buf, &full_buf, &splat_buf),
                    };
                    let flat_out = destination(&session, Kind::Compact, &BIG, dtype);
                    let mut slots = [
                        Slot::ptr(&flat_out.buffer),
                        Slot::ptr(lhs_splat),
                        Slot::ptr(rhs_splat),
                        Slot::i32(dtype.tag()),
                        Slot::i64(len),
                    ];
                    launch(
                        session.function(name),
                        &plan_elementwise(entry, dtype, len).unwrap(),
                        session.stream(),
                        &mut slots,
                    );
                    let out = destination(&session, Kind::Compact, &BIG, dtype);
                    let plan = plan_binary(binary, dtype, side, len).unwrap();
                    check_plan(&plan, len);
                    let mut slots = [
                        Slot::ptr(&out.buffer),
                        Slot::ptr(lhs),
                        Slot::ptr(rhs),
                        Slot::i32(dtype.tag()),
                        Slot::i32(binary.tag()),
                        Slot::i32(side.tag()),
                        Slot::i64(len),
                        Slot::i64(0),
                    ];
                    launch(
                        session.function("t4a_cuda_binary"),
                        &plan,
                        session.stream(),
                        &mut slots,
                    );
                    // Add and Sub have no operand-order freedom, so even NaN bits
                    // must agree here.
                    assert_eq!(
                        out.logical(&session),
                        flat_out.logical(&session),
                        "{binary:?} {dtype:?} {side:?} scalar {:02x?}",
                        &scalar[..size]
                    );
                }
            }
        }
    }
}
