//! Bit-identity golden for split reductions (issue #30).
//!
//! Each case runs the planned two-pass reduction (`plan_reduction` → pass 1 →
//! combine) through both the fast and the general-rank (`_any`) entries, and compares
//! with `tests/data/reduce-golden.tsv`. That fixture was captured from the `ab7395c`
//! kernels (release 0.1.2) before any of the combine or pass-1 changes. It holds:
//!
//! - the FNV-1a 64 hash and byte length of the whole pass-1 partials buffer, and
//! - the bytes of every final output, as hex.
//!
//! Every bit is compared, including signed zeros and NaN payloads. The fixture is
//! never regenerated to make a change pass.
//!
//! One listed exception, kept visible rather than absorbed into the golden:
//! `tests/data/reduce-golden-nan-payload.tsv` names the cases where #29's combine
//! load-ahead keeps a *different NaN payload* than `ab7395c` (f64/c64 additive combines
//! that meet two NaNs with different payloads; ptxas picks the `DADD` operand order). For
//! those keys the finals must equal the listed bytes exactly, and must still be NaN where
//! the golden is NaN. Any other difference, and any listed key that stops differing,
//! fails. `T4A_REDUCE_GOLDEN_WRITE=<path>` writes a
//! fresh capture to `<path>` so that two builds can be diffed; the test then
//! compares nothing.
//!
//! Run with `cargo test --release --features gpu-tests --test gpu_reduce_golden`;
//! requires a CUDA device.

mod gpu;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gpu::{launch_async, pub_sync, read_bytes, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_reduction, reduction_partials_layout, CLayout, CLayoutAny, DType, Layout, ReduceOp,
    ReductionPlan, WideLayout,
};

const SENTINEL: u8 = 0xCD;
/// Guard bytes after every destination, so a write past the end is visible.
const GUARD: usize = 256;
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/reduce-golden.tsv");
const PAYLOAD_LIST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/reduce-golden-nan-payload.tsv"
);

// ---- deterministic inputs -------------------------------------------------------

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1) on a 2^-52 grid, exactly representable in f64.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / (1u64 << 52) as f64) - 1.0
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").unwrap();
    }
    out
}

/// Component width (4 or 8 bytes) and components per element.
fn components(dtype: DType) -> (usize, usize) {
    match dtype {
        DType::F32 => (4, 1),
        DType::F64 => (8, 1),
        DType::C32 => (4, 2),
        DType::C64 => (8, 2),
        _ => unreachable!("golden covers float and complex dtypes"),
    }
}

/// A component value: an ordinary number, or raw bits written as they are.
#[derive(Clone, Copy)]
enum Comp {
    Num(f64),
    /// NaN with payload index `k` (distinct per k, sign bit alternating).
    Nan(u32),
}

fn put(bytes: &mut [u8], width: usize, value: Comp) {
    match (width, value) {
        (4, Comp::Num(x)) => bytes.copy_from_slice(&(x as f32).to_bits().to_le_bytes()),
        (8, Comp::Num(x)) => bytes.copy_from_slice(&x.to_bits().to_le_bytes()),
        (4, Comp::Nan(k)) => {
            let sign = if k % 2 == 1 { 0x8000_0000u32 } else { 0 };
            let bits = sign | 0x7fc0_0000 | (0x100 + k * 0x11);
            bytes.copy_from_slice(&bits.to_le_bytes());
        }
        (8, Comp::Nan(k)) => {
            let sign = if k % 2 == 1 { 1u64 << 63 } else { 0 };
            let bits = sign | 0x7ff8_0000_0000_0000 | (0x1000 + u64::from(k) * 0x111);
            bytes.copy_from_slice(&bits.to_le_bytes());
        }
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// Ordinary finite values (near 1 for `PROD`, so the product stays finite).
    Random,
    /// Random plus NaNs with distinct payloads, several meeting in one pass-1 lane, one
    /// chunk, and one combine lane.
    NanPayloads,
    /// Random plus `+inf` and `-inf` in different chunks (and a zero for `PROD`).
    Infs,
    /// Zeros of random sign.
    SignedZeros,
    /// Every element `-0.0`.
    NegZeros,
    /// `PROD` only: magnitudes in [1.5, 2.5), so the walk overflows, and one zero.
    Overflow,
}

const CLASSES: [Class; 6] = [
    Class::Random,
    Class::NanPayloads,
    Class::Infs,
    Class::SignedZeros,
    Class::NegZeros,
    Class::Overflow,
];

struct Shape {
    name: &'static str,
    extents: &'static [i64],
    axis: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "n1048576",
        extents: &[1_048_576],
        axis: 0,
    },
    Shape {
        name: "n1000003",
        extents: &[1_000_003],
        axis: 0,
    },
    Shape {
        name: "n4194304",
        extents: &[4_194_304],
        axis: 0,
    },
    Shape {
        name: "n65541",
        extents: &[65_541],
        axis: 0,
    },
    Shape {
        name: "n8195",
        extents: &[8_195],
        axis: 0,
    },
    Shape {
        name: "3x100003-axis1",
        extents: &[3, 100_003],
        axis: 1,
    },
    Shape {
        name: "131075x3-axis0",
        extents: &[131_075, 3],
        axis: 0,
    },
];

fn compact_strides(extents: &[i64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(extents.len());
    let mut stride = 1;
    for &extent in extents {
        strides.push(stride);
        stride *= extent;
    }
    strides
}

/// Element offset of output line `o`'s first reduced element.
fn line_base(extents: &[i64], axis: usize, mut o: i64) -> i64 {
    let strides = compact_strides(extents);
    let mut base = 0;
    for dim in 0..extents.len() {
        if dim == axis {
            continue;
        }
        base += (o % extents[dim]) * strides[dim];
        o /= extents[dim];
    }
    base
}

/// Axis positions that place specials where the passes meet them: lanes 0/1 of chunk 0
/// (the same lane twice), chunks that share a lane of every combine width, and the end.
fn nan_positions(axis_len: i64, splits: i64, chunk: i64) -> Vec<i64> {
    let mut out = vec![0, 1, 33];
    // Lane 0 of every pass-1 walk, every 32 elements of chunk 0: the scalar walk's lane 0, and
    // (for 4- and 8-byte elements) lane 0 of the vector walk's first batch and its whole-batch
    // body, so NaNs with different payloads meet inside one lane's fold.
    for k in 1..32i64 {
        out.push(32 * k);
    }
    for k in [5i64, 37, 69, 133, 517, 1029] {
        if k < splits {
            out.push(k * chunk + 7);
        }
    }
    out.push(axis_len - 1);
    out.retain(|&p| p < axis_len);
    out.sort_unstable();
    out.dedup();
    out
}

fn input_bytes(shape: &Shape, dtype: DType, op: ReduceOp, class: Class, splits: i64) -> Vec<u8> {
    let (width, comps) = components(dtype);
    let count: i64 = shape.extents.iter().product();
    let elem = width * comps;
    let seed = fnv1a64(format!("{}|{:?}|{:?}|{:?}", shape.name, dtype, op, class).as_bytes());
    let mut rng = SplitMix(seed);
    let prod = op == ReduceOp::Prod;
    let mut bytes = vec![0u8; count as usize * elem];
    for i in 0..count as usize {
        for c in 0..comps {
            let u = rng.unit();
            let value = match class {
                Class::SignedZeros => {
                    if u < 0.0 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                Class::NegZeros => -0.0,
                Class::Overflow => {
                    let magnitude = 2.0 + 0.5 * rng.unit();
                    if u < 0.0 {
                        -magnitude
                    } else {
                        magnitude
                    }
                }
                _ if prod && c == 0 => 1.0 + u * (1.0 / 512.0),
                _ if prod => u * (1.0 / 512.0),
                _ => u,
            };
            let at = i * elem + c * width;
            put(&mut bytes[at..at + width], width, Comp::Num(value));
        }
    }
    let axis_len = shape.extents[shape.axis];
    let axis_stride = compact_strides(shape.extents)[shape.axis];
    let outputs = count / axis_len;
    let chunk = axis_len / splits.max(1);
    let mut lines = vec![0];
    if outputs > 1 {
        lines.push(outputs - 1);
    }
    let mut k = 0u32;
    for &line in &lines {
        let base = line_base(shape.extents, shape.axis, line);
        let mut set = |position: i64, comp: usize, value: Comp| {
            let at = (base + position * axis_stride) as usize * elem + comp * width;
            put(&mut bytes[at..at + width], width, value);
        };
        match class {
            Class::NanPayloads => {
                for position in nan_positions(axis_len, splits, chunk) {
                    set(position, (k as usize) % comps, Comp::Nan(k));
                    k += 1;
                }
            }
            Class::Infs => {
                let late = if 41 < splits {
                    41 * chunk + 2
                } else {
                    40.min(axis_len - 1)
                };
                set(
                    (9 * chunk + 2).min(axis_len - 1),
                    0,
                    Comp::Num(f64::INFINITY),
                );
                set(late, 0, Comp::Num(f64::NEG_INFINITY));
                if prod {
                    set((3 * chunk + 1).min(axis_len - 1), 0, Comp::Num(0.0));
                }
            }
            Class::Overflow => set((3 * chunk + 1).min(axis_len - 1), 0, Comp::Num(0.0)),
            _ => {}
        }
    }
    bytes
}

// ---- launches ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Path {
    Fast,
    Any,
}

struct Geometry {
    fast: CLayout,
    general: CLayoutAny,
    _buffers: [Buf; 2],
}

fn geometry(session: &Session, layout: &WideLayout) -> Geometry {
    let fast =
        CLayout::new(&Layout::new(layout.extents(), layout.strides(), layout.offset()).unwrap());
    let extents = upload(session, layout.extents());
    let strides = upload(session, layout.strides());
    let general = CLayoutAny::new(layout.as_ref(), extents.as_ptr(), strides.as_ptr()).unwrap();
    Geometry {
        fast,
        general,
        _buffers: [extents, strides],
    }
}

fn layout_slot(path: Path, geometry: &Geometry) -> Slot {
    match path {
        Path::Fast => Slot::layout(&geometry.fast),
        Path::Any => Slot::layout_any(&geometry.general),
    }
}

/// Destination bytes with the guard checked and stripped.
fn take(session: &Session, buf: &Buf, bytes: usize, what: &str) -> Vec<u8> {
    let all = read_bytes(session, buf);
    assert!(
        all[bytes..bytes + GUARD].iter().all(|&b| b == SENTINEL),
        "{what}: a write landed past the destination"
    );
    all[..bytes].to_vec()
}

struct Outcome {
    plan: String,
    partials: Vec<u8>,
    finals: Vec<u8>,
}

fn run(
    session: &Session,
    src: &Buf,
    shape: &Shape,
    dtype: DType,
    op: ReduceOp,
    width: u32,
    path: Path,
) -> Outcome {
    let (w, comps) = components(dtype);
    let elem = w * comps;
    let in_layout = WideLayout::compact(shape.extents.to_vec()).unwrap();
    let mut out_extents = shape.extents.to_vec();
    out_extents[shape.axis] = 1;
    let out_layout = WideLayout::compact(out_extents).unwrap();
    let in_len = in_layout.element_count().unwrap();
    let out_len = out_layout.element_count().unwrap();
    let plan = plan_reduction(
        dtype,
        op,
        &out_layout,
        out_len,
        &in_layout,
        in_len,
        shape.axis as u32,
        width,
    )
    .unwrap();
    let ReductionPlan::Split {
        split,
        pass1,
        combine_strategy,
        combine_op,
        pass2,
    } = plan
    else {
        panic!("{}: golden shapes must split", shape.name);
    };
    let suffix = if path == Path::Any { "_any" } else { "" };
    let partials_fn = session.function(&format!("t4a_cuda_reduce_partials{suffix}"));
    let reduce_fn = session.function(&format!("t4a_cuda_reduce{suffix}"));
    let partials_layout =
        reduction_partials_layout(&in_layout, shape.axis as u32, split.splits).unwrap();
    let partials_len = partials_layout.element_count().unwrap() as usize;
    let in_geometry = geometry(session, &in_layout);
    let out_geometry = geometry(session, &out_layout);
    let partials_geometry = geometry(session, &partials_layout);
    let partials = written(session, &vec![SENTINEL; partials_len * elem + GUARD]);
    let out = written(session, &vec![SENTINEL; out_len as usize * elem + GUARD]);
    let mut slots1 = [
        Slot::ptr(&partials),
        Slot::ptr(src),
        Slot::i32(dtype.tag()),
        Slot::i32(op.tag()),
        layout_slot(path, &partials_geometry),
        layout_slot(path, &in_geometry),
        Slot::i32(shape.axis as i32),
        Slot::i64(out_len),
        Slot::i64(split.splits),
        Slot::i64(split.chunk_len),
    ];
    launch_async(partials_fn, &pass1, session.stream(), &mut slots1);
    let mut slots2 = [
        Slot::ptr(&out),
        Slot::ptr(&partials),
        Slot::i32(dtype.tag()),
        Slot::i32(combine_op.tag()),
        Slot::i32(combine_strategy.tag()),
        layout_slot(path, &out_geometry),
        layout_slot(path, &partials_geometry),
        Slot::i32(shape.axis as i32),
        Slot::i64(out_len),
    ];
    launch_async(reduce_fn, &pass2, session.stream(), &mut slots2);
    pub_sync(session.stream());
    let partials = take(session, &partials, partials_len * elem, "pass 1");
    for (index, element) in partials.chunks(elem).enumerate() {
        assert!(
            element.iter().any(|&b| b != SENTINEL),
            "{}: partial {index} was never written",
            shape.name
        );
    }
    Outcome {
        plan: format!(
            "splits={} chunk={} combine={:?}/{}",
            split.splits, split.chunk_len, combine_strategy, pass2.block[0]
        ),
        partials,
        finals: take(session, &out, out_len as usize * elem, "combine"),
    }
}

fn cases() -> Vec<(DType, ReduceOp)> {
    let mut out = Vec::new();
    for dtype in [DType::F32, DType::F64] {
        for op in [
            ReduceOp::Sum,
            ReduceOp::Prod,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::SumSquares,
        ] {
            out.push((dtype, op));
        }
    }
    for dtype in [DType::C32, DType::C64] {
        for op in [ReduceOp::Sum, ReduceOp::Prod] {
            out.push((dtype, op));
        }
    }
    out
}

/// One line per case: `key \t plan \t partials_bytes \t partials_fnv \t finals_hex`.
fn capture() -> BTreeMap<String, (String, usize, String, String)> {
    let session = Session::new("reduce");
    let mut out = BTreeMap::new();
    for shape in SHAPES {
        // The narrow widths only on one ragged shape: they must not change either.
        let widths: &[u32] = if shape.name == "n1000003" {
            &[32, 16, 8]
        } else {
            &[32]
        };
        for &width in widths {
            let axis_len = shape.extents[shape.axis];
            let splits = t4a_cuda_kernels::split_policy(axis_len, width)
                .unwrap()
                .map_or(1, |split| split.splits);
            for (dtype, op) in cases() {
                for class in CLASSES {
                    if class == Class::Overflow && op != ReduceOp::Prod {
                        continue;
                    }
                    let src = upload(&session, &input_bytes(shape, dtype, op, class, splits));
                    for path in [Path::Fast, Path::Any] {
                        let outcome = run(&session, &src, shape, dtype, op, width, path);
                        let key = format!(
                            "{}|w{width}|{dtype:?}|{op:?}|{class:?}|{}",
                            shape.name,
                            if path == Path::Fast { "fast" } else { "any" }
                        );
                        out.insert(
                            key,
                            (
                                outcome.plan,
                                outcome.partials.len(),
                                format!("{:016x}", fnv1a64(&outcome.partials)),
                                hex(&outcome.finals),
                            ),
                        );
                    }
                }
            }
        }
    }
    out
}

fn parse_hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn is_nan_bits(bytes: &[u8]) -> bool {
    match bytes.len() {
        4 => f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())).is_nan(),
        8 => f64::from_bits(u64::from_le_bytes(bytes.try_into().unwrap())).is_nan(),
        _ => unreachable!(),
    }
}

/// Whether two final buffers differ only in the payload of components that are NaN in both.
fn payload_only(dtype: DType, golden: &[u8], got: &[u8]) -> bool {
    let (width, _) = components(dtype);
    golden.len() == got.len()
        && golden
            .chunks(width)
            .zip(got.chunks(width))
            .all(|(a, b)| a == b || (is_nan_bits(a) && is_nan_bits(b)))
}

fn dtype_of(key: &str) -> DType {
    match key.split('|').nth(2).unwrap() {
        "F32" => DType::F32,
        "F64" => DType::F64,
        "C32" => DType::C32,
        "C64" => DType::C64,
        other => panic!("unknown dtype {other}"),
    }
}

#[test]
fn split_reductions_match_the_ab7395c_golden_bit_for_bit() {
    let got = capture();
    if let Ok(path) = std::env::var("T4A_REDUCE_GOLDEN_WRITE") {
        let nvcc = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .to_string()
            })
            .unwrap_or_default();
        let source = std::env::var("T4A_REDUCE_GOLDEN_SOURCE").unwrap_or_default();
        let mut text = format!(
            "# reduce golden: source {source}; nvcc {nvcc}; arch sm_80; {} cases\n\
             # key\tplan\tpartials_bytes\tpartials_fnv1a64\tfinals_hex\n",
            got.len()
        );
        for (key, (plan, bytes, hash, finals)) in &got {
            writeln!(text, "{key}\t{plan}\t{bytes}\t{hash}\t{finals}").unwrap();
        }
        std::fs::write(&path, text).unwrap();
        eprintln!("wrote {} cases to {path}", got.len());
        return;
    }
    let fixture = std::fs::read_to_string(FIXTURE).expect("tests/data/reduce-golden.tsv");
    let mut golden = BTreeMap::new();
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<&str> = line.split('\t').collect();
        golden.insert(
            fields[0].to_string(),
            (
                fields[1].to_string(),
                fields[2].parse::<usize>().unwrap(),
                fields[3],
                fields[4],
            ),
        );
    }
    assert_eq!(
        golden.keys().collect::<Vec<_>>(),
        got.keys().collect::<Vec<_>>(),
        "the case set changed; the golden must cover exactly the cases run"
    );
    let listed: BTreeMap<String, String> = std::fs::read_to_string(PAYLOAD_LIST)
        .expect("tests/data/reduce-golden-nan-payload.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            (fields[0].to_string(), fields[1].to_string())
        })
        .collect();
    let mut listed_seen = 0usize;
    let mut partials_diff = Vec::new();
    let mut value_diff = Vec::new();
    let mut payload_diff = Vec::new();
    for (key, (plan, bytes, hash, finals)) in &got {
        let (g_plan, g_bytes, g_hash, g_finals) = &golden[key];
        assert_eq!(plan, g_plan, "{key}: the split plan changed");
        if bytes != g_bytes || hash != g_hash {
            partials_diff.push(key.clone());
        }
        if finals != g_finals {
            let (a, b) = (parse_hex(g_finals), parse_hex(finals));
            if payload_only(dtype_of(key), &a, &b) && listed.get(key) == Some(finals) {
                listed_seen += 1;
            } else if payload_only(dtype_of(key), &a, &b) {
                payload_diff.push(format!("{key}\t{finals}\tgolden {g_finals}"));
            } else {
                value_diff.push(format!("{key}: golden {g_finals} got {finals}"));
            }
        }
    }
    let listed_stale: Vec<&String> = listed
        .keys()
        .filter(|key| got.get(*key).is_some_and(|row| row.3 == golden[*key].3))
        .collect();
    eprintln!(
        "cases {}  partials differ {}  finals differ: value/sign/nan-ness {}  \
         unlisted nan-payload-only {}  listed nan-payload-only {}/{}  listed but now equal {}",
        got.len(),
        partials_diff.len(),
        value_diff.len(),
        payload_diff.len(),
        listed_seen,
        listed.len(),
        listed_stale.len()
    );
    for line in partials_diff.iter().take(20) {
        eprintln!("partials differ: {line}");
    }
    for line in value_diff.iter().take(20) {
        eprintln!("final differs: {line}");
    }
    for line in &payload_diff {
        eprintln!("nan payload differs: {line}");
    }
    assert!(
        partials_diff.is_empty()
            && value_diff.is_empty()
            && payload_diff.is_empty()
            && listed_seen == listed.len()
            && listed_stale.is_empty(),
        "split reductions are not bit-identical to the ab7395c golden"
    );
}
