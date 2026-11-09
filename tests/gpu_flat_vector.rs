//! Vectorized flat entries (revision 14, tenferro-rs#2529) on a real device.
//!
//! Each entry runs at every vector width its dtypes admit. The planner picks
//! the width; the test steers it through the `alignment` argument. Every width
//! must write the values width 1 writes, guard region untouched: bit-identical
//! except the payload of a NaN result, which is not part of the contract. Width 1 is
//! the one-element-per-thread body the other GPU tests check against CPU
//! references; here it is also checked against a CPU reference for the
//! arithmetic entries. Inputs carry NaN payloads, signed zeros, infinities,
//! subnormals and integer extremes, and the lengths include a prime far larger
//! than one block, so a pack or tail lane that drops, duplicates or reorders an
//! element changes the bytes.

mod gpu;

use gpu::{launch, launches, read_bytes, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    address_alignment, plan_arithmetic, plan_clamp, plan_compare, plan_unary, CompareOp, DType,
    Entry, FlatLaunch, UnaryOp,
};

const SENTINEL: u8 = 0xCD;
const GUARD: usize = 13;
/// Lengths around every width's pack boundary, one block-crossing odd length and
/// a prime length of about four thousand blocks.
const LENGTHS: [i64; 9] = [0, 1, 2, 3, 4, 5, 7, 65_537, 1_000_003];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
}

/// `len` elements of `dtype` as raw bytes, special values mixed in.
fn values(dtype: DType, len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let f32s = |rng: &mut Rng| -> f32 {
        const SPECIAL: [u32; 8] = [
            0x7fc0_1234, // NaN with a payload
            0xffc0_0001, // negative NaN with a payload
            0x8000_0000, // -0
            0x0000_0000, // +0
            0x7f80_0000, // +inf
            0xff80_0000, // -inf
            0x0000_0001, // smallest subnormal
            0x807f_ffff, // largest negative subnormal
        ];
        let r = rng.next();
        if r % 7 == 0 {
            f32::from_bits(SPECIAL[(r / 7 % 8) as usize])
        } else {
            ((r % 2_000_001) as f32 - 1_000_000.0) / 977.0
        }
    };
    let f64s = |rng: &mut Rng| -> f64 {
        const SPECIAL: [u64; 8] = [
            0x7ff8_0000_0000_1234,
            0xfff8_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x800f_ffff_ffff_ffff,
        ];
        let r = rng.next();
        if r % 7 == 0 {
            f64::from_bits(SPECIAL[(r / 7 % 8) as usize])
        } else {
            ((r % 2_000_001) as f64 - 1_000_000.0) / 977.0
        }
    };
    let mut out = Vec::with_capacity(len * dtype.size());
    for _ in 0..len {
        match dtype {
            DType::F32 => out.extend(f32s(&mut rng).to_le_bytes()),
            DType::F64 => out.extend(f64s(&mut rng).to_le_bytes()),
            DType::C32 => {
                out.extend(f32s(&mut rng).to_le_bytes());
                out.extend(f32s(&mut rng).to_le_bytes());
            }
            DType::C64 => {
                out.extend(f64s(&mut rng).to_le_bytes());
                out.extend(f64s(&mut rng).to_le_bytes());
            }
            DType::I32 => {
                let r = rng.next();
                let v = match r % 9 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    _ => (r >> 4) as i32,
                };
                out.extend(v.to_le_bytes());
            }
            DType::I64 => {
                let r = rng.next();
                let v = match r % 9 {
                    0 => i64::MIN,
                    1 => i64::MAX,
                    _ => (r << 9) as i64 ^ (r as i64),
                };
                out.extend(v.to_le_bytes());
            }
            DType::Bool => out.push((rng.next() & 1) as u8),
        }
    }
    out
}

/// Run one planned launch into a fresh sentinel destination and return every
/// destination byte, guard included. `slots` gets the output buffer and the
/// planned width.
fn run(
    session: &Session,
    function: &str,
    plan: FlatLaunch,
    out_bytes: usize,
    slots: impl Fn(&Buf, i32) -> Vec<Slot>,
) -> Vec<u8> {
    let out = written(session, &vec![SENTINEL; out_bytes + GUARD]);
    let mut slots = slots(&out, plan.vector_width as i32);
    let before = launches();
    launch(
        session.function(function),
        &plan.launch,
        session.stream(),
        &mut slots,
    );
    assert_eq!(
        launches(),
        before + usize::from(!plan.is_empty()),
        "{function}: one physical launch per non-empty plan"
    );
    let bytes = read_bytes(session, &out);
    assert!(
        bytes[out_bytes..].iter().all(|&b| b == SENTINEL),
        "{function} width {}: the guard behind the output was written",
        plan.vector_width
    );
    bytes
}

/// Widths to exercise for operands whose widest element is `widest` bytes, with
/// the alignment that makes the planner choose each of them.
fn widths(widest: usize) -> Vec<(u32, u64)> {
    [1u32, 2, 4]
        .into_iter()
        .filter(|&w| widest * w as usize <= t4a_cuda_kernels::MAX_VECTOR_BYTES)
        .map(|w| (w, (widest * w as usize) as u64))
        .collect()
}

/// The width the planner must have chosen when asked for `wanted` over `len`.
fn expected_width(wanted: u32, len: i64) -> u32 {
    if len == 0 {
        1
    } else {
        [4u32, 2, 1]
            .into_iter()
            .find(|&w| w <= wanted && i64::from(w) <= len)
            .unwrap_or(1)
    }
}

/// A float component two widths may disagree on: its index, and both values.
type Allowance<'a> = &'a dyn Fn(usize, f64, f64) -> bool;

/// No difference beyond a NaN payload is allowed.
fn exact(_: usize, _: f64, _: f64) -> bool {
    false
}

/// First output component where `a` and `b` disagree under the cross-width
/// contract: integer and Bool components are bit-identical; a float component is
/// bit-identical unless both are NaN, whose payload is not part of the contract
/// (as for the reductions), or `allowed` accepts the pair. Complex outputs compare
/// their parts.
fn first_mismatch(out: DType, a: &[u8], b: &[u8], allowed: Allowance<'_>) -> Option<usize> {
    let (part, float) = match out {
        DType::F32 | DType::C32 => (4, true),
        DType::F64 | DType::C64 => (8, true),
        DType::I32 => (4, false),
        DType::I64 => (8, false),
        DType::Bool => (1, false),
    };
    let value = |bytes: &[u8]| match part {
        4 => f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
        _ => f64::from_le_bytes(bytes.try_into().unwrap()),
    };
    a.chunks_exact(part)
        .zip(b.chunks_exact(part))
        .enumerate()
        .position(|(index, (x, y))| {
            if x == y {
                return false;
            }
            if !float {
                return true;
            }
            let (x, y) = (value(x), value(y));
            !((x.is_nan() && y.is_nan()) || allowed(index, x, y))
        })
}

fn check_widths(
    what: &str,
    out: DType,
    widest: usize,
    len: i64,
    allowed: Allowance<'_>,
    plan_for: impl Fn(u64) -> FlatLaunch,
    run_plan: impl Fn(FlatLaunch) -> Vec<u8>,
) -> Vec<u8> {
    let scalar_plan = plan_for(1);
    assert_eq!(scalar_plan.vector_width, 1, "{what}");
    let scalar = run_plan(scalar_plan);
    let payload = len as usize * out.size();
    for (wanted, alignment) in widths(widest) {
        let plan = plan_for(alignment);
        assert_eq!(
            plan.vector_width,
            expected_width(wanted, len),
            "{what} len {len} alignment {alignment}"
        );
        let bytes = run_plan(plan);
        // `run` already proved the guard behind the payload untouched for both.
        if let Some(index) = first_mismatch(out, &bytes[..payload], &scalar[..payload], allowed) {
            let part = out.size()
                / if matches!(out, DType::C32 | DType::C64) {
                    2
                } else {
                    1
                };
            let hex = |buffer: &[u8]| {
                buffer[index * part..(index + 1) * part]
                    .iter()
                    .rev()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            };
            panic!(
                "{what} len {len}: width {} differs from width 1 at component {index}: {} vs {}",
                plan.vector_width,
                hex(&bytes),
                hex(&scalar)
            );
        }
    }
    scalar
}

#[test]
fn arithmetic_widths_write_the_scalar_bytes() {
    let session = Session::new("arithmetic");
    let cases = [
        (Entry::Add, DType::F32),
        (Entry::Add, DType::F64),
        (Entry::Add, DType::I32),
        (Entry::Add, DType::I64),
        (Entry::Add, DType::C32),
        (Entry::Add, DType::C64),
        (Entry::Sub, DType::F32),
        (Entry::Sub, DType::I64),
        (Entry::Mul, DType::F64),
        (Entry::Mul, DType::C32),
        (Entry::Mul, DType::C64),
        (Entry::Mul, DType::I32),
        (Entry::Div, DType::F32),
        (Entry::Div, DType::F64),
    ];
    for (entry, dtype) in cases {
        for len in LENGTHS {
            let n = len as usize;
            let lhs_host = values(dtype, n.max(1), 1 + len as u64);
            let rhs_host = values(dtype, n.max(1), 2 + len as u64);
            let lhs = written(&session, &lhs_host);
            let rhs = written(&session, &rhs_host);
            let what = format!("{entry:?} {dtype:?}");
            // Complex multiplication is `a.re*b.re - a.im*b.im` (and the imaginary
            // analogue), where the ABI leaves FMA contraction to the compiler. The pack
            // lanes and the scalar tail are separately compiled, so they may contract
            // differently; the difference is bounded normwise, not in ULPs of a result
            // that may cancel.
            let complex_mul = |index: usize, x: f64, y: f64| {
                let (part, eps) = match dtype {
                    DType::C32 => (4, f64::from(f32::EPSILON)),
                    DType::C64 => (8, f64::EPSILON),
                    _ => return false,
                };
                if entry != Entry::Mul {
                    return false;
                }
                let get = |bytes: &[u8], component: usize| match part {
                    4 => f64::from(f32::from_le_bytes(
                        bytes[component * 4..component * 4 + 4].try_into().unwrap(),
                    )),
                    _ => f64::from_le_bytes(
                        bytes[component * 8..component * 8 + 8].try_into().unwrap(),
                    ),
                };
                let element = index / 2;
                let (ar, ai) = (get(&lhs_host, 2 * element), get(&lhs_host, 2 * element + 1));
                let (br, bi) = (get(&rhs_host, 2 * element), get(&rhs_host, 2 * element + 1));
                let scale = if index % 2 == 0 {
                    (ar * br).abs() + (ai * bi).abs()
                } else {
                    (ar * bi).abs() + (ai * br).abs()
                };
                (x - y).abs() <= 2.0 * eps * scale
            };
            let scalar = check_widths(
                &what,
                dtype,
                dtype.size(),
                len,
                &complex_mul,
                |alignment| plan_arithmetic(entry, dtype, len, alignment).unwrap(),
                |plan| {
                    run(
                        &session,
                        entry.name(),
                        plan,
                        n * dtype.size(),
                        |out, width| {
                            vec![
                                Slot::ptr(out),
                                Slot::ptr(&lhs),
                                Slot::ptr(&rhs),
                                Slot::i32(dtype.tag()),
                                Slot::i64(len),
                                Slot::i32(width),
                            ]
                        },
                    )
                },
            );
            if matches!(dtype, DType::F64) && matches!(entry, Entry::Add) {
                let (l, r) = (&lhs_host, &rhs_host);
                for i in 0..n {
                    let at = |b: &[u8]| f64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
                    let expected = at(l) + at(r);
                    let actual = at(&scalar);
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || (actual.is_nan() && expected.is_nan()),
                        "add f64 len {len} element {i}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn unary_widths_write_the_scalar_bytes() {
    let session = Session::new("elementwise");
    let cases = [
        (UnaryOp::Exp, DType::F32, DType::F32),
        (UnaryOp::Exp, DType::F64, DType::F64),
        (UnaryOp::Tanh, DType::F64, DType::F64),
        (UnaryOp::Sin, DType::F32, DType::F32),
        (UnaryOp::Erf, DType::F64, DType::F64),
        (UnaryOp::Neg, DType::I32, DType::I32),
        (UnaryOp::Abs, DType::I64, DType::I64),
        (UnaryOp::Neg, DType::C32, DType::C32),
        (UnaryOp::Sign, DType::C64, DType::C64),
        (UnaryOp::Abs, DType::F32, DType::C32),
        (UnaryOp::Abs, DType::F64, DType::C64),
    ];
    for (op, dst, src) in cases {
        for len in LENGTHS {
            let n = len as usize;
            let input = written(&session, &values(src, n.max(1), 3 + len as u64));
            check_widths(
                &format!("{op:?} {dst:?}<-{src:?}"),
                dst,
                dst.size().max(src.size()),
                len,
                &exact,
                |alignment| plan_unary(op, dst, src, len, alignment).unwrap(),
                |plan| {
                    run(
                        &session,
                        "t4a_cuda_unary",
                        plan,
                        n * dst.size(),
                        |out, width| {
                            vec![
                                Slot::ptr(out),
                                Slot::ptr(&input),
                                Slot::i32(dst.tag()),
                                Slot::i32(src.tag()),
                                Slot::i32(op.tag()),
                                Slot::i64(len),
                                Slot::i32(width),
                            ]
                        },
                    )
                },
            );
        }
    }
}

#[test]
fn compare_and_clamp_widths_write_the_scalar_bytes() {
    let session = Session::new("elementwise");
    for dtype in [DType::F32, DType::F64, DType::I32, DType::I64] {
        for cmp in [CompareOp::Lt, CompareOp::Eq] {
            for len in LENGTHS {
                let n = len as usize;
                let lhs = written(&session, &values(dtype, n.max(1), 5 + len as u64));
                let rhs = written(&session, &values(dtype, n.max(1), 6 + len as u64));
                check_widths(
                    &format!("compare {cmp:?} {dtype:?}"),
                    DType::Bool,
                    dtype.size(),
                    len,
                    &exact,
                    |alignment| plan_compare(dtype, len, alignment).unwrap(),
                    |plan| {
                        run(&session, "t4a_cuda_compare", plan, n, |out, width| {
                            vec![
                                Slot::ptr(out),
                                Slot::ptr(&lhs),
                                Slot::ptr(&rhs),
                                Slot::i32(dtype.tag()),
                                Slot::i32(cmp.tag()),
                                Slot::i64(len),
                                Slot::i32(width),
                            ]
                        })
                    },
                );
            }
        }
    }
    for dtype in [DType::F32, DType::F64] {
        for len in LENGTHS {
            let n = len as usize;
            let input = written(&session, &values(dtype, n.max(1), 7 + len as u64));
            let lower = written(&session, &values(dtype, n.max(1), 8 + len as u64));
            let upper = written(&session, &values(dtype, n.max(1), 9 + len as u64));
            check_widths(
                &format!("clamp {dtype:?}"),
                dtype,
                dtype.size(),
                len,
                &exact,
                |alignment| plan_clamp(dtype, len, alignment).unwrap(),
                |plan| {
                    run(
                        &session,
                        "t4a_cuda_clamp",
                        plan,
                        n * dtype.size(),
                        |out, width| {
                            vec![
                                Slot::ptr(out),
                                Slot::ptr(&input),
                                Slot::ptr(&lower),
                                Slot::ptr(&upper),
                                Slot::i32(dtype.tag()),
                                Slot::i64(len),
                                Slot::i32(width),
                            ]
                        },
                    )
                },
            );
        }
    }
}

/// An operand one element past an allocation base is not aligned for a pack:
/// the planner falls back to width 1 and the result is the shifted operand's.
#[test]
fn misaligned_operand_falls_back_to_width_one() {
    let session = Session::new("arithmetic");
    let dtype = DType::F64;
    let len = 65_537i64;
    let n = len as usize;
    let lhs_values = values(dtype, n + 1, 11);
    let rhs_values = values(dtype, n, 12);
    let lhs = written(&session, &lhs_values);
    let rhs = written(&session, &rhs_values);
    let probe = written(&session, &[0u8; 8]);
    let alignment = address_alignment(&[
        probe.as_ptr() as u64,
        lhs.as_ptr() as u64 + 8,
        rhs.as_ptr() as u64,
    ]);
    let plan = plan_arithmetic(Entry::Add, dtype, len, alignment).unwrap();
    assert_eq!(plan.vector_width, 1, "alignment {alignment}");
    let aligned = plan_arithmetic(
        Entry::Add,
        dtype,
        len,
        address_alignment(&[
            probe.as_ptr() as u64,
            lhs.as_ptr() as u64,
            rhs.as_ptr() as u64,
        ]),
    )
    .unwrap();
    assert_eq!(
        aligned.vector_width, 2,
        "allocation bases admit a 16-byte pack"
    );
    let bytes = run(&session, Entry::Add.name(), plan, n * 8, |out, width| {
        vec![
            Slot::ptr(out),
            Slot::ptr_offset(&lhs, 8),
            Slot::ptr(&rhs),
            Slot::i32(dtype.tag()),
            Slot::i64(len),
            Slot::i32(width),
        ]
    });
    for i in 0..n {
        let at = |b: &[u8], k: usize| f64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap());
        let expected = at(&lhs_values, i + 1) + at(&rhs_values, i);
        let actual = at(&bytes, i);
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "element {i}: {actual} vs {expected}"
        );
    }
}
