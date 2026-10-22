//! Numerical GPU tests for the revision-7 linalg auxiliary entries.
//!
//! Every entry is planned by the facade, launched through the raw CUDA driver
//! API (one physical launch per call, counted through `gpu::launches`) and
//! compared against a CPU reference built from the same layouts, so a mis-mapped
//! stride, offset or batch axis cannot pass. The kernels under test are
//! translated from tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
//! (`crates/tenferro-linalg/src/gpu/kernels.rs`,
//! `crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs`), Copyright
//! (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors, MIT OR Apache-2.0.
//!
//! Every logical domain is exercised both below and above one 256-thread block,
//! so a grid that stops at the first block cannot pass. Copying entries are
//! compared byte for byte, including NaN payloads and zero signs; only the phase
//! arithmetic uses a tolerance.
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_linalg_aux`;
//! requires a CUDA device.

mod gpu;

use gpu::{as_bytes, from_bytes, launch, launches, read_bytes, upload, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_fill_one, plan_householder_assemble, plan_householder_explicit_v, plan_identity_shifted,
    plan_lu_apply_pivots, plan_lu_extract_outputs, plan_lu_parity, plan_matrix_adjoint,
    plan_qr_apply_phase, plan_qr_phase, plan_upper_trapezoidal_violation, CLayout, DType, Launch,
    Layout,
};

const SENTINEL: u8 = 0xCD;
/// Bytes reserved behind a logical output to catch a write past `len`.
const GUARD: usize = 13;
/// A `fill_one` domain above one 256-thread block, and not a multiple of it.
const BIG: usize = 1003;

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

fn cmul32(lhs: C32, rhs: C32) -> C32 {
    C32 {
        re: lhs.re * rhs.re - lhs.im * rhs.im,
        im: lhs.re * rhs.im + lhs.im * rhs.re,
    }
}

fn cmul64(lhs: C64, rhs: C64) -> C64 {
    C64 {
        re: lhs.re * rhs.re - lhs.im * rhs.im,
        im: lhs.re * rhs.im + lhs.im * rhs.re,
    }
}

fn conj32(value: C32) -> C32 {
    C32 {
        re: value.re,
        im: -value.im,
    }
}

fn conj64(value: C64) -> C64 {
    C64 {
        re: value.re,
        im: -value.im,
    }
}

// --- value traits -----------------------------------------------------------

/// Zero, one, negation and conjugation of the payload types the entries copy.
trait Unit: Copy + std::fmt::Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn negated(self) -> Self;
    /// The identity for the real and integer payloads.
    fn conjugated(self) -> Self {
        self
    }
}

macro_rules! unit_real {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Unit for $ty {
            fn zero() -> $ty {
                $zero
            }
            fn one() -> $ty {
                $one
            }
            fn negated(self) -> $ty {
                -self
            }
        }
    };
}

unit_real!(f64, 0.0, 1.0);
unit_real!(f32, 0.0, 1.0);
unit_real!(i32, 0, 1);
unit_real!(i64, 0, 1);

impl Unit for u8 {
    fn zero() -> u8 {
        0
    }
    fn one() -> u8 {
        1
    }
    // Unused: only the violation entry takes a Bool payload, and it never negates.
    fn negated(self) -> u8 {
        self.wrapping_neg()
    }
}

impl Unit for C32 {
    fn zero() -> C32 {
        C32 { re: 0.0, im: 0.0 }
    }
    fn one() -> C32 {
        C32 { re: 1.0, im: 0.0 }
    }
    fn negated(self) -> C32 {
        C32 {
            re: -self.re,
            im: -self.im,
        }
    }
    fn conjugated(self) -> C32 {
        conj32(self)
    }
}

impl Unit for C64 {
    fn zero() -> C64 {
        C64 { re: 0.0, im: 0.0 }
    }
    fn one() -> C64 {
        C64 { re: 1.0, im: 0.0 }
    }
    fn negated(self) -> C64 {
        C64 {
            re: -self.re,
            im: -self.im,
        }
    }
    fn conjugated(self) -> C64 {
        conj64(self)
    }
}

// --- harness helpers --------------------------------------------------------

fn sentinel(session: &Session, bytes: usize) -> Buf {
    gpu::written(session, &vec![SENTINEL; bytes])
}

/// Element index of one logical position, from the layout's own strides.
fn index_of(layout: &Layout, coords: &[i64]) -> usize {
    let mut index = layout.offset();
    for (axis, &coordinate) in coords.iter().enumerate() {
        index += coordinate * layout.strides()[axis];
    }
    assert!(index >= 0);
    index as usize
}

fn at<T: Copy>(buffer: &[T], layout: &Layout, coords: &[i64]) -> T {
    buffer[index_of(layout, coords)]
}

fn set<T: Copy>(buffer: &mut [T], layout: &Layout, coords: &[i64], value: T) {
    let index = index_of(layout, coords);
    buffer[index] = value;
}

/// Element indices a layout addresses, in flat logical order.
fn view_indices(layout: &Layout) -> Vec<i64> {
    let extents = layout.extents();
    let count: i64 = extents.iter().product();
    (0..count)
        .map(|mut flat| {
            let mut index = layout.offset();
            for axis in 0..extents.len() {
                index += (flat % extents[axis]) * layout.strides()[axis];
                flat /= extents[axis];
            }
            assert!(index >= 0);
            index
        })
        .collect()
}

/// Elements the layout can reach, i.e. the allocation length the facade is told.
fn span(layout: &Layout) -> usize {
    if layout.element_count().unwrap() == 0 {
        return 0;
    }
    let min = layout.min_reachable_offset().unwrap();
    assert!(min >= 0, "the test layouts start at or above zero");
    layout.max_reachable_offset().unwrap() as usize + 1
}

fn place<T: Copy>(buffer: &mut [T], layout: &Layout, values: &[T]) {
    assert_eq!(values.len(), layout.element_count().unwrap() as usize);
    for (&index, &value) in view_indices(layout).iter().zip(values) {
        buffer[index as usize] = value;
    }
}

fn pick<T: Copy>(buffer: &[T], layout: &Layout) -> Vec<T> {
    view_indices(layout)
        .into_iter()
        .map(|index| buffer[index as usize])
        .collect()
}

/// A buffer of the layout's reachable span holding `values` at its logical
/// positions; every other reachable element is the zero of the payload type.
fn operand<T: Unit>(session: &Session, layout: &Layout, values: &[T]) -> Buf {
    let mut host = vec![T::zero(); span(layout)];
    place(&mut host, layout, values);
    upload(session, &host)
}

/// An in-place operand: `values` at the layout's logical positions, the sentinel
/// in every hole and in a guard region behind the span, so the entry must leave
/// both untouched.
fn guarded_operand<T: Copy>(session: &Session, layout: &Layout, values: &[T]) -> Buf {
    let element = std::mem::size_of::<T>();
    let mut bytes = vec![SENTINEL; span(layout) * element + GUARD];
    for (&index, value) in view_indices(layout).iter().zip(values) {
        let offset = index as usize * element;
        bytes[offset..offset + element].copy_from_slice(as_bytes(std::slice::from_ref(value)));
    }
    gpu::written(session, &bytes)
}

/// A sentinel-filled destination sized for the layout's span, plus that span.
fn destination(session: &Session, layout: &Layout, dtype: DType) -> (Buf, i64) {
    let elements = span(layout);
    (
        sentinel(session, elements * dtype.size() + GUARD),
        elements as i64,
    )
}

/// Elements reserved for a logically empty output, so a stray write into it is
/// visible instead of landing in a one-byte allocation.
const EMPTY_ALLOC: usize = 5;

/// A multi-output destination: an empty layout still gets a sentinel allocation,
/// while the planner is told the reachable span (zero for an empty layout).
fn guarded_destination(session: &Session, layout: &Layout, dtype: DType) -> (Buf, i64) {
    let elements = span(layout);
    let allocation = if elements == 0 { EMPTY_ALLOC } else { elements };
    (
        sentinel(session, allocation * dtype.size() + GUARD),
        elements as i64,
    )
}

fn host_of<T: Copy>(bytes: &[u8], elements: usize) -> Vec<T> {
    from_bytes(&bytes[..elements * std::mem::size_of::<T>()])
}

fn assert_guard(bytes: &[u8], payload: usize, what: &str) {
    assert!(
        bytes[payload..].iter().all(|&byte| byte == SENTINEL),
        "{what}: the guard region behind the logical output was written"
    );
}

/// Bit-exact element comparison: a NaN payload and a zero sign must survive the
/// copy unchanged, so only the raw bytes may be compared.
fn assert_bits<T: Copy + std::fmt::Debug>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            as_bytes(std::slice::from_ref(actual)),
            as_bytes(std::slice::from_ref(expected)),
            "{what}[{index}]: observed {actual:?}, expected {expected:?}"
        );
    }
}

/// Assert the holes and the guard behind the span still hold the sentinel, and
/// return the values at the layout's logical positions.
fn pick_untouched<T: Copy>(bytes: &[u8], layout: &Layout, what: &str) -> Vec<T> {
    let element = std::mem::size_of::<T>();
    let elements = span(layout);
    let addresses = view_indices(layout);
    let mut addressed = vec![false; elements];
    for &index in &addresses {
        addressed[index as usize] = true;
    }
    for (index, &hit) in addressed.iter().enumerate() {
        assert!(
            hit || bytes[index * element..(index + 1) * element]
                .iter()
                .all(|&byte| byte == SENTINEL),
            "{what}: the hole at element {index} was written"
        );
    }
    assert_guard(bytes, elements * element, what);
    pick(&host_of::<T>(bytes, elements), layout)
}

/// Bit-exact comparison of a read-back image against the expected logical values.
fn assert_image<T: Copy + std::fmt::Debug>(
    bytes: &[u8],
    layout: &Layout,
    expected: &[T],
    what: &str,
) {
    assert_bits(&pick_untouched::<T>(bytes, layout, what), expected, what);
}

/// Enqueue one entry call and assert it cost exactly one physical launch, or zero
/// when the plan is a validated empty request.
fn run(session: &Session, function: &str, plan: &Launch, slots: &mut [Slot]) {
    let before = launches();
    launch(session.function(function), plan, session.stream(), slots);
    let expected = if plan.is_empty() { 0 } else { 1 };
    assert_eq!(
        launches(),
        before + expected,
        "{function}: physical launches for grid {:?}",
        plan.grid
    );
}

/// The `one` image of a dtype, as the bytes `fill_one` must store.
fn one_image(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => as_bytes(&[1.0f32]).to_vec(),
        DType::F64 => as_bytes(&[1.0f64]).to_vec(),
        DType::I32 => as_bytes(&[1i32]).to_vec(),
        DType::I64 => as_bytes(&[1i64]).to_vec(),
        DType::Bool => vec![1u8],
        DType::C32 => as_bytes(&[C32 { re: 1.0, im: 0.0 }]).to_vec(),
        DType::C64 => as_bytes(&[C64 { re: 1.0, im: 0.0 }]).to_vec(),
    }
}

/// The first two axes swapped, with the batch axes and offset untouched.
fn swapped(layout: &Layout) -> Layout {
    let mut extents = layout.extents().to_vec();
    let mut strides = layout.strides().to_vec();
    extents.swap(0, 1);
    strides.swap(0, 1);
    Layout::new(&extents, &strides, layout.offset()).unwrap()
}

// --- entry wrappers ---------------------------------------------------------

fn fill_one_bytes(session: &Session, dtype: DType, layout: &Layout, allocation: i64) -> Vec<u8> {
    let out = sentinel(session, allocation as usize * dtype.size() + GUARD);
    let plan = plan_fill_one(dtype, layout, allocation).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(layout)),
    ];
    run(session, "t4a_cuda_fill_one", &plan, &mut slots);
    read_bytes(session, &out)
}

fn adjoint_bytes(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    v_layout: &Layout,
    v: &Buf,
) -> Vec<u8> {
    let (out, out_len) = destination(session, out_layout, dtype);
    let v_len = span(v_layout) as i64;
    let plan = plan_matrix_adjoint(dtype, out_layout, out_len, v_layout, v_len).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(v),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(v_layout)),
    ];
    run(session, "t4a_cuda_matrix_adjoint", &plan, &mut slots);
    read_bytes(session, &out)
}

fn explicit_v_bytes(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    packed_layout: &Layout,
    packed: &Buf,
) -> Vec<u8> {
    let (out, out_len) = destination(session, out_layout, dtype);
    let packed_len = span(packed_layout) as i64;
    let plan =
        plan_householder_explicit_v(dtype, out_layout, out_len, packed_layout, packed_len).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(packed),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(packed_layout)),
    ];
    run(
        session,
        "t4a_cuda_householder_explicit_v",
        &plan,
        &mut slots,
    );
    read_bytes(session, &out)
}

fn identity_shifted_bytes(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    start: i64,
) -> Vec<u8> {
    let (out, out_len) = destination(session, out_layout, dtype);
    let plan = plan_identity_shifted(dtype, out_layout, out_len, start).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::i64(start),
    ];
    run(session, "t4a_cuda_identity_shifted", &plan, &mut slots);
    read_bytes(session, &out)
}

fn qr_phase_bytes(
    session: &Session,
    dtype: DType,
    phase_layout: &Layout,
    r_layout: &Layout,
    r: &Buf,
) -> Vec<u8> {
    let (phase, phase_len) = destination(session, phase_layout, dtype);
    let r_len = span(r_layout) as i64;
    let plan = plan_qr_phase(dtype, phase_layout, phase_len, r_layout, r_len).unwrap();
    let mut slots = [
        Slot::ptr(&phase),
        Slot::ptr(r),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(phase_layout)),
        Slot::layout(&CLayout::new(r_layout)),
    ];
    run(session, "t4a_cuda_qr_phase", &plan, &mut slots);
    read_bytes(session, &phase)
}

/// In place: returns the whole `q` and `r` buffers, guard included.
#[allow(clippy::too_many_arguments)]
fn qr_apply_phase_bytes(
    session: &Session,
    dtype: DType,
    q_layout: &Layout,
    q: &Buf,
    r_layout: &Layout,
    r: &Buf,
    phase_layout: &Layout,
    phase: &Buf,
    q_start: i64,
) -> (Vec<u8>, Vec<u8>) {
    let (q_len, r_len, phase_len) = (
        span(q_layout) as i64,
        span(r_layout) as i64,
        span(phase_layout) as i64,
    );
    let plan = plan_qr_apply_phase(
        dtype,
        q_layout,
        q_len,
        r_layout,
        r_len,
        phase_layout,
        phase_len,
        q_start,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(q),
        Slot::ptr(r),
        Slot::ptr(phase),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(q_layout)),
        Slot::layout(&CLayout::new(r_layout)),
        Slot::layout(&CLayout::new(phase_layout)),
        Slot::i64(q_start),
    ];
    run(session, "t4a_cuda_qr_apply_phase", &plan, &mut slots);
    (read_bytes(session, q), read_bytes(session, r))
}

fn violation_bytes(
    session: &Session,
    dtype: DType,
    violation_layout: &Layout,
    input_layout: &Layout,
    input: &Buf,
) -> Vec<u8> {
    let (violation, violation_len) = destination(session, violation_layout, DType::I32);
    let input_len = span(input_layout) as i64;
    let plan = plan_upper_trapezoidal_violation(
        dtype,
        violation_layout,
        violation_len,
        input_layout,
        input_len,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&violation),
        Slot::ptr(input),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(violation_layout)),
        Slot::layout(&CLayout::new(input_layout)),
    ];
    run(
        session,
        "t4a_cuda_upper_trapezoidal_violation",
        &plan,
        &mut slots,
    );
    read_bytes(session, &violation)
}

#[allow(clippy::too_many_arguments)]
fn assemble_bytes(
    session: &Session,
    dtype: DType,
    packed_out_layout: &Layout,
    packed_q_layout: &Layout,
    packed_q: &Buf,
    folded_r_layout: &Layout,
    folded_r: &Buf,
    coeff_out_len: i64,
    coeff_q: &Buf,
    coeff_q_len: i64,
    factor_width: i64,
) -> (Vec<u8>, Vec<u8>) {
    let (packed_out, packed_out_len) = destination(session, packed_out_layout, dtype);
    let coeff_out = sentinel(session, coeff_out_len as usize * dtype.size() + GUARD);
    let packed_q_len = span(packed_q_layout) as i64;
    let folded_r_len = span(folded_r_layout) as i64;
    let plan = plan_householder_assemble(
        dtype,
        packed_out_layout,
        packed_out_len,
        coeff_out_len,
        packed_q_layout,
        packed_q_len,
        folded_r_layout,
        folded_r_len,
        coeff_q_len,
        factor_width,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&packed_out),
        Slot::ptr(&coeff_out),
        Slot::ptr(packed_q),
        Slot::ptr(coeff_q),
        Slot::ptr(folded_r),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(packed_out_layout)),
        Slot::layout(&CLayout::new(packed_q_layout)),
        Slot::layout(&CLayout::new(folded_r_layout)),
        Slot::i64(coeff_out_len),
        Slot::i64(factor_width),
    ];
    run(session, "t4a_cuda_householder_assemble", &plan, &mut slots);
    (
        read_bytes(session, &packed_out),
        read_bytes(session, &coeff_out),
    )
}

#[allow(clippy::too_many_arguments)]
fn lu_extract_bytes(
    session: &Session,
    dtype: DType,
    p_layout: &Layout,
    l_layout: &Layout,
    u_layout: &Layout,
    work_layout: &Layout,
    work: &Buf,
    pivots: &Buf,
    k: i64,
    batches: i64,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (p, p_len) = guarded_destination(session, p_layout, dtype);
    let (l, l_len) = guarded_destination(session, l_layout, dtype);
    let (u, u_len) = guarded_destination(session, u_layout, dtype);
    let parity = sentinel(session, batches as usize * dtype.size() + GUARD);
    let work_len = span(work_layout) as i64;
    let plan = plan_lu_extract_outputs(
        dtype,
        p_layout,
        p_len,
        l_layout,
        l_len,
        u_layout,
        u_len,
        work_layout,
        work_len,
        k * batches,
        k,
        batches,
        batches,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&p),
        Slot::ptr(&l),
        Slot::ptr(&u),
        Slot::ptr(&parity),
        Slot::ptr(work),
        Slot::ptr(pivots),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(p_layout)),
        Slot::layout(&CLayout::new(l_layout)),
        Slot::layout(&CLayout::new(u_layout)),
        Slot::layout(&CLayout::new(work_layout)),
        Slot::i64(k),
        Slot::i64(batches),
    ];
    run(session, "t4a_cuda_lu_extract_outputs", &plan, &mut slots);
    (
        read_bytes(session, &p),
        read_bytes(session, &l),
        read_bytes(session, &u),
        read_bytes(session, &parity),
    )
}

fn lu_parity_bytes(session: &Session, dtype: DType, batches: i64, k: i64, pivots: &Buf) -> Vec<u8> {
    let parity = sentinel(session, batches as usize * dtype.size() + GUARD);
    let plan = plan_lu_parity(dtype, k * batches, batches, k).unwrap();
    let mut slots = [
        Slot::ptr(&parity),
        Slot::ptr(pivots),
        Slot::i32(dtype.tag()),
        Slot::i64(k),
        Slot::i64(batches),
    ];
    run(session, "t4a_cuda_lu_parity", &plan, &mut slots);
    read_bytes(session, &parity)
}

fn apply_pivots_bytes(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    in_layout: &Layout,
    input: &Buf,
    pivots: &Buf,
    k: i64,
    inverse: bool,
) -> Vec<u8> {
    let (out, out_len) = destination(session, out_layout, dtype);
    let in_len = span(in_layout) as i64;
    // Exactly one pivot per step of each batch, which is what the fixtures upload.
    let pivots_len = k * out_layout.extents()[2..].iter().product::<i64>();
    let plan =
        plan_lu_apply_pivots(dtype, out_layout, out_len, in_layout, in_len, pivots_len, k).unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(input),
        Slot::ptr(pivots),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(out_layout)),
        Slot::layout(&CLayout::new(in_layout)),
        Slot::i64(k),
        Slot::i32(inverse as i32),
    ];
    run(session, "t4a_cuda_lu_apply_pivots", &plan, &mut slots);
    read_bytes(session, &out)
}

// --- LU references ----------------------------------------------------------

struct LuShape {
    rows: i64,
    cols: i64,
    k: i64,
    batches: i64,
}

/// P, L, U and the parity, from the documented packed-LU contract.
fn lu_outputs_reference<T: Unit>(
    work: &[T],
    shape: &LuShape,
    pivots: &[i32],
) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>) {
    let &LuShape {
        rows,
        cols,
        k,
        batches,
    } = shape;
    let mut p = vec![T::zero(); (rows * rows * batches) as usize];
    let mut l = vec![T::zero(); (rows * k * batches) as usize];
    let mut u = vec![T::zero(); (k * cols * batches) as usize];
    let mut parity = vec![T::zero(); batches as usize];
    for batch in 0..batches {
        let pivot_at = |step: i64| pivots[(step + batch * k) as usize] as i64 - 1;
        for column in 0..rows {
            let mut final_row = column;
            for step in 0..k {
                let pivot = pivot_at(step);
                if final_row == step {
                    final_row = pivot;
                } else if final_row == pivot {
                    final_row = step;
                }
            }
            for row in 0..rows {
                let value = if final_row == row {
                    T::one()
                } else {
                    T::zero()
                };
                p[(row + rows * column + rows * rows * batch) as usize] = value;
            }
        }
        for row in 0..rows {
            for column in 0..k {
                let value = if row < column {
                    T::zero()
                } else if row == column {
                    T::one()
                } else {
                    work[(row + rows * column + rows * cols * batch) as usize]
                };
                l[(row + rows * column + rows * k * batch) as usize] = value;
            }
        }
        for row in 0..k {
            for column in 0..cols {
                let value = if row <= column {
                    work[(row + rows * column + rows * cols * batch) as usize]
                } else {
                    T::zero()
                };
                u[(row + k * column + k * cols * batch) as usize] = value;
            }
        }
        let mut sign = T::one();
        for step in 0..k {
            if pivot_at(step) != step {
                sign = sign.negated();
            }
        }
        parity[batch as usize] = sign;
    }
    (p, l, u, parity)
}

/// `out(row, column, batch) = input(source_row, column, batch)` with the pivots
/// applied in ascending order for the inverse and descending order for the
/// forward direction.
fn apply_reference<T: Unit>(input: &[T], shape: &LuShape, pivots: &[i32], inverse: bool) -> Vec<T> {
    let &LuShape {
        rows,
        cols,
        k,
        batches,
    } = shape;
    let mut out = vec![T::zero(); (rows * cols * batches) as usize];
    for batch in 0..batches {
        let pivot_at = |step: i64| pivots[(step + batch * k) as usize] as i64 - 1;
        for row in 0..rows {
            for column in 0..cols {
                let mut source_row = row;
                let steps: Vec<i64> = if inverse {
                    (0..k).collect()
                } else {
                    (0..k).rev().collect()
                };
                for step in steps {
                    let pivot = pivot_at(step);
                    if source_row == step {
                        source_row = pivot;
                    } else if source_row == pivot {
                        source_row = step;
                    }
                }
                out[(row + rows * column + rows * cols * batch) as usize] =
                    input[(source_row + rows * column + rows * cols * batch) as usize];
            }
        }
    }
    out
}

// --- 1. fill_one ------------------------------------------------------------

#[test]
fn fill_one_covers_compact_strided_all_dtypes_and_empty() {
    let session = Session::new("linalg_aux");

    // Compact, every dtype the entry serves.
    let layout = Layout::compact(&[5, 3]).unwrap();
    let elements = span(&layout);
    for dtype in DType::ALL {
        let bytes = fill_one_bytes(&session, dtype, &layout, elements as i64);
        let one = one_image(dtype);
        for index in 0..elements {
            let slice = &bytes[index * dtype.size()..(index + 1) * dtype.size()];
            assert_eq!(slice, &one[..], "fill_one {dtype:?}[{index}]");
        }
        assert_guard(
            &bytes,
            elements * dtype.size(),
            &format!("fill_one {dtype:?}"),
        );
    }

    // Strided with an offset, above one block: 59 * 17 = 1003 logical elements.
    let layout = Layout::new(&[59, 17], &[1, 64], 5).unwrap();
    let elements = span(&layout);
    let bytes = fill_one_bytes(&session, DType::F64, &layout, elements as i64);
    let mut covered = vec![false; elements];
    for index in view_indices(&layout) {
        covered[index as usize] = true;
    }
    assert_eq!(covered.iter().filter(|&&hit| hit).count(), BIG);
    let one = one_image(DType::F64);
    for index in 0..elements {
        let slice = &bytes[index * 8..(index + 1) * 8];
        if covered[index] {
            assert_eq!(slice, &one[..], "fill_one strided f64[{index}]");
        } else {
            assert!(
                slice.iter().all(|&byte| byte == SENTINEL),
                "fill_one strided f64[{index}]: an unaddressed element was written"
            );
        }
    }
    assert_guard(&bytes, elements * 8, "fill_one strided f64");

    // A validated empty plan launches nothing and writes nothing.
    let empty = Layout::new(&[0], &[1], 0).unwrap();
    let bytes = fill_one_bytes(&session, DType::I32, &empty, 0);
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));
}

// --- 2. matrix_adjoint ------------------------------------------------------

/// `out = conj?(v)` with the first two axes swapped; every output position is
/// compared byte for byte, holes and guard included.
fn adjoint_case<T: Unit>(
    session: &Session,
    dtype: DType,
    v_layout: &Layout,
    out_layout: &Layout,
    values: &[T],
    what: &str,
) {
    let mut host = vec![T::zero(); span(v_layout)];
    place(&mut host, v_layout, values);
    let expected: Vec<T> = pick(&host, &swapped(v_layout))
        .into_iter()
        .map(Unit::conjugated)
        .collect();
    let v = upload(session, &host);
    let bytes = adjoint_bytes(session, dtype, out_layout, v_layout, &v);
    assert_image(&bytes, out_layout, &expected, what);
}

#[test]
fn matrix_adjoint_transposes_and_conjugates_strided_and_batched_operands() {
    let session = Session::new("linalg_aux");

    // F64: asymmetric 3x5 strided operand into a strided 5x3 output. One source
    // element carries a NaN payload the copy must preserve unchanged.
    let v_layout = Layout::new(&[3, 5], &[7, 2], 3).unwrap();
    let out_layout = Layout::new(&[5, 3], &[1, 6], 2).unwrap();
    let mut v_values: Vec<f64> = (0..15).map(|index| index as f64 * 1.5 - 4.0).collect();
    v_values[4] = f64::from_bits(0x7ff8_0000_0000_1234);
    adjoint_case(
        &session,
        DType::F64,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint f64",
    );

    // F32: the same shapes.
    let v_values: Vec<f32> = (0..15).map(|index| index as f32 * 0.25 - 1.5).collect();
    adjoint_case(
        &session,
        DType::F32,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint f32",
    );

    // I32: no conjugation.
    let v_values: Vec<i32> = (0..15).map(|index| index * 7 - 30).collect();
    adjoint_case(
        &session,
        DType::I32,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint i32",
    );

    // I64: no conjugation.
    let v_values: Vec<i64> = (0..15).map(|index| index * 5 - 11).collect();
    adjoint_case(
        &session,
        DType::I64,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint i64",
    );

    // C32: conjugation of a single precision payload.
    let v_values: Vec<C32> = (0..15)
        .map(|index| C32 {
            re: index as f32 - 7.0,
            im: 0.25 * index as f32,
        })
        .collect();
    adjoint_case(
        &session,
        DType::C32,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint c32",
    );

    // C64: batched rank 3, asymmetric strided operand, conjugation.
    let v_layout = Layout::new(&[2, 3, 4], &[1, 5, 20], 1).unwrap();
    let out_layout = Layout::new(&[3, 2, 4], &[1, 7, 30], 2).unwrap();
    let v_values: Vec<C64> = (0..24)
        .map(|index| C64 {
            re: index as f64 - 9.0,
            im: 0.5 + index as f64,
        })
        .collect();
    adjoint_case(
        &session,
        DType::C64,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint c64 batched",
    );

    // Rank 4: two batch axes, so a body that dropped axes above axis 2 would read
    // the wrong plane.
    let v_layout = Layout::new(&[2, 3, 2, 2], &[1, 4, 12, 30], 1).unwrap();
    let out_layout = Layout::new(&[3, 2, 2, 2], &[1, 5, 14, 40], 2).unwrap();
    let v_values: Vec<f64> = (0..24).map(|index| index as f64 * 2.0 - 7.0).collect();
    adjoint_case(
        &session,
        DType::F64,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint f64 rank 4",
    );

    // Above one block: 16x20 batched 2, i.e. 640 output positions.
    let v_layout = Layout::new(&[16, 20, 2], &[1, 21, 500], 3).unwrap();
    let out_layout = Layout::new(&[20, 16, 2], &[1, 25, 400], 1).unwrap();
    let v_values: Vec<f64> = (0..640).map(|index| (index * 3) as f64 - 40.0).collect();
    adjoint_case(
        &session,
        DType::F64,
        &v_layout,
        &out_layout,
        &v_values,
        "adjoint f64 above one block",
    );
}

// --- 3. householder_explicit_v ----------------------------------------------

fn explicit_v_case<T: Unit>(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    packed_layout: &Layout,
    packed_values: &[T],
    what: &str,
) {
    let mut packed_host = vec![T::zero(); span(packed_layout)];
    place(&mut packed_host, packed_layout, packed_values);
    let (rows, cols) = (out_layout.extents()[0], out_layout.extents()[1]);
    let mut expected = Vec::new();
    for column in 0..cols {
        for row in 0..rows {
            expected.push(if row < column {
                T::zero()
            } else if row == column {
                T::one()
            } else {
                at(&packed_host, packed_layout, &[row, column])
            });
        }
    }
    let packed = upload(session, &packed_host);
    let bytes = explicit_v_bytes(session, dtype, out_layout, packed_layout, &packed);
    assert_image(&bytes, out_layout, &expected, what);
}

#[test]
fn householder_explicit_v_extracts_the_below_diagonal_part() {
    let session = Session::new("linalg_aux");

    // Packed 4x6 strided source into a narrower 4x3 explicit V.
    let packed_layout = Layout::new(&[4, 6], &[8, 1], 2).unwrap();
    let packed_values: Vec<f64> = (0..24).map(|index| index as f64 + 0.5).collect();

    let out_layout = Layout::new(&[4, 3], &[1, 4], 0).unwrap();
    explicit_v_case(
        &session,
        DType::F64,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v f64 narrow",
    );

    // Full width from a 4x6 packed source: the packed columns beyond the
    // narrow case must still be read below the diagonal.
    let out_layout = Layout::new(&[4, 6], &[1, 4], 0).unwrap();
    explicit_v_case(
        &session,
        DType::F64,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v f64 full",
    );

    // F32 through the same packed source.
    let packed_values: Vec<f32> = (0..24).map(|index| index as f32 * 0.5 - 3.0).collect();
    explicit_v_case(
        &session,
        DType::F32,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v f32",
    );

    // I32 through the same packed source.
    let packed_values: Vec<i32> = (0..24).map(|index| index * 3 - 20).collect();
    explicit_v_case(
        &session,
        DType::I32,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v i32",
    );

    // I64 through the same packed source.
    let packed_values: Vec<i64> = (0..24).map(|index| index * 9 - 100).collect();
    explicit_v_case(
        &session,
        DType::I64,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v i64",
    );

    // C32 through the same packed source.
    let packed_values: Vec<C32> = (0..24)
        .map(|index| C32 {
            re: index as f32,
            im: -(index as f32) * 0.5,
        })
        .collect();
    explicit_v_case(
        &session,
        DType::C32,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v c32",
    );

    // C64 and above one block: 20x18 = 360 output positions from a strided
    // 20x20 packed source.
    let out_layout = Layout::new(&[20, 18], &[1, 25], 4).unwrap();
    let packed_layout = Layout::new(&[20, 20], &[1, 24], 2).unwrap();
    let packed_values: Vec<C64> = (0..400)
        .map(|index| C64 {
            re: index as f64 + 0.5,
            im: 1.0 - index as f64,
        })
        .collect();
    explicit_v_case(
        &session,
        DType::C64,
        &out_layout,
        &packed_layout,
        &packed_values,
        "explicit_v c64 above one block",
    );
}

// --- 4. identity_shifted ----------------------------------------------------

fn identity_shifted_case<T: Unit>(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    start: i64,
    what: &str,
) {
    let (rows, cols) = (out_layout.extents()[0], out_layout.extents()[1]);
    let count = out_layout.element_count().unwrap();
    let mut expected = Vec::new();
    for flat in 0..count {
        let row = flat % rows;
        let column = (flat / rows) % cols;
        expected.push(if row == start + column {
            T::one()
        } else {
            T::zero()
        });
    }
    let bytes = identity_shifted_bytes(session, dtype, out_layout, start);
    assert_image(&bytes, out_layout, &expected, what);
}

#[test]
fn identity_shifted_places_the_shifted_diagonal_and_zero_complement_columns() {
    let session = Session::new("linalg_aux");

    // start = 0: the RRQR identity initialiser, in a strided output.
    let out_layout = Layout::new(&[4, 3], &[1, 5], 1).unwrap();
    identity_shifted_case::<f64>(
        &session,
        DType::F64,
        &out_layout,
        0,
        "identity_shifted start 0",
    );

    // start = 2 with 3 rows and 2 columns: the second column is entirely zero.
    let out_layout = Layout::compact(&[3, 2]).unwrap();
    identity_shifted_case::<f64>(
        &session,
        DType::F64,
        &out_layout,
        2,
        "identity_shifted start 2",
    );

    // F32.
    identity_shifted_case::<f32>(&session, DType::F32, &out_layout, 2, "identity_shifted f32");

    // I32.
    identity_shifted_case::<i32>(&session, DType::I32, &out_layout, 0, "identity_shifted i32");

    // I64.
    identity_shifted_case::<i64>(&session, DType::I64, &out_layout, 1, "identity_shifted i64");

    // C32.
    identity_shifted_case::<C32>(&session, DType::C32, &out_layout, 1, "identity_shifted c32");

    // start = 1, batched rank 3, complex payload.
    let out_layout = Layout::new(&[3, 2, 2], &[1, 4, 12], 3).unwrap();
    identity_shifted_case::<C64>(
        &session,
        DType::C64,
        &out_layout,
        1,
        "identity_shifted start 1 batched",
    );

    // Above one block: 20x14 = 280 positions, strided, with start + columns
    // past the row extent.
    let out_layout = Layout::new(&[20, 14], &[1, 24], 3).unwrap();
    identity_shifted_case::<f64>(
        &session,
        DType::F64,
        &out_layout,
        9,
        "identity_shifted f64 above one block",
    );
}

// --- 5. qr_phase ------------------------------------------------------------

fn phase32(diagonal: C32) -> C32 {
    let magnitude = diagonal.re.hypot(diagonal.im);
    if magnitude == 0.0 {
        C32 { re: 1.0, im: 0.0 }
    } else {
        C32 {
            re: diagonal.re / magnitude,
            im: diagonal.im / magnitude,
        }
    }
}

fn phase64(diagonal: C64) -> C64 {
    let magnitude = diagonal.re.hypot(diagonal.im);
    if magnitude == 0.0 {
        C64 { re: 1.0, im: 0.0 }
    } else {
        C64 {
            re: diagonal.re / magnitude,
            im: diagonal.im / magnitude,
        }
    }
}

/// Relative closeness for the phase arithmetic, which produces fresh values. A
/// NaN phase is only equal to a NaN phase; the payload is not part of the
/// contract there.
fn close64(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if expected == 0.0 {
        return actual == 0.0;
    }
    (actual - expected).abs() <= tolerance * expected.abs()
}

fn close32(actual: f32, expected: f32, tolerance: f32) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if expected == 0.0 {
        return actual == 0.0;
    }
    (actual - expected).abs() <= tolerance * expected.abs()
}

fn assert_phase64(actual: &[C64], expected: &[C64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close64(actual.re, expected.re, 1e-14) && close64(actual.im, expected.im, 1e-14),
            "{what}[{index}]: observed {actual:?}, expected {expected:?}"
        );
    }
}

fn assert_phase32(actual: &[C32], expected: &[C32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close32(actual.re, expected.re, 1e-6) && close32(actual.im, expected.im, 1e-6),
            "{what}[{index}]: observed {actual:?}, expected {expected:?}"
        );
    }
}

/// Every component exactly `0` or `+-1`, so multiplying by it is a sign flip or an
/// axis swap and the product must reproduce the operand's bytes.
fn exact64(phase: C64) -> bool {
    (phase.re == 0.0 || phase.re.abs() == 1.0) && (phase.im == 0.0 || phase.im.abs() == 1.0)
}

fn exact32(phase: C32) -> bool {
    (phase.re == 0.0 || phase.re.abs() == 1.0) && (phase.im == 0.0 || phase.im.abs() == 1.0)
}

/// Compare one in-place `qr_apply_phase` image. `phase` maps a flat logical position
/// to the phase entry that scales it, or `None` where the position is untouched (a
/// complement column): an untouched position must keep its bytes, an exact product
/// must match bit for bit, and only an inexact product is compared with a tolerance.
fn assert_applied64(
    actual: &[C64],
    expected: &[C64],
    phase: impl Fn(usize) -> Option<C64>,
    what: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let bitwise = phase(index).is_none_or(exact64);
        if bitwise {
            assert_eq!(
                as_bytes(std::slice::from_ref(got)),
                as_bytes(std::slice::from_ref(want)),
                "{what}[{index}]: observed {got:?}, expected {want:?}"
            );
        } else {
            assert!(
                close64(got.re, want.re, 1e-15) && close64(got.im, want.im, 1e-15),
                "{what}[{index}]: observed {got:?}, expected {want:?}"
            );
        }
    }
}

fn assert_applied32(
    actual: &[C32],
    expected: &[C32],
    phase: impl Fn(usize) -> Option<C32>,
    what: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let bitwise = phase(index).is_none_or(exact32);
        if bitwise {
            assert_eq!(
                as_bytes(std::slice::from_ref(got)),
                as_bytes(std::slice::from_ref(want)),
                "{what}[{index}]: observed {got:?}, expected {want:?}"
            );
        } else {
            assert!(
                close32(got.re, want.re, 1e-6) && close32(got.im, want.im, 1e-6),
                "{what}[{index}]: observed {got:?}, expected {want:?}"
            );
        }
    }
}

#[test]
fn qr_phase_matches_the_documented_phase_rules_bit_for_bit() {
    let session = Session::new("linalg_aux");

    // F64: 7 diagonals per batch covering positive, negative, +0, -0, zero,
    // negative and NaN; every off-diagonal entry is a poison -9 that would flip
    // a phase read from the wrong coordinate.
    let r_layout = Layout::new(&[7, 7, 2], &[1, 9, 64], 1).unwrap();
    let phase_layout = Layout::new(&[7, 2], &[1, 8], 2).unwrap();
    let diagonals: [[f64; 7]; 2] = [
        [3.0, -2.0, 0.0, -0.0, 1.0, -5.0, f64::NAN],
        [-3.0, 2.0, -0.0, 0.0, -1.0, 5.0, f64::NAN],
    ];
    let mut r_host = vec![-9.0f64; span(&r_layout)];
    for batch in 0..2i64 {
        for index in 0..7i64 {
            r_host[index_of(&r_layout, &[index, index, batch])] =
                diagonals[batch as usize][index as usize];
        }
    }
    let r = upload(&session, &r_host);
    let mut expected = Vec::new();
    for batch in 0..2 {
        for index in 0..7 {
            let diagonal = diagonals[batch][index];
            expected.push(if diagonal < 0.0 { -1.0f64 } else { 1.0 });
        }
    }
    let bytes = qr_phase_bytes(&session, DType::F64, &phase_layout, &r_layout, &r);
    assert_image(&bytes, &phase_layout, &expected, "qr_phase f64");

    // F32 through the same real rule, including the extreme magnitudes.
    let r_values = [2.5f32, -1.25, 0.0, -0.0, f32::NAN, -f32::MAX, -1e-45];
    let mut r_host = vec![-9.0f32; span(&r_layout)];
    for batch in 0..2i64 {
        for index in 0..7i64 {
            r_host[index_of(&r_layout, &[index, index, batch])] = r_values[index as usize];
        }
    }
    let r = upload(&session, &r_host);
    let mut expected = Vec::new();
    for _ in 0..2 {
        for &diagonal in &r_values {
            expected.push(if diagonal < 0.0 { -1.0f32 } else { 1.0 });
        }
    }
    let bytes = qr_phase_bytes(&session, DType::F32, &phase_layout, &r_layout, &r);
    assert_image(&bytes, &phase_layout, &expected, "qr_phase f32");

    // C64: exact-magnitude diagonals (3-4-5, 5-12-13 and axis-aligned), the zero
    // magnitude of (0, -0), and a NaN diagonal.
    let diagonals: [[C64; 7]; 2] = [
        [
            C64 { re: 3.0, im: 4.0 },
            C64 { re: -2.0, im: 0.0 },
            C64 { re: 0.0, im: 0.0 },
            C64 { re: 0.0, im: -0.0 },
            C64 { re: 0.0, im: 3.0 },
            C64 { re: -5.0, im: 12.0 },
            C64 {
                re: f64::NAN,
                im: 0.0,
            },
        ],
        [
            C64 { re: 0.0, im: -7.0 },
            C64 { re: 0.0, im: 7.0 },
            C64 { re: -1.0, im: 0.0 },
            C64 { re: 1.0, im: 0.0 },
            C64 { re: 12.0, im: 5.0 },
            C64 { re: -3.0, im: -4.0 },
            C64 {
                re: f64::NAN,
                im: f64::NAN,
            },
        ],
    ];
    let mut r_host = vec![C64 { re: -9.0, im: -9.0 }; span(&r_layout)];
    for batch in 0..2i64 {
        for index in 0..7i64 {
            r_host[index_of(&r_layout, &[index, index, batch])] =
                diagonals[batch as usize][index as usize];
        }
    }
    let r = upload(&session, &r_host);
    let expected: Vec<C64> = diagonals.iter().flatten().copied().map(phase64).collect();
    let bytes = qr_phase_bytes(&session, DType::C64, &phase_layout, &r_layout, &r);
    assert_phase64(
        &pick_untouched::<C64>(&bytes, &phase_layout, "qr_phase c64"),
        &expected,
        "qr_phase c64",
    );

    // C64 robustness: components whose squares overflow while the magnitude stays
    // finite (2.4e308 and 1.8e308 give 3e308), equal huge components (1e308 each
    // give 1.41e308), and subnormal magnitudes, where a naive `sqrt(re*re+im*im)`
    // would underflow to a zero magnitude.
    let robustness = [
        C64 {
            re: 1.2e308,
            im: 0.9e308,
        },
        C64 {
            re: 1.0e308,
            im: 1.0e308,
        },
        C64 {
            re: 1.0e-320,
            im: 0.0,
        },
        C64 {
            re: 3.0e-321,
            im: 4.0e-321,
        },
        C64 { re: 0.0, im: 0.0 },
        C64 {
            re: f64::NAN,
            im: 0.0,
        },
    ];
    let r_layout = Layout::compact(&[6, 6]).unwrap();
    let phase_layout = Layout::compact(&[6]).unwrap();
    let mut r_host = vec![C64 { re: -9.0, im: -9.0 }; span(&r_layout)];
    for index in 0..6i64 {
        r_host[index_of(&r_layout, &[index, index])] = robustness[index as usize];
    }
    let r = upload(&session, &r_host);
    let expected: Vec<C64> = robustness.iter().copied().map(phase64).collect();
    let bytes = qr_phase_bytes(&session, DType::C64, &phase_layout, &r_layout, &r);
    assert_phase64(
        &pick_untouched::<C64>(&bytes, &phase_layout, "qr_phase c64 robustness"),
        &expected,
        "qr_phase c64 robustness",
    );

    // C32 and above one block: 4 diagonals on each of 80 batches, i.e. 320 phase
    // entries, cycling the single-precision robustness values.
    let r_layout = Layout::new(&[4, 4, 80], &[1, 5, 30], 0).unwrap();
    let phase_layout = Layout::compact(&[4, 80]).unwrap();
    let robustness = [
        C32 {
            re: 2.0e38,
            im: 1.0e38,
        },
        C32 {
            re: 2.0e38,
            im: 2.0e38,
        },
        C32 {
            re: 1.0e-40,
            im: 0.0,
        },
        C32 {
            re: 3.0e-41,
            im: 4.0e-41,
        },
        C32 { re: 0.0, im: -0.0 },
        C32 {
            re: f32::NAN,
            im: 0.0,
        },
    ];
    let mut diagonals = vec![C32 { re: 0.0, im: 0.0 }; 4 * 80];
    for batch in 0..80usize {
        for index in 0..4usize {
            diagonals[index + 4 * batch] = robustness[(index + batch) % robustness.len()];
        }
    }
    let mut r_host = vec![C32 { re: -9.0, im: -9.0 }; span(&r_layout)];
    for batch in 0..80i64 {
        for index in 0..4i64 {
            r_host[index_of(&r_layout, &[index, index, batch])] =
                diagonals[(index + 4 * batch) as usize];
        }
    }
    let r = upload(&session, &r_host);
    let expected: Vec<C32> = diagonals.iter().copied().map(phase32).collect();
    let bytes = qr_phase_bytes(&session, DType::C32, &phase_layout, &r_layout, &r);
    assert_phase32(
        &pick_untouched::<C32>(&bytes, &phase_layout, "qr_phase c32 above one block"),
        &expected,
        "qr_phase c32 above one block",
    );
}

// --- 6. qr_apply_phase ------------------------------------------------------

#[test]
fn qr_apply_phase_scales_the_trailing_columns_and_the_rows_in_place() {
    let session = Session::new("linalg_aux");

    // F64, batched, q_start = 0: q columns 3 and 4 are past the phase entries and
    // must keep their bytes; r's rows are scaled without conjugation. Scaling by
    // +-1 is exact, so every position is compared byte for byte.
    let q_layout = Layout::new(&[4, 5, 2], &[1, 7, 40], 2).unwrap();
    let r_layout = Layout::new(&[3, 4, 2], &[1, 6, 30], 1).unwrap();
    let phase_layout = Layout::new(&[3, 2], &[1, 5], 3).unwrap();
    let q_values: Vec<f64> = (0..40).map(|index| index as f64 * 0.5 - 3.0).collect();
    let r_values: Vec<f64> = (0..24).map(|index| 2.0 - index as f64).collect();
    let phase_values: Vec<f64> = vec![-1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
    let q_expected: Vec<f64> = (0..40)
        .map(|flat| {
            let column = (flat / 4) % 5;
            let batch = flat / 20;
            if column < 3 {
                q_values[flat] * phase_values[column + 3 * batch]
            } else {
                q_values[flat]
            }
        })
        .collect();
    let r_expected: Vec<f64> = (0..24)
        .map(|flat| r_values[flat] * phase_values[(flat % 3) + 3 * (flat / 12)])
        .collect();
    let q = guarded_operand(&session, &q_layout, &q_values);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::F64,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        0,
    );
    assert_image(&q_bytes, &q_layout, &q_expected, "apply_phase f64 q");
    assert_image(&r_bytes, &r_layout, &r_expected, "apply_phase f64 r");

    // F32, above one block and with R larger than Q: q is 20x24 = 480 positions,
    // r is 20x40 = 800, so the domain is r's and q columns 20..23 are complement
    // columns past the 20 phase entries.
    let q_layout = Layout::new(&[20, 24], &[1, 21], 2).unwrap();
    let r_layout = Layout::new(&[20, 40], &[1, 21], 1).unwrap();
    let phase_layout = Layout::compact(&[20]).unwrap();
    let q_values: Vec<f32> = (0..480).map(|index| 0.5 * index as f32 - 60.0).collect();
    let r_values: Vec<f32> = (0..800).map(|index| 7.0 - index as f32).collect();
    let phase_values: Vec<f32> = (0..20)
        .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let q_expected: Vec<f32> = (0..480)
        .map(|flat| {
            let column = flat / 20;
            if column < 20 {
                q_values[flat] * phase_values[column]
            } else {
                q_values[flat]
            }
        })
        .collect();
    let r_expected: Vec<f32> = (0..800)
        .map(|flat| r_values[flat] * phase_values[flat % 20])
        .collect();
    let q = guarded_operand(&session, &q_layout, &q_values);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::F32,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        0,
    );
    assert_image(&q_bytes, &q_layout, &q_expected, "apply_phase f32 q");
    assert_image(&r_bytes, &r_layout, &r_expected, "apply_phase f32 r");

    // C64, batched and strided, q_start = 2: only q column 0 is scaled, and r is
    // scaled by the conjugated phase. Complex products are fresh arithmetic, so
    // only they use a tolerance; holes and guard are still byte-exact.
    let q_layout = Layout::new(&[4, 5, 2], &[1, 7, 40], 2).unwrap();
    let r_layout = Layout::new(&[3, 4, 2], &[1, 6, 30], 1).unwrap();
    let phase_layout = Layout::new(&[3, 2], &[1, 5], 3).unwrap();
    let phase_values: Vec<C64> = vec![
        C64 { re: 0.6, im: 0.8 },
        C64 { re: -1.0, im: 0.0 },
        C64 { re: 0.0, im: 1.0 },
        C64 { re: 0.0, im: -1.0 },
        C64 { re: -0.6, im: 0.8 },
        C64 { re: 1.0, im: 0.0 },
    ];
    let q_values: Vec<C64> = (0..40)
        .map(|index| C64 {
            re: index as f64 * 0.25 - 2.0,
            im: 1.5 - index as f64 * 0.125,
        })
        .collect();
    let r_values: Vec<C64> = (0..24)
        .map(|index| C64 {
            re: index as f64 + 1.0,
            im: -(index as f64) * 0.5,
        })
        .collect();
    let q_expected: Vec<C64> = (0..40)
        .map(|flat| {
            let column = (flat / 4) % 5;
            let batch = flat / 20;
            let index = 2 + column;
            if index < 3 {
                cmul64(q_values[flat], phase_values[index + 3 * batch])
            } else {
                q_values[flat]
            }
        })
        .collect();
    let r_expected: Vec<C64> = (0..24)
        .map(|flat| {
            let (row, batch) = (flat % 3, flat / 12);
            cmul64(r_values[flat], conj64(phase_values[row + 3 * batch]))
        })
        .collect();
    let q = guarded_operand(&session, &q_layout, &q_values);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::C64,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        2,
    );
    let q_got = pick_untouched::<C64>(&q_bytes, &q_layout, "apply_phase c64 q");
    assert_applied64(
        &q_got,
        &q_expected,
        |flat| {
            let column = (flat / 4) % 5;
            let index = 2 + column;
            (index < 3).then(|| phase_values[index + 3 * (flat / 20)])
        },
        "apply_phase c64 q",
    );
    let r_got = pick_untouched::<C64>(&r_bytes, &r_layout, "apply_phase c64 r");
    assert_applied64(
        &r_got,
        &r_expected,
        |flat| Some(phase_values[(flat % 3) + 3 * (flat / 12)]),
        "apply_phase c64 r",
    );

    // C32, q_start = 1: q columns 0 and 1 are scaled by phase 1 and 2, columns 2
    // and 3 are complement columns, and every r row is scaled by the conjugate.
    let q_layout = Layout::compact(&[3, 4]).unwrap();
    let r_layout = Layout::compact(&[3, 3]).unwrap();
    let phase_layout = Layout::compact(&[3]).unwrap();
    let phase_values: Vec<C32> = vec![
        C32 { re: -1.0, im: 0.0 },
        C32 { re: 0.0, im: 1.0 },
        C32 { re: 1.0, im: 0.0 },
    ];
    let q_values: Vec<C32> = (0..12)
        .map(|index| C32 {
            re: index as f32 * 0.5 - 2.0,
            im: 1.0 - index as f32 * 0.25,
        })
        .collect();
    let r_values: Vec<C32> = (0..9)
        .map(|index| C32 {
            re: index as f32 + 1.0,
            im: -(index as f32),
        })
        .collect();
    let q_expected: Vec<C32> = (0..12)
        .map(|flat| {
            let column = flat / 3;
            let index = 1 + column;
            if index < 3 {
                cmul32(q_values[flat], phase_values[index])
            } else {
                q_values[flat]
            }
        })
        .collect();
    let r_expected: Vec<C32> = (0..9)
        .map(|flat| cmul32(r_values[flat], conj32(phase_values[flat % 3])))
        .collect();
    let q = guarded_operand(&session, &q_layout, &q_values);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::C32,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        1,
    );
    let q_got = pick_untouched::<C32>(&q_bytes, &q_layout, "apply_phase c32 q");
    assert_applied32(
        &q_got,
        &q_expected,
        |flat| {
            let index = 1 + flat / 3;
            (index < 3).then(|| phase_values[index])
        },
        "apply_phase c32 q",
    );
    let r_got = pick_untouched::<C32>(&r_bytes, &r_layout, "apply_phase c32 r");
    assert_applied32(
        &r_got,
        &r_expected,
        |flat| Some(phase_values[flat % 3]),
        "apply_phase c32 r",
    );

    // C64 with a phase array that holds a signed zero and a NaN payload at an index
    // no thread may read: q_start = 4 scales q columns 0 and 1 (phase 4 and 5),
    // phase 3 is the NaN no column and no r row (r has 3 rows) may use. The
    // complement columns carry a NaN payload, a negative zero and a zero component,
    // so an off-by-one in either bound shows up in the bytes.
    let phase_values: Vec<C64> = vec![
        C64 { re: 1.0, im: -0.0 },
        C64 { re: -1.0, im: 0.0 },
        C64 { re: 0.0, im: -1.0 },
        C64 {
            re: f64::from_bits(0x7ff8_0000_dead_beef),
            im: 0.0,
        },
        C64 { re: -1.0, im: -0.0 },
        C64 { re: 0.0, im: 1.0 },
    ];
    let phase_layout = Layout::compact(&[6]).unwrap();
    let q_layout = Layout::new(&[3, 6], &[1, 5], 2).unwrap();
    let r_layout = Layout::new(&[3, 4], &[1, 3], 0).unwrap();
    let q_values: Vec<C64> = (0..18)
        .map(|index| {
            let column = index / 3;
            C64 {
                re: if column >= 2 && index % 3 == 0 {
                    -0.0
                } else {
                    index as f64 * 0.25 - 2.0
                },
                im: if column == 2 && index % 3 == 1 {
                    f64::from_bits(0x7ff0_0000_0000_0bad)
                } else if column >= 3 && index % 3 == 2 {
                    -0.0
                } else {
                    1.5 - index as f64 * 0.125
                },
            }
        })
        .collect();
    let r_values: Vec<C64> = (0..12)
        .map(|index| C64 {
            re: index as f64 + 1.0,
            im: -(index as f64) * 0.5 - 0.5,
        })
        .collect();
    let q_expected: Vec<C64> = (0..18)
        .map(|flat| {
            let index = 4 + flat / 3;
            match phase_values.get(index) {
                Some(&entry) => cmul64(q_values[flat], entry),
                None => q_values[flat],
            }
        })
        .collect();
    let r_expected: Vec<C64> = (0..12)
        .map(|flat| cmul64(r_values[flat], conj64(phase_values[flat % 3])))
        .collect();
    let q = guarded_operand(&session, &q_layout, &q_values);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::C64,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        4,
    );
    let q_got = pick_untouched::<C64>(&q_bytes, &q_layout, "apply_phase c64 signed zero q");
    assert_applied64(
        &q_got,
        &q_expected,
        |flat| phase_values.get(4 + flat / 3).copied(),
        "apply_phase c64 signed zero q",
    );
    let r_got = pick_untouched::<C64>(&r_bytes, &r_layout, "apply_phase c64 signed zero r");
    assert_applied64(
        &r_got,
        &r_expected,
        |flat| Some(phase_values[flat % 3]),
        "apply_phase c64 signed zero r",
    );
}

// --- 7. upper_trapezoidal_violation ------------------------------------------

#[test]
fn upper_trapezoidal_violation_flags_only_strictly_lower_nonzeros() {
    let session = Session::new("linalg_aux");

    // F32 over 64x37 (above one block): NaN and negative zeros in the upper part
    // must not be flagged, and one violation sits above the first block. A NaN
    // *below* the diagonal is non-zero (IEEE `NaN != 0`), so it violates the upper
    // trapezoidal shape and is flagged, as the pinned baseline's `input != 0` does.
    let input_layout = Layout::new(&[64, 37], &[1, 70], 3).unwrap();
    let violation_layout = Layout::compact(&[64, 37]).unwrap();
    let mut host = vec![0.0f32; span(&input_layout)];
    set(&mut host, &input_layout, &[40, 10], 5.0);
    set(&mut host, &input_layout, &[2, 0], -3.0);
    set(&mut host, &input_layout, &[0, 0], 2.0);
    set(&mut host, &input_layout, &[5, 1], 0.0);
    set(&mut host, &input_layout, &[1, 5], f32::NAN);
    set(&mut host, &input_layout, &[3, 7], -0.0);
    set(&mut host, &input_layout, &[2, 3], f32::NAN);
    set(&mut host, &input_layout, &[63, 0], -0.0);
    set(
        &mut host,
        &input_layout,
        &[7, 2],
        f32::from_bits(0x7fc0_1234),
    );
    let input = upload(&session, &host);
    let mut expected = vec![0i32; 64 * 37];
    for row in 0..64i64 {
        for column in 0..37i64 {
            let value = at(&host, &input_layout, &[row, column]);
            expected[(row + 64 * column) as usize] =
                if row > column && value != 0.0 { 1 } else { 0 };
        }
    }
    let bytes = violation_bytes(
        &session,
        DType::F32,
        &violation_layout,
        &input_layout,
        &input,
    );
    let got = pick_untouched::<i32>(&bytes, &violation_layout, "violation f32");
    assert_eq!(
        got.iter().filter(|&&value| value == 1).count(),
        3,
        "violation f32: the flags are (40, 10), (2, 0) and (7, 2)"
    );
    assert_bits(&got, &expected, "violation f32");

    // F64: a violation and a negative zero below the diagonal.
    let input_layout = Layout::compact(&[6, 5]).unwrap();
    let mut host = vec![0.0f64; span(&input_layout)];
    set(&mut host, &input_layout, &[4, 1], -2.5);
    set(&mut host, &input_layout, &[5, 4], -0.0);
    set(&mut host, &input_layout, &[1, 4], 3.0);
    let input = upload(&session, &host);
    let bytes = violation_bytes(&session, DType::F64, &input_layout, &input_layout, &input);
    let mut expected = vec![0i32; 30];
    expected[4 + 6 * 1] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &input_layout, "violation f64"),
        &expected,
        "violation f64",
    );

    // I32: the same shapes, integers.
    let input_layout = Layout::compact(&[4, 4]).unwrap();
    let mut host = vec![0i32; span(&input_layout)];
    set(&mut host, &input_layout, &[3, 1], -9);
    set(&mut host, &input_layout, &[2, 3], 5);
    let input = upload(&session, &host);
    let bytes = violation_bytes(&session, DType::I32, &input_layout, &input_layout, &input);
    let mut expected = vec![0i32; 16];
    expected[3 + 4 * 1] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &input_layout, "violation i32"),
        &expected,
        "violation i32",
    );

    // I64: integers, with a violation and a zero below the diagonal.
    let input_layout = Layout::compact(&[5, 4]).unwrap();
    let mut host = vec![0i64; 20];
    set(&mut host, &input_layout, &[3, 1], 7);
    set(&mut host, &input_layout, &[4, 3], 0);
    set(&mut host, &input_layout, &[1, 3], 9);
    let input = upload(&session, &host);
    let bytes = violation_bytes(&session, DType::I64, &input_layout, &input_layout, &input);
    let mut expected = vec![0i32; 20];
    expected[3 + 5 * 1] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &input_layout, "violation i64"),
        &expected,
        "violation i64",
    );

    // Bool: a one byte below the diagonal is a violation, a zero byte is not, and
    // an upper-triangle one is not either.
    let input_layout = Layout::new(&[4, 3], &[1, 5], 1).unwrap();
    let mut host = vec![0u8; span(&input_layout)];
    set(&mut host, &input_layout, &[2, 0], 1u8);
    set(&mut host, &input_layout, &[3, 2], 0u8);
    set(&mut host, &input_layout, &[0, 2], 1u8);
    let input = upload(&session, &host);
    let bytes = violation_bytes(&session, DType::Bool, &input_layout, &input_layout, &input);
    let mut expected = vec![0i32; 12];
    expected[2 + 4 * 0] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &input_layout, "violation bool"),
        &expected,
        "violation bool",
    );

    // C32: a complex value with a zero real part counts, a zero complex does not,
    // and a non-zero in the upper part stays unflagged.
    let input_layout = Layout::new(&[5, 4], &[1, 6], 0).unwrap();
    let mut host = vec![C32 { re: 0.0, im: 0.0 }; span(&input_layout)];
    set(&mut host, &input_layout, &[3, 0], C32 { re: 0.0, im: 2.5 });
    set(&mut host, &input_layout, &[4, 2], C32 { re: 0.0, im: -0.0 });
    let input = upload(&session, &host);
    let bytes = violation_bytes(&session, DType::C32, &input_layout, &input_layout, &input);
    let mut expected = vec![0i32; 20];
    expected[3 + 5 * 0] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &input_layout, "violation c32"),
        &expected,
        "violation c32",
    );

    // C64: a complex value with a zero real part counts, a zero complex does not,
    // and a non-zero in the upper part stays unflagged.
    let input_layout = Layout::new(&[9, 6], &[1, 10], 2).unwrap();
    let violation_layout = Layout::new(&[9, 6], &[1, 10], 0).unwrap();
    let mut host = vec![C64 { re: 0.0, im: 0.0 }; span(&input_layout)];
    set(&mut host, &input_layout, &[5, 0], C64 { re: 0.0, im: 2.0 });
    set(&mut host, &input_layout, &[4, 1], C64 { re: 0.0, im: -0.0 });
    set(&mut host, &input_layout, &[6, 3], C64 { re: -1.0, im: 0.0 });
    set(&mut host, &input_layout, &[1, 4], C64 { re: 0.0, im: 3.0 });
    let input = upload(&session, &host);
    let bytes = violation_bytes(
        &session,
        DType::C64,
        &violation_layout,
        &input_layout,
        &input,
    );
    let mut expected = vec![0i32; 54];
    expected[(5 + 9 * 0) as usize] = 1;
    expected[(6 + 9 * 3) as usize] = 1;
    assert_bits(
        &pick_untouched::<i32>(&bytes, &violation_layout, "violation c64"),
        &expected,
        "violation c64",
    );
}

// --- 8. householder_assemble ------------------------------------------------

/// `packed_out` and `coeff_out` from the two factor regions, with `factor_width`
/// possibly larger than both matrix dimensions and `coeff_out` possibly longer
/// than `packed_out`.
#[allow(clippy::too_many_arguments)]
fn assemble_case<T: Unit>(
    session: &Session,
    dtype: DType,
    out_layout: &Layout,
    folded_r_layout: &Layout,
    folded_r_values: &[T],
    packed_q_layout: &Layout,
    packed_q_values: &[T],
    coeff_out_len: i64,
    coeff_q_values: &[T],
    factor_width: i64,
    what: &str,
) {
    let mut folded_r_host = vec![T::zero(); span(folded_r_layout)];
    place(&mut folded_r_host, folded_r_layout, folded_r_values);
    let mut packed_q_host = vec![T::zero(); span(packed_q_layout)];
    place(&mut packed_q_host, packed_q_layout, packed_q_values);
    let (rows, cols) = (out_layout.extents()[0], out_layout.extents()[1]);
    let mut expected = Vec::new();
    for column in 0..cols {
        for row in 0..rows {
            expected.push(if row < factor_width && row <= column {
                at(&folded_r_host, folded_r_layout, &[row, column])
            } else if column < factor_width && row > column {
                at(&packed_q_host, packed_q_layout, &[row, column])
            } else {
                T::zero()
            });
        }
    }
    let mut coeff_expected = vec![T::zero(); coeff_out_len as usize];
    for (position, value) in coeff_expected.iter_mut().enumerate() {
        if (position as i64) < factor_width {
            *value = coeff_q_values[position];
        }
    }
    let folded_r = upload(session, &folded_r_host);
    let packed_q = upload(session, &packed_q_host);
    let coeff_q = upload(session, coeff_q_values);
    let (packed_bytes, coeff_bytes) = assemble_bytes(
        session,
        dtype,
        out_layout,
        packed_q_layout,
        &packed_q,
        folded_r_layout,
        &folded_r,
        coeff_out_len,
        &coeff_q,
        coeff_q_values.len() as i64,
        factor_width,
    );
    assert_image(&packed_bytes, out_layout, &expected, what);
    let coeff_what = format!("{what} coeff");
    assert_bits(
        &host_of::<T>(&coeff_bytes, coeff_out_len as usize),
        &coeff_expected,
        &coeff_what,
    );
    assert_guard(
        &coeff_bytes,
        coeff_out_len as usize * std::mem::size_of::<T>(),
        &coeff_what,
    );
}

#[test]
fn householder_assemble_merges_the_folded_r_and_packed_q_regions() {
    let session = Session::new("linalg_aux");

    // factor_width 0: every source is read for nothing, the output is all zeros
    // and the coefficient tail is empty.
    let out_layout = Layout::compact(&[4, 6]).unwrap();
    let tiny = Layout::compact(&[1, 1]).unwrap();
    assemble_case(
        &session,
        DType::F64,
        &out_layout,
        &tiny,
        &[42.0f64],
        &tiny,
        &[-42.0f64],
        4,
        &[7.0f64],
        0,
        "assemble fw 0",
    );

    // factor_width 1: a one-row folded_r and a one-column packed_q, both narrower
    // than the output, and a coefficient tail of length 1.
    let folded_r_1 = Layout::compact(&[1, 6]).unwrap();
    let packed_q_1 = Layout::compact(&[4, 1]).unwrap();
    assemble_case(
        &session,
        DType::F64,
        &out_layout,
        &folded_r_1,
        &[11.0f64, 12.0, 13.0, 14.0, 15.0, 16.0],
        &packed_q_1,
        &[21.0f64, 22.0, 23.0, 24.0],
        2,
        &[31.0f64],
        1,
        "assemble fw 1",
    );

    // factor_width 4 with a complex, strided output, a 4x4 packed_q and a
    // four-entry coefficient prefix in a six-entry coefficient array.
    let out_layout = Layout::new(&[4, 6], &[1, 5], 3).unwrap();
    let folded_r_4 = Layout::new(&[4, 6], &[1, 7], 1).unwrap();
    let packed_q_4 = Layout::new(&[4, 4], &[1, 6], 0).unwrap();
    let folded_r_values: Vec<C64> = (0..24)
        .map(|index| C64 {
            re: index as f64 + 0.25,
            im: -(index as f64),
        })
        .collect();
    let packed_q_values: Vec<C64> = (0..16)
        .map(|index| C64 {
            re: 100.0 - index as f64,
            im: 0.5 * index as f64,
        })
        .collect();
    let coeff_values: Vec<C64> = vec![
        C64 { re: 1.0, im: 1.0 },
        C64 { re: 2.0, im: -2.0 },
        C64 { re: 3.0, im: 0.5 },
        C64 { re: 4.0, im: 0.0 },
    ];
    assemble_case(
        &session,
        DType::C64,
        &out_layout,
        &folded_r_4,
        &folded_r_values,
        &packed_q_4,
        &packed_q_values,
        6,
        &coeff_values,
        4,
        "assemble c64 fw 4",
    );

    // F32, above one block, with factor_width 24 (larger than both 20-row/20-column
    // dimensions) and a 500-entry coefficient array longer than the 400-entry
    // output: the domain is the coefficient length.
    let out_layout = Layout::compact(&[20, 20]).unwrap();
    let full = Layout::compact(&[20, 20]).unwrap();
    let folded_r_values: Vec<f32> = (0..400).map(|index| index as f32 + 0.5).collect();
    let packed_q_values: Vec<f32> = (0..400).map(|index| 200.0 - index as f32).collect();
    let coeff_values: Vec<f32> = (0..24).map(|index| 10.0 - index as f32).collect();
    assemble_case(
        &session,
        DType::F32,
        &out_layout,
        &full,
        &folded_r_values,
        &full,
        &packed_q_values,
        500,
        &coeff_values,
        24,
        "assemble f32 above one block",
    );

    // I32 with factor_width 2.
    let out_layout = Layout::compact(&[4, 5]).unwrap();
    assemble_case(
        &session,
        DType::I32,
        &out_layout,
        &Layout::compact(&[2, 5]).unwrap(),
        &(0..10).map(|index| index * 3 - 12).collect::<Vec<i32>>(),
        &Layout::compact(&[4, 2]).unwrap(),
        &(0..8).map(|index| 50 - index).collect::<Vec<i32>>(),
        3,
        &[7i32, -8],
        2,
        "assemble i32",
    );

    // I64 with factor_width 3.
    let out_layout = Layout::compact(&[3, 4]).unwrap();
    assemble_case(
        &session,
        DType::I64,
        &out_layout,
        &Layout::compact(&[3, 4]).unwrap(),
        &(0..12)
            .map(|index| index as i64 * 7 - 30)
            .collect::<Vec<i64>>(),
        &Layout::compact(&[3, 3]).unwrap(),
        &(0..9).map(|index| index as i64 - 4).collect::<Vec<i64>>(),
        4,
        &[5i64, -6, 7],
        3,
        "assemble i64",
    );

    // C32 with factor_width 2 and a strided output.
    let out_layout = Layout::new(&[3, 3], &[1, 5], 2).unwrap();
    let folded_r_values: Vec<C32> = (0..6)
        .map(|index| C32 {
            re: index as f32,
            im: -(index as f32),
        })
        .collect();
    let packed_q_values: Vec<C32> = (0..6)
        .map(|index| C32 {
            re: 9.0 - index as f32,
            im: 0.5,
        })
        .collect();
    let coeff_values: Vec<C32> = vec![C32 { re: 1.0, im: 0.0 }, C32 { re: 0.0, im: -1.0 }];
    assemble_case(
        &session,
        DType::C32,
        &out_layout,
        &Layout::compact(&[2, 3]).unwrap(),
        &folded_r_values,
        &Layout::compact(&[3, 2]).unwrap(),
        &packed_q_values,
        2,
        &coeff_values,
        2,
        "assemble c32",
    );
}

// --- 9. lu_extract_outputs and lu_parity ------------------------------------

fn lu_case<T: Unit>(
    session: &Session,
    dtype: DType,
    shape: &LuShape,
    work_layout: &Layout,
    work_values: &[T],
    pivots: &[i32],
) {
    let (rows, cols, k, batches) = (shape.rows, shape.cols, shape.k, shape.batches);
    let (p_layout, l_layout, u_layout) = if work_layout.rank() == 2 {
        (
            Layout::compact(&[rows, rows]).unwrap(),
            Layout::compact(&[rows, k]).unwrap(),
            Layout::compact(&[k, cols]).unwrap(),
        )
    } else {
        (
            Layout::compact(&[rows, rows, batches]).unwrap(),
            Layout::compact(&[rows, k, batches]).unwrap(),
            Layout::compact(&[k, cols, batches]).unwrap(),
        )
    };
    lu_case_with(
        session,
        dtype,
        shape,
        &p_layout,
        &l_layout,
        &u_layout,
        work_layout,
        work_values,
        pivots,
    );
}

/// The same case with explicit P, L and U destination layouts, which may be strided
/// and offset independently of each other.
#[allow(clippy::too_many_arguments)]
fn lu_case_with<T: Unit>(
    session: &Session,
    dtype: DType,
    shape: &LuShape,
    p_layout: &Layout,
    l_layout: &Layout,
    u_layout: &Layout,
    work_layout: &Layout,
    work_values: &[T],
    pivots: &[i32],
) {
    let work = operand(session, work_layout, work_values);
    let pivot_buf = upload(session, pivots);
    let (k, batches) = (shape.k, shape.batches);
    let (p_bytes, l_bytes, u_bytes, parity_bytes) = lu_extract_bytes(
        session,
        dtype,
        p_layout,
        l_layout,
        u_layout,
        work_layout,
        &work,
        &pivot_buf,
        k,
        batches,
    );
    let (p, l, u, parity) = lu_outputs_reference(work_values, shape, pivots);
    assert_image(&p_bytes, p_layout, &p, "lu P");
    assert_image(&l_bytes, l_layout, &l, "lu L");
    assert_image(&u_bytes, u_layout, &u, "lu U");
    assert_bits(
        &host_of::<T>(&parity_bytes, parity.len()),
        &parity,
        "lu parity",
    );
    assert_guard(&parity_bytes, parity.len() * dtype.size(), "lu parity");

    // The standalone parity entry must agree with the extract entry.
    let standalone = lu_parity_bytes(session, dtype, batches, k, &pivot_buf);
    assert_bits(
        &host_of::<T>(&standalone, parity.len()),
        &parity,
        "lu_parity",
    );
    assert_guard(&standalone, parity.len() * dtype.size(), "lu_parity");
}

#[test]
fn lu_extract_outputs_and_parity_match_the_pivot_transpositions() {
    let session = Session::new("linalg_aux");

    // 1x1: no pivot moves, parity +1.
    let shape = LuShape {
        rows: 1,
        cols: 1,
        k: 1,
        batches: 1,
    };
    lu_case(
        &session,
        DType::F64,
        &shape,
        &Layout::compact(&[1, 1]).unwrap(),
        &[2.5f64],
        &[1],
    );

    // 2x2 with one row swap at step 0, parity -1.
    let shape = LuShape {
        rows: 2,
        cols: 2,
        k: 2,
        batches: 1,
    };
    lu_case(
        &session,
        DType::F64,
        &shape,
        &Layout::compact(&[2, 2]).unwrap(),
        &[0.5f64, 2.0, 3.0, 1.0],
        &[2, 2],
    );

    // 2x2 complex with the same single swap: the parity negation runs on the
    // complex payload.
    lu_case(
        &session,
        DType::C64,
        &shape,
        &Layout::new(&[2, 2], &[1, 3], 1).unwrap(),
        &[
            C64 { re: 0.25, im: -1.0 },
            C64 { re: 2.0, im: 0.0 },
            C64 { re: -3.0, im: 0.5 },
            C64 { re: 1.0, im: 2.0 },
        ],
        &[2, 2],
    );

    // 2x2 single precision complex: the same routes on the narrow payload.
    lu_case(
        &session,
        DType::C32,
        &shape,
        &Layout::compact(&[2, 2]).unwrap(),
        &[
            C32 { re: 0.5, im: 1.5 },
            C32 { re: -2.0, im: 0.0 },
            C32 { re: 3.0, im: -0.5 },
            C32 { re: 1.0, im: 1.0 },
        ],
        &[2, 2],
    );

    // Pivot-heavy 8x5, two batches, strided work layout and independently strided,
    // offset P, L and U destinations: a body that ignored a destination stride or
    // offset would scatter the factors over the wrong elements or the holes.
    let shape = LuShape {
        rows: 8,
        cols: 5,
        k: 5,
        batches: 2,
    };
    let work_layout = Layout::new(&[8, 5, 2], &[1, 9, 46], 2).unwrap();
    let p_layout = Layout::new(&[8, 8, 2], &[1, 11, 100], 3).unwrap();
    let l_layout = Layout::new(&[8, 5, 2], &[1, 10, 60], 2).unwrap();
    let u_layout = Layout::new(&[5, 5, 2], &[1, 7, 50], 1).unwrap();
    let work_values: Vec<f32> = (0..80)
        .map(|index| {
            let row = index % 8;
            let column = (index / 8) % 5;
            let batch = index / 40;
            (row * 7 + column * 3 + batch * 11) as f32 + 0.5
        })
        .collect();
    let pivots = [3, 1, 6, 3, 7, 1, 4, 2, 8, 5];
    lu_case_with(
        &session,
        DType::F32,
        &shape,
        &p_layout,
        &l_layout,
        &u_layout,
        &work_layout,
        &work_values,
        &pivots,
    );

    // Above one block: a full 20x20 factorisation of two batches, so P is 800
    // positions and the launch needs four blocks.
    let shape = LuShape {
        rows: 20,
        cols: 20,
        k: 20,
        batches: 2,
    };
    let work_values: Vec<f64> = (0..800)
        .map(|index| {
            let row = index % 20;
            let column = (index / 20) % 20;
            (row * 13 + column * 7) as f64 - 100.0
        })
        .collect();
    let pivots: Vec<i32> = (0..40).map(|index| ((index * 7) % 20 + 1) as i32).collect();
    lu_case(
        &session,
        DType::F64,
        &shape,
        &Layout::compact(&[20, 20, 2]).unwrap(),
        &work_values,
        &pivots,
    );

    // Above one block for the parity entry alone: 300 batches of 16 steps.
    let batches = 300i64;
    let k = 16i64;
    let pivots: Vec<i32> = (0..batches * k)
        .map(|index| ((index * 5) % k + 1) as i32)
        .collect();
    let pivot_buf = upload(&session, &pivots);
    let bytes = lu_parity_bytes(&session, DType::F64, batches, k, &pivot_buf);
    let expected: Vec<f64> = (0..batches as usize)
        .map(|batch| {
            let swaps = (0..k as usize)
                .filter(|&step| pivots[step + batch * k as usize] as i64 - 1 != step as i64)
                .count();
            if swaps % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    assert_bits(
        &host_of::<f64>(&bytes, batches as usize),
        &expected,
        "lu_parity above one block",
    );
    assert_guard(&bytes, batches as usize * 8, "lu_parity above one block");
}

// --- 10. lu_apply_pivots ----------------------------------------------------

fn apply_pivots_case<T: Unit>(
    session: &Session,
    dtype: DType,
    shape: &LuShape,
    in_layout: &Layout,
    values: &[T],
    pivots: &[i32],
) {
    let (rows, cols, batches) = (shape.rows, shape.cols, shape.batches);
    let out_layout = Layout::compact(&[rows, cols, batches]).unwrap();
    let pivot_buf = upload(session, pivots);
    let input = operand(session, in_layout, values);
    let forward = apply_reference(values, shape, pivots, false);
    let inverse = apply_reference(values, shape, pivots, true);

    let bytes = apply_pivots_bytes(
        session,
        dtype,
        &out_layout,
        in_layout,
        &input,
        &pivot_buf,
        shape.k,
        false,
    );
    assert_image(&bytes, &out_layout, &forward, "apply_pivots forward");

    let inverse_bytes = apply_pivots_bytes(
        session,
        dtype,
        &out_layout,
        in_layout,
        &input,
        &pivot_buf,
        shape.k,
        true,
    );
    assert_image(
        &inverse_bytes,
        &out_layout,
        &inverse,
        "apply_pivots inverse",
    );

    // apply(inverse) then apply(forward) restores the operand.
    let inverse_host = host_of::<T>(&inverse_bytes, span(&out_layout));
    let restored_input = upload(session, &inverse_host);
    let restored = apply_pivots_bytes(
        session,
        dtype,
        &out_layout,
        &out_layout,
        &restored_input,
        &pivot_buf,
        shape.k,
        false,
    );
    assert_image(&restored, &out_layout, values, "apply_pivots round trip");
}

#[test]
fn lu_apply_pivots_applies_the_transpositions_forwards_and_backwards() {
    let session = Session::new("linalg_aux");

    let shape = LuShape {
        rows: 4,
        cols: 3,
        k: 3,
        batches: 2,
    };
    let in_layout = Layout::new(&[4, 3, 2], &[1, 5, 30], 1).unwrap();
    let pivots = [2, 1, 4, 4, 3, 1];

    let values: Vec<f32> = (0..24).map(|index| index as f32 * 0.5 - 4.0).collect();
    apply_pivots_case(&session, DType::F32, &shape, &in_layout, &values, &pivots);

    // F64, with a NaN payload that the copy must not canonicalise.
    let mut values: Vec<f64> = (0..24).map(|index| index as f64 * 1.5 - 10.0).collect();
    values[6] = f64::from_bits(0x7ff0_0000_0000_4321);
    apply_pivots_case(&session, DType::F64, &shape, &in_layout, &values, &pivots);

    let values: Vec<i32> = (0..24).map(|index| index * 5 - 60).collect();
    apply_pivots_case(&session, DType::I32, &shape, &in_layout, &values, &pivots);

    let values: Vec<i64> = (0..24).map(|index| index as i64 * 7 - 100).collect();
    apply_pivots_case(&session, DType::I64, &shape, &in_layout, &values, &pivots);

    let values: Vec<C32> = (0..24)
        .map(|index| C32 {
            re: (index % 4) as f32 + 1.0,
            im: -((index / 4) as f32) - 0.5,
        })
        .collect();
    apply_pivots_case(&session, DType::C32, &shape, &in_layout, &values, &pivots);

    let values: Vec<C64> = (0..24)
        .map(|index| C64 {
            re: (index % 4) as f64 + 1.0,
            im: -((index / 4) as f64) - 0.5,
        })
        .collect();
    apply_pivots_case(&session, DType::C64, &shape, &in_layout, &values, &pivots);

    // A source whose rows are stored in reverse order (negative row stride and a
    // shifted offset): the permutation must index the layout, not a flat row.
    let reversed = Layout::new(&[4, 3, 2], &[-1, 5, 30], 3).unwrap();
    let values: Vec<f64> = (0..24).map(|index| index as f64 * 0.5 - 4.0).collect();
    apply_pivots_case(&session, DType::F64, &shape, &reversed, &values, &pivots);

    // Above one block: 20x20 batched 2, i.e. 800 output positions, with a full
    // 20-step pivot sequence per batch.
    let shape = LuShape {
        rows: 20,
        cols: 20,
        k: 20,
        batches: 2,
    };
    let in_layout = Layout::new(&[20, 20, 2], &[1, 21, 460], 1).unwrap();
    let values: Vec<f64> = (0..800)
        .map(|index| (index % 20) as f64 * 0.5 - index as f64 * 0.25)
        .collect();
    let pivots: Vec<i32> = (0..40)
        .map(|index| ((index * 11) % 20 + 1) as i32)
        .collect();
    apply_pivots_case(&session, DType::F64, &shape, &in_layout, &values, &pivots);
}

// --- 11. empty plans --------------------------------------------------------

#[test]
fn validated_empty_plans_launch_nothing() {
    let session = Session::new("linalg_aux");

    // fill_one over an empty layout.
    let empty = Layout::new(&[0], &[1], 0).unwrap();
    let bytes = fill_one_bytes(&session, DType::F64, &empty, 0);
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));

    // identity_shifted with a zero row extent.
    let empty = Layout::new(&[0, 3], &[1, 1], 0).unwrap();
    let bytes = identity_shifted_bytes(&session, DType::F64, &empty, 1);
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));

    // qr_phase with no phase entries.
    let r_layout = Layout::compact(&[4, 4]).unwrap();
    let r = operand(&session, &r_layout, &[1.0f64; 16]);
    let empty_phase = Layout::new(&[0], &[1], 0).unwrap();
    let bytes = qr_phase_bytes(&session, DType::F64, &empty_phase, &r_layout, &r);
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));

    // lu_parity with no batches.
    let pivots = upload(&session, &[1i32]);
    let bytes = lu_parity_bytes(&session, DType::F64, 0, 1, &pivots);
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));

    // lu_apply_pivots with a zero row extent.
    let out = Layout::new(&[0, 2], &[1, 1], 0).unwrap();
    let empty_operand = sentinel(&session, 16);
    let bytes = apply_pivots_bytes(
        &session,
        DType::F64,
        &out,
        &out,
        &empty_operand,
        &pivots,
        0,
        false,
    );
    assert!(bytes.iter().all(|&byte| byte == SENTINEL));
}

// --- 12. a zero-extent operand beside a nonempty sibling --------------------

/// A multi-output launch whose operands do not all have the same logical length: a
/// zero-extent operand is skipped, the nonempty siblings are written, and no empty
/// operand's bytes are touched.
#[test]
fn a_zero_extent_operand_is_skipped_beside_a_nonempty_sibling() {
    let session = Session::new("linalg_aux");

    // qr_apply_phase with a zero q column extent and a nonempty r: the launch still
    // runs for r (the `run` inside the wrapper asserts exactly one launch) and every
    // byte of q's allocation must still be the sentinel.
    let q_layout = Layout::new(&[3, 0, 2], &[1, 4, 8], 0).unwrap();
    let r_layout = Layout::new(&[3, 4, 2], &[1, 5, 20], 1).unwrap();
    let phase_layout = Layout::compact(&[3, 2]).unwrap();
    let phase_values: Vec<f64> = vec![-1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    let r_values: Vec<f64> = (0..24).map(|index| 3.0 - index as f64).collect();
    let r_expected: Vec<f64> = (0..24)
        .map(|flat| r_values[flat] * phase_values[(flat % 3) + 3 * (flat / 12)])
        .collect();
    let q = sentinel(&session, EMPTY_ALLOC * 8 + GUARD);
    let r = guarded_operand(&session, &r_layout, &r_values);
    let phase = operand(&session, &phase_layout, &phase_values);
    let (q_bytes, r_bytes) = qr_apply_phase_bytes(
        &session,
        DType::F64,
        &q_layout,
        &q,
        &r_layout,
        &r,
        &phase_layout,
        &phase,
        0,
    );
    assert!(
        q_bytes.iter().all(|&byte| byte == SENTINEL),
        "apply_phase empty q: the empty operand's buffer was written"
    );
    assert_image(&r_bytes, &r_layout, &r_expected, "apply_phase empty q r");

    // lu_extract_outputs with m = 4 and n = 0: p and the parity are nonempty, l and
    // u are empty, and k = min(m, n) = 0 so no pivot is read either.
    let p_layout = Layout::compact(&[4, 4, 2]).unwrap();
    let l_layout = Layout::new(&[4, 0, 2], &[1, 4, 8], 0).unwrap();
    let u_layout = Layout::new(&[0, 0, 2], &[1, 1, 1], 0).unwrap();
    let work_layout = Layout::new(&[4, 0, 2], &[1, 4, 8], 0).unwrap();
    let work = sentinel(&session, EMPTY_ALLOC * 8 + GUARD);
    let pivots = sentinel(&session, 8);
    let (p_bytes, l_bytes, u_bytes, parity_bytes) = lu_extract_bytes(
        &session,
        DType::F64,
        &p_layout,
        &l_layout,
        &u_layout,
        &work_layout,
        &work,
        &pivots,
        0,
        2,
    );
    let p_expected: Vec<f64> = (0..32)
        .map(|flat| if flat % 4 == (flat / 4) % 4 { 1.0 } else { 0.0 })
        .collect();
    assert_image(&p_bytes, &p_layout, &p_expected, "lu empty n P");
    assert_image(&l_bytes, &l_layout, &[] as &[f64], "lu empty n L");
    assert_image(&u_bytes, &u_layout, &[] as &[f64], "lu empty n U");
    assert_bits(
        &host_of::<f64>(&parity_bytes, 2),
        &[1.0, 1.0],
        "lu empty n parity",
    );
    assert_guard(&parity_bytes, 2 * 8, "lu empty n parity");
    assert!(
        read_bytes(&session, &work)
            .iter()
            .all(|&byte| byte == SENTINEL),
        "lu empty n: the empty work operand was written"
    );
}
