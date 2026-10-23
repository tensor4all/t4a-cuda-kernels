//! Paired A100 benchmark harness for this crate's ABI revision-8 entries.
//!
//! It drives the raw CUDA driver API with nvcc-compiled cubins (the harness the
//! GPU tests use, included through `#[path]`) and implements the candidate arm of
//! the frozen case list in `docs/worklogs/2026-10-21-complete-port-perf.md` §4.1,
//! so both sides submit a matched kernel sequence. The baseline arm is
//! `bench/cubecl-baseline/`.
//!
//! ```text
//! cargo run --release --features gpu-tests --example bench_kernels -- <case|all>...
//! ```
//!
//! Protocol per case: 10 warm-ups outside the profiler range, `cuProfilerStart`,
//! 100 timed iterations, one barrier at the observation boundary,
//! `cuProfilerStop`, single host thread. Buffers, plans and modules are prepared
//! before the range, and every allocation, transfer, zero fill, host wait and
//! physical launch the harness performs is counted; the measured region is
//! asserted to contain launches only.
//!
//! ## What one timed iteration measures
//!
//! Every case runs the same submission sequence 100 times, with no transfer, no
//! allocation and no host wait inside the profiler range, so the device time
//! nsys attributes to the region is the cost of that sequence. Two cases update an
//! operand in place, so their sequence is **not** reset between iterations and an
//! iteration measures the operation applied to the state the previous iteration
//! left behind:
//!
//! - **RRQR** (`bench_rrqr`): the host loop factorizes its work buffer in place, so
//!   iteration *n* factorizes the factorized remainder of iteration *n-1*, not a
//!   restored copy of the input matrix. The measured region starts from the state
//!   the reference-check factorization of the same input matrix left; the baseline
//!   arm (`bench/cubecl-baseline/baseline.patch`) does exactly the same, so both
//!   arms begin the region in equivalent states. No iteration — on either arm —
//!   starts from a restored matrix.
//! - **QR phase/apply** (`bench_qr_apply_phase`): Q's columns and R's rows are
//!   scaled in place every iteration, so the phases compound; the baseline arm
//!   submits the same in-place sequence against the same operands, and neither arm
//!   restores Q or R after its reference check.
//!
//! Every other case writes a destination it never reads or overwrites its state
//! completely (element-wise, copy/transpose/materialize, windowing, reductions,
//! gather, collision-heavy scatter, LU apply-pivots, Householder explicit-V), so
//! their iterations are independent of each other.
//!
//! `input_digest` in `BENCH_META` is an FNV-1a 64 of the case's initial host input
//! (one length-prefixed part per operand, in upload order), computed with the same
//! helper as the baseline arm, so two harnesses that print the same digest were
//! given the same data even for the in-place cases above.
//!
//! `scratch_bytes` counts every device buffer a case allocates besides the input
//! operands it uploads: for a split reduction the caller-provided partials, for the
//! RRQR loop the norms, coefficient, permutation, pivot, status and rank arrays, the
//! Q and R working outputs and the `m`-element packed-reflector buffer. Cases with no
//! such buffer report 0.

#[path = "../tests/gpu/mod.rs"]
mod gpu;
#[path = "../tests/gpu/prepare.rs"]
mod prepare;

use std::cell::Cell;

use gpu::{download, launch_async, launches, pub_sync, upload, written, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_apply_reflector, plan_binary, plan_column_norms, plan_copy_exact, plan_elementwise,
    plan_extract_r, plan_fused, plan_gather, plan_householder_explicit_v, plan_identity_shifted,
    plan_initialize_permutation, plan_lu_apply_pivots, plan_materialize_strided, plan_pad,
    plan_prefix_rank, plan_qr_apply_phase, plan_qr_phase, plan_reduction, plan_reflector,
    plan_scatter_add, plan_select_pivot, plan_slice, plan_swap_columns, plan_tiled_transpose,
    plan_unary, BinaryOp, CAxisValues, CGatherPlan, CLayout, CScatterPlan, CTransposeConfig, DType,
    Entry, FusionInstruction, FusionOp, FusionOperand, FusionPlan, GatherPlan, Layout, ReduceOp,
    ReductionPlan, ScalarSide, ScatterPlan, TransposeConfig, UnaryOp, FUSION_ENTRY,
};

const WARMUPS: usize = 10;
const ITERS: usize = 100;
const SENTINEL: u8 = 0xCD;
const PLANE_WIDTH: u32 = 32;

// ---------------------------------------------------------------------------
// Counters for the measured region
// ---------------------------------------------------------------------------

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static TRANSFER_BYTES: Cell<usize> = const { Cell::new(0) };
    static ZERO_FILLS: Cell<usize> = const { Cell::new(0) };
    static HOST_WAITS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Copy, Default)]
struct Counters {
    launches: usize,
    allocations: usize,
    transfer_bytes: usize,
    zero_fills: usize,
    host_waits: usize,
}

fn snapshot() -> Counters {
    Counters {
        launches: launches(),
        allocations: ALLOCATIONS.with(Cell::get),
        transfer_bytes: TRANSFER_BYTES.with(Cell::get),
        zero_fills: ZERO_FILLS.with(Cell::get),
        host_waits: HOST_WAITS.with(Cell::get),
    }
}

/// A device allocation outside the measured region, sentinel-filled so an
/// unstored byte is visible to the reference check.
fn alloc(session: &Session, bytes: usize) -> Buf {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + bytes));
    written(session, &vec![SENTINEL; bytes])
}

/// A device allocation that is zero-filled (flags, scatter initialization).
fn alloc_zero(session: &Session, bytes: usize) -> Buf {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + bytes));
    ZERO_FILLS.with(|count| count.set(count.get() + 1));
    written(session, &vec![0u8; bytes])
}

fn upload_counted<T: Copy>(session: &Session, values: &[T]) -> Buf {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + std::mem::size_of_val(values).max(1)));
    upload(session, values)
}

fn read_counted<T: Copy>(session: &Session, buf: &Buf, count: usize) -> Vec<T> {
    TRANSFER_BYTES.with(|slot| slot.set(slot.get() + count * std::mem::size_of::<T>()));
    HOST_WAITS.with(|slot| slot.set(slot.get() + 1));
    download::<T>(session, buf, count)
}

fn read_prefix(session: &Session, buf: &Buf, bytes: usize) -> Vec<u8> {
    TRANSFER_BYTES.with(|slot| slot.set(slot.get() + bytes));
    HOST_WAITS.with(|slot| slot.set(slot.get() + 1));
    gpu::read_prefix(session, buf, bytes)
}

fn sync(session: &Session) {
    HOST_WAITS.with(|slot| slot.set(slot.get() + 1));
    pub_sync(session.stream());
}

// ---------------------------------------------------------------------------
// Profiler range
// ---------------------------------------------------------------------------

type ProfilerFn = unsafe extern "C" fn(u32) -> i32;

fn profiler_symbol(name: &[u8]) -> ProfilerFn {
    // `cuProfilerStart`/`cuProfilerStop` are not bound by cudarc, so they are
    // resolved from the already-loaded driver library.
    let lib = unsafe { cudarc::driver::sys::culib() };
    *unsafe { lib.get::<ProfilerFn>(name) }.expect("driver profiler symbol")
}

fn profiler_start() -> i32 {
    unsafe { profiler_symbol(b"cuProfilerStart\0")(0) }
}

fn profiler_stop() -> i32 {
    unsafe { profiler_symbol(b"cuProfilerStop\0")(0) }
}

// ---------------------------------------------------------------------------
// Case driver
// ---------------------------------------------------------------------------

thread_local! {
    /// `input_digest` of the case currently running: recorded once by the case body
    /// from the host bytes it uploads, printed by `emit_meta`. The baseline arm
    /// implements the same helper, so equal digests prove the paired experiment gave
    /// both sides the same data.
    static INPUT_DIGEST: Cell<Option<u64>> = const { Cell::new(None) };
}

/// FNV-1a 64 over a case's initial host input: one length-prefixed part per operand,
/// in the order the case uploads them. Byte-for-byte identical to the helper in
/// `bench/cubecl-baseline/baseline.patch`.
fn input_digest(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        hash ^= part.len() as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        for byte in part.iter() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn record_input_digest(parts: &[&[u8]]) {
    let digest = input_digest(parts);
    INPUT_DIGEST.with(|cell| {
        assert!(
            cell.get().is_none(),
            "a case recorded its input digest twice"
        );
        cell.set(Some(digest));
    });
}

fn take_input_digest(case: &str) -> u64 {
    INPUT_DIGEST
        .with(|cell| cell.take())
        .unwrap_or_else(|| panic!("{case}: no input digest was recorded"))
}

struct Meta {
    kernels: &'static str,
    launches: usize,
    scratch_bytes: usize,
    check: &'static str,
}

fn emit_meta(case: &str, meta: &Meta, observed: Counters) {
    let digest = take_input_digest(case);
    println!(
        "BENCH_META case={case} kernels={} launches_per_iter={} scratch_bytes={} \
         host_waits_per_iter={} transfer_bytes_per_iter={} device_allocations_per_iter={} \
         zero_fills_per_iter={} boundary_barriers=1 input_digest=0x{digest:016x} check={}",
        meta.kernels,
        meta.launches,
        meta.scratch_bytes,
        observed.host_waits,
        observed.transfer_bytes,
        observed.allocations,
        observed.zero_fills,
        meta.check
    );
}

/// Warm-ups, then 100 timed iterations inside the profiler range, then the one
/// boundary barrier; asserts the measured region only launched kernels.
fn measure(case: &str, session: &Session, meta: &Meta, run: &mut impl FnMut()) -> Counters {
    for _ in 0..WARMUPS {
        run();
    }
    sync(session);
    let before = snapshot();
    assert_eq!(profiler_start(), 0, "cuProfilerStart");
    for _ in 0..ITERS {
        run();
    }
    let after = snapshot();
    // The single barrier at the observation boundary (not an iteration wait).
    pub_sync(session.stream());
    assert_eq!(profiler_stop(), 0, "cuProfilerStop");

    assert_eq!(
        after.launches - before.launches,
        meta.launches * ITERS,
        "{case}: launch count in the measured region"
    );
    assert_eq!(
        after.allocations, before.allocations,
        "{case}: the measured region allocated device memory"
    );
    assert_eq!(
        after.transfer_bytes, before.transfer_bytes,
        "{case}: the measured region transferred bytes"
    );
    assert_eq!(
        after.zero_fills, before.zero_fills,
        "{case}: the measured region zero-filled memory"
    );
    assert_eq!(
        after.host_waits, before.host_waits,
        "{case}: the measured region waited on the host"
    );
    Counters {
        launches: meta.launches,
        ..Counters::default()
    }
}

fn checked(case: &str, ok: bool) {
    assert!(ok, "reference check failed for {case}");
}

fn wanted(case: &str, filter: &[String]) -> bool {
    filter.is_empty() || filter.iter().any(|entry| entry == case || entry == "all")
}

// ---------------------------------------------------------------------------
// Host references (column-major, axis 0 fastest; never the kernel's index walk)
// ---------------------------------------------------------------------------

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

fn compact_strides(extents: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; extents.len()];
    let mut stride = 1i64;
    for (axis, &extent) in extents.iter().enumerate() {
        strides[axis] = stride;
        stride *= extent;
    }
    strides
}

fn layout(extents: &[i64], strides: &[i64], offset: i64) -> Layout {
    Layout::new(extents, strides, offset).expect("layout")
}

fn product(extents: &[i64]) -> i64 {
    extents.iter().product()
}

fn close_f32(actual: f32, expected: f32) -> bool {
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0)
}

fn close_f64(actual: f64, expected: f64) -> bool {
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0)
}

fn close_sum(actual: f64, expected: f64, magnitude: f64) -> bool {
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= 1e-5 * magnitude + 1e-6
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn c64(&mut self) -> C64 {
        C64 {
            re: self.f64() * 4.0,
            im: self.f64() * 4.0,
        }
    }

    fn i32(&mut self) -> i32 {
        (self.next_u64() % 2_000_001) as i32 - 1_000_000
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct C64 {
    re: f64,
    im: f64,
}

fn samples(len: usize, stride: usize) -> Vec<usize> {
    let mut points = vec![0, 1, 2, len / 3, len / 2, len - 2, len - 1];
    let mut index = 3;
    while index + 4 < len {
        points.push(index);
        index = index.saturating_add(stride);
    }
    points.retain(|&point| point < len);
    points.sort_unstable();
    points.dedup();
    points
}

// ---------------------------------------------------------------------------
// Flat elementwise entries
// ---------------------------------------------------------------------------

fn bench_add_f32(case: &str, len: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("arithmetic");
    let function = session.function(Entry::Add.name());
    let plan = plan_elementwise(Entry::Add, DType::F32, len as i64).unwrap();
    let mut rng = Rng::new(len as u64);
    let lhs: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    let rhs: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&lhs), gpu::as_bytes(&rhs)]);
    let lhs_buf = upload_counted(&session, &lhs);
    let rhs_buf = upload_counted(&session, &rhs);
    let out = alloc(&session, len * 4);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(DType::F32.tag()),
            Slot::i64(len as i64),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &out, len);
    checked(
        case,
        samples(len, 65_537)
            .into_iter()
            .all(|index| close_f32(got[index], lhs[index] + rhs[index])),
    );
    let meta = Meta {
        kernels: Entry::Add.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_exp_f32(case: &str, len: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("elementwise");
    let function = session.function(Entry::Unary.name());
    let plan = plan_unary(UnaryOp::Exp, DType::F32, DType::F32, len as i64).unwrap();
    let mut rng = Rng::new(0x5F5F_5555);
    let input: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&input)]);
    let input_buf = upload_counted(&session, &input);
    let out = alloc(&session, len * 4);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&input_buf),
            Slot::i32(DType::F32.tag()),
            Slot::i32(DType::F32.tag()),
            Slot::i32(UnaryOp::Exp.tag()),
            Slot::i64(len as i64),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &out, len);
    checked(
        case,
        samples(len, 65_537)
            .into_iter()
            .all(|index| close_f32(got[index], input[index].exp())),
    );
    let meta = Meta {
        kernels: Entry::Unary.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_mul_c64(case: &str, len: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("arithmetic");
    let function = session.function(Entry::Mul.name());
    let plan = plan_elementwise(Entry::Mul, DType::C64, len as i64).unwrap();
    let mut rng = Rng::new(0x1111_2222);
    let lhs: Vec<C64> = (0..len).map(|_| rng.c64()).collect();
    let rhs: Vec<C64> = (0..len).map(|_| rng.c64()).collect();
    record_input_digest(&[gpu::as_bytes(&lhs), gpu::as_bytes(&rhs)]);
    let lhs_buf = upload_counted(&session, &lhs);
    let rhs_buf = upload_counted(&session, &rhs);
    let out = alloc(&session, len * 16);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(DType::C64.tag()),
            Slot::i64(len as i64),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<C64>(&session, &out, len);
    checked(
        case,
        samples(len, 65_537).into_iter().all(|index| {
            let (a, b) = (lhs[index], rhs[index]);
            let expected = C64 {
                re: a.re * b.re - a.im * b.im,
                im: a.re * b.im + a.im * b.re,
            };
            close_f64(got[index].re, expected.re) && close_f64(got[index].im, expected.im)
        }),
    );
    let meta = Meta {
        kernels: Entry::Mul.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_div_c64(case: &str, len: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    // Complex division is not `Entry::Div` (float-only since revision 6); it is
    // revision 6's `t4a_cuda_binary` `DIV_COMPLEX` route, the scale-robust
    // component form the baseline's `div_complex_parts` kernel implements.
    let session = Session::new("elementwise");
    let function = session.function(Entry::Binary.name());
    let plan = plan_binary(
        BinaryOp::DivComplex,
        DType::C64,
        ScalarSide::Both,
        len as i64,
    )
    .unwrap();
    let mut rng = Rng::new(0x3333_4444);
    let lhs: Vec<C64> = (0..len).map(|_| rng.c64()).collect();
    let rhs: Vec<C64> = (0..len)
        .map(|_| {
            let value = rng.c64();
            C64 {
                re: value.re + 2.0,
                im: value.im + 1.0,
            }
        })
        .collect();
    record_input_digest(&[gpu::as_bytes(&lhs), gpu::as_bytes(&rhs)]);
    let lhs_buf = upload_counted(&session, &lhs);
    let rhs_buf = upload_counted(&session, &rhs);
    // `err` is required for the checked integer tags only and untouched by
    // `DIV_COMPLEX`; it is supplied anyway so the argument list matches the ABI.
    let err = alloc(&session, 4);
    let out = alloc(&session, len * 16);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&lhs_buf),
            Slot::ptr(&rhs_buf),
            Slot::i32(DType::C64.tag()),
            Slot::i32(BinaryOp::DivComplex.tag()),
            Slot::i32(ScalarSide::Both.tag()),
            Slot::i64(len as i64),
            Slot::ptr(&err),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<C64>(&session, &out, len);
    checked(
        case,
        samples(len, 65_537).into_iter().all(|index| {
            let (a, b) = (lhs[index], rhs[index]);
            let denominator = b.re * b.re + b.im * b.im;
            let expected = C64 {
                re: (a.re * b.re + a.im * b.im) / denominator,
                im: (a.im * b.re - a.re * b.im) / denominator,
            };
            close_f64(got[index].re, expected.re) && close_f64(got[index].im, expected.im)
        }),
    );
    let meta = Meta {
        kernels: Entry::Binary.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// Layout / transpose / window
// ---------------------------------------------------------------------------

fn bench_copy_f32(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("movement");
    let function = session.function(Entry::CopyExact.name());
    let len = side * side;
    let plan = plan_copy_exact(DType::F32, len as i64).unwrap();
    let mut rng = Rng::new(0xBBBB_CCCC);
    let values: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, len * 4);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F32.tag()),
            Slot::i64(len as i64),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, len);
    checked(case, got == values);
    let meta = Meta {
        kernels: Entry::CopyExact.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_tiled_transpose(
    case: &str,
    config: TransposeConfig,
    dst_fast: usize,
    src_fast: usize,
    batches: usize,
    src_offset: usize,
    filter: &[String],
) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("movement");
    let function = session.function(Entry::TiledTranspose.name());
    let matrix = dst_fast * src_fast;
    let span = matrix * batches;
    let plan = plan_tiled_transpose(
        DType::F32,
        config,
        src_offset as i64,
        matrix as i64,
        dst_fast as i64,
        src_fast as i64,
        batches as i64,
        (span + src_offset) as i64,
        span as i64,
    )
    .unwrap();
    let mut rng = Rng::new(0xDDDD_EEEE + src_offset as u64);
    let values: Vec<f32> = (0..span + src_offset).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, span * 4);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F32.tag()),
            Slot::i64(src_offset as i64),
            Slot::i64(matrix as i64),
            Slot::i64(dst_fast as i64),
            Slot::i64(src_fast as i64),
            Slot::config(&CTransposeConfig::from(config)),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, span);
    let mut ok = true;
    for batch in 0..batches {
        for i in 0..dst_fast {
            for j in 0..src_fast {
                let expected = values[src_offset + batch * matrix + i * src_fast + j];
                if !close_f32(got[batch * matrix + i + j * dst_fast], expected) {
                    ok = false;
                }
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::TiledTranspose.name(),
        launches: 1,
        scratch_bytes: config.tile() as usize * (config.tile() + config.padding()) as usize * 4,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_materialize_negative_stride(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("movement");
    let function = session.function(Entry::MaterializeStrided.name());
    let len = side * side;
    let extents = [side as i64, side as i64];
    let src_layout = layout(&extents, &[-1, -(side as i64)], (len - 1) as i64);
    let plan = plan_materialize_strided(DType::F32, &src_layout, len as i64).unwrap();
    let mut rng = Rng::new(0x1234_5678);
    let values: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, len * 4);
    let src_c = CLayout::new(&src_layout);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F32.tag()),
            Slot::layout(&src_c),
            Slot::i64(len as i64),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, len);
    let mut ok = true;
    for i in 0..side {
        for j in 0..side {
            let expected = values[len - 1 - i - j * side];
            if !close_f32(got[i + j * side], expected) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::MaterializeStrided.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_slice_f32(case: &str, set: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let side = 4_096i64;
    let (starts, sizes, steps): ([i64; 2], [i64; 2], [i64; 2]) = match set {
        1 => ([3, 5], [1_000, 2_000], [1, 1]),
        2 => ([0, 0], [2_048, 2_048], [2, 1]),
        3 => ([17, 4_095], [2_048, 1], [1, 1]),
        _ => ([1, 2], [4_095, 4_093], [1, 1]),
    };
    let session = Session::new("window");
    let function = session.function(Entry::Slice.name());
    let in_extents = [side, side];
    let in_layout = layout(&in_extents, &compact_strides(&in_extents), 0);
    let out_layout = layout(&sizes, &compact_strides(&sizes), 0);
    let out_len = product(&sizes);
    let plan = plan_slice(
        DType::F32,
        &out_layout,
        out_len,
        &in_layout,
        side * side,
        &starts,
        &steps,
    )
    .unwrap();
    let mut rng = Rng::new(0x9999 + set as u64);
    let values: Vec<f32> = (0..side * side).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, out_len as usize * 4);
    let in_c = CLayout::new(&in_layout);
    let out_c = CLayout::new(&out_layout);
    let starts_c = CAxisValues::new(&starts).unwrap();
    let steps_c = CAxisValues::new(&steps).unwrap();
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F32.tag()),
            Slot::layout(&out_c),
            Slot::layout(&in_c),
            Slot::axis_values(&starts_c),
            Slot::axis_values(&steps_c),
            Slot::i64(out_len),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, out_len as usize);
    let mut ok = true;
    for i in 0..sizes[0] {
        for j in 0..sizes[1] {
            let expected =
                values[(starts[0] + i * steps[0] + (starts[1] + j * steps[1]) * side) as usize];
            if !close_f32(got[(i + j * sizes[0]) as usize], expected) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::Slice.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_pad_f32(case: &str, set: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let side = 4_096i64;
    let (low, high, interior): ([i64; 2], [i64; 2], [i64; 2]) = match set {
        1 => ([1, 2], [3, 4], [0, 0]),
        2 => ([0, 0], [0, 1], [1, 0]),
        3 => ([5, 7], [0, 0], [0, 2]),
        _ => ([2, 2], [2, 2], [1, 1]),
    };
    let out_extents: [i64; 2] = [
        (side - 1) * (interior[0] + 1) + 1 + low[0] + high[0],
        (side - 1) * (interior[1] + 1) + 1 + low[1] + high[1],
    ];
    let session = Session::new("window");
    let function = session.function(Entry::Pad.name());
    let in_extents = [side, side];
    let in_layout = layout(&in_extents, &compact_strides(&in_extents), 0);
    let out_layout = layout(&out_extents, &compact_strides(&out_extents), 0);
    let out_len = product(&out_extents);
    let plan = plan_pad(
        DType::F32,
        &out_layout,
        out_len,
        &in_layout,
        side * side,
        &low,
        &high,
        &interior,
    )
    .unwrap();
    let mut rng = Rng::new(0xAAAA + set as u64);
    let values: Vec<f32> = (0..side * side).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc_zero(&session, out_len as usize * 4);
    let in_c = CLayout::new(&in_layout);
    let out_c = CLayout::new(&out_layout);
    let low_c = CAxisValues::new(&low).unwrap();
    let high_c = CAxisValues::new(&high).unwrap();
    let interior_c = CAxisValues::new(&interior).unwrap();
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&src),
            Slot::i32(DType::F32.tag()),
            Slot::layout(&out_c),
            Slot::layout(&in_c),
            Slot::axis_values(&low_c),
            Slot::axis_values(&high_c),
            Slot::axis_values(&interior_c),
            Slot::i64(out_len),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, out_len as usize);
    let mut ok = true;
    for i in 0..out_extents[0] {
        for j in 0..out_extents[1] {
            let i_shift = i - low[0];
            let j_shift = j - low[1];
            let spacing_i = interior[0] + 1;
            let spacing_j = interior[1] + 1;
            let expected = if i_shift < 0
                || j_shift < 0
                || i_shift % spacing_i != 0
                || j_shift % spacing_j != 0
            {
                0.0
            } else {
                let source_i = i_shift / spacing_i;
                let source_j = j_shift / spacing_j;
                if source_i < side && source_j < side {
                    values[(source_i + source_j * side) as usize]
                } else {
                    0.0
                }
            };
            if !close_f32(got[(i + j * out_extents[0]) as usize], expected) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::Pad.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Everything a reduction case needs to submit its matched sequence.
struct ReduceSetup {
    in_layout: Layout,
    out_layout: Layout,
    partials_layout: Option<Layout>,
    plan: ReductionPlan,
}

fn reduce_setup(dtype: DType, in_extents: &[i64], axis: u32, op: ReduceOp) -> ReduceSetup {
    let in_layout = layout(in_extents, &compact_strides(in_extents), 0);
    let mut out_extents = in_extents.to_vec();
    out_extents[axis as usize] = 1;
    let out_layout = layout(&out_extents, &compact_strides(&out_extents), 0);
    let plan = plan_reduction(
        dtype,
        op,
        &out_layout,
        product(&out_extents),
        &in_layout,
        product(in_extents),
        axis,
        PLANE_WIDTH,
    )
    .unwrap();
    let partials_layout = match &plan {
        ReductionPlan::Direct { .. } => None,
        ReductionPlan::Split { split, .. } => {
            let mut extents = out_extents.clone();
            extents[axis as usize] = split.splits;
            Some(layout(&extents, &compact_strides(&extents), 0))
        }
    };
    ReduceSetup {
        in_layout,
        out_layout,
        partials_layout,
        plan,
    }
}

fn reduce_launches(setup: &ReduceSetup) -> usize {
    match &setup.plan {
        ReductionPlan::Direct { .. } => 1,
        ReductionPlan::Split { .. } => 2,
    }
}

fn reduce_scratch(setup: &ReduceSetup, elem: usize) -> usize {
    setup
        .partials_layout
        .as_ref()
        .map(|layout| product(layout.extents()) as usize * elem)
        .unwrap_or(0)
}

fn bench_reduce_f32(case: &str, in_extents: &[i64], axis: u32, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["reduce"]);
    let reduce_fn = session.function(Entry::Reduce.name());
    let partials_fn = session.function(Entry::ReducePartials.name());
    let setup = reduce_setup(DType::F32, in_extents, axis, ReduceOp::Sum);
    let in_len = product(in_extents);
    let out_len = product(setup.out_layout.extents());
    let mut rng = Rng::new(0x1357_9BDF ^ in_extents[0] as u64);
    let values: Vec<f32> = (0..in_len).map(|_| rng.f32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, out_len as usize * 4);
    let partials = setup
        .partials_layout
        .as_ref()
        .map(|layout| alloc(&session, product(layout.extents()) as usize * 4));
    let in_c = CLayout::new(&setup.in_layout);
    let out_c = CLayout::new(&setup.out_layout);
    let partials_c = setup.partials_layout.as_ref().map(CLayout::new);
    let plan = setup.plan;
    let mut run = || match &plan {
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = [
                Slot::ptr(&dst),
                Slot::ptr(&src),
                Slot::i32(DType::F32.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::i32(strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(&in_c),
                Slot::i32(axis as i32),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, launch, session.stream(), &mut slots);
        }
        ReductionPlan::Split {
            split,
            pass1,
            combine_strategy,
            combine_op,
            pass2,
        } => {
            let partial_buf = partials.as_ref().unwrap();
            let partial_layout = partials_c.as_ref().unwrap();
            let mut slots1 = [
                Slot::ptr(partial_buf),
                Slot::ptr(&src),
                Slot::i32(DType::F32.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::layout(partial_layout),
                Slot::layout(&in_c),
                Slot::i32(axis as i32),
                Slot::i64(out_len),
                Slot::i64(split.splits),
                Slot::i64(split.chunk_len),
            ];
            launch_async(partials_fn, pass1, session.stream(), &mut slots1);
            let mut slots2 = [
                Slot::ptr(&dst),
                Slot::ptr(partial_buf),
                Slot::i32(DType::F32.tag()),
                Slot::i32(combine_op.tag()),
                Slot::i32(combine_strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(partial_layout),
                Slot::i32(axis as i32),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, pass2, session.stream(), &mut slots2);
        }
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, out_len as usize);
    let extents: Vec<i64> = in_extents.to_vec();
    let mut ok = true;
    for out_flat in 0..out_len {
        let mut coords = coordinates(out_flat, setup.out_layout.extents());
        let mut sum = 0.0f64;
        let mut magnitude = 0.0f64;
        for index in 0..in_extents[axis as usize] {
            coords[axis as usize] = index;
            let mut flat = 0i64;
            let strides = compact_strides(&extents);
            for (dim, &coordinate) in coords.iter().enumerate() {
                flat += coordinate * strides[dim];
            }
            let value = values[flat as usize] as f64;
            sum += value;
            magnitude += value.abs();
        }
        if !close_sum(got[out_flat as usize] as f64, sum, magnitude) {
            ok = false;
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: if reduce_launches(&setup) == 2 {
            "t4a_cuda_reduce_partials|t4a_cuda_reduce"
        } else {
            Entry::Reduce.name()
        },
        launches: reduce_launches(&setup),
        scratch_bytes: reduce_scratch(&setup, 4),
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_reduce_f64(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["reduce"]);
    let reduce_fn = session.function(Entry::Reduce.name());
    let partials_fn = session.function(Entry::ReducePartials.name());
    let in_extents = [side as i64, side as i64];
    let setup = reduce_setup(DType::F64, &in_extents, 0, ReduceOp::Sum);
    let in_len = product(&in_extents);
    let out_len = product(setup.out_layout.extents());
    let mut rng = Rng::new(0x0F0F_0F0F);
    let values: Vec<f64> = (0..in_len).map(|_| rng.f64()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, out_len as usize * 8);
    let partials = setup
        .partials_layout
        .as_ref()
        .map(|layout| alloc(&session, product(layout.extents()) as usize * 8));
    let in_c = CLayout::new(&setup.in_layout);
    let out_c = CLayout::new(&setup.out_layout);
    let partials_c = setup.partials_layout.as_ref().map(CLayout::new);
    let plan = setup.plan;
    let mut run = || match &plan {
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = [
                Slot::ptr(&dst),
                Slot::ptr(&src),
                Slot::i32(DType::F64.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::i32(strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, launch, session.stream(), &mut slots);
        }
        ReductionPlan::Split {
            split,
            pass1,
            combine_strategy,
            combine_op,
            pass2,
        } => {
            let partial_buf = partials.as_ref().unwrap();
            let partial_layout = partials_c.as_ref().unwrap();
            let mut slots1 = [
                Slot::ptr(partial_buf),
                Slot::ptr(&src),
                Slot::i32(DType::F64.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::layout(partial_layout),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
                Slot::i64(split.splits),
                Slot::i64(split.chunk_len),
            ];
            launch_async(partials_fn, pass1, session.stream(), &mut slots1);
            let mut slots2 = [
                Slot::ptr(&dst),
                Slot::ptr(partial_buf),
                Slot::i32(DType::F64.tag()),
                Slot::i32(combine_op.tag()),
                Slot::i32(combine_strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(partial_layout),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, pass2, session.stream(), &mut slots2);
        }
    };
    run();
    sync(&session);
    let got = read_counted::<f64>(&session, &dst, out_len as usize);
    let mut ok = true;
    for out_flat in 0..out_len as usize {
        let mut sum = 0.0f64;
        let mut magnitude = 0.0f64;
        for index in 0..side {
            let value = values[index + out_flat * side];
            sum += value;
            magnitude += value.abs();
        }
        if !close_sum(got[out_flat], sum, magnitude) {
            ok = false;
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: "t4a_cuda_reduce_partials|t4a_cuda_reduce",
        launches: reduce_launches(&setup),
        scratch_bytes: reduce_scratch(&setup, 8),
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_reduce_c64(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["reduce"]);
    let reduce_fn = session.function(Entry::Reduce.name());
    let partials_fn = session.function(Entry::ReducePartials.name());
    let in_extents = [side as i64, side as i64];
    let setup = reduce_setup(DType::C64, &in_extents, 0, ReduceOp::Sum);
    let in_len = product(&in_extents);
    let out_len = product(setup.out_layout.extents());
    let mut rng = Rng::new(0x0E0E_0E0E);
    let values: Vec<C64> = (0..in_len).map(|_| rng.c64()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, out_len as usize * 16);
    let partials = setup
        .partials_layout
        .as_ref()
        .map(|layout| alloc(&session, product(layout.extents()) as usize * 16));
    let in_c = CLayout::new(&setup.in_layout);
    let out_c = CLayout::new(&setup.out_layout);
    let partials_c = setup.partials_layout.as_ref().map(CLayout::new);
    let plan = setup.plan;
    let mut run = || match &plan {
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = [
                Slot::ptr(&dst),
                Slot::ptr(&src),
                Slot::i32(DType::C64.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::i32(strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, launch, session.stream(), &mut slots);
        }
        ReductionPlan::Split {
            split,
            pass1,
            combine_strategy,
            combine_op,
            pass2,
        } => {
            let partial_buf = partials.as_ref().unwrap();
            let partial_layout = partials_c.as_ref().unwrap();
            let mut slots1 = [
                Slot::ptr(partial_buf),
                Slot::ptr(&src),
                Slot::i32(DType::C64.tag()),
                Slot::i32(ReduceOp::Sum.tag()),
                Slot::layout(partial_layout),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
                Slot::i64(split.splits),
                Slot::i64(split.chunk_len),
            ];
            launch_async(partials_fn, pass1, session.stream(), &mut slots1);
            let mut slots2 = [
                Slot::ptr(&dst),
                Slot::ptr(partial_buf),
                Slot::i32(DType::C64.tag()),
                Slot::i32(combine_op.tag()),
                Slot::i32(combine_strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(partial_layout),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, pass2, session.stream(), &mut slots2);
        }
    };
    run();
    sync(&session);
    let got = read_counted::<C64>(&session, &dst, out_len as usize);
    let mut ok = true;
    for out_flat in 0..out_len as usize {
        let mut sum = C64 { re: 0.0, im: 0.0 };
        let mut magnitude = 0.0f64;
        for index in 0..side {
            let value = values[index + out_flat * side];
            sum.re += value.re;
            sum.im += value.im;
            magnitude += value.re.hypot(value.im);
        }
        if !close_sum(got[out_flat].re, sum.re, magnitude)
            || !close_sum(got[out_flat].im, sum.im, magnitude)
        {
            ok = false;
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: "t4a_cuda_reduce_partials|t4a_cuda_reduce",
        launches: reduce_launches(&setup),
        scratch_bytes: reduce_scratch(&setup, 16),
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_reduce_max_i32(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["reduce"]);
    let reduce_fn = session.function(Entry::Reduce.name());
    let partials_fn = session.function(Entry::ReducePartials.name());
    let in_extents = [side as i64, side as i64];
    let setup = reduce_setup(DType::I32, &in_extents, 0, ReduceOp::Max);
    let in_len = product(&in_extents);
    let out_len = product(setup.out_layout.extents());
    let mut rng = Rng::new(0x2468_ACE0);
    let values: Vec<i32> = (0..in_len).map(|_| rng.i32()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let src = upload_counted(&session, &values);
    let dst = alloc(&session, out_len as usize * 4);
    let partials = setup
        .partials_layout
        .as_ref()
        .map(|layout| alloc(&session, product(layout.extents()) as usize * 4));
    let in_c = CLayout::new(&setup.in_layout);
    let out_c = CLayout::new(&setup.out_layout);
    let partials_c = setup.partials_layout.as_ref().map(CLayout::new);
    let plan = setup.plan;
    let mut run = || match &plan {
        ReductionPlan::Direct { strategy, launch } => {
            let mut slots = [
                Slot::ptr(&dst),
                Slot::ptr(&src),
                Slot::i32(DType::I32.tag()),
                Slot::i32(ReduceOp::Max.tag()),
                Slot::i32(strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, launch, session.stream(), &mut slots);
        }
        ReductionPlan::Split {
            split,
            pass1,
            combine_strategy,
            combine_op,
            pass2,
        } => {
            let partial_buf = partials.as_ref().unwrap();
            let partial_layout = partials_c.as_ref().unwrap();
            let mut slots1 = [
                Slot::ptr(partial_buf),
                Slot::ptr(&src),
                Slot::i32(DType::I32.tag()),
                Slot::i32(ReduceOp::Max.tag()),
                Slot::layout(partial_layout),
                Slot::layout(&in_c),
                Slot::i32(0),
                Slot::i64(out_len),
                Slot::i64(split.splits),
                Slot::i64(split.chunk_len),
            ];
            launch_async(partials_fn, pass1, session.stream(), &mut slots1);
            let mut slots2 = [
                Slot::ptr(&dst),
                Slot::ptr(partial_buf),
                Slot::i32(DType::I32.tag()),
                Slot::i32(combine_op.tag()),
                Slot::i32(combine_strategy.tag()),
                Slot::layout(&out_c),
                Slot::layout(partial_layout),
                Slot::i32(0),
                Slot::i64(out_len),
            ];
            launch_async(reduce_fn, pass2, session.stream(), &mut slots2);
        }
    };
    run();
    sync(&session);
    let got = read_counted::<i32>(&session, &dst, out_len as usize);
    let mut ok = true;
    for out_flat in 0..out_len as usize {
        let mut best = i32::MIN;
        for index in 0..side {
            best = best.max(values[index + out_flat * side]);
        }
        if got[out_flat] != best {
            ok = false;
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: "t4a_cuda_reduce_partials|t4a_cuda_reduce",
        launches: reduce_launches(&setup),
        scratch_bytes: reduce_scratch(&setup, 4),
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// Gather and additive scatter
// ---------------------------------------------------------------------------

fn bench_gather(case: &str, rows: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["gather_scatter"]);
    let function = session.function(Entry::Gather.name());
    let cols = 256usize;
    let requested = 1024usize;
    let mut rng = Rng::new(0xFACE_0000 + rows as u64);
    let table: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
    let indices: Vec<i32> = (0..requested)
        .map(|_| (rng.next_u64() % rows as u64) as i32)
        .collect();
    record_input_digest(&[gpu::as_bytes(&table), gpu::as_bytes(&indices)]);
    let plan = GatherPlan {
        window_dims: vec![1],
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, cols as i64],
        index_vector_dim: 1,
    };
    let operand_layout = layout(
        &[rows as i64, cols as i64],
        &compact_strides(&[rows as i64, cols as i64]),
        0,
    );
    let indices_layout = layout(
        &[requested as i64, 1],
        &compact_strides(&[requested as i64, 1]),
        0,
    );
    let out_layout = layout(
        &[requested as i64, cols as i64],
        &compact_strides(&[requested as i64, cols as i64]),
        0,
    );
    let (launch, out_len) = plan_gather(
        DType::F32,
        DType::I32,
        &out_layout,
        product(out_layout.extents()),
        &operand_layout,
        product(operand_layout.extents()),
        &indices_layout,
        product(indices_layout.extents()),
        &plan,
    )
    .unwrap();
    let operand = upload_counted(&session, &table);
    let index_buf = upload_counted(&session, &indices);
    let dst = alloc(&session, out_len as usize * 4);
    let out_c = CLayout::new(&out_layout);
    let operand_c = CLayout::new(&operand_layout);
    let indices_c = CLayout::new(&indices_layout);
    let plan_c = CGatherPlan::new(&plan, 2).unwrap();
    let mut run = || {
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&operand),
            Slot::ptr(&index_buf),
            Slot::i32(DType::F32.tag()),
            Slot::i32(DType::I32.tag()),
            Slot::layout(&out_c),
            Slot::layout(&operand_c),
            Slot::layout(&indices_c),
            Slot::gather_plan(&plan_c),
            Slot::i64(out_len),
        ];
        launch_async(function, &launch, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f32>(&session, &dst, out_len as usize);
    let mut ok = true;
    for row in 0..requested {
        for col in 0..cols {
            let expected = table[indices[row] as usize + col * rows];
            if !close_f32(got[row + col * requested], expected) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::Gather.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_scatter_add(case: &str, complex: bool, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::with_modules(&["gather_scatter", "movement"]);
    let copy_fn = session.function(Entry::CopyExact.name());
    let scatter_fn = session.function(Entry::ScatterAdd.name());
    let slots_count = 16usize;
    let updates = 1_000_000usize;
    let dtype = if complex { DType::C64 } else { DType::F32 };
    let elem = if complex { 16 } else { 4 };
    let indices: Vec<i32> = (0..updates)
        .map(|index| ((index * 7 + index / 3) % slots_count) as i32)
        .collect();
    let mut rng = Rng::new(0xF00D_0000 + u64::from(complex));
    let operand_layout = layout(&[slots_count as i64], &[1], 0);
    let indices_layout = layout(
        &[updates as i64, 1],
        &compact_strides(&[updates as i64, 1]),
        0,
    );
    let updates_layout = layout(&[updates as i64], &[1], 0);
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let scatter_plan = plan_scatter_add(
        dtype,
        DType::I32,
        &operand_layout,
        slots_count as i64,
        &indices_layout,
        product(indices_layout.extents()),
        &updates_layout,
        updates as i64,
        &plan,
    )
    .unwrap();
    let copy_plan = plan_copy_exact(dtype, slots_count as i64).unwrap();
    let (operand_bytes, update_bytes, expected, magnitude): (Vec<u8>, Vec<u8>, Vec<C64>, Vec<f64>) =
        if complex {
            let operand: Vec<C64> = (0..slots_count)
                .map(|index| C64 {
                    re: index as f64,
                    im: -(index as f64),
                })
                .collect();
            let update_values: Vec<C64> = (0..updates).map(|_| rng.c64()).collect();
            let mut expected = operand.clone();
            let mut magnitude: Vec<f64> = operand
                .iter()
                .map(|value| value.re.hypot(value.im))
                .collect();
            for index in 0..updates {
                let slot = indices[index] as usize;
                expected[slot].re += update_values[index].re;
                expected[slot].im += update_values[index].im;
                magnitude[slot] += update_values[index].re.hypot(update_values[index].im);
            }
            (
                gpu::as_bytes(&operand).to_vec(),
                gpu::as_bytes(&update_values).to_vec(),
                expected,
                magnitude,
            )
        } else {
            let operand: Vec<f32> = (0..slots_count).map(|index| index as f32).collect();
            let update_values: Vec<f32> = (0..updates).map(|_| rng.f32()).collect();
            let mut expected = vec![0.0f64; slots_count];
            let mut magnitude = vec![0.0f64; slots_count];
            for slot in 0..slots_count {
                expected[slot] = operand[slot] as f64;
                magnitude[slot] = operand[slot].abs() as f64;
            }
            for index in 0..updates {
                let slot = indices[index] as usize;
                expected[slot] += update_values[index] as f64;
                magnitude[slot] += (update_values[index] as f64).abs();
            }
            (
                gpu::as_bytes(&operand).to_vec(),
                gpu::as_bytes(&update_values).to_vec(),
                expected
                    .into_iter()
                    .map(|value| C64 { re: value, im: 0.0 })
                    .collect(),
                magnitude,
            )
        };
    let operand = written(&session, &operand_bytes);
    record_input_digest(&[&operand_bytes, &update_bytes]);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + operand_bytes.len()));
    let index_buf = upload_counted(&session, &indices);
    let update_buf = written(&session, &update_bytes);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + update_bytes.len()));
    // The destination is written by the initialization copy every iteration, so
    // its initial content is irrelevant; it is zero-filled once, outside the
    // measured region, to make that explicit.
    let dst = alloc_zero(&session, slots_count * elem);
    let operand_c = CLayout::new(&operand_layout);
    let indices_c = CLayout::new(&indices_layout);
    let updates_c = CLayout::new(&updates_layout);
    let plan_c = CScatterPlan::new(1, 1, &plan).unwrap();
    let mut run = || {
        let mut copy_slots = [
            Slot::ptr(&dst),
            Slot::ptr(&operand),
            Slot::i32(dtype.tag()),
            Slot::i64(slots_count as i64),
        ];
        launch_async(copy_fn, &copy_plan, session.stream(), &mut copy_slots);
        let mut slots = [
            Slot::ptr(&dst),
            Slot::ptr(&index_buf),
            Slot::ptr(&update_buf),
            Slot::i32(dtype.tag()),
            Slot::i32(DType::I32.tag()),
            Slot::layout(&operand_c),
            Slot::layout(&indices_c),
            Slot::layout(&updates_c),
            Slot::scatter_plan(&plan_c),
            Slot::i64(updates as i64),
        ];
        launch_async(scatter_fn, &scatter_plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let mut ok = true;
    if complex {
        let got = read_counted::<C64>(&session, &dst, slots_count);
        for slot in 0..slots_count {
            if !close_sum(got[slot].re, expected[slot].re, magnitude[slot])
                || !close_sum(got[slot].im, expected[slot].im, magnitude[slot])
            {
                ok = false;
            }
        }
    } else {
        let got = read_counted::<f32>(&session, &dst, slots_count);
        // The f32 atomic accumulation rounds at the running-sum magnitude, so
        // the tolerance is relative to the accumulated magnitude.
        for slot in 0..slots_count {
            if !close_sum(got[slot] as f64, expected[slot].re, magnitude[slot]) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: "t4a_cuda_copy_exact|t4a_cuda_scatter_add",
        launches: 2,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// Linalg auxiliaries
// ---------------------------------------------------------------------------

fn bench_householder_explicit_v(
    case: &str,
    rows: usize,
    cols: usize,
    complex: bool,
    filter: &[String],
) {
    if !wanted(case, filter) {
        return;
    }
    let dtype = if complex { DType::C64 } else { DType::F64 };
    let elem = if complex { 16 } else { 8 };
    let session = Session::new("linalg_aux");
    let function = session.function(Entry::HouseholderExplicitV.name());
    let extents = [rows as i64, cols as i64];
    let packed_layout = layout(&extents, &compact_strides(&extents), 0);
    let out_layout = layout(&extents, &compact_strides(&extents), 0);
    let len = product(&extents);
    let plan = plan_householder_explicit_v(dtype, &out_layout, len, &packed_layout, len).unwrap();
    let mut rng = Rng::new(0x8080_0000 + rows as u64 + u64::from(complex));
    let values: Vec<C64> = if complex {
        (0..len).map(|_| rng.c64()).collect()
    } else {
        (0..len)
            .map(|_| C64 {
                re: rng.f64(),
                im: 0.0,
            })
            .collect()
    };
    let packed_bytes: Vec<u8> = if complex {
        gpu::as_bytes(&values).to_vec()
    } else {
        let real: Vec<f64> = values.iter().map(|value| value.re).collect();
        gpu::as_bytes(&real).to_vec()
    };
    let packed = written(&session, &packed_bytes);
    record_input_digest(&[&packed_bytes]);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + packed_bytes.len()));
    let out = alloc(&session, len as usize * elem);
    let out_c = CLayout::new(&out_layout);
    let packed_c = CLayout::new(&packed_layout);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&packed),
            Slot::i32(dtype.tag()),
            Slot::layout(&out_c),
            Slot::layout(&packed_c),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got_bytes = read_prefix(&session, &out, len as usize * elem);
    let got: Vec<C64> = (0..len as usize)
        .map(|index| {
            if complex {
                C64 {
                    re: f64::from_ne_bytes(
                        got_bytes[index * 16..index * 16 + 8].try_into().unwrap(),
                    ),
                    im: f64::from_ne_bytes(
                        got_bytes[index * 16 + 8..index * 16 + 16]
                            .try_into()
                            .unwrap(),
                    ),
                }
            } else {
                C64 {
                    re: f64::from_ne_bytes(got_bytes[index * 8..index * 8 + 8].try_into().unwrap()),
                    im: 0.0,
                }
            }
        })
        .collect();
    let mut ok = true;
    for i in 0..rows {
        for j in 0..cols {
            let expected = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                values[i + j * rows].re.hypot(values[i + j * rows].im)
            };
            let actual = got[i + j * rows].re.hypot(got[i + j * rows].im);
            if !close_f64(actual, expected) {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: Entry::HouseholderExplicitV.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_qr_apply_phase(case: &str, rows: usize, cols: usize, complex: bool, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let dtype = if complex { DType::C64 } else { DType::F64 };
    let elem = if complex { 16 } else { 8 };
    let session = Session::new("linalg_aux");
    let phase_fn = session.function(Entry::QrPhase.name());
    let apply_fn = session.function(Entry::QrApplyPhase.name());
    let q_extents = [rows as i64, cols as i64];
    let r_extents = [cols as i64, cols as i64];
    let phase_extents = [cols as i64];
    let q_layout = layout(&q_extents, &compact_strides(&q_extents), 0);
    let r_layout = layout(&r_extents, &compact_strides(&r_extents), 0);
    let phase_layout = layout(&phase_extents, &compact_strides(&phase_extents), 0);
    let phase_plan = plan_qr_phase(
        dtype,
        &phase_layout,
        cols as i64,
        &r_layout,
        product(&r_extents),
    )
    .unwrap();
    let apply_plan = plan_qr_apply_phase(
        dtype,
        &q_layout,
        product(&q_extents),
        &r_layout,
        product(&r_extents),
        &phase_layout,
        cols as i64,
        0,
    )
    .unwrap();
    let mut rng = Rng::new(0x7070_0000 + rows as u64 + u64::from(complex));
    let (q_bytes, r_bytes, q_values, r_values): (Vec<u8>, Vec<u8>, Vec<C64>, Vec<C64>) = if complex
    {
        let q_values: Vec<C64> = (0..rows * cols).map(|_| rng.c64()).collect();
        let mut r_values = vec![C64 { re: 0.0, im: 0.0 }; cols * cols];
        for j in 0..cols {
            for i in 0..=j {
                r_values[i + j * cols] = if i == j {
                    C64 {
                        re: 2.0 + i as f64,
                        im: 0.5,
                    }
                } else {
                    rng.c64()
                };
            }
        }
        (
            gpu::as_bytes(&q_values).to_vec(),
            gpu::as_bytes(&r_values).to_vec(),
            q_values,
            r_values,
        )
    } else {
        let q_values: Vec<f64> = (0..rows * cols).map(|_| rng.f64()).collect();
        let mut r_values = vec![0.0f64; cols * cols];
        for j in 0..cols {
            for i in 0..=j {
                r_values[i + j * cols] = if i == j {
                    if i % 2 == 0 {
                        2.0 + i as f64
                    } else {
                        -(2.0 + i as f64)
                    }
                } else {
                    rng.f64()
                };
            }
        }
        (
            gpu::as_bytes(&q_values).to_vec(),
            gpu::as_bytes(&r_values).to_vec(),
            q_values
                .iter()
                .map(|&value| C64 { re: value, im: 0.0 })
                .collect(),
            r_values
                .iter()
                .map(|&value| C64 { re: value, im: 0.0 })
                .collect(),
        )
    };
    let q_buf = written(&session, &q_bytes);
    record_input_digest(&[&q_bytes, &r_bytes]);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + q_bytes.len()));
    let r_buf = written(&session, &r_bytes);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + r_bytes.len()));
    let phase = alloc(&session, cols * elem);
    let q_c = CLayout::new(&q_layout);
    let r_c = CLayout::new(&r_layout);
    let phase_c = CLayout::new(&phase_layout);
    let mut run = || {
        let mut phase_slots = [
            Slot::ptr(&phase),
            Slot::ptr(&r_buf),
            Slot::i32(dtype.tag()),
            Slot::layout(&phase_c),
            Slot::layout(&r_c),
        ];
        launch_async(phase_fn, &phase_plan, session.stream(), &mut phase_slots);
        let mut apply_slots = [
            Slot::ptr(&q_buf),
            Slot::ptr(&r_buf),
            Slot::ptr(&phase),
            Slot::i32(dtype.tag()),
            Slot::layout(&q_c),
            Slot::layout(&r_c),
            Slot::layout(&phase_c),
            Slot::i64(0),
        ];
        launch_async(apply_fn, &apply_plan, session.stream(), &mut apply_slots);
    };
    run();
    sync(&session);
    let decode = |bytes: &[u8], count: usize| -> Vec<C64> {
        (0..count)
            .map(|index| {
                if complex {
                    C64 {
                        re: f64::from_ne_bytes(
                            bytes[index * 16..index * 16 + 8].try_into().unwrap(),
                        ),
                        im: f64::from_ne_bytes(
                            bytes[index * 16 + 8..index * 16 + 16].try_into().unwrap(),
                        ),
                    }
                } else {
                    C64 {
                        re: f64::from_ne_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap()),
                        im: 0.0,
                    }
                }
            })
            .collect()
    };
    let q_got = decode(
        &read_prefix(&session, &q_buf, rows * cols * elem),
        rows * cols,
    );
    let r_got = decode(
        &read_prefix(&session, &r_buf, cols * cols * elem),
        cols * cols,
    );
    let phase_of = |diagonal: C64| -> C64 {
        if complex {
            let magnitude = diagonal.re.hypot(diagonal.im);
            if magnitude == 0.0 {
                C64 { re: 1.0, im: 0.0 }
            } else {
                C64 {
                    re: diagonal.re / magnitude,
                    im: diagonal.im / magnitude,
                }
            }
        } else if diagonal.re < 0.0 {
            C64 { re: -1.0, im: 0.0 }
        } else {
            C64 { re: 1.0, im: 0.0 }
        }
    };
    let multiply = |a: C64, b: C64| C64 {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    };
    let conjugating = |value: C64| -> C64 {
        if complex {
            C64 {
                re: value.re,
                im: -value.im,
            }
        } else {
            value
        }
    };
    let mut ok = true;
    for j in 0..cols {
        let column_phase = phase_of(r_values[j + j * cols]);
        for i in 0..rows {
            let expected = multiply(q_values[i + j * rows], column_phase);
            if !close_f64(q_got[i + j * rows].re, expected.re)
                || !close_f64(q_got[i + j * rows].im, expected.im)
            {
                ok = false;
            }
        }
        for i in 0..cols {
            let row_phase = conjugating(phase_of(r_values[i + i * cols]));
            let expected = multiply(r_values[i + j * cols], row_phase);
            if !close_f64(r_got[i + j * cols].re, expected.re)
                || !close_f64(r_got[i + j * cols].im, expected.im)
            {
                ok = false;
            }
        }
    }
    checked(case, ok);
    let meta = Meta {
        kernels: "t4a_cuda_qr_phase|t4a_cuda_qr_apply_phase",
        launches: 2,
        scratch_bytes: cols * elem,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

fn bench_lu_apply_pivots(case: &str, side: usize, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let session = Session::new("linalg_aux");
    let function = session.function(Entry::LuApplyPivots.name());
    let extents = [side as i64, side as i64];
    let in_layout = layout(&extents, &compact_strides(&extents), 0);
    let out_layout = layout(&extents, &compact_strides(&extents), 0);
    let len = product(&extents);
    let pivots: Vec<i32> = (0..side).map(|index| index as i32 + 1).collect();
    let plan = plan_lu_apply_pivots(
        DType::F64,
        &out_layout,
        len,
        &in_layout,
        len,
        side as i64,
        side as i64,
    )
    .unwrap();
    let mut rng = Rng::new(0x6060);
    let values: Vec<f64> = (0..len).map(|_| rng.f64()).collect();
    record_input_digest(&[gpu::as_bytes(&values)]);
    let input = upload_counted(&session, &values);
    let pivots_buf = upload_counted(&session, &pivots);
    let out = alloc(&session, len as usize * 8);
    let out_c = CLayout::new(&out_layout);
    let in_c = CLayout::new(&in_layout);
    let mut run = || {
        let mut slots = [
            Slot::ptr(&out),
            Slot::ptr(&input),
            Slot::ptr(&pivots_buf),
            Slot::i32(DType::F64.tag()),
            Slot::layout(&out_c),
            Slot::layout(&in_c),
            Slot::i64(side as i64),
            Slot::i32(0),
        ];
        launch_async(function, &plan, session.stream(), &mut slots);
    };
    run();
    sync(&session);
    let got = read_counted::<f64>(&session, &out, len as usize);
    checked(case, got == values);
    let meta = Meta {
        kernels: Entry::LuApplyPivots.name(),
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// RRQR primitives (the host loop is caller code, as the ABI documents)
// ---------------------------------------------------------------------------

fn bench_rrqr(case: &str, m: usize, n: usize, complex: bool, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let dtype = if complex { DType::C64 } else { DType::F64 };
    let elem = if complex { 16 } else { 8 };
    // The norms buffer holds one real component per column (f64 either way), and
    // `select_pivot` reads that payload, so its dtype tag is the component type.
    let real_elem = 8;
    let norms_dtype = match dtype {
        DType::C32 => DType::F32,
        DType::C64 => DType::F64,
        other => other,
    };
    let k = m.min(n);
    // `identity_shifted` (the baseline's `initialize_q`) lives in `linalg_aux.cu`,
    // not `rrqr.cu`, and the copy of the reflector vector is `movement.cu`; every
    // module is loaded into one stream-ordered session.
    let session = Session::with_modules(&["rrqr", "linalg_aux", "movement"]);
    let init_perm_fn = session.function(Entry::InitializePermutation.name());
    let norms_fn = session.function(Entry::ColumnNorms.name());
    let pivot_fn = session.function(Entry::SelectPivot.name());
    let swap_fn = session.function(Entry::SwapColumns.name());
    let reflector_fn = session.function(Entry::Reflector.name());
    let apply_fn = session.function(Entry::ApplyReflector.name());
    let copy_fn = session.function(Entry::CopyExact.name());
    let extract_fn = session.function(Entry::ExtractR.name());
    let identity_fn = session.function(Entry::IdentityShifted.name());
    let rank_fn = session.function(Entry::PrefixRank.name());

    let work_layout = layout(
        &[m as i64, n as i64],
        &compact_strides(&[m as i64, n as i64]),
        0,
    );
    // A column-broadcast view of the compact reflector buffer: address
    // `offset + row * 1 + column * 0`, so `packed(row, step) == packed[row]`.
    // `apply_reflector` reads nothing but the reflector column `step`, which makes
    // the packed operand an `m`-element buffer instead of a second `m x n` matrix.
    let packed_layout = layout(&[m as i64, n as i64], &[1, 0], 0);
    let q_layout = layout(
        &[m as i64, k as i64],
        &compact_strides(&[m as i64, k as i64]),
        0,
    );
    let r_layout = layout(
        &[k as i64, n as i64],
        &compact_strides(&[k as i64, n as i64]),
        0,
    );
    let coeff_layout = layout(&[k as i64], &compact_strides(&[k as i64]), 0);
    let norms_layout = layout(&[n as i64], &compact_strides(&[n as i64]), 0);
    let perm_layout = layout(&[n as i64], &compact_strides(&[n as i64]), 0);
    let work_len = (m * n) as i64;
    let work_c = CLayout::new(&work_layout);
    let packed_c = CLayout::new(&packed_layout);
    let q_c = CLayout::new(&q_layout);
    let r_c = CLayout::new(&r_layout);
    let coeff_c = CLayout::new(&coeff_layout);
    let norms_c = CLayout::new(&norms_layout);
    let perm_c = CLayout::new(&perm_layout);

    let mut rng = Rng::new(0x5151 + (m * 31 + n) as u64);
    let values: Vec<C64> = if complex {
        (0..m * n)
            .map(|_| {
                let value = rng.c64();
                C64 {
                    re: value.re + 0.75,
                    im: value.im + 0.25,
                }
            })
            .collect()
    } else {
        (0..m * n)
            .map(|_| {
                let value = rng.f64() + 0.75;
                C64 { re: value, im: 0.0 }
            })
            .collect()
    };
    let (work_bytes, host_values): (Vec<u8>, Vec<C64>) = if complex {
        (gpu::as_bytes(&values).to_vec(), values.clone())
    } else {
        let real: Vec<f64> = values.iter().map(|value| value.re).collect();
        (gpu::as_bytes(&real).to_vec(), values.clone())
    };
    let work = written(&session, &work_bytes);
    record_input_digest(&[&work_bytes]);
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + work_bytes.len()));
    // Revision 8 requires the packed reflector storage and the trailing-update
    // target to be disjoint operands. The reflector vector of step `s` lives in
    // work's column `s` below the diagonal, which is a contiguous slice of the
    // compact matrix (`row + column * m`), so one `copy_exact` of `m - step`
    // elements per step moves it into the separate `packed` buffer.
    let packed = alloc(&session, m * elem);
    let perm = alloc_zero(&session, n * 8);
    let pivots = alloc_zero(&session, 8);
    let status = alloc_zero(&session, k * 8);
    let rank = alloc_zero(&session, 8);
    let norms = alloc_zero(&session, n * real_elem);
    let coeff = alloc_zero(&session, k * elem);
    let q = alloc(&session, m * k * elem);
    let r = alloc(&session, k * n * elem);

    let mut run = || {
        let mut slots = [Slot::ptr(&perm), Slot::i64(n as i64), Slot::i64(n as i64)];
        launch_async(
            init_perm_fn,
            &plan_initialize_permutation(n as i64, n as i64).unwrap(),
            session.stream(),
            &mut slots,
        );
        for step in 0..k {
            let step64 = step as i64;
            let mut slots = [
                Slot::ptr(&work),
                Slot::ptr(&norms),
                Slot::i32(dtype.tag()),
                Slot::layout(&work_c),
                Slot::layout(&norms_c),
                Slot::i64(step64),
            ];
            launch_async(
                norms_fn,
                &plan_column_norms(
                    dtype,
                    &work_layout,
                    work_len,
                    &norms_layout,
                    n as i64,
                    step64,
                )
                .unwrap(),
                session.stream(),
                &mut slots,
            );

            let mut slots = [
                Slot::ptr(&norms),
                Slot::ptr(&perm),
                Slot::ptr(&pivots),
                Slot::ptr(&status),
                Slot::i32(norms_dtype.tag()),
                Slot::layout(&norms_c),
                Slot::layout(&perm_c),
                Slot::i64(step64),
            ];
            launch_async(
                pivot_fn,
                &plan_select_pivot(
                    norms_dtype,
                    &norms_layout,
                    n as i64,
                    &perm_layout,
                    n as i64,
                    1,
                    k as i64,
                    step64,
                )
                .unwrap(),
                session.stream(),
                &mut slots,
            );

            let mut slots = [
                Slot::ptr(&work),
                Slot::ptr(&perm),
                Slot::ptr(&pivots),
                Slot::i32(dtype.tag()),
                Slot::layout(&work_c),
                Slot::layout(&perm_c),
                Slot::i64(step64),
            ];
            launch_async(
                swap_fn,
                &plan_swap_columns(
                    dtype,
                    &work_layout,
                    work_len,
                    &perm_layout,
                    n as i64,
                    1,
                    step64,
                )
                .unwrap(),
                session.stream(),
                &mut slots,
            );

            let mut slots = [
                Slot::ptr(&work),
                Slot::ptr(&coeff),
                Slot::i32(dtype.tag()),
                Slot::layout(&work_c),
                Slot::layout(&coeff_c),
                Slot::i64(step64),
            ];
            launch_async(
                reflector_fn,
                &plan_reflector(
                    dtype,
                    &work_layout,
                    work_len,
                    &coeff_layout,
                    k as i64,
                    step64,
                )
                .unwrap(),
                session.stream(),
                &mut slots,
            );

            let trailing = n - step - 1;
            if trailing > 0 {
                let mut slots = [
                    Slot::ptr_offset(&packed, step * elem),
                    Slot::ptr_offset(&work, (step * m + step) * elem),
                    Slot::i32(dtype.tag()),
                    Slot::i64((m - step) as i64),
                ];
                launch_async(
                    copy_fn,
                    &plan_copy_exact(dtype, (m - step) as i64).unwrap(),
                    session.stream(),
                    &mut slots,
                );
                let mut slots = [
                    Slot::ptr(&packed),
                    Slot::ptr(&work),
                    Slot::ptr(&coeff),
                    Slot::i32(dtype.tag()),
                    Slot::layout(&packed_c),
                    Slot::layout(&work_c),
                    Slot::layout(&coeff_c),
                    Slot::i64(step64),
                    Slot::i64((step + 1) as i64),
                    Slot::i64(trailing as i64),
                ];
                launch_async(
                    apply_fn,
                    &plan_apply_reflector(
                        dtype,
                        &packed_layout,
                        m as i64,
                        &work_layout,
                        work_len,
                        &coeff_layout,
                        k as i64,
                        step64,
                        (step + 1) as i64,
                        trailing as i64,
                    )
                    .unwrap(),
                    session.stream(),
                    &mut slots,
                );
            }
        }

        let mut slots = [
            Slot::ptr(&work),
            Slot::ptr(&r),
            Slot::i32(dtype.tag()),
            Slot::layout(&work_c),
            Slot::layout(&r_c),
        ];
        launch_async(
            extract_fn,
            &plan_extract_r(dtype, &work_layout, work_len, &r_layout, (k * n) as i64).unwrap(),
            session.stream(),
            &mut slots,
        );

        let mut slots = [
            Slot::ptr(&q),
            Slot::i32(dtype.tag()),
            Slot::layout(&q_c),
            Slot::i64(0),
        ];
        launch_async(
            identity_fn,
            &plan_identity_shifted(dtype, &q_layout, (m * k) as i64, 0).unwrap(),
            session.stream(),
            &mut slots,
        );

        for step in (0..k).rev() {
            // The Q update reads every completed reflector from `work` (a column is
            // never rewritten after its own step) and writes `q`: disjoint already.
            let mut slots = [
                Slot::ptr(&work),
                Slot::ptr(&q),
                Slot::ptr(&coeff),
                Slot::i32(dtype.tag()),
                Slot::layout(&work_c),
                Slot::layout(&q_c),
                Slot::layout(&coeff_c),
                Slot::i64(step as i64),
                Slot::i64(0),
                Slot::i64(k as i64),
            ];
            launch_async(
                apply_fn,
                &plan_apply_reflector(
                    dtype,
                    &work_layout,
                    work_len,
                    &q_layout,
                    (m * k) as i64,
                    &coeff_layout,
                    k as i64,
                    step as i64,
                    0,
                    k as i64,
                )
                .unwrap(),
                session.stream(),
                &mut slots,
            );
        }

        let mut slots = [
            Slot::ptr(&r),
            Slot::ptr(&rank),
            Slot::ptr(&status),
            Slot::i32(dtype.tag()),
            Slot::layout(&r_c),
            Slot::f64(0.0),
            Slot::f64(0.0),
        ];
        launch_async(
            rank_fn,
            &plan_prefix_rank(dtype, &r_layout, (k * n) as i64, 1, k as i64, 0.0, 0.0).unwrap(),
            session.stream(),
            &mut slots,
        );
    };
    run();
    sync(&session);
    let q_bytes = read_prefix(&session, &q, m * k * elem);
    let r_bytes = read_prefix(&session, &r, k * n * elem);
    let perm_values = read_counted::<i64>(&session, &perm, n);
    let read_c64 = |bytes: &[u8], count: usize| -> Vec<C64> {
        (0..count)
            .map(|index| {
                if complex {
                    let offset = index * 16;
                    C64 {
                        re: f64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap()),
                        im: f64::from_ne_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()),
                    }
                } else {
                    let offset = index * 8;
                    C64 {
                        re: f64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap()),
                        im: 0.0,
                    }
                }
            })
            .collect()
    };
    let q_values = read_c64(&q_bytes, m * k);
    let r_values = read_c64(&r_bytes, k * n);
    let mut residual = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut sum = C64 { re: 0.0, im: 0.0 };
            for l in 0..k {
                let a = q_values[i + l * m];
                let b = r_values[l + j * k];
                sum.re += a.re * b.re - a.im * b.im;
                sum.im += a.re * b.im + a.im * b.re;
            }
            let original = host_values[i + (perm_values[j] as usize) * m];
            residual += (sum.re - original.re).powi(2) + (sum.im - original.im).powi(2);
            scale += original.re * original.re + original.im * original.im;
        }
    }
    checked(case, residual.sqrt() <= 1e-8 * scale.sqrt().max(1.0));
    // The reference-check factorization above left `work` holding the packed
    // reflectors and the R factor. It is deliberately **not** restored: the
    // baseline arm's measured region starts from exactly the same state (one
    // factorization of the same input matrix), so the two arms begin the measured
    // region in equivalent states. Neither arm claims a restored matrix per
    // iteration -- each measured iteration factorizes the remainder of the previous
    // one, which is what both harness doc comments now state.

    // 1 initialize_permutation + k x (norms, pivot, swap, reflector)
    // + (k-1) x (reflector copy, trailing update) + extract_r
    // + identity_shifted + k x Q update + prefix_rank.
    let launches = 1 + k * 4 + 2 * (k - 1) + 1 + 1 + k + 1;
    // norms + coeff + (permutation, pivots, status, rank) + (Q, R) + the packed
    // reflector store the disjoint-operand contract needs.
    let scratch =
        n * real_elem + k * elem + (n + 1 + k + 1) * 8 + (m * k + k * n) * elem + m * elem;
    let meta = Meta {
        kernels: "t4a_cuda_initialize_permutation|t4a_cuda_column_norms|\
                  t4a_cuda_select_pivot|t4a_cuda_swap_columns|t4a_cuda_reflector|\
                  t4a_cuda_copy_exact|t4a_cuda_apply_reflector|t4a_cuda_extract_r|\
                  t4a_cuda_identity_shifted|t4a_cuda_prefix_rank",
        launches,
        scratch_bytes: scratch,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);
}

// ---------------------------------------------------------------------------
// Fused expressions through the prepare/compile/run consumer
// ---------------------------------------------------------------------------

struct FusedCase<'a> {
    case: &'a str,
    dtype: DType,
    plan: FusionPlan,
    len: usize,
    inputs: Vec<Vec<u8>>,
    outputs: usize,
    /// Reference values per output, element-wise in the dtype's scalar form.
    reference: Box<dyn Fn(usize, usize) -> C64 + 'a>,
}

fn operand_input(index: u32) -> FusionOperand {
    FusionOperand::Input(index)
}

fn operand_value(index: u32) -> FusionOperand {
    FusionOperand::Value(index)
}

fn instruction(op: FusionOp, operands: Vec<FusionOperand>) -> FusionInstruction {
    FusionInstruction { op, operands }
}

/// A device buffer holding a raw byte image, counted as an allocation + transfer
/// outside the measured region.
fn upload_bytes_counted(session: &Session, bytes: &[u8]) -> Buf {
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    TRANSFER_BYTES.with(|count| count.set(count.get() + bytes.len()));
    written(session, bytes)
}

/// One fused case through the caller-owned prepare/cache/launch boundary: NVRTC
/// compiles the emitted source once, the module is cached under the artifact key
/// and every timed iteration is one physical launch of the cached binding.
fn bench_fused(case: FusedCase<'_>, filter: &[String]) {
    if !wanted(case.case, filter) {
        return;
    }
    let scalar = case.dtype.size();
    let session = Session::with_modules(&[]);
    let consumer = prepare::Consumer::new(&session);
    let (_emitted, prepared) = consumer
        .prepare_fused(&case.plan)
        .expect("prepare of the fused plan");
    let plan = plan_fused(case.len as i64).expect("fused launch plan");
    let digest_parts: Vec<&[u8]> = case.inputs.iter().map(|bytes| bytes.as_slice()).collect();
    record_input_digest(&digest_parts);
    let input_bufs: Vec<Buf> = case
        .inputs
        .iter()
        .map(|bytes| upload_bytes_counted(&session, bytes))
        .collect();
    let output_bufs: Vec<Buf> = (0..case.outputs)
        .map(|_| alloc(&session, case.len * scalar))
        .collect();
    // Argument order of the emitted manifest: every output, then every input,
    // then the runtime element count.
    let mut run = || {
        let mut slots: Vec<Slot> = output_bufs.iter().map(Slot::ptr).collect();
        slots.extend(input_bufs.iter().map(Slot::ptr));
        slots.push(Slot::i64(case.len as i64));
        prepared.launch(&plan, &mut slots).expect("fused launch");
    };
    run();
    sync(&session);
    let mut ok = true;
    for (output_index, buf) in output_bufs.iter().enumerate() {
        let bytes = read_prefix(&session, buf, case.len * scalar);
        for index in samples(case.len, 65_537) {
            let actual = match (case.dtype, scalar) {
                (DType::C64, _) => C64 {
                    re: f64::from_ne_bytes(bytes[index * 16..index * 16 + 8].try_into().unwrap()),
                    im: f64::from_ne_bytes(
                        bytes[index * 16 + 8..index * 16 + 16].try_into().unwrap(),
                    ),
                },
                (_, 8) => C64 {
                    re: f64::from_ne_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap()),
                    im: 0.0,
                },
                _ => C64 {
                    re: f32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
                        as f64,
                    im: 0.0,
                },
            };
            let expected = (case.reference)(output_index, index);
            let tolerance = if scalar == 4 { 1e-4 } else { 1e-9 };
            if (actual.re - expected.re).abs() > tolerance * expected.re.abs().max(1.0)
                || (actual.im - expected.im).abs() > tolerance * expected.im.abs().max(1.0)
            {
                ok = false;
            }
        }
    }
    checked(case.case, ok);

    let meta = Meta {
        kernels: FUSION_ENTRY,
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case.case, &session, &meta, &mut run);
    emit_meta(case.case, &meta, observed);

    let counters = consumer.counters();
    let miss = consumer.last_miss_timings();
    println!(
        "BENCH_PREPARE case={} emits={} compile_attempts={} compile_successes={} \
         module_loads={} cache_hits={} cache_misses={} cold_compile_ms={:.3} cold_load_ms={:.3}",
        case.case,
        counters.emits,
        counters.compile_attempts,
        counters.compile_successes,
        counters.module_loads,
        counters.cache_hits,
        counters.cache_misses,
        miss.map(|timings| timings.compile.as_secs_f64() * 1e3)
            .unwrap_or(0.0),
        miss.map(|timings| timings.load.as_secs_f64() * 1e3)
            .unwrap_or(0.0),
    );
    // One plan and one artifact key: exactly one compile and one load, reused by
    // all 110 warm and timed launches.
    assert_eq!(counters.emits, 1, "{}: one emit", case.case);
    assert_eq!(
        counters.compile_attempts, 1,
        "{}: one NVRTC compile",
        case.case
    );
    assert_eq!(counters.module_loads, 1, "{}: one module load", case.case);
    assert_eq!(counters.cache_misses, 1, "{}: one cache miss", case.case);
}

fn fused_chain_f32(len: usize) -> FusedCase<'static> {
    // tanh(exp((x + x) * (x + x)))
    let plan = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: vec![
            instruction(FusionOp::Add, vec![operand_input(0), operand_input(0)]),
            instruction(FusionOp::Mul, vec![operand_value(0), operand_value(0)]),
            instruction(FusionOp::Exp, vec![operand_value(1)]),
            instruction(FusionOp::Tanh, vec![operand_value(2)]),
        ],
        outputs: vec![operand_value(3)],
    };
    let mut rng = Rng::new(0x5555_6666);
    let host: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    let bytes = gpu::as_bytes(&host).to_vec();
    FusedCase {
        case: if len == 1_000 {
            "fusion-chain-f32-1000"
        } else {
            "fusion-chain-f32-10000000"
        },
        dtype: DType::F32,
        plan,
        len,
        inputs: vec![bytes],
        outputs: 1,
        reference: Box::new(move |_, index| {
            let x = host[index] as f64;
            let sum = x + x;
            C64 {
                re: (sum * sum).exp().tanh(),
                im: 0.0,
            }
        }),
    }
}

fn fused_add_mul_c64(len: usize) -> FusedCase<'static> {
    // (x + x) * x * (x + x)
    let plan = FusionPlan {
        dtype: DType::C64,
        inputs: 1,
        instructions: vec![
            instruction(FusionOp::Add, vec![operand_input(0), operand_input(0)]),
            instruction(FusionOp::Mul, vec![operand_value(0), operand_input(0)]),
            instruction(FusionOp::Mul, vec![operand_value(1), operand_value(0)]),
        ],
        outputs: vec![operand_value(2)],
    };
    let mut rng = Rng::new(0x7777_8888);
    let host: Vec<C64> = (0..len).map(|_| rng.c64()).collect();
    let bytes = gpu::as_bytes(&host).to_vec();
    FusedCase {
        case: "fusion-add-mul-c64-10000000",
        dtype: DType::C64,
        plan,
        len,
        inputs: vec![bytes],
        outputs: 1,
        reference: Box::new(move |_, index| {
            let x = host[index];
            let multiply = |a: C64, b: C64| C64 {
                re: a.re * b.re - a.im * b.im,
                im: a.re * b.im + a.im * b.re,
            };
            let doubled = C64 {
                re: x.re + x.re,
                im: x.im + x.im,
            };
            multiply(multiply(doubled, x), doubled)
        }),
    }
}

fn fused_two_live_outs_f32(len: usize) -> FusedCase<'static> {
    // outputs: exp(x + x) and tanh(x + x)
    let plan = FusionPlan {
        dtype: DType::F32,
        inputs: 1,
        instructions: vec![
            instruction(FusionOp::Add, vec![operand_input(0), operand_input(0)]),
            instruction(FusionOp::Exp, vec![operand_value(0)]),
            instruction(FusionOp::Tanh, vec![operand_value(0)]),
        ],
        outputs: vec![operand_value(1), operand_value(2)],
    };
    let mut rng = Rng::new(0x9999_AAAA);
    let host: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
    let bytes = gpu::as_bytes(&host).to_vec();
    FusedCase {
        case: "fusion-two-live-outs-f32-10000000",
        dtype: DType::F32,
        plan,
        len,
        inputs: vec![bytes],
        outputs: 2,
        reference: Box::new(move |output, index| {
            let doubled = (host[index] + host[index]) as f64;
            C64 {
                re: if output == 0 {
                    doubled.exp()
                } else {
                    doubled.tanh()
                },
                im: 0.0,
            }
        }),
    }
}

/// Cold prepare: a fresh consumer emits and compiles the chain on its first call
/// and the warm launches reuse the cached module. The cold emit/compile/load is
/// timed with the host clock outside the profiler range; the warm launches then run
/// under the frozen protocol so this case also has a device-time number.
fn bench_prepare_cold(case: &str, filter: &[String]) {
    if !wanted(case, filter) {
        return;
    }
    let len = 10_000usize;
    let session = Session::with_modules(&[]);
    let consumer = prepare::Consumer::new(&session);
    let case_data = fused_chain_f32(len);
    record_input_digest(&[case_data.inputs[0].as_slice()]);
    let cold_start = std::time::Instant::now();
    let (_emitted, prepared) = consumer
        .prepare_fused(&case_data.plan)
        .expect("cold prepare");
    let cold = cold_start.elapsed();
    let plan = plan_fused(len as i64).expect("fused launch plan");
    let input = upload_bytes_counted(&session, &case_data.inputs[0]);
    let output = alloc(&session, len * 4);
    let mut run = || {
        let mut slots = [Slot::ptr(&output), Slot::ptr(&input), Slot::i64(len as i64)];
        prepared.launch(&plan, &mut slots).expect("fused launch");
    };
    // Reference check on the first launch.
    run();
    sync(&session);
    let got: Vec<f32> = read_prefix(&session, &output, len * 4)
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    checked(
        case,
        samples(len, 65_537).into_iter().all(|index| {
            let expected = (case_data.reference)(0, index);
            close_f32(got[index], expected.re as f32)
        }),
    );
    // Warm submission cost (10 launches, each observed), reported separately from
    // the device time.
    let warm_start = std::time::Instant::now();
    for _ in 0..WARMUPS {
        run();
        sync(&session);
    }
    let warm = warm_start.elapsed() / WARMUPS as u32;

    let meta = Meta {
        kernels: FUSION_ENTRY,
        launches: 1,
        scratch_bytes: 0,
        check: "ok",
    };
    let observed = measure(case, &session, &meta, &mut run);
    emit_meta(case, &meta, observed);

    let counters = consumer.counters();
    let miss = consumer.last_miss_timings();
    println!(
        "BENCH_PREPARE case={case} emits={} compile_attempts={} compile_successes={} \
         module_loads={} cache_hits={} cache_misses={} cold_ms={:.3} cold_compile_ms={:.3} \
         cold_load_ms={:.3} warm_ms={:.3}",
        counters.emits,
        counters.compile_attempts,
        counters.compile_successes,
        counters.module_loads,
        counters.cache_hits,
        counters.cache_misses,
        cold.as_secs_f64() * 1e3,
        miss.map(|timings| timings.compile.as_secs_f64() * 1e3)
            .unwrap_or(0.0),
        miss.map(|timings| timings.load.as_secs_f64() * 1e3)
            .unwrap_or(0.0),
        warm.as_secs_f64() * 1e3,
    );
    assert_eq!(counters.compile_attempts, 1, "one cold compile");
    assert_eq!(counters.module_loads, 1, "one module load");
    assert_eq!(counters.emits, 1, "one emit");
}

// ---------------------------------------------------------------------------
// Case table
// ---------------------------------------------------------------------------

fn main() {
    let filter: Vec<String> = std::env::args().skip(1).collect();
    if filter
        .iter()
        .any(|entry| entry == "--help" || entry == "-h")
    {
        println!("usage: bench_kernels [case|all]...");
        return;
    }

    for &len in &[1_000usize, 1_000_000, 100_000_000] {
        bench_add_f32(&format!("elementwise-add-f32-{len}"), len, &filter);
    }
    bench_exp_f32("elementwise-exp-f32-10000000", 10_000_000, &filter);
    bench_mul_c64("elementwise-mul-c64-10000000", 10_000_000, &filter);
    bench_div_c64("elementwise-div-c64-10000000", 10_000_000, &filter);

    bench_fused(fused_chain_f32(1_000), &filter);
    bench_fused(fused_chain_f32(10_000_000), &filter);
    bench_fused(fused_add_mul_c64(10_000_000), &filter);
    bench_fused(fused_two_live_outs_f32(10_000_000), &filter);
    bench_prepare_cold("prepare-fused-cold-compile", &filter);

    let default = TransposeConfig::new(16, 8, 1, 1).unwrap();
    bench_copy_f32("layout-copy-f32-4096sq", 4_096, &filter);
    bench_tiled_transpose(
        "layout-transpose-f32-4096sq",
        default,
        4_096,
        4_096,
        1,
        0,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-f32-4093x4099",
        default,
        4_093,
        4_099,
        1,
        0,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-f32-4099x4093",
        default,
        4_099,
        4_093,
        1,
        0,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-offset-f32-4096sq",
        default,
        4_096,
        4_096,
        1,
        64,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-batched-f32-64x256x256",
        default,
        256,
        256,
        64,
        0,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-32x8-p1-v1-f32-4096sq",
        TransposeConfig::new(32, 8, 1, 1).unwrap(),
        4_096,
        4_096,
        1,
        0,
        &filter,
    );
    bench_tiled_transpose(
        "layout-transpose-16x8-p1-v2-f32-4096sq",
        TransposeConfig::new(16, 8, 1, 2).unwrap(),
        4_096,
        4_096,
        1,
        0,
        &filter,
    );
    bench_materialize_negative_stride(
        "layout-materialize-negative-stride-f32-4096sq",
        4_096,
        &filter,
    );
    for set in 1..=4 {
        bench_slice_f32(&format!("layout-slice-f32-4096sq-set{set}"), set, &filter);
        bench_pad_f32(&format!("layout-pad-f32-4096sq-set{set}"), set, &filter);
    }

    for &len in &[511i64, 512, 513, 600] {
        bench_reduce_f32(
            &format!("reduce-sum-f32-axis-{len}"),
            &[len, 4_096],
            0,
            &filter,
        );
    }
    bench_reduce_f32("reduce-sum-f32-4096sq-axis0", &[4_096, 4_096], 0, &filter);
    bench_reduce_f32("reduce-sum-f32-4096sq-axis1", &[4_096, 4_096], 1, &filter);
    bench_reduce_f64("reduce-sum-f64-4096sq-axis0", 4_096, &filter);
    bench_reduce_c64("reduce-sum-c64-4096sq-axis0", 4_096, &filter);
    bench_reduce_max_i32("reduce-max-i32-4096sq-axis0", 4_096, &filter);
    for &len in &[1_000_000i64, 10_000_000] {
        bench_reduce_f32(&format!("reduce-sum-f32-full-{len}"), &[len, 1], 0, &filter);
    }

    bench_gather("indexing-gather-f32-1024x256-table1000", 1_000, &filter);
    bench_gather("indexing-gather-f32-1024x256-table100000", 100_000, &filter);
    bench_gather(
        "indexing-gather-f32-1024x256-table1000000",
        1_000_000,
        &filter,
    );
    bench_scatter_add("indexing-scatter-add-f32-1e6-into-16", false, &filter);
    bench_scatter_add("indexing-scatter-add-c64-1e6-into-16", true, &filter);

    bench_rrqr("linalg-rrqr-256x256-f64", 256, 256, false, &filter);
    bench_rrqr("linalg-rrqr-256x256-c64", 256, 256, true, &filter);
    bench_rrqr("linalg-rrqr-1024x256-f64", 1_024, 256, false, &filter);
    bench_rrqr("linalg-rrqr-1024x256-c64", 1_024, 256, true, &filter);
    bench_lu_apply_pivots("linalg-lu-apply-pivots-1024-f64", 1_024, &filter);
    bench_qr_apply_phase(
        "linalg-qr-apply-phase-1024x256-f64",
        1_024,
        256,
        false,
        &filter,
    );
    bench_qr_apply_phase(
        "linalg-qr-apply-phase-1024x256-c64",
        1_024,
        256,
        true,
        &filter,
    );
    bench_householder_explicit_v(
        "linalg-householder-explicit-v-1024x256-f64",
        1_024,
        256,
        false,
        &filter,
    );
    bench_householder_explicit_v(
        "linalg-householder-explicit-v-1024x256-c64",
        1_024,
        256,
        true,
        &filter,
    );
}
