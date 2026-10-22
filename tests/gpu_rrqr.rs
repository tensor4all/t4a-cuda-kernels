//! Numerical GPU tests for ABI revision 8: the rank-revealing QR primitives.
//!
//! Every entry is planned by the facade, launched through the raw CUDA driver API
//! (one physical launch per call, counted through `gpu::launches`) and compared
//! against a CPU reference built from the same layouts, so a mis-mapped stride,
//! offset or batch axis cannot pass. The kernels under test are translated from
//! tenferro-rs `14bc1ab933e3c7a9c966e08355670331ab9d993c`
//! (`crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs`), Copyright
//! (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors, MIT OR Apache-2.0.
//!
//! Copying entries are compared byte for byte, including NaN payloads; the
//! arithmetic entries (norms, reflector, rank) use a relative tolerance. Every
//! reduction entry is exercised with more than one 32-lane plane's worth of work,
//! and the flat entries above one 256-thread block.
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_rrqr`; requires a CUDA
//! device and `nvcc` on the invoking host.

mod gpu;

use gpu::{as_bytes, from_bytes, launch, launches, read_bytes, upload, Buf, Session, Slot};
use t4a_cuda_kernels::{
    plan_apply_reflector, plan_column_norms, plan_extract_r, plan_initialize_permutation,
    plan_prefix_rank, plan_reflector, plan_select_pivot, plan_swap_columns, CLayout, DType, Launch,
    Layout,
};

const SENTINEL: u8 = 0xCD;
/// Bytes reserved behind a logical output to catch a write past `len`.
const GUARD: usize = 13;
/// A domain above one 256-thread block, and not a multiple of it.
const BIG: i64 = 1003;
/// Elements reserved for a logically empty output, so a stray write is visible.
const EMPTY_ALLOC: usize = 5;

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

/// The payload types the entries under test accept, with the real-arithmetic view
/// the CPU references need.
trait El: Copy + std::fmt::Debug {
    fn zero() -> Self;
    /// `(re, im)` rounded to the payload precision.
    fn parts(re: f64, im: f64) -> Self;
    fn re(self) -> f64;
    fn im(self) -> f64;
    /// Magnitude in the payload's component precision, widened to f64.
    fn mag(self) -> f64;
}

impl El for f64 {
    fn zero() -> f64 {
        0.0
    }
    fn parts(re: f64, _im: f64) -> f64 {
        re
    }
    fn re(self) -> f64 {
        self
    }
    fn im(self) -> f64 {
        0.0
    }
    fn mag(self) -> f64 {
        self.abs()
    }
}

impl El for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn parts(re: f64, _im: f64) -> f32 {
        re as f32
    }
    fn re(self) -> f64 {
        self as f64
    }
    fn im(self) -> f64 {
        0.0
    }
    fn mag(self) -> f64 {
        self.abs() as f64
    }
}

impl El for C64 {
    fn zero() -> C64 {
        C64 { re: 0.0, im: 0.0 }
    }
    fn parts(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
    fn mag(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl El for C32 {
    fn zero() -> C32 {
        C32 { re: 0.0, im: 0.0 }
    }
    fn parts(re: f64, im: f64) -> C32 {
        C32 {
            re: re as f32,
            im: im as f32,
        }
    }
    fn re(self) -> f64 {
        self.re as f64
    }
    fn im(self) -> f64 {
        self.im as f64
    }
    fn mag(self) -> f64 {
        // The kernel's magnitude is a single-precision hypot.
        ((self.re as f64).hypot(self.im as f64) as f32) as f64
    }
}

impl El for i32 {
    fn zero() -> i32 {
        0
    }
    fn parts(re: f64, _im: f64) -> i32 {
        re as i32
    }
    fn re(self) -> f64 {
        self as f64
    }
    fn im(self) -> f64 {
        0.0
    }
    fn mag(self) -> f64 {
        (self as f64).abs()
    }
}

impl El for i64 {
    fn zero() -> i64 {
        0
    }
    fn parts(re: f64, _im: f64) -> i64 {
        re as i64
    }
    fn re(self) -> f64 {
        self as f64
    }
    fn im(self) -> f64 {
        0.0
    }
    fn mag(self) -> f64 {
        (self as f64).abs()
    }
}

/// The dtype of a `norms` or `rank` array: the real component type of the payload.
fn component(dtype: DType) -> DType {
    match dtype {
        DType::C32 => DType::F32,
        DType::C64 => DType::F64,
        other => other,
    }
}

/// Relative tolerance for the arithmetic an entry performs on this payload.
fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 1e-4,
        _ => 1e-12,
    }
}

// --- harness helpers --------------------------------------------------------

fn sentinel(session: &Session, bytes: usize) -> Buf {
    gpu::written(session, &vec![SENTINEL; bytes])
}

fn zeroed(session: &Session, bytes: usize) -> Buf {
    gpu::written(session, &vec![0u8; bytes])
}

/// A caller-zeroed payload with a sentinel guard behind it.
fn zeroed_guarded(session: &Session, payload: usize) -> Buf {
    let mut bytes = vec![0u8; payload];
    bytes.extend(std::iter::repeat_n(SENTINEL, GUARD));
    gpu::written(session, &bytes)
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

/// Index of one logical position inside a logical-order image (axis 0 fastest),
/// i.e. the layout's flat logical index rather than an element offset.
fn logical_index_of(layout: &Layout, coords: &[i64]) -> usize {
    let mut index = 0i64;
    let mut stride = 1i64;
    for (axis, &coordinate) in coords.iter().enumerate() {
        index += coordinate * stride;
        stride *= layout.extents()[axis];
    }
    assert!(index >= 0);
    index as usize
}

fn at_logical<T: Copy>(image: &[T], layout: &Layout, coords: &[i64]) -> T {
    image[logical_index_of(layout, coords)]
}

/// Element indices a layout addresses, in flat logical order (axis 0 fastest).
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
    assert!(layout.min_reachable_offset().unwrap() >= 0);
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

/// A span-sized host image with `values` at the layout's logical positions and the
/// zero of the payload everywhere the layout does not address.
fn host_image<T: El>(layout: &Layout, values: &[T]) -> Vec<T> {
    let mut host = vec![T::zero(); span(layout)];
    place(&mut host, layout, values);
    host
}

/// A read-only operand upload.
fn operand<T: El>(session: &Session, layout: &Layout, values: &[T]) -> Buf {
    upload(session, &host_image(layout, values))
}

/// An in-place operand: `values` at the layout's logical positions, the sentinel in
/// every hole and in a guard region behind the span, so the entry must leave both
/// untouched.
fn guarded_operand<T: El>(session: &Session, layout: &Layout, values: &[T]) -> Buf {
    let element = std::mem::size_of::<T>();
    let mut bytes = vec![SENTINEL; span(layout) * element + GUARD];
    for (&index, value) in view_indices(layout).iter().zip(values) {
        let offset = index as usize * element;
        bytes[offset..offset + element].copy_from_slice(as_bytes(&[*value]));
    }
    gpu::written(session, &bytes)
}

/// A sentinel-filled destination sized for the layout's reachable span, plus that
/// span as the capacity the facade is told. A logically empty layout still gets a
/// real allocation.
fn destination(session: &Session, layout: &Layout, dtype: DType) -> (Buf, i64) {
    let elements = span(layout);
    let allocation = elements.max(EMPTY_ALLOC);
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

/// Bit-exact element comparison: a NaN payload and a zero sign must survive
/// unchanged, so only the raw bytes may be compared.
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

/// Assert that the holes a layout does not address and the guard behind the span
/// still hold the sentinel.
fn assert_untouched_holes(bytes: &[u8], layout: &Layout, element: usize, what: &str) {
    let elements = span(layout);
    let mut addressed = vec![false; elements];
    for index in view_indices(layout) {
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
}

/// The holes and the guard are untouched; return the logical values.
fn pick_untouched<T: Copy>(bytes: &[u8], layout: &Layout, what: &str) -> Vec<T> {
    let element = std::mem::size_of::<T>();
    assert_untouched_holes(bytes, layout, element, what);
    pick(&host_of::<T>(bytes, span(layout)), layout)
}

fn is_sentinel(bytes: &[u8], layout: &Layout, coords: &[i64], element: usize) -> bool {
    let offset = index_of(layout, coords) * element;
    bytes[offset..offset + element]
        .iter()
        .all(|&byte| byte == SENTINEL)
}

fn read_value<T: Copy>(bytes: &[u8], layout: &Layout, coords: &[i64]) -> T {
    let element = std::mem::size_of::<T>();
    let offset = index_of(layout, coords) * element;
    from_bytes::<T>(&bytes[offset..offset + element])[0]
}

/// One `norms` or `rank` slot as f64, whatever the real component type is.
fn norm_at(dtype: DType, bytes: &[u8], layout: &Layout, coords: &[i64]) -> f64 {
    match component(dtype) {
        DType::F32 => read_value::<f32>(bytes, layout, coords) as f64,
        other => {
            assert_eq!(other, DType::F64);
            read_value::<f64>(bytes, layout, coords)
        }
    }
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

/// Relative closeness; a NaN only matches a NaN.
fn close_rel(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if expected == 0.0 {
        return actual == 0.0;
    }
    (actual - expected).abs() <= tolerance * expected.abs()
}

/// Absolute closeness relative to `scale`.
fn close(actual: f64, expected: f64, tolerance: f64, scale: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    (actual - expected).abs() <= tolerance * scale.max(f64::MIN_POSITIVE)
}

/// The 2-norm of `magnitudes` in the baseline's max-scale form: the plane maximum
/// first, then the scaled squares, then `scale * sqrt(sum)`.
fn max_scale_norm(magnitudes: &[f64]) -> f64 {
    let scale = magnitudes
        .iter()
        .fold(0.0f64, |scale, &value| scale.max(value.abs()));
    if scale > 0.0 {
        let sum: f64 = magnitudes
            .iter()
            .map(|value| (value.abs() / scale).powi(2))
            .sum();
        scale * sum.sqrt()
    } else {
        0.0
    }
}

// --- entry wrappers ---------------------------------------------------------

fn initialize_permutation_bytes(session: &Session, n: i64, len: i64) -> Vec<u8> {
    let out = sentinel(session, len.max(0) as usize * 8 + GUARD);
    let plan = plan_initialize_permutation(n, len).unwrap();
    let mut slots = [Slot::ptr(&out), Slot::i64(n), Slot::i64(len)];
    run(
        session,
        "t4a_cuda_initialize_permutation",
        &plan,
        &mut slots,
    );
    read_bytes(session, &out)
}

fn column_norms_bytes(
    session: &Session,
    dtype: DType,
    work_layout: &Layout,
    work: &Buf,
    norms_layout: &Layout,
    step: i64,
) -> Vec<u8> {
    let (norms, norms_len) = destination(session, norms_layout, component(dtype));
    let plan = plan_column_norms(
        dtype,
        work_layout,
        span(work_layout) as i64,
        norms_layout,
        norms_len,
        step,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(work),
        Slot::ptr(&norms),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(work_layout)),
        Slot::layout(&CLayout::new(norms_layout)),
        Slot::i64(step),
    ];
    run(session, "t4a_cuda_column_norms", &plan, &mut slots);
    read_bytes(session, &norms)
}

/// `norms` and `permutation` are the logical values of the two `[columns, batches]`
/// layouts, `pivots` and `status` one flat word per batch.
fn pivot_bytes<F: El>(
    session: &Session,
    dtype: DType,
    norms: &[F],
    permutation: &[i64],
    batches: i64,
    step: i64,
) -> (Vec<i64>, Vec<i64>) {
    let columns = norms.len() as i64 / batches;
    assert_eq!(permutation.len() as i64, columns * batches);
    // Neither array is compact, so a compact-layout assumption cannot pass.
    let norms_layout = Layout::new(&[columns, batches], &[1, columns + 2], 1).unwrap();
    let permutation_layout = Layout::new(&[columns, batches], &[1, columns + 1], 0).unwrap();
    let norms_buf = operand(session, &norms_layout, norms);
    let permutation_buf = operand(session, &permutation_layout, permutation);
    let words = batches.max(1) as usize;
    let pivots = sentinel(session, words * 8 + GUARD);
    let status = zeroed_guarded(session, words * 8);
    let plan = plan_select_pivot(
        dtype,
        &norms_layout,
        span(&norms_layout) as i64,
        &permutation_layout,
        span(&permutation_layout) as i64,
        batches,
        batches,
        step,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&norms_buf),
        Slot::ptr(&permutation_buf),
        Slot::ptr(&pivots),
        Slot::ptr(&status),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(&norms_layout)),
        Slot::layout(&CLayout::new(&permutation_layout)),
        Slot::i64(step),
    ];
    run(session, "t4a_cuda_select_pivot", &plan, &mut slots);
    let pivot_bytes = read_bytes(session, &pivots);
    let status_bytes = read_bytes(session, &status);
    assert_guard(&pivot_bytes, words * 8, "select_pivot pivots");
    assert_guard(&status_bytes, words * 8, "select_pivot status");
    (host_of(&pivot_bytes, words), host_of(&status_bytes, words))
}

/// In place: returns the whole work and permutation buffers, guard included.
fn swap_columns_bytes(
    session: &Session,
    dtype: DType,
    work_layout: &Layout,
    work: &Buf,
    permutation_layout: &Layout,
    permutation: &Buf,
    pivots: &[i64],
    step: i64,
) -> (Vec<u8>, Vec<u8>) {
    let pivots_buf = upload(session, pivots);
    let plan = plan_swap_columns(
        dtype,
        work_layout,
        span(work_layout) as i64,
        permutation_layout,
        span(permutation_layout) as i64,
        pivots.len() as i64,
        step,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(work),
        Slot::ptr(permutation),
        Slot::ptr(&pivots_buf),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(work_layout)),
        Slot::layout(&CLayout::new(permutation_layout)),
        Slot::i64(step),
    ];
    run(session, "t4a_cuda_swap_columns", &plan, &mut slots);
    (read_bytes(session, work), read_bytes(session, permutation))
}

/// In place on `work`: returns the whole work and coefficient buffers.
fn reflector_bytes(
    session: &Session,
    dtype: DType,
    work_layout: &Layout,
    work: &Buf,
    coeff_layout: &Layout,
    step: i64,
) -> (Vec<u8>, Vec<u8>) {
    let (coeff, coeff_len) = destination(session, coeff_layout, dtype);
    let plan = plan_reflector(
        dtype,
        work_layout,
        span(work_layout) as i64,
        coeff_layout,
        coeff_len,
        step,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(work),
        Slot::ptr(&coeff),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(work_layout)),
        Slot::layout(&CLayout::new(coeff_layout)),
        Slot::i64(step),
    ];
    run(session, "t4a_cuda_reflector", &plan, &mut slots);
    (read_bytes(session, work), read_bytes(session, &coeff))
}

/// In place on `target`: returns the whole target buffer.
#[allow(clippy::too_many_arguments)]
fn apply_reflector_bytes(
    session: &Session,
    dtype: DType,
    packed_layout: &Layout,
    packed: &Buf,
    target_layout: &Layout,
    target: &Buf,
    coeff_layout: &Layout,
    coeff: &Buf,
    step: i64,
    first_column: i64,
    column_count: i64,
) -> Vec<u8> {
    let plan = plan_apply_reflector(
        dtype,
        packed_layout,
        span(packed_layout) as i64,
        target_layout,
        span(target_layout) as i64,
        coeff_layout,
        span(coeff_layout) as i64,
        step,
        first_column,
        column_count,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(packed),
        Slot::ptr(target),
        Slot::ptr(coeff),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(packed_layout)),
        Slot::layout(&CLayout::new(target_layout)),
        Slot::layout(&CLayout::new(coeff_layout)),
        Slot::i64(step),
        Slot::i64(first_column),
        Slot::i64(column_count),
    ];
    run(session, "t4a_cuda_apply_reflector", &plan, &mut slots);
    read_bytes(session, target)
}

fn extract_r_bytes(
    session: &Session,
    dtype: DType,
    work_layout: &Layout,
    work: &Buf,
    r_layout: &Layout,
) -> Vec<u8> {
    let (r, r_len) = destination(session, r_layout, dtype);
    let plan = plan_extract_r(
        dtype,
        work_layout,
        span(work_layout) as i64,
        r_layout,
        r_len,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(work),
        Slot::ptr(&r),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(work_layout)),
        Slot::layout(&CLayout::new(r_layout)),
    ];
    run(session, "t4a_cuda_extract_r", &plan, &mut slots);
    read_bytes(session, &r)
}

fn prefix_rank_bytes(
    session: &Session,
    dtype: DType,
    r_layout: &Layout,
    r: &Buf,
    batches: i64,
    rtol: f64,
    atol: f64,
) -> (Vec<i64>, Vec<i64>) {
    let words = batches.max(1) as usize;
    let rank = sentinel(session, words * 8 + GUARD);
    let status = zeroed_guarded(session, words * 8);
    let plan = plan_prefix_rank(
        dtype,
        r_layout,
        span(r_layout) as i64,
        batches,
        batches,
        rtol,
        atol,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(r),
        Slot::ptr(&rank),
        Slot::ptr(&status),
        Slot::i32(dtype.tag()),
        Slot::layout(&CLayout::new(r_layout)),
        Slot::f64(rtol),
        Slot::f64(atol),
    ];
    run(session, "t4a_cuda_prefix_rank", &plan, &mut slots);
    let rank_bytes = read_bytes(session, &rank);
    let status_bytes = read_bytes(session, &status);
    assert_guard(&rank_bytes, words * 8, "prefix_rank rank");
    assert_guard(&status_bytes, words * 8, "prefix_rank status");
    (host_of(&rank_bytes, words), host_of(&status_bytes, words))
}

// --- references -------------------------------------------------------------

fn assert_reflector_identity<E: El>(
    original: &[E],
    post: &[E],
    coefficient: E,
    tolerance: f64,
    what: &str,
) {
    assert_eq!(post.len(), original.len(), "{what}: length");
    let length = original.len();
    // The stored vector: one at the step, the entries below it after the call.
    let v_re: Vec<f64> = std::iter::once(1.0)
        .chain(post[1..].iter().map(|value| value.re()))
        .collect();
    let v_im: Vec<f64> = std::iter::once(0.0)
        .chain(post[1..].iter().map(|value| value.im()))
        .collect();
    let x_re: Vec<f64> = original.iter().map(|value| value.re()).collect();
    let x_im: Vec<f64> = original.iter().map(|value| value.im()).collect();
    let (coefficient_re, coefficient_im) = (coefficient.re(), coefficient.im());
    let norm = max_scale_norm(&original.iter().map(|value| value.mag()).collect::<Vec<_>>());
    assert!(norm > 0.0, "{what}: the fixture must not be a zero column");
    let alpha = post[0];
    assert!(
        close(alpha.mag(), norm, tolerance, norm),
        "{what}: |alpha| is {} and the column 2-norm is {norm}",
        alpha.mag()
    );
    let x0_abs = x_re[0].hypot(x_im[0]);
    assert!(
        close(
            coefficient_re,
            1.0 + x0_abs / norm,
            tolerance,
            1.0 + x0_abs / norm
        ) && close(coefficient_im, 0.0, tolerance, 1.0),
        "{what}: coefficient is {:?}, expected {}",
        coefficient,
        1.0 + x0_abs / norm
    );
    // H x with H = I - coeff v v^H must be alpha at the step and zero below it.
    for row in 0..length {
        let mut dot_re = 0.0;
        let mut dot_im = 0.0;
        for (j, (&v_re, &v_im)) in v_re.iter().zip(&v_im).enumerate() {
            dot_re += v_re * x_re[j] + v_im * x_im[j];
            dot_im += v_re * x_im[j] - v_im * x_re[j];
        }
        let scaled_re = coefficient_re * dot_re - coefficient_im * dot_im;
        let scaled_im = coefficient_re * dot_im + coefficient_im * dot_re;
        let updated_re = x_re[row] - (v_re[row] * scaled_re - v_im[row] * scaled_im);
        let updated_im = x_im[row] - (v_re[row] * scaled_im + v_im[row] * scaled_re);
        let (expected_re, expected_im) = if row == 0 {
            (alpha.re(), alpha.im())
        } else {
            (0.0, 0.0)
        };
        assert!(
            close(updated_re, expected_re, tolerance, norm)
                && close(updated_im, expected_im, tolerance, norm),
            "{what}: row {row} of H x is ({updated_re}, {updated_im}), expected ({expected_re}, {expected_im})"
        );
    }
}

// --- initialize_permutation -------------------------------------------------

#[test]
fn initialize_permutation_is_the_modulo_image() {
    let session = Session::new("rrqr");
    for n in [1i64, 3, 7] {
        let bytes = initialize_permutation_bytes(&session, n, BIG);
        let values: Vec<i64> = host_of(&bytes, BIG as usize);
        for (pos, &value) in values.iter().enumerate() {
            assert_eq!(
                value,
                pos as i64 % n,
                "initialize_permutation n {n} at position {pos}"
            );
        }
        assert_guard(&bytes, BIG as usize * 8, "initialize_permutation");
    }
}

// --- column_norms -----------------------------------------------------------

#[test]
fn column_norms_matches_the_cpu_two_norm() {
    let session = Session::new("rrqr");
    let rows = 40i64;
    let columns = 6i64;
    let batches = 3i64;
    // A strided batched work matrix: axis 0 is the row, so the column stride is
    // `rows + 3` and the batch stride leaves a pad column behind every batch.
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 3, (rows + 3) * columns + 5],
        2,
    )
    .unwrap();
    // The one-axis-shorter operand: axis 0 is the column, axes 1.. the batch.
    let norms = Layout::new(&[columns, batches], &[1, columns + 1], 1).unwrap();

    fn sample<E: El>(row: i64, column: i64, batch: i64) -> E {
        let base = (row * 7 + column * 13 + batch * 29) as f64;
        E::parts(base * 0.125 - 3.0, base * 0.0625 - 1.0)
    }
    fn case<E: El>(session: &Session, dtype: DType, work: &Layout, norms: &Layout) {
        let extents = work.extents().to_vec();
        let mut values = Vec::new();
        for batch in 0..extents[2] {
            for column in 0..extents[1] {
                for row in 0..extents[0] {
                    values.push(sample::<E>(row, column, batch));
                }
            }
        }
        let host = host_image(work, &values);
        let buffer = upload(session, &host);
        let element = component(dtype).size();
        let tolerance = tolerance(dtype);
        for step in [0i64, 2] {
            let bytes = column_norms_bytes(session, dtype, work, &buffer, norms, step);
            assert_untouched_holes(&bytes, norms, element, "column_norms");
            for batch in 0..extents[2] {
                for column in 0..extents[1] {
                    let coords = [column, batch];
                    if column < step {
                        // A leading column is outside the trailing domain: its slot
                        // must keep the destination sentinel.
                        assert!(
                            is_sentinel(&bytes, norms, &coords, element),
                            "column_norms: the slot of column {column} was written at step {step}"
                        );
                        continue;
                    }
                    let magnitudes: Vec<f64> = (step..extents[0])
                        .map(|row| at(&host, work, &[row, column, batch]).mag())
                        .collect();
                    let expected = max_scale_norm(&magnitudes);
                    let observed = norm_at(dtype, &bytes, norms, &coords);
                    assert!(
                        close_rel(observed, expected, tolerance),
                        "column_norms {dtype:?} step {step} batch {batch} column {column}: observed {observed}, expected {expected}"
                    );
                }
            }
        }
    }
    case::<f32>(&session, DType::F32, &work, &norms);
    case::<f64>(&session, DType::F64, &work, &norms);
    case::<C32>(&session, DType::C32, &work, &norms);
    case::<C64>(&session, DType::C64, &work, &norms);

    // Three rows: lanes 3..31 of each plane own no element and must contribute their
    // zero accumulator instead of returning before the shuffles.
    let short = Layout::new(&[3, 5, 2], &[1, 6, 5 * 6 + 2], 1).unwrap();
    let short_norms = Layout::new(&[5, 2], &[1, 6], 0).unwrap();
    case::<f64>(&session, DType::F64, &short, &short_norms);
    case::<C64>(&session, DType::C64, &short, &short_norms);
}

#[test]
fn column_norms_flags_nonfinite_and_overflowing_columns() {
    let session = Session::new("rrqr");
    let rows = 64i64;
    let columns = 5i64;
    let batches = 2i64;

    /// Column 0 zero, 1 a huge and a tiny magnitude, 2 a NaN, 3 an infinity and 4 a
    /// column whose norm overflows the payload.
    fn special<E: El>(column: i64, row: i64, huge: f64) -> E {
        match column {
            0 => E::zero(),
            1 => {
                if row % 2 == 0 {
                    E::parts(1e30, 0.0)
                } else {
                    E::parts(1e-30, 0.0)
                }
            }
            2 => {
                if row == 17 {
                    E::parts(f64::NAN, 0.0)
                } else {
                    E::parts(1.0, 0.5)
                }
            }
            3 => {
                if row == 5 {
                    E::parts(f64::INFINITY, 0.0)
                } else {
                    E::parts(2.0, 0.0)
                }
            }
            _ => E::parts(huge, 0.0),
        }
    }
    fn case<E: El>(session: &Session, dtype: DType, huge: f64, shape: (i64, i64, i64)) {
        let (rows, columns, batches) = shape;
        let work = Layout::new(
            &[rows, columns, batches],
            &[1, rows + 1, (rows + 1) * columns + 2],
            1,
        )
        .unwrap();
        let norms = Layout::new(&[columns, batches], &[1, columns + 1], 0).unwrap();
        let mut values = Vec::new();
        for _batch in 0..batches {
            for column in 0..columns {
                for row in 0..rows {
                    values.push(special::<E>(column, row, huge));
                }
            }
        }
        let buffer = operand(session, &work, &values);
        let bytes = column_norms_bytes(session, dtype, &work, &buffer, &norms, 0);
        assert_untouched_holes(&bytes, &norms, component(dtype).size(), "column_norms");
        let tolerance = tolerance(dtype);
        for batch in 0..batches {
            for column in 0..columns {
                let coords = [column, batch];
                let observed = norm_at(dtype, &bytes, &norms, &coords);
                match column {
                    0 => assert_eq!(
                        observed, 0.0,
                        "column_norms {dtype:?}: a zero column must give 0, observed {observed}"
                    ),
                    1 => {
                        let magnitudes: Vec<f64> =
                            (0..rows).map(|row| special::<E>(1, row, huge).mag()).collect();
                        let expected = max_scale_norm(&magnitudes);
                        assert!(
                            close_rel(observed, expected, tolerance),
                            "column_norms {dtype:?}: the max-scale form gave {observed}, expected {expected}"
                        );
                    }
                    _ => assert_eq!(
                        observed, -1.0,
                        "column_norms {dtype:?} column {column} batch {batch}: the negative sentinel is expected, observed {observed}"
                    ),
                }
            }
        }
    }
    let shape = (rows, columns, batches);
    case::<f32>(&session, DType::F32, 1e38, shape);
    case::<f64>(&session, DType::F64, 1e308, shape);
    case::<C32>(&session, DType::C32, 1e38, shape);
    case::<C64>(&session, DType::C64, 1e308, shape);
}

// --- select_pivot -----------------------------------------------------------

#[test]
fn select_pivot_takes_the_largest_norm_then_the_tie_breaks() {
    let session = Session::new("rrqr");
    let identity = |columns: i64| -> Vec<i64> { (0..columns).collect() };
    let f32_norms =
        |values: &[f64]| -> Vec<f32> { values.iter().map(|&value| value as f32).collect() };

    // (a) a clear maximum.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[1.0, 5.0, 3.0, 2.0, 4.0, 0.0]),
        &identity(6),
        1,
        0,
    );
    assert_eq!((pivots, status), (vec![1i64], vec![0i64]));

    // (b) two equal maxima: the smaller original column wins.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[1.0, 5.0, 5.0, 2.0]),
        &[0, 3, 1, 2],
        1,
        0,
    );
    assert_eq!((pivots, status), (vec![2i64], vec![0i64]));

    // (c) equal maxima and equal originals: the smaller column wins.
    let (pivots, status) =
        pivot_bytes(&session, DType::F32, &f32_norms(&[5.0, 5.0]), &[7, 7], 1, 0);
    assert_eq!((pivots, status), (vec![0i64], vec![0i64]));

    // (e) step > 0 leaves the leading columns out of the scan.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[9.0, 1.0, 4.0, 5.0]),
        &identity(4),
        1,
        2,
    );
    assert_eq!((pivots, status), (vec![3i64], vec![0i64]));
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[9.0, 1.0, 4.0, 3.0]),
        &identity(4),
        1,
        2,
    );
    assert_eq!((pivots, status), (vec![2i64], vec![0i64]));

    // (f) 96 columns, so every lane of the 32-lane plane holds three candidates.
    let wide: Vec<f32> = (0..96).map(|column| column as f32).collect();
    let (pivots, status) = pivot_bytes(&session, DType::F32, &wide, &identity(96), 1, 0);
    assert_eq!((pivots, status), (vec![95i64], vec![0i64]));
    // A tie between lanes 5 and 37: the smaller original column wins.
    let mut tie = vec![1.0f32; 96];
    tie[5] = 10.0;
    tie[37] = 10.0;
    let mut permutation = identity(96);
    permutation[5] = 2;
    permutation[37] = 1;
    let (pivots, status) = pivot_bytes(&session, DType::F32, &tie, &permutation, 1, 0);
    assert_eq!((pivots, status), (vec![37i64], vec![0i64]));
    // The same norms and originals on lanes 5 and 37: the smaller column wins.
    permutation[5] = 1;
    let (pivots, status) = pivot_bytes(&session, DType::F32, &tie, &permutation, 1, 0);
    assert_eq!((pivots, status), (vec![5i64], vec![0i64]));

    // Batched: each batch picks from its own plane.
    let mut batched = vec![1.0f32; 64];
    batched[31] = 4.0;
    batched[32] = 7.0;
    let batches: Vec<i64> = (0..2).flat_map(|_| 0..32).collect();
    let (pivots, status) = pivot_bytes(&session, DType::F32, &batched, &batches, 2, 0);
    assert_eq!((pivots, status), (vec![31i64, 0], vec![0i64, 0]));

    // A float64 norm array.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F64,
        &[1.0f64, 2.0, 3.5],
        &identity(3),
        1,
        0,
    );
    assert_eq!((pivots, status), (vec![2i64], vec![0i64]));
}

#[test]
fn select_pivot_flags_the_invalid_sentinel() {
    let session = Session::new("rrqr");
    let f32_norms =
        |values: &[f64]| -> Vec<f32> { values.iter().map(|&value| value as f32).collect() };

    // A negative norm raises the sticky status word and is never selected.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[1.0, -1.0, 3.0]),
        &[0, 1, 2],
        1,
        0,
    );
    assert_eq!((pivots, status), (vec![2i64], vec![1i64]));

    // Every candidate invalid: lane 0 falls back to the step and the status stays set.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[-1.0, -1.0]),
        &[0, 1],
        1,
        0,
    );
    assert_eq!((pivots, status), (vec![0i64], vec![1i64]));

    // A sentinel before the step is outside the scan.
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[-1.0, 4.0, 2.0]),
        &[0, 1, 2],
        1,
        1,
    );
    assert_eq!((pivots, status), (vec![1i64], vec![0i64]));

    // Batched: only the batch holding the sentinel raises its word.
    let batches: Vec<i64> = (0..2).flat_map(|_| 0..2).collect();
    let (pivots, status) = pivot_bytes(
        &session,
        DType::F32,
        &f32_norms(&[1.0, 2.0, -1.0, 3.0]),
        &batches,
        2,
        0,
    );
    assert_eq!((pivots, status), (vec![1i64, 1], vec![0i64, 1]));
}

// --- swap_columns -----------------------------------------------------------

#[test]
fn swap_columns_exchanges_the_matrix_and_the_permutation() {
    let session = Session::new("rrqr");
    let rows = 20i64;
    let columns = 7i64;
    let batches = 16i64;
    // 20 rows times 16 batches is 320 threads: above one 256-thread block.
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 2, (rows + 2) * columns + 3],
        1,
    )
    .unwrap();
    let permutation_layout = Layout::new(&[columns, batches], &[1, columns + 1], 0).unwrap();

    fn case<T: El>(
        session: &Session,
        dtype: DType,
        work: &Layout,
        permutation_layout: &Layout,
        pivots: &[i64],
        step: i64,
    ) {
        let rows = work.extents()[0];
        let columns = work.extents()[1];
        let batches = work.extents()[2];
        let mut work_values = Vec::new();
        let mut permutation_values = Vec::new();
        for batch in 0..batches {
            for column in 0..columns {
                for row in 0..rows {
                    work_values.push(T::parts((row * 3 + column * 17 + batch * 101) as f64, 0.0));
                }
            }
            for column in 0..columns {
                permutation_values.push(column + 7 * batch);
            }
        }
        let original = pick(&host_image(work, &work_values), work);
        let permutation_original = pick(
            &host_image(permutation_layout, &permutation_values),
            permutation_layout,
        );
        let work_buf = guarded_operand(session, work, &work_values);
        let permutation_buf = guarded_operand(session, permutation_layout, &permutation_values);
        let (work_bytes, permutation_bytes) = swap_columns_bytes(
            session,
            dtype,
            work,
            &work_buf,
            permutation_layout,
            &permutation_buf,
            pivots,
            step,
        );
        let mut expected = original.clone();
        let mut expected_permutation = permutation_original.clone();
        for (batch, &pivot) in pivots.iter().enumerate() {
            if pivot == step {
                continue;
            }
            let batch = batch as i64;
            for row in 0..rows {
                let left = logical_index_of(work, &[row, step, batch]);
                let right = logical_index_of(work, &[row, pivot, batch]);
                expected.swap(left, right);
            }
            let left = logical_index_of(permutation_layout, &[step, batch]);
            let right = logical_index_of(permutation_layout, &[pivot, batch]);
            expected_permutation.swap(left, right);
        }
        assert_bits(
            &pick_untouched::<T>(&work_bytes, work, "swap_columns work"),
            &expected,
            "swap_columns work",
        );
        assert_bits(
            &pick_untouched::<i64>(&permutation_bytes, permutation_layout, "swap_columns"),
            &expected_permutation,
            "swap_columns permutation",
        );
    }

    // A no-op pivot for every batch, then a real swap for most of them; `step = 0`
    // moves the leading column as well.
    let no_op = vec![2i64; batches as usize];
    let mixed: Vec<i64> = (0..batches)
        .map(|batch| if batch % 3 == 0 { 2 } else { 3 + batch % 4 })
        .collect();
    let leading: Vec<i64> = (0..batches).map(|batch| 1 + batch % 6).collect();
    case::<f64>(&session, DType::F64, &work, &permutation_layout, &no_op, 2);
    case::<f64>(&session, DType::F64, &work, &permutation_layout, &mixed, 2);
    case::<f64>(
        &session,
        DType::F64,
        &work,
        &permutation_layout,
        &leading,
        0,
    );
    case::<i64>(&session, DType::I64, &work, &permutation_layout, &no_op, 2);
    case::<i64>(&session, DType::I64, &work, &permutation_layout, &mixed, 2);
    case::<i64>(
        &session,
        DType::I64,
        &work,
        &permutation_layout,
        &leading,
        0,
    );
    // The remaining payload arms: single precision, signed 32-bit and both
    // complex types.
    case::<f32>(&session, DType::F32, &work, &permutation_layout, &mixed, 2);
    case::<i32>(&session, DType::I32, &work, &permutation_layout, &mixed, 2);
    case::<C32>(&session, DType::C32, &work, &permutation_layout, &mixed, 2);
    case::<C64>(&session, DType::C64, &work, &permutation_layout, &mixed, 2);
    case::<C32>(
        &session,
        DType::C32,
        &work,
        &permutation_layout,
        &leading,
        0,
    );
}

// --- reflector --------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn reflector_case<E: El>(
    session: &Session,
    dtype: DType,
    work: &Layout,
    coeff: &Layout,
    values: &[E],
    step: i64,
    zero_column: bool,
) {
    let rows = work.extents()[0];
    let batches: i64 = work.extents()[2..].iter().product();
    let tolerance = tolerance(dtype);
    let original = pick(&host_image(work, values), work);
    let work_buf = guarded_operand(session, work, values);
    let (work_bytes, coeff_bytes) = reflector_bytes(session, dtype, work, &work_buf, coeff, step);
    let post = pick_untouched::<E>(&work_bytes, work, "reflector work");
    let coefficients = pick_untouched::<E>(&coeff_bytes, coeff, "reflector coeff");
    let mut expected = original.clone();
    for batch in 0..batches {
        let column = |image: &[E]| -> Vec<E> {
            (step..rows)
                .map(|row| at_logical(image, work, &[row, step, batch]))
                .collect()
        };
        let x = column(&original);
        let packed = column(&post);
        let coefficient = coefficients[logical_index_of(coeff, &[step, batch])];
        if zero_column {
            assert_eq!(
                as_bytes(std::slice::from_ref(&coefficient)),
                as_bytes(&[E::zero()]),
                "reflector {dtype:?}: a zero column must store a zero coefficient, observed {coefficient:?}"
            );
            assert_bits(&packed, &x, "reflector: a zero column must stay unchanged");
        } else {
            assert_reflector_identity(&x, &packed, coefficient, tolerance, "reflector");
            for (offset, row) in (step..rows).enumerate() {
                expected[logical_index_of(work, &[row, step, batch])] = packed[offset];
            }
        }
    }
    // Nothing outside the reflector column moved, and the coefficient slots outside
    // the step keep the destination sentinel.
    assert_bits(&post, &expected, "reflector: a position outside the column");
    let element = std::mem::size_of::<E>();
    for batch in 0..batches {
        for column in 0..coeff.extents()[0] {
            assert!(
                column == step
                    || is_sentinel(&coeff_bytes, coeff, &[column, batch], element),
                "reflector: the coefficient slot of column {column} batch {batch} was written at step {step}"
            );
        }
    }
}

#[test]
fn reflector_keeps_its_defining_identity() {
    let session = Session::new("rrqr");
    let rows = 40i64;
    let columns = 4i64;
    let batches = 2i64;
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 2, (rows + 2) * columns + 1],
        1,
    )
    .unwrap();
    let coeff = Layout::new(&[columns, batches], &[1, columns + 1], 0).unwrap();

    /// The column under the reflector gets `x0` at the step and a varied tail.
    fn fixture<E: El>(column: i64, row: i64, step: i64, x0: (f64, f64)) -> E {
        if column != step {
            E::parts((row * 5 + column) as f64 * 0.25 - 2.0, 0.5)
        } else if row == step {
            E::parts(x0.0, x0.1)
        } else {
            E::parts(row as f64 * 0.375 - 4.0, 0.25)
        }
    }
    fn case<E: El>(session: &Session, dtype: DType, work: &Layout, coeff: &Layout) {
        let rows = work.extents()[0];
        let columns = work.extents()[1];
        let batches: i64 = work.extents()[2..].iter().product();
        let build = |step: i64, x0: (f64, f64)| -> Vec<E> {
            let mut values = Vec::new();
            for _batch in 0..batches {
                for column in 0..columns {
                    for row in 0..rows {
                        values.push(fixture::<E>(column, row, step, x0));
                    }
                }
            }
            values
        };
        for step in [0i64, 1] {
            // A negative and a positive x0.
            for x0 in [(-3.5, 0.0), (2.25, 0.0)] {
                reflector_case(session, dtype, work, coeff, &build(step, x0), step, false);
            }
            // A complex x0 with both components non-zero.
            reflector_case(
                session,
                dtype,
                work,
                coeff,
                &build(step, (2.0, -1.5)),
                step,
                false,
            );
        }
    }
    case::<f32>(&session, DType::F32, &work, &coeff);
    case::<f64>(&session, DType::F64, &work, &coeff);
    case::<C32>(&session, DType::C32, &work, &coeff);
    case::<C64>(&session, DType::C64, &work, &coeff);

    // Three rows: only lanes 0..2 of each plane own an element.
    let short = Layout::new(&[3, 2, 2], &[1, 5, 2 * 5 + 1], 0).unwrap();
    let short_coeff = Layout::new(&[2, 2], &[1, 3], 0).unwrap();
    case::<f64>(&session, DType::F64, &short, &short_coeff);
}

#[test]
fn reflector_zero_column_writes_only_the_coefficient() {
    let session = Session::new("rrqr");
    let rows = 40i64;
    let columns = 4i64;
    let batches = 2i64;
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 1, (rows + 1) * columns],
        0,
    )
    .unwrap();
    let coeff = Layout::compact(&[columns, batches]).unwrap();
    fn case<E: El>(session: &Session, dtype: DType, work: &Layout, coeff: &Layout) {
        // Column `step` is exactly zero, so the reflector is the identity.
        let mut values = Vec::new();
        for _batch in 0..work.extents()[2] {
            for column in 0..work.extents()[1] {
                for row in 0..work.extents()[0] {
                    values.push(if column == 2 {
                        E::zero()
                    } else {
                        E::parts(row as f64 - 3.0, 0.5)
                    });
                }
            }
        }
        reflector_case(session, dtype, work, coeff, &values, 2, true);
    }
    case::<f64>(&session, DType::F64, &work, &coeff);
    case::<C32>(&session, DType::C32, &work, &coeff);
    case::<C64>(&session, DType::C64, &work, &coeff);
}

// --- apply_reflector --------------------------------------------------------

/// Assert the *first* application against the reflector's own definition: with `v`
/// equal to `1` at the step and the stored entries below it, the target column in
/// the window becomes `(I - coefficient * v v^H) x`. Columns outside the window and
/// rows above the step are untouched.
#[allow(clippy::too_many_arguments)]
fn assert_apply_reference<E: El>(
    original: &[E],
    single: &[E],
    packed: &[E],
    packed_layout: &Layout,
    target_layout: &Layout,
    coefficient: E,
    step: i64,
    first_column: i64,
    column_count: i64,
    tolerance: f64,
    scale: f64,
) {
    let rows = target_layout.extents()[0];
    let target_columns = target_layout.extents()[1];
    let batches: i64 = target_layout.extents()[2..].iter().product();
    let vector = |row: i64, batch: i64, image: &[E]| -> (f64, f64) {
        if row == step {
            (1.0, 0.0)
        } else {
            let value = at_logical(image, packed_layout, &[row, step, batch]);
            (value.re(), value.im())
        }
    };
    let (coefficient_re, coefficient_im) = (coefficient.re(), coefficient.im());
    for batch in 0..batches {
        for column in 0..target_columns {
            let applied = column >= first_column && column < first_column + column_count;
            // The plane dot of the kernel: sum of conj(v) * x over the window.
            let (mut dot_re, mut dot_im) = (0.0, 0.0);
            for row in step..rows {
                let (v_re, v_im) = vector(row, batch, packed);
                let value = at_logical(original, target_layout, &[row, column, batch]);
                dot_re += v_re * value.re() + v_im * value.im();
                dot_im += v_re * value.im() - v_im * value.re();
            }
            let scaled_re = coefficient_re * dot_re - coefficient_im * dot_im;
            let scaled_im = coefficient_re * dot_im + coefficient_im * dot_re;
            for row in 0..rows {
                let coords = [row, column, batch];
                let expected = at_logical(original, target_layout, &coords);
                let observed = at_logical(single, target_layout, &coords);
                if !applied || row < step {
                    assert!(
                        as_bytes(std::slice::from_ref(&observed))
                            == as_bytes(std::slice::from_ref(&expected)),
                        "apply_reflector reference: row {row} column {column} batch {batch} is outside the reflector and must stay {expected:?}, observed {observed:?}"
                    );
                    continue;
                }
                let (v_re, v_im) = vector(row, batch, packed);
                let (subtracted_re, subtracted_im) = (
                    v_re * scaled_re - v_im * scaled_im,
                    v_re * scaled_im + v_im * scaled_re,
                );
                let reference_re = expected.re() - subtracted_re;
                let reference_im = expected.im() - subtracted_im;
                // The update subtracts a term of magnitude `|v * coefficient * dot|`, so
                // that term, not the small result, bounds the round-off of both sides.
                let bound = scale.max(subtracted_re.hypot(subtracted_im));
                assert!(
                    close(observed.re(), reference_re, tolerance, bound)
                        && close(observed.im(), reference_im, tolerance, bound),
                    "apply_reflector reference: H x gave ({}, {}) at row {row} column {column} batch {batch}, expected ({reference_re}, {reference_im})",
                    observed.re(),
                    observed.im()
                );
            }
        }
    }
}

/// `coefficient` overrides the reflector-derived (always real) coefficient, which is
/// the only way to reach a complex one: with one supplied, `H` is not unitary and so
/// only the first application is checked against the reference.
#[allow(clippy::too_many_arguments)]
fn apply_reflector_case<E: El>(
    session: &Session,
    dtype: DType,
    work: &Layout,
    target_layout: &Layout,
    coeff_layout: &Layout,
    step: i64,
    first_column: i64,
    column_count: i64,
    coefficient: Option<(f64, f64)>,
) {
    let rows = work.extents()[0];
    let columns = work.extents()[1];
    let batches: i64 = work.extents()[2..].iter().product();
    let target_columns = target_layout.extents()[1];
    let tolerance = tolerance(dtype);
    // The reflector itself comes from the kernel under test.
    let mut packed_values = Vec::new();
    for _batch in 0..batches {
        for column in 0..columns {
            for row in 0..rows {
                packed_values.push(if column == step {
                    if row == step {
                        E::parts(-3.5, 0.5)
                    } else {
                        E::parts(row as f64 * 0.25 - 2.0, 0.75)
                    }
                } else {
                    E::parts((row + column) as f64, 0.0)
                });
            }
        }
    }
    let packed = guarded_operand(session, work, &packed_values);
    let (coeff, coefficients) = match coefficient {
        Some((re, im)) => {
            let values = vec![E::parts(re, im); (columns * batches) as usize];
            let buffer = operand(session, coeff_layout, &values);
            (buffer, values)
        }
        None => {
            let (_, coeff_bytes) =
                reflector_bytes(session, dtype, work, &packed, coeff_layout, step);
            let values = pick(
                &host_of::<E>(&coeff_bytes, span(coeff_layout)),
                coeff_layout,
            );
            (gpu::written(session, &coeff_bytes), values)
        }
    };
    // The target: the column at `first_column + 1` is entirely zero, so it must stay
    // zero, and the columns outside the applied range must not move at all.
    let zero_column = first_column + 1;
    let mut target_values = Vec::new();
    for _batch in 0..batches {
        for column in 0..target_columns {
            for row in 0..rows {
                target_values.push(if column == zero_column {
                    E::zero()
                } else {
                    E::parts((row * 3 + column) as f64 * 0.125 - 1.0, 0.5)
                });
            }
        }
    }
    let target = guarded_operand(session, target_layout, &target_values);
    let once = apply_reflector_bytes(
        session,
        dtype,
        work,
        &packed,
        target_layout,
        &target,
        coeff_layout,
        &coeff,
        step,
        first_column,
        column_count,
    );
    let original = pick(&host_image(target_layout, &target_values), target_layout);
    let single = pick_untouched::<E>(&once, target_layout, "apply_reflector");
    let packed_post = pick(
        &host_of::<E>(&read_bytes(session, &packed), span(work)),
        work,
    );
    let scale = original
        .iter()
        .fold(1.0f64, |scale, value| scale.max(value.mag()));
    for batch in 0..batches {
        assert_apply_reference(
            &original,
            &single,
            &packed_post,
            work,
            target_layout,
            at_logical(&coefficients, coeff_layout, &[step, batch]),
            step,
            first_column,
            column_count,
            tolerance,
            scale,
        );
    }
    if coefficient.is_some() {
        return;
    }
    let again = gpu::written(session, &once);
    let twice = apply_reflector_bytes(
        session,
        dtype,
        work,
        &packed,
        target_layout,
        &again,
        coeff_layout,
        &coeff,
        step,
        first_column,
        column_count,
    );
    let applied_twice = pick_untouched::<E>(&twice, target_layout, "apply_reflector twice");
    for batch in 0..batches {
        for column in 0..target_columns {
            let applied = column >= first_column && column < first_column + column_count;
            for row in 0..rows {
                let coords = [row, column, batch];
                let expected = at_logical(&original, target_layout, &coords);
                let observed = at_logical(&applied_twice, target_layout, &coords);
                if !applied || row < step {
                    assert!(
                        as_bytes(std::slice::from_ref(&observed))
                            == as_bytes(std::slice::from_ref(&expected)),
                        "apply_reflector {dtype:?}: row {row} column {column} batch {batch} is outside the reflector and must stay {expected:?}, observed {observed:?}"
                    );
                } else if column == zero_column {
                    for (image, what) in [(&single, "once"), (&applied_twice, "twice")] {
                        let value = at_logical(image, target_layout, &coords);
                        assert_eq!(
                            as_bytes(std::slice::from_ref(&value)),
                            as_bytes(&[E::zero()]),
                            "apply_reflector: a zero target column must stay zero {what} row {row} column {column}"
                        );
                    }
                } else {
                    assert!(
                        close(observed.re(), expected.re(), tolerance, scale)
                            && close(observed.im(), expected.im(), tolerance, scale),
                        "apply_reflector {dtype:?}: applying the reflector twice gave {:?} at row {row} column {column} batch {batch}, expected {expected:?}",
                        observed
                    );
                }
            }
        }
    }
}

/// The fixtures: a 70-row packed matrix and a 5-column target, both with a batch
/// axis, and the reflector window starting past column 0.
fn apply_reflector_fixture() -> (Layout, Layout, Layout) {
    let (rows, columns, batches) = (70i64, 4i64, 2i64);
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 1, (rows + 1) * columns + 2],
        1,
    )
    .unwrap();
    let target = Layout::new(&[rows, 5, batches], &[1, rows + 3, (rows + 3) * 5 + 2], 0).unwrap();
    let coeff = Layout::new(&[columns, batches], &[1, columns + 1], 0).unwrap();
    (work, target, coeff)
}

#[test]
fn apply_reflector_real_is_an_involution() {
    let session = Session::new("rrqr");
    let (work, target, coeff) = apply_reflector_fixture();
    let (step, first_column, column_count) = (1i64, 1i64, 3i64);
    apply_reflector_case::<f32>(
        &session,
        DType::F32,
        &work,
        &target,
        &coeff,
        step,
        first_column,
        column_count,
        None,
    );
    apply_reflector_case::<f64>(
        &session,
        DType::F64,
        &work,
        &target,
        &coeff,
        step,
        first_column,
        column_count,
        None,
    );
}

#[test]
fn apply_reflector_complex_is_an_involution() {
    let session = Session::new("rrqr");
    let (work, target, coeff) = apply_reflector_fixture();
    let (step, first_column, column_count) = (1i64, 1i64, 3i64);
    apply_reflector_case::<C64>(
        &session,
        DType::C64,
        &work,
        &target,
        &coeff,
        step,
        first_column,
        column_count,
        None,
    );
    apply_reflector_case::<C32>(
        &session,
        DType::C32,
        &work,
        &target,
        &coeff,
        step,
        first_column,
        column_count,
        None,
    );
}

/// A caller-supplied complex coefficient: the kernel's complex product must carry it
/// (a `coefficient` read as its slot index cannot match this reference).
#[test]
fn apply_reflector_complex_coefficient_matches_the_reference() {
    let session = Session::new("rrqr");
    let (work, target, coeff) = apply_reflector_fixture();
    let (step, first_column, column_count) = (1i64, 1i64, 3i64);
    apply_reflector_case::<C64>(
        &session,
        DType::C64,
        &work,
        &target,
        &coeff,
        step,
        first_column,
        column_count,
        Some((0.75, -1.25)),
    );
}

/// Three rows: only lanes 0..2 of each plane own an element, so a lane-local return
/// before the shuffles changes the update.
#[test]
fn apply_reflector_short_window_uses_every_lane() {
    let session = Session::new("rrqr");
    let (rows, batches) = (3i64, 2i64);
    let work = Layout::new(&[rows, 4, batches], &[1, rows + 1, (rows + 1) * 4 + 2], 1).unwrap();
    let target = Layout::new(&[rows, 5, batches], &[1, rows + 2, (rows + 2) * 5 + 1], 0).unwrap();
    let coeff = Layout::new(&[4, batches], &[1, 6], 0).unwrap();
    apply_reflector_case::<f64>(&session, DType::F64, &work, &target, &coeff, 1, 1, 3, None);
}

// --- extract_r --------------------------------------------------------------

#[test]
fn extract_r_copies_the_upper_triangle_and_zeroes_the_rest() {
    let session = Session::new("rrqr");
    let rows = 50i64;
    let columns = 8i64;
    let batches = 3i64;
    let k = 20i64;
    let work = Layout::new(
        &[rows, columns, batches],
        &[1, rows + 2, (rows + 2) * columns + 1],
        1,
    )
    .unwrap();
    let r = Layout::new(
        &[k, columns, batches],
        &[1, k + 3, (k + 3) * columns + 2],
        2,
    )
    .unwrap();

    fn case<E: El>(session: &Session, dtype: DType, work: &Layout, r: &Layout) {
        let rows = work.extents()[0];
        let columns = work.extents()[1];
        let batches = work.extents()[2];
        let k = r.extents()[0];
        let element = dtype.size();
        // A distinctive NaN payload in the upper triangle must survive bit for bit.
        let nan = if element == 4 {
            E::parts(f32::from_bits(0x7fc0_1234) as f64, 0.0)
        } else {
            E::parts(f64::from_bits(0x7ff8_0000_0000_1234), 0.0)
        };
        let mut values = Vec::new();
        for _batch in 0..batches {
            for column in 0..columns {
                for row in 0..rows {
                    values.push(match (row, column) {
                        (2, 5) => nan,
                        (7, 3) => nan,
                        _ => E::parts((row * 3 + column * 7) as f64 * 0.25, 0.5),
                    });
                }
            }
        }
        let host = host_image(work, &values);
        let buffer = upload(session, &host);
        let bytes = extract_r_bytes(session, dtype, work, &buffer, r);
        // Build the whole expected buffer byte for byte: the upper triangle is a raw
        // copy, the lower one is the payload zero, holes and the guard are sentinel.
        let mut expected = vec![SENTINEL; span(r) * element + GUARD];
        for batch in 0..batches {
            for column in 0..columns {
                for row in 0..k {
                    let value = if row <= column {
                        at(&host, work, &[row, column, batch])
                    } else {
                        E::zero()
                    };
                    let offset = index_of(r, &[row, column, batch]) * element;
                    expected[offset..offset + element].copy_from_slice(as_bytes(&[value]));
                }
            }
        }
        assert_eq!(
            bytes, expected,
            "extract_r {dtype:?}: the mask, the zero below it, the holes and the guard"
        );
    }
    case::<f32>(&session, DType::F32, &work, &r);
    case::<f64>(&session, DType::F64, &work, &r);
    case::<C32>(&session, DType::C32, &work, &r);
    case::<C64>(&session, DType::C64, &work, &r);
}

// --- prefix_rank ------------------------------------------------------------

fn prefix_rank_case<E: El>(
    session: &Session,
    dtype: DType,
    k: i64,
    n: i64,
    batches: i64,
    diagonal: &[f64],
    rtol: f64,
    atol: f64,
) -> (Vec<i64>, Vec<i64>) {
    let pairs: Vec<(f64, f64)> = diagonal.iter().map(|&value| (value, 0.0)).collect();
    prefix_rank_pairs_case::<E>(session, dtype, k, n, batches, &pairs, rtol, atol)
}

/// The same fixture with a caller-chosen imaginary part on each diagonal.
fn prefix_rank_pairs_case<E: El>(
    session: &Session,
    dtype: DType,
    k: i64,
    n: i64,
    batches: i64,
    diagonal: &[(f64, f64)],
    rtol: f64,
    atol: f64,
) -> (Vec<i64>, Vec<i64>) {
    let r = Layout::new(&[k, n, batches], &[1, k + 2, (k + 2) * n + 1], 1).unwrap();
    let mut values = Vec::new();
    for batch in 0..batches {
        for column in 0..n {
            for row in 0..k {
                values.push(if row == column {
                    let (re, im) = diagonal[(row + k * batch) as usize];
                    E::parts(re, im)
                } else {
                    // Outside the contract: a value far above every threshold would
                    // turn this into a full rank if the kernel read it.
                    E::parts(1e9, 0.0)
                });
            }
        }
    }
    let buffer = operand(session, &r, &values);
    let (rank, status) = prefix_rank_bytes(session, dtype, &r, &buffer, batches, rtol, atol);
    // Only the diagonal and the flat arrays are read; every hole and the guard stay.
    (rank, status)
}

#[test]
fn prefix_rank_stops_at_the_first_failing_diagonal() {
    let session = Session::new("rrqr");
    // Full rank.
    let (rank, status) = prefix_rank_case::<f64>(
        &session,
        DType::F64,
        6,
        8,
        1,
        &[1.0, 0.5, 0.4, 0.3, 0.2, 0.1],
        1e-6,
        0.0,
    );
    assert_eq!((rank, status), (vec![6i64], vec![0i64]));

    // Rank 0: the leading diagonal is already below the tolerance.
    let (rank, status) = prefix_rank_case::<f64>(
        &session,
        DType::F64,
        6,
        8,
        1,
        &[1e-30, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-10,
        1e-20,
    );
    assert_eq!((rank, status), (vec![0i64], vec![0i64]));

    // A limited rank: the fourth diagonal fails and a later large one is not reached.
    let (rank, status) = prefix_rank_case::<f64>(
        &session,
        DType::F64,
        6,
        8,
        1,
        &[1.0, 0.5, 0.25, 1e-9, 5.0, 5.0],
        1e-6,
        0.0,
    );
    assert_eq!((rank, status), (vec![3i64], vec![0i64]));

    // rtol dominance: a relative tolerance below every diagonal keeps all six.
    let diagonal = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01];
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &diagonal, 5e-3, 0.0);
    assert_eq!((rank, status), (vec![6i64], vec![0i64]));
    // The comparison is strict, so a diagonal exactly at `rtol * leading` fails.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &diagonal, 1e-2, 0.0);
    assert_eq!((rank, status), (vec![5i64], vec![0i64]));
    // atol dominance: the absolute floor rejects the second diagonal.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &diagonal, 0.0, 0.5);
    assert_eq!((rank, status), (vec![1i64], vec![0i64]));
    // Both set: `atol > rtol * leading` wins.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &diagonal, 1e-2, 0.5);
    assert_eq!((rank, status), (vec![1i64], vec![0i64]));
    // rtol above atol wins, and its threshold rejects the last diagonal.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &diagonal, 1e-2, 1e-3);
    assert_eq!((rank, status), (vec![5i64], vec![0i64]));

    // A NaN diagonal: the status word is raised and the NaN counts as a failure.
    let (rank, status) = prefix_rank_case::<f64>(
        &session,
        DType::F64,
        6,
        6,
        1,
        &[1.0, 1.0, 1.0, f64::NAN, 1.0, 1.0],
        1e-8,
        0.0,
    );
    assert_eq!((rank, status), (vec![3i64], vec![1i64]));
    let (rank, status) = prefix_rank_case::<f64>(
        &session,
        DType::F64,
        6,
        6,
        1,
        &[f64::NAN, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-8,
        0.0,
    );
    assert_eq!((rank, status), (vec![0i64], vec![1i64]));
}

#[test]
fn prefix_rank_wraps_past_32_diagonals_and_stays_batched() {
    let session = Session::new("rrqr");
    // 64 diagonals: every lane walks the diagonal twice.
    let mut diagonal = vec![1.0f64; 64];
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 64, 64, 1, &diagonal, 1e-8, 0.0);
    assert_eq!((rank, status), (vec![64i64], vec![0i64]));
    diagonal[35] = 1e-30;
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 64, 64, 1, &diagonal, 1e-8, 0.0);
    assert_eq!(
        (rank, status),
        (vec![35i64], vec![0i64]),
        "the failure at diagonal 35 belongs to lane 3's second pass"
    );

    // Batched: a full, a limited and an invalid batch in one launch.
    let mut diagonal = vec![0.0f64; 3 * 40];
    for batch in 0..3i64 {
        for row in 0..40i64 {
            diagonal[(row + 40 * batch) as usize] = 1.0;
        }
    }
    diagonal[(7 + 40) as usize] = 0.0;
    diagonal[(3 + 80) as usize] = f64::NAN;
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 40, 44, 3, &diagonal, 1e-8, 0.0);
    assert_eq!((rank, status), (vec![40i64, 7, 3], vec![0i64, 0, 1]));

    // The payload types with a single-precision component.
    let (rank, status) = prefix_rank_case::<f32>(
        &session,
        DType::F32,
        40,
        40,
        1,
        &[0.5f32 as f64; 40],
        1e-3,
        0.0,
    );
    assert_eq!((rank, status), (vec![40i64], vec![0i64]));
    let (rank, status) = prefix_rank_case::<C32>(
        &session,
        DType::C32,
        6,
        6,
        1,
        &[1.0, 0.5, 1e-9, 1.0, 1.0, 1.0],
        1e-6,
        0.0,
    );
    assert_eq!((rank, status), (vec![2i64], vec![0i64]));
    let (rank, status) = prefix_rank_case::<C64>(
        &session,
        DType::C64,
        6,
        6,
        1,
        &[1.0, 0.5, 0.25, 0.125, 0.0625, 0.0],
        1e-12,
        0.0,
    );
    assert_eq!((rank, status), (vec![5i64], vec![0i64]));
}

#[test]
fn prefix_rank_uses_the_magnitude_and_a_nan_tolerance() {
    let session = Session::new("rrqr");

    // Five diagonals: fewer than one plane, so lanes 5..31 own nothing.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 5, 8, 1, &[0.5; 5], 1e-6, 0.0);
    assert_eq!((rank, status), (vec![5i64], vec![0i64]));

    // Purely imaginary diagonals: `abs(re)` would read them as 0 and stop at rank 1.
    let pairs = [
        (1.0, 0.0),
        (0.0, 2.0),
        (0.0, -0.5),
        (0.0, 0.25),
        (0.5, 0.0),
        (0.0, 0.125),
    ];
    let (rank, status) =
        prefix_rank_pairs_case::<C64>(&session, DType::C64, 6, 6, 1, &pairs, 1e-12, 0.0);
    assert_eq!((rank, status), (vec![6i64], vec![0i64]));
    let (rank, status) =
        prefix_rank_pairs_case::<C32>(&session, DType::C32, 6, 6, 1, &pairs, 1e-4, 0.0);
    assert_eq!((rank, status), (vec![6i64], vec![0i64]));

    // A NaN rtol enters the baseline's threshold, so no diagonal exceeds it: rank 0
    // with the status word clear, because the diagonals themselves are finite.
    let (rank, status) =
        prefix_rank_case::<f64>(&session, DType::F64, 6, 6, 1, &[1.0; 6], f64::NAN, 0.0);
    assert_eq!((rank, status), (vec![0i64], vec![0i64]));
}

// --- empty plans ------------------------------------------------------------

#[test]
fn validated_empty_plans_launch_nothing() {
    let session = Session::new("rrqr");

    // column_norms with step == columns: no trailing column is measured.
    let work = Layout::compact(&[4, 5]).unwrap();
    let norms = Layout::compact(&[5]).unwrap();
    let work_buf = upload(&session, &[0.0f64; 20]);
    let (norms_buf, norms_len) = destination(&session, &norms, DType::F64);
    let plan =
        plan_column_norms(DType::F64, &work, span(&work) as i64, &norms, norms_len, 5).unwrap();
    assert!(plan.is_empty());
    let mut slots = [
        Slot::ptr(&work_buf),
        Slot::ptr(&norms_buf),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&work)),
        Slot::layout(&CLayout::new(&norms)),
        Slot::i64(5),
    ];
    run(&session, "t4a_cuda_column_norms", &plan, &mut slots);
    assert_guard(&read_bytes(&session, &norms_buf), 0, "column_norms empty");

    // swap_columns with an empty row extent.
    let flat = Layout::new(&[0, 5], &[1, 1], 0).unwrap();
    let permutation = Layout::compact(&[5]).unwrap();
    let (flat_buf, flat_len) = destination(&session, &flat, DType::F64);
    let permutation_buf = operand(&session, &permutation, &[0i64, 1, 2, 3, 4]);
    // One int64 pivot per batch of the empty work matrix.
    let pivots = zeroed(&session, 5 * 8);
    let plan = plan_swap_columns(
        DType::F64,
        &flat,
        flat_len,
        &permutation,
        span(&permutation) as i64,
        5,
        0,
    )
    .unwrap();
    assert!(plan.is_empty());
    let mut slots = [
        Slot::ptr(&flat_buf),
        Slot::ptr(&permutation_buf),
        Slot::ptr(&pivots),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&flat)),
        Slot::layout(&CLayout::new(&permutation)),
        Slot::i64(0),
    ];
    run(&session, "t4a_cuda_swap_columns", &plan, &mut slots);
    assert_guard(&read_bytes(&session, &flat_buf), 0, "swap_columns empty");

    // extract_r with an empty r row extent.
    let work = Layout::compact(&[6, 4]).unwrap();
    let r = Layout::new(&[0, 4], &[1, 1], 0).unwrap();
    let work_buf = upload(&session, &[1.0f64; 24]);
    let (r_buf, r_len) = destination(&session, &r, DType::F64);
    let plan = plan_extract_r(DType::F64, &work, span(&work) as i64, &r, r_len).unwrap();
    assert!(plan.is_empty());
    let mut slots = [
        Slot::ptr(&work_buf),
        Slot::ptr(&r_buf),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&work)),
        Slot::layout(&CLayout::new(&r)),
    ];
    run(&session, "t4a_cuda_extract_r", &plan, &mut slots);
    assert_guard(&read_bytes(&session, &r_buf), 0, "extract_r empty");

    // prefix_rank with zero batches.
    let r = Layout::new(&[3, 4, 0], &[1, 3, 12], 0).unwrap();
    let r_buf = zeroed(&session, 8);
    let rank = sentinel(&session, 8 + GUARD);
    let status = zeroed(&session, 8 + GUARD);
    let plan = plan_prefix_rank(DType::F64, &r, 0, 0, 0, 0.0, 0.0).unwrap();
    assert!(plan.is_empty());
    let mut slots = [
        Slot::ptr(&r_buf),
        Slot::ptr(&rank),
        Slot::ptr(&status),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&r)),
        Slot::f64(0.0),
        Slot::f64(0.0),
    ];
    run(&session, "t4a_cuda_prefix_rank", &plan, &mut slots);
    assert_guard(&read_bytes(&session, &rank), 0, "prefix_rank empty");

    // apply_reflector with no column to apply to.
    let packed = Layout::compact(&[4, 3]).unwrap();
    let target = Layout::compact(&[4, 3]).unwrap();
    let coeff = Layout::compact(&[3]).unwrap();
    let packed_buf = upload(&session, &[0.0f64; 12]);
    let (target_buf, target_len) = destination(&session, &target, DType::F64);
    let coeff_buf = upload(&session, &[2.0f64; 3]);
    let plan = plan_apply_reflector(
        DType::F64,
        &packed,
        span(&packed) as i64,
        &target,
        target_len,
        &coeff,
        span(&coeff) as i64,
        0,
        0,
        0,
    )
    .unwrap();
    assert!(plan.is_empty());
    let mut slots = [
        Slot::ptr(&packed_buf),
        Slot::ptr(&target_buf),
        Slot::ptr(&coeff_buf),
        Slot::i32(DType::F64.tag()),
        Slot::layout(&CLayout::new(&packed)),
        Slot::layout(&CLayout::new(&target)),
        Slot::layout(&CLayout::new(&coeff)),
        Slot::i64(0),
        Slot::i64(0),
        Slot::i64(0),
    ];
    run(&session, "t4a_cuda_apply_reflector", &plan, &mut slots);
    assert_guard(
        &read_bytes(&session, &target_buf),
        0,
        "apply_reflector empty",
    );

    // initialize_permutation with an empty domain.
    let out = sentinel(&session, GUARD);
    let plan = plan_initialize_permutation(3, 0).unwrap();
    assert!(plan.is_empty());
    let mut slots = [Slot::ptr(&out), Slot::i64(3), Slot::i64(0)];
    run(
        &session,
        "t4a_cuda_initialize_permutation",
        &plan,
        &mut slots,
    );
    assert_guard(
        &read_bytes(&session, &out),
        0,
        "initialize_permutation empty",
    );
}
