//! GPU tests for the fused-expression emitter (`src/fusion.rs`): the NVRTC reference
//! consumer of `docs/design/slice-fusion-emitter.md` §5 and §7.
//!
//! The crate never compiles anything, so this binary is the caller the design
//! describes: it emits a plan, compiles the emitted source with NVRTC, loads the PTX
//! with the raw driver API on the harness's context, launches the entry through the
//! harness's plan and slot machinery, and compares every live-out against a CPU
//! reference. Its own `compile_ptx_with_opts` and `cuModuleLoadData` call sites are
//! wrapped in counters, so the prepare/cache contract is observed rather than argued
//! from API shape.
//!
//! Run with `cargo test -j 16 --features gpu-tests --test gpu_fusion`; requires a CUDA
//! device and NVRTC.

mod gpu;

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::CString;

use cudarc::driver::result as drv;
use cudarc::driver::sys;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use gpu::{
    from_bytes, launch, launches, read_bytes, read_prefix, upload, written, Buf, Session, Slot,
};
use t4a_cuda_kernels::{
    artifact_key, emit_fusion, plan_binary, plan_combined_strided, plan_fused, ArtifactKey,
    BinaryOp, DType, EmittedFusion, FusionInstruction, FusionOp, FusionOperand, FusionPlan, Layout,
    ScalarSide, StridedOp,
};

/// The harness's device arch (`T4A_CUDA_ARCH`'s default), so the NVRTC target matches
/// the driver's device. It is also part of the option string the artifact key covers.
const ARCH: &str = "sm_80";
/// A byte pattern no kernel writes, filling every destination and its guard region.
const SENTINEL: u8 = 0xCD;
/// Bytes reserved behind a logical output to catch a write past `len`.
const GUARD: usize = 13;
/// Extents the design requires: a single element, a ragged tail above one 256-thread
/// block, and a large flat domain.
const EXTENTS: [i64; 3] = [1, 513, 65536];

thread_local! {
    static NVRTC_CALLS: Cell<usize> = const { Cell::new(0) };
    static MODULE_LOADS: Cell<usize> = const { Cell::new(0) };
}

/// NVRTC compilations this test thread performed.
fn nvrtc_calls() -> usize {
    NVRTC_CALLS.with(Cell::get)
}

/// `cuModuleLoadData` calls this test thread performed.
fn module_loads() -> usize {
    MODULE_LOADS.with(Cell::get)
}

fn count(counter: &'static std::thread::LocalKey<Cell<usize>>) {
    counter.with(|count| count.set(count.get() + 1));
}

/// The compiler identity the artifact key must include.
fn nvrtc_identity() -> String {
    let (mut major, mut minor) = (0i32, 0i32);
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .expect("nvrtcVersion");
    format!("nvrtc {major}.{minor}")
}

/// NVRTC options this consumer compiles with.
fn options() -> CompileOptions {
    CompileOptions {
        arch: Some(ARCH),
        ..Default::default()
    }
}

/// The canonical option string the artifact key covers. Every field of
/// [`CompileOptions`] that changes the generated code appears here, so two different
/// option strings cannot share an artifact.
fn option_key(opts: &CompileOptions) -> String {
    format!(
        "ftz={:?};prec-sqrt={:?};prec-div={:?};fmad={:?};fast-math={:?};maxrregcount={:?};\
         arch={:?};extra={:?}",
        opts.ftz,
        opts.prec_sqrt,
        opts.prec_div,
        opts.fmad,
        opts.use_fast_math,
        opts.maxrregcount,
        opts.arch,
        opts.options,
    )
}

/// A minimal reference consumer: one context/stream from the harness and one private
/// module cache keyed by the artifact key.
///
/// `prepare` is the whole prepare step of the design: compute the key, look it up, and
/// on a miss compile once with NVRTC and load once. Launching it again performs no
/// source generation, no NVRTC call and no key recomputation.
struct Consumer {
    session: Session,
    modules: Vec<sys::CUmodule>,
    cache: HashMap<ArtifactKey, sys::CUfunction>,
    target: String,
    compiler: String,
}

impl Consumer {
    /// A consumer whose harness session loads no nvcc module: this test compiles its
    /// kernel with NVRTC instead.
    fn new() -> Consumer {
        Consumer::with_session(Session::with_modules(&[]))
    }

    fn with_session(session: Session) -> Consumer {
        Consumer {
            session,
            modules: Vec::new(),
            cache: HashMap::new(),
            target: format!("nvrtc|ptx|{ARCH}"),
            compiler: nvrtc_identity(),
        }
    }

    fn session(&self) -> &Session {
        &self.session
    }

    /// The prepare step. Returns the artifact key it keyed on and the entry binding.
    fn prepare(
        &mut self,
        emitted: &EmittedFusion,
        opts: &CompileOptions,
    ) -> (ArtifactKey, sys::CUfunction) {
        let key = artifact_key(emitted, &option_key(opts), &self.target, &self.compiler);
        if let Some(function) = self.cache.get(&key) {
            return (key.clone(), *function);
        }
        count(&NVRTC_CALLS);
        let ptx = compile_ptx_with_opts(emitted.source(), opts.clone())
            .unwrap_or_else(|error| panic!("NVRTC rejected the emitted source: {error:?}"));
        let image = ptx.as_bytes().expect("NVRTC returned a PTX image");
        count(&MODULE_LOADS);
        let module = unsafe { drv::module::load_data(image.as_ptr().cast()) }
            .expect("cuModuleLoadData on the NVRTC image");
        let symbol = CString::new(emitted.manifest().entry).expect("entry name");
        let function =
            unsafe { drv::module::get_function(module, symbol) }.expect("cuModuleGetFunction");
        self.modules.push(module);
        self.cache.insert(key.clone(), function);
        (key, function)
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        unsafe { drv::stream::synchronize(self.session.stream()) }.unwrap();
        while let Some(module) = self.modules.pop() {
            unsafe { drv::module::unload(module) }.unwrap();
        }
    }
}

// --- launch -----------------------------------------------------------------

/// Launch one prepared kernel over `len` elements and return every live-out.
///
/// Every destination is a sentinel-filled buffer with `GUARD` bytes behind the logical
/// payload; the emitted kernel writes exactly `len` elements, so any byte past
/// `len * element_size` must still hold the sentinel (design §6, buffer contract).
fn run<T: Copy>(
    consumer: &Consumer,
    function: sys::CUfunction,
    dtype: DType,
    inputs: &[Vec<T>],
    outputs: usize,
    len: i64,
) -> Vec<Vec<T>> {
    let session = consumer.session();
    let payload = len as usize * dtype.size();
    let input_bufs: Vec<Buf> = inputs
        .iter()
        .map(|values| upload(session, values))
        .collect();
    let output_bufs: Vec<Buf> = (0..outputs)
        .map(|_| written(session, &vec![SENTINEL; payload + GUARD]))
        .collect();
    let plan = plan_fused(len).unwrap();
    // Argument order of the manifest: every output, then every input, then `len`.
    let mut slots: Vec<Slot> = output_bufs
        .iter()
        .map(Slot::ptr)
        .chain(input_bufs.iter().map(Slot::ptr))
        .collect();
    slots.push(Slot::i64(len));
    let before = launches();
    launch(function, &plan, session.stream(), &mut slots);
    assert_eq!(
        launches(),
        before + usize::from(!plan.is_empty()),
        "one physical launch per run (len {len})"
    );
    output_bufs
        .iter()
        .map(|buf| {
            let bytes = read_bytes(session, buf);
            assert!(
                bytes[payload..].iter().all(|&byte| byte == SENTINEL),
                "the emitted kernel wrote past len = {len}"
            );
            from_bytes(&bytes[..payload])
        })
        .collect()
}

/// [`run`] with every element widened to `f64`, so a launch of either real dtype is
/// compared the same way. The widening of an `f32` result is exact.
fn run_rows<T: Copy + Into<f64>>(
    consumer: &Consumer,
    function: sys::CUfunction,
    dtype: DType,
    inputs: &[Vec<T>],
    outputs: usize,
    len: i64,
) -> Vec<Vec<f64>> {
    run(consumer, function, dtype, inputs, outputs, len)
        .into_iter()
        .map(|column| column.into_iter().map(Into::into).collect())
        .collect()
}

// --- fixtures ---------------------------------------------------------------

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

/// A deterministic, well-spread sample of `[lo, hi)`.
fn spread(count: usize, lo: f64, hi: f64) -> Vec<f64> {
    const PHI: f64 = 0.618_033_988_749_894_9;
    (0..count)
        .map(|index| lo + (hi - lo) * (index as f64 * PHI).fract())
        .collect()
}

fn narrow(columns: &[Vec<f64>]) -> Vec<Vec<f32>> {
    columns
        .iter()
        .map(|column| column.iter().map(|&value| value as f32).collect())
        .collect()
}

fn widen(columns: &[Vec<f32>]) -> Vec<Vec<f64>> {
    columns
        .iter()
        .map(|column| column.iter().map(|&value| f64::from(value)).collect())
        .collect()
}

/// `a, b, c, d` of the two-live-out plan: `d` stays away from zero and `a*b + c` in a
/// range where both `exp` and `tanh(exp(..))` are far from either end, so an `f32`
/// launch and an `f64` reference stay comparable relatively.
fn real_main_inputs(len: i64) -> Vec<Vec<f64>> {
    let count = len as usize;
    vec![
        spread(count, 0.5, 1.5),
        spread(count, -1.5, 1.5),
        spread(count, -0.5, 0.5),
        spread(count, 0.75, 2.0),
    ]
}

// --- plans ------------------------------------------------------------------

fn input(index: u32) -> FusionOperand {
    FusionOperand::Input(index)
}

fn value(index: u32) -> FusionOperand {
    FusionOperand::Value(index)
}

fn instruction(op: FusionOp, operands: &[FusionOperand]) -> FusionInstruction {
    FusionInstruction {
        op,
        operands: operands.to_vec(),
    }
}

/// `y0 = (a*b + c) / d`, `y1 = tanh(exp(a*b + c))`, sharing `a*b`.
fn real_main_plan(dtype: DType) -> FusionPlan {
    FusionPlan {
        dtype,
        inputs: 4,
        instructions: vec![
            instruction(FusionOp::Mul, &[input(0), input(1)]), // v4 = a*b
            instruction(FusionOp::Add, &[value(0), input(2)]), // v5 = a*b + c
            instruction(FusionOp::Exp, &[value(1)]),           // v6 = exp(v5)
            instruction(FusionOp::Tanh, &[value(2)]),          // v7 = tanh(v6)
            instruction(FusionOp::Div, &[value(1), input(3)]), // v8 = v5 / d
        ],
        outputs: vec![value(4), value(3)],
    }
}

/// Every real op the two-live-out plan does not contain, all live out in one launch:
/// `y = [neg, abs, max, min, clamp, log, sin, cos, sqrt, rsqrt, pow, expm1, log1p, erf]`
/// over `x`, `y`, `lo`, `hi`.
fn real_wide_plan(dtype: DType) -> FusionPlan {
    FusionPlan {
        dtype,
        inputs: 4,
        instructions: vec![
            instruction(FusionOp::Neg, &[input(0)]),
            instruction(FusionOp::Abs, &[input(0)]),
            instruction(FusionOp::Max, &[input(0), input(1)]),
            instruction(FusionOp::Min, &[input(0), input(1)]),
            instruction(FusionOp::Clamp, &[input(0), input(2), input(3)]),
            instruction(FusionOp::Log, &[value(1)]),
            instruction(FusionOp::Sin, &[input(0)]),
            instruction(FusionOp::Cos, &[input(0)]),
            instruction(FusionOp::Sqrt, &[value(1)]),
            instruction(FusionOp::Rsqrt, &[value(1)]),
            instruction(FusionOp::Pow, &[value(1), input(1)]),
            instruction(FusionOp::Expm1, &[input(0)]),
            instruction(FusionOp::Log1p, &[input(0)]),
            instruction(FusionOp::Erf, &[input(0)]),
        ],
        outputs: (0..14).map(value).collect(),
    }
}

/// The fused extrema chain: `y0 = min(max(x, y), lo)`, `y1 = clamp(x, lo, hi)`.
fn extrema_plan(dtype: DType) -> FusionPlan {
    FusionPlan {
        dtype,
        inputs: 4,
        instructions: vec![
            instruction(FusionOp::Max, &[input(0), input(1)]),
            instruction(FusionOp::Min, &[value(0), input(2)]),
            instruction(FusionOp::Clamp, &[input(0), input(2), input(3)]),
        ],
        outputs: vec![value(1), value(2)],
    }
}

/// Complex chain over `u`, `v`: `y0 = (u*v + u) / (-v)`, `y1 = conj(u) * (u*v + u)`.
fn complex_chain_plan(dtype: DType) -> FusionPlan {
    FusionPlan {
        dtype,
        inputs: 2,
        instructions: vec![
            instruction(FusionOp::Mul, &[input(0), input(1)]),
            instruction(FusionOp::Add, &[value(0), input(0)]),
            instruction(FusionOp::Neg, &[input(1)]),
            instruction(FusionOp::Conj, &[input(0)]),
            instruction(FusionOp::Div, &[value(1), value(2)]),
            instruction(FusionOp::Mul, &[value(3), value(1)]),
        ],
        outputs: vec![value(4), value(5)],
    }
}

/// Complex division of extreme operands, two live-outs: `y0 = u / v`,
/// `y1 = (-u) / conj(v)`.
fn complex_extreme_plan(dtype: DType) -> FusionPlan {
    FusionPlan {
        dtype,
        inputs: 2,
        instructions: vec![
            instruction(FusionOp::Div, &[input(0), input(1)]),
            instruction(FusionOp::Neg, &[input(0)]),
            instruction(FusionOp::Conj, &[input(1)]),
            instruction(FusionOp::Div, &[value(1), value(2)]),
        ],
        outputs: vec![value(0), value(3)],
    }
}

// --- CPU references ---------------------------------------------------------

/// Relative closeness, with NaN matching NaN and infinities matching exactly.
fn relative(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    if expected == 0.0 {
        return actual == 0.0;
    }
    (actual - expected).abs() <= tolerance * expected.abs()
}

/// `erf` is not in Rust's standard library. The Maclaurin series is accurate to `f64`
/// rounding on `|x| <= 2.2` (the domain the fixtures use).
fn erf(x: f64) -> f64 {
    let square = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..400 {
        term *= -square / n as f64;
        let contribution = term / (2 * n + 1) as f64;
        sum += contribution;
        if contribution.abs() <= 1e-18 * sum.abs() {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// NaN-propagating maximum, `lhs`-then-`rhs`, as `t4a_cuda_nan_max` defines it.
fn nan_max(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else if lhs > rhs {
        lhs
    } else {
        rhs
    }
}

/// NaN-propagating minimum, `lhs`-then-`rhs`, as `t4a_cuda_nan_min` defines it.
fn nan_min(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else if lhs < rhs {
        lhs
    } else {
        rhs
    }
}

/// Evaluate a real plan at one element in `f64`, with the ABI's ordered rules.
fn real_reference(plan: &FusionPlan, inputs: &[f64]) -> Vec<f64> {
    let mut values: Vec<f64> = Vec::with_capacity(plan.instructions.len());
    for fused in &plan.instructions {
        let arg = |position: usize| match fused.operands[position] {
            FusionOperand::Input(index) => inputs[index as usize],
            FusionOperand::Value(index) => values[index as usize],
        };
        let lhs = arg(0);
        let result = match fused.op {
            FusionOp::Add => lhs + arg(1),
            FusionOp::Mul => lhs * arg(1),
            FusionOp::Div => lhs / arg(1),
            FusionOp::Neg => -lhs,
            FusionOp::Abs => lhs.abs(),
            FusionOp::Max => nan_max(lhs, arg(1)),
            FusionOp::Min => nan_min(lhs, arg(1)),
            FusionOp::Clamp => {
                let low = arg(1);
                let high = arg(2);
                if lhs < low {
                    low
                } else if lhs > high {
                    high
                } else {
                    lhs
                }
            }
            FusionOp::Exp => lhs.exp(),
            FusionOp::Log => lhs.ln(),
            FusionOp::Sin => lhs.sin(),
            FusionOp::Cos => lhs.cos(),
            FusionOp::Tanh => lhs.tanh(),
            FusionOp::Sqrt => lhs.sqrt(),
            FusionOp::Rsqrt => 1.0 / lhs.sqrt(),
            FusionOp::Pow => lhs.powf(arg(1)),
            FusionOp::Expm1 => lhs.exp_m1(),
            FusionOp::Log1p => lhs.ln_1p(),
            FusionOp::Erf => erf(lhs),
            FusionOp::Conj => panic!("Conj is not a real operation"),
        };
        values.push(result);
    }
    plan.outputs
        .iter()
        .map(|operand| match *operand {
            FusionOperand::Input(index) => inputs[index as usize],
            FusionOperand::Value(index) => values[index as usize],
        })
        .collect()
}

macro_rules! complex_division {
    ($name:ident, $f:ty, $max:expr, $min_positive:expr, $epsilon:expr, $nan:expr) => {
        /// Literal port of `t4a_cuda_div_complex_parts_at` (Baudin-Smith scaling with
        /// the subnormal/near-overflow rescale) in the element precision, as the
        /// revision-6 elementwise test states it. Derived from tenferro-rs
        /// `elementwise.rs`, which derives from Julia Base `complex.jl` (`cdiv`,
        /// `robust_cdiv1/2`, `scaling_cdiv`, `scaleargs_cdiv`); Copyright (c) 2009-2026
        /// Jeff Bezanson, Stefan Karpinski, Viral B. Shah and other Julia contributors,
        /// MIT. See `THIRD_PARTY_NOTICES.md`.
        fn $name(a0: $f, b0: $f, c0: $f, d0: $f) -> ($f, $f) {
            fn magnitude(value: $f) -> $f {
                if value < 0.0 {
                    -value
                } else {
                    value
                }
            }
            fn signed_unit(value: $f) -> $f {
                if value == 0.0 {
                    value
                } else if value < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
            fn term(a: $f, b: $f, c: $f, d: $f, r: $f, t: $f) -> $f {
                if r != 0.0 {
                    let br = b * r;
                    if br != 0.0 {
                        (a + br) * t
                    } else {
                        a * t + (b * t) * r
                    }
                } else {
                    (a + d * (b / c)) * t
                }
            }
            let max_value: $f = $max;
            let min_positive: $f = $min_positive;
            let epsilon: $f = $epsilon;
            let mut a = a0;
            let mut b = b0;
            let mut c = c0;
            let mut d = d0;
            if magnitude(c) > max_value || magnitude(d) > max_value {
                let a_finite = a == a && magnitude(a) <= max_value;
                let b_finite = b == b && magnitude(b) <= max_value;
                if a_finite && b_finite {
                    return (
                        0.0 * signed_unit(a) * signed_unit(c),
                        -0.0 * signed_unit(b) * signed_unit(d),
                    );
                }
                return ($nan, $nan);
            }
            let abs_a = magnitude(a);
            let abs_b = magnitude(b);
            let abs_c = magnitude(c);
            let abs_d = magnitude(d);
            let ab = if abs_a >= abs_b { abs_a } else { abs_b };
            let cd = if abs_c >= abs_d { abs_c } else { abs_d };
            let half: $f = 0.5;
            let two: $f = 2.0;
            let half_over = max_value * half;
            let two_under_eps = min_positive * two / epsilon;
            let mut scale: $f = 1.0;
            if ab >= half_over || ab <= two_under_eps || cd >= half_over || cd <= two_under_eps {
                let big: $f = two / (epsilon * epsilon);
                if ab >= half_over {
                    a = a * half;
                    b = b * half;
                    scale = scale * two;
                } else if ab <= two_under_eps {
                    a = a * big;
                    b = b * big;
                    scale = scale / big;
                }
                if cd >= half_over {
                    c = c * half;
                    d = d * half;
                    scale = scale * half;
                } else if cd <= two_under_eps {
                    c = c * big;
                    d = d * big;
                    scale = scale * big;
                }
            }
            if magnitude(d) <= magnitude(c) {
                let r = d / c;
                let t = 1.0 / (c + d * r);
                (
                    term(a, b, c, d, r, t) * scale,
                    term(b, -a, c, d, r, t) * scale,
                )
            } else {
                let r = c / d;
                let t = 1.0 / (c * r + d);
                (
                    term(b, a, d, c, r, t) * scale,
                    -term(a, -b, d, c, r, t) * scale,
                )
            }
        }
    };
}

complex_division!(
    cdiv32,
    f32,
    f32::MAX,
    f32::MIN_POSITIVE,
    f32::EPSILON,
    f32::from_bits(0x7fc0_0000)
);
complex_division!(
    cdiv64,
    f64,
    f64::MAX,
    f64::MIN_POSITIVE,
    f64::EPSILON,
    f64::from_bits(0x7ff8_0000_0000_0000)
);

macro_rules! complex_reference {
    ($name:ident, $f:ty, $cdiv:ident) => {
        /// Evaluate a complex plan (add, mul, div, neg, conj only) at one element.
        fn $name(plan: &FusionPlan, inputs: &[($f, $f)]) -> Vec<($f, $f)> {
            let mut values: Vec<($f, $f)> = Vec::with_capacity(plan.instructions.len());
            for fused in &plan.instructions {
                let arg = |position: usize| -> ($f, $f) {
                    match fused.operands[position] {
                        FusionOperand::Input(index) => inputs[index as usize],
                        FusionOperand::Value(index) => values[index as usize],
                    }
                };
                let (re, im) = arg(0);
                let result = match fused.op {
                    FusionOp::Add => {
                        let (rhs_re, rhs_im) = arg(1);
                        (re + rhs_re, im + rhs_im)
                    }
                    FusionOp::Mul => {
                        let (rhs_re, rhs_im) = arg(1);
                        (re * rhs_re - im * rhs_im, re * rhs_im + im * rhs_re)
                    }
                    FusionOp::Div => {
                        let (rhs_re, rhs_im) = arg(1);
                        $cdiv(re, im, rhs_re, rhs_im)
                    }
                    FusionOp::Neg => (-re, -im),
                    FusionOp::Conj => (re, -im),
                    other => panic!("{other:?} is not an eligible complex operation"),
                };
                values.push(result);
            }
            plan.outputs
                .iter()
                .map(|operand| match *operand {
                    FusionOperand::Input(index) => inputs[index as usize],
                    FusionOperand::Value(index) => values[index as usize],
                })
                .collect()
        }
    };
}

complex_reference!(complex_reference32, f32, cdiv32);
complex_reference!(complex_reference64, f64, cdiv64);

// --- comparisons ------------------------------------------------------------

/// Compare a launched real plan against the CPU reference over every live-out.
fn check_rows(
    what: &str,
    plan: &FusionPlan,
    inputs: &[Vec<f64>],
    rows: &[Vec<f64>],
    tolerance: f64,
) {
    assert_eq!(rows.len(), plan.outputs.len(), "{what}: live-out count");
    let len = inputs[0].len();
    for index in 0..len {
        let arguments: Vec<f64> = inputs.iter().map(|column| column[index]).collect();
        let expected = real_reference(plan, &arguments);
        for (output, &want) in expected.iter().enumerate() {
            let got = rows[output][index];
            assert!(
                relative(got, want, tolerance),
                "{what}: out{output}[{index}] = {got:?}, expected {want:?} for {arguments:?}"
            );
        }
    }
}

fn check_pair(what: &str, got: (f64, f64), expected: (f64, f64), tolerance: f64) {
    // Complex arithmetic cancels a component down to rounding noise, so the bound is
    // relative to the element's magnitude rather than to each component. A zero result
    // keeps the strict per-component bound, which is why the signed-zero fixtures below
    // are asserted separately.
    let scale = (expected.0.abs() + expected.1.abs()).max(f64::MIN_POSITIVE);
    let close = |actual: f64, want: f64| {
        if actual.is_nan() || want.is_nan() {
            return actual.is_nan() && want.is_nan();
        }
        if actual.is_infinite() || want.is_infinite() {
            return actual == want;
        }
        (actual - want).abs() <= tolerance * scale
    };
    assert!(
        close(got.0, expected.0) && close(got.1, expected.1),
        "{what}: got {got:?}, expected {expected:?} within {tolerance} of |{scale}|"
    );
}

// --- numerics ---------------------------------------------------------------

#[test]
fn real_two_live_outs_match_the_cpu_reference() {
    let mut consumer = Consumer::new();
    for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::F64, 1e-13)] {
        let plan = real_main_plan(dtype);
        let emitted = emit_fusion(&plan).unwrap();
        let (_, function) = consumer.prepare(&emitted, &options());
        for len in EXTENTS {
            let wide = real_main_inputs(len);
            let (rows, exact) = match dtype {
                DType::F64 => (
                    run_rows(&consumer, function, dtype, &wide, 2, len),
                    wide.clone(),
                ),
                _ => {
                    let narrow_inputs = narrow(&wide);
                    (
                        run_rows(&consumer, function, dtype, &narrow_inputs, 2, len),
                        widen(&narrow_inputs),
                    )
                }
            };
            let what = format!("{dtype:?} len {len}");
            check_rows(&what, &plan, &exact, &rows, tolerance);
            // The two live-outs are genuinely different values, so a body that wrote
            // one expression to both outputs cannot pass.
            assert!(
                rows[0].iter().zip(rows[1].iter()).any(|(a, b)| a != b),
                "{what}: both live-outs hold the same value"
            );
        }
    }
}

#[test]
fn every_eligible_real_op_runs_at_least_once() {
    let eligible = [
        FusionOp::Add,
        FusionOp::Mul,
        FusionOp::Div,
        FusionOp::Neg,
        FusionOp::Abs,
        FusionOp::Max,
        FusionOp::Min,
        FusionOp::Clamp,
        FusionOp::Exp,
        FusionOp::Log,
        FusionOp::Sin,
        FusionOp::Cos,
        FusionOp::Tanh,
        FusionOp::Sqrt,
        FusionOp::Rsqrt,
        FusionOp::Pow,
        FusionOp::Expm1,
        FusionOp::Log1p,
        FusionOp::Erf,
    ];
    let plans = [real_main_plan(DType::F64), real_wide_plan(DType::F64)];
    for op in eligible {
        assert!(
            plans
                .iter()
                .flat_map(|plan| plan.instructions.iter())
                .any(|fused| fused.op == op),
            "{op:?} is never launched"
        );
    }

    let mut consumer = Consumer::new();
    for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::F64, 1e-13)] {
        let plan = real_wide_plan(dtype);
        let emitted = emit_fusion(&plan).unwrap();
        let (_, function) = consumer.prepare(&emitted, &options());

        // A domain that keeps `log`, `sqrt`, `rsqrt` and `pow` on their positive
        // branch, plus an explicit fixture with negative `x` so `neg` and `abs` are
        // not no-ops.
        let len = 1024i64;
        let count = len as usize;
        let fixtures = vec![
            {
                let mut x = spread(count, 0.5, 2.2);
                x[0] = -1.5;
                x[1] = -0.5;
                x[2] = 0.25;
                x[3] = 2.0;
                x
            },
            spread(count, 0.5, 1.5),
            spread(count, -1.0, 0.0),
            spread(count, 0.5, 2.5),
        ];
        let (rows, exact) = match dtype {
            DType::F64 => (
                run_rows(&consumer, function, dtype, &fixtures, 14, len),
                fixtures.clone(),
            ),
            _ => {
                let narrow_inputs = narrow(&fixtures);
                (
                    run_rows(&consumer, function, dtype, &narrow_inputs, 14, len),
                    widen(&narrow_inputs),
                )
            }
        };
        check_rows(&format!("wide {dtype:?}"), &plan, &exact, &rows, tolerance);

        // The explicit fixture: |x| is what `abs`, `log`, `sqrt`, `rsqrt` and `pow`
        // see, so the negative inputs above exercise those branches, not a no-op.
        assert!(relative(rows[0][0], 1.5, tolerance), "neg(-1.5)");
        assert!(relative(rows[1][0], 1.5, tolerance), "abs(-1.5)");
        assert!(relative(rows[5][0], 1.5f64.ln(), tolerance), "log(|-1.5|)");
        assert!(
            relative(rows[8][0], 1.5f64.sqrt(), tolerance),
            "sqrt(|-1.5|)"
        );
        let expected_pow = 1.5f64.powf(exact[1][0]);
        assert!(
            relative(rows[10][0], expected_pow, tolerance),
            "pow(|x|, y)"
        );
    }
}

/// Complex fixtures of the revision-6 division test, in the element precision.
fn extreme64() -> (Vec<C64>, Vec<C64>) {
    let lhs = vec![
        C64 { re: 2.0, im: 0.0 },
        C64 {
            re: 1e-320,
            im: 1e-320,
        },
        C64 { re: 3.0, im: 4.0 },
        C64 { re: 3.0, im: -4.0 },
        C64 {
            re: f64::NAN,
            im: 1.0,
        },
        C64 { re: 1.0, im: 1.0 },
        C64 {
            re: 1e-320,
            im: 0.0,
        },
        C64 {
            re: 1e308,
            im: 1e308,
        },
    ];
    let rhs = vec![
        C64 { re: 1e38, im: 1e38 },
        C64 {
            re: 1e-320,
            im: 1e-320,
        },
        C64 {
            re: f64::INFINITY,
            im: 0.0,
        },
        C64 {
            re: f64::NEG_INFINITY,
            im: f64::INFINITY,
        },
        C64 { re: 2.0, im: 3.0 },
        C64 {
            re: f64::NAN,
            im: 0.0,
        },
        C64 { re: 1.0, im: 0.0 },
        C64 {
            re: 1e300,
            im: -1e300,
        },
    ];
    (lhs, rhs)
}

/// The same shapes at single precision.
fn extreme32() -> (Vec<C32>, Vec<C32>) {
    let lhs = vec![
        C32 { re: 2.0, im: 0.0 },
        C32 {
            re: 1e-42,
            im: 1e-42,
        },
        C32 { re: 3.0, im: 4.0 },
        C32 { re: 3.0, im: -4.0 },
        C32 {
            re: f32::NAN,
            im: 1.0,
        },
        C32 { re: 1.0, im: 1.0 },
        C32 { re: 1e-42, im: 0.0 },
        C32 { re: 1e30, im: 1e30 },
    ];
    let rhs = vec![
        C32 { re: 1e30, im: 1e30 },
        C32 {
            re: 1e-42,
            im: 1e-42,
        },
        C32 {
            re: f32::INFINITY,
            im: 0.0,
        },
        C32 {
            re: f32::NEG_INFINITY,
            im: f32::INFINITY,
        },
        C32 { re: 2.0, im: 3.0 },
        C32 {
            re: f32::NAN,
            im: 0.0,
        },
        C32 { re: 1.0, im: 0.0 },
        C32 {
            re: 1e20,
            im: -1e20,
        },
    ];
    (lhs, rhs)
}

#[test]
fn complex_two_live_outs_match_the_cpu_reference() {
    let mut consumer = Consumer::new();

    // A chain over eligible operations only: add, mul, div, neg, conj.
    for (dtype, tolerance) in [(DType::C32, 1e-5), (DType::C64, 1e-13)] {
        let plan = complex_chain_plan(dtype);
        let emitted = emit_fusion(&plan).unwrap();
        let (_, function) = consumer.prepare(&emitted, &options());

        for extent in EXTENTS {
            let count = extent as usize;
            let (u_re, u_im) = (spread(count, 0.5, 1.5), spread(count, -1.5, 1.5));
            let (v_re, v_im) = (spread(count, 1.0, 2.0), spread(count, 0.25, 1.75));
            let what = format!("complex chain {dtype:?} len {extent}");

            match dtype {
                DType::C64 => {
                    let u: Vec<C64> = (0..count)
                        .map(|index| C64 {
                            re: u_re[index],
                            im: u_im[index],
                        })
                        .collect();
                    let v: Vec<C64> = (0..count)
                        .map(|index| C64 {
                            re: v_re[index],
                            im: v_im[index],
                        })
                        .collect();
                    let out = run(
                        &consumer,
                        function,
                        dtype,
                        &[u.clone(), v.clone()],
                        2,
                        extent,
                    );
                    for index in 0..count {
                        let arguments = [(u[index].re, u[index].im), (v[index].re, v[index].im)];
                        let expected = complex_reference64(&plan, &arguments);
                        check_pair(
                            &format!("{what} out0[{index}]"),
                            (out[0][index].re, out[0][index].im),
                            expected[0],
                            tolerance,
                        );
                        check_pair(
                            &format!("{what} out1[{index}]"),
                            (out[1][index].re, out[1][index].im),
                            expected[1],
                            tolerance,
                        );
                    }
                }
                _ => {
                    assert_eq!(dtype, DType::C32);
                    let u: Vec<C32> = (0..count)
                        .map(|index| C32 {
                            re: u_re[index] as f32,
                            im: u_im[index] as f32,
                        })
                        .collect();
                    let v: Vec<C32> = (0..count)
                        .map(|index| C32 {
                            re: v_re[index] as f32,
                            im: v_im[index] as f32,
                        })
                        .collect();
                    let out = run(
                        &consumer,
                        function,
                        dtype,
                        &[u.clone(), v.clone()],
                        2,
                        extent,
                    );
                    for index in 0..count {
                        let arguments = [(u[index].re, u[index].im), (v[index].re, v[index].im)];
                        let expected = complex_reference32(&plan, &arguments);
                        check_pair(
                            &format!("{what} out0[{index}]"),
                            (f64::from(out[0][index].re), f64::from(out[0][index].im)),
                            (f64::from(expected[0].0), f64::from(expected[0].1)),
                            tolerance,
                        );
                        check_pair(
                            &format!("{what} out1[{index}]"),
                            (f64::from(out[1][index].re), f64::from(out[1][index].im)),
                            (f64::from(expected[1].0), f64::from(expected[1].1)),
                            tolerance,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_division_stays_scale_robust_in_a_fused_chain() {
    let mut consumer = Consumer::new();
    for (dtype, tolerance) in [(DType::C32, 1e-5), (DType::C64, 1e-13)] {
        let plan = complex_extreme_plan(dtype);
        let emitted = emit_fusion(&plan).unwrap();
        let (_, function) = consumer.prepare(&emitted, &options());

        // The first live-out of every fixture, widened to f64 for the checks below.
        let first: Vec<(f64, f64)> = if dtype == DType::C64 {
            let (lhs, rhs) = extreme64();
            let len = lhs.len() as i64;
            let out = run(
                &consumer,
                function,
                dtype,
                &[lhs.clone(), rhs.clone()],
                2,
                len,
            );
            for index in 0..len as usize {
                let arguments = [
                    (lhs[index].re, lhs[index].im),
                    (rhs[index].re, rhs[index].im),
                ];
                let expected = complex_reference64(&plan, &arguments);
                for (output, want) in expected.iter().enumerate() {
                    check_pair(
                        &format!("extreme C64 out{output}[{index}]"),
                        (out[output][index].re, out[output][index].im),
                        *want,
                        tolerance,
                    );
                }
            }
            out[0].iter().map(|value| (value.re, value.im)).collect()
        } else {
            let (lhs, rhs) = extreme32();
            let len = lhs.len() as i64;
            let out = run(
                &consumer,
                function,
                dtype,
                &[lhs.clone(), rhs.clone()],
                2,
                len,
            );
            for index in 0..len as usize {
                let arguments = [
                    (lhs[index].re, lhs[index].im),
                    (rhs[index].re, rhs[index].im),
                ];
                let expected = complex_reference32(&plan, &arguments);
                for (output, want) in expected.iter().enumerate() {
                    check_pair(
                        &format!("extreme C32 out{output}[{index}]"),
                        (
                            f64::from(out[output][index].re),
                            f64::from(out[output][index].im),
                        ),
                        (f64::from(want.0), f64::from(want.1)),
                        tolerance,
                    );
                }
            }
            out[0]
                .iter()
                .map(|value| (f64::from(value.re), f64::from(value.im)))
                .collect()
        };

        // 2 / (1e38 + 1e38 i) stays finite: the robust division, not a naive one.
        assert!(
            first[0].0.is_finite() && first[0].1.is_finite(),
            "{dtype:?}: 2 / (1e38 + 1e38 i) stays finite, got {:?}",
            first[0]
        );
        assert!(
            first[0].0 > 0.0 && first[0].1 < 0.0,
            "{dtype:?}: 2 / (1e38 + 1e38 i) has a positive real and negative imaginary part, \
             got {:?}",
            first[0]
        );
        // A subnormal divisor that equals its numerator divides to one.
        assert!(
            relative(first[1].0, 1.0, tolerance),
            "{dtype:?}: a subnormal divisor equal to its numerator divides to one, got {:?}",
            first[1]
        );
        // A finite numerator over an infinite divisor: the documented signed zeros.
        assert_eq!(
            first[2].0.to_bits(),
            0.0f64.to_bits(),
            "{dtype:?}: a finite numerator over an infinite divisor is +0 real"
        );
        assert_eq!(
            first[2].1.to_bits(),
            (-0.0f64).to_bits(),
            "{dtype:?}: a finite numerator over an infinite divisor is -0 imaginary"
        );
        // A NaN operand makes both components NaN.
        assert!(
            first[4].0.is_nan() && first[4].1.is_nan(),
            "{dtype:?}: a NaN operand makes both components NaN, got {:?}",
            first[4]
        );
    }
}

#[test]
fn fused_extrema_chain_follows_the_ordered_nan_rules() {
    let mut consumer = Consumer::new();
    // NaN operands on either side, both-NaN ties, signed zeros and an inverted range.
    let x = [
        0.0f64,
        -0.0,
        1.0,
        -5.0,
        5.0,
        f64::NAN,
        2.0,
        10.0,
        3.0,
        10.0,
        f64::NAN,
        f64::NAN,
        -0.0,
        -0.0,
    ];
    let y = [
        2.0f64,
        -1.0,
        -1.0,
        -5.0,
        -0.0,
        1.0,
        2.0,
        f64::NAN,
        3.0,
        1.0,
        f64::NAN,
        4.0,
        -0.0,
        -0.0,
    ];
    let lo = [
        -1.0f64,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        f64::NAN,
        5.0,
        5.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
    ];
    let hi = [
        1.0f64,
        1.0,
        3.0,
        1.0,
        3.0,
        1.0,
        9.0,
        1.0,
        1.0,
        f64::NAN,
        1.0,
        1.0,
        1.0,
        1.0,
    ];

    for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::F64, 1e-13)] {
        let plan = extrema_plan(dtype);
        let emitted = emit_fusion(&plan).unwrap();
        let (_, function) = consumer.prepare(&emitted, &options());
        let len = x.len() as i64;
        let (rows, exact) = match dtype {
            DType::F64 => {
                let columns = vec![x.to_vec(), y.to_vec(), lo.to_vec(), hi.to_vec()];
                (
                    run_rows(&consumer, function, dtype, &columns, 2, len),
                    columns,
                )
            }
            _ => {
                let columns = narrow(&[x.to_vec(), y.to_vec(), lo.to_vec(), hi.to_vec()]);
                (
                    run_rows(&consumer, function, dtype, &columns, 2, len),
                    widen(&columns),
                )
            }
        };
        check_rows(
            &format!("extrema {dtype:?}"),
            &plan,
            &exact,
            &rows,
            tolerance,
        );

        // The documented cases, by signed-zero sign: these ops only select operands,
        // so their results are exact for both dtypes.
        let is_negative_zero = |value: f64| value == 0.0 && value.is_sign_negative();
        assert!(
            is_negative_zero(rows[1][1]),
            "{dtype:?}: an in-range signed zero survives clamp, got {:?}",
            rows[1][1]
        );
        assert!(
            is_negative_zero(rows[0][13]),
            "{dtype:?}: min(max(-0.0, -0.0), 1.0) keeps the -0, got {:?}",
            rows[0][13]
        );
        assert!(
            rows[0][5].is_nan()
                && rows[0][7].is_nan()
                && rows[0][10].is_nan()
                && rows[0][11].is_nan(),
            "{dtype:?}: a NaN operand propagates through max then min"
        );
        assert!(
            rows[1][5].is_nan(),
            "{dtype:?}: a NaN input propagates through clamp"
        );
        assert_eq!(
            rows[1][6], 2.0,
            "{dtype:?}: a NaN lower bound disables that comparison"
        );
        assert_eq!(
            rows[1][7], 1.0,
            "{dtype:?}: inverted bounds clamp by statement order"
        );
        assert_eq!(
            rows[1][8], 5.0,
            "{dtype:?}: an inverted lower bound wins while x < lo"
        );
        assert_eq!(
            rows[1][9], 10.0,
            "{dtype:?}: a NaN upper bound disables that comparison"
        );
    }
}

#[test]
fn fused_contraction_is_a_documented_tolerance_level_difference() {
    // Design §3: the emitted chain is ordinary CUDA C++, so the compiler may contract
    // `a*b + c` into an FMA, exactly as a revision-6 per-op launch has the same freedom
    // inside one operation. With `a = FLT_MAX`, `b = 2`, `c = -FLT_MAX` the contracted
    // form is the finite FLT_MAX, while two separate launches overflow the product to
    // +inf and stay there (`inf - FLT_MAX = inf`). This fixture is therefore recorded
    // as a documented tolerance-level difference, never as bit equality: only the
    // documented finiteness alternatives are asserted.
    let (a, b, c) = (f32::MAX, 2.0f32, -f32::MAX);
    let plan = FusionPlan {
        dtype: DType::F32,
        inputs: 3,
        instructions: vec![
            instruction(FusionOp::Mul, &[input(0), input(1)]),
            instruction(FusionOp::Add, &[value(0), input(2)]),
        ],
        outputs: vec![value(1)],
    };

    // The separate launches need the revision-6 entries, so the harness session loads
    // their nvcc module as well as hosting the NVRTC module.
    let mut consumer = Consumer::with_session(Session::with_modules(&["elementwise"]));
    let emitted = emit_fusion(&plan).unwrap();
    let (_, function) = consumer.prepare(&emitted, &options());
    let fused: Vec<f32> = run(
        &consumer,
        function,
        DType::F32,
        &[vec![a], vec![b], vec![c]],
        1,
        1,
    )
    .remove(0);
    assert!(
        fused[0] == f32::MAX || fused[0] == f32::INFINITY,
        "the fused a*b + c is either the contracted FLT_MAX or the overflowing +inf, got {:?}",
        fused[0]
    );

    // Two separate launches: `MUL(a, b)` then the combined `AXPBY` with (1, 1), which is
    // the per-op form that adds `c` (the revision-6 elementwise entry has no plain add).
    let session = consumer.session();
    let a_buf = upload(session, &[a]);
    let b_buf = upload(session, &[b]);
    let c_buf = upload(session, &[c]);
    let coefficients = written(session, gpu::as_bytes(&[1.0f32, 1.0]));
    let mid = written(session, &vec![SENTINEL; 4 + GUARD]);
    let out = written(session, &vec![SENTINEL; 4 + GUARD]);
    let err = written(session, &[0u8; 4]);
    let mul = plan_binary(BinaryOp::Mul, DType::F32, ScalarSide::Right, 1).unwrap();
    let mut slots = [
        Slot::ptr(&mid),
        Slot::ptr(&a_buf),
        Slot::ptr(&b_buf),
        Slot::i32(DType::F32.tag()),
        Slot::i32(BinaryOp::Mul.tag()),
        Slot::i32(ScalarSide::Right.tag()),
        Slot::i64(1),
        Slot::ptr(&err),
    ];
    launch(
        session.function("t4a_cuda_binary"),
        &mul,
        session.stream(),
        &mut slots,
    );
    let product = from_bytes::<f32>(&read_prefix(session, &mid, 4))[0];
    assert_eq!(product, f32::INFINITY, "FLT_MAX * 2 overflows");

    let layout = Layout::compact(&[1]).unwrap();
    let add = plan_combined_strided(
        StridedOp::Axpby,
        DType::F32,
        &layout,
        1,
        &layout,
        1,
        &layout,
        1,
    )
    .unwrap();
    let mut slots = [
        Slot::ptr(&out),
        Slot::ptr(&mid),
        Slot::ptr(&c_buf),
        Slot::ptr(&coefficients),
        Slot::layout(&t4a_cuda_kernels::CLayout::new(&layout)),
        Slot::layout(&t4a_cuda_kernels::CLayout::new(&layout)),
        Slot::layout(&t4a_cuda_kernels::CLayout::new(&layout)),
        Slot::i32(DType::F32.tag()),
        Slot::i32(StridedOp::Axpby.tag()),
        Slot::i64(1),
    ];
    launch(
        session.function("t4a_cuda_combined_strided"),
        &add,
        session.stream(),
        &mut slots,
    );
    let separate = from_bytes::<f32>(&read_prefix(session, &out, 4))[0];
    assert_eq!(
        separate,
        f32::INFINITY,
        "the two separate launches overflow to +inf"
    );
}

// --- prepare/cache counters -------------------------------------------------

#[test]
fn prepare_and_cache_counters_hold_across_extents_options_and_archs() {
    let mut consumer = Consumer::new();
    let plan = real_main_plan(DType::F32);
    let emitted = emit_fusion(&plan).unwrap();
    let opts = options();
    let compiles = nvrtc_calls();
    let loads = module_loads();

    // One plan and one artifact key across four different extents: exactly one NVRTC
    // compile and one module load, and every extent still runs its own geometry.
    let (key0, function0) = consumer.prepare(&emitted, &opts);
    for len in [1i64, 513, 65536, 7] {
        let (key, function) = consumer.prepare(&emitted, &opts);
        assert_eq!(key, key0, "len {len} changed the artifact key");
        assert_eq!(function, function0, "len {len} reloaded the module");
        let inputs = narrow(&real_main_inputs(len));
        let rows = run_rows(&consumer, function, DType::F32, &inputs, 2, len);
        assert_eq!(
            rows[0].len(),
            len as usize,
            "len {len} wrote the wrong extent"
        );
        // Values, not just lengths: a kernel that wrote nothing would pass the length
        // check, so every extent is compared against the CPU reference.
        let exact = widen(&inputs);
        check_rows("cache reuse", &plan, &exact, &rows, 1e-5);
    }
    assert_eq!(
        nvrtc_calls(),
        compiles + 1,
        "one artifact key must cost exactly one NVRTC compile"
    );
    assert_eq!(
        module_loads(),
        loads + 1,
        "one artifact key must cost exactly one module load"
    );

    // Four warm launches: zero further NVRTC calls and one physical launch each.
    let (_, function) = consumer.prepare(&emitted, &opts);
    let inputs = narrow(&real_main_inputs(513));
    let exact = widen(&inputs);
    for warm in 0..4 {
        let before = launches();
        let rows = run_rows(&consumer, function, DType::F32, &inputs, 2, 513);
        assert_eq!(launches(), before + 1, "warm launch {warm} is one launch");
        assert_eq!(rows.len(), 2);
        check_rows("warm launch", &plan, &exact, &rows, 1e-5);
        assert_eq!(
            nvrtc_calls(),
            compiles + 1,
            "a warm launch must not compile"
        );
        assert_eq!(module_loads(), loads + 1, "a warm launch must not load");
    }

    // A different op set: a different key and a second compile.
    let other = emit_fusion(&real_wide_plan(DType::F32)).unwrap();
    let (key1, _) = consumer.prepare(&other, &opts);
    assert_ne!(key1, key0, "a different op set needs its own artifact");
    assert_eq!(nvrtc_calls(), compiles + 2);

    // A changed option string: a further distinct key and compile.
    let no_fmad = CompileOptions {
        arch: Some(ARCH),
        fmad: Some(false),
        ..Default::default()
    };
    let (key2, _) = consumer.prepare(&emitted, &no_fmad);
    assert_ne!(key2, key0, "a changed option string needs its own artifact");
    assert_ne!(key2, key1);
    assert_eq!(nvrtc_calls(), compiles + 3);

    // A changed arch: a further distinct key and compile.
    let narrow_arch = CompileOptions {
        arch: Some("sm_75"),
        ..Default::default()
    };
    let (key3, _) = consumer.prepare(&emitted, &narrow_arch);
    assert_ne!(key3, key0, "a changed target needs its own artifact");
    assert_ne!(key3, key2);
    assert_eq!(nvrtc_calls(), compiles + 4);
    assert_eq!(module_loads(), loads + 4, "one load per compiled artifact");
}

#[test]
fn two_independent_caches_load_their_own_module() {
    let mut first = Consumer::new();
    let mut second = Consumer::new();
    let plan = real_main_plan(DType::F32);
    let emitted = emit_fusion(&plan).unwrap();
    let opts = options();
    let compiles = nvrtc_calls();
    let loads = module_loads();

    let (key, first_function) = first.prepare(&emitted, &opts);
    let (other_key, second_function) = second.prepare(&emitted, &opts);
    assert_eq!(key, other_key, "both runtimes key on the same artifact");
    assert_eq!(
        nvrtc_calls(),
        compiles + 2,
        "each runtime compiles for its own cache"
    );
    assert_eq!(
        module_loads(),
        loads + 2,
        "each runtime loads its own module"
    );
    assert_ne!(
        first_function, second_function,
        "each runtime's cache holds its own loaded module"
    );

    let inputs = narrow(&real_main_inputs(513));
    let exact = widen(&inputs);
    drop(first);

    // The surviving runtime's binding is still usable after the other cache (and its
    // module) is gone.
    let rows = run_rows(&second, second_function, DType::F32, &inputs, 2, 513);
    check_rows("surviving cache", &plan, &exact, &rows, 1e-5);
    assert_eq!(
        nvrtc_calls(),
        compiles + 2,
        "the surviving cache must not recompile"
    );
}

#[test]
fn emitted_source_compiles_under_nvrtc_verbatim() {
    // Design §3/§7: the emitted source is self-contained and the caller only hands it to
    // NVRTC, so the pristine text — no include path, no repair — must compile. The two
    // defects this test found are fixed in the canonical header (the offset assertions
    // are skipped under `__CUDACC_RTC__`, and the value-form complex division is a pair
    // of overloads), so this check runs on the pristine text like every other test.
    for plan in [real_main_plan(DType::F64), complex_chain_plan(DType::C64)] {
        let emitted = emit_fusion(&plan).unwrap();
        assert_eq!(
            emitted.source().matches("#include").count(),
            0,
            "{:?}: the emitted source must not include anything",
            plan.dtype
        );
        compile_ptx_with_opts(&emitted.source(), options()).unwrap_or_else(|error| {
            panic!(
                "{:?}: NVRTC must accept the emitted source verbatim: {error:?}",
                plan.dtype
            )
        });
    }
}
