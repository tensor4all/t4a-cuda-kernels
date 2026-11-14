//! The Metal entries on a real Apple GPU. Needs `--features gpu-tests-metal`.
//!
//! Every test compares against a CPU reference over the whole result and fills the
//! destination with a sentinel past `len`, so a kernel that misses elements or
//! writes past the end fails here. The lengths include 4096 and 100_003 elements,
//! which no single threadgroup covers: that is the only size range in which a
//! kernel that never leaves the first thread fails.

use std::collections::HashMap;

use t4a_cuda_kernels::metal::{self, Dispatch, SYMBOLS};
use t4a_cuda_kernels::{DType, Entry, UnaryOp};
use t4a_metal_runtime::{
    source_hash, Arg, BoundFunction, Buffer, CompileRequest, LaunchConfig, Runtime,
};

/// Lengths every entry is exercised over. `1`, `3`, `255`, `257` and `100_003` are
/// not multiples of the threadgroup width, so their last group is a partial one.
const LENS: [usize; 7] = [1, 3, 255, 256, 257, 4096, 100_003];

/// Elements a destination carries beyond `len`, so a tail overrun is visible.
const SLACK: usize = 16;

/// The sentinel every destination starts with: an f32 NaN payload, so a stray
/// write into the slack is one word comparison away.
const SENTINEL: u32 = 0x7fc0_dead;

struct Device {
    runtime: Runtime,
    functions: HashMap<&'static str, BoundFunction>,
    sources: HashMap<&'static str, String>,
}

impl Device {
    /// Open the GPU and compile every Metal module once.
    ///
    /// A host without a usable device fails this lane instead of skipping it.
    fn open() -> Self {
        let mut runtime = Runtime::open(Default::default())
            .unwrap_or_else(|error| panic!("this lane needs an Apple GPU: {error}"));
        let mut functions = HashMap::new();
        let mut sources = HashMap::new();
        for module in metal::source::MODULES {
            let source = metal::source::compose(module);
            let artifact = runtime
                .compile(&CompileRequest::new(module.name, source.clone()))
                .unwrap_or_else(|error| panic!("{}: {error}", module.name));
            assert_eq!(
                artifact.source_hash(),
                source_hash(&source),
                "{}: the compiled artifact is the composed source this crate ships",
                module.name
            );
            for (entry, symbol) in SYMBOLS {
                if metal::source::module_of(*entry) == Some(module) {
                    let function = runtime
                        .bind(&artifact, symbol)
                        .unwrap_or_else(|error| panic!("{symbol}: {error}"));
                    functions.insert(*symbol, function);
                    sources.insert(*symbol, artifact.source_hash().to_string());
                }
            }
        }
        assert_eq!(
            functions.len(),
            SYMBOLS.len(),
            "every served entry is bound on the device"
        );
        Self {
            runtime,
            functions,
            sources,
        }
    }

    /// A destination of `len + SLACK` elements, sentinel-filled.
    fn sentinel_buffer(&mut self, dtype: DType, len: usize) -> Buffer {
        let bytes = (len + SLACK) * dtype.size();
        let buffer = self.runtime.alloc_zeroed(bytes).unwrap();
        // SAFETY: the buffer is fresh, `bytes` long, and no submission references it
        // yet, so the host may write every byte.
        unsafe {
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr().cast::<u32>(), bytes / 4);
            words.fill(SENTINEL);
        }
        buffer
    }

    /// A buffer holding `values`, with the slack sentinel-filled.
    fn host_buffer<T: Copy>(&mut self, dtype: DType, values: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(values);
        assert_eq!(
            bytes % dtype.size(),
            0,
            "the host slice must be a whole number of {dtype:?} elements"
        );
        let buffer = self.sentinel_buffer(dtype, bytes / dtype.size());
        // SAFETY: as above; the source slice is sized `T` and the buffer holds
        // `values.len()` such elements.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), buffer.host_ptr(), bytes);
        }
        buffer
    }

    /// One dispatch of `symbol` over the planned geometry, then a wait.
    ///
    /// Asserting the launch counter here is what makes "one entry call is one
    /// dispatch" mechanical rather than a claim.
    fn dispatch(&mut self, symbol: &'static str, plan: Dispatch, args: &[Arg<'_>]) {
        let function = self.functions.get(symbol).expect("a bound entry");
        let before = self.runtime.stats().launches;
        // SAFETY: every caller passes the operand layout the entry documents.
        unsafe {
            self.runtime
                .launch(
                    function,
                    LaunchConfig::tiled(plan.threadgroups, plan.threads_per_threadgroup),
                    args,
                )
                .unwrap_or_else(|error| panic!("{symbol}: {error}"));
        }
        self.runtime.commit().unwrap().wait().unwrap();
        assert_eq!(
            self.runtime.stats().launches,
            before + 1,
            "{symbol}: one entry call must be one dispatch"
        );
    }

    fn source_hash(&self, symbol: &str) -> &str {
        self.sources.get(symbol).expect("a bound entry")
    }
}

fn plan(entry: Entry, dtype: DType, len: usize) -> Dispatch {
    let plan = metal::plan_elementwise(entry, dtype, len as i64).unwrap();
    assert_eq!(plan.bytes, (len * dtype.size()) as u64);
    assert!(
        u64::from(plan.dispatch.threadgroups) * u64::from(plan.dispatch.threads_per_threadgroup)
            >= len as u64,
        "the plan must cover every element"
    );
    plan.dispatch
}

/// The `len` elements of `dtype` that start at element `offset`.
///
/// # Safety
///
/// The buffer holds at least `offset + len` elements of `dtype`, and every
/// submission that could touch it has completed.
unsafe fn read_f32(buffer: &Buffer, offset: usize, len: usize) -> &[f32] {
    std::slice::from_raw_parts(buffer.host_ptr().cast::<f32>().add(offset), len)
}

/// Element `index` of `dtype` as words, as read back from the device.
///
/// # Safety
///
/// As [`read_f32`].
unsafe fn element_words(buffer: &Buffer, dtype: DType, index: usize) -> Vec<u32> {
    let stride = dtype.size() / 4;
    std::slice::from_raw_parts(buffer.host_ptr().cast::<u32>().add(index * stride), stride).to_vec()
}

fn assert_tail_is_sentinel(buffer: &Buffer, dtype: DType, len: usize, what: &str) {
    for index in len..len + SLACK {
        // SAFETY: the buffer holds `len + SLACK` elements and every submission that
        // could touch it has completed.
        let words = unsafe { element_words(buffer, dtype, index) };
        assert!(
            words.iter().all(|word| *word == SENTINEL),
            "{what}: element {index} past len {len} was written"
        );
    }
}

/// Distance between two finite `f32` values in units in the last place.
fn ulps(a: f32, b: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

fn assert_within_ulps(got: &[f32], want: &[f32], tolerance: u32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        // Classification first: `f32::MAX` and `+inf` are adjacent bit patterns, so a
        // tolerance alone would accept an infinity where a finite value is expected.
        assert_eq!(
            got.is_nan(),
            want.is_nan(),
            "{what}: element {index} is {got}, the reference is {want}"
        );
        if want.is_nan() {
            continue;
        }
        assert_eq!(
            got.is_infinite(),
            want.is_infinite(),
            "{what}: element {index} is {got}, the reference is {want}"
        );
        if want.is_infinite() {
            assert_eq!(
                got.is_sign_positive(),
                want.is_sign_positive(),
                "{what}: element {index} is {got}, the reference is {want}"
            );
            continue;
        }
        assert!(
            ulps(*got, *want) <= tolerance,
            "{what}: element {index} is {got} but the reference is {want} ({} ulp)",
            ulps(*got, *want)
        );
    }
}

#[test]
fn every_module_compiles_and_binds() {
    let device = Device::open();
    for (_, symbol) in SYMBOLS {
        assert_eq!(
            device.source_hash(symbol).len(),
            64,
            "{symbol}: the artifact records the SHA-256 of its source"
        );
    }
}

#[test]
fn fill_zero_is_bit_exact_and_guards_the_tail() {
    let mut device = Device::open();
    for dtype in metal::DTYPES {
        for len in LENS {
            let out = device.sentinel_buffer(dtype, len);
            let dtype_tag = dtype.tag();
            let length = len as i64;
            device.dispatch(
                "t4a_metal_fill_zero",
                plan(Entry::FillZero, dtype, len),
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::pod(&length),
                ],
            );
            for index in 0..len {
                // SAFETY: every submission touching `out` has completed.
                let words = unsafe { element_words(&out, dtype, index) };
                assert!(
                    words.iter().all(|word| *word == 0),
                    "fill_zero {dtype:?} len {len}: element {index} is not zero"
                );
            }
            assert_tail_is_sentinel(&out, dtype, len, &format!("fill_zero {dtype:?} len {len}"));
        }
    }
}

#[test]
fn copy_exact_moves_bits_and_guards_the_tail() {
    let mut device = Device::open();
    // A NaN payload, a signed zero and a subnormal: an arithmetic copy would not
    // keep all three.
    let words = [0x7fc0_0001u32, 0x8000_0000, 0x0000_0001, 0x3f80_0000];
    for dtype in metal::DTYPES {
        for len in LENS {
            let elements = len * (dtype.size() / 4);
            let source: Vec<u32> = (0..elements).map(|i| words[i % words.len()]).collect();
            let input = device.host_buffer(dtype, &source);
            let out = device.sentinel_buffer(dtype, len);
            let dtype_tag = dtype.tag();
            let length = len as i64;
            device.dispatch(
                "t4a_metal_copy_exact",
                plan(Entry::CopyExact, dtype, len),
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::pod(&length),
                ],
            );
            let got = unsafe { read_f32(&out, 0, elements) };
            let want = unsafe { read_f32(&input, 0, elements) };
            assert_eq!(
                got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "copy_exact {dtype:?} len {len} must move bit patterns"
            );
            assert_tail_is_sentinel(&out, dtype, len, &format!("copy_exact {dtype:?} len {len}"));
        }
    }
}

/// Reference for one arithmetic entry on f32.
fn arithmetic_f32(entry: Entry, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| match entry {
            Entry::Add => a + b,
            Entry::Sub => a - b,
            Entry::Mul => a * b,
            Entry::Div => a / b,
            other => panic!("{other:?} is not an arithmetic entry"),
        })
        .collect()
}

/// Reference for one arithmetic entry on Complex32, as `(re, im)` pairs.
fn arithmetic_c32(entry: Entry, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.chunks_exact(2)
        .zip(rhs.chunks_exact(2))
        .flat_map(|(a, b)| {
            let (ar, ai) = (a[0], a[1]);
            let (br, bi) = (b[0], b[1]);
            match entry {
                Entry::Add => [ar + br, ai + bi],
                Entry::Sub => [ar - br, ai - bi],
                Entry::Mul => [ar * br - ai * bi, ar * bi + ai * br],
                other => panic!("{other:?} is not served for Complex32"),
            }
        })
        .collect()
}

fn sample_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| match i % 6 {
            0 => 0.5,
            1 => -1.25,
            2 => 3.0,
            3 => -0.0,
            4 => 1e-8,
            _ => 7.5,
        })
        .collect()
}

#[test]
fn arithmetic_matches_the_reference() {
    let mut device = Device::open();
    // A multiplication of two Complex32 values is three IEEE operations whose
    // rounding the compiler may fuse, so the comparison is in ulp, not bits.
    let tolerance = 4;
    for entry in [Entry::Add, Entry::Sub, Entry::Mul, Entry::Div] {
        let dtypes: &[DType] = if entry == Entry::Div {
            &[DType::F32]
        } else {
            &metal::DTYPES
        };
        for &dtype in dtypes {
            for len in LENS {
                let elements = len * (dtype.size() / 4);
                let lhs = sample_f32(elements);
                let rhs = sample_f32(elements)
                    .iter()
                    .map(|value| value + 0.25)
                    .collect::<Vec<_>>();
                let lhs_buffer = device.host_buffer(dtype, &lhs);
                let rhs_buffer = device.host_buffer(dtype, &rhs);
                let out = device.sentinel_buffer(dtype, len);
                let symbol = metal::symbol(entry).unwrap();
                let dtype_tag = dtype.tag();
                let length = len as i64;
                device.dispatch(
                    symbol,
                    plan(entry, dtype, len),
                    &[
                        Arg::Buffer {
                            buffer: &out,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &lhs_buffer,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &rhs_buffer,
                            offset: 0,
                        },
                        Arg::pod(&dtype_tag),
                        Arg::pod(&length),
                    ],
                );
                let want = if dtype == DType::F32 {
                    arithmetic_f32(entry, &lhs, &rhs)
                } else {
                    arithmetic_c32(entry, &lhs, &rhs)
                };
                let got = unsafe { read_f32(&out, 0, elements) };
                // Over normal-range operands every f32 entry is one correctly rounded
                // IEEE operation, and Complex32 add/sub are two, so their tolerance
                // stays zero. Complex32 mul is three operations the compiler may fuse.
                let tolerance = if dtype == DType::F32 || matches!(entry, Entry::Add | Entry::Sub) {
                    0
                } else {
                    tolerance
                };
                assert_within_ulps(
                    got,
                    &want,
                    tolerance,
                    &format!("{entry:?} {dtype:?} len {len}"),
                );
                assert_tail_is_sentinel(
                    &out,
                    dtype,
                    len,
                    &format!("{entry:?} {dtype:?} len {len}"),
                );
            }
        }
    }
}

#[test]
fn nonfinite_operands_propagate() {
    let mut device = Device::open();
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let lhs = [1.0f32, inf, 0.0, nan, -2.0, 3.0];
    let rhs = [inf, -1.0, 4.0, 1.0, nan, 0.0];
    for entry in [Entry::Add, Entry::Sub, Entry::Mul] {
        let len = lhs.len();
        let lhs_buffer = device.host_buffer(DType::F32, &lhs);
        let rhs_buffer = device.host_buffer(DType::F32, &rhs);
        let out = device.sentinel_buffer(DType::F32, len);
        let dtype_tag = DType::F32.tag();
        let length = len as i64;
        device.dispatch(
            metal::symbol(entry).unwrap(),
            plan(entry, DType::F32, len),
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &lhs_buffer,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &rhs_buffer,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::pod(&length),
            ],
        );
        let got = unsafe { read_f32(&out, 0, len) };
        assert_within_ulps(
            got,
            &arithmetic_f32(entry, &lhs, &rhs),
            0,
            &format!("{entry:?}"),
        );
    }
}

fn unary_reference(op: UnaryOp, value: f32) -> f32 {
    match op {
        UnaryOp::Neg => -value,
        UnaryOp::Exp => value.exp(),
        UnaryOp::Log => value.ln(),
        UnaryOp::Sin => value.sin(),
        UnaryOp::Cos => value.cos(),
        UnaryOp::Tanh => value.tanh(),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Rsqrt => 1.0 / value.sqrt(),
        UnaryOp::Abs => value.abs(),
        UnaryOp::Sign => {
            if value.is_nan() {
                value
            } else if value == 0.0 {
                0.0
            } else if value > 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        other => panic!("{other:?} has no Metal kernel"),
    }
}

#[test]
fn unary_matches_the_reference() {
    let mut device = Device::open();
    let positive = [0.125f32, 0.5, 1.0, 2.0, 4.0, 9.0];
    let signed = [0.0f32, -0.0, -0.5, -2.0, -4.0, 6.5];
    for op in metal::UNARY_OPS {
        // `Conj` is complex-only: `UnaryOp::accepts` rejects the f32 pair, and the
        // complex case is covered below.
        if op == UnaryOp::Conj {
            continue;
        }
        // log, sqrt and rsqrt need a positive operand.
        let fixture: &[f32] = if matches!(op, UnaryOp::Log | UnaryOp::Sqrt | UnaryOp::Rsqrt) {
            &positive
        } else {
            &signed
        };
        // The tolerance is the contract: negation, the absolute value and the sign
        // are exact, and the transcendentals agree with the host's libm.
        let tolerance = if matches!(op, UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sign) {
            0
        } else {
            8
        };
        // Every length, not one: a kernel that stopped after the first threadgroup
        // would match a six-element fixture exactly.
        for len in LENS {
            let values: Vec<f32> = (0..len)
                .map(|index| fixture[index % fixture.len()])
                .collect();
            let input = device.host_buffer(DType::F32, &values);
            let out = device.sentinel_buffer(DType::F32, len);
            let dst = DType::F32.tag();
            let src = DType::F32.tag();
            let tag = op.tag();
            let length = len as i64;
            device.dispatch(
                "t4a_metal_unary",
                metal::plan_unary(op, DType::F32, DType::F32, length)
                    .unwrap()
                    .dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&dst),
                    Arg::pod(&src),
                    Arg::pod(&tag),
                    Arg::pod(&length),
                ],
            );
            let want: Vec<f32> = values
                .iter()
                .map(|value| unary_reference(op, *value))
                .collect();
            let got = unsafe { read_f32(&out, 0, len) };
            let what = format!("unary {op:?} len {len}");
            assert_within_ulps(got, &want, tolerance, &what);
            assert_tail_is_sentinel(&out, DType::F32, len, &what);
        }
    }
}

#[test]
fn nonfinite_unary_operands_propagate() {
    // Case by case, because the reference pins only these: `sin(inf)` and
    // `tanh(inf)` are not worth asserting against two different libms.
    let mut device = Device::open();
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    let cases: &[(UnaryOp, &[f32], &[f32])] = &[
        (UnaryOp::Neg, &[inf, -inf, nan], &[-inf, inf, nan]),
        (UnaryOp::Abs, &[inf, -inf, nan], &[inf, inf, nan]),
        (
            UnaryOp::Sign,
            &[inf, -inf, nan, 0.0, -0.0],
            &[1.0, -1.0, nan, 0.0, 0.0],
        ),
        (UnaryOp::Exp, &[inf, -inf], &[inf, 0.0]),
        (UnaryOp::Log, &[inf, 0.0], &[inf, -inf]),
    ];
    for (op, values, want) in cases {
        let len = values.len();
        let input = device.host_buffer(DType::F32, values);
        let out = device.sentinel_buffer(DType::F32, len);
        let dst = DType::F32.tag();
        let src = DType::F32.tag();
        let tag = op.tag();
        let length = len as i64;
        device.dispatch(
            "t4a_metal_unary",
            metal::plan_unary(*op, DType::F32, DType::F32, length)
                .unwrap()
                .dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &input,
                    offset: 0,
                },
                Arg::pod(&dst),
                Arg::pod(&src),
                Arg::pod(&tag),
                Arg::pod(&length),
            ],
        );
        let got = unsafe { read_f32(&out, 0, len) };
        let what = format!("nonfinite unary {op:?}");
        assert_within_ulps(got, want, 0, &what);
        assert_tail_is_sentinel(&out, DType::F32, len, &what);
    }
}

/// One arithmetic case: the entry, its two operands and the expected result.
type SubnormalCase<'a> = (Entry, &'a [f32], &'a [f32], &'a [f32]);

#[test]
fn fast_math_flushes_subnormals_and_the_normals_stay_exact() {
    // Measured, because it decides what the numerics section of the ABI document may
    // claim: the runtime compiles with Metal\'s default options, where fast math is on,
    // and this device flushes a subnormal operand and a subnormal result to zero
    // (`smallest + smallest` is 0, not 2 * smallest). The CUDA entries do not flush,
    // because nvcc runs without `--use_fast_math`, so the two backends differ exactly
    // there. Asserting it here keeps the difference visible: a change of behaviour
    // fails this test instead of passing unnoticed.
    //
    // The seam that would remove it is `MTLCompileOptions.fastMathEnabled = NO`, which
    // `t4a-metal-runtime` does not expose yet (it passes nil options).
    let mut device = Device::open();
    let smallest = f32::from_bits(1); // 1.4e-45, the smallest subnormal
    let normal = f32::MIN_POSITIVE; // 1.17e-38, the smallest normal
    let max = f32::MAX;
    let cases: &[SubnormalCase<'_>] = &[
        // entry, lhs, rhs, reference: subnormal operands and results flush to zero
        (
            Entry::Add,
            &[smallest, smallest, normal, normal],
            &[smallest, 1.0, normal, -normal],
            &[0.0, 1.0, 2.0 * normal, 0.0],
        ),
        (
            Entry::Mul,
            &[smallest, normal, normal, max],
            &[2.0, 2.0, 0.5, 0.5],
            &[0.0, 2.0 * normal, 0.0, max * 0.5],
        ),
        (
            Entry::Div,
            &[smallest, normal, normal, normal, 0.0, 0.0, -1.0, smallest],
            &[2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, smallest],
            &[
                0.0,
                normal,
                0.0,
                f32::INFINITY,
                f32::NAN,
                f32::NAN,
                f32::NEG_INFINITY,
                f32::NAN,
            ],
        ),
    ];
    for (entry, lhs, rhs, want) in cases {
        let len = lhs.len();
        let lhs_buffer = device.host_buffer(DType::F32, lhs);
        let rhs_buffer = device.host_buffer(DType::F32, rhs);
        let out = device.sentinel_buffer(DType::F32, len);
        let dtype_tag = DType::F32.tag();
        let length = len as i64;
        device.dispatch(
            metal::symbol(*entry).unwrap(),
            plan(*entry, DType::F32, len),
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &lhs_buffer,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &rhs_buffer,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::pod(&length),
            ],
        );
        let got = unsafe { read_f32(&out, 0, len) };
        let what = format!("{entry:?} subnormal");
        assert_within_ulps(got, want, 0, &what);
        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "{what}: the bits, not only the value"
        );
        assert_tail_is_sentinel(&out, DType::F32, len, &what);
    }
}

#[test]
fn subnormal_complex_abs_and_sign_agree() {
    // `abs` flushes a subnormal magnitude to zero under fast math. `sign` divides by
    // that magnitude, so guarding only the exact (0, 0) would return a direction of
    // inf or NaN for an operand whose magnitude is the only thing that vanished.
    // Reference: `hypotf` says these are nonzero, the device flushes them, and both
    // entries must therefore agree on zero.
    let mut device = Device::open();
    let smallest = f32::from_bits(1);
    // The reference is nonzero: `hypotf(smallest, 0)` is `smallest`. So this is exactly
    // the case where the flush, not the formula, decides the answer.
    assert_eq!(
        complex_magnitude(smallest, 0.0).to_bits(),
        smallest.to_bits()
    );
    let values = [smallest, 0.0, 0.0, smallest, smallest, smallest];
    let len = values.len() / 2;
    let got_abs = complex_unary(
        &mut device,
        UnaryOp::Abs,
        DType::F32,
        &values,
        "subnormal abs",
    );
    assert_eq!(
        got_abs
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0u32; len],
        "abs of a subnormal Complex32 flushes to zero"
    );
    let got_sign = complex_unary(
        &mut device,
        UnaryOp::Sign,
        DType::C32,
        &values,
        "subnormal sign",
    );
    assert_eq!(
        got_sign
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0u32; values.len()],
        "sign of a subnormal Complex32 must be zero, not a direction built on 1/0"
    );
}

#[test]
fn division_is_correctly_rounded() {
    // `metal::precise::divide` should be the IEEE operation, which would make the
    // Metal entry bit-compatible with the CUDA one. Widely ranged operands catch a
    // reciprocal-based approximation, which is within a few ulp on average and
    // wrong on the last bit almost everywhere.
    let mut device = Device::open();
    for len in LENS {
        let lhs = sample_f32(len);
        let mut rhs = sample_f32(len);
        for (index, value) in rhs.iter_mut().enumerate() {
            *value = 0.1 + (index % 97) as f32 * 0.37;
        }
        let lhs_buffer = device.host_buffer(DType::F32, &lhs);
        let rhs_buffer = device.host_buffer(DType::F32, &rhs);
        let out = device.sentinel_buffer(DType::F32, len);
        let dtype_tag = DType::F32.tag();
        let length = len as i64;
        device.dispatch(
            metal::symbol(Entry::Div).unwrap(),
            plan(Entry::Div, DType::F32, len),
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &lhs_buffer,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &rhs_buffer,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::pod(&length),
            ],
        );
        let got = unsafe { read_f32(&out, 0, len) };
        let want = arithmetic_f32(Entry::Div, &lhs, &rhs);
        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "div len {len}: every quotient must be the correctly rounded one"
        );
    }
}

/// Dispatch one Complex32 unary operation over `values` and return the result as f32
/// words, asserting the destination tail.
fn complex_unary(
    device: &mut Device,
    op: UnaryOp,
    dst: DType,
    values: &[f32],
    what: &str,
) -> Vec<f32> {
    let len = values.len() / 2;
    let input = device.host_buffer(DType::C32, values);
    let out = device.sentinel_buffer(dst, len);
    let dst_tag = dst.tag();
    let src_tag = DType::C32.tag();
    let tag = op.tag();
    let length = len as i64;
    device.dispatch(
        "t4a_metal_unary",
        metal::plan_unary(op, dst, DType::C32, length)
            .unwrap()
            .dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &input,
                offset: 0,
            },
            Arg::pod(&dst_tag),
            Arg::pod(&src_tag),
            Arg::pod(&tag),
            Arg::pod(&length),
        ],
    );
    assert_tail_is_sentinel(&out, dst, len, what);
    let words = if dst == DType::F32 { len } else { values.len() };
    unsafe { read_f32(&out, 0, words) }.to_vec()
}

#[test]
fn complex_unary_and_abs_match_the_reference() {
    let mut device = Device::open();
    // (re, im) pairs: zeros, the extremes `abs` must survive, and the NaN cases whose
    // precedence `hypotf` pins — a NaN with a zero, a zero with a NaN, and a NaN with an
    // infinity, which is the infinity.
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let values = [
        1.0f32, 2.0, 0.0, 0.0, -0.0, -0.0, -3.0, 4.0, 1e-30, 1e-30, 1e30, 1e30, nan, 0.0, 0.0, nan,
        inf, nan,
    ];
    let pairs = values.len() / 2;
    let tolerance = 4;

    // Every operation over every length, with the fixture repeated: a kernel that
    // stopped after the first threadgroup would match the nine-element fixture exactly.
    for len in LENS {
        let repeated: Vec<f32> = (0..len)
            .flat_map(|index| {
                let pair = index % pairs;
                [values[2 * pair], values[2 * pair + 1]]
            })
            .collect();
        for op in [UnaryOp::Neg, UnaryOp::Conj, UnaryOp::Sign] {
            let what = format!("complex unary {op:?} len {len}");
            let got = complex_unary(&mut device, op, DType::C32, &repeated, &what);
            let want: Vec<f32> = repeated
                .chunks_exact(2)
                .flat_map(|pair| complex_reference(op, pair[0], pair[1]))
                .collect();
            assert_within_ulps(&got, &want, tolerance, &what);
        }
        // `abs` writes f32 from a Complex32 operand and must not overflow at 1e30.
        let what = format!("abs Complex32 to f32 len {len}");
        let got = complex_unary(&mut device, UnaryOp::Abs, DType::F32, &repeated, &what);
        let want: Vec<f32> = repeated
            .chunks_exact(2)
            .map(|pair| complex_magnitude(pair[0], pair[1]))
            .collect();
        assert_within_ulps(&got, &want, 8, &what);
    }

    // The fixture itself, index by index, so a wrong helper cannot agree with a wrong
    // reference.
    let got = complex_unary(
        &mut device,
        UnaryOp::Abs,
        DType::F32,
        &values,
        "abs fixture",
    );
    assert_eq!(got[0], 5.0f32.sqrt(), "abs(1 + 2i) is sqrt(5)");
    assert_eq!(got[1], 0.0, "abs(0) is zero");
    assert_eq!(got[2], 0.0, "abs(-0 - 0i) is zero");
    assert_eq!(got[3], 5.0, "abs(-3 + 4i) is 5");
    assert!(
        got[4].is_finite() && got[4] > 0.0,
        "abs of 1e-30 + 1e-30i must not underflow to zero, got {}",
        got[4]
    );
    assert!(
        got[5].is_finite() && got[5] > 1e30,
        "abs of 1e30 + 1e30i must not overflow, got {}",
        got[5]
    );
    // `fmax(NaN, 0)` is 0 under fast math, so a scaled magnitude that takes the zero
    // shortcut on that pair returns 0 where `hypotf` returns NaN.
    assert!(
        got[6].is_nan(),
        "abs of NaN + 0i must be NaN, got {}",
        got[6]
    );
    assert!(
        got[7].is_nan(),
        "abs of 0 + NaNi must be NaN, got {}",
        got[7]
    );
    assert_eq!(
        got[8], inf,
        "abs of inf + NaNi is inf, as the C standard specifies for hypot"
    );
}

/// The Complex32 unary reference: two f32 components, as `cuda/src/elementwise.cu` and
/// `unary_complex` in the pinned baseline define them.
fn complex_reference(op: UnaryOp, re: f32, im: f32) -> [f32; 2] {
    match op {
        UnaryOp::Neg => [-re, -im],
        UnaryOp::Conj => [re, -im],
        UnaryOp::Sign => {
            if re == 0.0 && im == 0.0 {
                [0.0, 0.0]
            } else {
                let magnitude = complex_magnitude(re, im);
                [re / magnitude, im / magnitude]
            }
        }
        other => panic!("{other:?} has no Complex32 kernel"),
    }
}

/// The robust Complex32 magnitude, the way `hypotf` computes it: inf beats NaN, NaN
/// propagates and (0, 0) is +0.
fn complex_magnitude(re: f32, im: f32) -> f32 {
    let (re, im) = (f64::from(re), f64::from(im));
    re.hypot(im) as f32
}

#[test]
fn an_empty_plan_launches_nothing() {
    let device = Device::open();
    for dtype in metal::DTYPES {
        let plan = metal::plan_fill_zero(dtype, 0).unwrap();
        assert!(plan.dispatch.is_empty());
        assert_eq!(plan.bytes, 0);
    }
    assert_eq!(device.runtime.stats().launches, 0);
}
