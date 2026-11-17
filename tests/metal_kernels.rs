//! The Metal entries on a real Apple GPU. Needs `--features gpu-tests-metal`.
//!
//! Every test compares against a CPU reference over the whole result and fills the
//! destination with a sentinel past `len`, so a kernel that misses elements or
//! writes past the end fails here. The lengths include 4096 and 100_003 elements,
//! which no single threadgroup covers: that is the only size range in which a
//! kernel that never leaves the first thread fails.

use std::collections::HashMap;

use t4a_cuda_kernels::metal::{self, Dispatch, SYMBOLS};
use t4a_cuda_kernels::{
    BinaryOp, CAxisValues, CompareOp, DType, Entry, LayoutRef, ReduceOp, ReduceStrategy,
    ScalarSide, UnaryOp,
};
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

// --- slices 2-4: movement, diagonals, convert, element-wise ops, reductions ---

/// Element index of every logical position of a layout, axis 0 fastest.
fn strided_indices(extents: &[i64], strides: &[i64], offset: i64) -> Vec<usize> {
    let count: i64 = extents.iter().product();
    let mut indices = Vec::with_capacity(count as usize);
    for flat in 0..count {
        let mut rest = flat;
        let mut index = offset;
        for (extent, stride) in extents.iter().zip(strides) {
            let coordinate = rest % extent;
            rest /= extent;
            index += coordinate * stride;
        }
        indices.push(index as usize);
    }
    indices
}

/// A distinct, finite pattern for word `word` of element `index`.
fn element_word(index: usize, word: usize) -> u32 {
    0x3f00_0000 | ((index as u32) << 4) | word as u32
}

/// The whole backing array as words, which is what every structured test compares.
fn backing_words(buffer: &Buffer, words: usize) -> Vec<u32> {
    // SAFETY: every submission that could touch `buffer` has completed, and the
    // buffer holds at least `words` words.
    unsafe { std::slice::from_raw_parts(buffer.host_ptr().cast::<u32>(), words).to_vec() }
}

/// Expected words of a destination that starts sentinel-filled: element
/// `indices[j]` holds the source words of logical position `j`.
fn expected_written(
    span: usize,
    words_per_element: usize,
    indices: &[usize],
    source: &[u32],
) -> Vec<u32> {
    let mut want = vec![SENTINEL; span * words_per_element];
    for (position, &index) in indices.iter().enumerate() {
        for word in 0..words_per_element {
            want[index * words_per_element + word] = source[position * words_per_element + word];
        }
    }
    want
}

/// `written` extended by the `SLACK` sentinel-filled elements every test destination
/// allocates, so a comparison covers the whole allocation: a write past the domain
/// lands on the sentinel and fails, and a threadgroup that stops early leaves one.
fn with_sentinel_slack(written: Vec<u32>, words_per_element: usize) -> Vec<u32> {
    let mut words = written;
    words.extend(std::iter::repeat_n(SENTINEL, SLACK * words_per_element));
    words
}

/// The source words of a strided layout in logical order.
fn ordered_source(source_words: &[u32], indices: &[usize], words_per_element: usize) -> Vec<u32> {
    indices
        .iter()
        .flat_map(|&index| {
            (0..words_per_element).map(move |word| source_words[index * words_per_element + word])
        })
        .collect()
}

/// A source buffer holding `span` elements of distinct patterns.
fn patterned_source(device: &mut Device, dtype: DType, span: usize) -> Buffer {
    let words_per_element = dtype.size() / 4;
    let words: Vec<u32> = (0..span)
        .flat_map(|index| (0..words_per_element).map(move |word| element_word(index, word)))
        .collect();
    device.host_buffer(dtype, &words)
}

#[test]
fn movement_entries_follow_their_layouts() {
    let mut device = Device::open();
    let cases = [
        ("offset", [4i64, 3], [1i64, 4], 2i64),
        ("negative strides", [3, 3], [-1, -3], 8),
        ("column-major", [3, 4], [4, 1], 0),
    ];
    for dtype in metal::DTYPES {
        let words_per_element = dtype.size() / 4;
        let dtype_tag = dtype.tag();
        for (name, extents, strides, offset) in cases {
            let indices = strided_indices(&extents, &strides, offset);
            let position_count = indices.len();
            let span = indices.iter().copied().max().unwrap() + 1;
            let source = patterned_source(&mut device, dtype, span);
            let source_words: Vec<u32> = (0..span)
                .flat_map(|index| (0..words_per_element).map(move |word| element_word(index, word)))
                .collect();
            let layout = LayoutRef::new(&extents, &strides, offset).unwrap();
            let mirror = metal::layout_mirror(layout).unwrap();
            let mirror_bytes = metal::layout_bytes(&mirror);
            let span_len = span as i64;
            let count_len = position_count as i64;

            // materialize_strided: a compact destination of the logical positions.
            let out = device.sentinel_buffer(dtype, position_count);
            let plan = metal::plan_materialize_strided(dtype, layout, span_len).unwrap();
            assert_eq!(plan.bytes, (position_count * dtype.size()) as u64);
            device.dispatch(
                "t4a_metal_materialize_strided",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &source,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::Bytes {
                        bytes: &mirror_bytes,
                    },
                    Arg::pod(&count_len),
                ],
            );
            let ordered = ordered_source(&source_words, &indices, words_per_element);
            let want = expected_written(
                position_count,
                words_per_element,
                &(0..position_count).collect::<Vec<_>>(),
                &ordered,
            );
            let want = with_sentinel_slack(want, words_per_element);
            assert_eq!(
                backing_words(&out, want.len()),
                want,
                "materialize_strided {dtype:?} {name}"
            );

            // fill_zero_view: every position the layout reaches becomes zero, and
            // nothing else is touched.
            let out = device.sentinel_buffer(dtype, span);
            let plan = metal::plan_fill_zero_view(dtype, layout, span_len).unwrap();
            device.dispatch(
                "t4a_metal_fill_zero_view",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::Bytes {
                        bytes: &mirror_bytes,
                    },
                    Arg::pod(&count_len),
                ],
            );
            let want = expected_written(
                span,
                words_per_element,
                &indices,
                &vec![0u32; position_count * words_per_element],
            );
            let want = with_sentinel_slack(want, words_per_element);
            assert_eq!(
                backing_words(&out, want.len()),
                want,
                "fill_zero_view {dtype:?} {name}"
            );

            // contiguous_to_view: the compact source lands on the layout.
            let out = device.sentinel_buffer(dtype, span);
            let plan = metal::plan_contiguous_to_view(dtype, layout, span_len).unwrap();
            device.dispatch(
                "t4a_metal_contiguous_to_view",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &source,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::Bytes {
                        bytes: &mirror_bytes,
                    },
                    Arg::pod(&count_len),
                ],
            );
            let want = expected_written(
                span,
                words_per_element,
                &indices,
                &source_words[..position_count * words_per_element],
            );
            let want = with_sentinel_slack(want, words_per_element);
            assert_eq!(
                backing_words(&out, want.len()),
                want,
                "contiguous_to_view {dtype:?} {name}"
            );

            // strided_to_strided: two layouts of the same extents.
            let dst_extents = extents;
            // A destination needs non-negative reachable indices; the absolute
            // strides keep the case's shape with an in-allocation span.
            let dst_strides = [strides[1].abs(), strides[0].abs()];
            let dst_indices = strided_indices(&dst_extents, &dst_strides, 0);
            let dst_span = dst_indices.iter().copied().max().unwrap() + 1;
            let out = device.sentinel_buffer(dtype, dst_span);
            let dst_layout = LayoutRef::new(&dst_extents, &dst_strides, 0).unwrap();
            let dst_mirror = metal::layout_mirror(dst_layout).unwrap();
            let dst_bytes = metal::layout_bytes(&dst_mirror);
            let plan = metal::plan_strided_to_strided(
                dtype,
                layout,
                span_len,
                dst_layout,
                dst_span as i64,
            )
            .unwrap();
            device.dispatch(
                "t4a_metal_strided_to_strided",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &source,
                        offset: 0,
                    },
                    Arg::pod(&dtype_tag),
                    Arg::Bytes {
                        bytes: &mirror_bytes,
                    },
                    Arg::Bytes { bytes: &dst_bytes },
                    Arg::pod(&count_len),
                ],
            );
            let want = expected_written(dst_span, words_per_element, &dst_indices, &ordered);
            let want = with_sentinel_slack(want, words_per_element);
            assert_eq!(
                backing_words(&out, want.len()),
                want,
                "strided_to_strided {dtype:?} {name}"
            );
        }
    }

    // A long domain, so the tail and a second threadgroup are exercised too.
    let len = 100_003usize;
    let extents = [len as i64];
    let strides = [1i64];
    let layout = LayoutRef::new(&extents, &strides, 1).unwrap();
    let source = patterned_source(&mut device, DType::F32, len + 1);
    let out = device.sentinel_buffer(DType::F32, len);
    let mirror = metal::layout_mirror(layout).unwrap();
    let mirror_bytes = metal::layout_bytes(&mirror);
    let (tag, length) = (DType::F32.tag(), len as i64);
    let plan = metal::plan_materialize_strided(DType::F32, layout, len as i64 + 1).unwrap();
    device.dispatch(
        "t4a_metal_materialize_strided",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &source,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes {
                bytes: &mirror_bytes,
            },
            Arg::pod(&length),
        ],
    );
    let want: Vec<u32> = (1..=len).map(|index| element_word(index, 0)).collect();
    assert_eq!(backing_words(&out, len), want, "a 100_003-element gather");
    assert_tail_is_sentinel(&out, DType::F32, len, "materialize_strided len 100_003");
}

#[test]
fn broadcast_in_dim_follows_its_dimension_map() {
    let mut device = Device::open();
    // out = [3, 4], in = [3, 1]: axis 0 keeps its extent, axis 1 broadcasts.
    let out_extents = [3i64, 4];
    let out_strides = [1i64, 3];
    let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let in_extents = [3i64, 1];
    let in_strides = [1i64, 3];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let input = device.host_buffer(DType::F32, &[10.0f32, 20.0, 30.0]);
    let out = device.sentinel_buffer(DType::F32, 12);
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let dims = CAxisValues::new(&[0, 1]).unwrap();
    let dims_bytes = metal::axis_values_bytes(&dims);
    let (tag, out_len) = (DType::F32.tag(), 12i64);
    let plan =
        metal::plan_broadcast_in_dim(DType::F32, out_layout, 12, in_layout, 3, &[0, 1]).unwrap();
    assert_eq!(plan.bytes, 48);
    device.dispatch(
        "t4a_metal_broadcast_in_dim",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &input,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::Bytes { bytes: &dims_bytes },
            Arg::pod(&out_len),
        ],
    );
    let want: Vec<u32> = [10.0f32, 20.0, 30.0]
        .iter()
        .cycle()
        .take(12)
        .map(|value| value.to_bits())
        .collect();
    let want = with_sentinel_slack(want, 1);
    assert_eq!(
        backing_words(&out, want.len()),
        want,
        "broadcast_in_dim 3x4"
    );

    // A rank-0 input: every destination element is the scalar.
    let scalar = device.host_buffer(DType::F32, &[7.5f32]);
    let out = device.sentinel_buffer(DType::F32, 5);
    let out_layout = LayoutRef::new(&[5], &[1], 0).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let in_layout = LayoutRef::new(&[], &[], 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let dims = CAxisValues::new(&[]).unwrap();
    let dims_bytes = metal::axis_values_bytes(&dims);
    let (tag, out_len) = (DType::F32.tag(), 5i64);
    let plan = metal::plan_broadcast_in_dim(DType::F32, out_layout, 5, in_layout, 1, &[]).unwrap();
    device.dispatch(
        "t4a_metal_broadcast_in_dim",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &scalar,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::Bytes { bytes: &dims_bytes },
            Arg::pod(&out_len),
        ],
    );
    let want = with_sentinel_slack(vec![7.5f32.to_bits(); 5], 1);
    assert_eq!(backing_words(&out, want.len()), want, "rank-0 broadcast");
}

/// Reference for the triangular mask over a compact `(rows, cols)` matrix per batch.
fn triangular_reference(data: &[f32], rows: i64, cols: i64, k: i64, keep_below: bool) -> Vec<f32> {
    data.iter()
        .enumerate()
        .map(|(flat, &value)| {
            let flat = flat as i64;
            let row = flat % rows;
            let col = (flat / rows) % cols;
            let offset = col - row;
            if if keep_below { offset <= k } else { offset >= k } {
                value
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn diagonal_entries_follow_the_corrected_map() {
    let mut device = Device::open();
    // The rank-3 extraction the baseline defect needed: out[o0, o1] = in[o0, o0, o1].
    let in_extents = [2i64, 3, 2];
    let in_strides = [1i64, 2, 6];
    let data: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let input = device.host_buffer(DType::F32, &data);
    let out_extents = [2i64, 2];
    let out_strides = [1i64, 2];
    let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let out = device.sentinel_buffer(DType::F32, 4);
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let (tag, len, axis_a, axis_b) = (DType::F32.tag(), 4i64, 0i32, 1i32);
    let plan =
        metal::plan_extract_diagonal(DType::F32, in_layout, 12, out_layout, 4, 0, 1).unwrap();
    device.dispatch(
        "t4a_metal_extract_diagonal",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &input,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::pod(&axis_a),
            Arg::pod(&axis_b),
            Arg::pod(&len),
        ],
    );
    let want: Vec<u32> = [0usize, 3, 6, 9]
        .iter()
        .map(|&index| data[index].to_bits())
        .collect();
    let want = with_sentinel_slack(want, 1);
    assert_eq!(
        backing_words(&out, want.len()),
        want,
        "rank-3 extract_diagonal"
    );

    // A strided (column-major) input reaches the same values through its strides.
    let in_strides = [6i64, 2, 1];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let out = device.sentinel_buffer(DType::F32, 4);
    let plan =
        metal::plan_extract_diagonal(DType::F32, in_layout, 12, out_layout, 4, 0, 1).unwrap();
    device.dispatch(
        "t4a_metal_extract_diagonal",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &input,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::pod(&axis_a),
            Arg::pod(&axis_b),
            Arg::pod(&len),
        ],
    );
    let want: Vec<u32> = [0usize, 8, 1, 9]
        .iter()
        .map(|&index| data[index].to_bits())
        .collect();
    let want = with_sentinel_slack(want, 1);
    assert_eq!(
        backing_words(&out, want.len()),
        want,
        "strided extract_diagonal"
    );

    // Embedding writes only the inserted diagonal into a pre-zeroed destination.
    let in_extents = [2i64, 2];
    let in_strides = [1i64, 2];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let source = device.host_buffer(DType::F32, &[1.0f32, 2.0, 3.0, 4.0]);
    let out_extents = [2i64, 2, 2];
    let out_strides = [1i64, 2, 4];
    let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let out = device.sentinel_buffer(DType::F32, 8);
    // Zero first: embed writes only diagonal positions.
    let zero_plan = metal::plan_fill_zero_view(DType::F32, out_layout, 8).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let (tag, len, axis_a, axis_b) = (DType::F32.tag(), 4i64, 0i32, 2i32);
    let zero_len = 8i64;
    device.dispatch(
        "t4a_metal_fill_zero_view",
        zero_plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::pod(&zero_len),
        ],
    );
    let plan =
        metal::plan_embed_diagonal_copy(DType::F32, in_layout, 4, out_layout, 8, 0, 2).unwrap();
    assert_eq!(plan.bytes, 32, "the destination's eight elements");
    device.dispatch(
        "t4a_metal_embed_diagonal_copy",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &source,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::pod(&axis_a),
            Arg::pod(&axis_b),
            Arg::pod(&len),
        ],
    );
    let mut want = vec![0u32; 8];
    // out[o0, o1, o0] = in[o0, o1]: index o0 + 2*o1 + 4*o0
    for (o0, o1) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
        want[o0 + 2 * o1 + 4 * o0] = [1.0f32, 2.0, 3.0, 4.0][o0 + 2 * o1].to_bits();
    }
    let want = with_sentinel_slack(want, 1);
    assert_eq!(backing_words(&out, want.len()), want, "embed_diagonal_copy");
}

#[test]
fn triangular_masks_match_the_reference() {
    let mut device = Device::open();
    let (rows, cols) = (3i64, 4i64);
    let len = (rows * cols) as usize;
    let data: Vec<f32> = (0..len).map(|value| (value as f32) + 0.5).collect();
    let input = device.host_buffer(DType::F32, &data);
    let (tag, length) = (DType::F32.tag(), len as i64);
    for k in [0i64, -1, 1, i64::MIN, i64::MAX] {
        for (symbol, keep_below, plan) in [
            (
                "t4a_metal_tril",
                true,
                metal::plan_tril(DType::F32, rows, cols, length).unwrap(),
            ),
            (
                "t4a_metal_triu",
                false,
                metal::plan_triu(DType::F32, rows, cols, length).unwrap(),
            ),
        ] {
            let out = device.sentinel_buffer(DType::F32, len);
            device.dispatch(
                symbol,
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&tag),
                    Arg::pod(&k),
                    Arg::pod(&rows),
                    Arg::pod(&cols),
                    Arg::pod(&length),
                ],
            );
            let want: Vec<u32> = triangular_reference(&data, rows, cols, k, keep_below)
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let want = with_sentinel_slack(want, 1);
            assert_eq!(backing_words(&out, want.len()), want, "{symbol} at k {k}");
        }
    }

    // The view twins read the input through its own layout.
    let in_extents = [3i64, 4];
    let in_strides = [4i64, 1];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let indices = strided_indices(&in_extents, &in_strides, 0);
    for k in [0i64, -1, 1] {
        for (symbol, keep_below, plan) in [
            (
                "t4a_metal_tril_view",
                true,
                metal::plan_tril_view(DType::F32, length, in_layout, length).unwrap(),
            ),
            (
                "t4a_metal_triu_view",
                false,
                metal::plan_triu_view(DType::F32, length, in_layout, length).unwrap(),
            ),
        ] {
            let out = device.sentinel_buffer(DType::F32, len);
            device.dispatch(
                symbol,
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&tag),
                    Arg::Bytes { bytes: &in_mirror },
                    Arg::pod(&k),
                    Arg::pod(&length),
                ],
            );
            let mask = triangular_reference(&data, rows, cols, k, keep_below);
            let want: Vec<u32> = mask
                .iter()
                .enumerate()
                .map(|(flat, value)| {
                    if value.to_bits() == 0 {
                        0u32
                    } else {
                        data[indices[flat]].to_bits()
                    }
                })
                .collect();
            let want = with_sentinel_slack(want, 1);
            assert_eq!(backing_words(&out, want.len()), want, "{symbol} at k {k}");
        }
    }
}

#[test]
fn convert_pairs_keep_the_component_they_own() {
    let mut device = Device::open();
    let len = 100_003usize;
    let data: Vec<f32> = (0..len)
        .map(|index| match index % 4 {
            0 => -0.0,
            1 => f32::from_bits(0x7fc0_0001),
            2 => 1.0e-40,
            _ => (index as f32) * 0.5,
        })
        .collect();
    let (tag, length) = (DType::F32.tag(), len as i64);

    // f32 -> Complex32: the real part moves bit for bit, the imaginary part is +0.
    let input = device.host_buffer(DType::F32, &data);
    let out = device.sentinel_buffer(DType::C32, len);
    let (dst_tag, src_tag) = (DType::C32.tag(), DType::F32.tag());
    let _ = tag;
    let plan = metal::plan_convert(DType::C32, DType::F32, length).unwrap();
    assert_eq!(plan.bytes, len as u64 * 8);
    device.dispatch(
        "t4a_metal_convert",
        plan.dispatch,
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
            Arg::pod(&length),
        ],
    );
    let mut want = Vec::with_capacity(len * 2);
    for value in &data {
        want.push(value.to_bits());
        want.push(0u32);
    }
    assert_eq!(backing_words(&out, len * 2), want);
    assert_tail_is_sentinel(&out, DType::C32, len, "convert to Complex32");

    // Complex32 -> f32: the real part only.
    let words: Vec<u32> = (0..len)
        .flat_map(|index| [element_word(index, 0), element_word(index, 1)])
        .collect();
    let input = device.host_buffer(DType::C32, &words);
    let out = device.sentinel_buffer(DType::F32, len);
    let (dst_tag, src_tag) = (DType::F32.tag(), DType::C32.tag());
    let plan = metal::plan_convert(DType::F32, DType::C32, length).unwrap();
    device.dispatch(
        "t4a_metal_convert",
        plan.dispatch,
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
            Arg::pod(&length),
        ],
    );
    let want: Vec<u32> = (0..len).map(|index| element_word(index, 0)).collect();
    assert_eq!(backing_words(&out, len), want);
    assert_tail_is_sentinel(&out, DType::F32, len, "convert to f32");
}

/// NaN-propagating maximum, as `t4a_cuda_nan_max` computes it.
fn nan_max(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.max(rhs)
    }
}

/// NaN-propagating minimum, as `t4a_cuda_nan_min` computes it.
fn nan_min(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() {
        lhs
    } else if rhs.is_nan() {
        rhs
    } else {
        lhs.min(rhs)
    }
}

/// The reference clamp: statement order, a NaN bound disables its own comparison.
fn clamp_reference(value: f32, low: f32, high: f32) -> f32 {
    if !value.is_nan() && !low.is_nan() && value < low {
        low
    } else if !value.is_nan() && !high.is_nan() && value > high {
        high
    } else {
        value
    }
}

/// Port of the scaled Baudin-Smith complex division the kernel carries.
fn div_complex_reference(pair: (f32, f32), divisor: (f32, f32)) -> (f32, f32) {
    const MAX: f32 = f32::MAX;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const EPSILON: f32 = f32::EPSILON;
    let (mut a, mut b) = pair;
    let (mut c, mut d) = divisor;
    if c.abs() > MAX || d.abs() > MAX {
        let a_finite = !a.is_nan() && a.abs() <= MAX;
        let b_finite = !b.is_nan() && b.abs() <= MAX;
        if a_finite && b_finite {
            fn unit(value: f32) -> f32 {
                if value == 0.0 {
                    value
                } else if value < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
            return (0.0 * unit(a) * unit(c), -0.0 * unit(b) * unit(d));
        }
        return (f32::NAN, f32::NAN);
    }
    let (abs_a, abs_b, abs_c, abs_d) = (a.abs(), b.abs(), c.abs(), d.abs());
    let ab = abs_a.max(abs_b);
    let cd = abs_c.max(abs_d);
    let half_over = MAX * 0.5;
    let two_under_eps = MIN_POSITIVE * 2.0 / EPSILON;
    let mut scale = 1.0f32;
    if ab >= half_over || ab <= two_under_eps || cd >= half_over || cd <= two_under_eps {
        let big = 2.0 / (EPSILON * EPSILON);
        if ab >= half_over {
            a *= 0.5;
            b *= 0.5;
            scale *= 2.0;
        } else if ab <= two_under_eps {
            a *= big;
            b *= big;
            scale /= big;
        }
        if cd >= half_over {
            c *= 0.5;
            d *= 0.5;
            scale *= 0.5;
        } else if cd <= two_under_eps {
            c *= big;
            d *= big;
            scale *= big;
        }
    }
    fn term(a: f32, b: f32, c: f32, d: f32, r: f32, t: f32) -> f32 {
        if r != 0.0 {
            let br = b * r;
            if br != 0.0 {
                return (a + br) * t;
            }
            return a * t + (b * t) * r;
        }
        (a + d * (b / c)) * t
    }
    if d.abs() <= c.abs() {
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

#[test]
fn binary_entries_match_the_reference() {
    let mut device = Device::open();
    let pairs = [
        (1.5f32, 2.5f32),
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::NAN, f32::NAN),
        (-0.0, 0.0),
        (f32::INFINITY, 1.0),
        (1.0e-40, 2.0),
        (-3.0, 0.5),
    ];
    let len = pairs.len();
    let lhs: Vec<f32> = pairs.iter().map(|(a, _)| *a).collect();
    let rhs: Vec<f32> = pairs.iter().map(|(_, b)| *b).collect();
    let left = device.host_buffer(DType::F32, &lhs);
    let right = device.host_buffer(DType::F32, &rhs);
    let (tag, length, side) = (DType::F32.tag(), len as i64, ScalarSide::Both.tag());

    for op in [BinaryOp::Max, BinaryOp::Min, BinaryOp::Pow, BinaryOp::Rem] {
        let out = device.sentinel_buffer(DType::F32, len);
        let plan = metal::plan_binary(op, DType::F32, ScalarSide::Both, length).unwrap();
        let op_tag = op.tag();
        device.dispatch(
            "t4a_metal_binary",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &left,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &right,
                    offset: 0,
                },
                Arg::pod(&tag),
                Arg::pod(&op_tag),
                Arg::pod(&side),
                Arg::pod(&length),
            ],
        );
        let got: Vec<f32> = unsafe { read_f32(&out, 0, len) }.to_vec();
        for (index, ((a, b), got)) in pairs.iter().zip(&got).enumerate() {
            let want = match op {
                BinaryOp::Max => nan_max(*a, *b),
                BinaryOp::Min => nan_min(*a, *b),
                // `powf` and `fmodf` are the functions the CUDA twin calls; the
                // precise MSL forms are the same IEEE operation, whose special
                // cases (`pow(1, NaN) == 1`) both sides share.
                BinaryOp::Pow => a.powf(*b),
                BinaryOp::Rem => a % b,
                other => panic!("{other:?} is not in this loop"),
            };
            let tolerance = match op {
                BinaryOp::Max | BinaryOp::Min | BinaryOp::Rem => 0,
                _ => 8,
            };
            // The sign of a zero is not the contract: `fmin(-0, +0)` may return
            // either sign, as Rust's own `min` documents.
            if want == 0.0 && *got == 0.0 {
                continue;
            }
            assert_within_ulps(
                &[*got],
                &[want],
                tolerance,
                &format!("{op:?} element {index}"),
            );
        }
    }

    // The scalar forms: one element on the right.
    let factor = device.host_buffer(DType::F32, &[2.0f32]);
    for (op, symbol_op) in [
        (BinaryOp::Mul, BinaryOp::Mul),
        (BinaryOp::Div, BinaryOp::Div),
    ] {
        let out = device.sentinel_buffer(DType::F32, len);
        let plan = metal::plan_binary(op, DType::F32, ScalarSide::Right, length).unwrap();
        let (op_tag, side) = (symbol_op.tag(), ScalarSide::Right.tag());
        device.dispatch(
            "t4a_metal_binary",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &left,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &factor,
                    offset: 0,
                },
                Arg::pod(&tag),
                Arg::pod(&op_tag),
                Arg::pod(&side),
                Arg::pod(&length),
            ],
        );
        let got: Vec<f32> = unsafe { read_f32(&out, 0, len) }.to_vec();
        for (index, ((a, _), got)) in pairs.iter().zip(&got).enumerate() {
            let want = match op {
                BinaryOp::Mul => a * 2.0,
                _ => a / 2.0,
            };
            // A subnormal result is flushed by the runtime's fast-math default,
            // which the slice-1 lane pins; anything else must be the plain
            // operation bit for bit.
            if want != 0.0 && want.abs() < f32::MIN_POSITIVE {
                assert_eq!(
                    got.abs().to_bits(),
                    0,
                    "{op:?} element {index}: a subnormal result must flush to zero"
                );
                continue;
            }
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{op:?} element {index}: a scalar form must be the plain operation"
            );
        }
    }
    let _ = device;
}

#[test]
fn complex_binary_and_division_match_the_reference() {
    let mut device = Device::open();
    let len = 1000usize;
    let magnitudes = [1.0e-30f32, 1.0e30, 0.0, f32::INFINITY, f32::NAN, 3.0, -2.5];
    let mut lhs = Vec::with_capacity(len * 2);
    let mut rhs = Vec::with_capacity(len * 2);
    for index in 0..len {
        let a = magnitudes[index % magnitudes.len()];
        let b = magnitudes[(index / 2) % magnitudes.len()];
        lhs.push(if index % 3 == 0 { 0.0 } else { a });
        lhs.push(if index % 5 == 0 { -0.0 } else { b });
        rhs.push(magnitudes[(index + 1) % magnitudes.len()]);
        rhs.push(magnitudes[(index + 3) % magnitudes.len()]);
    }
    let left = device.host_buffer(DType::C32, &lhs);
    let right = device.host_buffer(DType::C32, &rhs);
    let (tag, length, side) = (DType::C32.tag(), len as i64, ScalarSide::Both.tag());

    let out = device.sentinel_buffer(DType::C32, len);
    let plan =
        metal::plan_binary(BinaryOp::DivComplex, DType::C32, ScalarSide::Both, length).unwrap();
    assert_eq!(plan.bytes, len as u64 * 8);
    let op_tag = BinaryOp::DivComplex.tag();
    device.dispatch(
        "t4a_metal_binary",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &left,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &right,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&op_tag),
            Arg::pod(&side),
            Arg::pod(&length),
        ],
    );
    let got: Vec<f32> = unsafe { read_f32(&out, 0, len * 2) }.to_vec();
    for index in 0..len {
        let a = (lhs[2 * index], lhs[2 * index + 1]);
        let b = (rhs[2 * index], rhs[2 * index + 1]);
        let (want_re, want_im) = div_complex_reference(a, b);
        assert_within_ulps(
            &got[2 * index..2 * index + 2],
            &[want_re, want_im],
            8,
            &format!("DivComplex element {index}"),
        );
    }
    assert_tail_is_sentinel(&out, DType::C32, len, "DivComplex");

    // The scalar Complex32 multiplication reads the one-element operand on the
    // right and keeps the schoolbook form.
    let scalar = device.host_buffer(DType::C32, &[-1.5f32, 0.25]);
    let out = device.sentinel_buffer(DType::C32, len);
    let plan = metal::plan_binary(BinaryOp::Mul, DType::C32, ScalarSide::Right, length).unwrap();
    let (op_tag, side) = (BinaryOp::Mul.tag(), ScalarSide::Right.tag());
    device.dispatch(
        "t4a_metal_binary",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &left,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &scalar,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&op_tag),
            Arg::pod(&side),
            Arg::pod(&length),
        ],
    );
    let got: Vec<f32> = unsafe { read_f32(&out, 0, len * 2) }.to_vec();
    for index in 0..len {
        let (ar, ai) = (lhs[2 * index], lhs[2 * index + 1]);
        let (br, bi) = (-1.5f32, 0.25f32);
        let want = [ar * br - ai * bi, ar * bi + ai * br];
        assert_within_ulps(
            &got[2 * index..2 * index + 2],
            &want,
            2,
            &format!("Complex32 scalar Mul element {index}"),
        );
    }
}

#[test]
fn compare_select_and_clamp_match_the_reference() {
    let mut device = Device::open();
    let pairs = [
        (1.0f32, 2.0f32),
        (2.0, 2.0),
        (f32::NAN, 1.0),
        (-0.0, 0.0),
        (f32::INFINITY, f32::INFINITY),
    ];
    let len = pairs.len();
    let lhs: Vec<f32> = pairs.iter().map(|(a, _)| *a).collect();
    let rhs: Vec<f32> = pairs.iter().map(|(_, b)| *b).collect();
    let left = device.host_buffer(DType::F32, &lhs);
    let right = device.host_buffer(DType::F32, &rhs);
    let (tag, length) = (DType::F32.tag(), len as i64);
    for op in [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ] {
        let out = device.sentinel_buffer(DType::F32, len);
        let plan = metal::plan_compare(DType::F32, length).unwrap();
        assert_eq!(plan.bytes, len as u64, "one byte per position");
        let cmp = op.tag();
        device.dispatch(
            "t4a_metal_compare",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &left,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &right,
                    offset: 0,
                },
                Arg::pod(&tag),
                Arg::pod(&cmp),
                Arg::pod(&length),
            ],
        );
        // The destination is one byte per position: the first `len` bytes are the
        // predicates and the rest of the allocation keeps its sentinel.
        let mut want_bytes = Vec::with_capacity((len + SLACK) * 4);
        for _ in 0..len + SLACK {
            want_bytes.extend_from_slice(&SENTINEL.to_le_bytes());
        }
        for (index, (a, b)) in pairs.iter().enumerate() {
            let want = match op {
                CompareOp::Eq => a == b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            };
            want_bytes[index] = u8::from(want);
        }
        let want: Vec<u32> = want_bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        assert_eq!(
            backing_words(&out, len + SLACK),
            want,
            "{op:?} must write one byte per position and nothing else"
        );
    }

    // Selection: the predicate is one byte per position.
    let on_true = device.host_buffer(DType::F32, &[10.0f32, 20.0, 30.0]);
    let on_false = device.host_buffer(DType::F32, &[-1.0f32, -2.0, -3.0]);
    // One byte per position, read at the element index: predicates 1, 0, 1.
    let predicate = device.host_buffer(DType::F32, &[u32::from_le_bytes([1, 0, 1, 0])]);
    let out = device.sentinel_buffer(DType::F32, 3);
    let (tag, length) = (DType::F32.tag(), 3i64);
    let plan = metal::plan_select(DType::F32, length).unwrap();
    device.dispatch(
        "t4a_metal_select",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &predicate,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &on_true,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &on_false,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&length),
        ],
    );
    assert_eq!(
        backing_words(&out, 3),
        vec![10.0f32.to_bits(), (-2.0f32).to_bits(), 30.0f32.to_bits()]
    );
    assert_tail_is_sentinel(&out, DType::F32, 3, "select over three positions");

    // Clamp: every NaN placement and inverted bounds.
    let triples = [
        (0.5f32, 0.0f32, 1.0f32),
        (-1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (f32::NAN, 0.0, 1.0),
        (0.5, f32::NAN, 1.0),
        (0.5, 0.0, f32::NAN),
        (0.5, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    ];
    let len = triples.len();
    let values: Vec<f32> = triples.iter().map(|(v, _, _)| *v).collect();
    let lows: Vec<f32> = triples.iter().map(|(_, l, _)| *l).collect();
    let highs: Vec<f32> = triples.iter().map(|(_, _, h)| *h).collect();
    let (values, lows, highs) = (
        device.host_buffer(DType::F32, &values),
        device.host_buffer(DType::F32, &lows),
        device.host_buffer(DType::F32, &highs),
    );
    let out = device.sentinel_buffer(DType::F32, len);
    let (tag, length) = (DType::F32.tag(), len as i64);
    let plan = metal::plan_clamp(DType::F32, length).unwrap();
    device.dispatch(
        "t4a_metal_clamp",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &values,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &lows,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &highs,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&length),
        ],
    );
    let got: Vec<f32> = unsafe { read_f32(&out, 0, len) }.to_vec();
    for (index, ((value, low, high), got)) in triples.iter().zip(&got).enumerate() {
        let want = clamp_reference(*value, *low, *high);
        let equal = if want.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == want.to_bits()
        };
        assert!(
            equal,
            "clamp element {index}: ({value}, {low}, {high}) gave {got}, want {want}"
        );
    }
    assert_tail_is_sentinel(&out, DType::F32, len, "clamp over nine positions");
}

/// One row of a `(rows, axis_len)` layout with strides `[1, rows]`.
fn reduce_row(data: &[f32], row: usize, rows: usize, axis_len: usize) -> Vec<f32> {
    (0..axis_len)
        .map(|column| data[row + column * rows])
        .collect()
}

/// The serial fold the unit strategy must match bit for bit.
fn reduce_serial(op: ReduceOp, values: &[f32]) -> f32 {
    let mut acc = match op {
        ReduceOp::SumSquares => values[0] * values[0],
        _ => values[0],
    };
    for &value in &values[1..] {
        acc = match op {
            ReduceOp::Sum => acc + value,
            ReduceOp::Prod => acc * value,
            ReduceOp::SumSquares => acc + value * value,
            ReduceOp::Max => nan_max(acc, value),
            ReduceOp::Min => nan_min(acc, value),
        };
    }
    acc
}

#[test]
fn reductions_match_their_references() {
    let mut device = Device::open();
    let rows = 4i64;
    let out_extents = [rows, 1];
    let out_strides = [1i64, rows];
    let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let tag = DType::F32.tag();
    let axis = 1i32;

    for axis_len in [1i64, 2, 255, 256] {
        let in_extents = [rows, axis_len];
        let in_strides = [1i64, rows];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        let count = (rows * axis_len) as usize;
        let data: Vec<f32> = (0..count)
            .map(|index| (((index % 11) as f32) * 0.25) - 1.0)
            .collect();
        let input = device.host_buffer(DType::F32, &data);
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::SumSquares] {
            let out = device.sentinel_buffer(DType::F32, rows as usize);
            let plan = metal::plan_reduce(
                DType::F32,
                op,
                ReduceStrategy::Unit,
                out_layout,
                rows,
                in_layout,
                count as i64,
                1,
            )
            .unwrap();
            assert_eq!(plan.bytes, rows as u64 * 4);
            let (op_tag, strategy) = (op.tag(), ReduceStrategy::Unit.tag());
            device.dispatch(
                "t4a_metal_reduce",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&tag),
                    Arg::pod(&op_tag),
                    Arg::pod(&strategy),
                    Arg::Bytes { bytes: &out_mirror },
                    Arg::Bytes { bytes: &in_mirror },
                    Arg::pod(&axis),
                    Arg::pod(&rows),
                ],
            );
            let got: Vec<f32> = unsafe { read_f32(&out, 0, rows as usize) }.to_vec();
            assert_tail_is_sentinel(
                &out,
                DType::F32,
                rows as usize,
                &format!("{op:?} unit of axis {axis_len}"),
            );
            for (row, got) in got.iter().enumerate() {
                let values = reduce_row(&data, row, rows as usize, axis_len as usize);
                let want = reduce_serial(op, &values);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{op:?} unit row {row} of axis {axis_len} must be the serial fold"
                );
            }
        }
    }

    // The plane strategy: one threadgroup per output, tolerance against the exact
    // sum, because the threadgroup tree is not the serial order.
    for axis_len in [256i64, 257, 1024] {
        let in_extents = [rows, axis_len];
        let in_strides = [1i64, rows];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        for op in [
            ReduceOp::Sum,
            ReduceOp::Prod,
            ReduceOp::SumSquares,
            ReduceOp::Max,
            ReduceOp::Min,
        ] {
            let count = (rows * axis_len) as usize;
            let data: Vec<f32> = (0..count)
                .map(|index| {
                    if op == ReduceOp::Prod {
                        // ±1 along the axis: nonzero, exactly representable, so the fold
                        // order cannot change the result — a product that always returns
                        // zero fails here.
                        if (index / rows as usize) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        (((index % 7) as f32) * 0.125) - 0.5
                    }
                })
                .collect();
            let input = device.host_buffer(DType::F32, &data);
            let out = device.sentinel_buffer(DType::F32, rows as usize);
            let plan = metal::plan_reduce(
                DType::F32,
                op,
                ReduceStrategy::Plane,
                out_layout,
                rows,
                in_layout,
                count as i64,
                1,
            )
            .unwrap();
            assert_eq!(
                plan.dispatch.threadgroups, rows as u32,
                "one threadgroup per output"
            );
            let (op_tag, strategy) = (op.tag(), ReduceStrategy::Plane.tag());
            device.dispatch(
                "t4a_metal_reduce",
                plan.dispatch,
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &input,
                        offset: 0,
                    },
                    Arg::pod(&tag),
                    Arg::pod(&op_tag),
                    Arg::pod(&strategy),
                    Arg::Bytes { bytes: &out_mirror },
                    Arg::Bytes { bytes: &in_mirror },
                    Arg::pod(&axis),
                    Arg::pod(&rows),
                ],
            );
            let got: Vec<f32> = unsafe { read_f32(&out, 0, rows as usize) }.to_vec();
            assert_tail_is_sentinel(
                &out,
                DType::F32,
                rows as usize,
                &format!("{op:?} plane of axis {axis_len}"),
            );
            for (row, got) in got.iter().enumerate() {
                let values = reduce_row(&data, row, rows as usize, axis_len as usize);
                let exact: f64 = match op {
                    ReduceOp::Sum => values.iter().map(|value| f64::from(*value)).sum(),
                    ReduceOp::SumSquares => values
                        .iter()
                        .map(|value| f64::from(*value) * f64::from(*value))
                        .sum(),
                    ReduceOp::Prod => values.iter().map(|value| f64::from(*value)).product(),
                    ReduceOp::Max => {
                        f64::from(values.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    }
                    ReduceOp::Min => {
                        f64::from(values.iter().copied().fold(f32::INFINITY, f32::min))
                    }
                };
                if matches!(op, ReduceOp::Max | ReduceOp::Min) {
                    assert_eq!(
                        got.to_bits(),
                        (exact as f32).to_bits(),
                        "{op:?} plane row {row} must be exact"
                    );
                    continue;
                }
                let relative = ((f64::from(*got) - exact) / exact.max(1.0)).abs();
                assert!(
                    relative <= 1e-5,
                    "{op:?} plane row {row} of axis {axis_len}: {got} against the exact {exact} \
                     (relative {relative})"
                );
            }
        }
    }

    // Non-finite inputs decide the fold: a NaN propagates, a zero annihilates a
    // product, an infinity survives a sum.
    let in_extents = [3i64, 256];
    let in_strides = [1i64, 3];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    // Row r occupies indices `r + 3 * column`.
    let mut data = vec![1.0f32; 3 * 256];
    data[5 * 3] = f32::NAN;
    data[1 + 7 * 3] = 0.0;
    data[2 + 9 * 3] = f32::INFINITY;
    let input = device.host_buffer(DType::F32, &data);
    for (op, strategy, row, want) in [
        (ReduceOp::Sum, ReduceStrategy::Plane, 0usize, "nan"),
        (ReduceOp::Max, ReduceStrategy::Plane, 0, "nan"),
        (ReduceOp::Min, ReduceStrategy::Plane, 0, "nan"),
        (ReduceOp::Prod, ReduceStrategy::Plane, 1, "zero"),
        (ReduceOp::Sum, ReduceStrategy::Plane, 2, "inf"),
    ] {
        let out = device.sentinel_buffer(DType::F32, 3);
        let out_extents = [3i64, 1];
        let out_strides = [1i64, 3];
        let out_layout3 = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let out_mirror3 = metal::layout_bytes(&metal::layout_mirror(out_layout3).unwrap());
        let plan = metal::plan_reduce(DType::F32, op, strategy, out_layout3, 3, in_layout, 768, 1)
            .unwrap();
        let (op_tag, strategy_tag) = (op.tag(), strategy.tag());
        let out_len = 3i64;
        device.dispatch(
            "t4a_metal_reduce",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &input,
                    offset: 0,
                },
                Arg::pod(&tag),
                Arg::pod(&op_tag),
                Arg::pod(&strategy_tag),
                Arg::Bytes {
                    bytes: &out_mirror3,
                },
                Arg::Bytes { bytes: &in_mirror },
                Arg::pod(&axis),
                Arg::pod(&out_len),
            ],
        );
        let got: Vec<f32> = unsafe { read_f32(&out, 0, 3) }.to_vec();
        assert_tail_is_sentinel(&out, DType::F32, 3, &format!("{op:?} of a non-finite row"));
        match want {
            "nan" => assert!(got[row].is_nan(), "{op:?} must propagate the NaN"),
            "zero" => assert_eq!(
                got[row].to_bits(),
                0.0f32.to_bits(),
                "a zero annihilates the product"
            ),
            "inf" => assert!(
                got[row].is_infinite() && got[row] > 0.0,
                "the infinity survives"
            ),
            other => panic!("unknown expectation {other}"),
        }
    }

    // Complex32 sum and product. The product uses unit-circle factors, so it is
    // exact, bounded and nonzero — unlike a fixture with a zero in it, which a
    // product that always returns zero would also pass.
    let in_extents = [2i64, 256];
    let in_strides = [1i64, 2];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let out_extents = [2i64, 1];
    let out_layout = LayoutRef::new(&out_extents, &[1i64, 2], 0).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    for op in [ReduceOp::Sum, ReduceOp::Prod] {
        // Row `r` occupies element `r + 2 * column` of a `(2, 256)` layout.
        // The input allocation is 512 Complex32 elements of a `(2, 256)` layout, so
        // element `e` belongs to row `e % 2`.
        let parts: Vec<f32> = (0..512)
            .flat_map(|element| {
                if op == ReduceOp::Prod {
                    // Column 0 of each row is `i` and the rest are one, so the product
                    // is exactly `i`: bounded, exact, and not a real number.
                    if element < 2 {
                        [0.0, 1.0]
                    } else {
                        [1.0, 0.0]
                    }
                } else {
                    [((element % 5) as f32) * 0.5, ((element % 3) as f32) * 0.25]
                }
            })
            .collect();
        let input = device.host_buffer(DType::C32, &parts);
        let out = device.sentinel_buffer(DType::C32, 2);
        let plan = metal::plan_reduce(
            DType::C32,
            op,
            ReduceStrategy::Plane,
            out_layout,
            2,
            in_layout,
            512,
            1,
        )
        .unwrap();
        let (tag, op_tag, strategy_tag) = (DType::C32.tag(), op.tag(), ReduceStrategy::Plane.tag());
        let out_len = 2i64;
        device.dispatch(
            "t4a_metal_reduce",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &input,
                    offset: 0,
                },
                Arg::pod(&tag),
                Arg::pod(&op_tag),
                Arg::pod(&strategy_tag),
                Arg::Bytes { bytes: &out_mirror },
                Arg::Bytes { bytes: &in_mirror },
                Arg::pod(&axis),
                Arg::pod(&out_len),
            ],
        );
        let got: Vec<f32> = unsafe { read_f32(&out, 0, 2 * 2) }.to_vec();
        assert_tail_is_sentinel(&out, DType::C32, 2, "reduce Complex32");
        for row in 0..2usize {
            let values: Vec<(f32, f32)> = (0..256)
                .map(|column| {
                    let element = row + 2 * column;
                    (parts[2 * element], parts[2 * element + 1])
                })
                .collect();
            let mut acc = values[0];
            for value in &values[1..] {
                acc = if op == ReduceOp::Prod {
                    (
                        acc.0 * value.0 - acc.1 * value.1,
                        acc.0 * value.1 + acc.1 * value.0,
                    )
                } else {
                    (acc.0 + value.0, acc.1 + value.1)
                };
            }
            for (component, want) in [(0usize, acc.0), (1, acc.1)] {
                let got = got[2 * row + component];
                let relative = ((got - want) / want.abs().max(1.0)).abs();
                assert!(
                    relative <= 1e-5,
                    "{op:?} row {row} component {component}: {got} against {want}"
                );
            }
        }
    }

    // A long axis whose values sum exactly, so the tree and the serial order agree.
    let len = 100_003i64;
    let long_extents = [1i64, len];
    let long_strides = [1i64, 1];
    let in_layout = LayoutRef::new(&long_extents, &long_strides, 0).unwrap();
    let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
    let out_extents = [1i64, 1];
    let out_layout = LayoutRef::new(&out_extents, &long_strides, 0).unwrap();
    let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
    let data = vec![0.5f32; len as usize];
    let input = device.host_buffer(DType::F32, &data);
    let out = device.sentinel_buffer(DType::F32, 1);
    let plan = metal::plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        out_layout,
        1,
        in_layout,
        len,
        1,
    )
    .unwrap();
    assert_eq!(plan.dispatch.threadgroups, 1);
    let (op_tag, strategy_tag, out_len) = (ReduceOp::Sum.tag(), ReduceStrategy::Plane.tag(), 1i64);
    device.dispatch(
        "t4a_metal_reduce",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &input,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&op_tag),
            Arg::pod(&strategy_tag),
            Arg::Bytes { bytes: &out_mirror },
            Arg::Bytes { bytes: &in_mirror },
            Arg::pod(&axis),
            Arg::pod(&out_len),
        ],
    );
    let got: f32 = unsafe { read_f32(&out, 0, 1) }[0];
    assert_eq!(
        got.to_bits(),
        (0.5 * len as f32).to_bits(),
        "100_003 halves sum exactly"
    );
    assert_tail_is_sentinel(&out, DType::F32, 1, "the 100_003-element sum");
}

/// 600 positions is three threadgroups of 256, the last one two-thirds empty: a
/// kernel that launches one threadgroup, or that stops at the first partial one,
/// fails here.
#[test]
fn structural_entries_cross_a_threadgroup() {
    let mut device = Device::open();
    let len = 600usize;
    let dim = len as i64;

    for dtype in metal::DTYPES {
        let words = dtype.size() / 4;
        let dtype_tag = dtype.tag();
        let source = patterned_source(&mut device, dtype, len);
        let source_words: Vec<u32> = (0..len)
            .flat_map(|index| (0..words).map(move |word| element_word(index, word)))
            .collect();

        // fill_zero_view over a compact layout.
        let extents = [dim];
        let strides = [1i64];
        let layout = LayoutRef::new(&extents, &strides, 0).unwrap();
        let mirror = metal::layout_bytes(&metal::layout_mirror(layout).unwrap());
        let out = device.sentinel_buffer(dtype, len);
        let plan = metal::plan_fill_zero_view(dtype, layout, dim).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
        assert_eq!(plan.bytes, (len * dtype.size()) as u64);
        device.dispatch(
            "t4a_metal_fill_zero_view",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &mirror },
                Arg::pod(&dim),
            ],
        );
        let want = with_sentinel_slack(vec![0u32; len * words], words);
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "fill_zero_view {dtype:?} over three threadgroups"
        );

        // contiguous_to_view into a column-major destination: the same span as a
        // compact source, so the two disagree in every position that is not on the
        // diagonal.
        let dst_extents = [24i64, 25];
        let dst_strides = [25i64, 1];
        let dst = LayoutRef::new(&dst_extents, &dst_strides, 0).unwrap();
        let dst_indices = strided_indices(&dst_extents, &dst_strides, 0);
        let dst_mirror = metal::layout_bytes(&metal::layout_mirror(dst).unwrap());
        let out = device.sentinel_buffer(dtype, len);
        let plan = metal::plan_contiguous_to_view(dtype, dst, dim).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
        device.dispatch(
            "t4a_metal_contiguous_to_view",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &source,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &dst_mirror },
                Arg::pod(&dim),
            ],
        );
        let want = with_sentinel_slack(
            expected_written(len, words, &dst_indices, &source_words),
            words,
        );
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "contiguous_to_view {dtype:?} over three threadgroups"
        );

        // materialize_strided from a column-major source into a compact one.
        let src_extents = [24i64, 25];
        let src_strides = [25i64, 1];
        let src = LayoutRef::new(&src_extents, &src_strides, 0).unwrap();
        let src_mirror = metal::layout_bytes(&metal::layout_mirror(src).unwrap());
        let src_indices = strided_indices(&src_extents, &src_strides, 0);
        let out = device.sentinel_buffer(dtype, len);
        let plan = metal::plan_materialize_strided(dtype, src, dim).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
        device.dispatch(
            "t4a_metal_materialize_strided",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &source,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &src_mirror },
                Arg::pod(&dim),
            ],
        );
        let ordered = ordered_source(&source_words, &src_indices, words);
        let want = with_sentinel_slack(
            expected_written(len, words, &(0..len).collect::<Vec<_>>(), &ordered),
            words,
        );
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "materialize_strided {dtype:?} over three threadgroups"
        );

        // strided_to_strided into the same destination.
        let out = device.sentinel_buffer(dtype, len);
        let plan = metal::plan_strided_to_strided(dtype, src, dim, dst, dim).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
        device.dispatch(
            "t4a_metal_strided_to_strided",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &source,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &src_mirror },
                Arg::Bytes { bytes: &dst_mirror },
                Arg::pod(&dim),
            ],
        );
        let want = with_sentinel_slack(expected_written(len, words, &dst_indices, &ordered), words);
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "strided_to_strided {dtype:?} over three threadgroups"
        );

        // broadcast_in_dim: [3, 300] from [3, 1], 900 positions.
        let out_extents = [3i64, 300];
        let out_strides = [1i64, 3];
        let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
        let in_extents = [3i64, 1];
        let in_strides = [1i64, 3];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        let input = device.host_buffer(dtype, &source_words[..3 * words]);
        let out = device.sentinel_buffer(dtype, 900);
        let dims = metal::axis_values_bytes(&CAxisValues::new(&[0, 1]).unwrap());
        let plan =
            metal::plan_broadcast_in_dim(dtype, out_layout, 900, in_layout, 3, &[0, 1]).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 4, "900 positions");
        let out_len = 900i64;
        device.dispatch(
            "t4a_metal_broadcast_in_dim",
            plan.dispatch,
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
                Arg::Bytes { bytes: &out_mirror },
                Arg::Bytes { bytes: &in_mirror },
                Arg::Bytes { bytes: &dims },
                Arg::pod(&out_len),
            ],
        );
        let want: Vec<u32> = (0..900)
            .flat_map(|position| {
                let row = position % 3;
                (0..words)
                    .map(move |word| element_word(row, word))
                    .collect::<Vec<_>>()
            })
            .collect();
        let want = with_sentinel_slack(want, words);
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "broadcast_in_dim {dtype:?} over four threadgroups"
        );

        // extract_diagonal from a 300x300 matrix: 300 positions, two threadgroups.
        let in_extents = [300i64, 300];
        let in_strides = [1i64, 300];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        let matrix: Vec<u32> = (0..90_000)
            .flat_map(|index| (0..words).map(move |word| element_word(index + word, word)))
            .collect();
        let matrix = device.host_buffer(dtype, &matrix);
        let out_extents = [300i64];
        let out_strides = [1i64];
        let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
        let out = device.sentinel_buffer(dtype, 300);
        let plan =
            metal::plan_extract_diagonal(dtype, in_layout, 90_000, out_layout, 300, 0, 1).unwrap();
        assert_eq!(plan.dispatch.threadgroups, 2, "300 diagonal positions");
        let (axis_a, axis_b, count) = (0i32, 1i32, 300i64);
        device.dispatch(
            "t4a_metal_extract_diagonal",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &matrix,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &out_mirror },
                Arg::Bytes { bytes: &in_mirror },
                Arg::pod(&axis_a),
                Arg::pod(&axis_b),
                Arg::pod(&count),
            ],
        );
        let want: Vec<u32> = (0..300)
            .flat_map(|diagonal| {
                let index = diagonal + 300 * diagonal;
                (0..words).map(move |word| element_word(index + word, word))
            })
            .collect();
        let want = with_sentinel_slack(want, words);
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "extract_diagonal {dtype:?} over two threadgroups"
        );
    }

    // The triangle masks over a 300x2 matrix: 600 positions, both dtypes.
    let rows = 300i64;
    let cols = 2i64;
    let real: Vec<f32> = (0..len).map(|value| (value as f32) + 0.5).collect();
    for dtype in metal::DTYPES {
        let words = dtype.size() / 4;
        let dtype_tag = dtype.tag();
        let data: Vec<f32> = real
            .iter()
            .flat_map(|value| (0..words).map(move |word| value + word as f32 * 0.25))
            .collect();
        let input = device.host_buffer(dtype, &data);
        let (tag, length) = (dtype_tag, dim);
        for k in [0i64, -1, i64::MAX] {
            for (symbol, keep_below) in [("t4a_metal_tril", true), ("t4a_metal_triu", false)] {
                let plan = if keep_below {
                    metal::plan_tril(dtype, rows, cols, length).unwrap()
                } else {
                    metal::plan_triu(dtype, rows, cols, length).unwrap()
                };
                assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
                let out = device.sentinel_buffer(dtype, len);
                device.dispatch(
                    symbol,
                    plan.dispatch,
                    &[
                        Arg::Buffer {
                            buffer: &out,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &input,
                            offset: 0,
                        },
                        Arg::pod(&tag),
                        Arg::pod(&k),
                        Arg::pod(&rows),
                        Arg::pod(&cols),
                        Arg::pod(&length),
                    ],
                );
                let mask = triangular_reference(&real, rows, cols, k, keep_below);
                let mut want: Vec<u32> = Vec::with_capacity(len * words);
                for (flat, value) in mask.iter().enumerate() {
                    for word in 0..words {
                        want.push(if value.to_bits() == 0 {
                            0
                        } else {
                            data[words * flat + word].to_bits()
                        });
                    }
                }
                let want = with_sentinel_slack(want, words);
                assert_eq!(
                    backing_words(&out, want.len()),
                    want,
                    "{symbol} {dtype:?} at k {k} over three threadgroups"
                );
            }
        }

        // The view twins over a transposed 300x2 input: the same 600 positions read
        // through the layout.
        let in_extents = [300i64, 2];
        let in_strides = [2i64, 1];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        let indices = strided_indices(&in_extents, &in_strides, 0);
        for k in [0i64, -1] {
            for (symbol, keep_below) in [
                ("t4a_metal_tril_view", true),
                ("t4a_metal_triu_view", false),
            ] {
                let plan = if keep_below {
                    metal::plan_tril_view(dtype, length, in_layout, length).unwrap()
                } else {
                    metal::plan_triu_view(dtype, length, in_layout, length).unwrap()
                };
                assert_eq!(plan.dispatch.threadgroups, 3, "600 positions");
                let out = device.sentinel_buffer(dtype, len);
                device.dispatch(
                    symbol,
                    plan.dispatch,
                    &[
                        Arg::Buffer {
                            buffer: &out,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &input,
                            offset: 0,
                        },
                        Arg::pod(&tag),
                        Arg::Bytes { bytes: &in_mirror },
                        Arg::pod(&k),
                        Arg::pod(&length),
                    ],
                );
                let mask = triangular_reference(&real, rows, cols, k, keep_below);
                let mut want: Vec<u32> = Vec::with_capacity(len * words);
                for (flat, value) in mask.iter().enumerate() {
                    for word in 0..words {
                        want.push(if value.to_bits() == 0 {
                            0
                        } else {
                            data[words * indices[flat] + word].to_bits()
                        });
                    }
                }
                let want = with_sentinel_slack(want, words);
                assert_eq!(
                    backing_words(&out, want.len()),
                    want,
                    "{symbol} {dtype:?} at k {k} over three threadgroups"
                );
            }
        }
    }

    // Embedding into a destination big enough to need two threadgroups: a rank-1 input
    // of 300 elements inserts an axis of extent 300, so the destination is 300x300. The
    // destination is zero-filled first because only diagonal positions are written.
    for dtype in metal::DTYPES {
        let words = dtype.size() / 4;
        let dtype_tag = dtype.tag();
        let in_extents = [300i64];
        let in_strides = [1i64];
        let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let in_mirror = metal::layout_bytes(&metal::layout_mirror(in_layout).unwrap());
        let source: Vec<f32> = (0..300)
            .flat_map(|index| (0..words).map(move |word| index as f32 + word as f32 * 0.25))
            .collect();
        let source = device.host_buffer(dtype, &source);
        let out_extents = [300i64, 300];
        let out_strides = [1i64, 300];
        let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let out_mirror = metal::layout_bytes(&metal::layout_mirror(out_layout).unwrap());
        let out = device.sentinel_buffer(dtype, 90_000);
        let zero_plan = metal::plan_fill_zero_view(dtype, out_layout, 90_000).unwrap();
        let zero_len = 90_000i64;
        device.dispatch(
            "t4a_metal_fill_zero_view",
            zero_plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &out_mirror },
                Arg::pod(&zero_len),
            ],
        );
        let plan = metal::plan_embed_diagonal_copy(dtype, in_layout, 300, out_layout, 90_000, 0, 1)
            .unwrap();
        assert_eq!(plan.dispatch.threadgroups, 2, "300 positions");
        assert_eq!(plan.bytes, 90_000 * dtype.size() as u64);
        let (axis_a, axis_b, count) = (0i32, 1i32, 300i64);
        device.dispatch(
            "t4a_metal_embed_diagonal_copy",
            plan.dispatch,
            &[
                Arg::Buffer {
                    buffer: &out,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &source,
                    offset: 0,
                },
                Arg::pod(&dtype_tag),
                Arg::Bytes { bytes: &out_mirror },
                Arg::Bytes { bytes: &in_mirror },
                Arg::pod(&axis_a),
                Arg::pod(&axis_b),
                Arg::pod(&count),
            ],
        );
        let mut want = vec![0u32; 90_000 * words];
        for (row, column) in (0..300).map(|index| (index, index)) {
            let index = row + 300 * column;
            for word in 0..words {
                want[index * words + word] = (row as f32 + word as f32 * 0.25).to_bits();
            }
        }
        let want = with_sentinel_slack(want, words);
        assert_eq!(
            backing_words(&out, want.len()),
            want,
            "embed_diagonal_copy {dtype:?} over a 300x300 destination"
        );
    }

    // The flat elementwise-op entries at 600 positions.
    let lhs: Vec<f32> = (0..len).map(|index| (index % 13) as f32 * 0.5).collect();
    let rhs: Vec<f32> = (0..len).map(|index| (index % 5) as f32 - 2.0).collect();
    let left = device.host_buffer(DType::F32, &lhs);
    let right = device.host_buffer(DType::F32, &rhs);
    let out = device.sentinel_buffer(DType::F32, len);
    let plan = metal::plan_compare(DType::F32, dim).unwrap();
    let (tag, cmp) = (DType::F32.tag(), CompareOp::Lt.tag());
    device.dispatch(
        "t4a_metal_compare",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &left,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &right,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&cmp),
            Arg::pod(&dim),
        ],
    );
    let mut want_bytes = vec![0u8; (len + SLACK) * 4];
    for word in want_bytes.chunks_exact_mut(4) {
        word.copy_from_slice(&SENTINEL.to_le_bytes());
    }
    for index in 0..len {
        want_bytes[index] = u8::from(lhs[index] < rhs[index]);
    }
    let want: Vec<u32> = want_bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    assert_eq!(
        backing_words(&out, want.len()),
        want,
        "compare over three threadgroups"
    );

    // select: one byte per position, every third byte nonzero.
    let predicate: Vec<u32> = (0..len.div_ceil(4) + SLACK)
        .map(|word| {
            (0..4).fold(0u32, |packed, byte| {
                let index = word * 4 + byte;
                let set = index < len && index % 3 == 0;
                packed | (u32::from(set) << (8 * byte))
            })
        })
        .collect();
    let predicate = device.host_buffer(DType::F32, &predicate);
    let on_true = device.host_buffer(DType::F32, &lhs);
    let on_false = device.host_buffer(DType::F32, &rhs);
    let out = device.sentinel_buffer(DType::F32, len);
    let plan = metal::plan_select(DType::F32, dim).unwrap();
    device.dispatch(
        "t4a_metal_select",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &predicate,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &on_true,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &on_false,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&dim),
        ],
    );
    let want: Vec<u32> = (0..len)
        .map(|index| {
            if index % 3 == 0 {
                lhs[index].to_bits()
            } else {
                rhs[index].to_bits()
            }
        })
        .collect();
    let want = with_sentinel_slack(want, 1);
    assert_eq!(
        backing_words(&out, want.len()),
        want,
        "select over three threadgroups"
    );

    // clamp with a NaN every 97th element and bounds that move per element.
    let values: Vec<f32> = (0..len)
        .map(|index| {
            if index % 97 == 0 {
                f32::NAN
            } else {
                (index % 11) as f32 * 0.25 - 1.0
            }
        })
        .collect();
    let lows: Vec<f32> = (0..len)
        .map(|index| -0.5 - (index % 3) as f32 * 0.1)
        .collect();
    let highs: Vec<f32> = (0..len)
        .map(|index| 0.5 + (index % 4) as f32 * 0.1)
        .collect();
    let (values, lows, highs) = (
        device.host_buffer(DType::F32, &values),
        device.host_buffer(DType::F32, &lows),
        device.host_buffer(DType::F32, &highs),
    );
    let out = device.sentinel_buffer(DType::F32, len);
    let plan = metal::plan_clamp(DType::F32, dim).unwrap();
    device.dispatch(
        "t4a_metal_clamp",
        plan.dispatch,
        &[
            Arg::Buffer {
                buffer: &out,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &values,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &lows,
                offset: 0,
            },
            Arg::Buffer {
                buffer: &highs,
                offset: 0,
            },
            Arg::pod(&tag),
            Arg::pod(&dim),
        ],
    );
    let got: Vec<f32> = unsafe { read_f32(&out, 0, len + SLACK) }.to_vec();
    for (index, got) in got.iter().take(len).enumerate() {
        let index_f = index as f32;
        let value = if index % 97 == 0 {
            f32::NAN
        } else {
            (index_f % 11.0) * 0.25 - 1.0
        };
        let low = -0.5 - (index_f % 3.0) * 0.1;
        let high = 0.5 + (index_f % 4.0) * 0.1;
        let want = clamp_reference(value, low, high);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "clamp element {index} over three threadgroups"
        );
    }
    assert_tail_is_sentinel(&out, DType::F32, len, "clamp over three threadgroups");
}
