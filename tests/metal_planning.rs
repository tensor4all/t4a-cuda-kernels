//! The Metal source set and planning target. Needs no device.

use t4a_cuda_kernels::metal::{self, Dispatch, Plan, SYMBOLS, THREADGROUP};
use t4a_cuda_kernels::{DType, Entry, Error, UnaryOp};

/// One MSL `#define` whose value is a number.
fn msl_define(header: &str, name: &str) -> u32 {
    let prefix = format!("#define {name} ");
    header
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("the MSL header defines {name}"))
        .strip_prefix(&prefix)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn abi_revision_matches_the_msl_header() {
    let header = metal::source::ABI_HEADER;
    assert_eq!(
        msl_define(header, "T4A_METAL_ABI_REVISION"),
        metal::ABI_REVISION,
        "the MSL header and the crate must declare the same Metal revision"
    );
    assert_ne!(
        metal::ABI_REVISION,
        0,
        "a revision starts at one, so an artifact key can never mean 'none'"
    );
}

#[test]
fn dtype_tags_match_the_msl_header() {
    let header = metal::source::ABI_HEADER;
    assert_eq!(msl_define(header, "T4A_METAL_F32"), DType::F32.tag() as u32);
    assert_eq!(msl_define(header, "T4A_METAL_C32"), DType::C32.tag() as u32);
    // The catalogue serves exactly these two and no wider one.
    assert_eq!(metal::DTYPES, [DType::F32, DType::C32]);
}

#[test]
fn unary_tags_match_the_msl_header() {
    let header = metal::source::ABI_HEADER;
    for op in metal::UNARY_OPS {
        let name = match op {
            UnaryOp::Neg => "T4A_METAL_UNARY_NEG",
            UnaryOp::Exp => "T4A_METAL_UNARY_EXP",
            UnaryOp::Log => "T4A_METAL_UNARY_LOG",
            UnaryOp::Sin => "T4A_METAL_UNARY_SIN",
            UnaryOp::Cos => "T4A_METAL_UNARY_COS",
            UnaryOp::Tanh => "T4A_METAL_UNARY_TANH",
            UnaryOp::Sqrt => "T4A_METAL_UNARY_SQRT",
            UnaryOp::Rsqrt => "T4A_METAL_UNARY_RSQRT",
            UnaryOp::Expm1 => "T4A_METAL_UNARY_EXPM1",
            UnaryOp::Log1p => "T4A_METAL_UNARY_LOG1P",
            UnaryOp::Erf => "T4A_METAL_UNARY_ERF",
            UnaryOp::Abs => "T4A_METAL_UNARY_ABS",
            UnaryOp::Sign => "T4A_METAL_UNARY_SIGN",
            UnaryOp::Conj => "T4A_METAL_UNARY_CONJ",
        };
        assert_eq!(
            msl_define(header, name),
            op.tag() as u32,
            "{name} and UnaryOp::tag must agree"
        );
    }
    // The three tags MSL cannot serve stay documented but unusable.
    for op in [UnaryOp::Expm1, UnaryOp::Log1p, UnaryOp::Erf] {
        assert!(!metal::UNARY_OPS.contains(&op));
        assert_eq!(
            metal::plan_unary(op, DType::F32, DType::F32, 8),
            Err(Error::UnsupportedOp)
        );
    }
}

#[test]
fn symbols_are_the_cuda_names_with_the_metal_prefix() {
    for (entry, symbol) in SYMBOLS {
        let expected = entry
            .name()
            .strip_prefix("t4a_cuda_")
            .map(|suffix| format!("t4a_metal_{suffix}"))
            .expect("every entry name carries the CUDA prefix");
        assert_eq!(*symbol, expected);
        assert_eq!(metal::symbol(*entry), Some(*symbol));
    }
}

#[test]
fn every_symbol_has_one_kernel_and_no_other_kernel_exists() {
    for (entry, symbol) in SYMBOLS {
        let declaring = metal::source::MODULES
            .iter()
            .filter(|module| module.source.contains(&format!("kernel void {symbol}(")))
            .count();
        assert_eq!(
            declaring, 1,
            "{symbol} ({entry:?}) must be declared in exactly one module"
        );
        assert!(metal::source::module_of(*entry).is_some());
    }
    let declared = metal::source::MODULES
        .iter()
        .map(|module| module.source.matches("kernel void ").count())
        .sum::<usize>();
    assert_eq!(
        declared,
        SYMBOLS.len(),
        "the Metal text declares exactly the served entries and nothing else, so \
         no MSL kernel exists for an operation MetalPerformanceShaders owns"
    );
}

#[test]
fn composed_modules_carry_the_header_and_no_include() {
    for module in metal::source::MODULES {
        assert_eq!(
            module
                .source
                .lines()
                .filter(|line| line.trim() == metal::source::INCLUDE_LINE)
                .count(),
            1,
            "{}: a module includes the header once, for a file-based compile",
            module.name
        );
        let text = metal::source::compose(module);
        assert!(text.starts_with(metal::source::ABI_HEADER));
        assert!(
            !text.contains(metal::source::INCLUDE_LINE),
            "{}: compose prepends the header, so the module's own include is removed",
            module.name
        );
        assert_eq!(
            text.matches("#ifndef T4A_METAL_ABI_MSL").count(),
            1,
            "{}: the composed module holds the header exactly once",
            module.name
        );
    }
}

#[test]
fn the_metal_text_holds_no_wider_float() {
    let mut text = metal::source::ABI_HEADER.to_string();
    for module in metal::source::MODULES {
        text.push_str(module.source);
    }
    for wider in ["double", "float64", "long double"] {
        assert!(
            !text.contains(wider),
            "Metal has no {wider}; f64 work belongs to the CPU engine"
        );
    }
}

#[test]
fn complex32_has_the_host_layout() {
    #[repr(C)]
    struct HostComplex32 {
        re: f32,
        im: f32,
    }
    assert_eq!(core::mem::size_of::<HostComplex32>(), 8);
    assert_eq!(core::mem::align_of::<HostComplex32>(), 4);
    assert_eq!(core::mem::offset_of!(HostComplex32, re), 0);
    assert_eq!(core::mem::offset_of!(HostComplex32, im), 4);
    // The MSL header asserts the same numbers for its own struct, which the device
    // lane compiles; the sizes here are the ones a caller's buffer uses.
    assert_eq!(DType::C32.size(), 8);
}

#[test]
fn the_plan_is_a_whole_number_of_simd_groups() {
    assert_eq!(THREADGROUP % metal::SIMD_WIDTH, 0);
}

#[test]
fn unsupported_dtypes_are_rejected() {
    for (entry, _) in SYMBOLS {
        for dtype in [DType::F64, DType::C64, DType::I32, DType::I64, DType::Bool] {
            assert_eq!(
                metal::plan_elementwise(*entry, dtype, 8),
                Err(Error::UnsupportedDType),
                "{entry:?} must reject {dtype:?}"
            );
        }
    }
    // Complex32 division is deferred: `Entry::Div` serves f32 and f64 only.
    assert_eq!(
        metal::plan_elementwise(Entry::Div, DType::C32, 8),
        Err(Error::UnsupportedDType)
    );
}

#[test]
fn elementwise_entries_without_a_kernel_are_rejected() {
    for entry in [
        Entry::FillOne,
        Entry::Binary,
        Entry::Compare,
        Entry::Select,
        Entry::Clamp,
    ] {
        assert!(entry.is_elementwise());
        assert_eq!(metal::symbol(entry), None);
        assert_eq!(
            metal::plan_elementwise(entry, DType::F32, 8),
            Err(Error::UnsupportedOp),
            "{entry:?} has no Metal kernel, so it must not produce a plan"
        );
    }
    assert_eq!(
        metal::plan_elementwise(Entry::Reduce, DType::F32, 8),
        Err(Error::NotElementwise)
    );
}

#[test]
fn unary_admission_follows_the_pair_then_the_msl_set() {
    assert!(metal::plan_unary(UnaryOp::Abs, DType::F32, DType::C32, 8).is_ok());
    assert!(metal::plan_unary(UnaryOp::Conj, DType::C32, DType::C32, 8).is_ok());
    assert!(metal::plan_unary(UnaryOp::Sign, DType::C32, DType::C32, 8).is_ok());
    // A pair the shared matrix rejects is an unsupported operation.
    assert_eq!(
        metal::plan_unary(UnaryOp::Exp, DType::C32, DType::C32, 8),
        Err(Error::UnsupportedOp)
    );
    assert_eq!(
        metal::plan_unary(UnaryOp::Conj, DType::F32, DType::F32, 8),
        Err(Error::UnsupportedOp)
    );
    assert_eq!(
        metal::plan_unary(UnaryOp::Abs, DType::C32, DType::C32, 8),
        Err(Error::UnsupportedOp)
    );
    // A pair the matrix admits outside the Metal dtype set is an unsupported dtype.
    assert_eq!(
        metal::plan_unary(UnaryOp::Exp, DType::F64, DType::F64, 8),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        metal::plan_unary(UnaryOp::Abs, DType::F64, DType::C64, 8),
        Err(Error::UnsupportedDType)
    );
}

#[test]
fn geometry_covers_the_length() {
    let cases = [
        (0, 0u32),
        (1, 1),
        (255, 1),
        (256, 1),
        (257, 2),
        (100_003, 391),
    ];
    for (len, threadgroups) in cases {
        let plan = metal::plan_fill_zero(DType::F32, len).unwrap();
        assert_eq!(plan.dispatch.threadgroups, threadgroups, "len {len}");
        assert_eq!(plan.dispatch.threads_per_threadgroup, THREADGROUP);
        assert_eq!(plan.dispatch.is_empty(), len == 0);
        assert_eq!(plan.bytes, len as u64 * 4);
        let covered = u64::from(plan.dispatch.threadgroups) * u64::from(THREADGROUP);
        assert!(covered >= len as u64, "len {len} is not covered");
    }
    assert_eq!(
        metal::plan_copy_exact(DType::C32, 3).unwrap(),
        Plan {
            dispatch: Dispatch {
                threadgroups: 1,
                threads_per_threadgroup: THREADGROUP,
            },
            bytes: 24,
        }
    );
}

#[test]
fn bad_lengths_are_rejected() {
    assert_eq!(
        metal::plan_fill_zero(DType::F32, -1),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        metal::plan_fill_zero(DType::F32, i64::MAX),
        Err(Error::LaunchLimit)
    );
    // The last length whose element positions fit the 32-bit `uint` a kernel reads
    // is accepted; one more is not.
    let longest = i64::from(u32::MAX);
    let plan = metal::plan_fill_zero(DType::C32, longest).unwrap();
    assert_eq!(plan.bytes, u64::from(u32::MAX) * 8);
    assert_eq!(plan.dispatch.threadgroups, u32::MAX.div_ceil(THREADGROUP));
    assert_eq!(
        metal::plan_fill_zero(DType::C32, longest + 1),
        Err(Error::LaunchLimit)
    );
}

#[test]
fn the_complex_copy_widens_before_doubling_the_word_index() {
    // `copy_exact` moves a Complex32 element as two 32-bit words. `2 * index` in
    // 32-bit arithmetic would wrap at index 2^31, on a length the planner accepts,
    // so the text must double a widened index. No device beyond this machine's
    // memory can execute that length, which is why the guard is textual.
    let text = metal::source::compose(&metal::source::MODULES[0]);
    assert!(
        text.contains("2 * ulong(index)"),
        "the Complex32 copy must widen the word index before doubling it"
    );
}
