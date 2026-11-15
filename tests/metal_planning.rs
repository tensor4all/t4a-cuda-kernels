//! The Metal source set and planning target. Needs no device.

use t4a_cuda_kernels::metal::{self, Dispatch, Plan, SYMBOLS, THREADGROUP};
use t4a_cuda_kernels::{
    BinaryOp, CAxisValues, CLayout, CompareOp, DType, Entry, Error, LayoutRef, ReduceOp,
    ReduceStrategy, ScalarSide, UnaryOp,
};

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
    for entry in metal::FLAT_ENTRIES {
        for dtype in [DType::F64, DType::C64, DType::I32, DType::I64, DType::Bool] {
            assert_eq!(
                metal::plan_elementwise(entry, dtype, 8),
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
fn the_flat_planner_serves_exactly_its_six_entries() {
    let flat = LayoutRef::new(&[8], &[1], 0).unwrap();
    for entry in metal::FLAT_ENTRIES {
        assert!(
            metal::plan_elementwise(entry, DType::F32, 8).is_ok(),
            "{entry:?}"
        );
    }
    // An element-wise entry with its own parameter block is refused here, whatever
    // its dtype: the slice-1 planner cannot describe it.
    for entry in [
        Entry::Unary,
        Entry::Binary,
        Entry::Compare,
        Entry::Select,
        Entry::Clamp,
        Entry::FillOne,
    ] {
        assert_eq!(
            metal::plan_elementwise(entry, DType::F32, 8),
            Err(Error::UnsupportedOp),
            "{entry:?} has its own planner"
        );
    }
    // `FillZeroView` is not in the shared element-wise set at all: it takes a layout.
    assert_eq!(
        metal::plan_elementwise(Entry::FillZeroView, DType::F32, 8),
        Err(Error::NotElementwise)
    );
    for entry in [
        Entry::MaterializeStrided,
        Entry::ContiguousToView,
        Entry::StridedToStrided,
        Entry::BroadcastInDim,
        Entry::ExtractDiagonal,
        Entry::EmbedDiagonalCopy,
        Entry::Tril,
        Entry::Triu,
        Entry::TrilView,
        Entry::TriuView,
        Entry::Convert,
    ] {
        assert_eq!(
            metal::plan_elementwise(entry, DType::F32, 8),
            Err(Error::NotElementwise),
            "{entry:?} is not in the shared element-wise set"
        );
    }
    assert_eq!(
        metal::plan_elementwise(Entry::Reduce, DType::F32, 8),
        Err(Error::NotElementwise)
    );
    // The six entries that do not fit the flat planner still plan through their own.
    assert!(metal::plan_compare(DType::F32, 8).is_ok());
    assert!(metal::plan_select(DType::F32, 8).is_ok());
    assert!(metal::plan_clamp(DType::F32, 8).is_ok());
    assert!(metal::plan_fill_zero_view(DType::F32, flat, 8).is_ok());
    assert!(metal::plan_binary(BinaryOp::Max, DType::F32, ScalarSide::Both, 8).is_ok());
    assert!(metal::plan_convert(DType::C32, DType::F32, 8).is_ok());
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

#[test]
fn tags_match_the_msl_header() {
    let header = metal::source::ABI_HEADER;
    assert_eq!(
        msl_define(header, "T4A_METAL_ABI_REVISION"),
        metal::ABI_REVISION
    );
    assert_eq!(
        msl_define(header, "T4A_METAL_THREADGROUP"),
        THREADGROUP,
        "the reduction fold's static tree is sized from the header's width"
    );
    assert_eq!(
        msl_define(header, "T4A_METAL_MAX_RANK") as usize,
        t4a_cuda_kernels::MAX_RANK
    );
    for (op, name) in [
        (BinaryOp::Max, "T4A_METAL_BINARY_MAX"),
        (BinaryOp::Min, "T4A_METAL_BINARY_MIN"),
        (BinaryOp::Pow, "T4A_METAL_BINARY_POW"),
        (BinaryOp::Rem, "T4A_METAL_BINARY_REM"),
        (BinaryOp::Mul, "T4A_METAL_BINARY_MUL"),
        (BinaryOp::DivComplex, "T4A_METAL_BINARY_DIV_COMPLEX"),
        (BinaryOp::Div, "T4A_METAL_BINARY_DIV"),
        (BinaryOp::DivChecked, "T4A_METAL_BINARY_DIV_CHECKED"),
        (BinaryOp::RemChecked, "T4A_METAL_BINARY_REM_CHECKED"),
        (BinaryOp::PowChecked, "T4A_METAL_BINARY_POW_CHECKED"),
        (BinaryOp::MixedAdd, "T4A_METAL_BINARY_MIXED_ADD"),
        (BinaryOp::MixedSub, "T4A_METAL_BINARY_MIXED_SUB"),
        (BinaryOp::MixedMul, "T4A_METAL_BINARY_MIXED_MUL"),
        (BinaryOp::MixedDiv, "T4A_METAL_BINARY_MIXED_DIV"),
    ] {
        assert_eq!(msl_define(header, name), op.tag() as u32, "{name}");
    }
    for (side, name) in [
        (ScalarSide::Both, "T4A_METAL_SCALAR_BOTH"),
        (ScalarSide::Left, "T4A_METAL_SCALAR_LHS"),
        (ScalarSide::Right, "T4A_METAL_SCALAR_RHS"),
    ] {
        assert_eq!(msl_define(header, name), side.tag() as u32, "{name}");
    }
    for (op, name) in [
        (CompareOp::Eq, "T4A_METAL_COMPARE_EQ"),
        (CompareOp::Lt, "T4A_METAL_COMPARE_LT"),
        (CompareOp::Le, "T4A_METAL_COMPARE_LE"),
        (CompareOp::Gt, "T4A_METAL_COMPARE_GT"),
        (CompareOp::Ge, "T4A_METAL_COMPARE_GE"),
    ] {
        assert_eq!(msl_define(header, name), op.tag() as u32, "{name}");
    }
    for (op, name) in [
        (ReduceOp::Sum, "T4A_METAL_REDUCE_SUM"),
        (ReduceOp::Prod, "T4A_METAL_REDUCE_PROD"),
        (ReduceOp::SumSquares, "T4A_METAL_REDUCE_SUM_SQUARES"),
        (ReduceOp::Max, "T4A_METAL_REDUCE_MAX"),
        (ReduceOp::Min, "T4A_METAL_REDUCE_MIN"),
    ] {
        assert_eq!(msl_define(header, name), op.tag() as u32, "{name}");
    }
    for (strategy, name) in [
        (ReduceStrategy::Unit, "T4A_METAL_REDUCE_UNIT"),
        (ReduceStrategy::Plane, "T4A_METAL_REDUCE_PLANE"),
    ] {
        assert_eq!(msl_define(header, name), strategy.tag() as u32, "{name}");
    }
}

#[test]
fn the_layout_mirror_is_the_msl_struct() {
    assert_eq!(core::mem::size_of::<CLayout>(), metal::LAYOUT_BYTES);
    assert_eq!(core::mem::align_of::<CLayout>(), 8);
    assert_eq!(
        core::mem::size_of::<CAxisValues>(),
        metal::AXIS_VALUES_BYTES
    );
    assert_eq!(core::mem::align_of::<CAxisValues>(), 8);

    let extents = [2i64, 3];
    let strides = [1i64, 2];
    let layout = LayoutRef::new(&extents, &strides, 5).unwrap();
    let mirror = metal::layout_mirror(layout).unwrap();
    assert_eq!(mirror.rank, 2);
    assert_eq!(mirror.offset, 5);
    assert_eq!(&mirror.extents[..2], &[2, 3]);
    assert_eq!(&mirror.extents[2..], &[0; 6], "slots past rank are zeroed");
    assert_eq!(&mirror.strides[..2], &[1, 2]);
    assert_eq!(&mirror.strides[2..], &[0; 6]);
    let bytes = metal::layout_bytes(&mirror);
    assert_eq!(bytes.len(), metal::LAYOUT_BYTES);
    // Field by field, little-endian, in declaration order: rank, reserved, offset,
    // extents, strides.
    assert_eq!(&bytes[0..4], &2i32.to_le_bytes());
    assert_eq!(&bytes[4..8], &0i32.to_le_bytes());
    assert_eq!(&bytes[8..16], &5i64.to_le_bytes());
    assert_eq!(&bytes[16..24], &2i64.to_le_bytes());
    assert_eq!(&bytes[80..88], &1i64.to_le_bytes());

    let deep = [1i64; 9];
    assert_eq!(
        metal::layout_mirror(LayoutRef::new(&deep, &deep, 0).unwrap()),
        Err(Error::RankOutOfRange)
    );

    let dims = CAxisValues::new(&[0, 2]).unwrap();
    let bytes = metal::axis_values_bytes(&dims);
    assert_eq!(bytes.len(), metal::AXIS_VALUES_BYTES);
    assert_eq!(&bytes[0..4], &2i32.to_le_bytes());
    assert_eq!(&bytes[8..16], &0i64.to_le_bytes());
    assert_eq!(
        CAxisValues::new(&[0; 9]).err(),
        Some(Error::RankOutOfRange),
        "a value vector deeper than the mirror is refused"
    );
}

#[test]
fn every_new_planner_covers_its_domain() {
    for len in [0i64, 1, 255, 256, 257, 4096, 100_003] {
        let extents = [len];
        let strides = [1i64];
        let compact = LayoutRef::new(&extents, &strides, 0).unwrap();
        let cell = [len, 1];
        let cell_strides = [1i64, len.max(1)];
        let matrix = LayoutRef::new(&cell, &cell_strides, 0).unwrap();

        let plans = [
            (
                metal::plan_fill_zero_view(DType::F32, compact, len).unwrap(),
                4u64,
            ),
            (
                metal::plan_materialize_strided(DType::C32, compact, len).unwrap(),
                8,
            ),
            (
                metal::plan_contiguous_to_view(DType::F32, compact, len).unwrap(),
                4,
            ),
            (
                metal::plan_strided_to_strided(DType::F32, compact, len, compact, len).unwrap(),
                4,
            ),
            (
                metal::plan_broadcast_in_dim(DType::F32, compact, len, compact, len, &[0]).unwrap(),
                4,
            ),
            (metal::plan_compare(DType::F32, len).unwrap(), 1),
            (metal::plan_select(DType::F32, len).unwrap(), 4),
            (metal::plan_clamp(DType::F32, len).unwrap(), 4),
            (
                metal::plan_binary(BinaryOp::Max, DType::F32, ScalarSide::Both, len).unwrap(),
                4,
            ),
            (metal::plan_convert(DType::C32, DType::F32, len).unwrap(), 8),
            (metal::plan_convert(DType::F32, DType::C32, len).unwrap(), 4),
            (metal::plan_tril(DType::F32, 1, len, len).unwrap(), 4),
            (metal::plan_triu(DType::C32, 1, len, len).unwrap(), 8),
            (
                metal::plan_tril_view(DType::F32, len, matrix, len).unwrap(),
                4,
            ),
        ];
        for (plan, size) in plans {
            assert_eq!(
                plan.dispatch.threadgroups as u64,
                (len as u64).div_ceil(u64::from(THREADGROUP)),
                "len {len}"
            );
            assert_eq!(plan.dispatch.threads_per_threadgroup, THREADGROUP);
            assert_eq!(plan.dispatch.is_empty(), len == 0);
            assert!(
                u64::from(plan.dispatch.threadgroups) * u64::from(THREADGROUP) >= len as u64,
                "len {len}"
            );
            assert_eq!(plan.bytes, len as u64 * size, "len {len}");
        }
    }
}

#[test]
fn the_embed_destination_is_larger_than_the_position_domain() {
    // The launch walks the input; the destination gains the inserted axis, so
    // `bytes` is the output allocation. A plan whose bytes were the position count
    // would under-allocate the destination the engine writes.
    let in_extents = [3i64, 4];
    let in_strides = [1i64, 3];
    let input = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_extents = [3i64, 4, 3];
    let out_strides = [1i64, 3, 12];
    let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let plan = metal::plan_embed_diagonal_copy(DType::F32, input, 12, out, 36, 0, 2).unwrap();
    assert_eq!(plan.dispatch.threadgroups, 1, "twelve positions");
    assert_eq!(
        plan.bytes,
        36 * 4,
        "the output's 36 elements, not the 12 positions"
    );

    // The diagonal map: the diagonal of axes (0, 1) lands at output axis 0.
    let out_extents = [3i64];
    let out_strides = [1i64];
    let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    let plan = metal::plan_extract_diagonal(DType::F32, input, 12, out, 3, 0, 1).unwrap();
    assert_eq!(plan.dispatch.threadgroups, 1);
    assert_eq!(plan.bytes, 3 * 4);
}

#[test]
fn reductions_split_the_strategies_at_the_threadgroup() {
    // Axis 1 of a (4, axis) input, keepdims output (4, 1).
    for axis_len in [1i64, 2, 255, 256] {
        let in_extents = [4i64, axis_len];
        let in_strides = [1i64, 4];
        let input = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let out_extents = [4i64, 1];
        let out_strides = [1i64, 4];
        let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let plan = metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            out,
            4,
            input,
            4 * axis_len,
            1,
        )
        .unwrap();
        assert_eq!(
            plan.dispatch.threadgroups, 1,
            "four outputs in one threadgroup"
        );
        assert_eq!(plan.dispatch.threads_per_threadgroup, THREADGROUP);
        assert_eq!(plan.bytes, 4 * 4);
    }
    for axis_len in [255i64, 256, 257, 1024] {
        let in_extents = [4i64, axis_len];
        let in_strides = [1i64, 4];
        let input = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
        let out_extents = [4i64, 1];
        let out_strides = [1i64, 4];
        let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
        let strategy = ReduceStrategy::Plane;
        let plan = metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            strategy,
            out,
            4,
            input,
            4 * axis_len,
            1,
        );
        if axis_len < i64::from(THREADGROUP) {
            assert_eq!(
                plan,
                Err(Error::StrategyMismatch),
                "plane needs a whole plane"
            );
            continue;
        }
        let plan = plan.unwrap();
        assert_eq!(plan.dispatch.threadgroups, 4, "one threadgroup per output");
        assert_eq!(plan.bytes, 4 * 4);
    }
    // Unit above one threadgroup is refused too.
    let in_extents = [4i64, 257];
    let in_strides = [1i64, 4];
    let input = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_extents = [4i64, 1];
    let out_strides = [1i64, 4];
    let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    assert_eq!(
        metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            out,
            4,
            input,
            1028,
            1
        ),
        Err(Error::StrategyMismatch)
    );
    // The output keeps the axis at extent one.
    let out_extents = [4i64, 2];
    let out = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    assert_eq!(
        metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            out,
            8,
            input,
            1028,
            1
        ),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn the_32_bit_cliffs_are_refused() {
    let longest = i64::from(u32::MAX);
    let extents = [longest];
    let strides = [1i64];
    let compact = LayoutRef::new(&extents, &strides, 0).unwrap();
    let plan = metal::plan_materialize_strided(DType::C32, compact, longest).unwrap();
    assert_eq!(plan.dispatch.threadgroups, u32::MAX.div_ceil(THREADGROUP));
    assert_eq!(plan.bytes, u64::from(u32::MAX) * 8);

    let over = [longest + 1];
    let compact = LayoutRef::new(&over, &strides, 0).unwrap();
    assert_eq!(
        metal::plan_materialize_strided(DType::F32, compact, longest + 1),
        Err(Error::LaunchLimit)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, compact, longest + 1, compact, longest + 1, &[0]),
        Err(Error::LaunchLimit)
    );

    // A component-length tag: `DivComplex` addresses `2 * position`.
    let half = i64::from(u32::MAX / 2);
    let plan =
        metal::plan_binary(BinaryOp::DivComplex, DType::C32, ScalarSide::Both, half).unwrap();
    assert_eq!(plan.bytes, half as u64 * 8);
    assert_eq!(
        metal::plan_binary(BinaryOp::DivComplex, DType::C32, ScalarSide::Both, half + 1),
        Err(Error::LaunchLimit)
    );
    // A scalar Complex32 multiplication addresses whole elements only.
    assert!(metal::plan_binary(BinaryOp::Mul, DType::C32, ScalarSide::Right, longest).is_ok());

    // A keepdims output wider than a 32-bit threadgroup index.
    let wide = [1i64, longest + 1];
    let wide_strides = [1i64, 1];
    let wide_in = [1i64, longest + 1];
    let input = LayoutRef::new(&wide_in, &wide_strides, 0).unwrap();
    let out = LayoutRef::new(&wide, &wide_strides, 0).unwrap();
    assert_eq!(
        metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            out,
            longest + 1,
            input,
            longest + 1,
            0
        ),
        Err(Error::LaunchLimit)
    );
}

#[test]
fn the_new_planners_reject_what_they_cannot_serve() {
    let extents = [4i64, 3];
    let strides = [1i64, 4];
    let compact = LayoutRef::new(&extents, &strides, 0).unwrap();
    let transposed = LayoutRef::new(&extents, &[3i64, 1], 0).unwrap();
    let broadcast = LayoutRef::new(&extents, &[0i64, 1], 0).unwrap();

    // Dtypes.
    for dtype in [DType::F64, DType::C64, DType::I32, DType::I64, DType::Bool] {
        assert_eq!(
            metal::plan_materialize_strided(dtype, compact, 12),
            Err(Error::UnsupportedDType)
        );
        assert_eq!(
            metal::plan_tril(dtype, 4, 3, 12),
            Err(Error::UnsupportedDType)
        );
        assert_eq!(metal::plan_compare(dtype, 12), Err(Error::UnsupportedDType));
    }
    assert_eq!(
        metal::plan_compare(DType::C32, 12),
        Err(Error::UnsupportedDType),
        "the shared table compares real and integer dtypes only"
    );
    assert_eq!(
        metal::plan_select(DType::C32, 12),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        metal::plan_clamp(DType::C32, 12),
        Err(Error::UnsupportedDType)
    );

    // Convert pairs: identity and out-of-set.
    assert_eq!(
        metal::plan_convert(DType::F32, DType::F32, 12),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        metal::plan_convert(DType::F32, DType::F64, 12),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        metal::plan_convert(DType::C64, DType::C32, 12),
        Err(Error::UnsupportedDType)
    );

    // Binary tags.
    for op in [
        BinaryOp::MixedAdd,
        BinaryOp::MixedSub,
        BinaryOp::MixedMul,
        BinaryOp::MixedDiv,
    ] {
        assert_eq!(
            metal::plan_binary(op, DType::F32, ScalarSide::Right, 12),
            Err(Error::UnsupportedOp),
            "{op:?} has no Metal caller"
        );
    }
    for op in [BinaryOp::Pow, BinaryOp::Rem, BinaryOp::Max] {
        assert_eq!(
            metal::plan_binary(op, DType::C32, ScalarSide::Both, 12),
            Err(Error::UnsupportedOp)
        );
    }
    for op in [
        BinaryOp::DivChecked,
        BinaryOp::RemChecked,
        BinaryOp::PowChecked,
    ] {
        assert_eq!(
            metal::plan_binary(op, DType::F32, ScalarSide::Both, 12),
            Err(Error::UnsupportedOp)
        );
    }
    assert_eq!(
        metal::plan_binary(BinaryOp::Mul, DType::F32, ScalarSide::Both, 12),
        Err(Error::UnsupportedOp),
        "the Both form of multiplication stays with Entry::Mul"
    );

    // Layouts.
    assert_eq!(
        metal::plan_fill_zero_view(DType::F32, broadcast, 12),
        Err(Error::BroadcastDestination),
        "a stride-zero destination is not injective"
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, transposed, 12, compact, 12, &[0, 1]),
        Err(Error::NotCompact)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, compact, 12, compact, 12, &[0]),
        Err(Error::RankLength)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, compact, 12, compact, 12, &[0, 0]),
        Err(Error::DuplicateAxis)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, compact, 12, compact, 12, &[0, 2]),
        Err(Error::InvalidAxis)
    );
    let wider = [4i64, 5];
    let wider_strides = [1i64, 4];
    let wider = LayoutRef::new(&wider, &wider_strides, 0).unwrap();
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, wider, 20, compact, 12, &[0, 1]),
        Err(Error::ShapeMismatch),
        "a mapped axis keeps its extent unless it is one"
    );
    assert_eq!(
        metal::plan_embed_diagonal_copy(DType::F32, transposed, 12, compact, 12, 0, 1),
        Err(Error::NotCompact)
    );
    let other_extents = [4i64, 4];
    let other = LayoutRef::new(&other_extents, &strides, 0).unwrap();
    assert_eq!(
        metal::plan_strided_to_strided(DType::F32, compact, 12, other, 16),
        Err(Error::CountMismatch),
        "the extents decide, not the strides"
    );
    assert_eq!(
        metal::plan_strided_to_strided(DType::F32, compact, 12, broadcast, 12),
        Err(Error::BroadcastDestination)
    );
    assert_eq!(
        metal::plan_tril_view(DType::F32, 12, LayoutRef::new(&[12], &[1], 0).unwrap(), 12),
        Err(Error::RankOutOfRange),
        "the triangle needs two axes"
    );
    assert_eq!(
        metal::plan_tril_view(DType::F32, 8, compact, 12),
        Err(Error::OffsetOutOfSpan),
        "the compact output of the input's extents must fit"
    );

    // Shapes and lengths.
    assert_eq!(
        metal::plan_tril(DType::F32, 4, 3, 11),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        metal::plan_tril(DType::F32, -1, 3, 12),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        metal::plan_tril(DType::F32, 0, 3, 12),
        Err(Error::ShapeMismatch)
    );
    assert!(metal::plan_tril(DType::F32, 0, 3, 0)
        .unwrap()
        .dispatch
        .is_empty());
    assert_eq!(
        metal::plan_compare(DType::F32, -1),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        metal::plan_materialize_strided(DType::F32, compact, 11),
        Err(Error::OffsetOutOfSpan),
        "the layout reaches past the caller's allocation"
    );
}

#[test]
fn the_reduction_text_folds_in_a_threadgroup() {
    let reduce = metal::source::MODULES
        .iter()
        .find(|module| module.name == "reduce")
        .unwrap();
    assert!(reduce
        .source
        .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    assert!(reduce.source.contains("T4A_METAL_THREADGROUP"));
    assert!(reduce
        .source
        .contains("threadgroup float scratch_f32[T4A_METAL_THREADGROUP]"));
    // The movement and diagonal modules address through the by-value layout, and no
    // module reaches past the catalogue: no atomics, no warp intrinsics that would
    // not be threadgroup-local, and no MPS class.
    let movement = metal::source::MODULES
        .iter()
        .find(|module| module.name == "movement")
        .unwrap();
    assert!(movement.source.contains("t4a_metal_decode"));
    let mut text = String::new();
    for module in metal::source::MODULES {
        text.push_str(&metal::source::compose(module));
    }
    for forbidden in ["atomic_", "simd_shuffle", "MPSMatrix", "MPSGraph"] {
        assert!(
            !text.contains(forbidden),
            "the Metal text must not use {forbidden}"
        );
    }
}

#[test]
fn a_view_destination_plans_its_allocation_not_its_element_count() {
    // The engine allocates `plan.bytes` and hands it to the kernel as element zero, so
    // a destination that starts at an offset or strides past its element count must
    // plan the whole backing span: `[2]` with stride 2 and offset 1 writes element 3.
    let extents = [2i64];
    let strides = [2i64];
    let view = LayoutRef::new(&extents, &strides, 1).unwrap();
    assert_eq!(view.element_count().unwrap(), 2);
    let alloc = 4i64;
    for (plan, what) in [
        (
            metal::plan_fill_zero_view(DType::F32, view, alloc).unwrap(),
            "fill_zero_view",
        ),
        (
            metal::plan_contiguous_to_view(DType::F32, view, alloc).unwrap(),
            "contiguous_to_view",
        ),
        (
            metal::plan_strided_to_strided(DType::F32, view, alloc, view, alloc).unwrap(),
            "strided_to_strided",
        ),
    ] {
        assert_eq!(
            plan.bytes,
            (alloc as u64) * 4,
            "{what} must plan the destination allocation"
        );
        assert_eq!(plan.dispatch.threadgroups, 1);
    }

    // Embedding gains an axis, so a compact two-element input plans the eight-element
    // destination it writes into.
    let in_extents = [2i64, 2];
    let in_strides = [1i64, 2];
    let input = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_extents = [2i64, 2, 2];
    let out_strides = [1i64, 2, 4];
    let out = LayoutRef::new(&out_extents, &out_strides, 1).unwrap();
    let plan = metal::plan_embed_diagonal_copy(DType::F32, input, 4, out, 9, 0, 2).unwrap();
    assert_eq!(
        plan.bytes,
        9 * 4,
        "the destination allocation, offset included"
    );

    // A reduce output that is a view plans its allocation too.
    let in_extents = [2i64, 4];
    let in_strides = [1i64, 2];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_extents = [2i64, 1];
    let out_strides = [3i64, 1];
    let out_view = LayoutRef::new(&out_extents, &out_strides, 2).unwrap();
    let plan = metal::plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        out_view,
        8,
        in_layout,
        8,
        1,
    )
    .unwrap();
    assert_eq!(plan.bytes, 8 * 4, "reduce plans the output allocation");
}

#[test]
fn layouts_the_mirror_cannot_carry_are_refused() {
    // The 144-byte mirror has MAX_RANK extent and stride slots; a ninth axis would be
    // dropped by `layout_mirror`, so every planner that takes a layout refuses it.
    let extents = [1i64; 9];
    let strides = [1i64; 9];
    let rank_nine = LayoutRef::new(&extents, &strides, 0).unwrap();
    let compact = LayoutRef::new(&extents[..1], &strides[..1], 0).unwrap();
    assert_eq!(
        metal::plan_fill_zero_view(DType::F32, rank_nine, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_materialize_strided(DType::F32, rank_nine, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_contiguous_to_view(DType::F32, rank_nine, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_strided_to_strided(DType::F32, rank_nine, 1, rank_nine, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, rank_nine, 1, compact, 1, &[0]),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_broadcast_in_dim(DType::F32, compact, 1, rank_nine, 1, &[0; 9]),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_extract_diagonal(DType::F32, rank_nine, 1, compact, 1, 0, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_embed_diagonal_copy(DType::F32, compact, 1, rank_nine, 1, 0, 0),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_tril_view(DType::F32, 1, rank_nine, 1),
        Err(Error::RankOutOfRange)
    );
    assert_eq!(
        metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            compact,
            1,
            rank_nine,
            1,
            0
        ),
        Err(Error::RankOutOfRange)
    );
    // A rank-8 input cannot embed into a rank-9 output.
    let eight = [1i64; 8];
    let rank_eight = LayoutRef::new(&eight, &eight, 0).unwrap();
    assert_eq!(
        metal::plan_embed_diagonal_copy(DType::F32, rank_eight, 1, rank_nine, 1, 0, 0),
        Err(Error::RankOutOfRange)
    );
    // The rank-8 layouts themselves still plan.
    assert_eq!(
        metal::plan_materialize_strided(DType::F32, rank_eight, 1)
            .unwrap()
            .bytes,
        4
    );
}

#[test]
fn a_reduction_axis_must_fit_the_pointer_walk() {
    // The plane lane steps by the threadgroup width, so the axis is part of the 32-bit
    // position bound: one element past u32::MAX is refused whatever the stride.
    let rows = 1i64;
    let huge = u64::from(u32::MAX) + 1;
    let in_extents = [rows, huge as i64];
    let in_strides = [1i64, 0];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let out_extents = [rows, 1];
    let out_strides = [1i64, 1];
    let out_layout = LayoutRef::new(&out_extents, &out_strides, 0).unwrap();
    assert_eq!(
        metal::plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            out_layout,
            1,
            in_layout,
            1,
            1
        ),
        Err(Error::LaunchLimit)
    );
    // At the limit it plans.
    let in_extents = [rows, u32::MAX as i64];
    let in_layout = LayoutRef::new(&in_extents, &in_strides, 0).unwrap();
    let plan = metal::plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        out_layout,
        1,
        in_layout,
        1,
        1,
    )
    .unwrap();
    assert_eq!(plan.dispatch.threadgroups, 1, "one threadgroup per output");
}
