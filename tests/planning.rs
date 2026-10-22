//! Host-side validation, planning and manifest consistency. Needs no CUDA.

use t4a_cuda_kernels::{
    bytes, plan_contiguous_to_view, plan_copy_exact, plan_elementwise, plan_fill_zero,
    plan_fill_zero_view, plan_materialize_strided, plan_strided_to_strided, plan_tiled_transpose,
    CLayout, CTransposeConfig, DType, Entry, Error, Launch, Layout, TransposeConfig, ABI_REVISION,
    DEFAULT_SHARED_BYTES, FLAT_BLOCK, MAX_RANK,
};

#[test]
fn abi_revision_matches_the_c_header() {
    let header = include_str!("../cuda/include/t4a/abi.cuh");
    let declared = header
        .lines()
        .find(|line| line.starts_with("#define T4A_CUDA_ABI_REVISION "))
        .expect("the header declares the ABI revision")
        .rsplit(' ')
        .next()
        .expect("a numeric revision");
    assert_eq!(
        declared.trim().parse::<u32>().unwrap(),
        ABI_REVISION,
        "the C header and the Rust crate must declare the same revision"
    );
}

#[test]
fn abi_revision_is_documented() {
    let abi = include_str!("../docs/abi/c-kernel-abi.md");
    assert!(abi.contains(&format!("revision {ABI_REVISION}")) || abi.contains("Revision"));
    assert!(abi.contains(&format!("T4A_CUDA_ABI_REVISION == {ABI_REVISION}")));
}

#[test]
fn layout_rejects_bad_inputs() {
    assert_eq!(Layout::new(&[1; 9], &[1; 9], 0), Err(Error::RankOutOfRange));
    assert_eq!(Layout::new(&[1, 2], &[1], 0), Err(Error::RankLength));
    assert_eq!(Layout::new(&[-1], &[1], 0), Err(Error::NegativeExtent));
}

#[test]
fn rank_zero_is_a_scalar_view() {
    let layout = Layout::new(&[], &[], 7).unwrap();
    assert_eq!(layout.rank(), 0);
    assert_eq!(layout.element_count().unwrap(), 1);
    assert_eq!(layout.min_reachable_offset().unwrap(), 7);
    assert_eq!(layout.max_reachable_offset().unwrap(), 7);
    assert!(layout.rejects_broadcast_axis());
    let launch = plan_fill_zero_view(DType::F64, &layout, 8).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
}

#[test]
fn compact_layout_has_axis_zero_fastest() {
    let layout = Layout::compact(&[2, 3, 4]).unwrap();
    assert_eq!(layout.strides(), &[1, 2, 6]);
    assert_eq!(layout.element_count().unwrap(), 24);
    assert_eq!(layout.max_reachable_offset().unwrap(), 23);
}

#[test]
fn negative_strides_report_the_reachable_span() {
    let layout = Layout::new(&[3], &[-2], 4).unwrap();
    assert_eq!(layout.min_reachable_offset().unwrap(), 0);
    assert_eq!(layout.max_reachable_offset().unwrap(), 4);
    // A three-element reversed view at offset 2 spans [0, 2].
    let reversed = Layout::new(&[3], &[-1], 2).unwrap();
    assert_eq!(reversed.min_reachable_offset().unwrap(), 0);
    assert_eq!(reversed.max_reachable_offset().unwrap(), 2);
}

#[test]
fn extent_product_overflow_is_rejected() {
    let layout = Layout::new(&[i64::MAX, 2], &[1, 1], 0).unwrap();
    assert_eq!(layout.element_count(), Err(Error::Overflow));
    assert_eq!(Layout::compact(&[i64::MAX, 2]), Err(Error::Overflow));
}

#[test]
fn broadcast_axis_is_not_a_destination() {
    let broadcast = Layout::new(&[3, 4], &[1, 0], 0).unwrap();
    assert!(!broadcast.rejects_broadcast_axis());
    assert_eq!(
        plan_contiguous_to_view(DType::F32, &broadcast, 12),
        Err(Error::BroadcastDestination)
    );
    // A broadcast read is fine.
    assert!(plan_materialize_strided(DType::F32, &broadcast, 12).is_ok());
}

#[test]
fn fill_zero_plans_one_block_per_256_elements() {
    assert_eq!(plan_fill_zero(DType::F32, 0).unwrap(), Launch::EMPTY);
    assert_eq!(plan_fill_zero(DType::F32, 1).unwrap().grid, [1, 1, 1]);
    assert_eq!(plan_fill_zero(DType::F32, 256).unwrap().grid, [1, 1, 1]);
    assert_eq!(plan_fill_zero(DType::F32, 257).unwrap().grid, [2, 1, 1]);
    let launch = plan_fill_zero(DType::F32, 1_000_000).unwrap();
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);
    assert_eq!(u64::from(launch.grid[0]), 1_000_000u64.div_ceil(256));
    assert_eq!(plan_fill_zero(DType::F32, -1), Err(Error::NegativeCount));
}

#[test]
fn flat_plan_rejects_negative_and_oversized_counts() {
    let too_many = i64::from(i32::MAX) * i64::from(FLAT_BLOCK) + 1;
    assert_eq!(
        plan_fill_zero(DType::F32, too_many),
        Err(Error::LaunchLimit)
    );
}

#[test]
fn view_entries_derive_the_count_from_the_extents() {
    // Three logical elements, spaced by two, inside a five-element allocation.
    let layout = Layout::new(&[3], &[2], 0).unwrap();
    let launch = plan_fill_zero_view(DType::F32, &layout, 5).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    let launch = plan_materialize_strided(DType::F32, &layout, 5).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
}

#[test]
fn view_entries_reject_out_of_span_layouts() {
    // Three elements spaced by two reach element 4, past a four-element span.
    let layout = Layout::new(&[3], &[2], 0).unwrap();
    assert_eq!(
        plan_fill_zero_view(DType::F32, &layout, 4),
        Err(Error::OffsetOutOfSpan)
    );
    let sparse = Layout::new(&[3], &[3], 0).unwrap();
    let compact = Layout::new(&[3], &[1], 0).unwrap();
    assert_eq!(
        plan_strided_to_strided(DType::F32, &compact, 3, &sparse, 3),
        Err(Error::OffsetOutOfSpan)
    );
    // A negative reachable offset is out of span too.
    let negative = Layout::new(&[3], &[-1], 1).unwrap();
    assert_eq!(
        plan_contiguous_to_view(DType::F32, &negative, 3),
        Err(Error::OffsetOutOfSpan)
    );
    // A destination that repeats an address is a data race even when every
    // write stores the same value, so a broadcast destination is rejected.
    let broadcast = Layout::new(&[3], &[0], 0).unwrap();
    assert_eq!(
        plan_fill_zero_view(DType::F32, &broadcast, 1),
        Err(Error::BroadcastDestination)
    );
    // The destination side is checked as well.
    assert_eq!(
        plan_strided_to_strided(DType::F32, &compact, 3, &sparse, 6),
        Err(Error::OffsetOutOfSpan)
    );
}

#[test]
fn strided_to_strided_requires_equal_extents() {
    let src = Layout::compact(&[2, 2]).unwrap();
    let dst = Layout::compact(&[4]).unwrap();
    assert_eq!(
        plan_strided_to_strided(DType::F32, &src, 4, &dst, 4),
        Err(Error::CountMismatch)
    );
}

#[test]
fn binary_entries_follow_the_dtype_matrix() {
    assert_eq!(
        plan_elementwise(Entry::Add, DType::Bool, 4),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        plan_elementwise(Entry::Div, DType::I32, 4),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        plan_elementwise(Entry::Div, DType::C64, 4),
        Err(Error::UnsupportedDType)
    );
    for dtype in [DType::F32, DType::F64, DType::I32, DType::I64] {
        assert!(plan_elementwise(Entry::Add, dtype, 4).is_ok());
        assert!(plan_elementwise(Entry::Mul, dtype, 4).is_ok());
    }
    assert!(plan_elementwise(Entry::Sub, DType::C32, 4).is_ok());
    assert_eq!(
        plan_elementwise(Entry::TiledTranspose, DType::F32, 4),
        Err(Error::NotElementwise)
    );
    assert_eq!(plan_copy_exact(DType::Bool, 0).unwrap(), Launch::EMPTY);
}

#[test]
fn transpose_configuration_range() {
    assert!(TransposeConfig::new(12, 8, 1, 1).is_err());
    assert!(TransposeConfig::new(16, 3, 1, 1).is_err());
    assert!(TransposeConfig::new(16, 8, 2, 1).is_err());
    assert!(TransposeConfig::new(16, 8, 1, 8).is_err());
    assert!(TransposeConfig::new(8, 16, 0, 4).is_ok());
    assert_eq!(TransposeConfig::DEFAULT.tile(), 16);
}

#[test]
fn transpose_plans_grid_shared_memory_and_batches() {
    let config = TransposeConfig::DEFAULT;
    let span = 512 * 1024 * 3;
    let launch =
        plan_tiled_transpose(DType::F64, config, 0, 512 * 1024, 512, 1024, 3, span, span).unwrap();
    assert_eq!(launch.grid, [32, 64, 3]);
    assert_eq!(launch.block, [16, 8, 1]);
    assert_eq!(launch.shared_bytes, 16 * 17 * 8);

    // Vector width splits the x block.
    let wide = TransposeConfig::new(16, 8, 1, 2).unwrap();
    let span = 512 * 1024;
    let launch =
        plan_tiled_transpose(DType::F32, wide, 0, 512 * 1024, 512, 1024, 1, span, span).unwrap();
    assert_eq!(launch.block, [8, 8, 1]);

    // Ragged extents round up.
    let launch = plan_tiled_transpose(DType::F32, config, 0, 5 * 7, 5, 7, 1, 35, 35).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
}

#[test]
fn transpose_rejects_bad_batches_and_limits() {
    let config = TransposeConfig::DEFAULT;
    // batch_stride must be the matrix product.
    assert_eq!(
        plan_tiled_transpose(DType::F32, config, 0, 5, 2, 3, 1, 6, 6),
        Err(Error::InvalidTransposeConfig)
    );
    assert_eq!(
        plan_tiled_transpose(DType::F32, config, -1, 6, 2, 3, 1, 6, 6),
        Err(Error::NegativeCount)
    );
    // The source span must include every batch.
    assert_eq!(
        plan_tiled_transpose(DType::F32, config, 1, 6, 2, 3, 1, 6, 6),
        Err(Error::OffsetOutOfSpan)
    );
    // grid.y is capped at 65535, while grid.x may use the full 2^31-1 range.
    let wide_rows = 65_535 * 16 + 1;
    assert_eq!(
        plan_tiled_transpose(
            DType::F32,
            config,
            0,
            wide_rows,
            1,
            wide_rows,
            1,
            wide_rows,
            wide_rows
        ),
        Err(Error::LaunchLimit)
    );
    let long_columns = 65_535 * 16 + 1;
    let plan = plan_tiled_transpose(
        DType::F32,
        config,
        0,
        long_columns,
        long_columns,
        1,
        1,
        long_columns,
        long_columns,
    )
    .unwrap();
    assert_eq!(plan.grid, [65_536, 1, 1]);
    // A zero extent plans no launch, but keeps the configuration's block.
    let empty = plan_tiled_transpose(DType::F32, config, 0, 0, 0, 8, 4, 0, 0).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.block, [16, 8, 1]);
    // Complex64 with the largest tile still fits the shared-memory budget.
    let big = TransposeConfig::new(32, 8, 1, 1).unwrap();
    assert!(big.shared_bytes(DType::C64.size()).unwrap() <= DEFAULT_SHARED_BYTES);
}

#[test]
fn byte_lengths_are_checked() {
    assert_eq!(bytes(DType::C64, 3).unwrap(), 48);
    assert_eq!(bytes(DType::Bool, 0).unwrap(), 0);
    assert_eq!(bytes(DType::F32, -1), Err(Error::NegativeCount));
    assert_eq!(bytes(DType::C64, i64::MAX), Err(Error::Overflow));
}

#[test]
fn c_layout_mirror_round_trips() {
    let layout = Layout::new(&[2, 3], &[1, 2], 4).unwrap();
    let mirror = CLayout::new(&layout);
    assert_eq!(mirror.rank, 2);
    assert_eq!(mirror.reserved, 0);
    assert_eq!(mirror.offset, 4);
    assert_eq!(&mirror.extents[..3], &[2, 3, 0]);
    assert_eq!(&mirror.strides[..3], &[1, 2, 0]);
    assert_eq!(std::mem::size_of::<CLayout>(), 144);
    assert_eq!(std::mem::size_of::<CTransposeConfig>(), 16);
    let config = TransposeConfig::DEFAULT;
    let c_config: CTransposeConfig = config.into();
    assert_eq!(c_config.tile, 16);
    assert_eq!(c_config.vector_width, 1);
}

#[test]
fn entry_names_are_unique_and_exported_by_the_sources() {
    let sources = concat!(
        include_str!("../cuda/src/movement.cu"),
        include_str!("../cuda/src/arithmetic.cu"),
        include_str!("../cuda/src/diagonal.cu"),
        include_str!("../cuda/src/window.cu"),
        include_str!("../cuda/src/validation.cu"),
        include_str!("../cuda/src/reduce.cu"),
        include_str!("../cuda/src/gather_scatter.cu"),
        include_str!("../cuda/src/convert.cu"),
        include_str!("../cuda/src/elementwise.cu"),
        include_str!("../cuda/src/linalg_aux.cu"),
        include_str!("../cuda/src/rrqr.cu")
    );
    for (index, entry) in Entry::ALL.iter().enumerate() {
        let name = entry.name();
        assert!(name.starts_with("t4a_cuda_"), "{name}");
        assert!(
            Entry::ALL
                .iter()
                .skip(index + 1)
                .all(|other| other.name() != name),
            "duplicate {name}"
        );
        let declaration = format!("extern \"C\" __global__ void {name}(");
        assert!(sources.contains(&declaration), "{name} is not exported");
    }
    // Nothing else is exported: the source declares exactly these entries.
    assert_eq!(
        sources.matches("extern \"C\" __global__ void ").count(),
        Entry::ALL.len()
    );
}

#[test]
fn entry_dtype_matrix_matches_the_abi_document() {
    let abi = include_str!("../docs/abi/c-kernel-abi.md");
    for entry in Entry::ALL {
        assert!(
            abi.contains(&format!("`{}`", entry.name())),
            "{}",
            entry.name()
        );
    }
    assert_eq!(Entry::Div.supported_dtypes(), &[DType::F32, DType::F64]);
    assert_eq!(
        Entry::ValidateRealCast.supported_dtypes(),
        &[DType::F32, DType::F64]
    );
    assert_eq!(Entry::Pad.supported_dtypes(), &DType::ALL);
    assert_eq!(Entry::Add.supported_dtypes().len(), 6);
    assert_eq!(Entry::FillZero.supported_dtypes(), &DType::ALL);
    assert_eq!(MAX_RANK, 8);
    assert!((0..7).all(|tag| DType::from_tag(tag).map(DType::tag) == Some(tag)));
    assert_eq!(DType::from_tag(7), None);
}

#[test]
fn empty_views_need_no_span() {
    // A zero extent addresses nothing, so a leading zero axis must not be
    // rejected by the reachability check.
    let layout = Layout::new(&[0, 3], &[1, 1], 0).unwrap();
    assert!(plan_fill_zero_view(DType::F32, &layout, 0)
        .unwrap()
        .is_empty());
    assert!(plan_materialize_strided(DType::F32, &layout, 0)
        .unwrap()
        .is_empty());
    assert!(plan_contiguous_to_view(DType::F32, &layout, 0)
        .unwrap()
        .is_empty());
}

#[test]
fn offsets_must_fit_a_64_bit_byte_address() {
    // A single element offset that is representable as an element count but not
    // as `offset * element_size` bytes would overflow device address arithmetic.
    let offset = i64::MAX / 16 + 1;
    let layout = Layout::new(&[1], &[1], offset).unwrap();
    let allocation = offset + 1;
    assert_eq!(
        plan_fill_zero_view(DType::C64, &layout, allocation),
        Err(Error::Overflow)
    );
    // The same element offset is fine for a four-byte element.
    assert!(plan_fill_zero_view(DType::F32, &layout, allocation).is_ok());
}

// --- revision 2: diagonal, windowing and validation ------------------------

use t4a_cuda_kernels::{
    plan_dynamic_slice, plan_embed_diagonal_copy, plan_extract_diagonal,
    plan_extract_invalid_float_index, plan_extract_invalid_real_cast, plan_init_validation,
    plan_pad, plan_slice, plan_tril, plan_triu, plan_validate_float_indices,
    plan_validate_real_cast, CAxisValues,
};

#[test]
fn extract_diagonal_derives_the_output_layout() {
    let input = Layout::compact(&[2, 3, 4]).unwrap();
    // axes (0, 2): remove axis 2, diagonal length min(2, 4) = 2 at axis 0.
    let output = Layout::compact(&[2, 3]).unwrap();
    assert!(plan_extract_diagonal(DType::F32, &input, 24, &output, 6, 0, 2).is_ok());
    // The diagonal output position is derived, so a different supplied shape is
    // refused instead of reinterpreted.
    let wrong = Layout::compact(&[3, 2]).unwrap();
    assert_eq!(
        plan_extract_diagonal(DType::F32, &input, 24, &wrong, 6, 0, 2),
        Err(Error::ShapeMismatch)
    );
    // axes (2, 0) put the diagonal at output position 1: extents [3, 2].
    let output = Layout::compact(&[3, 2]).unwrap();
    assert!(plan_extract_diagonal(DType::F32, &input, 24, &output, 6, 2, 0).is_ok());
    assert_eq!(
        plan_extract_diagonal(DType::F32, &input, 24, &output, 6, 0, 0),
        Err(Error::DuplicateAxis)
    );
    assert_eq!(
        plan_extract_diagonal(DType::F32, &input, 24, &output, 6, 0, 3),
        Err(Error::InvalidAxis)
    );
    let strided_out = Layout::new(&[2, 3], &[1, 4], 0).unwrap();
    assert_eq!(
        plan_extract_diagonal(DType::F32, &input, 24, &strided_out, 12, 0, 2),
        Err(Error::NotCompact)
    );
}

#[test]
fn embed_diagonal_allows_an_insertion_position_at_the_end() {
    let input = Layout::compact(&[2, 3]).unwrap();
    let output = Layout::compact(&[2, 3, 3]).unwrap();
    // axis_a = 1 (input), axis_b = 2 (insertion at the end), equal axes allowed.
    assert!(plan_embed_diagonal_copy(DType::F32, &input, 6, &output, 18, 1, 2).is_ok());
    assert!(plan_embed_diagonal_copy(DType::F32, &input, 6, &output, 18, 1, 3).is_err());
    let same = Layout::compact(&[2, 3, 3]).unwrap();
    assert!(plan_embed_diagonal_copy(DType::F32, &input, 6, &same, 18, 1, 1).is_ok());
    let strided_in = Layout::new(&[2, 3], &[1, 4], 0).unwrap();
    assert_eq!(
        plan_embed_diagonal_copy(DType::F32, &strided_in, 12, &output, 18, 1, 2),
        Err(Error::NotCompact)
    );
    let wrong = Layout::compact(&[2, 3]).unwrap();
    assert_eq!(
        plan_embed_diagonal_copy(DType::F32, &input, 6, &wrong, 6, 1, 2),
        Err(Error::ShapeMismatch)
    );
    // A destination that repeats an address is a data race.
    let broadcast_out = Layout::new(&[2, 3, 3], &[0, 0, 0], 0).unwrap();
    assert_eq!(
        plan_embed_diagonal_copy(DType::F32, &input, 6, &broadcast_out, 1, 1, 2),
        Err(Error::BroadcastDestination)
    );
}

#[test]
fn triangular_plans_accept_batches_and_extreme_offsets() {
    assert_eq!(plan_tril(DType::F32, 3, 4, 12).unwrap().grid, [1, 1, 1]);
    // Three 3x4 matrices: the mask repeats per batch.
    assert!(plan_tril(DType::F32, 3, 4, 36).is_ok());
    assert_eq!(plan_tril(DType::F32, 3, 4, 13), Err(Error::ShapeMismatch));
    assert!(plan_tril(DType::F32, 0, 4, 0).is_ok());
    assert_eq!(plan_triu(DType::F32, 2, 2, -1), Err(Error::NegativeCount));
    assert!(plan_triu(DType::C64, 2, 2, 4).is_ok());
}

#[test]
fn slice_requires_the_reachable_count() {
    let input = Layout::compact(&[10, 4]).unwrap();
    let output = Layout::compact(&[5, 2]).unwrap();
    // starts (0, 0), steps (2, 2): ceil(10/2) = 5 and ceil(4/2) = 2.
    assert!(plan_slice(DType::F32, &output, 10, &input, 40, &[0, 0], &[2, 2]).is_ok());
    assert_eq!(
        plan_slice(DType::F32, &output, 10, &input, 40, &[0, 0], &[0, 2]),
        Err(Error::InvalidStep)
    );
    assert_eq!(
        plan_slice(DType::F32, &output, 10, &input, 40, &[11, 0], &[2, 2]),
        Err(Error::OffsetOutOfSpan)
    );
    // A shorter output is the caller's slice limit and stays valid.
    let shorter = Layout::compact(&[4, 2]).unwrap();
    assert!(plan_slice(DType::F32, &shorter, 8, &input, 40, &[0, 0], &[2, 2]).is_ok());
    // Longer than the reachable count is not.
    let wrong_output = Layout::compact(&[6, 2]).unwrap();
    assert_eq!(
        plan_slice(DType::F32, &wrong_output, 12, &input, 40, &[0, 0], &[2, 2]),
        Err(Error::ShapeMismatch)
    );
    let strided_out = Layout::new(&[5, 2], &[1, 7], 0).unwrap();
    assert_eq!(
        plan_slice(DType::F32, &strided_out, 20, &input, 40, &[0, 0], &[2, 2]),
        Err(Error::NotCompact)
    );
}

#[test]
fn dynamic_slice_checks_windows_and_index_dtype() {
    let input = Layout::compact(&[10, 4]).unwrap();
    let output = Layout::compact(&[3, 4]).unwrap();
    assert!(plan_dynamic_slice(DType::F32, DType::I64, &output, 12, &input, 40, 2).is_ok());
    assert!(plan_dynamic_slice(DType::F32, DType::F64, &output, 12, &input, 40, 2).is_ok());
    assert_eq!(
        plan_dynamic_slice(DType::F32, DType::Bool, &output, 12, &input, 40, 2),
        Err(Error::UnsupportedDType)
    );
    let too_wide = Layout::compact(&[11, 4]).unwrap();
    assert_eq!(
        plan_dynamic_slice(DType::F32, DType::I64, &too_wide, 44, &input, 40, 2),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_dynamic_slice(DType::F32, DType::I64, &output, 12, &input, 40, 3),
        Err(Error::RankLength)
    );
}

#[test]
fn pad_derives_the_padded_extents() {
    let input = Layout::compact(&[3, 4]).unwrap();
    // low (1, 0), high (2, 1), interior (0, 1): [3 + 0 + 3, 4 + 1 + 4] with
    // interior spacing 2 giving (4-1)*2 + 1 = 7.
    let output = Layout::compact(&[6, 8]).unwrap();
    assert!(plan_pad(
        DType::F32,
        &output,
        48,
        &input,
        12,
        &[1, 0],
        &[2, 1],
        &[0, 1]
    )
    .is_ok());
    assert_eq!(
        plan_pad(
            DType::F32,
            &output,
            48,
            &input,
            12,
            &[1, 0],
            &[2, 1],
            &[0, -1]
        ),
        Err(Error::InvalidPadding)
    );
    let wrong = Layout::compact(&[6, 7]).unwrap();
    assert_eq!(
        plan_pad(
            DType::F32,
            &wrong,
            42,
            &input,
            12,
            &[1, 0],
            &[2, 1],
            &[0, 1]
        ),
        Err(Error::ShapeMismatch)
    );
    // A negative edge low is allowed and reduces the output extent.
    let negative = Layout::compact(&[2, 4]).unwrap();
    assert!(plan_pad(
        DType::F32,
        &negative,
        8,
        &input,
        12,
        &[-1, 0],
        &[0, 0],
        &[0, 0]
    )
    .is_ok());
    // An empty input axis pads from zero.
    let empty = Layout::compact(&[0]).unwrap();
    let padded = Layout::compact(&[2]).unwrap();
    assert!(plan_pad(DType::F32, &padded, 2, &empty, 0, &[1], &[1], &[5]).is_ok());
}

#[test]
fn validation_plans_cap_the_domain_and_check_stride() {
    assert_eq!(plan_init_validation(DType::F32).unwrap().block, [1, 1, 1]);
    assert_eq!(
        plan_init_validation(DType::I32),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        plan_validate_float_indices(DType::F64, i64::from(u32::MAX) + 1),
        Err(Error::DomainTooLarge)
    );
    assert!(plan_validate_float_indices(DType::F64, i64::from(u32::MAX)).is_ok());
    assert_eq!(
        plan_validate_float_indices(DType::I32, 4),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        plan_extract_invalid_float_index(DType::F32).unwrap().block,
        [1, 1, 1]
    );
    assert_eq!(
        plan_validate_real_cast(DType::F32, 8, 0),
        Err(Error::InvalidStride)
    );
    // Sampling every second element halves the launch domain.
    let launch = plan_validate_real_cast(DType::F32, 1_000, 2).unwrap();
    assert_eq!(u64::from(launch.grid[0]), 500u64.div_ceil(256));
    // The cap applies to the sample count, so a buffer twice the cap is fine at
    // stride 2 while one sample more than the cap is not.
    assert!(plan_validate_real_cast(DType::F32, 2 * i64::from(u32::MAX), 2).is_ok());
    assert_eq!(
        plan_validate_real_cast(DType::F32, 2 * i64::from(u32::MAX) + 1, 2),
        Err(Error::DomainTooLarge)
    );
    // A tiny buffer with a huge stride is a legal one-sample plan.
    assert!(plan_validate_real_cast(DType::F64, 3, i64::MAX).is_ok());
    assert_eq!(
        plan_extract_invalid_real_cast(DType::F64).unwrap().grid,
        [1, 1, 1]
    );
}

#[test]
fn c_axis_values_mirror_layout() {
    let mirror = CAxisValues::new(&[1, -2, 3]).unwrap();
    assert_eq!(mirror.rank, 3);
    assert_eq!(&mirror.values[..4], &[1, -2, 3, 0]);
    assert_eq!(std::mem::size_of::<CAxisValues>(), 72);
    assert_eq!(CAxisValues::new(&[0; 9]), Err(Error::RankOutOfRange));
}

// --- revision 3: reductions -------------------------------------------------

use t4a_cuda_kernels::{
    plan_reduce, plan_reduce_partials, plan_reduction, reduction_partials_layout, split_policy,
    ReduceOp, ReduceStrategy, ReductionPlan, Split, MAX_SPLITS,
};

fn reduced_layouts(extents: &[i64], axis: usize) -> (Layout, Layout) {
    let input = Layout::compact(extents).unwrap();
    let mut out_extents = extents.to_vec();
    out_extents[axis] = 1;
    (input, Layout::compact(&out_extents).unwrap())
}

#[test]
fn split_policy_matches_the_baseline() {
    // min_chunk = plane_width * 8 = 256 at width 32.
    assert_eq!(split_policy(255, 32), Ok(None));
    assert_eq!(split_policy(256, 32), Ok(None));
    assert_eq!(
        split_policy(512, 32),
        Ok(Some(Split {
            splits: 2,
            chunk_len: 256
        }))
    );
    assert_eq!(
        split_policy(513, 32),
        Ok(Some(Split {
            splits: 2,
            chunk_len: 256
        }))
    );
    assert_eq!(
        split_policy(1000, 32),
        Ok(Some(Split {
            splits: 3,
            chunk_len: 333
        }))
    );
    // 33 chunks: the combine pass is a plane reduction again.
    assert_eq!(
        split_policy(8448, 32),
        Ok(Some(Split {
            splits: 33,
            chunk_len: 256
        }))
    );
    // 1e7 elements: capped at 4096 chunks, floor chunk length.
    let capped = split_policy(10_000_000, 32).unwrap().unwrap();
    assert_eq!(capped.splits, MAX_SPLITS);
    assert_eq!(capped.chunk_len, 10_000_000 / MAX_SPLITS);
    // Narrower planes split later.
    assert_eq!(
        split_policy(256, 16),
        Ok(Some(Split {
            splits: 2,
            chunk_len: 128
        }))
    );
    assert_eq!(
        split_policy(255, 8),
        Ok(Some(Split {
            splits: 3,
            chunk_len: 85
        }))
    );
    assert_eq!(split_policy(0, 32), Ok(None));
    assert_eq!(split_policy(512, 0), Err(Error::InvalidPlaneWidth));
}

#[test]
fn strategy_boundary_is_the_plane_width() {
    let (input, output) = reduced_layouts(&[32, 3], 0);
    assert!(plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &output,
        3,
        &input,
        96,
        0,
        32
    )
    .is_ok());
    // Exactly one plane is a valid plane reduction.
    assert!(plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &output,
        3,
        &input,
        96,
        0,
        32
    )
    .is_ok());
    // One element short of a plane cannot be a plane reduction.
    let (short_input, short_output) = reduced_layouts(&[31, 3], 0);
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            &short_output,
            3,
            &short_input,
            93,
            0,
            32
        ),
        Err(Error::StrategyMismatch)
    );
    let (input, output) = reduced_layouts(&[33, 3], 0);
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            &output,
            3,
            &input,
            99,
            0,
            32
        ),
        Err(Error::StrategyMismatch)
    );
    assert!(plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &output,
        3,
        &input,
        99,
        0,
        32
    )
    .is_ok());
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            &output,
            3,
            &input,
            99,
            0,
            24
        ),
        Err(Error::InvalidPlaneWidth)
    );
}

#[test]
fn reduction_geometry_and_shapes() {
    let (input, output) = reduced_layouts(&[4, 64, 2], 1);
    // 4 * 2 = 8 outputs, axis 64 > 32: plane, one block per output.
    let launch = plan_reduce(
        DType::F32,
        ReduceOp::Max,
        ReduceStrategy::Plane,
        &output,
        8,
        &input,
        512,
        1,
        32,
    )
    .unwrap();
    assert_eq!(launch.grid, [8, 1, 1]);
    assert_eq!(launch.block, [32, 1, 1]);

    // A short axis uses one thread per output.
    let (input, output) = reduced_layouts(&[4, 4, 2], 1);
    let launch = plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Unit,
        &output,
        8,
        &input,
        32,
        1,
        8,
    )
    .unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);

    // Non-keepdims output, mismatched extents and a zero axis are rejected.
    let (input, _output) = reduced_layouts(&[4, 64], 1);
    let not_keepdims = Layout::compact(&[4]).unwrap();
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            &not_keepdims,
            4,
            &input,
            256,
            1,
            32
        )
        .map(|_| ()),
        Err(Error::RankLength)
    );
    let (input, _output) = reduced_layouts(&[4, 64], 1);
    let wrong = Layout::compact(&[5, 1]).unwrap();
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            &wrong,
            5,
            &input,
            256,
            1,
            32
        ),
        Err(Error::ShapeMismatch)
    );
    let (zero_axis_in, zero_axis_out) = reduced_layouts(&[4, 0], 1);
    assert_eq!(
        plan_reduce(
            DType::F32,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            &zero_axis_out,
            4,
            &zero_axis_in,
            0,
            1,
            32
        ),
        Err(Error::NegativeCount)
    );
    // An empty output on another axis plans no launch.
    let (input, output) = reduced_layouts(&[0, 64], 1);
    let launch = plan_reduce(
        DType::F32,
        ReduceOp::Sum,
        ReduceStrategy::Plane,
        &output,
        0,
        &input,
        0,
        1,
        32,
    )
    .unwrap();
    assert!(launch.is_empty());
}

#[test]
fn reduction_dtype_gates_follow_the_operation() {
    let (input, output) = reduced_layouts(&[64, 2], 0);
    assert_eq!(
        plan_reduce(
            DType::C64,
            ReduceOp::Max,
            ReduceStrategy::Plane,
            &output,
            2,
            &input,
            128,
            0,
            32
        ),
        Err(Error::UnsupportedDType)
    );
    assert_eq!(
        plan_reduce(
            DType::I32,
            ReduceOp::SumSquares,
            ReduceStrategy::Plane,
            &output,
            2,
            &input,
            128,
            0,
            32
        ),
        Err(Error::UnsupportedDType)
    );
    assert!(plan_reduce(
        DType::C64,
        ReduceOp::Prod,
        ReduceStrategy::Plane,
        &output,
        2,
        &input,
        128,
        0,
        32
    )
    .is_ok());
    assert_eq!(ReduceOp::SumSquares.combine(), ReduceOp::Sum);
    assert_eq!(ReduceOp::Max.combine(), ReduceOp::Max);
}

#[test]
fn reduction_plan_applies_the_auto_policy() {
    // Short axis: one direct pass.
    let (input, output) = reduced_layouts(&[64, 4], 0);
    match plan_reduction(DType::F64, ReduceOp::Sum, &output, 4, &input, 256, 0, 32).unwrap() {
        ReductionPlan::Direct { strategy, launch } => {
            assert_eq!(strategy, ReduceStrategy::Plane);
            assert_eq!(launch.grid, [4, 1, 1]);
        }
        other => panic!("expected a direct plan, got {other:?}"),
    }

    // 512-long axis: two partials chunks, then a direct combine (2 <= 32).
    let (input, output) = reduced_layouts(&[512, 4], 0);
    match plan_reduction(DType::F64, ReduceOp::Sum, &output, 4, &input, 2048, 0, 32).unwrap() {
        ReductionPlan::Split {
            split,
            pass1,
            combine_strategy,
            combine_op,
            pass2,
            ..
        } => {
            assert_eq!(
                split,
                Split {
                    splits: 2,
                    chunk_len: 256
                }
            );
            assert_eq!(pass1.grid, [8, 1, 1]);
            assert_eq!(pass1.block, [32, 1, 1]);
            assert_eq!(combine_strategy, ReduceStrategy::Unit);
            assert_eq!(combine_op, ReduceOp::Sum);
            assert_eq!(pass2.grid, [1, 1, 1]);
        }
        other => panic!("expected a split plan, got {other:?}"),
    }

    // 8448-long axis: 33 chunks, so the combine is a plane reduction, and
    // sum-of-squares combines as a plain sum.
    let (input, output) = reduced_layouts(&[8448, 2], 0);
    match plan_reduction(
        DType::F32,
        ReduceOp::SumSquares,
        &output,
        2,
        &input,
        16896,
        0,
        32,
    )
    .unwrap()
    {
        ReductionPlan::Split {
            split,
            combine_strategy,
            combine_op,
            pass2,
            ..
        } => {
            assert_eq!(split.splits, 33);
            assert_eq!(combine_strategy, ReduceStrategy::Plane);
            assert_eq!(combine_op, ReduceOp::Sum);
            assert_eq!(pass2.grid, [2, 1, 1]);
        }
        other => panic!("expected a split plan, got {other:?}"),
    }
}

#[test]
fn partials_plans_enforce_the_floor_split() {
    let (input, _) = reduced_layouts(&[512, 4], 0);
    let partials = Layout::compact(&[2, 4]).unwrap();
    let split = Split {
        splits: 2,
        chunk_len: 256,
    };
    let launch = plan_reduce_partials(
        DType::F32,
        ReduceOp::Sum,
        &partials,
        8,
        &input,
        2048,
        0,
        split,
        32,
    )
    .unwrap();
    assert_eq!(launch.grid, [8, 1, 1]);

    // A chunk shorter than a plane, or not the floor split, is refused: a lane
    // would seed outside its own slice.
    let too_short = Split {
        splits: 64,
        chunk_len: 1,
    };
    assert_eq!(
        plan_reduce_partials(
            DType::F32,
            ReduceOp::Sum,
            &partials,
            8,
            &input,
            2048,
            0,
            too_short,
            32
        ),
        Err(Error::InvalidSplit)
    );
    let not_floor = Split {
        splits: 2,
        chunk_len: 200,
    };
    assert_eq!(
        plan_reduce_partials(
            DType::F32,
            ReduceOp::Sum,
            &partials,
            8,
            &input,
            2048,
            0,
            not_floor,
            32
        ),
        Err(Error::InvalidSplit)
    );
    let single = Split {
        splits: 1,
        chunk_len: 512,
    };
    assert_eq!(
        plan_reduce_partials(
            DType::F32,
            ReduceOp::Sum,
            &partials,
            8,
            &input,
            2048,
            0,
            single,
            32
        ),
        Err(Error::InvalidSplit)
    );
    // The partials axis extent must equal the split count.
    let wrong = Layout::compact(&[3, 4]).unwrap();
    assert_eq!(
        plan_reduce_partials(
            DType::F32,
            ReduceOp::Sum,
            &wrong,
            12,
            &input,
            2048,
            0,
            split,
            32
        ),
        Err(Error::ShapeMismatch)
    );
}

// --- revision 4: gather and scatter -----------------------------------------

use t4a_cuda_kernels::{
    plan_gather, plan_scatter_add, window_dims, CGatherPlan, CScatterPlan, GatherPlan, ScatterPlan,
};

fn gather_case() -> (Layout, Layout, Layout, GatherPlan) {
    // Operand [4, 3]: axis 0 is collapsed, so the window lives on axis 1.
    let operand = Layout::compact(&[4, 3]).unwrap();
    // Index tensor [2, 5]: two batches of one start component.
    let indices = Layout::compact(&[2, 1]).unwrap();
    let out = Layout::compact(&[2, 3]).unwrap();
    let plan = GatherPlan {
        window_dims: window_dims(2, &[0]),
        offset_dims: vec![1],
        start_index_map: vec![0],
        slice_sizes: vec![1, 3],
        index_vector_dim: 1,
    };
    (operand, indices, out, plan)
}

#[test]
fn gather_derives_the_output_and_plans_one_flat_launch() {
    let (operand, indices, out, plan) = gather_case();
    let (launch, output_len) = plan_gather(
        DType::F32,
        DType::I64,
        &out,
        6,
        &operand,
        12,
        &indices,
        2,
        &plan,
    )
    .unwrap();
    assert_eq!(output_len, 6);
    assert_eq!(launch.grid, [1, 1, 1]);

    // A mismatched output layout is refused instead of reinterpreted.
    let wrong = Layout::compact(&[2, 2]).unwrap();
    assert_eq!(
        plan_gather(
            DType::F32,
            DType::I64,
            &wrong,
            4,
            &operand,
            12,
            &indices,
            2,
            &plan
        ),
        Err(Error::ShapeMismatch)
    );
    // The index vector size must match the start map length.
    let bad_vector = GatherPlan {
        index_vector_dim: 0,
        ..plan.clone()
    };
    assert_eq!(
        plan_gather(
            DType::F32,
            DType::I64,
            &out,
            6,
            &operand,
            12,
            &indices,
            2,
            &bad_vector
        ),
        Err(Error::IndexVectorMismatch)
    );
    // A slice larger than the operand axis is refused.
    let big = GatherPlan {
        slice_sizes: vec![1, 5],
        ..plan.clone()
    };
    assert_eq!(
        plan_gather(
            DType::F32,
            DType::I64,
            &out,
            6,
            &operand,
            12,
            &indices,
            2,
            &big
        ),
        Err(Error::ShapeMismatch)
    );
    // Duplicate or out-of-range axes are refused.
    let duplicate = GatherPlan {
        start_index_map: vec![0, 0],
        ..plan.clone()
    };
    assert!(matches!(
        plan_gather(
            DType::F32,
            DType::I64,
            &out,
            6,
            &operand,
            12,
            &indices,
            2,
            &duplicate
        ),
        Err(Error::IndexVectorMismatch) | Err(Error::DuplicateAxis)
    ));
    // Index tensors must be compact: an offset view would need strides the
    // entries do not transport.
    let offset_indices = Layout::new(&[2, 1], &[1, 2], 3).unwrap();
    assert_eq!(
        plan_gather(
            DType::F32,
            DType::I64,
            &out,
            6,
            &operand,
            12,
            &offset_indices,
            8,
            &plan
        ),
        Err(Error::NotCompact)
    );
    // A rank-0 output is a scalar gather of one element.
    let scalar_plan = GatherPlan {
        window_dims: vec![],
        offset_dims: vec![],
        start_index_map: vec![0],
        slice_sizes: vec![1],
        index_vector_dim: 0,
    };
    let scalar_operand = Layout::compact(&[5]).unwrap();
    let scalar_indices = Layout::compact(&[1]).unwrap();
    let scalar_out = Layout::compact(&[]).unwrap();
    let (launch, output_len) = plan_gather(
        DType::F64,
        DType::I32,
        &scalar_out,
        1,
        &scalar_operand,
        5,
        &scalar_indices,
        1,
        &scalar_plan,
    )
    .unwrap();
    assert_eq!(output_len, 1);
    assert_eq!(launch.grid, [1, 1, 1]);
}

#[test]
fn scatter_validates_the_batch_shape_and_ignores_operand_fit() {
    // Operand [6], indices [4, 1], updates [4, 2]: four batches, window 2.
    let out = Layout::compact(&[6]).unwrap();
    let indices = Layout::compact(&[4, 1]).unwrap();
    let updates = Layout::compact(&[4, 2]).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![0],
        update_window_dims: vec![1],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let launch = plan_scatter_add(
        DType::F32,
        DType::I64,
        &out,
        6,
        &indices,
        4,
        &updates,
        8,
        &plan,
    )
    .unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);

    // The batch extents come from the updates non-window axes; a mismatch is a
    // shape error rather than a silent reinterpretation.
    let wrong_updates = Layout::compact(&[3, 2]).unwrap();
    assert_eq!(
        plan_scatter_add(
            DType::F32,
            DType::I64,
            &out,
            6,
            &indices,
            4,
            &wrong_updates,
            6,
            &plan
        ),
        Err(Error::ShapeMismatch)
    );
    // A window larger than the operand axis is allowed: the kernel skips those
    // windows element by element.
    let big_window = Layout::compact(&[4, 9]).unwrap();
    assert!(plan_scatter_add(
        DType::F32,
        DType::I64,
        &out,
        6,
        &indices,
        4,
        &big_window,
        36,
        &plan
    )
    .is_ok());
    // Integer and boolean data are not additive-scatter types.
    assert_eq!(
        plan_scatter_add(
            DType::I32,
            DType::I64,
            &out,
            6,
            &indices,
            4,
            &updates,
            8,
            &plan
        ),
        Err(Error::UnsupportedDType)
    );
    // Index tensors must be compact.
    let offset_indices = Layout::new(&[4, 1], &[1, 2], 1).unwrap();
    assert_eq!(
        plan_scatter_add(
            DType::F32,
            DType::I64,
            &out,
            6,
            &offset_indices,
            8,
            &updates,
            8,
            &plan
        ),
        Err(Error::NotCompact)
    );
}

#[test]
fn gather_and_scatter_mirrors_have_the_documented_layout() {
    let (_, _, _, plan) = gather_case();
    let mirror = CGatherPlan::new(&plan, 2).unwrap();
    assert_eq!(mirror.operand_rank, 2);
    assert_eq!(mirror.out_rank, 2);
    assert_eq!(mirror.window_rank, 1);
    assert_eq!(mirror.start_map_len, 1);
    assert_eq!(mirror.index_vector_dim, 1);
    assert_eq!(&mirror.window_dims[..2], &[1, 0]);
    assert_eq!(&mirror.offset_dims[..2], &[1, 0]);
    assert_eq!(&mirror.slice_sizes[..3], &[1, 3, 0]);
    assert_eq!(std::mem::size_of::<CGatherPlan>(), 280);

    let scatter = ScatterPlan {
        window_dims: vec![0],
        update_window_dims: vec![1],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let mirror = CScatterPlan::new(1, 2, &scatter).unwrap();
    assert_eq!(mirror.window_rank, 1);
    assert_eq!(mirror.scatter_map_len, 1);
    assert_eq!(std::mem::size_of::<CScatterPlan>(), 216);
}

#[test]
fn window_dims_helper_drops_collapsed_axes() {
    assert_eq!(window_dims(3, &[1]), vec![0, 2]);
    assert_eq!(window_dims(2, &[]), vec![0, 1]);
    assert_eq!(window_dims(1, &[0]), Vec::<u32>::new());
}

#[test]
fn gather_requires_unit_slices_on_collapsed_axes() {
    // Operand [5]: a start map on axis 0 with a collapsed (non-window) axis must
    // still carry slice size 1, or the kernel would read outside the operand.
    let operand = Layout::compact(&[5]).unwrap();
    let indices = Layout::compact(&[1]).unwrap();
    let out = Layout::compact(&[]).unwrap();
    let bad = GatherPlan {
        window_dims: vec![],
        offset_dims: vec![],
        start_index_map: vec![0],
        slice_sizes: vec![0],
        index_vector_dim: 0,
    };
    assert_eq!(
        plan_gather(
            DType::F32,
            DType::I64,
            &out,
            1,
            &operand,
            5,
            &indices,
            5,
            &bad
        ),
        Err(Error::ShapeMismatch)
    );
    let good = GatherPlan {
        slice_sizes: vec![1],
        ..bad
    };
    assert!(plan_gather(
        DType::F32,
        DType::I64,
        &out,
        1,
        &operand,
        5,
        &indices,
        5,
        &good
    )
    .is_ok());
    // Gather moves Bool data.
    assert_eq!(Entry::Gather.supported_dtypes(), &DType::ALL);
}

#[test]
fn scatter_with_an_empty_destination_plans_nothing() {
    let out = Layout::compact(&[0]).unwrap();
    let indices = Layout::compact(&[1]).unwrap();
    let updates = Layout::compact(&[1]).unwrap();
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 1,
    };
    let launch = plan_scatter_add(
        DType::F32,
        DType::I64,
        &out,
        0,
        &indices,
        1,
        &updates,
        1,
        &plan,
    )
    .unwrap();
    assert!(launch.is_empty());
}

// --- revision 5: conversions ------------------------------------------------

use t4a_cuda_kernels::plan_convert;

#[test]
fn convert_accepts_every_unequal_pair_and_rejects_identity() {
    let mut accepted = 0;
    for dst in DType::ALL {
        for src in DType::ALL {
            let plan = plan_convert(dst, src, 257);
            if dst == src {
                assert_eq!(plan, Err(Error::ShapeMismatch), "{dst:?} -> {src:?}");
            } else {
                let plan = plan.unwrap();
                assert_eq!(plan.grid, [2, 1, 1]);
                assert_eq!(plan.block, [256, 1, 1]);
                accepted += 1;
            }
        }
    }
    assert_eq!(accepted, 42, "all unequal pairs of the seven dtypes");
    assert_eq!(
        plan_convert(DType::F32, DType::F64, -1),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        plan_convert(DType::F32, DType::F64, 0).unwrap(),
        Launch::EMPTY
    );
    let too_many = i64::from(i32::MAX) * i64::from(FLAT_BLOCK) + 1;
    assert_eq!(
        plan_convert(DType::F32, DType::F64, too_many),
        Err(Error::LaunchLimit)
    );
    assert_eq!(Entry::Convert.supported_dtypes(), &DType::ALL);
}

// --- revision 6: elementwise, comparison and strided combination -----------

use t4a_cuda_kernels::{
    plan_binary, plan_clamp, plan_combined_strided, plan_compare, plan_select, plan_unary,
    BinaryOp, CompareOp, ScalarSide, StridedOp, UnaryOp,
};

const UNARY_OPS: [UnaryOp; 14] = [
    UnaryOp::Neg,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tanh,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Expm1,
    UnaryOp::Log1p,
    UnaryOp::Erf,
    UnaryOp::Abs,
    UnaryOp::Sign,
    UnaryOp::Conj,
];

const BINARY_OPS: [BinaryOp; 14] = [
    BinaryOp::Max,
    BinaryOp::Min,
    BinaryOp::Pow,
    BinaryOp::Rem,
    BinaryOp::Mul,
    BinaryOp::DivComplex,
    BinaryOp::DivChecked,
    BinaryOp::RemChecked,
    BinaryOp::PowChecked,
    BinaryOp::MixedAdd,
    BinaryOp::MixedSub,
    BinaryOp::MixedMul,
    BinaryOp::MixedDiv,
    BinaryOp::Div,
];

const SIDES: [ScalarSide; 3] = [ScalarSide::Both, ScalarSide::Left, ScalarSide::Right];

#[test]
fn revision_six_tags_are_stable_and_unique() {
    for (tag, op) in UNARY_OPS.iter().enumerate() {
        assert_eq!(op.tag(), tag as i32, "{op:?}");
    }
    for (tag, op) in BINARY_OPS.iter().enumerate() {
        assert_eq!(op.tag(), tag as i32, "{op:?}");
    }
    assert_eq!(StridedOp::Mul.tag(), 0);
    assert_eq!(StridedOp::Axpby.tag(), 1);
    assert_eq!(ScalarSide::Both.tag(), 0);
    assert_eq!(ScalarSide::Left.tag(), 1);
    assert_eq!(ScalarSide::Right.tag(), 2);
    let compares = [
        (CompareOp::Eq, 0),
        (CompareOp::Lt, 1),
        (CompareOp::Le, 2),
        (CompareOp::Gt, 3),
        (CompareOp::Ge, 4),
    ];
    for (op, tag) in compares {
        assert_eq!(op.tag(), tag);
    }
}

#[test]
fn unary_accepts_exactly_the_documented_pairs() {
    let pairs = |op: UnaryOp| {
        let mut accepted = DType::ALL
            .iter()
            .flat_map(|dst| DType::ALL.iter().map(move |src| (*dst, *src)))
            .filter(|(dst, src)| plan_unary(op, *dst, *src, 3).is_ok())
            .collect::<Vec<_>>();
        accepted.sort_by_key(|(dst, src)| (dst.tag(), src.tag()));
        accepted
    };
    let same = |dtypes: &[DType]| {
        let mut pairs = dtypes
            .iter()
            .map(|dtype| (*dtype, *dtype))
            .collect::<Vec<_>>();
        pairs.sort_by_key(|(dst, src)| (dst.tag(), src.tag()));
        pairs
    };

    let real_or_complex = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::C32,
        DType::C64,
    ];
    for op in [UnaryOp::Neg, UnaryOp::Sign] {
        assert_eq!(pairs(op), same(&real_or_complex), "{op:?}");
    }
    for op in [
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tanh,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Expm1,
        UnaryOp::Log1p,
        UnaryOp::Erf,
    ] {
        assert_eq!(pairs(op), same(&[DType::F32, DType::F64]), "{op:?}");
    }
    let mut magnitude = same(&[DType::F32, DType::F64, DType::I32, DType::I64])
        .into_iter()
        .chain([(DType::F32, DType::C32), (DType::F64, DType::C64)])
        .collect::<Vec<_>>();
    magnitude.sort_by_key(|(dst, src)| (dst.tag(), src.tag()));
    assert_eq!(
        pairs(UnaryOp::Abs),
        magnitude,
        "complex magnitude is the only unequal pair"
    );
    assert_eq!(pairs(UnaryOp::Conj), same(&[DType::C32, DType::C64]));
    // A same-dtype complex ABS is not an operation.
    assert_eq!(
        plan_unary(UnaryOp::Abs, DType::C32, DType::C32, 4),
        Err(Error::UnsupportedOp)
    );
    assert_eq!(
        plan_unary(UnaryOp::Neg, DType::Bool, DType::Bool, 4),
        Err(Error::UnsupportedOp)
    );
    let launch = plan_unary(UnaryOp::Exp, DType::F32, DType::F32, 513).unwrap();
    assert_eq!(launch.grid, [3, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);
    assert!(plan_unary(UnaryOp::Exp, DType::F32, DType::F32, 0)
        .unwrap()
        .is_empty());
    assert_eq!(
        plan_unary(UnaryOp::Exp, DType::F32, DType::F32, -1),
        Err(Error::NegativeCount)
    );
}

#[test]
fn binary_accepts_exactly_the_documented_op_dtype_side_triples() {
    let mut accepted = Vec::new();
    for op in BINARY_OPS {
        for dtype in DType::ALL {
            for side in SIDES {
                if plan_binary(op, dtype, side, 7).is_ok() {
                    accepted.push((op.tag(), dtype.tag(), side.tag()));
                }
            }
        }
    }
    let expected: Vec<(i32, i32, i32)> = vec![
        // MAX / MIN over the ordered dtypes, every side.
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (0, 3, 0),
        (0, 3, 1),
        (0, 3, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (1, 3, 0),
        (1, 3, 1),
        (1, 3, 2),
        // POW / REM over the floats, every side.
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (3, 0, 0),
        (3, 0, 1),
        (3, 0, 2),
        (3, 1, 0),
        (3, 1, 1),
        (3, 1, 2),
        // MUL with one scalar side.
        (4, 0, 1),
        (4, 0, 2),
        (4, 1, 1),
        (4, 1, 2),
        (4, 2, 1),
        (4, 2, 2),
        (4, 3, 1),
        (4, 3, 2),
        (4, 5, 1),
        (4, 5, 2),
        (4, 6, 1),
        (4, 6, 2),
        // DIV_COMPLEX, both elementwise.
        (5, 5, 0),
        (5, 6, 0),
        // Checked integer ops, every side.
        (6, 2, 0),
        (6, 2, 1),
        (6, 2, 2),
        (6, 3, 0),
        (6, 3, 1),
        (6, 3, 2),
        (7, 2, 0),
        (7, 2, 1),
        (7, 2, 2),
        (7, 3, 0),
        (7, 3, 1),
        (7, 3, 2),
        (8, 2, 0),
        (8, 2, 1),
        (8, 2, 2),
        (8, 3, 0),
        (8, 3, 1),
        (8, 3, 2),
        // Mixed real/complex with the real operand named by the side.
        (9, 0, 1),
        (9, 0, 2),
        (9, 1, 1),
        (9, 1, 2),
        (10, 0, 1),
        (10, 0, 2),
        (10, 1, 1),
        (10, 1, 2),
        (11, 0, 1),
        (11, 0, 2),
        (11, 1, 1),
        (11, 1, 2),
        (12, 0, 1),
        (12, 0, 2),
        (12, 1, 1),
        (12, 1, 2),
        // Float division with one scalar side.
        (13, 0, 1),
        (13, 0, 2),
        (13, 1, 1),
        (13, 1, 2),
    ];
    assert_eq!(accepted, expected);

    // The both-elementwise forms of multiplication and float division stay with
    // the revision-1 entries; a complex scalar side is never meaningful.
    assert_eq!(
        plan_binary(BinaryOp::Mul, DType::F32, ScalarSide::Both, 4),
        Err(Error::UnsupportedOp)
    );
    assert_eq!(
        plan_binary(BinaryOp::Div, DType::F64, ScalarSide::Both, 4),
        Err(Error::UnsupportedOp)
    );
    assert_eq!(
        plan_binary(BinaryOp::DivComplex, DType::C32, ScalarSide::Left, 4),
        Err(Error::UnsupportedOp)
    );
    // A component-length tag must not overflow the doubled buffer length.
    let too_long = i64::MAX / 2 + 1;
    assert_eq!(
        plan_binary(BinaryOp::DivComplex, DType::C64, ScalarSide::Both, too_long),
        Err(Error::Overflow)
    );
    assert_eq!(
        plan_binary(BinaryOp::Mul, DType::C64, ScalarSide::Left, 1000)
            .unwrap()
            .grid,
        [4, 1, 1],
        "a whole-value complex operation does not double the length"
    );
    assert_eq!(
        plan_binary(BinaryOp::Max, DType::F32, ScalarSide::Both, -2),
        Err(Error::NegativeCount)
    );
    assert!(plan_binary(BinaryOp::Max, DType::F32, ScalarSide::Both, 0)
        .unwrap()
        .is_empty());
}

#[test]
fn compare_select_and_clamp_reject_complex_and_bool() {
    for dtype in [DType::C32, DType::C64, DType::Bool] {
        assert_eq!(plan_compare(dtype, 4), Err(Error::UnsupportedDType));
        assert_eq!(plan_select(dtype, 4), Err(Error::UnsupportedDType));
        assert_eq!(plan_clamp(dtype, 4), Err(Error::UnsupportedDType));
    }
    for dtype in [DType::F32, DType::F64, DType::I32, DType::I64] {
        assert_eq!(plan_compare(dtype, 300).unwrap().grid, [2, 1, 1]);
        assert_eq!(plan_select(dtype, 300).unwrap().grid, [2, 1, 1]);
    }
    assert_eq!(
        plan_clamp(DType::F64, 256).unwrap().grid,
        [1, 1, 1],
        "one block exactly"
    );
    assert_eq!(plan_clamp(DType::F64, 257).unwrap().grid, [2, 1, 1]);
    assert!(plan_compare(DType::F32, 0).unwrap().is_empty());
    assert_eq!(plan_select(DType::F32, -1), Err(Error::NegativeCount));
    assert_eq!(plan_clamp(DType::F32, -1), Err(Error::NegativeCount));
}

#[test]
fn combined_strided_validates_the_shared_domain_and_spans() {
    let out = Layout::compact(&[3, 4]).unwrap();
    // a broadcasts axis 1, b is axis 0 reversed (negative stride, shifted offset).
    let a = Layout::new(&[3, 4], &[1, 0], 0).unwrap();
    let b = Layout::new(&[3, 4], &[4, -1], 3).unwrap();
    let launch =
        plan_combined_strided(StridedOp::Axpby, DType::F32, &out, 12, &a, 3, &b, 12).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);

    // The domain comes from the destination, so a different operand shape is a
    // shape mismatch rather than a silently truncated loop.
    let wrong = Layout::new(&[3, 3], &[1, 0], 0).unwrap();
    assert_eq!(
        plan_combined_strided(StridedOp::Mul, DType::F32, &out, 12, &wrong, 3, &a, 3),
        Err(Error::ShapeMismatch)
    );
    // A broadcast destination would race two threads onto one element.
    assert_eq!(
        plan_combined_strided(StridedOp::Mul, DType::F32, &a, 3, &out, 12, &out, 12),
        Err(Error::BroadcastDestination)
    );
    // Spans are checked against the caller's allocation lengths.
    assert_eq!(
        plan_combined_strided(StridedOp::Mul, DType::F32, &out, 12, &a, 3, &out, 11),
        Err(Error::OffsetOutOfSpan)
    );
    assert_eq!(
        plan_combined_strided(StridedOp::Mul, DType::F32, &out, 12, &a, 2, &out, 12),
        Err(Error::OffsetOutOfSpan)
    );
    // AXPBY is real/complex only; the wrapping integer multiply is MUL.
    assert_eq!(
        plan_combined_strided(StridedOp::Axpby, DType::I32, &out, 12, &out, 12, &out, 12),
        Err(Error::UnsupportedOp)
    );
    assert!(
        plan_combined_strided(StridedOp::Mul, DType::I64, &out, 12, &out, 12, &out, 12).is_ok()
    );
    assert_eq!(
        plan_combined_strided(StridedOp::Mul, DType::Bool, &out, 12, &out, 12, &out, 12),
        Err(Error::UnsupportedOp)
    );
    // An empty domain needs no span.
    let empty = Layout::new(&[0, 4], &[4, 1], 0).unwrap();
    assert!(
        plan_combined_strided(StridedOp::Mul, DType::F32, &empty, 0, &empty, 0, &empty, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn revision_six_entries_are_in_the_manifest() {
    for (entry, name) in [
        (Entry::Unary, "t4a_cuda_unary"),
        (Entry::Binary, "t4a_cuda_binary"),
        (Entry::Compare, "t4a_cuda_compare"),
        (Entry::Select, "t4a_cuda_select"),
        (Entry::Clamp, "t4a_cuda_clamp"),
        (Entry::CombinedStrided, "t4a_cuda_combined_strided"),
    ] {
        assert_eq!(entry.name(), name);
        assert!(Entry::ALL.contains(&entry));
    }
    assert!(Entry::Unary.is_elementwise());
    assert!(Entry::Binary.is_elementwise());
    assert!(Entry::Compare.is_elementwise());
    assert!(Entry::Select.is_elementwise());
    assert!(Entry::Clamp.is_elementwise());
    assert!(!Entry::CombinedStrided.is_elementwise());
    assert_eq!(Entry::Clamp.supported_dtypes(), &[DType::F32, DType::F64]);
    assert_eq!(
        Entry::Compare.supported_dtypes(),
        &[DType::F32, DType::F64, DType::I32, DType::I64]
    );
    assert_eq!(Entry::CombinedStrided.supported_dtypes().len(), 6);
}

#[test]
fn a_split_reduction_exposes_its_partials_layout() {
    let input = Layout::compact(&[1_000_000]).unwrap();
    let out = Layout::compact(&[1]).unwrap();
    let plan =
        plan_reduction(DType::F64, ReduceOp::Sum, &out, 1, &input, 1_000_000, 0, 32).unwrap();
    match plan {
        ReductionPlan::Split { split, .. } => {
            let partials = reduction_partials_layout(&input, 0, split.splits).unwrap();
            assert_eq!(partials.extents(), &[split.splits]);
            assert_eq!(partials.element_count().unwrap(), split.splits);
        }
        ReductionPlan::Direct { .. } => panic!("a million-element sum must split"),
    }
}

#[test]
fn structural_extras_are_expressible_with_revision_one_entries() {
    // broadcast_in_dim: source [2,3] broadcast into output [2,4,3] on axes (0,2).
    // The expanded source read layout carries the output extents, the source
    // strides on the mapped axes and stride zero on inserted axes.
    let broadcast = Layout::new(&[2, 4, 3], &[3, 0, 1], 0).unwrap();
    assert_eq!(broadcast.element_count().unwrap(), 24);
    let launch = plan_materialize_strided(DType::F32, &broadcast, 6).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);

    // reverse: reverse axis 2 of [2,3,4]; the axis flips to a negative stride and
    // its offset moves to the far end of that axis.
    let reversed = Layout::new(&[2, 3, 4], &[1, 2, -6], 18).unwrap();
    assert_eq!(reversed.element_count().unwrap(), 24);
    assert!(plan_materialize_strided(DType::F32, &reversed, 24).is_ok());
    assert!(plan_strided_to_strided(
        DType::F32,
        &reversed,
        24,
        &Layout::compact(&[2, 3, 4]).unwrap(),
        24
    )
    .is_ok());

    // concatenate_copy: write a compact [2,3] operand at axis_offset 4 of a
    // compact [6,3] output. The destination is the shifted *view*, so its extents
    // equal the input's and the revision-1 strided copy serves it.
    let output = Layout::compact(&[6, 3]).unwrap();
    let shifted = Layout::new(&[2, 3], output.strides(), 4 * output.strides()[0]).unwrap();
    assert!(plan_strided_to_strided(
        DType::F32,
        &Layout::compact(&[2, 3]).unwrap(),
        6,
        &shifted,
        18
    )
    .is_ok());
    // A shifted view that leaves the buffer is still refused.
    assert_eq!(
        plan_strided_to_strided(
            DType::F32,
            &Layout::compact(&[2, 3]).unwrap(),
            6,
            &shifted,
            17
        ),
        Err(Error::OffsetOutOfSpan)
    );

    // scale_in_place is the binary scalar multiply with out == lhs.
    let scaled = plan_binary(BinaryOp::Mul, DType::C64, ScalarSide::Right, 9).unwrap();
    assert_eq!(scaled.grid, [1, 1, 1]);
}

// --- revision 7: linalg auxiliaries -----------------------------------------

use t4a_cuda_kernels::{
    plan_fill_one, plan_householder_assemble, plan_householder_explicit_v, plan_identity_shifted,
    plan_lu_apply_pivots, plan_lu_extract_outputs, plan_lu_parity, plan_matrix_adjoint,
    plan_qr_apply_phase, plan_qr_phase, plan_upper_trapezoidal_violation,
};

#[test]
fn revision_seven_entries_are_in_the_manifest() {
    for (entry, name) in [
        (Entry::FillOne, "t4a_cuda_fill_one"),
        (Entry::MatrixAdjoint, "t4a_cuda_matrix_adjoint"),
        (
            Entry::HouseholderExplicitV,
            "t4a_cuda_householder_explicit_v",
        ),
        (Entry::IdentityShifted, "t4a_cuda_identity_shifted"),
        (Entry::QrPhase, "t4a_cuda_qr_phase"),
        (Entry::QrApplyPhase, "t4a_cuda_qr_apply_phase"),
        (
            Entry::UpperTrapezoidalViolation,
            "t4a_cuda_upper_trapezoidal_violation",
        ),
        (Entry::HouseholderAssemble, "t4a_cuda_householder_assemble"),
        (Entry::LuExtractOutputs, "t4a_cuda_lu_extract_outputs"),
        (Entry::LuParity, "t4a_cuda_lu_parity"),
        (Entry::LuApplyPivots, "t4a_cuda_lu_apply_pivots"),
    ] {
        assert_eq!(entry.name(), name);
        assert!(Entry::ALL.contains(&entry));
    }
    assert_eq!(Entry::FillOne.supported_dtypes(), &DType::ALL);
    assert!(Entry::FillOne.is_elementwise());
    assert!(!Entry::MatrixAdjoint.is_elementwise());
    assert_eq!(
        Entry::QrPhase.supported_dtypes(),
        &[DType::F32, DType::F64, DType::C32, DType::C64]
    );
    assert_eq!(Entry::LuApplyPivots.supported_dtypes().len(), 6);
    assert_eq!(
        Entry::UpperTrapezoidalViolation.supported_dtypes(),
        &DType::ALL
    );
}

#[test]
fn fill_one_is_a_layout_fill() {
    let view = Layout::new(&[2, 3], &[1, 2], 1).unwrap();
    let launch = plan_fill_one(DType::F64, &view, 7).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);
    assert!(plan_fill_one(DType::Bool, &view, 7).is_ok());
    assert_eq!(
        plan_fill_one(DType::F64, &view, 6),
        Err(Error::OffsetOutOfSpan)
    );
    assert_eq!(
        plan_fill_one(DType::F64, &Layout::new(&[2, 3], &[1, 0], 0).unwrap(), 2),
        Err(Error::BroadcastDestination)
    );
}

#[test]
fn matrix_adjoint_swaps_the_first_two_axes() {
    let v = Layout::compact(&[2, 3, 4]).unwrap();
    let out = Layout::compact(&[3, 2, 4]).unwrap();
    assert!(plan_matrix_adjoint(DType::C64, &out, 24, &v, 24).is_ok());
    assert!(plan_matrix_adjoint(DType::I64, &out, 24, &v, 24).is_ok());
    // Same extents is not a transpose.
    assert_eq!(
        plan_matrix_adjoint(DType::F32, &v, 24, &v, 24),
        Err(Error::ShapeMismatch)
    );
    // Batch axes must match.
    let wrong_batch = Layout::compact(&[3, 2, 5]).unwrap();
    assert_eq!(
        plan_matrix_adjoint(DType::F32, &wrong_batch, 30, &v, 24),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_matrix_adjoint(DType::Bool, &out, 24, &v, 24),
        Err(Error::UnsupportedDType)
    );
    let vector = Layout::compact(&[2]).unwrap();
    assert_eq!(
        plan_matrix_adjoint(DType::F32, &vector, 2, &vector, 2),
        Err(Error::RankOutOfRange)
    );
}

#[test]
fn explicit_v_may_be_narrower_than_the_packed_factors() {
    let packed = Layout::compact(&[4, 6]).unwrap();
    let narrow = Layout::compact(&[4, 3]).unwrap();
    assert!(plan_householder_explicit_v(DType::F64, &narrow, 12, &packed, 24).is_ok());
    let wide = Layout::compact(&[4, 7]).unwrap();
    assert_eq!(
        plan_householder_explicit_v(DType::F64, &wide, 28, &packed, 24),
        Err(Error::ShapeMismatch)
    );
    let short = Layout::compact(&[3, 3]).unwrap();
    assert_eq!(
        plan_householder_explicit_v(DType::F64, &short, 9, &packed, 24),
        Err(Error::ShapeMismatch)
    );
    // Rank 2 only: the baseline reads row and column and never a batch axis.
    let batched = Layout::compact(&[4, 3, 2]).unwrap();
    assert_eq!(
        plan_householder_explicit_v(DType::F64, &batched, 24, &packed, 24),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn identity_shifted_allows_zero_complement_columns() {
    // rows = 3, start = 2, columns = 2: the second column is entirely zero, which is
    // the baseline's complement-column behaviour.
    let out = Layout::compact(&[3, 2]).unwrap();
    assert!(plan_identity_shifted(DType::F64, &out, 6, 2).is_ok());
    assert_eq!(
        plan_identity_shifted(DType::F64, &out, 6, -1),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        plan_identity_shifted(DType::F64, &out, 6, i64::MAX),
        Err(Error::Overflow)
    );
    // The RRQR identity is the batched start = 0 case.
    let batched = Layout::compact(&[4, 3, 2]).unwrap();
    assert!(plan_identity_shifted(DType::C32, &batched, 24, 0).is_ok());
}

#[test]
fn qr_phase_bounds_the_diagonal_it_reads() {
    let r = Layout::compact(&[3, 3, 2]).unwrap();
    let phase = Layout::compact(&[3, 2]).unwrap();
    let launch = plan_qr_phase(DType::F64, &phase, 6, &r, 18).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    // More phase entries than the R diagonal.
    let too_long = Layout::compact(&[4, 2]).unwrap();
    assert_eq!(
        plan_qr_phase(DType::F64, &too_long, 8, &r, 18),
        Err(Error::ShapeMismatch)
    );
    // Phase rank must be the matrix rank minus one.
    let same_rank = Layout::compact(&[3, 3, 2]).unwrap();
    assert_eq!(
        plan_qr_phase(DType::F64, &same_rank, 18, &r, 18),
        Err(Error::ShapeMismatch)
    );
    // Batch extents must match.
    let wrong_batch = Layout::compact(&[3, 5]).unwrap();
    assert_eq!(
        plan_qr_phase(DType::F64, &wrong_batch, 15, &r, 18),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_qr_phase(DType::I32, &phase, 6, &r, 18),
        Err(Error::UnsupportedDType)
    );
}

#[test]
fn qr_apply_phase_checks_the_r_rows_and_the_start() {
    let r = Layout::compact(&[3, 3, 2]).unwrap();
    let q = Layout::compact(&[3, 4, 2]).unwrap();
    let phase = Layout::compact(&[3, 2]).unwrap();
    let launch = plan_qr_apply_phase(DType::F64, &q, 24, &r, 18, &phase, 6, 0).unwrap();
    assert_eq!(launch.grid, [1, 1, 1], "the domain is max(q, r) = 24");
    assert!(plan_qr_apply_phase(DType::F64, &q, 24, &r, 18, &phase, 6, 1).is_ok());
    // R rows beyond the phase entries would read past the phase array.
    let tall = Layout::compact(&[5, 3, 2]).unwrap();
    assert_eq!(
        plan_qr_apply_phase(DType::F64, &q, 24, &tall, 30, &phase, 6, 0),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_qr_apply_phase(DType::F64, &q, 24, &r, 18, &phase, 6, -1),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        plan_qr_apply_phase(DType::F64, &q, 24, &r, 18, &phase, 6, i64::MAX - 1),
        Err(Error::Overflow)
    );
}

#[test]
fn violation_matrix_is_i32_whatever_the_input_dtype() {
    let input = Layout::compact(&[3, 4]).unwrap();
    let violation = Layout::compact(&[3, 4]).unwrap();
    // The payload is four bytes per element even for a C64 input.
    let launch = plan_upper_trapezoidal_violation(DType::C64, &violation, 12, &input, 12).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert!(plan_upper_trapezoidal_violation(DType::Bool, &violation, 12, &input, 12).is_ok());
    let wrong = Layout::compact(&[4, 3]).unwrap();
    assert_eq!(
        plan_upper_trapezoidal_violation(DType::F32, &wrong, 12, &input, 12),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn assemble_requires_only_the_regions_its_branches_read() {
    let out = Layout::compact(&[4, 6]).unwrap();
    let narrow_r = Layout::compact(&[2, 6]).unwrap();
    let narrow_q = Layout::compact(&[4, 2]).unwrap();
    // factor_width = 2 reads folded_r rows 0..2 and packed_q columns 0..2.
    assert!(
        plan_householder_assemble(DType::F64, &out, 24, 2, &narrow_q, 8, &narrow_r, 12, 2, 2)
            .is_ok()
    );
    // One row short of the branch's reach.
    let too_short = Layout::compact(&[1, 6]).unwrap();
    assert_eq!(
        plan_householder_assemble(DType::F64, &out, 24, 2, &narrow_q, 8, &too_short, 6, 2, 2),
        Err(Error::ShapeMismatch)
    );
    // factor_width = 0 never touches the sources, so nothing is required of them.
    let empty = Layout::compact(&[0, 0]).unwrap();
    assert!(plan_householder_assemble(DType::F64, &out, 24, 2, &empty, 0, &empty, 0, 0, 0).is_ok());
    assert_eq!(
        plan_householder_assemble(DType::F64, &out, 24, 2, &narrow_q, 8, &narrow_r, 12, 2, -1),
        Err(Error::NegativeCount)
    );
    // The coefficient prefix must exist in the source.
    assert_eq!(
        plan_householder_assemble(DType::F64, &out, 24, 4, &narrow_q, 8, &narrow_r, 12, 1, 2),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn lu_extract_checks_the_pinned_shape_relationships() {
    let work = Layout::compact(&[5, 3, 2]).unwrap();
    let p = Layout::compact(&[5, 5, 2]).unwrap();
    let l = Layout::compact(&[5, 3, 2]).unwrap();
    let u = Layout::compact(&[3, 3, 2]).unwrap();
    let launch =
        plan_lu_extract_outputs(DType::F64, &p, 50, &l, 30, &u, 18, &work, 30, 6, 3, 2, 2).unwrap();
    assert_eq!(
        launch.grid,
        [1, 1, 1],
        "the domain is max(P, L, U, parity) = 50"
    );
    // A non-square P cannot be a permutation matrix.
    let p_bad = Layout::compact(&[5, 4, 2]).unwrap();
    assert_eq!(
        plan_lu_extract_outputs(
            DType::F64,
            &p_bad,
            40,
            &l,
            30,
            &u,
            18,
            &work,
            30,
            6,
            3,
            2,
            2
        ),
        Err(Error::ShapeMismatch)
    );
    // L must be exactly [m, k] and U exactly [k, n]; vary one at a time with a
    // valid P so a rejection cannot come from the other operand.
    let l_wide = Layout::compact(&[5, 4, 2]).unwrap();
    assert_eq!(
        plan_lu_extract_outputs(
            DType::F64,
            &p,
            50,
            &l_wide,
            40,
            &u,
            18,
            &work,
            30,
            6,
            3,
            2,
            2
        ),
        Err(Error::ShapeMismatch)
    );
    let l_narrow = Layout::compact(&[5, 2, 2]).unwrap();
    assert_eq!(
        plan_lu_extract_outputs(
            DType::F64,
            &p,
            50,
            &l_narrow,
            20,
            &u,
            18,
            &work,
            30,
            6,
            3,
            2,
            2
        ),
        Err(Error::ShapeMismatch)
    );
    let u_short = Layout::compact(&[2, 3, 2]).unwrap();
    assert_eq!(
        plan_lu_extract_outputs(
            DType::F64,
            &p,
            50,
            &l,
            30,
            &u_short,
            12,
            &work,
            30,
            6,
            3,
            2,
            2
        ),
        Err(Error::ShapeMismatch)
    );
    let u_bad = Layout::compact(&[4, 3, 2]).unwrap();
    assert_eq!(
        plan_lu_extract_outputs(
            DType::F64,
            &p,
            50,
            &l,
            30,
            &u_bad,
            24,
            &work,
            30,
            6,
            3,
            2,
            2
        ),
        Err(Error::ShapeMismatch)
    );
    // The pivot array must hold k * batches pivots.
    assert_eq!(
        plan_lu_extract_outputs(DType::F64, &p, 50, &l, 30, &u, 18, &work, 30, 5, 3, 2, 2),
        Err(Error::ShapeMismatch)
    );
    // k must be the factorisation step count, or a U row could leave `work`.
    assert_eq!(
        plan_lu_extract_outputs(DType::F64, &p, 50, &l, 30, &u, 18, &work, 30, 3, 2, 2, 2),
        Err(Error::ShapeMismatch)
    );
    // batches must be the batch extent product, and parity must hold them.
    assert_eq!(
        plan_lu_extract_outputs(DType::F64, &p, 50, &l, 30, &u, 18, &work, 30, 9, 3, 3, 2),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_lu_extract_outputs(DType::F64, &p, 50, &l, 30, &u, 18, &work, 30, 6, 3, 2, 1),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_lu_extract_outputs(DType::I32, &p, 50, &l, 30, &u, 18, &work, 30, 6, 3, 2, 2),
        Err(Error::UnsupportedDType)
    );
    // The standalone parity entry needs only the batch count.
    assert_eq!(
        plan_lu_parity(DType::C64, 21, 7, 3).unwrap().grid,
        [1, 1, 1]
    );
    // A pivot array that cannot hold k * batches pivots, or an overflowing count.
    assert_eq!(
        plan_lu_parity(DType::C64, 20, 7, 3),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_lu_parity(DType::C64, i64::MAX, 2, i64::MAX),
        Err(Error::Overflow)
    );
    // An element count is not enough: the pivot byte span must fit a device address.
    assert_eq!(
        plan_lu_parity(DType::C64, i64::MAX, 1, i64::MAX),
        Err(Error::Overflow)
    );
    assert_eq!(
        plan_lu_parity(DType::C64, 0, -1, 3),
        Err(Error::NegativeCount)
    );
}

#[test]
fn lu_apply_pivots_needs_matching_extents_and_a_bounded_k() {
    let input = Layout::compact(&[4, 3]).unwrap();
    let out = Layout::compact(&[4, 3]).unwrap();
    assert!(plan_lu_apply_pivots(DType::F32, &out, 12, &input, 12, 3, 3).is_ok());
    // One pivot per step per batch: a shorter array is refused even though the
    // dimensions agree.
    assert_eq!(
        plan_lu_apply_pivots(DType::F32, &out, 12, &input, 12, 2, 3),
        Err(Error::ShapeMismatch)
    );
    let batched = Layout::compact(&[2, 1, 2]).unwrap();
    assert_eq!(
        plan_lu_apply_pivots(DType::F32, &batched, 4, &batched, 4, 1, 1),
        Err(Error::ShapeMismatch)
    );
    assert!(plan_lu_apply_pivots(DType::F32, &batched, 4, &batched, 4, 2, 1).is_ok());
    assert_eq!(
        plan_lu_apply_pivots(DType::F32, &out, 12, &input, 12, 3, 5),
        Err(Error::ShapeMismatch)
    );
    let other = Layout::compact(&[3, 4]).unwrap();
    assert_eq!(
        plan_lu_apply_pivots(DType::F32, &out, 12, &other, 12, 3, 3),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_lu_apply_pivots(DType::F32, &out, 12, &input, 12, 3, -1),
        Err(Error::NegativeCount)
    );
    assert!(plan_lu_apply_pivots(DType::I64, &out, 12, &input, 12, 0, 0).is_ok());
}

// --- revision 8: rank-revealing QR primitives ------------------------------

use t4a_cuda_kernels::{
    plan_apply_reflector, plan_column_norms, plan_extract_r, plan_initialize_permutation,
    plan_prefix_rank, plan_reflector, plan_select_pivot, plan_swap_columns, PLANE_BLOCK,
};

fn plane_grid(tasks: i64) -> [u32; 3] {
    [tasks as u32, 1, 1]
}

#[test]
fn revision_eight_entries_are_in_the_manifest() {
    for (entry, name) in [
        (
            Entry::InitializePermutation,
            "t4a_cuda_initialize_permutation",
        ),
        (Entry::ColumnNorms, "t4a_cuda_column_norms"),
        (Entry::SelectPivot, "t4a_cuda_select_pivot"),
        (Entry::SwapColumns, "t4a_cuda_swap_columns"),
        (Entry::Reflector, "t4a_cuda_reflector"),
        (Entry::ApplyReflector, "t4a_cuda_apply_reflector"),
        (Entry::ExtractR, "t4a_cuda_extract_r"),
        (Entry::PrefixRank, "t4a_cuda_prefix_rank"),
    ] {
        assert_eq!(entry.name(), name);
        assert!(Entry::ALL.contains(&entry));
    }
    assert_eq!(
        Entry::SelectPivot.supported_dtypes(),
        &[DType::F32, DType::F64]
    );
    assert_eq!(Entry::SwapColumns.supported_dtypes().len(), 6);
    assert_eq!(
        Entry::ColumnNorms.supported_dtypes(),
        &[DType::F32, DType::F64, DType::C32, DType::C64]
    );
    assert_eq!(Entry::InitializePermutation.supported_dtypes(), &DType::ALL);
}

#[test]
fn initialize_permutation_is_the_modulo_image() {
    let launch = plan_initialize_permutation(4, 10).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);
    assert_eq!(
        plan_initialize_permutation(0, 10),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        plan_initialize_permutation(4, -1),
        Err(Error::NegativeCount)
    );
    assert!(plan_initialize_permutation(4, 0).unwrap().is_empty());
}

#[test]
fn column_norms_maps_one_task_per_trailing_column() {
    let work = Layout::compact(&[5, 4, 2]).unwrap();
    let norms = Layout::compact(&[4, 2]).unwrap();
    // step 1 leaves three columns per batch, so six tasks.
    let launch = plan_column_norms(DType::C32, &work, 40, &norms, 8, 1).unwrap();
    assert_eq!(launch.grid, plane_grid(6));
    assert_eq!(launch.block, [PLANE_BLOCK, 1, 1]);
    // The norms array carries the component type, so a shorter capacity is caught.
    assert_eq!(
        plan_column_norms(DType::C64, &work, 40, &norms, 7, 1),
        Err(Error::OffsetOutOfSpan)
    );
    // The shorter operand rule: the norms must drop exactly the row axis.
    let square = Layout::compact(&[4, 2]).unwrap();
    assert!(plan_column_norms(DType::F64, &work, 40, &square, 8, 0).is_ok());
    let same_rank = Layout::compact(&[5, 4, 2]).unwrap();
    assert_eq!(
        plan_column_norms(DType::F64, &work, 40, &same_rank, 40, 0),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_column_norms(DType::F64, &work, 40, &norms, 8, 5),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_column_norms(DType::F64, &work, 40, &norms, 8, -1),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_column_norms(DType::I32, &work, 40, &norms, 8, 0),
        Err(Error::UnsupportedDType)
    );
    // A step past the rows would leave every lane without an accumulator.
    let flat = Layout::compact(&[2, 3]).unwrap();
    let norms = Layout::compact(&[3]).unwrap();
    assert_eq!(
        plan_column_norms(DType::F32, &flat, 6, &norms, 3, 2),
        Err(Error::ShapeMismatch)
    );
    assert!(plan_column_norms(DType::F32, &flat, 6, &norms, 3, 3)
        .unwrap()
        .is_empty());
}

#[test]
fn select_pivot_requires_matching_arrays_and_capacities() {
    let norms = Layout::compact(&[6, 2]).unwrap();
    let permutation = Layout::compact(&[6, 2]).unwrap();
    let launch = plan_select_pivot(DType::F32, &norms, 12, &permutation, 12, 2, 2, 1).unwrap();
    assert_eq!(launch.grid, plane_grid(2));
    assert_eq!(launch.block, [PLANE_BLOCK, 1, 1]);
    // The flat pivot and status arrays need one word per batch.
    assert_eq!(
        plan_select_pivot(DType::F32, &norms, 12, &permutation, 12, 1, 2, 1),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_select_pivot(DType::F32, &norms, 12, &permutation, 12, 2, 1, 1),
        Err(Error::ShapeMismatch)
    );
    // The permutation is int64 whatever the norm type is.
    assert_eq!(
        plan_select_pivot(DType::F64, &norms, 12, &permutation, 11, 2, 2, 1),
        Err(Error::OffsetOutOfSpan)
    );
    let other = Layout::compact(&[6, 3]).unwrap();
    assert_eq!(
        plan_select_pivot(DType::F32, &norms, 12, &other, 18, 2, 2, 1),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_select_pivot(DType::F32, &norms, 12, &permutation, 12, 2, 2, 7),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_select_pivot(DType::C64, &norms, 12, &permutation, 12, 2, 2, 1),
        Err(Error::UnsupportedDType)
    );
}

#[test]
fn swap_columns_bounds_the_step_and_the_permutation() {
    let work = Layout::compact(&[4, 3, 2]).unwrap();
    let permutation = Layout::compact(&[3, 2]).unwrap();
    let launch = plan_swap_columns(DType::F64, &work, 24, &permutation, 6, 2, 1).unwrap();
    assert_eq!(launch.grid, [1, 1, 1]);
    assert_eq!(launch.block, [FLAT_BLOCK, 1, 1]);
    // rows * batches above one block: 20 rows times 20 batches.
    let big = Layout::compact(&[20, 20, 20]).unwrap();
    let big_permutation = Layout::compact(&[20, 20]).unwrap();
    assert_eq!(
        plan_swap_columns(DType::F64, &big, 8000, &big_permutation, 400, 20, 3)
            .unwrap()
            .grid,
        [2, 1, 1],
        "400 threads at 256 per block"
    );
    assert_eq!(
        plan_swap_columns(DType::F64, &work, 24, &permutation, 6, 2, 3),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_swap_columns(DType::F64, &work, 24, &permutation, 6, 2, -1),
        Err(Error::ShapeMismatch)
    );
    let short = Layout::compact(&[2, 2]).unwrap();
    assert_eq!(
        plan_swap_columns(DType::F64, &work, 24, &short, 4, 2, 0),
        Err(Error::ShapeMismatch)
    );
    // One pivot word per batch, or a batched swap reads past its array.
    assert_eq!(
        plan_swap_columns(DType::F64, &work, 24, &permutation, 6, 1, 1),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn reflector_bounds_the_step_and_the_coefficients() {
    let work = Layout::compact(&[5, 4, 2]).unwrap();
    let coeff = Layout::compact(&[4, 2]).unwrap();
    let launch = plan_reflector(DType::C64, &work, 40, &coeff, 8, 3).unwrap();
    assert_eq!(launch.grid, plane_grid(2));
    assert_eq!(launch.block, [PLANE_BLOCK, 1, 1]);
    // step must stay below the coefficient extent as well as the matrix.
    let short_coeff = Layout::compact(&[2, 2]).unwrap();
    assert_eq!(
        plan_reflector(DType::C64, &work, 40, &short_coeff, 4, 2),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_reflector(DType::C64, &work, 40, &coeff, 8, 4),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_reflector(DType::C64, &work, 40, &coeff, 8, -1),
        Err(Error::ShapeMismatch)
    );
    let batched_coeff = Layout::compact(&[4]).unwrap();
    assert_eq!(
        plan_reflector(DType::C64, &work, 40, &batched_coeff, 4, 0),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn apply_reflector_checks_the_target_window() {
    let packed = Layout::compact(&[5, 4, 2]).unwrap();
    let coeff = Layout::compact(&[4, 2]).unwrap();
    let target = Layout::compact(&[5, 6, 2]).unwrap();
    let launch =
        plan_apply_reflector(DType::F64, &packed, 40, &target, 60, &coeff, 8, 1, 2, 3).unwrap();
    assert_eq!(
        launch.grid,
        plane_grid(6),
        "two batches times three columns"
    );
    // The window must stay inside the target's columns.
    assert_eq!(
        plan_apply_reflector(DType::F64, &packed, 40, &target, 60, &coeff, 8, 1, 4, 3),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_apply_reflector(DType::F64, &packed, 40, &target, 60, &coeff, 8, 1, -1, 3),
        Err(Error::NegativeCount)
    );
    assert_eq!(
        plan_apply_reflector(DType::F64, &packed, 40, &target, 60, &coeff, 8, 1, 0, -3),
        Err(Error::NegativeCount)
    );
    // Rows and batch axes must agree.
    let short = Layout::compact(&[4, 6, 2]).unwrap();
    assert_eq!(
        plan_apply_reflector(DType::F64, &packed, 40, &short, 48, &coeff, 8, 1, 0, 1),
        Err(Error::ShapeMismatch)
    );
    // The coefficient slot must exist for this step.
    assert_eq!(
        plan_apply_reflector(DType::F64, &packed, 40, &target, 60, &coeff, 8, 4, 0, 1),
        Err(Error::ShapeMismatch)
    );
    // The reflector vector lives in packed column `step`, so a step inside the rows
    // but outside the columns is refused.
    let narrow = Layout::compact(&[5, 1, 2]).unwrap();
    let coeff_narrow = Layout::compact(&[4, 2]).unwrap();
    let target_narrow = Layout::compact(&[5, 2, 2]).unwrap();
    assert_eq!(
        plan_apply_reflector(
            DType::F64,
            &narrow,
            10,
            &target_narrow,
            20,
            &coeff_narrow,
            8,
            1,
            0,
            1
        ),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn extract_r_copies_the_upper_triangle_only() {
    let work = Layout::compact(&[5, 4, 2]).unwrap();
    let r = Layout::compact(&[3, 4, 2]).unwrap();
    assert_eq!(
        plan_extract_r(DType::C32, &work, 40, &r, 24).unwrap().grid,
        [1, 1, 1]
    );
    let too_many_rows = Layout::compact(&[6, 4, 2]).unwrap();
    assert_eq!(
        plan_extract_r(DType::C32, &work, 40, &too_many_rows, 48),
        Err(Error::ShapeMismatch)
    );
    let wrong_columns = Layout::compact(&[3, 5, 2]).unwrap();
    assert_eq!(
        plan_extract_r(DType::C32, &work, 40, &wrong_columns, 30),
        Err(Error::ShapeMismatch)
    );
}

#[test]
fn prefix_rank_bounds_the_diagonal_domain() {
    let r = Layout::compact(&[3, 4, 2]).unwrap();
    let launch = plan_prefix_rank(DType::F64, &r, 24, 2, 2, 1e-12, 0.0).unwrap();
    assert_eq!(launch.grid, plane_grid(2));
    assert_eq!(launch.block, [PLANE_BLOCK, 1, 1]);
    // One rank and one status word per batch.
    assert_eq!(
        plan_prefix_rank(DType::F64, &r, 24, 1, 2, 1e-12, 0.0),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_prefix_rank(DType::F64, &r, 24, 2, 1, 1e-12, 0.0),
        Err(Error::ShapeMismatch)
    );
    // More rows than columns would read past the matrix on the diagonal.
    let tall = Layout::compact(&[5, 3, 2]).unwrap();
    assert_eq!(
        plan_prefix_rank(DType::F64, &tall, 30, 2, 2, 1e-12, 0.0),
        Err(Error::ShapeMismatch)
    );
    let empty = Layout::compact(&[0, 3]).unwrap();
    assert_eq!(
        plan_prefix_rank(DType::F64, &empty, 0, 1, 1, 1e-12, 0.0),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(
        plan_prefix_rank(DType::I64, &r, 24, 2, 2, 1e-12, 0.0),
        Err(Error::UnsupportedDType)
    );
}
