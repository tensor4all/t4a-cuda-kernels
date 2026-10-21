//! Host-side validation, planning and manifest consistency. Needs no CUDA.

use t4a_cuda_kernels::{
    bytes, plan_contiguous_to_view, plan_copy_exact, plan_elementwise, plan_fill_zero,
    plan_fill_zero_view, plan_materialize_strided, plan_strided_to_strided, plan_tiled_transpose,
    CLayout, CTransposeConfig, DType, Entry, Error, Launch, Layout, TransposeConfig, ABI_REVISION,
    DEFAULT_SHARED_BYTES, FLAT_BLOCK, MAX_RANK,
};

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
        include_str!("../cuda/src/gather_scatter.cu")
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
    plan_reduce, plan_reduce_partials, plan_reduction, split_policy, ReduceOp, ReduceStrategy,
    ReductionPlan, Split, MAX_SPLITS,
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
