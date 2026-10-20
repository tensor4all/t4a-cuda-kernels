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
        include_str!("../cuda/src/validation.cu")
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
