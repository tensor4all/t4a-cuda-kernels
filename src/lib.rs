//! Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.
//!
//! This crate owns kernel semantics, the C-compatible kernel ABI and checked
//! host-side planning. It does not depend on `t4a-cuda-runtime`, tenferro,
//! CubeCL or CubeK, never allocates device memory and never launches anything:
//! the caller supplies context, stream, buffers, module loading, workspace and
//! completion management.
//!
//! The kernel sources live in `cuda/`; [`Entry`] names every entry point and
//! its supported dtypes, and the ABI is documented in `docs/abi/c-kernel-abi.md`.
//!
//! The host-side shape derivation of the diagonal and pad entries is translated
//! from tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c
//! `crates/tenferro-gpu/src/cubecl/mod.rs` (`extract_diagonal_shape` L5352,
//! `embed_diagonal_shape` L5369, `pad_output_shape` L5504, `slice_output_shape`).
//! Copyright (c) 2026 Hiroshi Shinaoka and tenferro-rs contributors;
//! MIT OR Apache-2.0.

#![forbid(unsafe_code)]

use core::fmt;

/// Maximum supported tensor rank. Deeper layouts need a later ABI revision.
pub const MAX_RANK: usize = 8;
/// Version of the C ABI described by this crate.
pub const ABI_REVISION: u32 = 2;
/// Threads per block for the flat (element-wise) entries.
pub const FLAT_BLOCK: u32 = 256;
/// Largest block count for grid dimensions y and z.
pub const MAX_GRID_YZ: u32 = 65_535;
/// Largest block count for grid dimension x.
pub const MAX_GRID_X: u32 = i32::MAX as u32;
/// Default per-block shared-memory budget on current CUDA architectures.
pub const DEFAULT_SHARED_BYTES: u64 = 48 * 1024;

/// Element type of a kernel invocation, as a runtime tag in the C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    /// One byte per element, value 0 or 1.
    Bool,
    /// Interleaved `(re, im)` single precision, alignment 4.
    C32,
    /// Interleaved `(re, im)` double precision, alignment 8.
    C64,
}

impl DType {
    pub const ALL: [DType; 7] = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];

    /// C ABI tag.
    pub const fn tag(self) -> i32 {
        match self {
            DType::F32 => 0,
            DType::F64 => 1,
            DType::I32 => 2,
            DType::I64 => 3,
            DType::Bool => 4,
            DType::C32 => 5,
            DType::C64 => 6,
        }
    }

    /// Inverse of [`DType::tag`].
    pub const fn from_tag(tag: i32) -> Option<DType> {
        match tag {
            0 => Some(DType::F32),
            1 => Some(DType::F64),
            2 => Some(DType::I32),
            3 => Some(DType::I64),
            4 => Some(DType::Bool),
            5 => Some(DType::C32),
            6 => Some(DType::C64),
            _ => None,
        }
    }

    pub const fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
            DType::C32 => 8,
            DType::C64 => 16,
        }
    }
}

/// Host-side rejection of an invocation the kernels cannot serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `rank > MAX_RANK`.
    RankOutOfRange,
    /// The number of extents or strides does not match the rank.
    RankLength,
    /// A negative extent.
    NegativeExtent,
    /// A negative element count.
    NegativeCount,
    /// The extent product or an address computation overflowed `i64`.
    Overflow,
    /// `len` differs from the layout's element count.
    CountMismatch,
    /// The layout's reachable element span leaves the caller's allocation.
    OffsetOutOfSpan,
    /// A destination axis with extent greater than one has stride zero.
    BroadcastDestination,
    /// The entry does not support this dtype.
    UnsupportedDType,
    /// `entry` is not a flat element-wise entry.
    NotElementwise,
    /// Grid, block or shared-memory request exceeds the device limits.
    LaunchLimit,
    /// Unsupported or inconsistent transpose configuration.
    InvalidTransposeConfig,
    /// Axis index outside the operand rank.
    InvalidAxis,
    /// The same axis used for both sides of an axis pair.
    DuplicateAxis,
    /// An operand that the entry requires to be compact is not.
    NotCompact,
    /// Operand extents or ranks are inconsistent with the operation.
    ShapeMismatch,
    /// A slice step below one.
    InvalidStep,
    /// Negative interior padding.
    InvalidPadding,
    /// A validation domain above the 32-bit status word.
    DomainTooLarge,
    /// A sampling stride below one.
    InvalidStride,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Error::RankOutOfRange => "rank exceeds the supported maximum",
            Error::RankLength => "extent/stride count does not match the rank",
            Error::NegativeExtent => "negative extent",
            Error::NegativeCount => "negative element count",
            Error::Overflow => "size or address arithmetic overflowed",
            Error::CountMismatch => "element count differs from the extent product",
            Error::OffsetOutOfSpan => "layout reaches outside the supplied allocation",
            Error::BroadcastDestination => "destination axis with extent > 1 has stride 0",
            Error::UnsupportedDType => "entry does not support this dtype",
            Error::NotElementwise => "entry is not a flat element-wise entry",
            Error::LaunchLimit => "launch geometry exceeds the device limits",
            Error::InvalidTransposeConfig => "invalid or unsupported transpose configuration",
            Error::InvalidAxis => "axis outside the operand rank",
            Error::DuplicateAxis => "the same axis appears twice",
            Error::NotCompact => "operand must be compact",
            Error::ShapeMismatch => "operand extents do not match the operation",
            Error::InvalidStep => "slice step must be at least one",
            Error::InvalidPadding => "interior padding must be non-negative",
            Error::DomainTooLarge => "validation domain exceeds the 32-bit status word",
            Error::InvalidStride => "sampling stride must be at least one",
        };
        f.write_str(text)
    }
}

impl std::error::Error for Error {}

/// Logical layout of a strided operand.
///
/// Axis 0 is the fastest-varying logical axis: a flat logical index is decoded
/// with `coordinate = flat % extent`, `flat /= extent` from axis 0 upwards, and
/// the element address is `offset + sum(coordinate * stride)`. Strides and the
/// offset are signed element counts; stride zero denotes a broadcast axis.
///
/// Rank is limited to [`MAX_RANK`]; a rank-0 layout is a single scalar at
/// `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    rank: u32,
    offset: i64,
    extents: [i64; MAX_RANK],
    strides: [i64; MAX_RANK],
}

impl Layout {
    /// Build a layout; extents and strides must have equal length `<= MAX_RANK`.
    pub fn new(extents: &[i64], strides: &[i64], offset: i64) -> Result<Layout, Error> {
        if extents.len() != strides.len() {
            return Err(Error::RankLength);
        }
        if extents.len() > MAX_RANK {
            return Err(Error::RankOutOfRange);
        }
        if extents.iter().any(|&extent| extent < 0) {
            return Err(Error::NegativeExtent);
        }
        let mut layout = Layout {
            rank: extents.len() as u32,
            offset,
            extents: [0; MAX_RANK],
            strides: [0; MAX_RANK],
        };
        layout.extents[..extents.len()].copy_from_slice(extents);
        layout.strides[..strides.len()].copy_from_slice(strides);
        Ok(layout)
    }

    /// Compact row-major layout over `extents`, with axis 0 fastest.
    pub fn compact(extents: &[i64]) -> Result<Layout, Error> {
        let mut strides = [0; MAX_RANK];
        let mut stride: i64 = 1;
        for (axis, &extent) in extents.iter().enumerate() {
            if axis == MAX_RANK {
                return Err(Error::RankOutOfRange);
            }
            strides[axis] = stride;
            stride = stride.checked_mul(extent).ok_or(Error::Overflow)?;
        }
        Layout::new(extents, &strides[..extents.len()], 0)
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn offset(&self) -> i64 {
        self.offset
    }

    pub fn extents(&self) -> &[i64] {
        &self.extents[..self.rank as usize]
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides[..self.rank as usize]
    }

    /// Product of the extents; one for a rank-0 layout.
    pub fn element_count(&self) -> Result<i64, Error> {
        self.extents().iter().try_fold(1i64, |acc, &extent| {
            acc.checked_mul(extent).ok_or(Error::Overflow)
        })
    }

    /// Smallest element index the layout can address.
    pub fn min_reachable_offset(&self) -> Result<i64, Error> {
        self.reachable_offset(false)
    }

    /// Largest element index the layout can address.
    pub fn max_reachable_offset(&self) -> Result<i64, Error> {
        self.reachable_offset(true)
    }

    fn reachable_offset(&self, maximum: bool) -> Result<i64, Error> {
        let mut total = self.offset;
        for axis in 0..self.rank as usize {
            let step = self.extents[axis]
                .checked_sub(1)
                .ok_or(Error::Overflow)?
                .checked_mul(self.strides[axis])
                .ok_or(Error::Overflow)?;
            let step = if maximum { step.max(0) } else { step.min(0) };
            total = total.checked_add(step).ok_or(Error::Overflow)?;
        }
        Ok(total)
    }

    /// Whether the layout is the compact form of its extents with offset zero.
    pub fn is_compact(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let mut stride: i64 = 1;
        for axis in 0..self.rank as usize {
            if self.strides[axis] != stride {
                return false;
            }
            match stride.checked_mul(self.extents[axis]) {
                Some(next) => stride = next,
                None => return false,
            }
        }
        true
    }

    /// Necessary condition for an injective layout: no broadcast axis with more
    /// than one element.
    ///
    /// This is not sufficient in general (for example `extents = [2, 2]` with
    /// `strides = [1, 1]`), so a caller that writes through a layout must still
    /// prove injectivity, as the consumer planners already do.
    pub fn rejects_broadcast_axis(&self) -> bool {
        (0..self.rank as usize).all(|axis| self.extents[axis] <= 1 || self.strides[axis] != 0)
    }

    /// Check the layout against a caller allocation of `len` elements.
    ///
    /// `element_size` is checked as well: device address arithmetic is
    /// `base + index * element_size` bytes, so an element offset that does not
    /// fit in a 64-bit byte offset is rejected rather than left to overflow.
    fn check_span(&self, len: i64, element_size: usize) -> Result<(), Error> {
        if len < 0 {
            return Err(Error::NegativeCount);
        }
        // An empty layout addresses nothing, so no span can be violated.
        if self.element_count()? == 0 {
            return Ok(());
        }
        let min = self.min_reachable_offset()?;
        let max = self.max_reachable_offset()?;
        if min < 0 || max >= len {
            return Err(Error::OffsetOutOfSpan);
        }
        max.checked_mul(i64::try_from(element_size).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        Ok(())
    }
}

/// Tile configuration of [`Entry::TiledTranspose`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposeConfig {
    tile: u32,
    block_rows: u32,
    padding: u32,
    vector_width: u32,
}

impl TransposeConfig {
    /// The baseline default configuration.
    pub const DEFAULT: TransposeConfig = TransposeConfig {
        tile: 16,
        block_rows: 8,
        padding: 1,
        vector_width: 1,
    };

    /// Validate a configuration against the supported algorithm family.
    pub fn new(
        tile: u32,
        block_rows: u32,
        padding: u32,
        vector_width: u32,
    ) -> Result<TransposeConfig, Error> {
        if !matches!(tile, 8 | 16 | 32)
            || !matches!(block_rows, 4 | 8 | 16)
            || padding > 1
            || !matches!(vector_width, 1 | 2 | 4)
            || tile % vector_width != 0
        {
            return Err(Error::InvalidTransposeConfig);
        }
        Ok(TransposeConfig {
            tile,
            block_rows,
            padding,
            vector_width,
        })
    }

    pub fn tile(self) -> u32 {
        self.tile
    }

    pub fn block_rows(self) -> u32 {
        self.block_rows
    }

    pub fn padding(self) -> u32 {
        self.padding
    }

    pub fn vector_width(self) -> u32 {
        self.vector_width
    }

    /// Dynamic shared-memory bytes this configuration requires.
    pub fn shared_bytes(self, element_size: usize) -> Result<u64, Error> {
        let pitch = u64::from(self.tile) + u64::from(self.padding);
        u64::from(self.tile)
            .checked_mul(pitch)
            .and_then(|elements| elements.checked_mul(element_size as u64))
            .ok_or(Error::Overflow)
    }
}

/// Kernel entry point and its supported dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    FillZero,
    FillZeroView,
    CopyExact,
    MaterializeStrided,
    ContiguousToView,
    StridedToStrided,
    TiledTranspose,
    Add,
    Sub,
    Mul,
    Div,
    ExtractDiagonal,
    EmbedDiagonalCopy,
    Tril,
    Triu,
    Slice,
    DynamicSlice,
    Pad,
    InitValidation,
    ValidateFloatIndices,
    ExtractInvalidFloatIndex,
    ValidateRealCast,
    ExtractInvalidRealCast,
}

impl Entry {
    pub const ALL: [Entry; 23] = [
        Entry::FillZero,
        Entry::FillZeroView,
        Entry::CopyExact,
        Entry::MaterializeStrided,
        Entry::ContiguousToView,
        Entry::StridedToStrided,
        Entry::TiledTranspose,
        Entry::Add,
        Entry::Sub,
        Entry::Mul,
        Entry::Div,
        Entry::ExtractDiagonal,
        Entry::EmbedDiagonalCopy,
        Entry::Tril,
        Entry::Triu,
        Entry::Slice,
        Entry::DynamicSlice,
        Entry::Pad,
        Entry::InitValidation,
        Entry::ValidateFloatIndices,
        Entry::ExtractInvalidFloatIndex,
        Entry::ValidateRealCast,
        Entry::ExtractInvalidRealCast,
    ];

    /// Exported CUDA symbol; equal to the name documented in the ABI.
    pub const fn name(self) -> &'static str {
        match self {
            Entry::FillZero => "t4a_cuda_fill_zero",
            Entry::FillZeroView => "t4a_cuda_fill_zero_view",
            Entry::CopyExact => "t4a_cuda_copy_exact",
            Entry::MaterializeStrided => "t4a_cuda_materialize_strided",
            Entry::ContiguousToView => "t4a_cuda_contiguous_to_view",
            Entry::StridedToStrided => "t4a_cuda_strided_to_strided",
            Entry::TiledTranspose => "t4a_cuda_tiled_transpose",
            Entry::Add => "t4a_cuda_add",
            Entry::Sub => "t4a_cuda_sub",
            Entry::Mul => "t4a_cuda_mul",
            Entry::Div => "t4a_cuda_div",
            Entry::ExtractDiagonal => "t4a_cuda_extract_diagonal",
            Entry::EmbedDiagonalCopy => "t4a_cuda_embed_diagonal_copy",
            Entry::Tril => "t4a_cuda_tril",
            Entry::Triu => "t4a_cuda_triu",
            Entry::Slice => "t4a_cuda_slice",
            Entry::DynamicSlice => "t4a_cuda_dynamic_slice",
            Entry::Pad => "t4a_cuda_pad",
            Entry::InitValidation => "t4a_cuda_init_validation",
            Entry::ValidateFloatIndices => "t4a_cuda_validate_float_indices",
            Entry::ExtractInvalidFloatIndex => "t4a_cuda_extract_invalid_float_index",
            Entry::ValidateRealCast => "t4a_cuda_validate_real_cast",
            Entry::ExtractInvalidRealCast => "t4a_cuda_extract_invalid_real_cast",
        }
    }

    /// Dtypes this entry serves. Arithmetic excludes Bool, matching the
    /// baseline dispatch; division is float-only.
    pub const fn supported_dtypes(self) -> &'static [DType] {
        match self {
            Entry::Div
            | Entry::InitValidation
            | Entry::ValidateFloatIndices
            | Entry::ExtractInvalidFloatIndex
            | Entry::ValidateRealCast
            | Entry::ExtractInvalidRealCast => &[DType::F32, DType::F64],
            Entry::Add | Entry::Sub | Entry::Mul => &[
                DType::F32,
                DType::F64,
                DType::I32,
                DType::I64,
                DType::C32,
                DType::C64,
            ],
            _ => &DType::ALL,
        }
    }

    pub fn supports(self, dtype: DType) -> bool {
        self.supported_dtypes().contains(&dtype)
    }

    /// Whether the entry is a flat element-wise kernel over `len` elements.
    pub const fn is_elementwise(self) -> bool {
        matches!(
            self,
            Entry::FillZero | Entry::CopyExact | Entry::Add | Entry::Sub | Entry::Mul | Entry::Div
        )
    }
}

/// Planned launch geometry for one kernel invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
}

impl Launch {
    /// A validated empty request: the caller skips the launch.
    pub const EMPTY: Launch = Launch {
        grid: [0, 0, 0],
        block: [FLAT_BLOCK, 1, 1],
        shared_bytes: 0,
    };

    pub fn is_empty(&self) -> bool {
        self.grid == [0, 0, 0]
    }
}

fn check_dtype(entry: Entry, dtype: DType) -> Result<(), Error> {
    if entry.supports(dtype) {
        Ok(())
    } else {
        Err(Error::UnsupportedDType)
    }
}

fn flat_launch(len: i64) -> Result<Launch, Error> {
    if len < 0 {
        return Err(Error::NegativeCount);
    }
    if len == 0 {
        return Ok(Launch::EMPTY);
    }
    let blocks = (len as u64).div_ceil(u64::from(FLAT_BLOCK));
    if blocks > u64::from(MAX_GRID_X) {
        return Err(Error::LaunchLimit);
    }
    Ok(Launch {
        grid: [blocks as u32, 1, 1],
        block: [FLAT_BLOCK, 1, 1],
        shared_bytes: 0,
    })
}

/// Plan an element-wise entry over `len` contiguous elements.
pub fn plan_elementwise(entry: Entry, dtype: DType, len: i64) -> Result<Launch, Error> {
    if !entry.is_elementwise() {
        return Err(Error::NotElementwise);
    }
    check_dtype(entry, dtype)?;
    flat_launch(len)
}

/// Plan [`Entry::FillZero`].
pub fn plan_fill_zero(dtype: DType, len: i64) -> Result<Launch, Error> {
    plan_elementwise(Entry::FillZero, dtype, len)
}

/// Plan [`Entry::FillZeroView`]: zero `dst.element_count()` logical elements.
///
/// `dst_len` is the caller's destination allocation length in elements; the
/// layout's reachable span must fit inside it.
pub fn plan_fill_zero_view(dtype: DType, dst: &Layout, dst_len: i64) -> Result<Launch, Error> {
    check_dtype(Entry::FillZeroView, dtype)?;
    let len = dst.element_count()?;
    require_destination(dst)?;
    dst.check_span(dst_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::CopyExact`].
pub fn plan_copy_exact(dtype: DType, len: i64) -> Result<Launch, Error> {
    plan_elementwise(Entry::CopyExact, dtype, len)
}

/// Plan [`Entry::MaterializeStrided`]: gather `src.element_count()` strided
/// elements into a compact destination of the same length.
pub fn plan_materialize_strided(dtype: DType, src: &Layout, src_len: i64) -> Result<Launch, Error> {
    check_dtype(Entry::MaterializeStrided, dtype)?;
    let len = src.element_count()?;
    src.check_span(src_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::ContiguousToView`]: scatter a compact source of
/// `dst.element_count()` elements into `dst`.
pub fn plan_contiguous_to_view(dtype: DType, dst: &Layout, dst_len: i64) -> Result<Launch, Error> {
    check_dtype(Entry::ContiguousToView, dtype)?;
    let len = dst.element_count()?;
    require_destination(dst)?;
    dst.check_span(dst_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::StridedToStrided`]: both layouts share the source's extents.
pub fn plan_strided_to_strided(
    dtype: DType,
    src: &Layout,
    src_len: i64,
    dst: &Layout,
    dst_len: i64,
) -> Result<Launch, Error> {
    check_dtype(Entry::StridedToStrided, dtype)?;
    if src.extents() != dst.extents() {
        return Err(Error::CountMismatch);
    }
    let len = src.element_count()?;
    require_destination(dst)?;
    src.check_span(src_len, dtype.size())?;
    dst.check_span(dst_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::TiledTranspose`] over `batches` compact `(M, N)` matrices.
///
/// `src_len` and `dst_len` are the two allocations in elements.
#[allow(clippy::too_many_arguments)]
pub fn plan_tiled_transpose(
    dtype: DType,
    config: TransposeConfig,
    src_offset: i64,
    batch_stride: i64,
    dst_fast_extent: i64,
    src_fast_extent: i64,
    batches: i64,
    src_len: i64,
    dst_len: i64,
) -> Result<Launch, Error> {
    check_dtype(Entry::TiledTranspose, dtype)?;
    if dst_fast_extent < 0 || src_fast_extent < 0 || batches < 0 || src_offset < 0 {
        return Err(Error::NegativeCount);
    }
    let matrix = dst_fast_extent
        .checked_mul(src_fast_extent)
        .ok_or(Error::Overflow)?;
    if batch_stride != matrix {
        return Err(Error::InvalidTransposeConfig);
    }
    let span = matrix.checked_mul(batches).ok_or(Error::Overflow)?;
    if src_offset.checked_add(span).ok_or(Error::Overflow)? > src_len || span > dst_len {
        return Err(Error::OffsetOutOfSpan);
    }
    let shared_bytes = config.shared_bytes(dtype.size())?;
    if shared_bytes > DEFAULT_SHARED_BYTES {
        return Err(Error::LaunchLimit);
    }
    let grid_x = div_ceil_u32(dst_fast_extent, config.tile())?;
    let grid_y = div_ceil_u32(src_fast_extent, config.tile())?;
    let grid_z = u32::try_from(batches).map_err(|_| Error::LaunchLimit)?;
    if grid_x > MAX_GRID_X || grid_y > MAX_GRID_YZ || grid_z > MAX_GRID_YZ {
        return Err(Error::LaunchLimit);
    }
    let launch = Launch {
        grid: [grid_x, grid_y, grid_z],
        block: [
            config.tile() / config.vector_width(),
            config.block_rows(),
            1,
        ],
        shared_bytes: u32::try_from(shared_bytes).map_err(|_| Error::Overflow)?,
    };
    if launch.grid.contains(&0) {
        return Ok(Launch {
            grid: [0, 0, 0],
            ..launch
        });
    }
    Ok(launch)
}

/// Diagonal output axis for extraction: the caller derives it from the two input
/// axes (`axis_a - (axis_b < axis_a)`), matching the baseline caller.
fn diagonal_output_axis(axis_a: u32, axis_b: u32) -> u32 {
    if axis_a < axis_b {
        axis_a
    } else {
        axis_a - 1
    }
}

/// Expected output extents of [`Entry::ExtractDiagonal`].
fn extract_diagonal_extents(
    in_layout: &Layout,
    axis_a: u32,
    axis_b: u32,
) -> Result<Vec<i64>, Error> {
    let mut extents: Vec<i64> = in_layout
        .extents()
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis as u32 != axis_b)
        .map(|(_, &extent)| extent)
        .collect();
    let diagonal = in_layout.extents()[axis_a as usize].min(in_layout.extents()[axis_b as usize]);
    let axis = diagonal_output_axis(axis_a, axis_b) as usize;
    extents[axis] = diagonal;
    Ok(extents)
}

/// Expected output extents of [`Entry::EmbedDiagonalCopy`].
fn embed_diagonal_extents(in_layout: &Layout, axis_a: u32, axis_b: u32) -> Vec<i64> {
    let mut extents = in_layout.extents().to_vec();
    let diagonal = in_layout.extents()[axis_a as usize];
    extents.insert(axis_b as usize, diagonal);
    extents
}

fn require_compact(layout: &Layout) -> Result<(), Error> {
    if layout.is_compact() {
        Ok(())
    } else {
        Err(Error::NotCompact)
    }
}

fn require_axis(axis: u32, rank: u32) -> Result<(), Error> {
    if axis < rank {
        Ok(())
    } else {
        Err(Error::InvalidAxis)
    }
}

fn expected_out_extents(out_layout: &Layout, expected: &[i64]) -> Result<(), Error> {
    if out_layout.extents() == expected {
        Ok(())
    } else {
        Err(Error::ShapeMismatch)
    }
}

/// Plan [`Entry::ExtractDiagonal`]. The output layout must be the compact
/// `in` without `axis_b` and with the diagonal length at
/// `axis_a - (axis_b < axis_a)`.
pub fn plan_extract_diagonal(
    dtype: DType,
    in_layout: &Layout,
    in_len: i64,
    out_layout: &Layout,
    out_len: i64,
    axis_a: u32,
    axis_b: u32,
) -> Result<Launch, Error> {
    check_dtype(Entry::ExtractDiagonal, dtype)?;
    require_axis(axis_a, in_layout.rank())?;
    require_axis(axis_b, in_layout.rank())?;
    if axis_a == axis_b {
        return Err(Error::DuplicateAxis);
    }
    let expected = extract_diagonal_extents(in_layout, axis_a, axis_b)?;
    expected_out_extents(out_layout, &expected)?;
    require_compact(out_layout)?;
    let len = out_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::EmbedDiagonalCopy`]. The input must be compact and the output
/// must be zero-filled before the launch, because only diagonal positions are
/// written; `axis_b` is an insertion position in `[0, in.rank]`.
pub fn plan_embed_diagonal_copy(
    dtype: DType,
    in_layout: &Layout,
    in_len: i64,
    out_layout: &Layout,
    out_len: i64,
    axis_a: u32,
    axis_b: u32,
) -> Result<Launch, Error> {
    check_dtype(Entry::EmbedDiagonalCopy, dtype)?;
    require_axis(axis_a, in_layout.rank())?;
    if axis_b > in_layout.rank() {
        return Err(Error::InvalidAxis);
    }
    require_compact(in_layout)?;
    require_destination(out_layout)?;
    let expected = embed_diagonal_extents(in_layout, axis_a, axis_b);
    expected_out_extents(out_layout, &expected)?;
    let len = in_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    flat_launch(len)
}

fn plan_triangular(
    entry: Entry,
    dtype: DType,
    rows: i64,
    cols: i64,
    len: i64,
) -> Result<Launch, Error> {
    check_dtype(entry, dtype)?;
    if rows < 0 || cols < 0 || len < 0 {
        return Err(Error::NegativeCount);
    }
    let matrix = rows.checked_mul(cols).ok_or(Error::Overflow)?;
    if matrix == 0 {
        if len == 0 {
            return Ok(Launch::EMPTY);
        }
        return Err(Error::ShapeMismatch);
    }
    // Higher axes batch the mask, so the length is a whole number of matrices.
    if len % matrix != 0 {
        return Err(Error::ShapeMismatch);
    }
    flat_launch(len)
}

/// Plan [`Entry::Tril`] over compact operands with `rows` and `cols` on axes 0
/// and 1; any higher axis batches the mask.
pub fn plan_tril(dtype: DType, rows: i64, cols: i64, len: i64) -> Result<Launch, Error> {
    plan_triangular(Entry::Tril, dtype, rows, cols, len)
}

/// Plan [`Entry::Triu`]; see [`plan_tril`].
pub fn plan_triu(dtype: DType, rows: i64, cols: i64, len: i64) -> Result<Launch, Error> {
    plan_triangular(Entry::Triu, dtype, rows, cols, len)
}

/// Plan [`Entry::Slice`]. `steps[a] >= 1`, `0 <= starts[a] <= in.extents[a]`,
/// and `out.extents[a]` is the slice length: at most the reachable count
/// `(in - start).div_ceil(step)`. A shorter length is the caller's explicit
/// slice limit, which the baseline expresses with its `limit` argument.
pub fn plan_slice(
    dtype: DType,
    out_layout: &Layout,
    out_len: i64,
    in_layout: &Layout,
    in_len: i64,
    starts: &[i64],
    steps: &[i64],
) -> Result<Launch, Error> {
    check_dtype(Entry::Slice, dtype)?;
    require_compact(out_layout)?;
    let rank = in_layout.rank() as usize;
    if out_layout.rank() as usize != rank || starts.len() != rank || steps.len() != rank {
        return Err(Error::RankLength);
    }
    for axis in 0..rank {
        let extent = in_layout.extents()[axis];
        let start = starts[axis];
        let step = steps[axis];
        if start < 0 || start > extent {
            return Err(Error::OffsetOutOfSpan);
        }
        if step < 1 {
            return Err(Error::InvalidStep);
        }
        let reachable = ceil_div(extent.checked_sub(start).ok_or(Error::Overflow)?, step);
        // A shorter output is the caller's slice limit, not an error.
        if out_layout.extents()[axis] > reachable {
            return Err(Error::ShapeMismatch);
        }
    }
    let len = out_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::DynamicSlice`]. `starts` is a device-resident vector of
/// `starts_rank == in.rank` index values; the window sizes are the output
/// extents, which must not exceed the input extents.
pub fn plan_dynamic_slice(
    dtype: DType,
    index_dtype: DType,
    out_layout: &Layout,
    out_len: i64,
    in_layout: &Layout,
    in_len: i64,
    starts_rank: u32,
) -> Result<Launch, Error> {
    check_dtype(Entry::DynamicSlice, dtype)?;
    if !matches!(
        index_dtype,
        DType::F32 | DType::F64 | DType::I32 | DType::I64
    ) {
        return Err(Error::UnsupportedDType);
    }
    require_compact(out_layout)?;
    let rank = in_layout.rank();
    if out_layout.rank() != rank || starts_rank != rank {
        return Err(Error::RankLength);
    }
    for axis in 0..rank as usize {
        if out_layout.extents()[axis] > in_layout.extents()[axis] {
            return Err(Error::ShapeMismatch);
        }
    }
    let len = out_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    flat_launch(len)
}

/// Plan [`Entry::Pad`]. Interior padding must be non-negative and the output
/// extents must equal the padded extents derived from the input.
#[allow(clippy::too_many_arguments)]
pub fn plan_pad(
    dtype: DType,
    out_layout: &Layout,
    out_len: i64,
    in_layout: &Layout,
    in_len: i64,
    edge_low: &[i64],
    edge_high: &[i64],
    interior: &[i64],
) -> Result<Launch, Error> {
    check_dtype(Entry::Pad, dtype)?;
    require_compact(out_layout)?;
    let rank = in_layout.rank() as usize;
    if out_layout.rank() as usize != rank
        || edge_low.len() != rank
        || edge_high.len() != rank
        || interior.len() != rank
    {
        return Err(Error::RankLength);
    }
    for axis in 0..rank {
        if interior[axis] < 0 {
            return Err(Error::InvalidPadding);
        }
        let spacing = interior[axis].checked_add(1).ok_or(Error::Overflow)?;
        let extent = in_layout.extents()[axis];
        let base = if extent == 0 {
            0
        } else {
            extent
                .checked_sub(1)
                .and_then(|value| value.checked_mul(spacing))
                .and_then(|value| value.checked_add(1))
                .ok_or(Error::Overflow)?
        };
        let expected = edge_low[axis]
            .checked_add(edge_high[axis])
            .and_then(|value| value.checked_add(base))
            .ok_or(Error::Overflow)?;
        if expected < 0 || out_layout.extents()[axis] != expected {
            return Err(Error::ShapeMismatch);
        }
    }
    let len = out_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    flat_launch(len)
}

/// One thread in one block, for the validation status entries.
fn ceil_div(value: i64, divisor: i64) -> i64 {
    // `value >= 0` and `divisor >= 1` at every call site.
    value / divisor + i64::from(value % divisor != 0)
}

fn single_thread_launch() -> Launch {
    Launch {
        grid: [1, 1, 1],
        block: [1, 1, 1],
        shared_bytes: 0,
    }
}

fn check_validation_domain(len: i64) -> Result<(), Error> {
    if len < 0 {
        return Err(Error::NegativeCount);
    }
    if len > i64::from(u32::MAX) {
        return Err(Error::DomainTooLarge);
    }
    Ok(())
}

/// Plan [`Entry::InitValidation`]: one thread writes the flag and the value slot.
pub fn plan_init_validation(dtype: DType) -> Result<Launch, Error> {
    check_dtype(Entry::InitValidation, dtype)?;
    Ok(single_thread_launch())
}

/// Plan [`Entry::ValidateFloatIndices`].
pub fn plan_validate_float_indices(dtype: DType, len: i64) -> Result<Launch, Error> {
    check_dtype(Entry::ValidateFloatIndices, dtype)?;
    check_validation_domain(len)?;
    flat_launch(len)
}

/// Plan [`Entry::ExtractInvalidFloatIndex`]: one thread reads the flag.
pub fn plan_extract_invalid_float_index(dtype: DType) -> Result<Launch, Error> {
    check_dtype(Entry::ExtractInvalidFloatIndex, dtype)?;
    Ok(single_thread_launch())
}

/// Plan [`Entry::ValidateRealCast`] sampling every `stride`-th element of a
/// `len`-element buffer. The status word holds a sampling position, so the cap
/// applies to the sample count `ceil(len / stride)`, not to `len` itself.
pub fn plan_validate_real_cast(dtype: DType, len: i64, stride: i64) -> Result<Launch, Error> {
    check_dtype(Entry::ValidateRealCast, dtype)?;
    if len < 0 {
        return Err(Error::NegativeCount);
    }
    if stride < 1 {
        return Err(Error::InvalidStride);
    }
    check_validation_domain(ceil_div(len, stride))?;
    flat_launch(ceil_div(len, stride))
}

/// Plan [`Entry::ExtractInvalidRealCast`]: one thread reads the flag.
pub fn plan_extract_invalid_real_cast(dtype: DType) -> Result<Launch, Error> {
    check_dtype(Entry::ExtractInvalidRealCast, dtype)?;
    Ok(single_thread_launch())
}

fn div_ceil_u32(extent: i64, tile: u32) -> Result<u32, Error> {
    let extent = u64::try_from(extent).map_err(|_| Error::NegativeCount)?;
    let blocks = extent.div_ceil(u64::from(tile));
    u32::try_from(blocks).map_err(|_| Error::LaunchLimit)
}

fn require_destination(layout: &Layout) -> Result<(), Error> {
    if layout.rejects_broadcast_axis() {
        Ok(())
    } else {
        Err(Error::BroadcastDestination)
    }
}

/// Byte length of `count` elements of `dtype`.
pub fn bytes(dtype: DType, count: i64) -> Result<u64, Error> {
    let count = u64::try_from(count).map_err(|_| Error::NegativeCount)?;
    count
        .checked_mul(dtype.size() as u64)
        .ok_or(Error::Overflow)
}

/// C ABI mirror of [`Layout`], passed to the kernels by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CLayout {
    pub rank: i32,
    pub reserved: i32,
    pub offset: i64,
    pub extents: [i64; MAX_RANK],
    pub strides: [i64; MAX_RANK],
}

impl CLayout {
    pub fn new(layout: &Layout) -> CLayout {
        let mut value = CLayout {
            rank: layout.rank() as i32,
            reserved: 0,
            offset: layout.offset(),
            extents: [0; MAX_RANK],
            strides: [0; MAX_RANK],
        };
        value.extents[..layout.rank() as usize].copy_from_slice(layout.extents());
        value.strides[..layout.rank() as usize].copy_from_slice(layout.strides());
        value
    }
}

impl From<&Layout> for CLayout {
    fn from(layout: &Layout) -> CLayout {
        CLayout::new(layout)
    }
}

/// C ABI mirror of [`TransposeConfig`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CTransposeConfig {
    pub tile: i32,
    pub block_rows: i32,
    pub padding: i32,
    pub vector_width: i32,
}

impl From<TransposeConfig> for CTransposeConfig {
    fn from(config: TransposeConfig) -> CTransposeConfig {
        CTransposeConfig {
            tile: config.tile() as i32,
            block_rows: config.block_rows() as i32,
            padding: config.padding() as i32,
            vector_width: config.vector_width() as i32,
        }
    }
}

// The C header asserts the same numbers; a change here is an ABI revision.
const _: () = {
    assert!(core::mem::size_of::<CLayout>() == 144);
    assert!(core::mem::align_of::<CLayout>() == 8);
    assert!(core::mem::offset_of!(CLayout, rank) == 0);
    assert!(core::mem::offset_of!(CLayout, offset) == 8);
    assert!(core::mem::offset_of!(CLayout, extents) == 16);
    assert!(core::mem::offset_of!(CLayout, strides) == 16 + 8 * MAX_RANK);
    assert!(core::mem::size_of::<CTransposeConfig>() == 16);
    assert!(core::mem::offset_of!(CTransposeConfig, block_rows) == 4);
    assert!(core::mem::offset_of!(CTransposeConfig, vector_width) == 12);
};

/// C ABI mirror of the per-axis runtime values of the windowing entries.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CAxisValues {
    pub rank: i32,
    pub reserved: i32,
    pub values: [i64; MAX_RANK],
}

impl CAxisValues {
    /// Mirror a per-axis vector; the length becomes the rank.
    pub fn new(values: &[i64]) -> Result<CAxisValues, Error> {
        if values.len() > MAX_RANK {
            return Err(Error::RankOutOfRange);
        }
        let mut mirror = CAxisValues {
            rank: values.len() as i32,
            reserved: 0,
            values: [0; MAX_RANK],
        };
        mirror.values[..values.len()].copy_from_slice(values);
        Ok(mirror)
    }
}

const _: () = {
    assert!(core::mem::size_of::<CAxisValues>() == 72);
    assert!(core::mem::align_of::<CAxisValues>() == 8);
    assert!(core::mem::offset_of!(CAxisValues, reserved) == 4);
    assert!(core::mem::offset_of!(CAxisValues, values) == 8);
};
