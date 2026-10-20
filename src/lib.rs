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

#![forbid(unsafe_code)]

use core::fmt;

/// Maximum supported tensor rank. Deeper layouts need a later ABI revision.
pub const MAX_RANK: usize = 8;
/// Version of the C ABI described by this crate.
pub const ABI_REVISION: u32 = 1;
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
}

impl Entry {
    pub const ALL: [Entry; 11] = [
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
        }
    }

    /// Dtypes this entry serves. Arithmetic excludes Bool, matching the
    /// baseline dispatch; division is float-only.
    pub const fn supported_dtypes(self) -> &'static [DType] {
        match self {
            Entry::Div => &[DType::F32, DType::F64],
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
