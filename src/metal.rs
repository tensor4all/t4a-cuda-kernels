//! The Metal source set and its planning target.
//!
//! Metal kernels are MSL, which is not CUDA-source compatible, so they are a
//! separate source set ([`source`]) that this crate ships as data and the caller
//! compiles at run time with `newLibraryWithSource:` — the same contract the CUDA
//! text has with NVRTC. Nothing here allocates, launches or synchronizes.
//!
//! Every Metal tensor is f32 or Complex32: MSL rejects `double` outright, and f64
//! work belongs to the paired CPU engine. The dtype tags, the entry names and the
//! operation tags are the ones [`crate::DType`], [`crate::Entry`], [`crate::UnaryOp`],
//! [`crate::BinaryOp`], [`crate::CompareOp`], [`crate::ScalarSide`],
//! [`crate::ReduceOp`] and [`crate::ReduceStrategy`] already define; the MSL header
//! repeats their numbers so the two sides can be checked against each other.
//!
//! Launch geometry is one threadgroup per [`THREADGROUP`] elements. Every Metal
//! entry guards its own tail, because the runtime's tiled dispatch does not
//! bound-check an index, and every entry reads its position as a 32-bit `uint`, so a
//! length above `u32::MAX` is refused with [`crate::Error::LaunchLimit`] rather than
//! planned. The component-length entries additionally refuse a length whose
//! component index would not fit 32 bits.
//!
//! A contraction or an FFT is deliberately absent: GEMM and contractions go
//! through `MPSMatrixMultiplication`, and FFT is a later unit, so neither is a
//! kernel in this catalogue. `docs/abi/c-kernel-abi.md` records that split, the
//! entries this revision does not serve, and why.

use crate::Entry;

/// The SIMD-group width of every Apple GPU: the planning input, as wave size is
/// for HIP.
pub const SIMD_WIDTH: u32 = 32;

/// Threads per threadgroup of the flat launch form, a whole number of SIMD groups.
///
/// The reduction's plane strategy uses the same width, and its threadgroup tree is
/// sized from it at compile time: the MSL header's `T4A_METAL_THREADGROUP` must
/// equal this, which the host lane checks.
pub const THREADGROUP: u32 = 8 * SIMD_WIDTH;

/// Revision of the MSL header and of the entry set it describes.
///
/// Independent of [`crate::ABI_REVISION`]: a Metal change does not revize the CUDA
/// ABI, and the two are never mixed in one module.
pub const ABI_REVISION: u32 = 2;

/// The entries this source set serves, with their MSL kernel names.
///
/// An entry missing here has no Metal kernel; its planner rejects it with
/// [`crate::Error::UnsupportedOp`] rather than returning a plan the catalogue
/// cannot keep.
pub const SYMBOLS: &[(Entry, &str)] = &[
    (Entry::FillZero, "t4a_metal_fill_zero"),
    (Entry::CopyExact, "t4a_metal_copy_exact"),
    (Entry::Add, "t4a_metal_add"),
    (Entry::Sub, "t4a_metal_sub"),
    (Entry::Mul, "t4a_metal_mul"),
    (Entry::Div, "t4a_metal_div"),
    (Entry::Unary, "t4a_metal_unary"),
    (Entry::Binary, "t4a_metal_binary"),
    (Entry::Compare, "t4a_metal_compare"),
    (Entry::Select, "t4a_metal_select"),
    (Entry::Clamp, "t4a_metal_clamp"),
    (Entry::FillZeroView, "t4a_metal_fill_zero_view"),
    (Entry::MaterializeStrided, "t4a_metal_materialize_strided"),
    (Entry::ContiguousToView, "t4a_metal_contiguous_to_view"),
    (Entry::StridedToStrided, "t4a_metal_strided_to_strided"),
    (Entry::BroadcastInDim, "t4a_metal_broadcast_in_dim"),
    (Entry::ExtractDiagonal, "t4a_metal_extract_diagonal"),
    (Entry::EmbedDiagonalCopy, "t4a_metal_embed_diagonal_copy"),
    (Entry::Tril, "t4a_metal_tril"),
    (Entry::Triu, "t4a_metal_triu"),
    (Entry::TrilView, "t4a_metal_tril_view"),
    (Entry::TriuView, "t4a_metal_triu_view"),
    (Entry::Convert, "t4a_metal_convert"),
    (Entry::Reduce, "t4a_metal_reduce"),
];

/// The MSL kernel name of `entry`, when this source set serves it.
pub fn symbol(entry: Entry) -> Option<&'static str> {
    SYMBOLS
        .iter()
        .find(|(served, _)| *served == entry)
        .map(|(_, symbol)| *symbol)
}

/// The dtypes the Metal catalogue serves.
pub const DTYPES: [crate::DType; 2] = [crate::DType::F32, crate::DType::C32];

/// The unary operations [`source`] implements.
///
/// `Expm1`, `Log1p` and `Erf` are absent because MSL has no such functions, and
/// `Conj` is absent from the f32 path because [`crate::UnaryOp::accepts`] already
/// restricts it to the complex dtypes.
pub const UNARY_OPS: [crate::UnaryOp; 11] = [
    crate::UnaryOp::Neg,
    crate::UnaryOp::Exp,
    crate::UnaryOp::Log,
    crate::UnaryOp::Sin,
    crate::UnaryOp::Cos,
    crate::UnaryOp::Tanh,
    crate::UnaryOp::Sqrt,
    crate::UnaryOp::Rsqrt,
    crate::UnaryOp::Abs,
    crate::UnaryOp::Sign,
    crate::UnaryOp::Conj,
];

/// The entries [`plan_elementwise`] plans.
///
/// They share one parameter block — a dtype tag and a length — so the plan's byte
/// size is the operand's size for both. Every other element-wise entry needs a
/// scalar, a second dtype, or a destination whose element size differs
/// ([`Entry::Compare`] writes one byte per position), and has its own planner.
pub const FLAT_ENTRIES: [Entry; 6] = [
    Entry::FillZero,
    Entry::CopyExact,
    Entry::Add,
    Entry::Sub,
    Entry::Mul,
    Entry::Div,
];

/// A checked Metal dispatch.
///
/// `threadgroups * threads_per_threadgroup` is at least the element count, so a
/// kernel must guard its tail: Metal does not bound-check an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// Threadgroups per grid, at least one.
    pub threadgroups: u32,
    /// Threads per threadgroup, [`THREADGROUP`] for the flat form.
    pub threads_per_threadgroup: u32,
}

impl Dispatch {
    /// An empty request: the caller skips the launch.
    pub const EMPTY: Dispatch = Dispatch {
        threadgroups: 0,
        threads_per_threadgroup: THREADGROUP,
    };

    /// No work to dispatch.
    pub const fn is_empty(&self) -> bool {
        self.threadgroups == 0
    }
}

/// A validated launch: the dispatch and the destination's byte size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Threadgroups and threadgroup width.
    pub dispatch: Dispatch,
    /// Bytes of the destination allocation this plan writes.
    ///
    /// The caller allocates from this, so it is the destination's element count
    /// times its element size — not always the launch's position count:
    /// [`plan_embed_diagonal_copy`] inserts an axis and writes more elements than it
    /// launches over, and [`plan_compare`] writes one byte per position.
    pub bytes: u64,
}

/// Plan [`Entry::FillZero`].
pub fn plan_fill_zero(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    plan_elementwise(Entry::FillZero, dtype, len)
}

/// Plan [`Entry::CopyExact`].
pub fn plan_copy_exact(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    plan_elementwise(Entry::CopyExact, dtype, len)
}

/// Plan a flat element-wise entry of [`FLAT_ENTRIES`] over `len` contiguous
/// elements.
///
/// This is the slice-1 entry point, narrowed to the entries whose parameter block
/// it can describe. An element-wise entry outside that set is
/// [`crate::Error::UnsupportedOp`] here — it has its own planner — and a
/// non-element-wise entry stays [`crate::Error::NotElementwise`].
pub fn plan_elementwise(entry: Entry, dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    if !entry.is_elementwise() {
        return Err(crate::Error::NotElementwise);
    }
    if !FLAT_ENTRIES.contains(&entry) {
        return Err(crate::Error::UnsupportedOp);
    }
    check_dtype(entry, dtype)?;
    plan_len(dtype, len)
}

/// Plan [`Entry::Unary`]: `len` elements transform from `src` to `dst`.
pub fn plan_unary(
    op: crate::UnaryOp,
    dst: crate::DType,
    src: crate::DType,
    len: i64,
) -> Result<Plan, crate::Error> {
    if !op.accepts(dst, src) {
        return Err(crate::Error::UnsupportedOp);
    }
    if !DTYPES.contains(&dst) || !DTYPES.contains(&src) {
        return Err(crate::Error::UnsupportedDType);
    }
    if !UNARY_OPS.contains(&op) {
        return Err(crate::Error::UnsupportedOp);
    }
    plan_len(dst, len)
}

/// Plan [`Entry::Binary`] over `len` elements.
///
/// The served (tag, dtype) pairs are the ones a Metal caller can reach: `Max`,
/// `Min`, `Pow`, `Rem`, scalar `Mul` and scalar `Div` on f32, and scalar `Mul` and
/// `DivComplex` on Complex32. `DivComplex` reads and writes two interleaved
/// components per element, so its length is also capped at `u32::MAX / 2`.
pub fn plan_binary(
    op: crate::BinaryOp,
    dtype: crate::DType,
    side: crate::ScalarSide,
    len: i64,
) -> Result<Plan, crate::Error> {
    if !op.accepts(dtype, side) {
        return Err(crate::Error::UnsupportedOp);
    }
    check_dtype(Entry::Binary, dtype)?;
    if !binary_served(op, dtype) {
        return Err(crate::Error::UnsupportedOp);
    }
    if len < 0 {
        return Err(crate::Error::NegativeCount);
    }
    if op.uses_component_length() {
        // The component position is `2 * position`; both halves must fit the
        // 32-bit index the kernel reads.
        if u64::try_from(len).map_or(true, |len| len > u64::from(u32::MAX) / 2) {
            return Err(crate::Error::LaunchLimit);
        }
    }
    plan_len(dtype, len)
}

/// Whether [`plan_binary`] serves `op` for `dtype`.
fn binary_served(op: crate::BinaryOp, dtype: crate::DType) -> bool {
    use crate::BinaryOp;
    let f32_op = matches!(
        op,
        BinaryOp::Max
            | BinaryOp::Min
            | BinaryOp::Pow
            | BinaryOp::Rem
            | BinaryOp::Mul
            | BinaryOp::Div
    );
    let c32_op = matches!(op, BinaryOp::Mul | BinaryOp::DivComplex);
    match dtype {
        crate::DType::F32 => f32_op,
        crate::DType::C32 => c32_op,
        _ => false,
    }
}

/// Plan [`Entry::Compare`]: `len` element pairs reduce to one byte each, so the
/// destination is `len` bytes.
pub fn plan_compare(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    check_dtype(Entry::Compare, dtype)?;
    Ok(plan_positions(len)?
        .with_bytes(u64::try_from(len).map_err(|_| crate::Error::NegativeCount)?))
}

/// Plan [`Entry::Select`]: `len` predicate bytes select between two operands.
pub fn plan_select(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    check_dtype(Entry::Select, dtype)?;
    plan_len(dtype, len)
}

/// Plan [`Entry::Clamp`]: `len` elements with per-element bounds.
pub fn plan_clamp(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    check_dtype(Entry::Clamp, dtype)?;
    plan_len(dtype, len)
}

/// Plan [`Entry::Convert`] over `len` contiguous elements.
///
/// The Metal dtype set is f32 and Complex32, so the served pairs are f32 → Complex32
/// and Complex32 → f32; an identity pair belongs to [`Entry::CopyExact`] and is
/// [`crate::Error::ShapeMismatch`], as in the shared planner. A pair with a dtype
/// outside the set is [`crate::Error::UnsupportedDType`].
pub fn plan_convert(dst: crate::DType, src: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    if dst == src {
        return Err(crate::Error::ShapeMismatch);
    }
    if !DTYPES.contains(&dst) || !DTYPES.contains(&src) {
        return Err(crate::Error::UnsupportedDType);
    }
    plan_len(dst, len)
}

/// Plan [`Entry::FillZeroView`]: zero `dst.element_count()` logical elements.
///
/// `dst_len` is the caller's destination allocation length in elements; the
/// layout's reachable span must fit inside it.
pub fn plan_fill_zero_view(
    dtype: crate::DType,
    dst: crate::LayoutRef<'_>,
    dst_len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::FillZeroView, dtype)?;
    require_mirror_rank(dst)?;
    let len = dst.element_count()?;
    crate::require_destination(dst)?;
    dst.check_span(dst_len, dtype.size())?;
    plan_into(dtype, len, dst_len)
}

/// Plan [`Entry::MaterializeStrided`]: gather `src.element_count()` strided
/// elements into a compact destination of the same length.
pub fn plan_materialize_strided(
    dtype: crate::DType,
    src: crate::LayoutRef<'_>,
    src_len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::MaterializeStrided, dtype)?;
    require_mirror_rank(src)?;
    let len = src.element_count()?;
    src.check_span(src_len, dtype.size())?;
    plan_len(dtype, len)
}

/// Plan [`Entry::ContiguousToView`]: scatter a compact source of `dst.element_count()`
/// elements into `dst`.
pub fn plan_contiguous_to_view(
    dtype: crate::DType,
    dst: crate::LayoutRef<'_>,
    dst_len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::ContiguousToView, dtype)?;
    require_mirror_rank(dst)?;
    let len = dst.element_count()?;
    crate::require_destination(dst)?;
    dst.check_span(dst_len, dtype.size())?;
    plan_into(dtype, len, dst_len)
}

/// Plan [`Entry::StridedToStrided`]: both layouts share the source's extents.
pub fn plan_strided_to_strided(
    dtype: crate::DType,
    src: crate::LayoutRef<'_>,
    src_len: i64,
    dst: crate::LayoutRef<'_>,
    dst_len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::StridedToStrided, dtype)?;
    require_mirror_rank(src)?;
    require_mirror_rank(dst)?;
    if src.extents() != dst.extents() {
        return Err(crate::Error::CountMismatch);
    }
    let len = src.element_count()?;
    crate::require_destination(dst)?;
    src.check_span(src_len, dtype.size())?;
    dst.check_span(dst_len, dtype.size())?;
    plan_into(dtype, len, dst_len)
}

/// Plan [`Entry::BroadcastInDim`]: a rank-mapped broadcast from `input` into the
/// compact destination `out`.
///
/// `dims[src_axis]` is the destination axis that carries the input's `src_axis`; an
/// input axis with extent one is broadcast, so its coordinate is zero. `dims.len()`
/// is the input rank and may be zero (a scalar input).
pub fn plan_broadcast_in_dim(
    dtype: crate::DType,
    out: crate::LayoutRef<'_>,
    out_len: i64,
    input: crate::LayoutRef<'_>,
    in_len: i64,
    dims: &[i64],
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::BroadcastInDim, dtype)?;
    require_mirror_rank(out)?;
    require_mirror_rank(input)?;
    crate::require_compact(out)?;
    crate::require_destination(out)?;
    if dims.len() != input.rank() as usize {
        return Err(crate::Error::RankLength);
    }
    if dims.len() > crate::MAX_RANK {
        return Err(crate::Error::RankOutOfRange);
    }
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        if dst_axis < 0 || dst_axis >= out.rank() as i64 {
            return Err(crate::Error::InvalidAxis);
        }
        let dst_axis = dst_axis as usize;
        if dims[..src_axis].contains(&(dst_axis as i64)) {
            return Err(crate::Error::DuplicateAxis);
        }
        let in_extent = input.extents()[src_axis];
        let out_extent = out.extents()[dst_axis];
        if in_extent != out_extent && in_extent != 1 {
            return Err(crate::Error::ShapeMismatch);
        }
    }
    let len = out.element_count()?;
    input.check_span(in_len, dtype.size())?;
    out.check_span(out_len, dtype.size())?;
    plan_len(dtype, len)
}

/// Plan [`Entry::ExtractDiagonal`]. The output layout must be the compact `in`
/// without `axis_b` and with the diagonal length at `axis_a - (axis_b < axis_a)`.
#[allow(clippy::too_many_arguments)]
pub fn plan_extract_diagonal(
    dtype: crate::DType,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
    out_layout: crate::LayoutRef<'_>,
    out_len: i64,
    axis_a: u32,
    axis_b: u32,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::ExtractDiagonal, dtype)?;
    require_mirror_rank(in_layout)?;
    require_mirror_rank(out_layout)?;
    crate::require_axis(axis_a, in_layout.rank())?;
    crate::require_axis(axis_b, in_layout.rank())?;
    if axis_a == axis_b {
        return Err(crate::Error::DuplicateAxis);
    }
    let expected = crate::extract_diagonal_extents(in_layout, axis_a, axis_b)?;
    crate::expected_out_extents(out_layout, &expected)?;
    crate::require_compact(out_layout)?;
    let len = out_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    plan_len(dtype, len)
}

/// Plan [`Entry::EmbedDiagonalCopy`]. The input must be compact and the output must
/// be zero-filled before the launch, because only diagonal positions are written;
/// `axis_b` is an insertion position in `[0, in.rank]`.
///
/// The launch covers the input's `element_count()` positions, but the destination
/// gains the inserted axis, so [`Plan::bytes`] is the *output* allocation.
#[allow(clippy::too_many_arguments)]
pub fn plan_embed_diagonal_copy(
    dtype: crate::DType,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
    out_layout: crate::LayoutRef<'_>,
    out_len: i64,
    axis_a: u32,
    axis_b: u32,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::EmbedDiagonalCopy, dtype)?;
    require_mirror_rank(in_layout)?;
    require_mirror_rank(out_layout)?;
    // The output gains the inserted axis, so it is one rank above the input.
    if in_layout.rank() as usize + 1 > crate::MAX_RANK {
        return Err(crate::Error::RankOutOfRange);
    }
    crate::require_axis(axis_a, in_layout.rank())?;
    if axis_b > in_layout.rank() {
        return Err(crate::Error::InvalidAxis);
    }
    crate::require_compact(in_layout)?;
    crate::require_destination(out_layout)?;
    let expected = crate::embed_diagonal_extents(in_layout, axis_a, axis_b);
    crate::expected_out_extents(out_layout, &expected)?;
    let len = in_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    out_layout.check_span(out_len, dtype.size())?;
    plan_into(dtype, len, out_len)
}

/// Plan [`Entry::Tril`] over compact operands with `rows` and `cols` on axes 0 and
/// 1; any higher axis batches the mask.
pub fn plan_tril(
    dtype: crate::DType,
    rows: i64,
    cols: i64,
    len: i64,
) -> Result<Plan, crate::Error> {
    plan_triangular(Entry::Tril, dtype, rows, cols, len)
}

/// Plan [`Entry::Triu`]; see [`plan_tril`].
pub fn plan_triu(
    dtype: crate::DType,
    rows: i64,
    cols: i64,
    len: i64,
) -> Result<Plan, crate::Error> {
    plan_triangular(Entry::Triu, dtype, rows, cols, len)
}

fn plan_triangular(
    entry: Entry,
    dtype: crate::DType,
    rows: i64,
    cols: i64,
    len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(entry, dtype)?;
    if rows < 0 || cols < 0 || len < 0 {
        return Err(crate::Error::NegativeCount);
    }
    let matrix = rows.checked_mul(cols).ok_or(crate::Error::Overflow)?;
    if matrix == 0 {
        if len == 0 {
            return plan_len(dtype, 0);
        }
        return Err(crate::Error::ShapeMismatch);
    }
    if len % matrix != 0 {
        return Err(crate::Error::ShapeMismatch);
    }
    plan_len(dtype, len)
}

/// Plan [`Entry::TrilView`]: mask the triangle of axes 0 and 1 of a strided input
/// into a compact output that has the input's extents.
///
/// `k` is a launch scalar and is not part of the plan. The output takes no layout
/// argument: its offset is zero and its compact strides follow from the input's
/// extents, so `out_len` is only the caller's output allocation length and must hold
/// `in.element_count()` elements.
pub fn plan_tril_view(
    dtype: crate::DType,
    out_len: i64,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
) -> Result<Plan, crate::Error> {
    plan_triangular_view(Entry::TrilView, dtype, out_len, in_layout, in_len)
}

/// Plan [`Entry::TriuView`]; see [`plan_tril_view`].
pub fn plan_triu_view(
    dtype: crate::DType,
    out_len: i64,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
) -> Result<Plan, crate::Error> {
    plan_triangular_view(Entry::TriuView, dtype, out_len, in_layout, in_len)
}

fn plan_triangular_view(
    entry: Entry,
    dtype: crate::DType,
    out_len: i64,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
) -> Result<Plan, crate::Error> {
    check_dtype(entry, dtype)?;
    require_mirror_rank(in_layout)?;
    crate::require_matrix(in_layout)?;
    let len = in_layout.element_count()?;
    in_layout.check_span(in_len, dtype.size())?;
    if out_len < 0 {
        return Err(crate::Error::NegativeCount);
    }
    if out_len < len {
        return Err(crate::Error::OffsetOutOfSpan);
    }
    plan_len(dtype, len)
}

/// Plan [`Entry::Reduce`] over one axis with a keepdims output.
///
/// The strategy's threadgroup width is [`THREADGROUP`]: `Unit` is legal when the axis
/// is no longer than one threadgroup, `Plane` when it is at least that long. The
/// output length bounds the threadgroup count in both strategies, so it must fit
/// `u32`.
#[allow(clippy::too_many_arguments)]
pub fn plan_reduce(
    dtype: crate::DType,
    op: crate::ReduceOp,
    strategy: crate::ReduceStrategy,
    out_layout: crate::LayoutRef<'_>,
    out_len: i64,
    in_layout: crate::LayoutRef<'_>,
    in_len: i64,
    axis: u32,
) -> Result<Plan, crate::Error> {
    check_dtype(Entry::Reduce, dtype)?;
    require_mirror_rank(out_layout)?;
    require_mirror_rank(in_layout)?;
    let (reduce_len, output_len) =
        crate::reduce_problem(dtype, op, out_layout, out_len, in_layout, in_len, axis)?;
    // A plane lane steps by the threadgroup width, so the axis must fit the 32-bit
    // position its walk is bounded by.
    if u64::try_from(reduce_len).map_or(true, |len| len > u64::from(u32::MAX)) {
        return Err(crate::Error::LaunchLimit);
    }
    crate::check_strategy(strategy, reduce_len, THREADGROUP)?;
    let plan = plan_into(dtype, output_len, out_len)?;
    match strategy {
        crate::ReduceStrategy::Unit => Ok(plan),
        crate::ReduceStrategy::Plane => {
            // One threadgroup per output, so the output length is the grid.
            let threadgroups = u32::try_from(output_len).map_err(|_| crate::Error::LaunchLimit)?;
            Ok(plan.with_threadgroups(threadgroups))
        }
    }
}

/// Reject a dtype the entry and the Metal set both must serve.
fn check_dtype(entry: Entry, dtype: crate::DType) -> Result<(), crate::Error> {
    if !entry.supports(dtype) || !DTYPES.contains(&dtype) {
        return Err(crate::Error::UnsupportedDType);
    }
    Ok(())
}

/// Refuse a negative length, then compute the geometry.
///
/// Every entry receives its position as a 32-bit `uint` (`[[thread_position_in_grid]]`
/// or `[[threadgroup_position_in_grid]]`), so `len` above `u32::MAX` names elements
/// no thread can reach; that is a device limit, not an arithmetic accident, and it
/// is refused as such. It also keeps `bytes` well inside `u64` for every dtype the
/// set can serve.
fn plan_len(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    Ok(plan_positions(len)?.with_bytes(crate::bytes(dtype, len)?))
}

/// Reject a layout the 144-byte mirror cannot carry.
fn require_mirror_rank(layout: crate::LayoutRef<'_>) -> Result<(), crate::Error> {
    if layout.rank() as usize > crate::MAX_RANK {
        return Err(crate::Error::RankOutOfRange);
    }
    Ok(())
}

/// The plan for an entry that writes `positions` positions into a destination whose
/// validated allocation is `alloc_len` elements.
///
/// [`Plan::bytes`] is the allocation the caller must provide, not the logical element
/// count: a view can place its last element past `element_count()`, and the engine
/// allocates exactly this many bytes.
fn plan_into(dtype: crate::DType, positions: i64, alloc_len: i64) -> Result<Plan, crate::Error> {
    if positions == 0 {
        return plan_positions(0);
    }
    Ok(plan_positions(positions)?.with_bytes(crate::bytes(dtype, alloc_len)?))
}

/// Validate the position count and compute the flat geometry.
fn plan_positions(len: i64) -> Result<Plan, crate::Error> {
    if len < 0 {
        return Err(crate::Error::NegativeCount);
    }
    if u64::try_from(len).map_or(true, |len| len > u64::from(u32::MAX)) {
        return Err(crate::Error::LaunchLimit);
    }
    if len == 0 {
        return Ok(Plan {
            dispatch: Dispatch::EMPTY,
            bytes: 0,
        });
    }
    let threadgroups = (len as u64).div_ceil(u64::from(THREADGROUP));
    Ok(Plan {
        dispatch: Dispatch {
            threadgroups: threadgroups as u32,
            threads_per_threadgroup: THREADGROUP,
        },
        bytes: 0,
    })
}

impl Plan {
    /// The same dispatch with a different destination byte size.
    #[must_use]
    fn with_bytes(self, bytes: u64) -> Plan {
        Plan { bytes, ..self }
    }

    /// The same byte size with an explicit threadgroup count.
    #[must_use]
    fn with_threadgroups(self, threadgroups: u32) -> Plan {
        Plan {
            dispatch: Dispatch {
                threadgroups,
                threads_per_threadgroup: self.dispatch.threads_per_threadgroup,
            },
            ..self
        }
    }
}

/// The by-value layout a Metal kernel reads, from a borrowed layout.
///
/// Slots past `rank` are zero-filled, so the whole 144-byte struct is initialised and
/// [`layout_bytes`] can hand it to `setBytes:`.
pub fn layout_mirror(layout: crate::LayoutRef<'_>) -> Result<crate::CLayout, crate::Error> {
    let rank = layout.rank() as usize;
    if rank > crate::MAX_RANK {
        return Err(crate::Error::RankOutOfRange);
    }
    let mut mirror = crate::CLayout {
        rank: rank as i32,
        reserved: 0,
        offset: layout.offset(),
        extents: [0; crate::MAX_RANK],
        strides: [0; crate::MAX_RANK],
    };
    mirror.extents[..rank].copy_from_slice(layout.extents());
    mirror.strides[..rank].copy_from_slice(layout.strides());
    Ok(mirror)
}

/// Byte length of [`layout_bytes`]'s encoding: the C struct's size.
pub const LAYOUT_BYTES: usize = 144;

/// Byte length of [`axis_values_bytes`]'s encoding: the C struct's size.
pub const AXIS_VALUES_BYTES: usize = 72;

/// Encode a layout mirror as the by-value kernel parameter the MSL header
/// declares.
///
/// The encoding is field by field in declaration order and little-endian, which is
/// the C struct's representation on every host Apple silicon supports; it is written
/// out rather than reinterpreted because this crate forbids unsafe code. `setBytes:`
/// copies the result into the command stream, so it may be dropped as soon as the
/// launch returns.
pub fn layout_bytes(mirror: &crate::CLayout) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(LAYOUT_BYTES);
    bytes.extend_from_slice(&mirror.rank.to_le_bytes());
    bytes.extend_from_slice(&mirror.reserved.to_le_bytes());
    bytes.extend_from_slice(&mirror.offset.to_le_bytes());
    for extent in &mirror.extents {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    for stride in &mirror.strides {
        bytes.extend_from_slice(&stride.to_le_bytes());
    }
    assert_eq!(bytes.len(), LAYOUT_BYTES, "CLayout is 144 bytes of fields");
    bytes
}

/// Encode a per-axis value mirror as the by-value kernel parameter the MSL header
/// declares; see [`layout_bytes`] for the encoding rule.
pub fn axis_values_bytes(values: &crate::CAxisValues) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(AXIS_VALUES_BYTES);
    bytes.extend_from_slice(&values.rank.to_le_bytes());
    bytes.extend_from_slice(&values.reserved.to_le_bytes());
    for value in &values.values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        bytes.len(),
        AXIS_VALUES_BYTES,
        "CAxisValues is 72 bytes of fields"
    );
    bytes
}

/// The MSL sources a caller compiles, composed from this crate's own text.
///
/// `newLibraryWithSource:` has no include path, so a module carries no include
/// line of its own and [`compose`] puts [`ABI_HEADER`] in front of it — the same
/// contract [`crate::source::compose`] has with NVRTC.
pub mod source {
    use super::symbol;

    /// The shared MSL header, prepended to every module.
    pub const ABI_HEADER: &str = include_str!("../metal/include/t4a/abi.msl");

    /// One translation unit: the module name and its source text.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Module {
        /// Module name, matching `metal/src/<name>.msl`.
        pub name: &'static str,
        /// The module's source text.
        pub source: &'static str,
    }

    /// Every translation unit the Metal entries are exported from.
    pub const MODULES: &[Module] = &[
        Module {
            name: "elementwise",
            source: include_str!("../metal/src/elementwise.msl"),
        },
        Module {
            name: "movement",
            source: include_str!("../metal/src/movement.msl"),
        },
        Module {
            name: "diagonal",
            source: include_str!("../metal/src/diagonal.msl"),
        },
        Module {
            name: "convert",
            source: include_str!("../metal/src/convert.msl"),
        },
        Module {
            name: "reduce",
            source: include_str!("../metal/src/reduce.msl"),
        },
    ];

    /// The module that exports `entry`, from [`super::SYMBOLS`].
    pub fn module_of(entry: crate::Entry) -> Option<&'static Module> {
        let symbol = symbol(entry)?;
        MODULES
            .iter()
            .find(|module| module.source.contains(&format!("{symbol}(")))
    }

    /// The include line every module starts with.
    pub const INCLUDE_LINE: &str = "#include <t4a/abi.msl>";

    /// One module's source preceded by the shared header, ready for
    /// `newLibraryWithSource:`.
    ///
    /// The library has no include path, so the header is prepended and the
    /// module's own include line removed; the composition is otherwise the file
    /// text.
    pub fn compose(module: &Module) -> String {
        let mut text = String::with_capacity(ABI_HEADER.len() + module.source.len() + 1);
        text.push_str(ABI_HEADER);
        text.push('\n');
        for line in module.source.lines() {
            if line.trim() == INCLUDE_LINE {
                continue;
            }
            text.push_str(line);
            text.push('\n');
        }
        text
    }
}
