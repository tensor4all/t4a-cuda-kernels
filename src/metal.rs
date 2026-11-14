//! The Metal source set and its planning target.
//!
//! Metal kernels are MSL, which is not CUDA-source compatible, so they are a
//! separate source set ([`source`]) that this crate ships as data and the caller
//! compiles at run time with `newLibraryWithSource:` — the same contract the CUDA
//! text has with NVRTC. Nothing here allocates, launches or synchronizes.
//!
//! Every Metal tensor is f32 or Complex32: MSL rejects `double` outright, and f64
//! work belongs to the paired CPU engine. The dtype tags, the entry names and the
//! unary tags are the ones [`crate::DType`], [`crate::Entry`] and [`crate::UnaryOp`]
//! already define; the MSL header repeats their numbers so the two sides can be
//! checked against each other.
//!
//! Launch geometry is one threadgroup per [`THREADGROUP`] elements with the
//! SIMD-group width as a planning input. Every Metal entry guards its own tail,
//! because the runtime's tiled dispatch does not bound-check an index, and every
//! entry reads its position as a 32-bit `uint`, so a length above `u32::MAX` is
//! refused with [`crate::Error::LaunchLimit`] rather than planned.
//!
//! A contraction or an FFT is deliberately absent: GEMM and contractions go
//! through `MPSMatrixMultiplication`, and FFT is a later unit, so neither is a
//! kernel in this catalogue. `docs/abi/c-kernel-abi.md` records that split.

use crate::Entry;

/// The SIMD-group width of every Apple GPU: the planning input, as wave size is
/// for HIP.
pub const SIMD_WIDTH: u32 = 32;

/// Threads per threadgroup of the flat launch form, a whole number of SIMD groups.
pub const THREADGROUP: u32 = 8 * SIMD_WIDTH;

/// Revision of the MSL header and of the entry set it describes.
///
/// Independent of [`crate::ABI_REVISION`]: a Metal change does not revize the CUDA
/// ABI, and the two are never mixed in one module.
pub const ABI_REVISION: u32 = 1;

/// The entries this source set serves, with their MSL kernel names.
///
/// An entry missing here has no Metal kernel; [`plan_elementwise`] rejects it with
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
    /// Bytes of one operand of this dtype at this length.
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

/// Plan a flat element-wise entry over `len` contiguous elements.
pub fn plan_elementwise(entry: Entry, dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    if !entry.is_elementwise() {
        return Err(crate::Error::NotElementwise);
    }
    if !entry.supports(dtype) || !DTYPES.contains(&dtype) {
        return Err(crate::Error::UnsupportedDType);
    }
    if symbol(entry).is_none() {
        return Err(crate::Error::UnsupportedOp);
    }
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

/// Refuse a length the kernels cannot address, then compute the geometry.
///
/// Every entry receives its position as a 32-bit `uint` (`[[thread_position_in_grid]]`),
/// so `len` above `u32::MAX` names elements no thread can reach; that is a device
/// limit, not an arithmetic accident, and it is refused as such. It also keeps
/// `bytes` well inside `u64` for every dtype the set can serve.
fn plan_len(dtype: crate::DType, len: i64) -> Result<Plan, crate::Error> {
    if len < 0 {
        return Err(crate::Error::NegativeCount);
    }
    if u64::try_from(len).map_or(true, |len| len > u64::from(u32::MAX)) {
        return Err(crate::Error::LaunchLimit);
    }
    let bytes = (len as u64) * dtype.size() as u64;
    if len == 0 {
        return Ok(Plan {
            dispatch: Dispatch::EMPTY,
            bytes,
        });
    }
    let threadgroups = (len as u64).div_ceil(u64::from(THREADGROUP));
    Ok(Plan {
        dispatch: Dispatch {
            threadgroups: threadgroups as u32,
            threads_per_threadgroup: THREADGROUP,
        },
        bytes,
    })
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
    pub const MODULES: &[Module] = &[Module {
        name: "elementwise",
        source: include_str!("../metal/src/elementwise.msl"),
    }];

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
