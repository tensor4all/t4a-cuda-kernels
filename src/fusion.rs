//! Fused element-wise expression emitter.
//!
//! A [`FusionPlan`] is a topologically ordered instruction list over the element-wise
//! operations revision 6 defines. [`emit_fusion`] turns it into a self-contained CUDA
//! C++ kernel (`t4a_fused_elementwise`), a manifest and the canonical artifact key.
//!
//! This module generates *source*; it never compiles, loads or launches anything. The
//! caller compiles the emitted source (NVRTC or NVCC), loads the module in its own
//! context and caches both under [`EmittedFusion::key`]. See
//! `docs/design/slice-fusion-emitter.md` for the contract.
//!
//! Classification: the dtype, the op set, their arity and the output count are
//! compile-time because they shape the emitted code; the element count, every buffer
//! address and the launch geometry are runtime and never enter the key.

use crate::{flat_launch, DType, Error, Launch, FLAT_BLOCK};

/// Largest accepted CUDA kernel-parameter block. CUDA 12.0 allows 4 KiB of kernel
/// parameters; the emitted entry passes one pointer per operand plus `len`, so the
/// operand count is bounded by this budget as well as by the instruction count.
pub const MAX_FUSION_PARAMETER_BYTES: usize = 4096;

/// Largest accepted instruction count. A fused region is a handful of element-wise
/// operations; the bound keeps a pathological plan from producing unbounded source.
pub const MAX_FUSION_INSTRUCTIONS: usize = 256;

/// The canonical CUDA ABI header, embedded verbatim into every emitted source so the
/// helper semantics and their notices come from the single canonical definitions
/// rather than a second copy that could drift.
const ABI_HEADER: &str = include_str!("../cuda/include/t4a/abi.cuh");

/// Entry point of every emitted kernel.
pub const FUSION_ENTRY: &str = "t4a_fused_elementwise";

/// One element-wise operation a fused region may contain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionOp {
    Add,
    Mul,
    Div,
    Neg,
    Abs,
    Conj,
    Max,
    Min,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    Erf,
}

impl FusionOp {
    /// Number of operands this operation takes.
    pub const fn arity(self) -> usize {
        match self {
            FusionOp::Clamp => 3,
            FusionOp::Neg
            | FusionOp::Abs
            | FusionOp::Conj
            | FusionOp::Exp
            | FusionOp::Log
            | FusionOp::Sin
            | FusionOp::Cos
            | FusionOp::Tanh
            | FusionOp::Sqrt
            | FusionOp::Rsqrt
            | FusionOp::Expm1
            | FusionOp::Log1p
            | FusionOp::Erf => 1,
            FusionOp::Add
            | FusionOp::Mul
            | FusionOp::Div
            | FusionOp::Max
            | FusionOp::Min
            | FusionOp::Pow => 2,
        }
    }

    /// Whether this operation is eligible for `dtype`, matching the pinned baseline's
    /// `classify.rs` matrix: the real dtypes take the full set, the complex dtypes only
    /// the algebraic ones.
    pub const fn accepts(self, dtype: DType) -> bool {
        let real = matches!(dtype, DType::F32 | DType::F64);
        let complex = matches!(dtype, DType::C32 | DType::C64);
        match self {
            FusionOp::Add | FusionOp::Mul | FusionOp::Div | FusionOp::Neg => real || complex,
            // `conj` is meaningful only for complex payloads and `abs` only for the real
            // (and integer) ones; the pinned classifier lists exactly that split.
            FusionOp::Conj => complex,
            FusionOp::Abs
            | FusionOp::Max
            | FusionOp::Min
            | FusionOp::Clamp
            | FusionOp::Exp
            | FusionOp::Log
            | FusionOp::Sin
            | FusionOp::Cos
            | FusionOp::Tanh
            | FusionOp::Sqrt
            | FusionOp::Rsqrt
            | FusionOp::Pow
            | FusionOp::Expm1
            | FusionOp::Log1p
            | FusionOp::Erf => real,
        }
    }
}

/// Operand of a fused instruction: a region input or an earlier instruction's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionOperand {
    Input(u32),
    Value(u32),
}

/// One instruction of a fused region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionInstruction {
    pub op: FusionOp,
    pub operands: Vec<FusionOperand>,
}

/// A fused element-wise region: one dtype, `inputs` identity-view operands, a
/// topologically ordered instruction list and the live-out values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionPlan {
    pub dtype: DType,
    pub inputs: u32,
    pub instructions: Vec<FusionInstruction>,
    pub outputs: Vec<FusionOperand>,
}

/// What the caller needs to compile, load and launch an emitted kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusionManifest {
    pub entry: &'static str,
    pub abi_revision: u32,
    pub dtype: DType,
    pub inputs: u32,
    pub outputs: u32,
    pub block: u32,
    pub index_width: u32,
}

/// Canonical artifact-key material: the ordered, length-prefixed inputs that decide
/// whether two prepared kernels are the same artifact.
///
/// It is deliberately not a digest. It implements [`Eq`] and [`Hash`], so an in-memory
/// cache can key on it directly, and a persistent cache hashes it with whatever digest
/// it already trusts. Nothing can be dropped from the key silently, because the
/// material *is* the list of inputs, in order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey {
    material: Vec<u8>,
}

impl ArtifactKey {
    /// The key material, for a caller that wants to store or hash it.
    pub fn material(&self) -> &[u8] {
        &self.material
    }
}

/// An emitted kernel: source text, manifest and artifact key.
///
/// The three are read-only together: a caller clones the bundle, not one field, so the
/// key can never describe a different source than the one it hands to its compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedFusion {
    source: String,
    manifest: FusionManifest,
    key: ArtifactKey,
}

impl EmittedFusion {
    /// The generated CUDA C++ source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The manifest describing the emitted entry.
    pub fn manifest(&self) -> FusionManifest {
        self.manifest
    }

    /// The canonical artifact key of this source and plan.
    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }
}

/// Plan the launch of an emitted kernel over `len` logical elements.
///
/// The grid follows each runtime length; a prepared artifact reuses its code and
/// bindings, never a stale grid.
pub fn plan_fused(len: i64) -> Result<Launch, Error> {
    flat_launch(len)
}

/// Validate `plan` and emit the CUDA C++ source, manifest and artifact key.
pub fn emit_fusion(plan: &FusionPlan) -> Result<EmittedFusion, Error> {
    validate(plan)?;

    let dtype = plan.dtype;
    let scalar = scalar_name(dtype);
    let complex = matches!(dtype, DType::C32 | DType::C64);

    let mut source = String::with_capacity(ABI_HEADER.len() + 256 * 48);
    source.push_str("// Generated by t4a-cuda-kernels: fused element-wise region.\n");
    source.push_str(
        "// The canonical helper definitions follow verbatim from the CUDA ABI header.\n",
    );
    source.push_str(ABI_HEADER);
    source.push('\n');
    source.push_str("extern \"C\" __global__ void ");
    source.push_str(FUSION_ENTRY);
    source.push('(');
    let mut first = true;
    for output in 0..plan.outputs.len() {
        if !first {
            source.push_str(", ");
        }
        first = false;
        source.push_str(&format!("{scalar}* out{output}"));
    }
    for input in 0..plan.inputs {
        source.push_str(&format!(", const {scalar}* in{input}"));
    }
    source.push_str(
        ", long long len) {\n  const long long index = \
         static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n  \
         if (index >= len) {\n    return;\n  }\n",
    );
    for input in 0..plan.inputs {
        source.push_str(&format!("  const {scalar} v{input} = in{input}[index];\n"));
    }
    for (position, instruction) in plan.instructions.iter().enumerate() {
        source.push_str(&format!(
            "  const {scalar} v{} = {};\n",
            plan.inputs as usize + position,
            render(instruction, complex, plan.inputs),
        ));
    }
    for (output, operand) in plan.outputs.iter().enumerate() {
        source.push_str(&format!(
            "  out{output}[index] = {};\n",
            operand_text(*operand, plan.inputs),
        ));
    }
    source.push_str("}\n");

    let manifest = FusionManifest {
        entry: FUSION_ENTRY,
        abi_revision: crate::ABI_REVISION,
        dtype,
        inputs: plan.inputs,
        outputs: plan.outputs.len() as u32,
        block: FLAT_BLOCK,
        index_width: 64,
    };

    let mut material = Vec::new();
    push_field(&mut material, &crate::ABI_REVISION.to_le_bytes());
    push_text(&mut material, FUSION_ENTRY);
    push_field(&mut material, &[dtype.tag() as u8]);
    push_field(&mut material, &plan.inputs.to_le_bytes());
    push_field(&mut material, &(plan.outputs.len() as u32).to_le_bytes());
    push_text(&mut material, &source);

    Ok(EmittedFusion {
        source,
        manifest,
        key: ArtifactKey { material },
    })
}

/// Complete the artifact key with the caller's compiler identity.
///
/// Two prepared kernels are the same artifact only when the canonical material, the
/// compiler options, the target (architecture and output kind) and the compiler
/// version all agree.
pub fn artifact_key(
    emitted: &EmittedFusion,
    options: &str,
    target: &str,
    compiler: &str,
) -> ArtifactKey {
    let mut material = emitted.key.material.clone();
    push_text(&mut material, options);
    push_text(&mut material, target);
    push_text(&mut material, compiler);
    ArtifactKey { material }
}

fn push_field(material: &mut Vec<u8>, field: &[u8]) {
    material.extend_from_slice(&(field.len() as u64).to_le_bytes());
    material.extend_from_slice(field);
}

fn push_text(material: &mut Vec<u8>, field: &str) {
    push_field(material, field.as_bytes());
}

fn validate(plan: &FusionPlan) -> Result<(), Error> {
    if plan.outputs.is_empty() || plan.inputs == 0 {
        return Err(Error::FusionEmpty);
    }
    // The dtype is checked before anything else: a plan with no instructions would
    // otherwise emit `float*` accesses for a one-byte Bool plan.
    if !matches!(
        plan.dtype,
        DType::F32 | DType::F64 | DType::C32 | DType::C64
    ) {
        return Err(Error::FusionEligibility);
    }
    if plan.instructions.len() > MAX_FUSION_INSTRUCTIONS {
        return Err(Error::FusionTooLarge);
    }
    // One pointer per input and per output, plus the element count.
    let parameters = (plan.inputs as usize)
        .checked_add(plan.outputs.len())
        .and_then(|count| count.checked_add(1))
        .and_then(|count| count.checked_mul(8))
        .ok_or(Error::FusionTooLarge)?;
    if parameters > MAX_FUSION_PARAMETER_BYTES {
        return Err(Error::FusionTooLarge);
    }
    for (position, instruction) in plan.instructions.iter().enumerate() {
        if !instruction.op.accepts(plan.dtype) {
            return Err(Error::FusionEligibility);
        }
        if instruction.operands.len() != instruction.op.arity() {
            return Err(Error::FusionArity);
        }
        for operand in &instruction.operands {
            match *operand {
                FusionOperand::Input(index) if index < plan.inputs => {}
                FusionOperand::Value(index) if (index as usize) < position => {}
                _ => return Err(Error::FusionOperand),
            }
        }
    }
    let available = plan.instructions.len() as u32;
    for operand in &plan.outputs {
        match *operand {
            FusionOperand::Input(index) if index < plan.inputs => {}
            FusionOperand::Value(index) if index < available => {}
            _ => return Err(Error::FusionOperand),
        }
    }
    Ok(())
}

/// Region values are emitted as `v0..` for the inputs and continue past them for the
/// instructions, so an instruction value's name is its operand index shifted by the
/// input count.
fn operand_text(operand: FusionOperand, inputs: u32) -> String {
    match operand {
        FusionOperand::Input(index) => format!("v{index}"),
        FusionOperand::Value(index) => format!("v{}", inputs + index),
    }
}

fn scalar_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::C32 => "t4a_cuda_c32",
        DType::C64 => "t4a_cuda_c64",
        DType::I32 | DType::I64 | DType::Bool => "float",
    }
}

/// One instruction as a C++ expression over the canonical helpers, so a fused chain
/// and the equivalent per-operation launches share one definition of each operation.
fn render(instruction: &FusionInstruction, complex: bool, inputs: u32) -> String {
    let operand = |position: usize| operand_text(instruction.operands[position], inputs);
    let lhs = operand(0);
    match instruction.op {
        FusionOp::Add => format!("t4a_cuda_add({lhs}, {})", operand(1)),
        FusionOp::Mul => format!("t4a_cuda_mul({lhs}, {})", operand(1)),
        FusionOp::Div if complex => format!("t4a_cuda_div_complex({lhs}, {})", operand(1)),
        FusionOp::Div => format!("({lhs} / {})", operand(1)),
        FusionOp::Neg => format!("t4a_cuda_neg({lhs})"),
        FusionOp::Abs => format!("t4a_cuda_fabs({lhs})"),
        FusionOp::Conj => format!("t4a_cuda_conj({lhs})"),
        FusionOp::Max => format!("t4a_cuda_max({lhs}, {})", operand(1)),
        FusionOp::Min => format!("t4a_cuda_min({lhs}, {})", operand(1)),
        FusionOp::Clamp => format!("t4a_cuda_clamp({lhs}, {}, {})", operand(1), operand(2)),
        FusionOp::Exp => format!("t4a_cuda_exp({lhs})"),
        FusionOp::Log => format!("t4a_cuda_log({lhs})"),
        FusionOp::Sin => format!("t4a_cuda_sin({lhs})"),
        FusionOp::Cos => format!("t4a_cuda_cos({lhs})"),
        FusionOp::Tanh => format!("t4a_cuda_tanh({lhs})"),
        FusionOp::Sqrt => format!("t4a_cuda_sqrt({lhs})"),
        FusionOp::Rsqrt => format!("t4a_cuda_rsqrt({lhs})"),
        FusionOp::Pow => format!("t4a_cuda_pow({lhs}, {})", operand(1)),
        FusionOp::Expm1 => format!("t4a_cuda_expm1({lhs})"),
        FusionOp::Log1p => format!("t4a_cuda_log1p({lhs})"),
        FusionOp::Erf => format!("t4a_cuda_erf({lhs})"),
    }
}
