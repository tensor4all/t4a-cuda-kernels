# Slice H: fused-expression emitter and prepare/cache contract

Status: design, 2026-10-21. Ninth and last implementation slice of
[the migration design](cubecl-kernel-port.md). This one is not a port: the pinned
baseline builds CubeCL IR at run time
(`crates/tenferro-gpu/src/cubecl/fusion/{classify,codegen,launch}.rs`), and the
migration replaces that with an **independent expression description plus a small
CUDA C++ source emitter** whose artifact the caller compiles and caches.

## 1. Scope

The crate gains a Rust-side API that turns a runtime-independent operation
description into a self-contained CUDA C++ source, a manifest and an artifact key.
It gains no new `t4a_cuda_*` kernel entry (the emitted entry is
`t4a_fused_elementwise`), no dependency, and no compilation: **compiling and loading
stay a caller/runtime service**, as the migration design requires.

Preserved from the baseline:

- eligibility: F32/F64 support add, multiply, divide, negate, abs, maximum, minimum,
  clamp, exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p and erf; C32/C64
  support add, multiply, divide, negate and conj (the pinned `classify.rs` matrix);
- identity-view inputs only, all of the same shape, one dtype per region;
- **all live-outs**, i.e. several outputs from one launch;
- straight-line arithmetic with per-element intermediates in local values;
- 256 threads per block with a tail guard.

Not preserved deliberately: CubeCL IR itself, its client/launch plumbing, and any
WebGPU-only baked-layout variant.

## 2. Description

```rust
pub enum FusionOp { Add, Mul, Div, Neg, Abs, Conj, Max, Min, Clamp,
                    Exp, Log, Sin, Cos, Tanh, Sqrt, Rsqrt, Pow, Expm1, Log1p, Erf }

pub enum FusionOperand { Input(u32), Value(u32) }   // Value = instruction index

pub struct FusionInstruction { pub op: FusionOp, pub operands: Vec<FusionOperand> }

pub struct FusionPlan {
    pub dtype: DType,
    pub inputs: u32,
    pub instructions: Vec<FusionInstruction>,   // topologically ordered
    pub outputs: Vec<FusionOperand>,
}

pub fn emit_fusion(plan: &FusionPlan) -> Result<EmittedFusion, Error>;
pub fn plan_fused(len: i64) -> Result<Launch, Error>;
```

The plan is a **topologically ordered instruction list**, not a recursive tree: an
operand names either an input or an earlier instruction, so a repeated value is
referenced once. Eight successive `v = v*v` operations are eight instructions, where
a tree would need 511 nodes. This also matches the pinned codegen, which references
earlier values by id.

`FusionPlan` carries no shape, stride, offset or length: those are runtime launch
arguments, so one artifact serves every size. Validation: a `Value` operand must
name an earlier instruction, an `Input` operand must be `< inputs`, arity must match
the op (`Clamp` is ternary, the rest unary or binary), and the instruction count is
bounded (`MAX_FUSION_INSTRUCTIONS = 256`); a plan with no outputs, no inputs or a
repeated/backward reference is rejected.

`plan_fused` computes the flat launch for each *runtime* `len` with the same checked
grid arithmetic as the other flat entries. Reusing a prepared artifact means reusing
the code and its bindings, never a stale grid: the geometry follows `len`.

## 3. Emitted source

Self-contained CUDA C++, no `#include`, so the artifact key is a function of the
source alone (the manifest records `includes: []`):

```cpp
extern "C" __global__ void t4a_fused_elementwise(
    float* out0, float* out1, const float* in0, const float* in1, long long len) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= len) return;
  const float v0 = in0[index];
  const float v1 = in1[index];
  const float t2 = v0 + v0;          // shared subtree
  out0[index] = ...;
  out1[index] = ...;
}
```

- Argument order: every output pointer in plan order, then every input pointer, then
  `long long len`. The manifest states it.
- **The helper text is not duplicated.** The emitted source embeds the whole of
  `cuda/include/t4a/abi.cuh` as its prelude (the header includes nothing, so the
  artifact needs no include path), and every operation resolves to a canonical helper
  from it: `t4a_cuda_fabs` for real `abs`, the sign-preserving magnitude only inside the
  complex division, the ordered clamp, the NaN extrema and the scale-robust complex
  division. A test asserts the emitted source contains no `#include`.
- 64-bit indexing is unconditional. The baseline selects an index width from the
  element count; this ABI does not, deliberately: the emitted kernel is flat and
  correct for any `len`, and no performance-parity claim is made for the fused path
  until a paired benchmark covers it.
- `NaN` behaviour: `Max`/`Min` propagate in `lhs`-then-`rhs` order and `Clamp` is the
  ordered expression, exactly as `t4a_cuda_binary` defines them.
- **Contraction.** The emitted chain is ordinary CUDA C++, so the compiler may
  contract `a*b + c` into an FMA; the per-op revision-6 launches give the same
  freedom inside one op. Fused and separate launches therefore use the same tolerance
  contract, not bit equality, and a caller that needs bit parity with a separate
  launch passes `--fmad=false` (the option string is part of the key, so the two
  configurations never share an artifact).
- Integer and Bool dtypes are not eligible (the baseline fuses none), and complex
  payloads use the component forms with the robust division.

## 4. Manifest and artifact key

```rust
pub struct FusionManifest {
    pub entry: &'static str,   // "t4a_fused_elementwise"
    pub abi_revision: u32,
    pub dtype: DType,
    pub inputs: u32,
    pub outputs: u32,
    pub block: u32,            // 256
    pub index_width: u32,      // 64
}

pub struct ArtifactKey { /* canonical, length-prefixed material */ }
pub fn artifact_key(emitted: &EmittedFusion, options: &str, target: &str,
                    compiler: &str) -> ArtifactKey;
```

**The key is the canonical key material, not a digest.** It is a length-prefixed
encoding of `ABI_REVISION`, the emitted source (which already contains the canonical
header text, so the header cannot change without changing the source), the dtype,
input/output counts and entry name, and the caller's compiler option string, target
string (architecture and output kind) and **compiler identity** (NVRTC or toolchain
version). It implements `Eq`/`Hash`, so it is usable directly as an in-memory cache
key; a persistent cache hashes it with whatever digest it already trusts. This
avoids both a hand-written hash and a new dependency, and it cannot silently drop an
input: the material *is* the inputs, in order.

Runtime extents, strides, offsets and buffer addresses are never inputs, so changing
them reuses the artifact.

## 5. Prepare and cache (caller side)

The contract, which this slice documents and exercises but does not implement in the
crate:

1. Prepare fixes dtype, op set and outputs; it computes the key and looks up an
   artifact.
2. On a miss it compiles the emitted source with NVRTC using its own options and
   target, reports diagnostics, and stores the artifact under the key.
3. It loads the artifact as a module in each runtime's private cache (revision 7's
   entries already take caller-owned modules; this entry does the same through the
   caller's loader).
4. Run performs no source generation, NVRTC call or key recomputation: the prepared
   source and its loaded module are reused, and changing ordinary extents does not
   recompile — only `plan_fused(len)` recomputes the grid.

Cache levels stay separate, as the migration design states: compilation artifacts key
on the emitted source and compiler identity, loaded modules on artifact identity plus
load options within one runtime/context.

The tests in §7 include a minimal reference consumer that does exactly this, because
the contract's evidence requires observed NVRTC counts rather than an argument from
API shape.

## 6. Validation in the crate

`emit_fusion` rejects, with a typed `Error`:

- an empty output list or `inputs == 0`;
- an input index outside `0..inputs`;
- a dtype other than F32/F64/C32/C64, checked before the instruction list so an
  instruction-free plan cannot emit four-byte accesses over a one-byte Bool buffer;
- an op outside the dtype's eligibility matrix (`Conj` is complex-only and `Abs`
  real-only, as the pinned classifier states);
- an arity that does not match the op (`Clamp` is ternary, the rest unary or binary);
- a `Value` operand that does not name an earlier instruction, or a repeated output;
- more than `MAX_FUSION_INSTRUCTIONS = 256` instructions;
- a kernel-parameter block above `MAX_FUSION_PARAMETER_BYTES = 4096` (CUDA 12.0's
  limit): one pointer per operand plus `len`.

New `Error` variants: `FusionEmpty`, `FusionTooLarge`, `FusionArity`,
`FusionEligibility`, `FusionOperand`.

**Buffer contract of the emitted entry** (documented in the ABI note for consumers):
every output and input buffer holds at least `len` elements of the manifest dtype
(`len` logical complex elements for C32/C64, whose components are interleaved), bases
are 16-byte aligned, outputs are pairwise disjoint and disjoint from the inputs, and
each launch is `plan_fused(len)` on the caller's stream. Values are written only for
`index < len`.

## 7. Tests

Host-side (no GPU):

- the eligibility matrix per `FusionOp` and dtype, including every rejected pair;
- arity, input-index and size/depth bounds;
- emitted source structure: the entry name, the argument order, one `len` guard, one
  temporary per distinct subtree (CSE), no numeric extent/stride constant anywhere
  in the source for two otherwise identical plans of different sizes
  (they are the same plan — asserted by the absence of any size argument);
- helper emission only when used;
- the key: stable across emissions of the same plan; different for a different op
  set, dtype, option string, target and compiler identity; unchanged when only the
  runtime `len` differs;
- the emitted source contains no `#include` and embeds the canonical header text, so
  the helper semantics cannot drift from the per-op entries.

GPU (local A100, `tests/gpu_fusion.rs`), as the reference consumer:

- NVRTC-compile the emitted source through `cudarc`'s NVRTC bindings, load the module
  and launch it, for:
  - F32 and F64 two-live-out real plans (`y0 = (a*b + c) / d`, `y1 = tanh(exp(a*b + c))`
    with a shared `a*b`), covering every eligible real op at least once across the
    plans, compared against a CPU reference at extents 1, 513 (ragged tail) and
    65536;
  - a C32 and a C64 plan built only from eligible complex operations
    (`add`, `mul`, `div`, `neg`, `conj`) with two live-outs, including the extreme
    complex-division fixtures the revision-6 tests use;
  - the NaN/signed-zero behaviour of a fused `Max`/`Min`/`Clamp` chain against the
    same CPU reference the revision-6 tests use;
- a contraction fixture (`a*b + c` at `FLT_MAX`/`-FLT_MAX`) recorded as a documented
  tolerance-level difference from two separate launches, not as a failure;
- **prepare/cache counters** observed at the consumer's own NVRTC call boundary: one
  artifact key across four extents → one NVRTC attempt and one module load, four warm
  launches with zero compiles and one physical launch each (harness counter); a plan
  with a different op set → a second key and a second compile; a changed option or
  target string → a third; two independent runtimes each load their own module and
  clearing one cache leaves the other's binding alive.

## 8. Deliberate exclusions

- Strided/broadcast fused inputs: the baseline fuses identity views only. A
  stride-capable fused path is a later extension, not a prerequisite.
- Code generation for non-elementwise work (reductions, scatter, linalg) and any
  CUDA-graph or scheduler facility.
- Narrow-index variants (64-bit indexing only, declared above).
- Disk persistence of artifacts, registry publication, and HIP/Metal.
