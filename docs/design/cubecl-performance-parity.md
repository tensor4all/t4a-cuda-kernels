# CubeCL performance preservation contract

Status: implementation requirements, 2026-10-20. This is a source/policy audit; no native CUDA port or new performance measurement is claimed.

This contract supplements [the migration design](cubecl-kernel-port.md). The audited tenferro baseline is [14bc1ab933e3c7a9c966e08355670331ab9d993c](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c); its CubeCL fork is [11b52669f13e27bbe188f988fd696df6d989a562](https://github.com/tensor4all/cubecl/tree/11b52669f13e27bbe188f988fd696df6d989a562). Preserve useful behavior and measured fast paths. CubeCL-specific ownership workarounds and recorded compiler defects are not migration requirements.

## 1. Runtime values and classified specialization

The existing [kernel metadata contract](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/gpu-backend-design.md#kernel-metadata-contract) distinguishes layout values from code specialization. The native CUDA ABI must preserve that distinction.

| Parameter class | Native CUDA treatment |
| --- | --- |
| Axis extents, tensor strides, buffer lengths, element counts and flattened products | Runtime arguments; never part of the generated source or compilation key as their concrete values. |
| Element offsets, slice starts, padding amounts, slice/window sizes and running concatenation offsets | Runtime arguments, including values derived from layout or position within an operation. |
| Rank / sequence length | A classified compile-time loop bound is allowed for fixed-size local storage and unrolled axis loops. The rank is validated at the boundary and carries no extent/stride values. |
| Operation attributes | Transpose permutation, broadcast/gather/scatter dimension mappings, axis sets and static slice-step attributes may specialize code. Distinguish static slice steps from tensor strides. |
| Algorithm configuration | Tile size, block rows, shared-memory padding, vector width, index/address arithmetic width and kernel blueprints may select compiled variants. Validate their capability requirements. |
| Ordinary numerical scalar arguments | Runtime values. Dispatch operation/mode choices outside element loops; data-dependent mathematical comparisons remain part of the operation. |

Every emitted/template constant is classified as rank, operation/axis mapping, or algorithm configuration in a reviewed kernel/specialization manifest. Source-contract checks and runtime compile counters enforce this policy. Avoid a generic “shape specialization” escape that turns arbitrary layout values into constants.

### Metadata transport and shape reuse

[CubeCL's runtime_sequence](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/cubecl/dispatch.rs#L153) sends per-axis values as launch scalars; only sequence length affects specialization. Preserve this property for common small-rank native CUDA paths with a documented by-value scalar/POD argument ABI for host-known layout metadata. Data-dependent index/start arrays remain device inputs; do not download them just to fill scalar arguments. Do not replace every shape/stride array with a freshly allocated and uploaded device array.

Larger metadata may use explicit caller-provided device storage when necessary. The plan declares its size, transport and lifetime; transfers/allocations are observable. Ordinary metadata transport must not add a per-operation host barrier.

At a fixed dtype, rank, attribute set, target and algorithm variant, changing extents, strides, offsets, padding/window sizes or concatenation offsets reuses compiled code. Recompute launch geometry and workspace requirements separately. Selecting a different documented variant, such as 32- versus 64-bit indexing, may use a different artifact. This does not relax a prepared program's declared input-shape constraints.

The native-permutation `*_comptime_layout_kernel` twins are a [WebGPU/Metal-only exception](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/tests/integration/kernel_metadata_contract.rs). CUDA uses the runtime-layout versions. Exclude those twins from the CUDA port; keep the useful tile configuration, not their baked layout.

## 2. Planning, indexing and memory traffic

- Validate layout, bounds, aliasing, dtype and checked shape/byte/launch arithmetic once at the owning boundary. Kernel bodies use the resulting invariants without repeating host-style validation per element.
- Preserve compact linear-buffer elementwise/cast paths; do not add rank-wide coordinate decoding to dense operations.
- Preserve [bilateral affine-axis fusion and linear/strided/tiled classification](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/native_permutation.rs). Fuse compatible adjacent axes before choosing effective rank/permutation where observable semantics permit it. Keep numerical extents/strides as runtime values after this normalization.
- Use narrow index arithmetic where the full reachable range, products, offsets and intermediate computations are proved to fit. Preserve a checked wide path. Signed view strides and large offsets require more than a logical element-count check.
- Preserve direct borrowed/view reads where the current operation supports them. In particular, gather must not copy an entire embedding table just to read a few rows. Same-device canonicalization remains an explicit supported-layout boundary, with its traffic included in measurements.
- Nonempty reductions read their original input for the first axis. Empty-axis operations that semantically return independent storage still copy; do not erase ownership semantics as an optimization.
- Full-overwrite outputs/scratch use the caller's correctly acquired uninitialized storage. Skip preliminary zero fills only when every value is written before any read. Scatter initialization and other read-modify-write operations keep their required initialization.
- Perform operation/mode/dtype dispatch at the outer boundary. Keep source generation, full-plan cloning, string formatting and source/header hashing out of warm execution. Prepare stable artifact/binding identifiers and reusable argument storage; use bounded inline host metadata for the common small-rank path.

The [#2112 performance record](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/gpu-performance-invariants-2112.md) supplies evidence for borrowed gather, output reuse and semantic-layer degenerate-dot rewrites. The rewrites remain tenferro's responsibility. Kernel artifact keys exclude runtime sizes; shape-dependent vendor-plan keys are a separate cache and may require extents/strides.

## 3. Launch and algorithm families

### Elementwise and fusion

The current dense launch baseline is [256 threads per block](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/cubecl/dispatch.rs#L41). Carry it as a comparison point, with checked grid limits and a tail guard; it is not a universal optimal block size.

The source emitter lowers supported expressions to typed straight-line arithmetic with per-element intermediates in local values. Preserve all live-out outputs, input reuse and the current dtype/op eligibility matrix, including its real/complex distinctions. Ordinary intermediates must not become separate device arrays and launches.

Tenferro retains fusion grouping, dependency/liveness handling, and the decision to materialize a view only when the supported path justifies it. Preserve [prepared-region multi-output and completion rules](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/gpu-prepared-elementwise-fusion.md). The existing view-copy amortization policy is a baseline to compare, not a reason to require copies in a new stride-capable path.

Count executor commands and physical CUDA launches separately. For an eligible dense elementwise chain, require one physical fused kernel launch as well as correct values; a single executor command alone does not prove that property. Composite reductions, scatter and vendor operations have their own launch counts.

### Data movement and transpose

Preserve linear copies, generic signed-stride copies, and [shared-memory tiled transpose](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L165).

The existing planner exposes tile/block-row/padding/vector configurations, with `16x8-p1-v1` as its default and explicit 8/16/32 tile and v1/v2/v4 alternatives. Preserve the algorithm family and capability/eligibility checks. Coalesced reads/writes, shared-memory padding, destination-fast grid ordering, tail handling, batching and grid/shared-memory limits are migration requirements. Vectorization must respect the actual alignment, layout and tail; a configuration name is not proof of valid wide loads.

Copy remains bit-preserving for real/complex/Bool/integer payloads. Keep the [exact-copy and tiled-destination regression cases](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/worklogs/issue-1891-exact-cuda-copy-and-tiled-transpose-destination.md). Existing vendor permutation routes remain separate; do not replace every route with a slow generic strided kernel.

### Reductions

Port the current [Auto strategy and split planner](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/routines.rs) and [launch logic](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/launch.rs), not a serial reduction reference.

- Unit-per-output work is bounded by the warp width; longer axes use warp/parallel reductions.
- Long axes with few outputs also split across blocks. A single warp walking the whole input reintroduces the defect fixed by #2134.
- Baseline split policy: `min_chunk = warp_width * 8`, `splits = clamp(reduce_len / min_chunk, 1, 4096)`, and floor-divided chunk length with the last chunk absorbing the remainder. These are planner parameters, not tensor-size constants in generated code.
- Pass 1 writes explicit caller-provided partials; pass 2 combines without recursively splitting them again. For sum-of-squares, pass 2 is a plain sum.
- Preserve integer wrapping, NaN propagation and the chosen deterministic reduction topology. Floating results use the established tolerance contract; no bitwise agreement with a differently ordered CPU sum is promised.
- No host synchronization between partial and final passes on one stream. The two-pass algorithm can outperform one launch; do not sacrifice the split to satisfy a blanket one-launch target.

The [#2134 record](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/gpu-reduce-split-2134.md) contains historical A100 evidence. Its prose includes superseded details; the pinned current planner/launch source is authoritative.

### Indexing, scatter and linalg helpers

Indexing kernels cover the output/update domain in parallel. Scatter keeps the parallel initialization copy followed by parallel updates; overlapping floating and complex-component updates keep the supported atomic semantics and documented nondeterminism. Do not serialize tensor-sized work in one CUDA thread.

Linalg/RRQR helpers preserve device-resident state and their current numerical conventions. Record dependency-driven multi-launch sequences and required status observations. This design does not require a new GEMM/SVD/FFT implementation or a general CUDA task scheduler.

## 4. Compilation and execution overhead

The [pinned CubeCL CUDA compilation path](https://github.com/tensor4all/cubecl/blob/11b52669f13e27bbe188f988fd696df6d989a562/crates/cubecl-cuda/src/compute/context.rs#L124) already uses NVRTC. Preserve explicit target/options and numerical behavior while moving ownership to the caller/runtime.

Prepare compiles or retrieves all required artifacts. Late-bound engines load/bind those artifacts explicitly before numerical execution. Warm run performs no source generation or NVRTC calls. Artifact keys include classified specialization, source/headers, ABI, compiler/options and target; runtime layout values are excluded. Bound modules, pools and scratch stay private to each runtime, even for two runtimes on one GPU.

Reuse caller-owned outputs/scratch and prepared launch metadata. Avoid per-item argument heap allocation and repeated code/plan reconstruction where the ABI permits reusable storage. No hidden metadata upload, extra materialization, device allocation or host barrier may be introduced as a convenience of the wrapper.

The native runtime's retirement must keep buffers/modules alive asynchronously. Do not carry over CubeCL-specific queue/server/interop synchronization merely as habit; preserve the safety obligation through the new ownership model and completion witnesses. Explicit status/readback, resource-pressure waits and teardown are documented and counted.

Keep default IEEE behavior. Do not obtain speedups by enabling broad fast math or adopting the [recorded unsound constant folds](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/cubecl-011-ieee-constant-fold.md). Validate signed zero, NaN/Inf, complex division and contraction/FMA-sensitive results under the declared numerical contract.

## 5. Required evidence before claiming performance parity

### Structural and numerical checks

- [ ] Each compiled/template parameter has a classification; extents/strides and the other runtime values above never enter emitted constants or compilation keys.
- [ ] At least four distinct layouts per fixed specialization vary extents, strides, offsets and relevant starts/padding/window/concatenation values, with reference-checked outputs and unchanged compilation counts.
- [ ] Instrument the actual NVRTC boundary. The current [kernel_specialization_tests](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/cubecl/tests/kernel_specialization_tests.rs#L1) retain numerical checks but their compile-count assertion is disabled because the [observation hook is missing](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/design/cubecl-011-compile-count-hook.md). Porting those test names does not prove reuse.
- [ ] Cover legitimate rank/attribute/algorithm/target changes separately, plus header/options invalidation and runtime-private cache clear/eviction. Measure compile attempts/successes, module loads and cache hits/misses independently.
- [ ] Preserve compact/view reads, no preliminary full overwrite fill, tiled transpose, parallel scatter and split reductions through structural checks and real CUDA numerical tests.
- [ ] Test multi-output fusion, correct dependency/completion coverage, and one physical launch for the eligible dense chain. Warm run has zero code-generation/NVRTC invocations.

### Initial paired benchmark matrix

Freeze the exact case list and comparison protocol before running the candidate. This is the required starting coverage, not permission to select only favorable results.

| Area | Cases to include |
| --- | --- |
| Dense elementwise/fusion | Small launch-bound and large bandwidth-bound tensors; the existing four-op real chain; real/complex routes; multiple live-outs. |
| Layout/transpose | Compact copies, odd/ragged matrices, both transpose orientations, batched matrices, offset/misaligned and signed-stride views; baseline tile configurations. |
| Reduction | Lengths 511/512/513/600 and 1e6/1e7; full and few-output long-axis reductions, both axis orientations; F32/F64 and supported integer/complex families. |
| Indexing/scatter | Gather whose table grows while the requested rows stay fixed; padding/slicing with varied runtime metadata; collision-heavy real/complex scatter. |
| Linalg helpers | Representative QR/LU/RRQR shapes from the existing numerical suite; warm kernel and orchestration counts. |
| Cache/runtime overhead | Many layouts at fixed specialization, cold prepare, repeated warm run, output/scratch reuse, and two independent runtimes on one physical GPU. |

Use a paired release experiment on the same GPU/driver/toolkit and equivalent algorithms, with pinned baseline/candidate and benchmark source revisions. Predeclare repetition policy (at least three repetitions with spread), statistics, noise/validity rules and non-regression tolerances before candidate timings. Extra tuning beyond preserving existing paths first establishes that the path matters in a representative workload.

Report cold preparation, NVRTC and module loading separately from warm host submission and device elapsed time. Record physical launches, host waits, transfer/copy bytes, allocations/zero fills, scratch requirements and cache residency. Timing barriers belong at observation boundaries, not between every operation in the measured chain.

Compare reusable kernels through an equivalent external caller, then run tenferro integration workloads when adoption is evaluated. Preserve both microbenchmark and end-to-end evidence. Historical numbers in linked documents are context, not new measurements or portable performance guarantees. Required unexplained regressions block a performance-parity claim; missing hardware is reported as not run. An invalid/noisy experiment is inconclusive, and is rerun as a whole under the declared protocol.
