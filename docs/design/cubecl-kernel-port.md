# Runtime-independent CUDA kernel migration

Status: design and source inventory, 2026-10-20. No kernel has been ported or GPU-tested in this repository yet.

This repository, **tensor4all/t4a-cuda-kernels**, will provide the Rust crate `t4a-cuda-kernels` (import `t4a_cuda_kernels`) and reusable CUDA C++ sources. It is a separate repository from [t4a-gpu-backend](https://github.com/tensor4all/t4a-gpu-backend).

**Hard dependency boundary: this crate does not depend on t4a-cuda-runtime, tenferro Tensor/Engine/graph/AD crates, CubeCL or CubeK.** External CUDA applications must be able to use the kernels with their own streams, allocations, module loading and synchronization.

## 1. Source baseline and inventory

Reviewed source: [tensor4all/tenferro-rs at 14bc1ab933e3c7a9c966e08355670331ab9d993c](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c). The local checkout was older; this inventory was read from the pinned main commit, not from the checkout's older kernel bodies.

The eight primary kernel/helper files below contain **4,947 physical source lines**, including comments/blank lines and before generic dtype instantiation. They are migration candidates, not a requirement to preserve every symbol or duplicate helper. Port the currently supported CUDA behavior, consolidate duplicates and verify host dispatch/capability gates.

See the [symbol inventory](../kernel-inventory.md) and [machine-readable inventory](../kernel-inventory.json) for pinned source links and macro-generated entry names.

| Family and source | Candidates and important variants | Migration |
| --- | --- | --- |
| [elementwise.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs) — 882 lines | Real/complex add/sub/mul/div, integer wrapping arithmetic and checked div/rem/pow, scalar/mixed-real-complex forms, broadcast multiply, strided-source axpby, analytic unary operations, abs/sign, NaN-aware extrema, compare/select/clamp/conjugation. Robust complex division has its own helpers. | Port to typed CUDA kernels and numerical helper headers. Preserve current dtype/operation gates. |
| [structural.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs) — 589 lines | Signed-stride/offset materialization and view writes, strided-to-strided copy, tiled transpose and layout-specialized variants, broadcast, conversion/cast validation, reverse, concatenate, zero fill and in-place scaling. | Port data-movement kernels; exact copy must not become arithmetic scaling. Compact copy/zero may use caller-side CUDA memcpy/memset. |
| [indexing.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs) — 558 lines | Static/dynamic slice, pad, gather, additive real/complex scatter, floating-index validation and extraction of invalid values. | Port indexing semantics and collision-safe updates; expose device status buffers instead of hidden host readbacks. |
| [diagonal.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs) — 87 lines | Extract/embed diagonal, lower/upper triangular masks. | Port, reusing common strided/zero helpers. |
| [reduce/kernels.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs) — 1,438 lines | Sum/product, sum of squares, min/max; serial-per-output, warp/plane and chunked variants; real/complex and wrapping integer paths; NaN propagation. | Port the algorithm families and explicit partial-workspace requirements, not CubeCL launch types. |
| [helpers.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs) — 140 lines | Wrapping arithmetic, NaN extrema, coordinate/axis helpers. | Consolidate into CUDA headers and host validation helpers; not standalone exported operations. |
| [linalg gpu/kernels.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs) — 491 lines | One fill, real/complex adjoint and magnitude, Householder-vector/Q assembly, QR phase/gauge extraction and application, upper-trapezoidal validation, LU output extraction/parity/pivot application. | Port the supporting kernels used around vendor calls; they are part of CUDA numerical parity. |
| [rank_revealing_qr.rs](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs) — 762 lines | Permutation initialization, real/complex column norms, pivot selection, column swaps, Householder construction/application, Q initialization, R extraction and prefix rank. | Port device primitives and explicit state/status arrays. Keep the bounded host orchestration separate from runtime ownership. |

### Additional paths to account for

- [Reduction launch/routines](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce): preserve strategy selection, split/finalize behavior and scratch requirements. Remove CubeCL client queries/launch plumbing; accept explicit device capabilities where planning needs them.
- [Native permutation planning](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/native_permutation.rs): a host planning reference, not a CUDA kernel body. Preserve useful layout classification without importing tenferro types.
- [Generated fusion](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/cubecl/fusion): the current implementation builds CubeCL IR dynamically. Moving static kernels alone will not replace this path. Tenferro keeps fusion selection and its graph representation; this repository supplies a small, independent expression-to-CUDA emitter using its numerical helper headers. The caller/runtime compiles and loads its output. Do not make this crate depend on `ElementwiseFusionPlan` or build a universal compiler as a prerequisite.
- [WebGPU packing](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/webgpu/kernels.rs): `pack_lhs_dot_general` / `pack_rhs_dot_general` serve the WebGPU/CubeK route. They are not required for current CUDA parity; reuse a generic pack primitive only if a CUDA consumer needs one.
- [Sample CubeCL kernel](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/samples/cubecl-kernel/src/kernel.rs): consumer example, not another production kernel family.

## 2. Vendor operations and exclusions

GEMM/contractions, general vendor permutation routes, QR/SVD/eigh/solve and FFT are not all handwritten CubeCL kernels. Existing cuBLAS/cuTENSOR/cuSOLVER/cuFFT calls must be tracked separately. The CUDA FFT source tree has no separate static CubeCL FFT kernel definitions; its layout/postprocessing can depend on the generic kernels above.

This port does not request replacement GEMM, SVD or FFT implementations. Vendor handles, operation plans, stream binding and vendor-library loader policy stay with consumers or separate adapters. Avoid bringing the existing vendor loaders and Tensor dispatch into this kernel crate.

Currently unsupported operations/dtypes are not silently added or advertised by a mechanical generic CUDA translation. Examples include the current dynamic-update-slice gap and Bool/integer additive-scatter gaps. Treat the pinned capability matrix and tests as the baseline, and record deliberate changes.

## 3. Dependency and execution boundary

```text
tenferro-gpu or another CUDA consumer
    +-- t4a-cuda-runtime, plain cudarc, or its own CUDA runtime
    +-- t4a-cuda-kernels
           +-- optional thin CUDA Driver API bindings
```

No dependency edge runs between this crate and t4a-cuda-runtime. This applies to the production crate, public types and standalone examples/tests.

Primary reusable assets are CUDA C++ `.cu` / `.cuh` sources with C-compatible kernel entry points and documented parameter layouts. Non-Rust consumers can compile them with NVCC/NVRTC and launch them directly. The Rust facade provides source/entry manifests, pure host validation and launch/workspace planning; a thin optional launcher may use CUDA driver handles directly.

The default source/planning build should not require an installed CUDA toolkit or initialize a GPU. Compilation of device code and GPU execution are exercised by explicit test/build commands. CUBIN is architecture-specific; a precompiled artifact must declare its target. Do not make a single architecture's CUBIN the universal distribution.

The caller owns:

- context, stream, allocations and initialization state;
- module compilation/loading, cache, loaded function handles and their lifetime;
- output and scratch/status/descriptor buffers;
- cross-stream dependencies, completion, buffer/module retirement and numerical-status observation.

The kernel layer owns:

- operation semantics, typed CUDA implementation and CUDA helper headers;
- validation of provided shape/layout/launch metadata and checked workspace-size queries;
- explicit launch plans and the ABI expected by each entry point.

Kernel launch success is not proof of asynchronous completion or device-side numerical validity.

### Preparation, compilation and private caches

The kernel crate provides a small source emitter for supported fused elementwise expressions using a runtime-independent operation description. Tenferro chooses fusion groups and converts its graph representation to that description. Source generation, ABI metadata and numerical helpers stay independent of tenferro and t4a-cuda-runtime; CUDA compilation/loading remains a caller/runtime service.

For traced execution, compilation is part of **prepare**, rather than the numerical run loop:

1. Prepare fixes the selected target capabilities and kernel/fusion variants, then generates CUDA C++ and a source/header/ABI manifest.
2. For each target, obtain a compatible precompiled or cached artifact; otherwise compile with NVRTC using explicit options and report diagnostics from prepare.
3. Record entry points, launch/workspace requirements and required artifacts before prepare succeeds. Treat ordinary shape/stride values as runtime arguments, as specified below.
4. Materialize loaded modules/functions in each supplied runtime's private cache during preparation. If engines are supplied only after the shareable plan is prepared, perform an explicit engine-binding/warmup stage before numerical execution. That stage loads the already prepared artifact in the actual context and validates the required target.
5. Run uses the prepared code and compatible engine-owned bindings. It performs no CUDA source generation or NVRTC invocation. A missing artifact or incompatible target is a preparation/binding error; run does not silently JIT-compile after numerical submission begins.

A shareable prepared plan may contain immutable source/artifact descriptions and requirements, but never captures an engine, CUDA context, stream or context-bound function handle. Loaded modules and mutable workspace remain engine-owned. A replacement runtime on the same ordinal gets its own loaded module and cache; the ordinal alone is not a module compatibility key.

Eager callers may compile on first use before the operation is enqueued, or explicitly prewarm. Subsequent compatible calls reuse the artifact/module; changing ordinary runtime shape values does not trigger recompilation.

There are two distinct cache levels:

- **Compilation artifacts:** key by the generated source and all included header contents, kernel/ABI revision, selected dtype and declared specialization, complete compiler options, NVRTC/toolchain version and target architecture/output kind. Concrete tensor layout/size values and buffer addresses are excluded. Classified rank, operation-attribute and algorithm variants are included; see the performance preservation contract.
- **Loaded modules/functions:** key by artifact identity and load options within the owning runtime/context generation. Do not use a process-global cache or share module/cache state between independent runtimes, even on one physical GPU.

The caller/runtime owns bounded cache capacity, clear/eviction, statistics and module retirement. Active prepared bindings and queued work retain the modules they need independently of cache membership. Clearing a cache must not unload code still in use. Disk persistence is optional; if supplied, it must validate the same compilation key and must never persist CUDA handles.

## 4. API contract

Exact Rust names are provisional; keep the public surface small.

A planned invocation describes its entry point(s), scalar/layout arguments, grid/block/shared-memory settings and workspace/metadata/status requirements. Multi-pass operations describe every pass and their dependencies; do not label a whole reduction as one physical launch when it needs partials and a final pass.

Execution accepts caller-provided CUDA function handle(s), stream and device pointers/spans. No runtime-specific buffer wrapper is required. Raw launches are `unsafe`: the caller proves context/device compatibility, allocation bounds and initialization, aliasing, ordering, and lifetime until GPU completion. Host-side argument checks do not prove those facts.

- Shape extents, offsets, strides and argument-structure ABI use documented fixed-width representations. Preserve signed strides, offsets and broadcast stride zero where supported.
- Use column-major logical indexing for the port, with explicit strides. No silent row-major repacking or host roundtrip.
- Check products, byte lengths, pointer offsets and launch/workspace sizes before allocation-sensitive planning or launch; validate even empty/no-op requests.
- Runtime-valued layout metadata must have explicit host-scalar versus device-buffer transport. Host scalar argument storage needs to survive CUDA enqueue; device metadata must survive execution.
- Scratch is queried before launch and supplied by the caller with documented alignment and lifetime. Production kernels never allocate/free device memory or create streams/events.
- No implicit upload/download, hidden global/default-stream execution, per-call module compilation, GPU worker/server thread, or per-item host synchronization.
- Data-dependent errors such as invalid floating indices, invalid casts or integer division by zero use explicit device status storage where required. The consumer decides when to wait/read and translate status into its own errors.
- Host planning/compilation may allocate ordinary host memory. This does not authorize hidden device allocation or payload transfer.
- Exact in-place/overlap rules are documented per entry. The metadata API must not imply that arbitrary overlapping pointers are safe.

Tensor shape extents, tensor strides, offsets, buffer lengths, flattened products, element/reduction counts, slice starts, padding amounts, slice/window sizes, running concatenation offsets and ordinary numerical scalar arguments are **runtime values**. Do not emit these concrete values as literals, template arguments or generated defines merely because prepare knows them.

Preserve the existing classified specialization boundary: validated rank/sequence length may be a compile-time loop bound for local storage and unrolling; permutation/axis mappings, axis sets and static slice-step attributes are operation attributes; tile/block/padding/vector settings, index/address width and kernel blueprints are algorithm configuration. Each compiled parameter is classified in a reviewed manifest and included in the artifact key. Numerical scalar arguments remain distinct from operation/mode dispatch.

For common small-rank paths, pass host-known runtime per-axis layout values as documented by-value scalar/POD launch arguments, matching CubeCL's runtime Sequence behavior without a separate device allocation/upload. Data-dependent index/start arrays remain device inputs without host readback. Larger metadata may use explicitly supplied device storage; declare and account for its transport and lifetime.

At a fixed dtype, rank, attribute set, target and algorithm variant, varying runtime layout values reuses compiled code. Prepare recomputes launch dimensions and workspace sizes separately. This does not by itself make a whole prepared program dynamically shaped: its input-shape constraints are a separate consumer contract.

CUDA does not inherit the WebGPU/Metal `*_comptime_layout_kernel` exception. Port the runtime-layout CUDA entries and retain useful rank/attribute/algorithm specializations. See the [performance preservation contract](cubecl-performance-parity.md) for the complete classification, fast paths and validation requirements.

CubeCL-specific workarounds are not required ABI features. For example, a complex coefficient can use a documented POD scalar argument instead of a one-element device array if the CUDA ABI supports the chosen representation.

## 5. Numerical and performance contracts

Port behavior, not known DSL restrictions or accidental implementation bugs.

- Support F32/F64/I32/I64/Bool/C32/C64 only where the particular operation is currently supported. Define Bool storage and complex real/imaginary layout explicitly; external consumers do not need a tenferro dtype enum.
- Integer add/sub/mul/neg/abs/power and integer reductions preserve defined wrapping behavior. CUDA C++ signed-overflow UB and `INT_MIN / -1` must be handled deliberately.
- Preserve scalar operand order, real/complex result types, conjugation and robust extreme-value complex division; do not replace scaled division with the overflow-prone textbook expression.
- Preserve NaN/signed-zero semantics of extrema, remainder, casts and data movement. A copy moves values/bits; multiplying complex values by one is not an exact copy for infinities.
- Preserve gather/dynamic-slice clamping, floating-index validation and additive-scatter collision semantics. Complex scatter updates both components correctly. Float atomic capability and reproducibility limitations must be stated.
- RRQR pivot ties follow the existing lowest-original-index rule. Test rank-deficient/scaled and complex cases. Whole RRQR orchestration is not promised to be one kernel.
- Default compiler options must not silently enable fast math that changes the above contract.
- Preserve compact/vectorized, tiled-transpose, warp-reduction and chunked/multi-pass paths where useful. Start with measured, correct implementations rather than porting every CubeCL specialization.
- Device synchronization inside a kernel is distinct from host synchronization. Record launch counts for composite operations and size buckets.

Relevant prior design evidence: [strided movement](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/worklogs/issue-1832-1834-cuda-strided-data-movement.md), [exact complex copy/tiled destination](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/worklogs/issue-1891-exact-cuda-copy-and-tiled-transpose-destination.md), [RRQR CUDA](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/docs/worklogs/2026-09-05-rank-revealing-qr-cuda.md).

### CubeCL performance preservation

The [performance preservation contract](cubecl-performance-parity.md) is part of this migration's acceptance criteria. It records the audited metadata/specialization rules, by-value runtime metadata, affine-axis fusion, dense and tiled copy paths, bounded unit/warp/split reductions, multi-output fusion, view reads, allocation reuse and required cold/warm overhead measurements. Preserve those paths or provide paired evidence for a deliberate replacement.

Rank, operation attributes and algorithm configuration may specialize code; concrete tensor sizes/layout values remain runtime arguments. The WebGPU-only baked-layout twins are excluded from CUDA. Existing compile-count tests have a disabled observation, so new tests must count actual NVRTC calls rather than rely on their names or source equality alone.

## 6. Implementation sequence

1. Define the C-compatible ABI, source packaging, classified specialization manifest and runtime-free planning; implement zero/one/copy/strided materialization/transpose, elementary arithmetic and reusable scalar helpers.
2. Implement reduction families and explicit scratch planning, conversions/status kernels and indexing/scatter/diagonal operations.
3. Implement linalg auxiliary kernels and RRQR primitives; consolidate shared fill, adjoint, phase, norm and pivot helpers.
4. Implement the independent fused-expression source emitter and validate a generated kernel with runtime shape arguments. Connect prepare-time compilation/cache lookup and explicit engine binding in a standalone CUDA consumer; tenferro adapts its fusion graph separately.
5. Compare with the pinned CUDA numerical tests, test an external plain-CUDA consumer, and run the performance contract's paired cold/warm benchmark matrix with actual compile/launch/transfer counters. Record supported coverage and unresolved regressions.

These are work packages, not authorization to publish or replace tenferro's backend. The port can be developed and tested without waiting for t4a-cuda-runtime.

## 7. Acceptance criteria

- [ ] No direct/transitive production dependency on t4a-cuda-runtime, tenferro, CubeCL or CubeK. Standalone tests/examples also use an independent CUDA consumer.
- [ ] CUDA source/header assets and ABI are packaged so NVCC/NVRTC consumers need no Rust runtime object. Ordinary source/planning checks build without CUDA initialization.
- [ ] Prepare obtains cached/precompiled artifacts or invokes NVRTC, and reports compilation failures before numerical run. Warm run performs zero source-generation/NVRTC calls; late engine binding is completed explicitly before submission.
- [ ] Every compiled parameter is classified as rank, operation attribute/axis mapping or algorithm configuration. Runtime extents/strides/offsets/lengths/starts/padding/window/concatenation values are excluded; CUDA does not use the WebGPU-only baked-layout exception.
- [ ] At fixed dtype/rank/attributes/target/algorithm variant, at least four runtime layouts reuse compiled code. Check numerical results and actual NVRTC invocation counts; the baseline test names alone do not establish this. Common small-rank host-known layout metadata uses by-value launch arguments without an added device upload.
- [ ] Cache tests cover header/options/compiler/target invalidation, bounded eviction/clear, and independent module/cache state for two runtimes on the same GPU. Clearing a cache cannot invalidate active bindings or in-flight kernels.
- [ ] Every source inventory family has a tested replacement, a consolidated replacement mapping, or an explicitly documented exclusion. Macro-generated symbols and status helpers are accounted for.
- [ ] GPU tests compare numerical results with trusted CPU references or established numerical residuals, not only shapes or successful launches.
- [ ] Coverage includes empty/odd sizes, offsets, reversed/broadcast views, batched layouts, all supported dtype routes, NaN/Inf/signed zero, integer boundaries and invalid index/cast/division cases.
- [ ] Test reduction partial/final paths, collision-heavy scatter, complex arithmetic and bit-preserving data movement.
- [ ] Linalg tests include reconstruction, orthogonality, pivot/permutation direction, phase conventions and rank deficiency.
- [ ] Use an external caller-owned stream and buffers with plain cudarc or raw CUDA. Show that calls do not create their own streams/pools, allocate hidden device scratch or host-synchronize after each item.
- [ ] Validate two independent consumer contexts/streams where supported; the caller supplies compatible modules and owns completion. A multi-GPU absence is reported explicitly.
- [ ] Preserve dense/view reads, affine-axis fusion, tiled transpose, long-axis split reduction, scatter initialization/update and eligible multi-output fusion. Verify commands and physical launches separately.
- [ ] Complete the [performance preservation contract](cubecl-performance-parity.md): paired release comparisons with predeclared cases/tolerances, cold prepare versus warm host/device costs, copy/allocation/wait counters and an explicit account of unresolved regressions. Record tested hardware/compiler combinations.
- [ ] Preserve required derivation notices and license texts before translating upstream-derived code.
- [ ] Consumer/backend adoption is reviewed separately; no existing tenferro/CubeCL path is removed by this design-only setup.

## 8. Provenance

The source is tenferro's existing code, not an independently invented implementation.

The reduction sources already record CubeK Reduce derivation at commit `9cf90b797107d46829e1c9d9355ce801c3dd4a7d` and dual-license notices. The complex-division sources record Julia Base derivation; the original consulted Julia revision was not recorded. Preserve that fact rather than inventing a pin, and verify the applicable original attribution when porting.

Use [the existing notices](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/THIRD_PARTY_NOTICES.md) as a starting record and audit the actual inherited notices before translation. Language translation does not remove provenance obligations. This initial repository setup contains documentation/inventory only, not translated third-party source.

## 9. Related projects

- [t4a-gpu-backend#1](https://github.com/tensor4all/t4a-gpu-backend/issues/1): t4a-cuda-runtime, a separate optional consumer execution foundation.
- [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208): conditional backend adoption.
- [tenferro-rs#2206](https://github.com/tensor4all/tenferro-rs/issues/2206): caller-held engines and eager/traced execution.
