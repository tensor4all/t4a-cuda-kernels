# Kernel and ABI review checklist

Applies to every change in this repository, and to every ported kernel family.
Run it on the diff before requesting review or opening a PR, and give the same
list to an independent reviewer. Each item states how to verify it and what
counts as a failure; "the test passes" alone is not evidence for any of them.

Items marked **[AI]** are the ones agents most often miss, usually by writing
correct-looking code that never actually runs in parallel or that quietly
serializes tensor-sized work.

## 1. Dependency and execution boundary

- [ ] No direct or transitive dependency on `t4a-cuda-runtime`, tenferro,
      CubeCL or CubeK. Verify: `cargo tree`; `grep -rn "t4a_cuda_runtime\|tenferro\|cubecl"`.
      Failure: any hit outside a doc/comment or an explicit exclusion note.
- [ ] The crate compiles and its CPU tests pass with no CUDA toolkit and no GPU.
      Failure: a build script, `#[link]`, or a lazy static that touches CUDA.
- [ ] Kernels never allocate, free, create a stream/event, or synchronize.
      Verify: read the `.cu` body; `grep -n "cudaMalloc\|cudaFree\|cudaDeviceSynchronize\|cudaStreamCreate\|new \|malloc"`.
- [ ] Workspace, metadata, status and output buffers are caller-provided, with
      documented size, alignment and lifetime.
- [ ] New launch parameters are classified: runtime value (extents, strides,
      offsets, counts, scalars), operation attribute, or algorithm
      configuration. Failure: a concrete extent/stride emitted as a constant or
      folded into a compiled variant without an entry in the manifest.
- [ ] No implicit host round trip, no per-item host wait, no hidden
      upload/download, no default-stream launch.

## 2. Parallelism and threading **[AI]**

This is the item an AI-written kernel most often fails: it compiles, it is
numerically right for the tested size, and it runs the whole tensor in one
thread.

- [ ] The launch domain covers the output or update domain: every output
      element or update is reached by some thread, and no thread iterates over
      an unbounded tensor domain. Verify: read the entry, name the index that
      each thread computes, and confirm `grid * block` covers `len`. Failure:
      `if (threadIdx.x == 0)` / `blockIdx.x == 0` guarding a loop over `n`;
      a "flat" entry that only uses `blockIdx`; a grid of 1 for tensor-sized
      work.
- [ ] The grid and block actually come from the facade's `Launch` plan, and the
      plan is derived from the extent product, not a constant. Verify: the
      consumer test launches with the planned values and asserts the physical
      launch count for two very different sizes (e.g. 1 and 1e6).
- [ ] No serialization hidden in a barrier: `__syncthreads()` is reached by
      every thread in the block, on every path, including tail blocks.
      Failure: `return` before a barrier in a partial tile.
- [ ] Reductions and scans use bounded per-thread work plus block/warp
      cooperation; a documented threshold exists if a serial unit-thread
      fallback is allowed. Failure: one warp walking the whole axis.
- [ ] No unbounded register/local-memory spill from a rank loop that is
      unrolled over runtime rank. Verify: `nvcc --ptxas-options=-v` for the
      changed entries; check registers and spill stores.
- [ ] Grid, block and shared-memory values are checked against the device
      limits (65535 per grid dimension, block threads, 48 KiB default shared)
      before a plan is returned, not asserted after a failed launch.
- [ ] The test suite exercises a size far larger than one block for every new
      entry, and compares against a CPU reference over the whole result. A
      test at 8 elements cannot detect a single-threaded kernel.

## 3. Indexing and memory safety contract

- [ ] Logical order is stated per entry (which axis is fastest, how a flat
      index decodes) and matches the ported source.
- [ ] Signed strides and offsets are preserved; broadcast stride zero and
      negative strides are either supported or rejected explicitly.
- [ ] Every reachable address is proved inside the caller's allocation using
      signed arithmetic, including the minimum reachable offset.
- [ ] Injection and overlap rules are documented per entry: destinations that
      must be injective, buffers that may alias, buffers that may not.
      Failure: "the caller must not overlap" left implicit for a read-modify-
      write or scatter entry.
- [ ] A whole-view entry's element count equals the extent product (including
      the empty and rank-zero cases), or the entry documents partial-prefix
      behavior and validates it.
- [ ] Exact copies move bits: no arithmetic, no scaling by one, no type
      round-trip. Applies to every movement route (flat copy, materialize,
      contiguous→view, strided→strided, transpose), not just the flat one.

## 4. Numerical contract

- [ ] Supported dtype per entry is stated and enforced at the boundary; new
      dtype routes are not added silently (for example Bool arithmetic, which
      the baseline does not have).
- [ ] Integer add/sub/mul wrap deliberately; `INT_MIN / -1`, division by zero
      and cast saturation follow a documented choice.
- [ ] NaN, inf, signed zero and subnormals behave as documented per entry;
      copies are bit-exact, arithmetic is stated to be bit-exact only where the
      ported source guarantees it.
- [ ] Complex operations use the ported robust forms; extreme-magnitude operand
      tests exist for anything involving division, norms or phases.
- [ ] Compiler options are explicit: no `--use_fast_math`, no flush-to-zero, and
      any FMA contraction policy is declared. Verify: the exact nvcc/NVRTC
      option list recorded next to the test evidence.

## 5. ABI

- [ ] The C struct layout is fixed and asserted: size, field offsets,
      alignment, and the Rust `#[repr(C)]` counterpart.
- [ ] Element pointer alignment and offset units (elements, not bytes) are
      documented; vectorized paths validate the alignment they assume.
- [ ] Runtime dtype tags, entry names and supported-dtype matrix are listed in
      the manifest and the ABI document, and a test keeps the three in sync.
- [ ] ABI changes bump the documented ABI revision; the revision is part of any
      artifact or cache key that carries compiled code.

## 6. Provenance and licensing

- [ ] Each ported file names its source (project, file, function or line range,
      pinned commit) in a file-level comment, plus the nature of the reference
      (translated, derived, validated against).
- [ ] The upstream copyright and license notice is preserved, the upstream
      license text is in the repository, and the SPDX metadata matches. Adding
      new attribution beside the original is fine; replacing it is not.
- [ ] Third-party notices that apply only to families not yet ported are not
      silently dropped: they move with the family, or the gap is recorded.

## 7. Evidence

- [ ] The GPU evidence records hardware, driver, toolkit, exact command and
      observed result; a hosted-gate result alone does not document local
      behavior.
- [ ] Physical launch counts are counted, not inferred from one API call.
- [ ] Numerical comparison is against a CPU reference or a stated residual, for
      every dtype route, over empty, length-1, odd, offset, strided, batched and
      misaligned cases.
- [ ] Bit-pattern equality is checked where the contract is bit-exact, including
      nonfinite values, NaN payloads, signed zero, holes and tails of a
      sentinel-filled destination.
- [ ] Deliberate exclusions and deferred variants are written into the design or
      work log, with the reason, instead of being left as an unexplained gap.

## 8. Repository rules

The shared [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md)
still apply, in particular:

- `common/performance.md`: no hidden CPU/GPU transfer or host round trip; GPU
  kernels map the output/update domain to the launch domain; no unbounded serial
  work in one thread; no zero-initialization of buffers that are fully
  overwritten; cache ownership with bounds, clear and stats.
- `common/provenance.md`: record references at writing time, comply with
  copyright, credit algorithms and ask before publishing credit changes.
- `rust/performance.md` (GPU kernels, threading): no single-thread fallbacks for
  tensor-sized work, no large compile-time parameter sets duplicating runtime
  metadata, one declared source of truth for threading policy.
- `common/repository.md`, `common/docs-and-tests.md`: invariant markers,
  documentation audits, local validation before hosted gates.
