# t4a-cuda-kernels — repository rules

Runtime-independent CUDA kernels for tensor4all and external CUDA consumers.

## Read first

- [Migration design](docs/design/cubecl-kernel-port.md) — dependency boundary,
  ABI contract, prepare/cache contract, acceptance criteria.
- [CubeCL performance preservation contract](docs/design/cubecl-performance-parity.md)
  — normative for what must not regress.
- [Kernel and ABI review checklist](docs/review-checklist.md) — every change
  must pass it, and an independent reviewer must be given it.
- [Symbol inventory](docs/kernel-inventory.md) — pinned source of the port.

## Hard boundary

This crate does not depend on `t4a-cuda-runtime`, tenferro, CubeCL or CubeK, and
does not allocate device memory, create streams or synchronize. Callers own the
CUDA context, stream, buffers, module loading and completion. The default
build and its tests must work without a CUDA installation; device compilation
and GPU runs are explicit.

## Workflow

- Read the workspace rules in `../AGENTS.md` and the shared
  [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md)
  (`common/performance.md`, `common/provenance.md`, `rust/performance.md`) before
  implementing.
- Plan the slice in `docs/design/` before code, and run the independent review
  (`independent-review` skill, `openai/gpt-6.1-sol`) on the plan before
  implementing and on the finished diff before opening a PR.
- Verify against [the review checklist](docs/review-checklist.md) — in
  particular section 2, parallelism: a kernel that compiles and matches a small
  CPU reference proves nothing about whether any thread beyond the first runs.
- Ported code keeps its derivation notice and the upstream license text; new
  files are `MIT OR Apache-2.0`.

## Commands

```bash
cargo fmt --check
cargo clippy -j 16 --all-targets -- -D warnings
cargo test -j 16                     # CPU-only: planning, validation, manifest
cargo test -j 16 --features gpu-tests  # local A100: nvcc -> cubin -> driver API
```

GPU evidence must record the GPU model, driver and toolkit versions, the exact
command and the observed result. Do not push to a hosted GPU gate to check.
