# Exposed kernel sources for NVRTC prepare (ABI revision 10)

Date: 2026-10-23. Follow-up to
[t4a-cuda-kernels#14](https://github.com/tensor4all/t4a-cuda-kernels/issues/14) for
tenferro's adoption ([tenferro-rs#2208](https://github.com/tenferro-rs/issues/2208),
unit U2).

## Why

The crate ships planners and entry names, but a consumer that prepares artifacts
through NVRTC needs the kernel text. Tenferro's new CUDA path compiles the kernels it
launches itself, and it cannot reach a dependency's files, so the crate now embeds them:
`t4a_cuda_kernels::source` exposes the shared header and every translation unit, and
`source::compose(module)` returns the header followed by the module with its own
`#include <t4a/abi.cuh>` line removed, which is what an in-memory NVRTC compilation
takes.

## What landed

- `source::ABI_HEADER`, `source::MODULES` (eleven translation units, `include_str!`),
  `source::Module { name, source }`, `source::module_of(Entry)`,
  `source::INCLUDE_LINE` and `source::compose(&Module)`.
- `the_exposed_sources_match_the_files` (CPU): the embedded header and every module equal
  their files, the module set equals `cuda/src/*.cu`, `compose` carries every line
  except the include, and every `Entry` resolves to exactly one module.
- `the_composed_sources_compile_under_nvrtc` (GPU lane, local A100): all eleven composed
  sources compile with `cudarc::nvrtc` for `compute_80`, which is the prepare path's
  first check.

## Local gates

A100 80GB, driver 580.126.09, CUDA 12.6: `cargo fmt --check`, `cargo clippy -j 16
--all-targets -- -D warnings`, 83 planning + 7 fusion CPU tests, `cargo doc` without
warnings, and the `gpu-tests` suites including the new NVRTC composition test (85 s for
eleven translation units).

## Not done

- No artifact cache, plan key or module binding here: those belong to the consumer's
  prepare pipeline (tenferro `U2`/`U3`), which now has the text it needs.
